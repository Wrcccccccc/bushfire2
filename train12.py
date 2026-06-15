"""
    python unet_bushfire.py \
        --mode train \
        --date_start 2018-12-01 \
        --date_end   2019-01-31 \
        --tiles T56HKJ T56HKH T56HKG \
        --data_root E:/wrcccccccc/bushfirewrc/data \
        --output_dir ./output \
        --epochs 50 --batch_size 2  # batch_size=2, lr=1e-4, weight_decay=1e-4
        --use_amp 1  # 启用混合精度
"""

import os
import re
import glob
import argparse
import fnmatch
import random
import gc
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import mask as rmask
from rasterio.transform import from_bounds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from osgeo import gdal
# AMP相关导入
from torch.cuda.amp import autocast, GradScaler

# ─────────────────────────────────────────────
#  全局常量
# ─────────────────────────────────────────────
TILE_EXTENTS = {
    'T56HKJ': '199980.0 6290200.0 309780.0 6400000.0',
    'T56HKH': '199980.0 6190240.0 309780.0 6300040.0',
    'T56HKG': '199980.0 6090220.0 309780.0 6200020.0',
}

# 扩展气象变量
WEATHER_VARS_AVG = ['ET', 'Max_temp', 'Rel_Humid', ]  # 取均值的气象变量
WEATHER_VARS_SUM = []  # 取累计的气象变量
# 合并所有气象变量
ALL_WEATHER_VARS = WEATHER_VARS_AVG + WEATHER_VARS_SUM

N_S2_BANDS = 10
N_WEATHER = len(ALL_WEATHER_VARS)
N_FIRE_FREQ = 1
N_DEM = 1
N_IGN_POINTS = 3  # 起火点采样通道数
N_CHANNELS = N_S2_BANDS + N_WEATHER + N_FIRE_FREQ + N_IGN_POINTS + N_DEM

PATCH_SIZE_TRAIN = 256
PATCH_SIZE_INFER = 512
PATCH_STRIDE = 64
NODATA = -32768
FREQ = {}  # tile -> np.ndarray缓存


def get_date_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r'(\d{8})', base)
    return m.group(1) if m else ''


def parse_fire_date(date_str: str) -> datetime:
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析日期：{date_str}")


def filter_fire_by_date(fire_gdf: gpd.GeoDataFrame,
                        date_start: str,
                        date_end: str) -> gpd.GeoDataFrame:
    dt_start = datetime.strptime(date_start, '%Y-%m-%d')
    dt_end = datetime.strptime(date_end, '%Y-%m-%d')
    mask1 = fire_gdf['StartDate'].apply(
        lambda s: dt_start <= parse_fire_date(s) <= dt_end
    )
    mask2 = fire_gdf['Area_ha'].apply(lambda s: s > 100)
    mask = mask1 & mask2
    filtered = fire_gdf[mask].reset_index(drop=True)
    print(f"[过滤] {date_start} ~ {date_end}：共 {len(filtered)} 条火灾记录")
    return filtered


class FireSample:
    def __init__(self, fire_row, tile: str, data_root: str,
                 n_ignition: int = 5, patch_size: int = PATCH_SIZE_TRAIN):
        self.row = fire_row
        self.tile = tile
        self.data_root = data_root
        self.n_ignition = n_ignition
        self.patch_size = patch_size
        self.fire_no = str(fire_row['FireNo'])
        self.start_date = fire_row['StartDate']
        self.freq = None

    def _load_tif(self, path: str, band_idx: int = 1) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read(band_idx).astype(np.float32)
        arr[arr == NODATA] = np.nan
        return arr

    def _load_s2_mosaic(self) -> np.ndarray:
        pattern = os.path.join(
            self.data_root, 'dataset',
            f'{self.tile}_Mosaics2',
            f'S2_{self.fire_no}_{self.start_date}_10bands_ndvi.tif'
        )
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"找不到S2 mosaic: {pattern}")
        path = candidates[0]
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        arr[arr == NODATA] = np.nan
        arr = arr / 10000.0
        return arr

    def _load_weather_mosaic(self, var_name: str) -> np.ndarray:
        for mode in ('avg', 'sum'):
            pattern = os.path.join(
                self.data_root, 'vector', 'outputMasaic',
                f'{self.tile}_Weather', var_name,
                f'{var_name}_{self.fire_no}_{self.start_date}_{mode}.tif'
            )
            candidates = glob.glob(pattern)
            if candidates:
                return self._load_tif(candidates[0])
        raise FileNotFoundError(f"找不到气象合成: {var_name} / {self.fire_no} / {self.start_date}")

    def _load_dem(self, target_shape: tuple) -> np.ndarray:
        proj_path = r"D:\anaconda\envs\bushfire310\Lib\site-packages\rasterio\proj_data"
        if os.path.exists(proj_path):
            os.environ['PROJ_LIB'] = proj_path

        dem_path = os.path.join(
            self.data_root, "raster", "Predictors",
            f"DEM_SRTM_{self.tile}_UTM.tif"
        )
        ds = gdal.Open(dem_path)
        band = ds.GetRasterBand(1)
        dem_data = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        del ds

        dem_data[dem_data == nodata] = np.nan
        dem_data[dem_data == -32768.0] = np.nan
        dem_data[dem_data == 32768.0] = np.nan

        valid_min = np.nanmin(dem_data)
        valid_max = np.nanmax(dem_data)
        dem_data = np.nan_to_num(dem_data, nan=0.0)

        dem_tensor = torch.from_numpy(dem_data).float().unsqueeze(0).unsqueeze(0)
        dem_tensor = torch.nn.functional.interpolate(
            dem_tensor, size=target_shape, mode="bilinear", align_corners=False
        )
        dem = dem_tensor.squeeze().numpy()
        dem = (dem - valid_min) / (valid_max - valid_min)
        dem = np.clip(dem, 0.0, 1.0)

        return dem.astype(np.float32)

    def _load_fire_freq(self, tile, target_shape: tuple) -> np.ndarray:
        tif_path = os.path.join(self.data_root, 'raster', 'Fire_NPWSFireHistory_1902_2020_freq_UTM.tif')
        aoi_shp = os.path.join(self.data_root, 'vector', f'S2_{tile}_bbox_UTM.shp')

        aoi = gpd.read_file(aoi_shp)
        geom = [aoi.geometry.iloc[0]]

        with rasterio.open(tif_path) as src:
            clipped_data, clipped_transform = rmask.mask(
                src, geom, crop=True, nodata=src.nodata, filled=True
            )

        arr = clipped_data.astype(np.float32)
        with rasterio.open(tif_path) as _src:
            _nodata_val = _src.nodata
        if _nodata_val is not None:
            arr[arr == _nodata_val] = np.nan
        arr[arr == -32768.0] = np.nan
        arr[arr < 0] = np.nan
        arr = arr.squeeze(0)
        if arr.shape != target_shape:
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=target_shape, mode='bilinear', align_corners=False)
            arr = t.squeeze().numpy()
        arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def _load_burn_mask(self, target_shape: tuple) -> np.ndarray:
        mask_pattern = os.path.join(
            self.data_root, 'vector', 'Masks',
            f'{self.tile}_Masks',
            f'Mask_{self.fire_no}_{self.start_date}_*.tif'
        )
        candidates = glob.glob(mask_pattern)
        if not candidates:
            return None
        arr = self._load_tif(candidates[0])
        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.clip(arr, 0, 1)
        if arr.shape != target_shape:
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=target_shape, mode='nearest')
            arr = t.squeeze().numpy()
        return arr.astype(np.uint8)

    def _sample_ignition_points(self, burn_mask: np.ndarray) -> np.ndarray:
        H, W = burn_mask.shape
        channels = np.zeros((self.n_ignition, H, W), dtype=np.float32)
        burned_yx = np.argwhere(burn_mask == 1)
        if len(burned_yx) == 0:
            return channels
        n = min(self.n_ignition, len(burned_yx))
        chosen = burned_yx[np.random.choice(len(burned_yx), n, replace=False)]
        yy, xx = np.mgrid[:H, :W]
        sigma = max(H, W) * 0.05
        for k, (py, px) in enumerate(chosen):
            gauss = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sigma ** 2))
            channels[k] = gauss.astype(np.float32)
        return channels

    def load(self):
        """加载输入特征图和燃烧掩码"""
        try:
            s2 = self._load_s2_mosaic()
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")
            return None, None

        H, W = s2.shape[1], s2.shape[2]
        weather_channels = []
        weather_scale = {
            'ET': 10.0, 'Max_temp': 50.0, 'Rel_Humid': 100.0, 'Sol_Rad': 30.0,
            'Evaporation': 20.0, 'Total_rain': 50.0, 'Vap_Press': 10.0
        }
        for var in ALL_WEATHER_VARS:
            try:
                w = self._load_weather_mosaic(var)
            except FileNotFoundError:
                w = np.zeros((H, W), dtype=np.float32)
            if w.shape != (H, W):
                t = torch.from_numpy(w).unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
                w = t.squeeze().numpy()
            w = w / weather_scale.get(var, 1.0)
            weather_channels.append(w)
        weather_stack = np.stack(weather_channels, axis=0)

        if self.tile not in FREQ:
            try:
                FREQ[self.tile] = self._load_fire_freq(self.tile, (H, W))
            except Exception as E:
                print("未正常加载FREQ", E)
                FREQ[self.tile] = np.zeros((H, W), dtype=np.float32)
        freq = FREQ[self.tile][np.newaxis, ...] / 20.0

        burn_mask = self._load_burn_mask((H, W))
        if burn_mask is None:
            return None, None
        ign = self._sample_ignition_points(burn_mask)

        s2 = np.nan_to_num(s2, nan=0.0)
        weather_stack = np.nan_to_num(weather_stack, nan=0.0)
        freq = np.nan_to_num(freq, nan=0.0)

        try:
            dem = self._load_dem((H, W))
        except Exception as e:
            print(f"[DEM加载失败] {e}，使用全0")
            dem = np.zeros((H, W), dtype=np.float32)
        dem = dem[np.newaxis, ...]
        dem = np.nan_to_num(dem, nan=0.0)

        input_stack = np.concatenate([s2, weather_stack, freq, ign, dem], axis=0)
        assert input_stack.shape[0] == N_CHANNELS, f"通道数不匹配：{input_stack.shape[0]} != {N_CHANNELS}"

        return input_stack.astype(np.float32), burn_mask


class BushfireDataset(Dataset):
    def __init__(self, samples, patch_size=PATCH_SIZE_TRAIN,
                 stride=PATCH_STRIDE, augment=True):
        self.samples = samples
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.indices = []
        self._build_indices()

    def _build_indices(self):
        P = self.patch_size
        S = self.stride
        hi_pos, lo_pos, neg_indices = [], [], []
        burned_px = 0
        total_px = 0
        for sample_idx, (inp, mask) in enumerate(self.samples):
            if inp is None:
                continue
            H, W = inp.shape[1:]
            for y in range(0, H - P + 1, S):
                for x in range(0, W - P + 1, S):
                    mask_p = mask[y:y + P, x:x + P]
                    ratio = mask_p.sum() / mask_p.size
                    if ratio >= 0.85:
                        hi_pos.append((sample_idx, y, x))
                        burned_px += mask_p.sum()
                        total_px += mask_p.size
                    elif ratio > 0:
                        lo_pos.append((sample_idx, y, x))
                        burned_px += mask_p.sum()
                        total_px += mask_p.size
                    else:
                        neg_indices.append((sample_idx, y, x))

        hi_sampled = random.sample(hi_pos, len(hi_pos) // 2) if hi_pos else []
        effective_pos = lo_pos + hi_sampled
        n_neg = min(len(neg_indices), len(effective_pos) // 8)
        print(len(hi_sampled), len(lo_pos), n_neg,"++++++++++++++++++++++++++++++")
        sampled_neg = random.sample(neg_indices, n_neg) if n_neg > 0 else []
        for sidx, y, x in sampled_neg:
            total_px += P * P

        self.indices = effective_pos + sampled_neg
        random.shuffle(self.indices)

        print(f"[Dataset] 共生成 {len(self.indices)} 个训练块"
              f"（高密度正:{len(hi_pos)} 低密度正:{len(lo_sampled)}/{len(lo_pos)} 负:{n_neg}）")
        pos_ratio = burned_px / (total_px + 1e-6)
        pos_weight = (1 - pos_ratio) / (pos_ratio + 1e-6)
        self.pos_weight = pos_weight
        print(f"[数据] 过火像素比例：{pos_ratio:.4f}，正类权重：{pos_weight:.2f}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, y, x = self.indices[idx]
        inp, mask = self.samples[sample_idx]
        P = self.patch_size
        inp_p = inp[:, y:y + P, x:x + P]
        mask_p = mask[y:y + P, x:x + P]

        inp_p = torch.from_numpy(inp_p).float()
        mask_p = torch.from_numpy(mask_p).long()

        if self.augment:
            if random.random() > 0.5:
                inp_p = torch.flip(inp_p, dims=[2])
                mask_p = torch.flip(mask_p, dims=[1])
            if random.random() > 0.5:
                inp_p = torch.flip(inp_p, dims=[1])
                mask_p = torch.flip(mask_p, dims=[0])
            k = random.randint(0, 3)
            if k > 0:
                inp_p = torch.rot90(inp_p, k, dims=[1, 2])
                mask_p = torch.rot90(mask_p, k, dims=[0, 1])

        return inp_p, mask_p


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = N_CHANNELS,
                 base_features: int = 32,
                 n_classes: int = 2,
                 bilinear: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        f = base_features
        self.inc = DoubleConv(in_channels, f)
        self.d1 = Down(f, f * 2, dropout)
        self.d2 = Down(f * 2, f * 4, dropout)
        self.d3 = Down(f * 4, f * 8, dropout)
        self.d4 = Down(f * 8, f * 16, dropout)
        self.u1 = Up(f * 16 + f * 8, f * 8, bilinear)
        self.u2 = Up(f * 8 + f * 4, f * 4, bilinear)
        self.u3 = Up(f * 4 + f * 2, f * 2, bilinear)
        self.u4 = Up(f * 2 + f, f, bilinear)
        self.out = nn.Conv2d(f, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.out(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)[:, 1, :, :]


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        targets = targets.float()

        # 将 Batch 整体展平，进行全局 Batch-wise Dice 计算，提高负样本训练稳定性
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
                probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.6,
                 dice_weight=0.4,
                 pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            weight = torch.tensor([1.0, pos_weight], dtype=torch.float32)
        else:
            weight = None
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss()
        self.w_ce = ce_weight
        self.w_dice = dice_weight

    def forward(self, logits, targets):
        return (
                self.w_ce * self.ce(logits, targets)
                + self.w_dice * self.dice(logits, targets)
        )


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = (pred > 0.5).long()
    target = target.long()
    inter = (pred & target).sum().item()
    union = (pred | target).sum().item()
    return inter / (union + 1e-6)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    for inp, mask in tqdm(loader, desc='  Train', leave=False):
        inp, mask = inp.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(inp)
            loss = criterion(logits, mask)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        del inp, mask, logits, loss
        torch.cuda.empty_cache()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    n_samples = 0
    for inp, mask in tqdm(loader, desc='  Val  ', leave=False):
        inp, mask = inp.to(device, non_blocking=True), mask.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(inp)
            loss = criterion(logits, mask)

        total_loss += loss.item()
        proba = torch.softmax(logits, dim=1)[:, 1, :, :]
        for i in range(proba.shape[0]):
            if mask[i].sum() > 0:
                total_iou += compute_iou(proba[i], mask[i])
                n_samples += 1

        del inp, mask, logits, loss, proba
        torch.cuda.empty_cache()

    iou = total_iou / n_samples if n_samples > 0 else 0.0
    return total_loss / len(loader), iou


def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[训练] 使用设备：{device}")
    print(f"[AMP] 混合精度训练：{'启用' if args.use_amp else '禁用'}")

    all_samples = []
    for tile in args.tiles:
        shp = os.path.join(args.data_root, 'vector', 'Masks',
                           f'FireHistory_2018_2020_UTM_{tile}.shp')
        if not os.path.exists(shp):
            print(f"[警告] 找不到shp：{shp}，跳过 {tile}")
            continue
        fire_gdf = gpd.read_file(shp)
        fire_gdf = filter_fire_by_date(fire_gdf, args.date_start, args.date_end)

        for _, row in fire_gdf.iterrows():
            if row['Area_ha'] < 100:
                continue
            fs = FireSample(row, tile, args.data_root,
                            n_ignition=N_IGN_POINTS,
                            patch_size=PATCH_SIZE_TRAIN)
            inp, mask = fs.load()
            if inp is not None:
                all_samples.append((inp, mask))
            gc.collect()
            torch.cuda.empty_cache()

    if len(all_samples) == 0:
        print("[错误] 无有效样本")
        return

    print(f"\n[数据] 共加载 {len(all_samples)} 个火灾样本")
    burned_px = sum(m.sum() for _, m in all_samples)
    total_px = sum(m.size for _, m in all_samples)
    pos_ratio = burned_px / total_px
    pos_weight = (1 - pos_ratio) / (pos_ratio + 1e-6)
    print(f"[数据前统计] 过火像素比例：{pos_ratio:.4f}，正类权重：{pos_weight:.2f}")

    sample_ratios = [m.sum() / m.size for _, m in all_samples]
    sorted_idx = sorted(range(len(all_samples)), key=lambda i: sample_ratios[i])
    val_sample_idx = sorted_idx[::5]
    train_sample_idx = [i for i in sorted_idx if i not in set(val_sample_idx)]
    train_samples = [all_samples[i] for i in train_sample_idx]
    val_samples = [all_samples[i] for i in val_sample_idx]
    train_ratio_mean = sum(sample_ratios[i] for i in train_sample_idx) / len(train_sample_idx)
    val_ratio_mean = sum(sample_ratios[i] for i in val_sample_idx) / len(val_sample_idx)
    print(f"[划分] 训练:{len(train_samples)}场 (均值过火率{train_ratio_mean:.3f})  "
          f"验证:{len(val_samples)}场 (均值过火率{val_ratio_mean:.3f})")

    train_ds = BushfireDataset(train_samples, patch_size=PATCH_SIZE_TRAIN,
                               stride=PATCH_STRIDE, augment=True)
    val_ds = BushfireDataset(val_samples, patch_size=PATCH_SIZE_TRAIN,
                             stride=PATCH_STRIDE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=False)

    model = UNet(in_channels=N_CHANNELS, base_features=32, dropout=0.3).to(device)
    WARMUP_EPOCHS = 5
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine scheduler 的 T_max 定义为余弦退火阶段的总长度
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs - WARMUP_EPOCHS, 1), eta_min=1e-6)

    train_pos_weight = getattr(train_ds, 'pos_weight', pos_weight)
    print(f"[Loss] 训练集实际正类权重：{train_pos_weight:.2f}（上限截断为 10.0）")
    criterion = CombinedLoss(
        ce_weight=0.6,
        dice_weight=0.4,
        pos_weight=min(train_pos_weight, 10.0)
    ).to(device)

    scaler = GradScaler() if args.use_amp else None

    print(f"\n[模型] 参数量：{sum(p.numel() for p in model.parameters()):,}")
    os.makedirs(args.output_dir, exist_ok=True)
    best_iou = -1.0
    best_path = os.path.join(args.output_dir, 'best_unet.pth')
    log_path = os.path.join(args.output_dir, 'train_log.csv')

    with open(log_path, 'w') as flog:
        flog.write('epoch,train_loss,val_loss,val_iou\n')

    for epoch in range(1, args.epochs + 1):
        if epoch <= WARMUP_EPOCHS:
            # 线性 Warmup
            warmup_lr = args.lr * (0.1 + 0.9 * epoch / WARMUP_EPOCHS)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            current_lr = cosine_scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch}/{args.epochs}  lr={current_lr:.2e}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, scaler, args.use_amp)
        val_loss, val_iou = validate(model, val_loader, criterion, device, args.use_amp)

        # 仅在 Warmup 结束后才执行调度器的 step
        if epoch > WARMUP_EPOCHS:
            cosine_scheduler.step()

        print(f"  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_IoU={val_iou:.4f}")

        with open(log_path, 'a') as flog:
            flog.write(f'{epoch},{tr_loss:.6f},{val_loss:.6f},{val_iou:.6f}\n')

        if val_iou > best_iou:
            best_iou = val_iou
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_iou': val_iou,
                'n_channels': N_CHANNELS,
                'optimizer': optimizer.state_dict(),
                'scheduler': cosine_scheduler.state_dict()
            }
            if args.use_amp:
                save_dict['scaler'] = scaler.state_dict()
            torch.save(save_dict, best_path)
            print(f"  最佳IoU更新：{val_iou:.4f}，模型已保存")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[完成] 最佳验证 IoU = {best_iou:.4f}，模型已保存至 {best_path}")


def spline_window_2d(patch_size: int) -> np.ndarray:
    wind1d = np.sin(np.linspace(0, np.pi, patch_size)) ** 2
    return np.outer(wind1d, wind1d).astype(np.float32)


@torch.no_grad()
def predict_full_tile(model, input_stack: np.ndarray,
                      patch_size: int = PATCH_SIZE_INFER,
                      stride: int = PATCH_STRIDE,
                      device: torch.device = torch.device('cpu'),
                      use_amp: bool = True) -> np.ndarray:
    model.eval()
    _, H, W = input_stack.shape
    proba_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    window = spline_window_2d(patch_size)

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    for y in tqdm(ys, desc='推理行'):
        for x in xs:
            patch = input_stack[:, y:y + patch_size, x:x + patch_size]
            tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits = model(tensor)
                prob = torch.softmax(logits, dim=1)[0, 1, :, :].cpu().numpy()

            proba_map[y:y + patch_size, x:x + patch_size] += prob * window
            weight_map[y:y + patch_size, x:x + patch_size] += window

            del tensor, logits, prob
            torch.cuda.empty_cache()

    proba_map /= (weight_map + 1e-6)
    return proba_map


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = UNet(in_channels=ckpt.get('n_channels', N_CHANNELS)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"[推理] 模型加载完成（Epoch {ckpt['epoch']}，IoU={ckpt['val_iou']:.4f}）")

    os.makedirs(args.output_dir, exist_ok=True)

    for tile in args.tiles:
        shp = os.path.join(args.data_root, 'vector', 'Masks',
                           f'FireHistory_2018_2020_UTM_{tile}.shp')
        if not os.path.exists(shp):
            continue
        fire_gdf = gpd.read_file(shp)
        fire_gdf = filter_fire_by_date(fire_gdf, args.date_start, args.date_end)

        for _, row in fire_gdf.iterrows():
            fs = FireSample(row, tile, args.data_root, n_ignition=N_IGN_POINTS)
            inp, mask = fs.load()
            if inp is None:
                continue

            print(f"\n[推理] {tile} / 火灾 {row['FireNo']} / {row['StartDate']}")
            proba = predict_full_tile(model, inp,
                                      patch_size=PATCH_SIZE_INFER,
                                      stride=PATCH_STRIDE, device=device,
                                      use_amp=args.use_amp)

            out_path = os.path.join(
                args.output_dir,
                f'proba_{tile}_{row["FireNo"]}_{row["StartDate"]}.tif'
            )
            ref_pattern = os.path.join(
                args.data_root, 'dataset',
                f'{tile}_Mosaics2',
                f'S2_{row["FireNo"]}_{row["StartDate"]}_10bands_ndvi.tif'
            )
            refs = glob.glob(ref_pattern)
            if refs:
                with rasterio.open(refs[0]) as src:
                    profile = src.profile
                profile.update(count=1, dtype='float32', nodata=None)
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(proba.astype(np.float32), 1)
            else:
                np.save(out_path.replace('.tif', '.npy'), proba)
            print(f"  → 概率图保存至：{out_path}")

            if mask is not None:
                pred_bin = (proba > 0.5).astype(np.uint8)
                acc = (pred_bin == mask).mean()
                iou = compute_iou(
                    torch.from_numpy(proba),
                    torch.from_numpy(mask.astype(np.float32)).long()
                )
                print(f"  像素精度={acc:.4f}  IoU={iou:.4f}")

            gc.collect()
            torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description='丛林火灾过火面积预测 U-Net (AMP优化版)')
    p.add_argument('--mode', choices=['train', 'predict'], default='train',
                   help='运行模式')
    p.add_argument('--date_start', type=str, default='2018-12-01',
                   help='起始日期')
    p.add_argument('--date_end', type=str, default='2020-07-31',
                   help='截止日期')
    p.add_argument('--tiles', nargs='+', default=['T56HKJ'],
                   help='要处理的Sentinel-2图幅列表')
    p.add_argument('--data_root', type=str, default=r"F:\Qiao\bushifire\data",
                   help='数据根目录')
    p.add_argument('--output_dir', type=str, default='./output',
                   help='模型和预测结果输出目录')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--use_amp', type=bool, default=True)
    p.add_argument('--checkpoint', type=str, default='./output/best_unet.pth')
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(" 丛林火灾过火面积预测 —— U-Net Pipeline (AMP优化)")
    print(f" 模式：{args.mode}")
    print(f" 时间范围：{args.date_start}  ~  {args.date_end}")
    print(f" 图幅：{args.tiles}")
    print(f" 输入通道数：{N_CHANNELS} (10 S2 + {N_WEATHER} 气象 + 1 火频 + {N_IGN_POINTS} 起火点 + 1 DEM)")
    print(f" 混合精度训练：{'启用' if args.use_amp else '禁用'}")
    print("=" * 60)

    if args.mode == 'train':
        run_training(args)
    else:
        run_inference(args)


if __name__ == '__main__':
    main()