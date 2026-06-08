"""
实验步骤1：数据加载与预处理
============================
输入：
  - VNP46A2.A2022008.h21v04.002.2025114204956.h5  (VIIRS 战前 2022-01-08)
  - VNP46A2.A2022128.h21v04.002.2025119063131.h5  (VIIRS 战后 2022-05-08)
  - 战前1/ (Sentinel-2 T37TDN 2022-01-08)
  - 战前2/ (Sentinel-2 T37TCN 2022-01-08)
  - 战后1/ (Sentinel-2 T37TCN 2022-05-08)
  - 战后2/ (Sentinel-2 T37TDN 2022-05-08)

输出（写入 exp_outputs/）：
  - exp_v1_viirs_dntl.npy        : 战前-战后 NTL 差值的子区域 (float32)
  - exp_v1_viirs_lat.npy         : 子区域纬度
  - exp_v1_viirs_lon.npy         : 子区域经度
  - exp_v1_s2_dndvi_t37tdn.npy   : T37TDN ΔNDVI
  - exp_v1_s2_dnbr_t37tdn.npy    : T37TDN ΔNBR
  - exp_v1_s2_dndbi_t37tdn.npy   : T37TDN ΔNDBI
  - exp_v1_s2_scene_mask_t37tdn.npy : 有效像素掩膜
  - (T37TCN 同理)

说明：
  - 本脚本命名为 exp_v1（实验版本1），所有输出均带 exp_v1 前缀
  - 仅为实验演示，非最终可发表版本
  - 已知问题：Sentinel-2 战前(1月)vs 战后(5月)存在季节混淆，已在报告中讨论
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import h5py
import rasterio
import os
from pathlib import Path

# ============================================================
# 0. 路径设置
# ============================================================
BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs'
OUT.mkdir(exist_ok=True)

VIIRS_PRE  = BASE / 'VNP46A2.A2022008.h21v04.002.2025114204956.h5'
VIIRS_POST = BASE / 'VNP46A2.A2022128.h21v04.002.2025119063131.h5'

S2_DIRS = {
    'pre_t37tdn':  BASE / '战前1',
    'pre_t37tcn':  BASE / '战前2',
    'post_t37tcn': BASE / '战后1',
    'post_t37tdn': BASE / '战后2',
}

# Sentinel-2 关注区域：马里乌波尔
AOI_LAT_MIN, AOI_LAT_MAX = 46.85, 47.86
AOI_LON_MIN, AOI_LON_MAX = 37.66, 39.14

print("=" * 60)
print("实验 v1：数据加载与预处理")
print("=" * 60)

# ============================================================
# 1. VIIRS 夜间灯光处理
# ============================================================
print("\n[1/3] 加载 VIIRS 夜间灯光数据...")

f_pre = h5py.File(VIIRS_PRE, 'r')
f_post = h5py.File(VIIRS_POST, 'r')

ntl_pre  = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL'][:]
ntl_post = f_post['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL'][:]
lat_v = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lat'][:]
lon_v = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lon'][:]

# 质量掩膜
qf_pre  = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Mandatory_Quality_Flag'][:]
qf_post = f_post['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Mandatory_Quality_Flag'][:]

f_pre.close()
f_post.close()

# 质量控制：只保留 HQ (flag=0 或 1) 且 NTL>0 的像素
# Mandatory_Quality_Flag: 0=high quality, 1=good, 2=poor, 3=no retrieval
quality_mask = (qf_pre <= 1) & (qf_post <= 1) & (ntl_pre > 0) & (ntl_post > 0)
print(f"  VIIRS 全图: {ntl_pre.shape}, 高质量像素: {np.sum(quality_mask)} / {quality_mask.size}")

# 提取马里乌波尔子区域
lat_2d = lat_v[:, np.newaxis]   # (2400, 1)
lon_2d = lon_v[np.newaxis, :]   # (1, 2400)
aoi_mask = ((lat_2d >= AOI_LAT_MIN) & (lat_2d <= AOI_LAT_MAX) &
            (lon_2d >= AOI_LON_MIN) & (lon_2d <= AOI_LON_MAX))
full_mask = aoi_mask & quality_mask

# 提取子区域
rows = np.any(full_mask, axis=1)
cols = np.any(full_mask, axis=0)
lat_sub = lat_v[rows]
lon_sub = lon_v[cols]

dntl = ntl_post[rows, :][:, cols] - ntl_pre[rows, :][:, cols]
mask_sub = quality_mask[rows, :][:, cols]
ntl_pre_sub = ntl_pre[rows, :][:, cols]

dntl_valid = np.where(mask_sub, dntl, np.nan)

print(f"  AOI 子区域: {dntl.shape[0]}行 × {dntl.shape[1]}列")
print(f"  ΔNTL 范围: [{np.nanmin(dntl_valid):.2f}, {np.nanmax(dntl_valid):.2f}]")
print(f"  ΔNTL 均值: {np.nanmean(dntl_valid):.2f}")
print(f"  灯光下降像素: {np.sum(dntl_valid < 0)} / {np.sum(mask_sub)} ({np.sum(dntl_valid < 0)/np.sum(mask_sub)*100:.1f}%)")

# 保存
np.save(OUT / 'exp_v1_viirs_dntl.npy', dntl_valid.astype(np.float32))
np.save(OUT / 'exp_v1_viirs_lat.npy', lat_sub)
np.save(OUT / 'exp_v1_viirs_lon.npy', lon_sub)
print("  → 已保存 exp_v1_viirs_dntl.npy, lat.npy, lon.npy")

# ============================================================
# 2. Sentinel-2 多光谱处理
# ============================================================
print("\n[2/3] 加载 Sentinel-2 多光谱数据...")

def find_band_in_dir(img_dir, band_pattern):
    """在 Sentinel-2 目录结构中找指定波段的 .jp2 文件"""
    granule_dir = list((img_dir / 'GRANULE').glob('L2A_*'))[0]
    for res in ['R10m', 'R20m', 'R60m']:
        img_data = granule_dir / 'IMG_DATA' / res
        if img_data.exists():
            for f in img_data.glob('*.jp2'):
                if band_pattern in f.name:
                    return f
    raise FileNotFoundError(f"找不到 {band_pattern} in {img_dir}")

def load_s2_band_10m(img_dir, band_pattern):
    """加载 10m 波段，返回 (B, H, W)"""
    fp = find_band_in_dir(img_dir, band_pattern)
    with rasterio.open(fp) as src:
        data = src.read()
    return data

def compute_index(img_dir, index_name, verbose=True):
    """
    计算单一遥感指数。输入为 S2 目录，输出为 NDVI/NBR/NDBI 等。
    均使用 10m 数据以保持最高空间分辨率。
    """
    if index_name == 'NDVI':
        red = load_s2_band_10m(img_dir, 'B04_10m').astype(float)
        nir = load_s2_band_10m(img_dir, 'B08_10m').astype(float)
        idx = (nir - red) / (nir + red + 1e-10)
    elif index_name == 'NDWI':
        green = load_s2_band_10m(img_dir, 'B03_10m').astype(float)
        nir = load_s2_band_10m(img_dir, 'B08_10m').astype(float)
        idx = (green - nir) / (green + nir + 1e-10)
    elif index_name == 'NBR':
        # NBR 需要 SWIR (B12)，只在 20m 分辨率可用
        # 使用 B8A(20m) 替代 B08 作为 NIR
        nir_20 = load_s2_band_20m(img_dir, 'B8A_20m').astype(float)
        swir2_20 = load_s2_band_20m(img_dir, 'B12_20m').astype(float)
        idx = (nir_20 - swir2_20) / (nir_20 + swir2_20 + 1e-10)
    elif index_name == 'NDBI':
        swir1_20 = load_s2_band_20m(img_dir, 'B11_20m').astype(float)
        nir_20 = load_s2_band_20m(img_dir, 'B8A_20m').astype(float)
        idx = (swir1_20 - nir_20) / (swir1_20 + nir_20 + 1e-10)
    else:
        raise ValueError(f"Unknown index: {index_name}")
    return idx.squeeze()  # (H, W)

def load_s2_band_20m(img_dir, band_pattern):
    """加载 20m 波段"""
    fp = find_band_in_dir(img_dir, band_pattern)
    with rasterio.open(fp) as src:
        data = src.read(1)
    return data

def load_scl(img_dir):
    """加载场景分类图层 (20m)"""
    fp = find_band_in_dir(img_dir, 'SCL_20m')
    with rasterio.open(fp) as src:
        return src.read(1)

# 为每个 tile 计算指数
tiles = ['t37tdn', 't37tcn']
results = {}

for tile in tiles:
    print(f"\n  处理 tile {tile.upper()}...")
    pre_key = f'pre_{tile}'
    post_key = f'post_{tile}'

    pre_dir = S2_DIRS[pre_key]
    post_dir = S2_DIRS[post_key]

    # 加载 SCL 掩膜
    scl_pre = load_scl(pre_dir)
    scl_post = load_scl(post_dir)
    # 有效像素: SCL 2-7 (排除 0=no data, 1=saturated, 8-10=cloud, 11=snow)
    scl_mask = ((scl_pre >= 2) & (scl_pre <= 7) &
                (scl_post >= 2) & (scl_post <= 7))

    tile_res = {'mask': scl_mask}

    for idx_name in ['NDVI', 'NDWI', 'NBR', 'NDBI']:
        try:
            pre_idx = compute_index(pre_dir, idx_name)
            post_idx = compute_index(post_dir, idx_name)

            if pre_idx.shape != scl_mask.shape:
                # 需要将 10m mask 上采样到 20m 或 10m 下采样
                # NBR/NDBI 是 20m(NBR的NIR是20m)，NDVI/NDWI是10m
                pass

            d_idx = post_idx - pre_idx
            tile_res[f'd{idx_name}'] = d_idx

            # 统计
            if scl_mask.shape == d_idx.shape:
                valid_d = d_idx[scl_mask]
            elif scl_mask.shape[0] * 2 == d_idx.shape[0]:
                # 20m mask 需要上采样到 10m
                valid_d = d_idx[::2, ::2][scl_mask]
            else:
                valid_d = d_idx[scl_mask[:d_idx.shape[0], :d_idx.shape[1]]]

            print(f"    Δ{idx_name}: mean={np.nanmean(valid_d):.4f}, std={np.nanstd(valid_d):.4f}")
        except Exception as e:
            print(f"    Δ{idx_name}: 计算失败 - {e}")
            tile_res[f'd{idx_name}'] = None

    results[tile] = tile_res

# ============================================================
# 3. 保存 Sentinel-2 中间结果
# ============================================================
print("\n[3/3] 保存预处理结果...")

for tile in tiles:
    tile_res = results[tile]
    np.save(OUT / f'exp_v1_s2_mask_{tile}.npy', tile_res['mask'])
    for idx_name in ['NDVI', 'NDWI', 'NBR', 'NDBI']:
        key = f'd{idx_name}'
        if tile_res.get(key) is not None:
            np.save(OUT / f'exp_v1_s2_{key}_{tile}.npy', tile_res[key].astype(np.float32))
            print(f"  → exp_v1_s2_{key}_{tile}.npy  shape={tile_res[key].shape}")

print("\n" + "=" * 60)
print("预处理完成。输出文件列表：")
for f in sorted(OUT.glob('exp_v1_*')):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name} ({size_mb:.1f} MB)")
print("=" * 60)
