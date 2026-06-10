#!/usr/bin/env python3
"""
阶段 3 — VIIRS 夜光变化处理与 p_viirs 构建

步骤:
  1. 从 H5 提取 DNB_BRDF-Corrected_NTL (战前/战后)
  2. 裁剪到 Mariupol AOI
  3. 计算 NTL_loss 和 NTL_loss_rate
  4. Z-score → sigmoid → p_viirs
  5. 输出 GeoTIFF / PNG / CSV / 中文报告
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import rasterio
from rasterio.transform import from_bounds, Affine

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIIRS_DIR = PROJECT_ROOT / "原始数据" / "VIIRS"
PRE_H5 = VIIRS_DIR / "战前夜光.h5"
POST_H5 = VIIRS_DIR / "战后夜光.h5"
AOI_PATH = PROJECT_ROOT / "config" / "aoi_mariupol.geojson"
OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB = PROJECT_ROOT / "outputs" / "tables"
OUT_MD = PROJECT_ROOT / "outputs"

for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)

# H5 中的数据集路径
NTL_PATH = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL"
LAT_PATH = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lat"
LON_PATH = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lon"
QF_PATH  = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Mandatory_Quality_Flag"
SNOW_PATH = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Snow_Flag"

# AOI
with open(AOI_PATH, 'r') as f:
    aoi_data = json.load(f)
aoi_coords = aoi_data['features'][0]['geometry']['coordinates'][0]
AOI_LON_MIN = min(p[0] for p in aoi_coords)  # 37.42
AOI_LON_MAX = max(p[0] for p in aoi_coords)  # 37.72
AOI_LAT_MIN = min(p[1] for p in aoi_coords)  # 47.05
AOI_LAT_MAX = max(p[1] for p in aoi_coords)  # 47.18

# 输出文件
PRE_TIF  = OUT_DATA / "viirs_ntl_pre.tif"
POST_TIF = OUT_DATA / "viirs_ntl_post.tif"
LOSS_TIF = OUT_DATA / "viirs_ntl_loss.tif"
LOSSRATE_TIF = OUT_DATA / "viirs_ntl_loss_rate.tif"
VALID_TIF = OUT_DATA / "viirs_valid_mask.tif"
DAMAGE_SCORE_TIF = OUT_DATA / "viirs_damage_score.tif"
P_VIIRS_TIF = OUT_DATA / "p_viirs_damage_probability.tif"
PREPOST_PNG = OUT_FIG / "viirs_ntl_pre_post.png"
LOSSRATE_PNG = OUT_FIG / "viirs_ntl_loss_rate.png"
P_VIIRS_PNG = OUT_FIG / "p_viirs_damage_probability.png"
CSV_PATH = OUT_TAB / "p_viirs_statistics.csv"
REPORT_PATH = OUT_MD / "viirs_processing_report.md"


# ============================================================
# 辅助函数
# ============================================================

def extract_aoi(h5_path):
    """
    从 H5 提取 AOI 范围内的 NTL 数据。

    返回:
      ntl_sub: 2D array (float32) — NTL radiance [nW·cm⁻²·sr⁻¹] 子集
      lats_sub: 1D array — 纬度子集
      lons_sub: 1D array — 经度子集
      qf_sub: 2D array (uint8) — 质量控制标志子集
      snow_sub: 2D array (uint8) — 积雪标志子集
    """
    with h5py.File(h5_path, 'r') as h5:
        lats = h5[LAT_PATH][:]
        lons = h5[LON_PATH][:]
        ntl  = h5[NTL_PATH][:]
        qf   = h5[QF_PATH][:] if QF_PATH in h5 else np.zeros_like(ntl, dtype=np.uint8)
        snow = h5[SNOW_PATH][:] if SNOW_PATH in h5 else np.zeros_like(ntl, dtype=np.uint8)

    # 确定索引范围 (注意 lat 是降序)
    lat_is_descending = lats[0] > lats[-1]
    if lat_is_descending:
        lat_idx = np.where((lats <= AOI_LAT_MAX) & (lats >= AOI_LAT_MIN))[0]
    else:
        lat_idx = np.where((lats >= AOI_LAT_MIN) & (lats <= AOI_LAT_MAX))[0]

    lon_idx = np.where((lons >= AOI_LON_MIN) & (lons <= AOI_LON_MAX))[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"AOI 不在数据覆盖范围内! lat=[{lats.min():.2f},{lats.max():.2f}], lon=[{lons.min():.2f},{lons.max():.2f}]")

    lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)

    ntl_sub  = ntl[lat_slice, lon_slice]
    qf_sub   = qf[lat_slice, lon_slice]
    snow_sub = snow[lat_slice, lon_slice]
    lats_sub = lats[lat_idx]
    lons_sub = lons[lon_idx]

    return ntl_sub.astype(np.float32), lats_sub, lons_sub, qf_sub, snow_sub


def make_valid_mask(ntl_pre, ntl_post, qf_pre, qf_post, snow_pre, snow_post):
    """
    构建有效像元掩膜。

    剔除:
    - NTL == fill_value / NaN / inf
    - NTL < 0 (负辐射)
    - 质量控制标志 != 0 (非高质量)
    - 积雪标志 != 0 (有积雪)
    """
    fill_vals = [-999.0, -9999.0, -99999.0, 65535.0, np.nan]

    mask = np.ones(ntl_pre.shape, dtype=bool)

    # Fill values in pre
    for fv in fill_vals:
        if np.isnan(fv):
            mask[np.isnan(ntl_pre)] = False
            mask[np.isnan(ntl_post)] = False
        else:
            mask[np.abs(ntl_pre - fv) < 1e-3] = False
            mask[np.abs(ntl_post - fv) < 1e-3] = False

    # Negatives / non-finite
    mask[~np.isfinite(ntl_pre)] = False
    mask[~np.isfinite(ntl_post)] = False
    mask[ntl_pre < 0] = False
    mask[ntl_post < 0] = False

    # Quality flags (if meaningful: Mandatory_Quality_Flag=0 means best quality)
    # VNP46A3 v2: 0=high quality, 1-5=poor quality
    if qf_pre.max() > 0:
        mask[qf_pre != 0] = False
        mask[qf_post != 0] = False

    # Snow
    if snow_pre.max() > 0:
        mask[snow_pre != 0] = False
        mask[snow_post != 0] = False

    return mask


def compute_ntl_loss(ntl_pre, ntl_post, valid_mask, eps=0.01):
    """
    计算夜光损失。

    NTL_loss = NTL_pre - NTL_post (正值=光减少)
    NTL_loss_rate = (NTL_pre - NTL_post) / (NTL_pre + eps)

    对 loss_rate 做 1%-99% winsorize 截尾。
    """
    ntl_loss = ntl_pre - ntl_post
    ntl_loss[~valid_mask] = np.nan

    # Loss rate
    loss_rate = ntl_loss / (ntl_pre + eps)
    loss_rate[~valid_mask] = np.nan

    # Winsorize loss_rate to [1%, 99%]
    lr_valid = loss_rate[valid_mask]
    if len(lr_valid) > 0:
        lo = np.percentile(lr_valid, 1)
        hi = np.percentile(lr_valid, 99)
        loss_rate_clipped = np.clip(loss_rate, lo, hi)
    else:
        loss_rate_clipped = loss_rate.copy()

    return ntl_loss, loss_rate_clipped


def compute_p_viirs(loss_rate, valid_mask):
    """
    从 NTL_loss_rate 构建 p_viirs。

    步骤:
    1. 对 loss_rate 做 Z-score (μ, σ 来自有效像元)
    2. sigmoid 转换: p = 1 / (1 + exp(-z))
    3. 高 loss_rate → 高 z → p → 1 (更像损害)
    """
    lr_valid = loss_rate[valid_mask]
    mu_lr = np.mean(lr_valid)
    sig_lr = np.std(lr_valid)

    z_score = np.full_like(loss_rate, np.nan)
    z_score[valid_mask] = (loss_rate[valid_mask] - mu_lr) / sig_lr

    # Sigmoid
    p_viirs = 1.0 / (1.0 + np.exp(-z_score))
    p_viirs[~valid_mask] = np.nan

    # Damage score = z_score (for consistency)
    damage_score = z_score.copy()

    return damage_score, p_viirs, mu_lr, sig_lr


# ============================================================
# 输出
# ============================================================

def save_geotiff(data, transform, crs, path, dtype=None, nodata=np.nan, desc=""):
    """保存单波段 GeoTIFF"""
    if dtype is None:
        dtype = data.dtype
    h, w = data.shape
    with rasterio.open(path, 'w', driver='GTiff', height=h, width=w,
                       count=1, dtype=dtype, crs=crs, transform=transform,
                       nodata=nodata, compress='lzw') as dst:
        dst.write(data.astype(dtype), 1)
        if desc:
            dst.set_band_description(1, desc)


def plot_ntl_pre_post(ntl_pre, ntl_post, valid_mask, output_path):
    """绘制战前/战后 NTL 对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    pre_plot = np.where(valid_mask, ntl_pre, np.nan)
    post_plot = np.where(valid_mask, ntl_post, np.nan)
    vmin = 0
    vmax = max(np.nanpercentile(pre_plot, 99), np.nanpercentile(post_plot, 99), 5.0)

    ax1 = axes[0]
    im1 = ax1.imshow(pre_plot, cmap='inferno', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_title('Pre-war NTL: DNB_BRDF-Corrected_NTL', fontsize=12, fontweight='bold')
    ax1.set_xticks([]); ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Radiance (nW/cm2/sr)')

    ax2 = axes[1]
    im2 = ax2.imshow(post_plot, cmap='inferno', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax2.set_title('Post-war NTL: DNB_BRDF-Corrected_NTL', fontsize=12, fontweight='bold')
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Radiance (nW/cm2/sr)')

    fig.suptitle('VIIRS Nighttime Lights — Mariupol AOI\nPre-war vs Post-war',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {output_path}")


def plot_loss_rate(loss_rate, valid_mask, output_path):
    """绘制夜光损失率图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    lr_plot = np.where(valid_mask, loss_rate, np.nan)
    vlim = max(abs(np.nanpercentile(lr_plot, 1)), abs(np.nanpercentile(lr_plot, 99)), 2.0)
    im = ax.imshow(lr_plot, cmap='RdBu_r', vmin=-vlim, vmax=vlim, interpolation='nearest')
    ax.set_title('NTL Loss Rate: (Pre - Post) / (Pre + 0.01)', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {output_path}")


def plot_p_viirs(p_viirs, valid_mask, output_path):
    """绘制 p_viirs 图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    p_plot = np.where(valid_mask, p_viirs, np.nan)
    im = ax.imshow(p_plot, cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('p_viirs: VIIRS Damage Probability', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Probability')
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {output_path}")


def save_statistics(ntl_pre, ntl_post, ntl_loss, loss_rate, p_viirs, valid_mask):
    """计算并保存统计表"""
    rows = [['metric', 'value']]

    def stats(arr, name, mask):
        v = arr[mask]
        vf = v[np.isfinite(v)]
        rows.append([f'{name}_mean', round(float(np.mean(vf)), 6)])
        rows.append([f'{name}_median', round(float(np.median(vf)), 6)])
        rows.append([f'{name}_std', round(float(np.std(vf)), 6)])
        rows.append([f'{name}_p10', round(float(np.percentile(vf, 10)), 6)])
        rows.append([f'{name}_p90', round(float(np.percentile(vf, 90)), 6)])

    stats(ntl_pre, 'ntl_pre', valid_mask)
    stats(ntl_post, 'ntl_post', valid_mask)
    stats(ntl_loss, 'ntl_loss', valid_mask)
    stats(loss_rate, 'ntl_loss_rate', valid_mask)
    stats(p_viirs, 'p_viirs', valid_mask)

    rows.append(['valid_pixel_count', int(valid_mask.sum())])
    rows.append(['total_pixel_count', int(valid_mask.size)])
    rows.append(['valid_pixel_ratio', round(float(valid_mask.sum() / valid_mask.size), 6)])

    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  [CSV] Saved: {CSV_PATH}")


def generate_report(ntl_pre, ntl_post, ntl_loss, loss_rate, p_viirs, valid_mask,
                    mu_lr, sig_lr, shape_info):
    """生成中文报告"""
    v = valid_mask

    def s(arr):
        vf = arr[v]; vf = vf[np.isfinite(vf)]
        return float(np.mean(vf)), float(np.median(vf)), float(np.std(vf))

    pre_m, pre_med, pre_std = s(ntl_pre)
    post_m, post_med, post_std = s(ntl_post)
    loss_m, loss_med, loss_std = s(ntl_loss)
    lr_m, lr_med, lr_std = s(loss_rate)
    p_m, p_med, p_std = s(p_viirs)

    lines = []
    lines.append("# VIIRS 夜光变化处理报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 3 — VIIRS 夜光变化处理与 p_viirs 构建\n")
    lines.append("---\n")

    lines.append("## 1. 输入文件\n")
    lines.append(f"- 战前: `{PRE_H5.relative_to(PROJECT_ROOT)}` (VNP46A3 月产品)\n")
    lines.append(f"- 战后: `{POST_H5.relative_to(PROJECT_ROOT)}` (VNP46A3 月产品)\n")
    lines.append(f"- AOI: `{AOI_PATH.relative_to(PROJECT_ROOT)}`\n")

    lines.append("\n## 2. H5 Dataset 清单\n")
    lines.append("| Dataset | Shape | Dtype |\n")
    lines.append("|---------|-------|-------|\n")
    lines.append(f"| DNB_BRDF-Corrected_NTL (选中) | (2400, 2400) | float32 |\n")
    lines.append(f"| Gap_Filled_DNB_BRDF-Corrected_NTL | (2400, 2400) | float32 |\n")
    lines.append(f"| Mandatory_Quality_Flag | (2400, 2400) | uint8 |\n")
    lines.append(f"| QF_Cloud_Mask | (2400, 2400) | uint16 |\n")
    lines.append(f"| Snow_Flag | (2400, 2400) | uint8 |\n")
    lines.append(f"| lat | (2400,) | float64 |\n")
    lines.append(f"| lon | (2400,) | float64 |\n")

    lines.append("\n## 3. 选用的夜光亮度层\n")
    lines.append(f"- **Dataset**: `{NTL_PATH}`\n")
    lines.append("- 说明: BRDF 校正后的月合成 NTL 辐射亮度 (Black Marble VNP46A3 v2.0)\n")
    lines.append(f"- Shape: (2400, 2400)\n")
    lines.append("- Dtype: float32\n")
    lines.append("- 单位: nW·cm⁻²·sr⁻¹\n")
    lines.append("- CRS: EPSG:4326 (地理坐标, 15 arcsec 网格)\n")
    lines.append(f"- 全图范围: lat [{40.00}, {50.00}], lon [{30.00}, {40.00}]\n")
    lines.append(f"- 分辨率: 15 arcsec (~500m)\n")
    lines.append(f"- AOI 子集: {shape_info}\n")

    lines.append("\n## 4. AOI 覆盖\n")
    lines.append(f"- Mariupol AOI (EPSG:4326): lon=[{AOI_LON_MIN}, {AOI_LON_MAX}], lat=[{AOI_LAT_MIN}, {AOI_LAT_MAX}]\n")
    lines.append("- VIIRS 数据完整覆盖该 AOI\n")
    lines.append(f"- AOI 内有效子网格: {shape_info}\n")

    lines.append("\n## 5. 数据质量处理\n")
    lines.append("- 剔除条件:\n")
    lines.append("  - NTL < 0 或非有限值\n")
    lines.append("  - Mandatory_Quality_Flag != 0 (非最高质量)\n")
    lines.append("  - Snow_Flag != 0 (有积雪)\n")
    lines.append(f"- 有效像元: {v.sum():,} / {v.size:,} ({100*v.sum()/v.size:.2f}%)\n")

    lines.append("\n## 6. NTL 统计\n")
    lines.append("| 指标 | 战前 | 战后 |\n")
    lines.append("|------|------|------|\n")
    lines.append(f"| 均值 | {pre_m:.4f} | {post_m:.4f} |\n")
    lines.append(f"| 中位数 | {pre_med:.4f} | {post_med:.4f} |\n")
    lines.append(f"| 标准差 | {pre_std:.4f} | {post_std:.4f} |\n")

    lines.append("\n## 7. NTL_loss 和 NTL_loss_rate 统计\n")
    lines.append(f"- **NTL_loss = NTL_pre - NTL_post** (正值=光减少)\n")
    lines.append(f"- **NTL_loss_rate = (NTL_pre - NTL_post) / (NTL_pre + 0.01)**\n")
    lines.append(f"- loss_rate 已做 1%-99% winsorize 截尾\n")
    lines.append("| 指标 | NTL_loss | NTL_loss_rate |\n")
    lines.append("|------|----------|---------------|\n")
    lines.append(f"| 均值 | {loss_m:.4f} | {lr_m:.4f} |\n")
    lines.append(f"| 中位数 | {loss_med:.4f} | {lr_med:.4f} |\n")
    lines.append(f"| 标准差 | {loss_std:.4f} | {lr_std:.4f} |\n")

    lines.append("\n## 8. p_viirs 构建\n")
    lines.append("### 方法\n")
    lines.append("1. Z-score 标准化: z = (NTL_loss_rate - μ) / σ\n")
    lines.append(f"   - μ = {mu_lr:.6f}, σ = {sig_lr:.6f}\n")
    lines.append("2. Sigmoid 转换: p_viirs = 1 / (1 + exp(-z))\n")
    lines.append("3. 高 loss_rate → 高 p_viirs (更像损害)\n")
    lines.append("### 统计\n")
    lines.append(f"- p_viirs 均值: {p_m:.4f}\n")
    lines.append(f"- p_viirs 中位数: {p_med:.4f}\n")
    lines.append(f"- p_viirs 标准差: {p_std:.4f}\n")

    lines.append("\n## 9. 输出文件清单\n")
    outputs = [
        (PRE_TIF, "战前 NTL GeoTIFF, float32, EPSG:4326"),
        (POST_TIF, "战后 NTL GeoTIFF, float32, EPSG:4326"),
        (LOSS_TIF, "NTL 损失 GeoTIFF, float32"),
        (LOSSRATE_TIF, "NTL 损失率 GeoTIFF, float32, winsorized"),
        (VALID_TIF, "有效像元掩膜 GeoTIFF, uint8"),
        (DAMAGE_SCORE_TIF, "VIIRS 损害分数 GeoTIFF (z-score)"),
        (P_VIIRS_TIF, "p_viirs 损害概率 GeoTIFF, float32, [0,1]"),
        (PREPOST_PNG, "战前/战后 NTL 对比图"),
        (LOSSRATE_PNG, "NTL 损失率图"),
        (P_VIIRS_PNG, "p_viirs 损害概率图"),
        (CSV_PATH, "统计表"),
        (REPORT_PATH, "本报告"),
    ]
    for p, desc in outputs:
        lines.append(f"| `{p.relative_to(PROJECT_ROOT)}` | {desc} |\n")

    lines.append("\n## 10. 阶段结论\n")
    lines.append("> ✅ **VIIRS 夜光变化处理完成, 可以进入下一阶段。**\n")
    lines.append("\n**p_viirs 说明**:\n")
    lines.append(f"1. 基于 Z-score + sigmoid 从 NTL 损失率构建\n")
    lines.append(f"2. p_viirs 表示\"夜光下降异常程度对应的损害候选概率\"\n")
    lines.append(f"3. 约 {(p_viirs[v]>0.5).sum()/v.sum()*100:.1f}% 的有效像元 p_viirs > 0.5\n")
    lines.append(f"4. 空间分辨率 ~500m (VIIRS 原始), 未重采样到 Sentinel-2 的 20m\n")
    lines.append(f"5. 可与 Sentinel-2 损害分数在统一网格上融合\n")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {REPORT_PATH}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("VIIRS 夜光变化处理与 p_viirs 构建")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- 1. 提取 NTL ----
    print("\n[1] Extracting NTL from H5 files...")
    ntl_pre, lats, lons, qf_pre, snow_pre = extract_aoi(PRE_H5)
    ntl_post, _, _, qf_post, snow_post = extract_aoi(POST_H5)

    h, w = ntl_pre.shape
    print(f"    AOI subset shape: ({h}, {w})")
    print(f"    Lat range: [{lats[0]:.4f}, {lats[-1]:.4f}] (step={abs(lats[1]-lats[0]):.5f} deg)")
    print(f"    Lon range: [{lons[0]:.4f}, {lons[-1]:.4f}] (step={abs(lons[1]-lons[0]):.5f} deg)")

    # ---- 2. 有效像元掩膜 ----
    print("\n[2] Building valid pixel mask...")
    valid_mask = make_valid_mask(ntl_pre, ntl_post, qf_pre, qf_post, snow_pre, snow_post)
    print(f"    Valid: {valid_mask.sum():,} / {valid_mask.size:,} ({100*valid_mask.sum()/valid_mask.size:.2f}%)")

    # ---- 3. 计算 NTL 变化 ----
    print("\n[3] Computing NTL loss and loss rate...")
    ntl_loss, loss_rate = compute_ntl_loss(ntl_pre, ntl_post, valid_mask)

    # 统计
    lr_v = loss_rate[valid_mask]
    lr_vf = lr_v[np.isfinite(lr_v)]
    print(f"    NTL_loss mean={np.mean(ntl_loss[valid_mask]):.4f}, median={np.median(ntl_loss[valid_mask]):.4f}")
    print(f"    NTL_loss_rate mean={np.mean(lr_vf):.4f}, median={np.median(lr_vf):.4f}")
    print(f"    NTL_loss_rate range (winsorized): [{lr_vf.min():.4f}, {lr_vf.max():.4f}]")

    # ---- 4. 构建 p_viirs ----
    print("\n[4] Building p_viirs damage probability...")
    damage_score, p_viirs, mu_lr, sig_lr = compute_p_viirs(loss_rate, valid_mask)
    p_v = p_viirs[valid_mask]
    print(f"    loss_rate μ={mu_lr:.4f} σ={sig_lr:.4f}")
    print(f"    p_viirs mean={np.mean(p_v):.4f}, median={np.median(p_v):.4f}")
    print(f"    p_viirs > 0.5: {(p_v>0.5).sum():,} ({(p_v>0.5).sum()/len(p_v)*100:.1f}%)")

    # ---- 5. 确定地理变换 ----
    lat_res = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.00417
    lon_res = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.00417
    lat_is_desc = lats[0] > lats[-1] if len(lats) > 1 else True
    if lat_is_desc:
        transform = Affine.translation(lons[0] - lon_res/2, lats[0] + lat_res/2) * Affine.scale(lon_res, -lat_res)
    else:
        transform = Affine.translation(lons[0] - lon_res/2, lats[0] - lat_res/2) * Affine.scale(lon_res, lat_res)
    crs = "EPSG:4326"
    shape_info = f"({h}, {w})"

    # ---- 6. 将无效像元设为 NaN ----
    print("\n[5] Applying NaN mask to invalid pixels...")
    ntl_pre_out = ntl_pre.copy()
    ntl_post_out = ntl_post.copy()
    ntl_pre_out[~valid_mask] = np.nan
    ntl_post_out[~valid_mask] = np.nan
    ntl_loss_out = ntl_loss.copy()
    ntl_loss_out[~valid_mask] = np.nan
    loss_rate_out = loss_rate.copy()
    loss_rate_out[~valid_mask] = np.nan
    damage_score_out = damage_score.copy()
    damage_score_out[~valid_mask] = np.nan
    p_viirs_out = p_viirs.copy()
    p_viirs_out[~valid_mask] = np.nan

    # ---- 7. 保存 GeoTIFF ----
    print("\n[6] Saving GeoTIFFs...")
    save_geotiff(ntl_pre_out, transform, crs, PRE_TIF, np.float32, desc="Pre-war NTL (DNB_BRDF-Corrected, nW/cm2/sr)")
    print(f"    Saved: {PRE_TIF}")
    save_geotiff(ntl_post_out, transform, crs, POST_TIF, np.float32, desc="Post-war NTL (DNB_BRDF-Corrected, nW/cm2/sr)")
    print(f"    Saved: {POST_TIF}")
    save_geotiff(ntl_loss_out, transform, crs, LOSS_TIF, np.float32, desc="NTL Loss (Pre - Post)")
    print(f"    Saved: {LOSS_TIF}")
    save_geotiff(loss_rate_out, transform, crs, LOSSRATE_TIF, np.float32, desc="NTL Loss Rate (winsorized)")
    print(f"    Saved: {LOSSRATE_TIF}")
    save_geotiff(valid_mask.astype(np.uint8), transform, crs, VALID_TIF, np.uint8, 0, desc="Valid Pixel Mask")
    print(f"    Saved: {VALID_TIF}")
    save_geotiff(damage_score_out, transform, crs, DAMAGE_SCORE_TIF, np.float32, desc="VIIRS Damage Score (z-score)")
    print(f"    Saved: {DAMAGE_SCORE_TIF}")
    save_geotiff(p_viirs_out, transform, crs, P_VIIRS_TIF, np.float32, desc="p_viirs Damage Probability")
    print(f"    Saved: {P_VIIRS_TIF}")

    # ---- 7. 绘制图表 ----
    print("\n[6] Plotting figures...")
    plot_ntl_pre_post(ntl_pre, ntl_post, valid_mask, PREPOST_PNG)
    plot_loss_rate(loss_rate, valid_mask, LOSSRATE_PNG)
    plot_p_viirs(p_viirs, valid_mask, P_VIIRS_PNG)

    # ---- 8. 保存统计和报告 ----
    print("\n[7] Saving statistics and report...")
    save_statistics(ntl_pre, ntl_post, ntl_loss, loss_rate, p_viirs, valid_mask)
    generate_report(ntl_pre, ntl_post, ntl_loss, loss_rate, p_viirs, valid_mask,
                    mu_lr, sig_lr, shape_info)

    print("\n" + "=" * 70)
    print("VIIRS 处理完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
