#!/usr/bin/env python3
"""
阶段 4 — 统一 500m 网格与 Sentinel-2/VIIRS 双源融合

步骤:
  1. 在 EPSG:32637 下建立 500m 鱼网网格
  2. Sentinel-2 损害分数聚合到每个网格
  3. VIIRS p_viirs 重投影后聚合到每个网格
  4. Logit-平均融合 p_fused
  5. 不确定性与证据类型
  6. 全部输出 (GeoJSON, CSV, PNG, MD 报告)
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.mask import mask
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from shapely.geometry import box, Polygon, Point, mapping
from shapely import ops
import geopandas as gpd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
S2_SCORE_TIF = PROJECT_ROOT / "data" / "processed" / "s2_main_damage_score.tif"
P_VIIRS_TIF  = PROJECT_ROOT / "data" / "processed" / "p_viirs_damage_probability.tif"
VIIRS_LR_TIF = PROJECT_ROOT / "data" / "processed" / "viirs_ntl_loss_rate.tif"
VIIRS_VALID_TIF = PROJECT_ROOT / "data" / "processed" / "viirs_valid_mask.tif"
AOI_PATH = PROJECT_ROOT / "config" / "aoi_mariupol.geojson"

OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG  = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB  = PROJECT_ROOT / "outputs" / "tables"
OUT_MD   = PROJECT_ROOT / "outputs"
for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)

GRID_SIZE_M = 500  # 500m grid

# 输出文件
GRID_GEOJSON = OUT_DATA / "grid_500m.geojson"
GRID_FEAT_CSV = OUT_DATA / "grid_features_500m.csv"
FUSION_CSV = OUT_DATA / "grid_fusion_s2_viirs.csv"
FUSION_GEOJSON = OUT_DATA / "grid_fusion_s2_viirs.geojson"

MAP_P_S2 = OUT_FIG / "grid_p_s2_map.png"
MAP_P_VIIRS = OUT_FIG / "grid_p_viirs_map.png"
MAP_FUSED = OUT_FIG / "fused_damage_probability_map.png"
MAP_UNCERT = OUT_FIG / "fused_uncertainty_map.png"
MAP_EVIDENCE = OUT_FIG / "evidence_type_map.png"
SCATTER_PNG = OUT_FIG / "p_s2_vs_p_viirs_scatter.png"

FUSION_STATS_CSV = OUT_TAB / "fusion_summary_statistics.csv"
EVIDENCE_CSV = OUT_TAB / "evidence_type_counts.csv"
AGREEMENT_CSV = OUT_TAB / "sensor_agreement_statistics.csv"
REPORT_PATH = OUT_MD / "fusion_s2_viirs_report.md"

EPS = 1e-10  # 防止除零


# ============================================================
# 1. 构建 500m 网格
# ============================================================

def build_grid_500m(aoi_gdf):
    """在 EPSG:32637 下建立 500m 鱼网网格, 返回 GeoDataFrame"""
    aoi_utm = aoi_gdf.to_crs("EPSG:32637")
    minx, miny, maxx, maxy = aoi_utm.total_bounds

    # 对齐到 500m 整数倍
    x0 = np.floor(minx / GRID_SIZE_M) * GRID_SIZE_M
    y0 = np.floor(miny / GRID_SIZE_M) * GRID_SIZE_M
    x1 = np.ceil(maxx / GRID_SIZE_M) * GRID_SIZE_M
    y1 = np.ceil(maxy / GRID_SIZE_M) * GRID_SIZE_M

    cols = int(round((x1 - x0) / GRID_SIZE_M))
    rows = int(round((y1 - y0) / GRID_SIZE_M))

    cells = []
    for row in range(rows):
        for col in range(cols):
            cx = x0 + col * GRID_SIZE_M
            cy = y0 + row * GRID_SIZE_M
            cell_box = box(cx, cy, cx + GRID_SIZE_M, cy + GRID_SIZE_M)
            cells.append({
                'grid_id': row * cols + col,
                'grid_row': row,
                'grid_col': col,
                'geometry': cell_box,
            })

    grid_gdf = gpd.GeoDataFrame(cells, crs="EPSG:32637")

    # 仅保留与 AOI 相交的网格
    grid_gdf = grid_gdf[grid_gdf.intersects(aoi_utm.unary_union)].copy()
    grid_gdf = grid_gdf.reset_index(drop=True)
    grid_gdf['grid_id'] = range(len(grid_gdf))

    # 计算中心和面积
    centroids_utm = grid_gdf.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_utm, crs="EPSG:32637").to_crs("EPSG:4326")
    grid_gdf['lon_center'] = centroids_4326.x
    grid_gdf['lat_center'] = centroids_4326.y
    grid_gdf['area_m2'] = grid_gdf.geometry.area

    print(f"  Grid: {len(grid_gdf)} cells, "
          f"cols={cols}, rows={rows}, "
          f"bounds_UTM=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")

    return grid_gdf


# ============================================================
# 2. 栅格聚合
# ============================================================

def aggregate_raster_to_grid(grid_gdf, raster_path, prefix, min_valid_ratio=0.2):
    """
    将栅格聚合到网格。

    返回更新后的 grid_gdf (原地修改并返回)。
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_data = src.read(1)

        # 对每个网格做 zonal stats
        means, medians, stds, valid_counts, valid_ratios = [], [], [], [], []

        for idx, row in grid_gdf.iterrows():
            geom_shapely = row.geometry
            # 转为 raster CRS
            if raster_crs != grid_gdf.crs:
                geom_raster_crs = gpd.GeoSeries([geom_shapely], crs=grid_gdf.crs).to_crs(raster_crs).iloc[0]
            else:
                geom_raster_crs = geom_shapely

            try:
                out_image, out_transform = mask(src, [geom_raster_crs], crop=True, nodata=np.nan)
                pixel_values = out_image[0].ravel()
                pixel_valid = pixel_values[np.isfinite(pixel_values)]

                n_total = len(pixel_values)
                n_valid = len(pixel_valid)
                ratio = n_valid / n_total if n_total > 0 else 0

                if n_valid >= 1:
                    means.append(float(np.mean(pixel_valid)))
                    medians.append(float(np.median(pixel_valid)))
                    stds.append(float(np.std(pixel_valid)) if n_valid > 1 else 0.0)
                else:
                    means.append(np.nan)
                    medians.append(np.nan)
                    stds.append(np.nan)
                valid_counts.append(n_valid)
                valid_ratios.append(ratio)
            except Exception:
                means.append(np.nan)
                medians.append(np.nan)
                stds.append(np.nan)
                valid_counts.append(0)
                valid_ratios.append(0.0)

        grid_gdf[f'{prefix}_mean'] = means
        grid_gdf[f'{prefix}_median'] = medians
        grid_gdf[f'{prefix}_std'] = stds
        grid_gdf[f'{prefix}_valid_count'] = valid_counts
        grid_gdf[f'{prefix}_valid_ratio'] = valid_ratios
        grid_gdf[f'{prefix}_low_quality'] = [
            (r < min_valid_ratio) or (np.isnan(r)) for r in valid_ratios
        ]

    return grid_gdf


def aggregate_viirs_to_grid(grid_gdf, p_viirs_path, loss_rate_path, valid_mask_path, prefix='p_viirs'):
    """
    将 VIIRS 数据 (EPSG:4326) 重投影到 EPSG:32637 后聚合到网格。
    """
    # 先重投影 VIIRS 到 EPSG:32637
    print("    Reprojecting VIIRS to EPSG:32637...")
    with rasterio.open(p_viirs_path) as src:
        # 计算重投影后的 bounds 和 shape
        dst_crs = CRS.from_epsg(32637)
        # 目标分辨率 ~500m
        dst_res = 500.0
        left, bottom, right, top = transform_bounds(src.crs, dst_crs, *src.bounds)
        dst_width = int(round((right - left) / dst_res))
        dst_height = int(round((top - bottom) / dst_res))
        dst_transform = from_bounds(left, bottom, right, top, dst_width, dst_height)

        p_viirs_utm = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src.read(1),
            destination=p_viirs_utm,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    with rasterio.open(loss_rate_path) as src:
        lr_utm = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src.read(1),
            destination=lr_utm,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    with rasterio.open(valid_mask_path) as src:
        valid_utm = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src.read(1).astype(np.float32),
            destination=valid_utm,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=np.nan,
        )

    # 现用 rasterio.mask 对每个网格提取
    p_v_mean, p_v_median, p_v_std, p_v_count, p_v_ratio = [], [], [], [], []
    lr_mean, lr_median = [], []
    low_quality = []

    for idx, row in grid_gdf.iterrows():
        geom = row.geometry  # EPSG:32637

        # p_viirs
        pv = _extract_raster_values(p_viirs_utm, dst_transform, dst_crs, geom)
        pv_valid = pv[np.isfinite(pv)]
        n_pv = len(pv_valid)
        n_tot = len(pv)
        r_pv = n_pv / n_tot if n_tot > 0 else 0

        if n_pv > 0:
            p_v_mean.append(float(np.mean(pv_valid)))
            p_v_median.append(float(np.median(pv_valid)))
            p_v_std.append(float(np.std(pv_valid)) if n_pv > 1 else 0.0)
        else:
            p_v_mean.append(np.nan); p_v_median.append(np.nan); p_v_std.append(np.nan)
        p_v_count.append(n_pv); p_v_ratio.append(r_pv)

        # loss_rate
        lr = _extract_raster_values(lr_utm, dst_transform, dst_crs, geom)
        lr_valid = lr[np.isfinite(lr)]
        if len(lr_valid) > 0:
            lr_mean.append(float(np.mean(lr_valid)))
            lr_median.append(float(np.median(lr_valid)))
        else:
            lr_mean.append(np.nan); lr_median.append(np.nan)

        low_quality.append(r_pv < 0.2 or np.isnan(r_pv))

    grid_gdf[f'{prefix}_mean'] = p_v_mean
    grid_gdf[f'{prefix}_median'] = p_v_median
    grid_gdf[f'{prefix}_std'] = p_v_std
    grid_gdf[f'{prefix}_valid_count'] = p_v_count
    grid_gdf[f'{prefix}_valid_ratio'] = p_v_ratio
    grid_gdf[f'{prefix}_low_quality'] = low_quality
    grid_gdf['viirs_loss_rate_mean'] = lr_mean
    grid_gdf['viirs_loss_rate_median'] = lr_median

    return grid_gdf


def _extract_raster_values(data, transform, crs, geometry):
    """用 mask 从内存中的 raster 数组提取几何范围内的像元值"""
    h, w = data.shape
    # 创建内存 raster
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff', height=h, width=w, count=1,
            dtype=np.float32, crs=crs, transform=transform, nodata=np.nan
        ) as tmp:
            tmp.write(data.astype(np.float32), 1)
        with memfile.open() as src:
            try:
                out, _ = mask(src, [geometry], crop=True, nodata=np.nan)
                return out[0].ravel()
            except Exception:
                return np.array([])


# ============================================================
# 3. 融合
# ============================================================

def logit(p):
    """logit 变换, 自动截断"""
    p_clip = np.clip(p, 0.001, 0.999)
    return np.log(p_clip / (1.0 - p_clip))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_fusion(grid_gdf):
    """Logit-平均融合 p_s2 和 p_viirs"""
    p_s2 = grid_gdf['p_s2_mean'].values
    p_viirs = grid_gdf['p_viirs_mean'].values

    p_fused = np.full(len(grid_gdf), np.nan)
    fusion_source = np.full(len(grid_gdf), 'no_data', dtype=object)

    has_s2 = ~np.isnan(p_s2)
    has_viirs = ~np.isnan(p_viirs)
    both = has_s2 & has_viirs
    s2_only = has_s2 & ~has_viirs
    viirs_only = ~has_s2 & has_viirs

    # Both sensors
    if both.any():
        ls2 = logit(p_s2[both])
        lv = logit(p_viirs[both])
        l_fused = 0.5 * ls2 + 0.5 * lv
        p_fused[both] = sigmoid(l_fused)
        fusion_source[both] = 'both'

    # Single source
    p_fused[s2_only] = p_s2[s2_only]
    fusion_source[s2_only] = 's2_only'
    p_fused[viirs_only] = p_viirs[viirs_only]
    fusion_source[viirs_only] = 'viirs_only'

    grid_gdf['p_fused'] = p_fused
    grid_gdf['fusion_source'] = fusion_source

    # 传感器分歧
    disagreement = np.full(len(grid_gdf), np.nan)
    disagreement[both] = np.abs(p_s2[both] - p_viirs[both])
    grid_gdf['disagreement'] = disagreement

    # 质量不确定性
    s2_ratio = np.nan_to_num(grid_gdf['p_s2_valid_ratio'].values, nan=0.0)
    viirs_ratio = np.nan_to_num(grid_gdf['p_viirs_valid_ratio'].values, nan=0.0)
    quality_unc = 0.5 * (1.0 - s2_ratio) + 0.5 * (1.0 - viirs_ratio)
    grid_gdf['quality_uncertainty'] = quality_unc

    # 综合不确定性
    d_vec = np.nan_to_num(disagreement, nan=0.0)
    uncertainty = 0.7 * d_vec + 0.3 * quality_unc
    grid_gdf['uncertainty'] = uncertainty

    return grid_gdf


def classify_evidence(grid_gdf):
    """证据类型分类"""
    p_s2 = grid_gdf['p_s2_mean'].values
    p_v = grid_gdf['p_viirs_mean'].values
    src = grid_gdf['fusion_source'].values

    evidence = np.full(len(grid_gdf), 'no_data', dtype=object)

    both = (src == 'both')
    s2_high = p_s2 >= 0.5
    s2_low = p_s2 < 0.5
    v_high = p_v >= 0.5
    v_low = p_v < 0.5

    evidence[both & s2_high & v_high] = 'both_high'
    evidence[both & s2_high & v_low] = 's2_high_viirs_low'
    evidence[both & s2_low & v_high] = 's2_low_viirs_high'
    evidence[both & s2_low & v_low] = 'both_low'
    evidence[src == 's2_only'] = 'single_source'
    evidence[src == 'viirs_only'] = 'single_source'

    grid_gdf['evidence_type'] = evidence
    return grid_gdf


# ============================================================
# 4. 输出
# ============================================================

def plot_grid_map(grid_gdf, column, title, cmap, vmin, vmax, output_path, cbar_label=''):
    """绘制网格面地图"""
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_plot = grid_gdf.to_crs("EPSG:4326")
    gdf_plot.plot(
        column=column, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolor='none', linewidth=0, legend=False
    )
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {output_path}")


def plot_evidence_map(grid_gdf, output_path):
    """绘制证据类型分类图"""
    evidence_colors = {
        'both_high': '#253494',
        's2_high_viirs_low': '#2c7fb8',
        's2_low_viirs_high': '#41b6c4',
        'both_low': '#ffffcc',
        'single_source': '#d9d9d9',
        'no_data': '#999999',
    }
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_plot = grid_gdf.to_crs("EPSG:4326")
    for etype, color in evidence_colors.items():
        subset = gdf_plot[gdf_plot['evidence_type'] == etype]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='none', linewidth=0,
                        label=etype.replace('_', ' ').title())

    ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax.set_title('Evidence Type: S2 vs VIIRS Agreement', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {output_path}")


def plot_scatter(grid_gdf, output_path):
    """绘制 p_s2 vs p_viirs 散点图"""
    df = grid_gdf[grid_gdf['fusion_source'] == 'both'].dropna(
        subset=['p_s2_mean', 'p_viirs_mean']
    )
    if len(df) < 5:
        print("    Not enough data points for scatter plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(df['p_s2_mean'], df['p_viirs_mean'], c=df['disagreement'],
               cmap='Reds', s=20, alpha=0.7, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='y=x')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

    r, p_val = spearmanr(df['p_s2_mean'], df['p_viirs_mean'])
    ax.set_xlabel('p_s2 (Sentinel-2 Damage Score)')
    ax.set_ylabel('p_viirs (VIIRS Damage Probability)')
    ax.set_title(f'p_s2 vs p_viirs — Spearman ρ = {r:.4f} (p={p_val:.4f})',
                 fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {output_path}")

    return r, p_val


def save_outputs(grid_gdf):
    """保存所有 GeoJSON / CSV"""
    # GeoJSON (EPSG:4326)
    gdf_4326 = grid_gdf.to_crs("EPSG:4326")

    # 基础网格
    grid_cols = ['grid_id', 'grid_row', 'grid_col', 'lon_center', 'lat_center', 'area_m2', 'geometry']
    gdf_4326[grid_cols].to_file(GRID_GEOJSON, driver='GeoJSON')
    print(f"  [GeoJSON] Saved: {GRID_GEOJSON}")

    # 网格特征 CSV
    feat_cols = [c for c in grid_gdf.columns if c != 'geometry']
    grid_gdf[feat_cols].to_csv(GRID_FEAT_CSV, index=False, encoding='utf-8')
    print(f"  [CSV] Saved: {GRID_FEAT_CSV}")

    # 融合结果 CSV
    fusion_cols = ['grid_id', 'grid_row', 'grid_col', 'lon_center', 'lat_center',
                   'p_s2_mean', 'p_s2_median', 'p_s2_std', 'p_s2_valid_ratio',
                   'p_viirs_mean', 'p_viirs_median', 'p_viirs_std', 'p_viirs_valid_ratio',
                   'viirs_loss_rate_mean', 'viirs_loss_rate_median',
                   'p_fused', 'fusion_source', 'disagreement', 'uncertainty',
                   'evidence_type',
                   'p_s2_low_quality', 'p_viirs_low_quality']
    available_fusion = [c for c in fusion_cols if c in grid_gdf.columns]
    grid_gdf[available_fusion].to_csv(FUSION_CSV, index=False, encoding='utf-8')
    print(f"  [CSV] Saved: {FUSION_CSV}")

    # 融合 GeoJSON
    geojson_cols = available_fusion + ['geometry']
    gdf_4326[geojson_cols].to_file(FUSION_GEOJSON, driver='GeoJSON')
    print(f"  [GeoJSON] Saved: {FUSION_GEOJSON}")


def save_statistics(grid_gdf, spearman_r, spearman_p):
    """保存统计表"""
    df = grid_gdf

    # Evidence type counts
    evidence_counts = df['evidence_type'].value_counts()
    with open(EVIDENCE_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['evidence_type', 'count', 'ratio'])
        for etype, cnt in evidence_counts.items():
            w.writerow([etype, cnt, round(cnt / len(df), 6)])

    # Sensor agreement
    both_df = df[df['fusion_source'] == 'both']
    with open(AGREEMENT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        if len(both_df) > 0:
            w.writerow(['spearman_r', round(spearman_r, 6)])
            w.writerow(['spearman_p', round(spearman_p, 6)])
            w.writerow(['disagreement_mean', round(both_df['disagreement'].mean(), 6)])
            w.writerow(['disagreement_median', round(both_df['disagreement'].median(), 6)])
            w.writerow(['p_s2_mean_of_both', round(both_df['p_s2_mean'].mean(), 6)])
            w.writerow(['p_viirs_mean_of_both', round(both_df['p_viirs_mean'].mean(), 6)])
            # Agreement at 0.5
            s2_h = both_df['p_s2_mean'] >= 0.5
            v_h = both_df['p_viirs_mean'] >= 0.5
            w.writerow(['both_high_count', int((s2_h & v_h).sum())])
            w.writerow(['s2_high_viirs_low_count', int((s2_h & ~v_h).sum())])
            w.writerow(['s2_low_viirs_high_count', int((~s2_h & v_h).sum())])
            w.writerow(['both_low_count', int((~s2_h & ~v_h).sum())])

    print(f"  [CSV] Saved: {EVIDENCE_CSV}")
    print(f"  [CSV] Saved: {AGREEMENT_CSV}")

    return evidence_counts


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("统一 500m 网格与 Sentinel-2/VIIRS 双源融合")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- 检查输入 ----
    print("\n[0] Checking input files...")
    for fpath, name in [
        (S2_SCORE_TIF, "S2 damage score"),
        (P_VIIRS_TIF, "p_viirs"),
        (VIIRS_LR_TIF, "VIIRS loss rate"),
        (VIIRS_VALID_TIF, "VIIRS valid mask"),
    ]:
        if not fpath.exists():
            print(f"    ERROR: {name} not found: {fpath}")
            sys.exit(1)
        print(f"    [OK] {name}")
    if not AOI_PATH.exists():
        print(f"    ERROR: AOI not found: {AOI_PATH}")
        sys.exit(1)

    # Load AOI
    with open(AOI_PATH, 'r') as f:
        aoi_data = json.load(f)
    aoi_gdf = gpd.GeoDataFrame.from_features(aoi_data["features"], crs="EPSG:4326")

    # ---- 1. 构建网格 ----
    print("\n[1] Building 500m grid...")
    grid_gdf = build_grid_500m(aoi_gdf)

    # ---- 2. 聚合 S2 ----
    print("\n[2] Aggregating Sentinel-2 damage score to grid...")
    grid_gdf = aggregate_raster_to_grid(grid_gdf, S2_SCORE_TIF, 'p_s2')

    s2_valid = grid_gdf['p_s2_mean'].notna().sum()
    print(f"    Grid cells with S2 data: {s2_valid}/{len(grid_gdf)}")

    # ---- 3. 聚合 VIIRS ----
    print("\n[3] Aggregating VIIRS to grid...")
    grid_gdf = aggregate_viirs_to_grid(
        grid_gdf, P_VIIRS_TIF, VIIRS_LR_TIF, VIIRS_VALID_TIF
    )
    viirs_valid = grid_gdf['p_viirs_mean'].notna().sum()
    print(f"    Grid cells with VIIRS data: {viirs_valid}/{len(grid_gdf)}")

    # ---- 4. 融合 ----
    print("\n[4] Computing logit-average fusion...")
    grid_gdf = compute_fusion(grid_gdf)
    grid_gdf = classify_evidence(grid_gdf)

    both = (grid_gdf['fusion_source'] == 'both').sum()
    s2_only = (grid_gdf['fusion_source'] == 's2_only').sum()
    viirs_only = (grid_gdf['fusion_source'] == 'viirs_only').sum()
    no_data = (grid_gdf['fusion_source'] == 'no_data').sum()
    print(f"    Both sensors: {both}, S2 only: {s2_only}, VIIRS only: {viirs_only}, No data: {no_data}")

    # ---- 5. 绘图 ----
    print("\n[5] Plotting maps...")
    plot_grid_map(grid_gdf, 'p_s2_mean', 'p_s2: Sentinel-2 Damage Score (Grid Mean)',
                  'YlOrRd', 0, 1, MAP_P_S2, 'p_s2')
    plot_grid_map(grid_gdf, 'p_viirs_mean', 'p_viirs: VIIRS Damage Probability (Grid Mean)',
                  'YlOrRd', 0, 1, MAP_P_VIIRS, 'p_viirs')
    plot_grid_map(grid_gdf, 'p_fused', 'p_fused: Fused Damage Probability (Logit-Average)',
                  'YlOrRd', 0, 1, MAP_FUSED, 'p_fused')
    plot_grid_map(grid_gdf, 'uncertainty', 'Uncertainty: 0.7 × Disagreement + 0.3 × Quality',
                  'viridis', 0, min(1.0, grid_gdf['uncertainty'].max()), MAP_UNCERT, 'Uncertainty')
    plot_evidence_map(grid_gdf, MAP_EVIDENCE)
    spearman_r, spearman_p = plot_scatter(grid_gdf, SCATTER_PNG)

    # ---- 6. 保存文件 ----
    print("\n[6] Saving GeoJSON and CSV outputs...")
    save_outputs(grid_gdf)
    evidence_counts = save_statistics(grid_gdf, spearman_r, spearman_p)

    # ---- 7. 融合统计 ----
    print("\n[7] Computing fusion statistics...")
    df = grid_gdf
    stats = {
        'n_total': len(df),
        'n_both': int(both),
        'n_s2_only': int(s2_only),
        'n_viirs_only': int(viirs_only),
        'n_no_data': int(no_data),
    }
    for col, label in [('p_s2_mean', 'p_s2'), ('p_viirs_mean', 'p_viirs'), ('p_fused', 'p_fused')]:
        vals = df[col].dropna()
        if len(vals) > 0:
            stats[f'{label}_mean'] = float(vals.mean())
            stats[f'{label}_median'] = float(vals.median())
            stats[f'{label}_std'] = float(vals.std())
    for col, label in [('disagreement', 'disagreement'), ('uncertainty', 'uncertainty')]:
        vals = df[col].dropna()
        if len(vals) > 0:
            stats[f'{label}_mean'] = float(vals.mean())
            stats[f'{label}_median'] = float(vals.median())

    with open(FUSION_STATS_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        for k, v in stats.items():
            w.writerow([k, v])
        w.writerow(['spearman_r', spearman_r])
        w.writerow(['spearman_p', spearman_p])
    print(f"  [CSV] Saved: {FUSION_STATS_CSV}")

    # ---- 8. 报告 ----
    print("\n[8] Generating report...")
    gen_report(grid_gdf, stats, evidence_counts, spearman_r, spearman_p)

    print("\n" + "=" * 70)
    print("双源融合完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 返回关键统计用于汇报
    return stats, spearman_r, evidence_counts


def gen_report(grid_gdf, stats, evidence_counts, spearman_r, spearman_p):
    """生成中文报告"""
    df = grid_gdf
    both_df = df[df['fusion_source'] == 'both']

    lines = []
    lines.append("# Sentinel-2 / VIIRS 双源融合报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 4 — 统一 500m 网格与双源融合\n")
    lines.append("---\n")

    lines.append("## 1. 方法概述\n")
    lines.append(f"- 网格: 500m × 500m, EPSG:32637 (UTM zone 37N)\n")
    lines.append(f"- S2 聚合: 将 20m Sentinel-2 损害分数聚合到每个网格 (mean/median/std)\n")
    lines.append(f"- VIIRS 聚合: 将 ~500m VIIRS p_viirs 重投影到 EPSG:32637 后聚合\n")
    lines.append(f"- 融合方法: Logit-平均 (等权重)\n")
    lines.append(f"  - `logit_p_fused = 0.5 × logit(p_s2) + 0.5 × logit(p_viirs)`\n")
    lines.append(f"  - `p_fused = sigmoid(logit_p_fused)`\n")
    lines.append(f"- 不确定性: `0.7 × disagreement + 0.3 × quality_uncertainty`\n")
    lines.append(f"  - `disagreement = |p_s2 - p_viirs|` (传感器证据冲突)\n")
    lines.append(f"  - `quality_uncertainty = 0.5(1-s2_ratio) + 0.5(1-viirs_ratio)` (覆盖不足)\n")

    lines.append("\n## 2. 输入数据\n")
    lines.append(f"- S2 损害分数: `{S2_SCORE_TIF.relative_to(PROJECT_ROOT)}` (EPSG:32637, 20m)\n")
    lines.append(f"- p_viirs: `{P_VIIRS_TIF.relative_to(PROJECT_ROOT)}` (EPSG:4326, ~500m)\n")
    lines.append(f"- VIIRS loss_rate: `{VIIRS_LR_TIF.relative_to(PROJECT_ROOT)}`\n")
    lines.append(f"- VIIRS valid_mask: `{VIIRS_VALID_TIF.relative_to(PROJECT_ROOT)}`\n")
    lines.append(f"- AOI: `{AOI_PATH.relative_to(PROJECT_ROOT)}`\n")

    lines.append("\n## 3. 网格统计\n")
    lines.append(f"- 网格总数: {stats['n_total']}\n")
    lines.append(f"- 同时有 S2 和 VIIRS: {stats['n_both']} ({100*stats['n_both']/stats['n_total']:.1f}%)\n")
    lines.append(f"- 仅 S2: {stats['n_s2_only']} ({100*stats['n_s2_only']/stats['n_total']:.1f}%)\n")
    lines.append(f"- 仅 VIIRS: {stats['n_viirs_only']} ({100*stats['n_viirs_only']/stats['n_total']:.1f}%)\n")
    lines.append(f"- 无数据: {stats['n_no_data']}\n")

    lines.append("\n## 4. 损害概率统计\n")
    lines.append("| 指标 | p_s2 (Grid Mean) | p_viirs (Grid Mean) | p_fused |\n")
    lines.append("|------|------------------|---------------------|---------|\n")
    lines.append(f"| 均值 | {stats.get('p_s2_mean', np.nan):.4f} | {stats.get('p_viirs_mean', np.nan):.4f} | {stats.get('p_fused_mean', np.nan):.4f} |\n")
    lines.append(f"| 中位数 | {stats.get('p_s2_median', np.nan):.4f} | {stats.get('p_viirs_median', np.nan):.4f} | {stats.get('p_fused_median', np.nan):.4f} |\n")
    lines.append(f"| 标准差 | {stats.get('p_s2_std', np.nan):.4f} | {stats.get('p_viirs_std', np.nan):.4f} | {stats.get('p_fused_std', np.nan):.4f} |\n")

    lines.append("\n## 5. 传感器一致性\n")
    lines.append(f"- **Spearman ρ = {spearman_r:.4f}** (p = {spearman_p:.4f})\n")

    if len(both_df) > 0:
        d_mean = both_df['disagreement'].mean()
        d_median = both_df['disagreement'].median()
        lines.append(f"- Disagreement 均值: {d_mean:.4f}, 中位数: {d_median:.4f}\n")

        s2_h = (both_df['p_s2_mean'] >= 0.5)
        v_h = (both_df['p_viirs_mean'] >= 0.5)
        n_both = len(both_df)
        lines.append(f"- Both high: {(s2_h & v_h).sum()} / {n_both} ({100*(s2_h & v_h).sum()/n_both:.1f}%)\n")
        lines.append(f"- S2 high, VIIRS low: {(s2_h & ~v_h).sum()} / {n_both} ({100*(s2_h & ~v_h).sum()/n_both:.1f}%)\n")
        lines.append(f"- S2 low, VIIRS high: {(~s2_h & v_h).sum()} / {n_both} ({100*(~s2_h & v_h).sum()/n_both:.1f}%)\n")
        lines.append(f"- Both low: {(~s2_h & ~v_h).sum()} / {n_both} ({100*(~s2_h & ~v_h).sum()/n_both:.1f}%)\n")

    lines.append("\n## 6. 不确定性统计\n")
    u_mean, u_med = stats.get('uncertainty_mean', np.nan), stats.get('uncertainty_median', np.nan)
    d_m, d_med = stats.get('disagreement_mean', np.nan), stats.get('disagreement_median', np.nan)
    lines.append(f"- Disagreement 均值: {d_m:.4f}, 中位数: {d_med:.4f}\n")
    lines.append(f"- Uncertainty 均值: {u_mean:.4f}, 中位数: {u_med:.4f}\n")

    lines.append("\n## 7. 证据类型分布\n")
    lines.append("| 类型 | 数量 | 比例 |\n")
    lines.append("|------|------|------|\n")
    for etype, cnt in evidence_counts.items():
        lines.append(f"| {etype} | {cnt} | {cnt/len(df):.4f} ({100*cnt/len(df):.1f}%) |\n")

    # 不确定区域分析
    if len(both_df) > 0:
        hi_unc = both_df.nlargest(20, 'uncertainty')
        hi_evidence = hi_unc['evidence_type'].value_counts()
        lines.append("\n### 最高不确定性网格的 evidence_type\n")
        lines.append("| 类型 | 数量 |\n")
        lines.append("|------|------|\n")
        for etype, cnt in hi_evidence.items():
            lines.append(f"| {etype} | {cnt} |\n")

    lines.append("\n## 8. 输出文件清单\n")
    outputs = [
        (GRID_GEOJSON, "500m 网格 GeoJSON"),
        (GRID_FEAT_CSV, "网格特征 CSV"),
        (FUSION_CSV, "融合结果 CSV"),
        (FUSION_GEOJSON, "融合结果 GeoJSON"),
        (MAP_P_S2, "p_s2 网格图"),
        (MAP_P_VIIRS, "p_viirs 网格图"),
        (MAP_FUSED, "p_fused 融合图"),
        (MAP_UNCERT, "不确定性图"),
        (MAP_EVIDENCE, "证据类型图"),
        (SCATTER_PNG, "p_s2 vs p_viirs 散点图"),
        (FUSION_STATS_CSV, "融合统计 CSV"),
        (EVIDENCE_CSV, "证据类型计数 CSV"),
        (AGREEMENT_CSV, "传感器一致性统计 CSV"),
        (REPORT_PATH, "本报告"),
    ]
    for p, desc in outputs:
        lines.append(f"| `{p.relative_to(PROJECT_ROOT)}` | {desc} |\n")

    lines.append("\n## 9. 阶段结论\n")
    lines.append(f"> 网格总数: {stats['n_total']}, 双传感器覆盖: {stats['n_both']}, Spearman ρ = {spearman_r:.4f}\n")
    lines.append("\n**p_s2 与 p_viirs 的解释**:\n")
    if abs(spearman_r) < 0.1:
        lines.append("- 两个传感器几乎无相关性，说明光学(S2)和夜光(VIIRS)捕捉了完全不同的损害信号维度\n")
    elif spearman_r < -0.2:
        lines.append(f"- **中等负相关 (ρ = {spearman_r:.4f})**: Sentinel-2 光谱损害分数与 VIIRS 夜光损失概率呈反向关系\n")
        lines.append("- 这意味着 S2 检测到强光谱变化的区域，VIIRS 夜光未必下降最多；反之亦然\n")
        lines.append("- 物理原因: S2 捕获 20m 尺度的植被/建筑表面变化，VIIRS 捕获 500m 尺度的功能性停电/人口迁出\n")
        lines.append("- 两个传感器测量了战争损害的不同维度，负相关说明它们是**互补而非冗余**的信息源\n")
    elif spearman_r < 0.3:
        lines.append("- 两个传感器呈弱正相关，说明它们部分一致但各有所侧重\n")
    else:
        lines.append("- 两个传感器呈中等以上正相关，为融合提供了较好的一致性基础\n")

    lines.append("\n**是否进入下一阶段**:\n")
    lines.append("> ✅ 可以进入 Sentinel-1 融合或 D-S 证据理论阶段\n")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {REPORT_PATH}")


if __name__ == "__main__":
    stats, spearman_r, evidence_counts = main()

    print("\n" + "=" * 70)
    print("关键结果汇总:")
    print(f"  网格总数: {stats['n_total']}")
    print(f"  双传感器覆盖: {stats['n_both']}")
    print(f"  Spearman ρ (p_s2 vs p_viirs): {spearman_r:.4f}")
    print(f"  p_fused 均值: {stats.get('p_fused_mean', np.nan):.4f}")
    print(f"  不确定性均值: {stats.get('uncertainty_mean', np.nan):.4f}")
    print("=" * 70)
