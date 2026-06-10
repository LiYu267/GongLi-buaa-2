#!/usr/bin/env python3
"""
阶段 2 — Sentinel-2 单源损害分数计算

方法:
  - 从 delta 指数 (dNDVI, dNBR, dNDBI, dNDWI) 出发
  - Z-score 标准化: z_i = (x_i - μ) / σ  (消除不同指数的量纲差异)
  - 损害方向投影:
      • dNDVI < 0 (植被覆盖减少)  → 损害方向为正
      • dNBR  < 0 (SWIR 反射上升, 瓦砾暴露) → 损害方向为正
      • dNDBI > 0 (建成区/裸土增加) → 损害方向为正
      • dNDWI 不用于方向判断 (建筑物损害信号弱)
    损害原始分 = max(0, -z_dNDVI) + max(0, -z_dNBR) + max(0, z_dNDBI)
  - 归一化到 [0, 1]
  - 基于百分位数分级: 极低 / 低 / 中等 / 高 / 极高

输出:
  - data/processed/s2_main_damage_score.tif         损害分数 (float32, [0,1])
  - data/processed/s2_main_damage_category.tif      损害等级 (uint8, 1-5)
  - outputs/figures/s2_main_damage_score.png        损害图
  - outputs/tables/s2_main_damage_statistics.csv    统计表
  - outputs/sentinel2_damage_score_report.md        中文报告
"""

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELTA_TIF = PROJECT_ROOT / "data" / "processed" / "s2_main_delta_20210523_20220508_indices.tif"
AOI_PATH = PROJECT_ROOT / "config" / "aoi_mariupol.geojson"
OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB = PROJECT_ROOT / "outputs" / "tables"
OUT_MD = PROJECT_ROOT / "outputs"

for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)

SCORE_TIF = OUT_DATA / "s2_main_damage_score.tif"
CATEGORY_TIF = OUT_DATA / "s2_main_damage_category.tif"
SCORE_PNG = OUT_FIG / "s2_main_damage_score.png"
HIST_PNG = OUT_FIG / "s2_main_damage_histogram.png"
CSV_PATH = OUT_TAB / "s2_main_damage_statistics.csv"
REPORT_PATH = OUT_MD / "sentinel2_damage_score_report.md"

# 损害等级定义
CATEGORY_LABELS = {
    0: "No Data",
    1: "Very Low",
    2: "Low",
    3: "Medium",
    4: "High",
    5: "Very High",
}
CATEGORY_COLORS = {
    0: '#d9d9d9',
    1: '#ffffcc',
    2: '#a1dab4',
    3: '#41b6c4',
    4: '#2c7fb8',
    5: '#253494',
}
PERCENTILE_THRESHOLDS = [25, 50, 75, 90]  # 对应 Very Low / Low / Medium / High / Very High


# ============================================================
# 核心计算
# ============================================================

def load_delta_indices():
    """加载 delta 指数和元数据"""
    with rasterio.open(DELTA_TIF) as src:
        dNDVI = src.read(1)
        dNBR = src.read(2)
        dNDBI = src.read(3)
        dNDWI = src.read(4)
        transform = src.transform
        crs = src.crs
        h, w = src.shape
    return dNDVI, dNBR, dNDBI, dNDWI, transform, crs, (h, w)


def compute_valid_mask(dNDVI, dNBR, dNDBI, dNDWI):
    """所有四个指数均有限 → 有效像元"""
    return (
        np.isfinite(dNDVI) &
        np.isfinite(dNBR) &
        np.isfinite(dNDBI) &
        np.isfinite(dNDWI)
    )


def compute_damage_score(dNDVI, dNBR, dNDBI, dNDWI, valid_mask):
    """
    计算单源损害分数。

    步骤:
    1. Z-score 标准化各 delta 指数
    2. 损害方向投影
    3. 归一化到 [0, 1]
    """
    # --- Z-score ---
    # 仅用有效像元计算 μ, σ
    v_ndvi = dNDVI[valid_mask]
    v_nbr  = dNBR[valid_mask]
    v_ndbi = dNDBI[valid_mask]

    mu_ndvi, sig_ndvi = np.mean(v_ndvi), np.std(v_ndvi)
    mu_nbr,  sig_nbr  = np.mean(v_nbr),  np.std(v_nbr)
    mu_ndbi, sig_ndbi = np.mean(v_ndbi), np.std(v_ndbi)

    z_ndvi = np.full_like(dNDVI, np.nan)
    z_nbr  = np.full_like(dNBR,  np.nan)
    z_ndbi = np.full_like(dNDBI, np.nan)

    z_ndvi[valid_mask] = (dNDVI[valid_mask] - mu_ndvi) / sig_ndvi
    z_nbr[valid_mask]  = (dNBR[valid_mask]  - mu_nbr)  / sig_nbr
    z_ndbi[valid_mask] = (dNDBI[valid_mask] - mu_ndbi) / sig_ndbi

    # --- 损害方向投影 ---
    # 只计正贡献 (即变化方向符合损害特征)
    contrib_ndvi = np.maximum(0.0, -z_ndvi)  # NDVI↓ → 正分
    contrib_nbr  = np.maximum(0.0, -z_nbr)   # NBR↓  → 正分
    contrib_ndbi = np.maximum(0.0,  z_ndbi)  # NDBI↑ → 正分

    damage_raw = contrib_ndvi + contrib_nbr + contrib_ndbi

    # --- 归一化到 [0, 1] ---
    dmin = np.nanmin(damage_raw[valid_mask])
    dmax = np.nanmax(damage_raw[valid_mask])
    if dmax - dmin < 1e-10:
        damage_score = np.zeros_like(damage_raw)
    else:
        damage_score = (damage_raw - dmin) / (dmax - dmin)

    damage_score[~valid_mask] = np.nan

    # 诊断信息
    z_stats = {
        'mu_ndvi': mu_ndvi, 'sig_ndvi': sig_ndvi,
        'mu_nbr': mu_nbr, 'sig_nbr': sig_nbr,
        'mu_ndbi': mu_ndbi, 'sig_ndbi': sig_ndbi,
        'raw_min': dmin, 'raw_max': dmax,
    }

    return damage_score, z_stats


def categorize_damage(damage_score, valid_mask):
    """基于百分位数分级, 处理 score=0 的边缘情况"""
    scores_valid = damage_score[valid_mask]
    thresholds = np.percentile(scores_valid, PERCENTILE_THRESHOLDS)

    # 检测 edge case: 若 P25 == 0 且大量像素 score==0,
    # 则将 score==0 的像素归为 Very Low, 剩余像素按等量分到 Low/Medium/High/Very High
    n_zero = int(np.sum(scores_valid == 0))
    n_total = len(scores_valid)

    categories = np.full_like(damage_score, 0, dtype=np.uint8)

    if thresholds[0] <= 1e-8 and n_zero > n_total * 0.05:
        # 存在大量零分像素, 单独处理
        score_zero_mask = (damage_score <= 1e-8) & valid_mask
        score_pos_mask = (damage_score > 1e-8) & valid_mask
        categories[score_zero_mask] = 1  # Very Low
        categories[~valid_mask] = 0

        # 对正分像素重新分级: 均分四等份
        scores_pos = damage_score[score_pos_mask]
        if len(scores_pos) >= 4:
            p33, p66 = np.percentile(scores_pos, [33.333, 66.667])
            categories[(damage_score > 1e-8) & (damage_score < p33)] = 2
            categories[(damage_score >= p33) & (damage_score < p66)] = 3
            categories[(damage_score >= p66) & (damage_score < thresholds[3])] = 4
            categories[damage_score >= thresholds[3]] = 5
        else:
            # 正分像素太少, 用原始阈值
            categories[(damage_score >= thresholds[0]) & (damage_score < thresholds[1])] = 2
            categories[(damage_score >= thresholds[1]) & (damage_score < thresholds[2])] = 3
            categories[(damage_score >= thresholds[2]) & (damage_score < thresholds[3])] = 4
            categories[damage_score >= thresholds[3]] = 5
    else:
        # 标准分级
        categories[(damage_score >= 0)        & (damage_score < thresholds[0])] = 1
        categories[(damage_score >= thresholds[0]) & (damage_score < thresholds[1])] = 2
        categories[(damage_score >= thresholds[1]) & (damage_score < thresholds[2])] = 3
        categories[(damage_score >= thresholds[2]) & (damage_score < thresholds[3])] = 4
        categories[damage_score >= thresholds[3]] = 5
        categories[~valid_mask] = 0

    return categories, thresholds


def compute_stats(damage_score, categories, valid_mask):
    """计算各等级统计"""
    counts = {}
    for cat_id in range(6):
        if cat_id == 0:
            counts['No Data'] = (~valid_mask).sum()
        else:
            counts[CATEGORY_LABELS[cat_id]] = (categories[valid_mask] == cat_id).sum()

    total = valid_mask.sum()
    ratios = {k: v / total for k, v in counts.items() if k != 'No Data'}

    # 分数统计
    scores = damage_score[valid_mask]
    score_stats = {
        'mean': float(np.mean(scores)),
        'median': float(np.median(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'p25': float(np.percentile(scores, 25)),
        'p50': float(np.percentile(scores, 50)),
        'p75': float(np.percentile(scores, 75)),
        'p90': float(np.percentile(scores, 90)),
    }

    return counts, ratios, score_stats


# ============================================================
# 输出
# ============================================================

def save_damage_score_geotiff(damage_score, transform, crs, output_path):
    """保存损害分数 GeoTIFF"""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=damage_score.shape[0],
        width=damage_score.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress='lzw',
    ) as dst:
        dst.write(damage_score.astype(np.float32), 1)
        dst.set_band_description(1, 'Damage Score [0,1]')


def save_category_geotiff(categories, transform, crs, output_path):
    """保存损害等级 GeoTIFF"""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=categories.shape[0],
        width=categories.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=0,
        compress='lzw',
    ) as dst:
        dst.write(categories, 1)
        dst.set_band_description(1, 'Damage Category (1-5)')


def plot_damage_map(damage_score, categories, output_path, aoi_gdf):
    """绘制损害图和损害等级图"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- 左: 损害分数 (连续) ---
    ax1 = axes[0]
    score_valid = np.where(np.isfinite(damage_score), damage_score, np.nan)
    im1 = ax1.imshow(score_valid, cmap='YlOrRd', vmin=0, vmax=1,
                     interpolation='nearest')
    ax1.set_title('Damage Score (Continuous)', fontsize=13, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Score [0, 1]')

    # --- 右: 损害等级 (离散) ---
    ax2 = axes[1]
    cat_colors = [CATEGORY_COLORS[i] for i in range(6)]
    cmap_cat = mcolors.ListedColormap(cat_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm_cat = mcolors.BoundaryNorm(bounds, cmap_cat.N)
    im2 = ax2.imshow(categories, cmap=cmap_cat, norm=norm_cat, interpolation='nearest')
    ax2.set_title('Damage Category', fontsize=13, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # 图例
    legend_patches = []
    for cat_id in [1, 2, 3, 4, 5]:
        legend_patches.append(Patch(
            facecolor=CATEGORY_COLORS[cat_id],
            label=CATEGORY_LABELS[cat_id]
        ))
    ax2.legend(handles=legend_patches, loc='lower right',
               fontsize=9, framealpha=0.9)

    fig.suptitle(
        'Sentinel-2 Single-Source Damage Score — Mariupol\n'
        '2021-05-23 (Pre-war) → 2022-05-08 (Post-war)',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {output_path}")


def plot_histogram(damage_score, categories, thresholds, valid_mask, output_path):
    """绘制损害分数直方图 (按等级着色)"""
    scores = damage_score[valid_mask]
    cats_valid = categories[valid_mask]

    fig, ax = plt.subplots(figsize=(12, 6))

    for cat_id in [1, 2, 3, 4, 5]:
        mask = cats_valid == cat_id
        ax.hist(scores[mask], bins=60, color=CATEGORY_COLORS[cat_id],
                alpha=0.85, label=CATEGORY_LABELS[cat_id], edgecolor='white',
                linewidth=0.3)

    # 标注阈值线
    for i, thr in enumerate(thresholds):
        ax.axvline(x=thr, color='#333333', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(thr, ax.get_ylim()[1] * 0.95, f'P{PERCENTILE_THRESHOLDS[i]}={thr:.3f}',
                rotation=90, va='top', fontsize=8, color='#333333')

    ax.set_xlabel('Damage Score', fontsize=12)
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title(
        'Distribution of Sentinel-2 Damage Score — Mariupol\n'
        f'Valid pixels: {valid_mask.sum():,}',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {output_path}")


def save_statistics_csv(score_stats, counts, ratios, thresholds, z_stats):
    """保存统计表"""
    rows = [
        ['metric', 'value'],
        ['valid_pixels', counts.get(list(counts.keys())[-1], 0)],
        ['score_mean', score_stats['mean']],
        ['score_median', score_stats['median']],
        ['score_std', score_stats['std']],
        ['score_min', score_stats['min']],
        ['score_max', score_stats['max']],
        ['score_p25', score_stats['p25']],
        ['score_p50', score_stats['p50']],
        ['score_p75', score_stats['p75']],
        ['score_p90', score_stats['p90']],
        ['threshold_VeryLow_to_Low', thresholds[0]],
        ['threshold_Low_to_Medium', thresholds[1]],
        ['threshold_Medium_to_High', thresholds[2]],
        ['threshold_High_to_VeryHigh', thresholds[3]],
        ['', ''],
        ['category', 'count', 'ratio'],
    ]
    for cat_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
        rows.append([cat_name, counts.get(cat_name, 0),
                     round(ratios.get(cat_name, 0), 6)])
    rows.append(['No Data', counts.get('No Data', 0), ''])

    # Z-score params
    rows.append(['', '', ''])
    rows.append(['zscore_param', 'mu', 'sigma'])
    rows.append(['dNDVI', z_stats['mu_ndvi'], z_stats['sig_ndvi']])
    rows.append(['dNBR', z_stats['mu_nbr'], z_stats['sig_nbr']])
    rows.append(['dNDBI', z_stats['mu_ndbi'], z_stats['sig_ndbi']])

    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  [CSV] Saved: {CSV_PATH}")


def generate_report(z_stats, score_stats, thresholds, counts, ratios):
    """生成中文报告"""
    lines = []
    lines.append("# Sentinel-2 单源损害分数计算报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 2 — Sentinel-2 单源损害分数\n")
    lines.append("---\n")

    lines.append("## 1. 方法论\n")
    lines.append("### 1.1 输入数据\n")
    lines.append(f"- 来源: `data/processed/s2_main_delta_20210523_20220508_indices.tif`\n")
    lines.append("- 包含 4 个 delta 指数: dNDVI, dNBR, dNDBI, dNDWI\n")
    lines.append("- 分辨率: 20m, CRS: EPSG:32637\n")

    lines.append("### 1.2 损害分数计算步骤\n")
    lines.append("**步骤 1 — Z-score 标准化**\n")
    lines.append("对每个 delta 指数进行 Z-score 标准化，消除量纲差异：\n")
    lines.append("```\n")
    lines.append("z_i = (x_i - μ) / σ\n")
    lines.append("```\n")
    lines.append(f"其中 μ, σ 为 AOI 范围内有效像元的均值和标准差。\n")
    lines.append("\n**步骤 2 — 损害方向投影**\n")
    lines.append("战争建筑物损害典型表现为以下光谱变化：\n")
    lines.append("- **NDVI 下降** (植被覆盖减少/移除)\n")
    lines.append("- **NBR 下降** (建筑物瓦砾暴露 → SWIR 反射率上升)\n")
    lines.append("- **NDBI 上升** (建成区/裸土的暴露增加)\n")
    lines.append("- NDWI 不用于方向判断 (建筑物损害信号弱)\n")
    lines.append("\n损害原始分计算：\n")
    lines.append("```\n")
    lines.append("damage_raw = max(0, -z_dNDVI) + max(0, -z_dNBR) + max(0, z_dNDBI)\n")
    lines.append("```\n")
    lines.append("仅计入损害方向一致的正贡献。\n")

    lines.append("**步骤 3 — 归一化**\n")
    lines.append("```\n")
    lines.append("damage_score = (damage_raw - min) / (max - min)  ∈ [0, 1]\n")
    lines.append("```\n")

    lines.append("**步骤 4 — 损害等级划分**\n")
    lines.append("基于有效像元损害分数的百分位数分级：\n")
    lines.append("| 等级 | 百分位范围 |\n")
    lines.append("|------|------------|\n")
    for i, (cat_id, p_range) in enumerate([
        (1, '[0, P25)'),
        (2, '[P25, P50)'),
        (3, '[P50, P75)'),
        (4, '[P75, P90)'),
        (5, '[P90, 1]'),
    ]):
        lines.append(f"| {CATEGORY_LABELS[cat_id]} | {p_range} |\n")

    lines.append("\n## 2. Z-score 标准化参数\n")
    lines.append("| Delta 指数 | μ (均值) | σ (标准差) |\n")
    lines.append("|------------|----------|------------|\n")
    lines.append(f"| dNDVI | {z_stats['mu_ndvi']:.6f} | {z_stats['sig_ndvi']:.6f} |\n")
    lines.append(f"| dNBR  | {z_stats['mu_nbr']:.6f} | {z_stats['sig_nbr']:.6f} |\n")
    lines.append(f"| dNDBI | {z_stats['mu_ndbi']:.6f} | {z_stats['sig_ndbi']:.6f} |\n")
    lines.append(f"\n损害原始分范围: [{z_stats['raw_min']:.4f}, {z_stats['raw_max']:.4f}]\n")

    lines.append("\n## 3. 损害分数统计\n")
    lines.append("| 统计量 | 值 |\n")
    lines.append("|--------|-----|\n")
    for key, val in [
        ('均值', score_stats['mean']),
        ('中位数', score_stats['median']),
        ('标准差', score_stats['std']),
        ('最小值', score_stats['min']),
        ('最大值', score_stats['max']),
        ('P25', score_stats['p25']),
        ('P50', score_stats['p50']),
        ('P75', score_stats['p75']),
        ('P90', score_stats['p90']),
    ]:
        lines.append(f"| {key} | {val:.6f} |\n")

    lines.append("\n## 4. 损害等级分布\n")
    total_valid = sum(c for k, c in counts.items() if k != 'No Data')
    lines.append("| 等级 | 像元数 | 占比 |\n")
    lines.append("|------|--------|------|\n")
    for cat_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
        c = counts.get(cat_name, 0)
        r = ratios.get(cat_name, 0) if cat_name != 'No Data' else 0
        lines.append(f"| {cat_name} | {c} | {r:.4f} ({100*r:.2f}%) |\n")
    lines.append(f"| No Data | {counts.get('No Data', 0)} | — |\n")

    lines.append("\n## 5. 分级阈值\n")
    lines.append("| 阈值 | 分数值 |\n")
    lines.append("|------|--------|\n")
    lines.append(f"| Very Low → Low | {thresholds[0]:.6f} |\n")
    lines.append(f"| Low → Medium   | {thresholds[1]:.6f} |\n")
    lines.append(f"| Medium → High  | {thresholds[2]:.6f} |\n")
    lines.append(f"| High → Very High | {thresholds[3]:.6f} |\n")

    lines.append("\n## 6. 输出文件\n")
    lines.append("| 文件 | 说明 |\n")
    lines.append("|------|------|\n")
    lines.append(f"| `{SCORE_TIF.relative_to(PROJECT_ROOT)}` | 损害分数 GeoTIFF, float32, [0,1] |\n")
    lines.append(f"| `{CATEGORY_TIF.relative_to(PROJECT_ROOT)}` | 损害等级 GeoTIFF, uint8, 1-5 |\n")
    lines.append(f"| `{SCORE_PNG.relative_to(PROJECT_ROOT)}` | 损害分数与等级图 |\n")
    lines.append(f"| `{HIST_PNG.relative_to(PROJECT_ROOT)}` | 分数分布直方图 |\n")
    lines.append(f"| `{CSV_PATH.relative_to(PROJECT_ROOT)}` | 统计表 |\n")
    lines.append(f"| `{REPORT_PATH.relative_to(PROJECT_ROOT)}` | 本报告 |\n")

    lines.append("\n## 7. 阶段结论\n")
    lines.append("> ✅ **Sentinel-2 单源损害分数计算完成**\n")
    lines.append("\n**要点**:\n")
    lines.append(f"1. 损害分数基于 3 个 delta 指数的 Z-score 方向投影合成\n")
    lines.append(f"2. 有效像元 {total_valid:,} 个，覆盖 Mariupol AOI 20m 分辨率的 {100*total_valid/(744*1152):.1f}%\n")
    high_pct = 100 * (ratios.get('High', 0) + ratios.get('Very High', 0))
    lines.append(f"3. High + Very High 等级合计占比 {high_pct:.2f}%\n")
    lines.append("4. 分数可进一步用于与 VIIRS / InSAR / GRD 的贝叶斯融合\n")

    lines.append("\n## 8. 方法局限性说明\n")
    lines.append("1. **相对归一化**: 分数在 AOI 内部归一化，适合 AOI 内部比较，但不同 AOI 之间不可直接比较\n")
    lines.append("2. **非监督方法**: 未使用 UNOSAT 等标签进行校准，高分仅表示相对异常变化\n")
    lines.append("3. **百分位阈值**: 分级阈值依赖于 AOI 内像元的分布，若受损面积占比大，阈值会被抬高\n")
    lines.append("4. **季节性混淆**: 战前 (5月) 与战后 (5月) 同月，但年际差异仍可能存在气候/农业影响\n")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {REPORT_PATH}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("Sentinel-2 单源损害分数计算")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- 1. 加载数据 ----
    print("\n[1] Loading delta indices...")
    dNDVI, dNBR, dNDBI, dNDWI, transform, crs, shape = load_delta_indices()
    print(f"    Shape: {shape}, CRS: {crs}")

    # ---- 2. 有效像元掩膜 ----
    print("\n[2] Computing valid mask...")
    valid_mask = compute_valid_mask(dNDVI, dNBR, dNDBI, dNDWI)
    print(f"    Valid: {valid_mask.sum():,} / {valid_mask.size:,} "
          f"({100*valid_mask.sum()/valid_mask.size:.2f}%)")

    # ---- 3. 计算损害分数 ----
    print("\n[3] Computing damage score...")
    damage_score, z_stats = compute_damage_score(dNDVI, dNBR, dNDBI, dNDWI, valid_mask)
    print(f"    dNDVI μ={z_stats['mu_ndvi']:.4f} σ={z_stats['sig_ndvi']:.4f}")
    print(f"    dNBR  μ={z_stats['mu_nbr']:.4f} σ={z_stats['sig_nbr']:.4f}")
    print(f"    dNDBI μ={z_stats['mu_ndbi']:.4f} σ={z_stats['sig_ndbi']:.4f}")
    print(f"    Raw damage range: [{z_stats['raw_min']:.4f}, {z_stats['raw_max']:.4f}]")

    # ---- 4. 损害分级 ----
    print("\n[4] Categorizing damage...")
    categories, thresholds = categorize_damage(damage_score, valid_mask)
    print(f"    Thresholds: {[f'{t:.4f}' for t in thresholds]}")

    # ---- 5. 统计 ----
    print("\n[5] Computing statistics...")
    counts, ratios, score_stats = compute_stats(damage_score, categories, valid_mask)
    for cat_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
        print(f"    {cat_name}: {counts.get(cat_name, 0):,} ({100*ratios.get(cat_name, 0):.2f}%)")

    # ---- 6. 保存 ----
    print("\n[6] Saving outputs...")

    save_damage_score_geotiff(damage_score, transform, crs, SCORE_TIF)
    print(f"    Saved: {SCORE_TIF}")

    save_category_geotiff(categories, transform, crs, CATEGORY_TIF)
    print(f"    Saved: {CATEGORY_TIF}")

    # Load AOI for visualization
    with open(AOI_PATH, 'r', encoding='utf-8') as f:
        aoi_geojson = json.load(f)
    aoi_gdf = gpd.GeoDataFrame.from_features(aoi_geojson["features"], crs="EPSG:4326")

    plot_damage_map(damage_score, categories, SCORE_PNG, aoi_gdf)
    plot_histogram(damage_score, categories, thresholds, valid_mask, HIST_PNG)
    save_statistics_csv(score_stats, counts, ratios, thresholds, z_stats)
    generate_report(z_stats, score_stats, thresholds, counts, ratios)

    print("\n" + "=" * 70)
    print("单源损害分数计算完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
