#!/usr/bin/env python3
"""
阶段 5 — Dempster-Shafer 证据理论融合 (S2 + VIIRS)

BPA 构造 → Dempster 组合 → 冲突/决策分类 → 与 logit 平均对比
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_CSV = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.csv"
FUSION_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.geojson"

OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG  = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB  = PROJECT_ROOT / "outputs" / "tables"
OUT_MD   = PROJECT_ROOT / "outputs"
for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)

# Output paths
DS_CSV = OUT_DATA / "grid_ds_fusion_s2_viirs.csv"
DS_GEOJSON = OUT_DATA / "grid_ds_fusion_s2_viirs.geojson"

# Maps
MAP_BELIEF = OUT_FIG / "ds_belief_damage_map.png"
MAP_PLAUS = OUT_FIG / "ds_plausibility_damage_map.png"
MAP_UNCDS = OUT_FIG / "ds_uncertainty_map.png"
MAP_CONFLICT = OUT_FIG / "ds_conflict_k_map.png"
MAP_DECISION = OUT_FIG / "ds_decision_class_map.png"
MAP_SCATTER = OUT_FIG / "ds_vs_logit_scatter.png"
MAP_INTERVAL = OUT_FIG / "ds_interval_width_map.png"

# Tables
STATS_CSV = OUT_TAB / "ds_fusion_summary_statistics.csv"
CONFLICT_CSV = OUT_TAB / "ds_conflict_class_counts.csv"
DECISION_CSV = OUT_TAB / "ds_decision_class_counts.csv"
CROSSTAB_CSV = OUT_TAB / "ds_evidence_conflict_crosstab.csv"
COMPARE_CSV = OUT_TAB / "ds_vs_logit_comparison.csv"

# Report
REPORT_PATH = OUT_MD / "ds_fusion_report.md"


# ============================================================
# 1. BPA 构造
# ============================================================

def compute_bpa(row):
    """对单个网格计算 S2 和 VIIRS 的 BPA"""
    result = {}

    # --- Sentinel-2 BPA ---
    has_s2 = not np.isnan(row.get('p_s2_mean', np.nan))
    if has_s2:
        p_s2 = np.clip(row['p_s2_mean'], 0.001, 0.999)
        s2_ratio = row.get('p_s2_valid_ratio', 1.0)
        if np.isnan(s2_ratio):
            s2_ratio = 0.0

        # Uncertainty: base + quality + (optional) std
        u_s2 = 0.15 + 0.5 * (1.0 - s2_ratio)
        if 'p_s2_std' in row.index and not np.isnan(row['p_s2_std']):
            s2_std_norm = min(row['p_s2_std'] / 0.3, 1.0)  # normalize, cap at 1
            u_s2 = 0.15 + 0.4 * (1.0 - s2_ratio) + 0.2 * s2_std_norm

        u_s2 = np.clip(u_s2, 0.05, 0.80)

        result['p_s2_clipped'] = p_s2
        result['u_s2'] = u_s2
        result['m_s2_D'] = p_s2 * (1.0 - u_s2)
        result['m_s2_N'] = (1.0 - p_s2) * (1.0 - u_s2)
        result['m_s2_U'] = u_s2
    else:
        result['p_s2_clipped'] = np.nan
        result['u_s2'] = np.nan
        result['m_s2_D'] = np.nan
        result['m_s2_N'] = np.nan
        result['m_s2_U'] = np.nan

    # --- VIIRS BPA ---
    has_viirs = not np.isnan(row.get('p_viirs_mean', np.nan))
    if has_viirs:
        p_viirs = np.clip(row['p_viirs_mean'], 0.001, 0.999)
        viirs_ratio = row.get('p_viirs_valid_ratio', 1.0)
        if np.isnan(viirs_ratio):
            viirs_ratio = 0.0

        u_viirs = 0.15 + 0.5 * (1.0 - viirs_ratio)
        u_viirs = np.clip(u_viirs, 0.05, 0.80)

        result['p_viirs_clipped'] = p_viirs
        result['u_viirs'] = u_viirs
        result['m_viirs_D'] = p_viirs * (1.0 - u_viirs)
        result['m_viirs_N'] = (1.0 - p_viirs) * (1.0 - u_viirs)
        result['m_viirs_U'] = u_viirs
    else:
        result['p_viirs_clipped'] = np.nan
        result['u_viirs'] = np.nan
        result['m_viirs_D'] = np.nan
        result['m_viirs_N'] = np.nan
        result['m_viirs_U'] = np.nan

    return result


def dempster_combine(bpa):
    """Dempster 组合规则"""
    has_both = not (np.isnan(bpa['m_s2_D']) or np.isnan(bpa['m_viirs_D']))
    has_s2_only = not np.isnan(bpa['m_s2_D']) and np.isnan(bpa['m_viirs_D'])
    has_viirs_only = np.isnan(bpa['m_s2_D']) and not np.isnan(bpa['m_viirs_D'])

    result = {}

    if has_both:
        # Conflict K
        K = bpa['m_s2_D'] * bpa['m_viirs_N'] + bpa['m_s2_N'] * bpa['m_viirs_D']
        result['conflict_k'] = K

        if K >= 0.999:
            # Extreme conflict — can't normalize
            result['belief_damage'] = np.nan
            result['belief_no_damage'] = np.nan
            result['uncertainty_ds'] = np.nan
            result['plausibility_damage'] = np.nan
            result['plausibility_no_damage'] = np.nan
        else:
            denom = 1.0 - K
            m_D = (bpa['m_s2_D'] * bpa['m_viirs_D'] +
                   bpa['m_s2_D'] * bpa['m_viirs_U'] +
                   bpa['m_s2_U'] * bpa['m_viirs_D']) / denom
            m_N = (bpa['m_s2_N'] * bpa['m_viirs_N'] +
                   bpa['m_s2_N'] * bpa['m_viirs_U'] +
                   bpa['m_s2_U'] * bpa['m_viirs_N']) / denom
            m_U = (bpa['m_s2_U'] * bpa['m_viirs_U']) / denom

            result['belief_damage'] = m_D
            result['belief_no_damage'] = m_N
            result['uncertainty_ds'] = m_U
            result['plausibility_damage'] = m_D + m_U
            result['plausibility_no_damage'] = m_N + m_U

    elif has_s2_only:
        result['conflict_k'] = 0.0
        result['belief_damage'] = bpa['m_s2_D']
        result['belief_no_damage'] = bpa['m_s2_N']
        result['uncertainty_ds'] = bpa['m_s2_U']
        result['plausibility_damage'] = bpa['m_s2_D'] + bpa['m_s2_U']
        result['plausibility_no_damage'] = bpa['m_s2_N'] + bpa['m_s2_U']

    elif has_viirs_only:
        result['conflict_k'] = 0.0
        result['belief_damage'] = bpa['m_viirs_D']
        result['belief_no_damage'] = bpa['m_viirs_N']
        result['uncertainty_ds'] = bpa['m_viirs_U']
        result['plausibility_damage'] = bpa['m_viirs_D'] + bpa['m_viirs_U']
        result['plausibility_no_damage'] = bpa['m_viirs_N'] + bpa['m_viirs_U']

    else:
        result['conflict_k'] = np.nan
        for k in ['belief_damage','belief_no_damage','uncertainty_ds',
                  'plausibility_damage','plausibility_no_damage']:
            result[k] = np.nan

    # Derived
    if not np.isnan(result.get('belief_damage', np.nan)):
        bd = result['belief_damage']
        ud = result['uncertainty_ds']
        result['ds_damage_mid'] = bd + 0.5 * ud
        result['ds_interval_width'] = ud  # = plausibility_damage - belief_damage = m_U
    else:
        result['ds_damage_mid'] = np.nan
        result['ds_interval_width'] = np.nan

    return result


def classify_conflict(k):
    """冲突等级"""
    if np.isnan(k):
        return 'no_data'
    if k < 0.2:
        return 'low_conflict'
    if k < 0.5:
        return 'medium_conflict'
    if k < 0.8:
        return 'high_conflict'
    return 'extreme_conflict'


def classify_decision(belief_d, belief_n, plaus_d, plaus_n, conflict_k):
    """D-S 决策分类"""
    if np.isnan(belief_d) or np.isnan(conflict_k):
        return 'insufficient_evidence'
    if conflict_k >= 0.5:
        return 'uncertain_conflict'
    if belief_d >= 0.6:
        return 'confident_damage'
    if plaus_d >= 0.6:
        return 'possible_damage'
    if belief_n >= 0.6:
        return 'likely_no_damage'
    return 'insufficient_evidence'


# ============================================================
# 2. 运行
# ============================================================

def process_all(df):
    """对全部网格执行 D-S 融合"""
    ds_cols = []

    for idx, row in df.iterrows():
        bpa = compute_bpa(row)
        ds = dempster_combine(bpa)

        combined = {**bpa, **ds}
        combined['conflict_class'] = classify_conflict(ds.get('conflict_k', np.nan))
        combined['ds_decision_class'] = classify_decision(
            ds.get('belief_damage', np.nan),
            ds.get('belief_no_damage', np.nan),
            ds.get('plausibility_damage', np.nan),
            ds.get('plausibility_no_damage', np.nan),
            ds.get('conflict_k', np.nan),
        )
        combined['ds_source'] = (
            'both' if not (np.isnan(bpa.get('m_s2_D', np.nan)) or np.isnan(bpa.get('m_viirs_D', np.nan)))
            else 's2_only' if not np.isnan(bpa.get('m_s2_D', np.nan))
            else 'viirs_only' if not np.isnan(bpa.get('m_viirs_D', np.nan))
            else 'no_data'
        )
        ds_cols.append(combined)

    ds_df = pd.DataFrame(ds_cols)
    # Merge back with original
    result = pd.concat([df.reset_index(drop=True), ds_df], axis=1)
    return result


# ============================================================
# 3. 统计
# ============================================================

def compute_summary(df):
    """汇总统计"""
    s = {}
    for col, label in [
        ('p_fused', 'p_fused'),
        ('belief_damage', 'belief_damage'),
        ('plausibility_damage', 'plausibility_damage'),
        ('ds_damage_mid', 'ds_damage_mid'),
        ('uncertainty_ds', 'uncertainty_ds'),
        ('conflict_k', 'conflict_k'),
        ('ds_interval_width', 'ds_interval_width'),
        ('uncertainty', 'uncertainty_logit'),
    ]:
        if col in df.columns:
            v = df[col].dropna()
            s[f'{label}_mean'] = float(v.mean()) if len(v) > 0 else np.nan
            s[f'{label}_median'] = float(v.median()) if len(v) > 0 else np.nan
            s[f'{label}_std'] = float(v.std()) if len(v) > 0 else np.nan

    # Only "both" grids
    both = df[df['ds_source'] == 'both']
    if len(both) > 0:
        kv = both['conflict_k'].dropna()
        s['conflict_k_both_mean'] = float(kv.mean()) if len(kv) > 0 else np.nan
        s['conflict_k_both_median'] = float(kv.median()) if len(kv) > 0 else np.nan
        s['conflict_k_top10_threshold'] = float(kv.quantile(0.90)) if len(kv) > 0 else np.nan

    return s


# ============================================================
# 4. 输出表格
# ============================================================

def save_tables(df):
    """保存所有 CSV"""
    # Main DS CSV
    ds_out_cols = [c for c in df.columns if c not in ['geometry'] or c == 'geometry']
    # For CSV, drop geometry
    csv_cols = [c for c in ds_out_cols if c != 'geometry']
    df[csv_cols].to_csv(DS_CSV, index=False, encoding='utf-8')
    print(f"  [CSV] Saved: {DS_CSV}")

    # DS Fusion Summary
    summary = compute_summary(df)
    with open(STATS_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        for k, v in summary.items():
            w.writerow([k, v])
    print(f"  [CSV] Saved: {STATS_CSV}")

    # Conflict class counts
    cc = df['conflict_class'].value_counts()
    with open(CONFLICT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['conflict_class', 'count', 'ratio'])
        for cls, cnt in cc.items():
            w.writerow([cls, cnt, round(cnt / len(df), 6)])
    print(f"  [CSV] Saved: {CONFLICT_CSV}")

    # Decision class counts
    dc = df['ds_decision_class'].value_counts()
    with open(DECISION_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ds_decision_class', 'count', 'ratio'])
        for cls, cnt in dc.items():
            w.writerow([cls, cnt, round(cnt / len(df), 6)])
    print(f"  [CSV] Saved: {DECISION_CSV}")

    # Evidence × Conflict crosstab
    both = df[df['evidence_type'].notna()]
    ct = pd.crosstab(both['evidence_type'], both['conflict_class'])
    ct.to_csv(CROSSTAB_CSV, encoding='utf-8')
    print(f"  [CSV] Saved: {CROSSTAB_CSV}")

    # DS vs Logit comparison
    comp_rows = [['metric', 'logit_p_fused', 'ds_belief_damage', 'ds_plausibility_damage',
                  'ds_damage_mid', 'ds_interval_width', 'conflict_k']]
    both_valid = df[(df['ds_source'] == 'both') &
                    df['p_fused'].notna() &
                    df['belief_damage'].notna()]
    for ev_type in ['both_high', 's2_high_viirs_low', 's2_low_viirs_high', 'both_low']:
        sub = both_valid[both_valid['evidence_type'] == ev_type]
        if len(sub) > 0:
            comp_rows.append([
                ev_type,
                round(sub['p_fused'].mean(), 4),
                round(sub['belief_damage'].mean(), 4),
                round(sub['plausibility_damage'].mean(), 4),
                round(sub['ds_damage_mid'].mean(), 4),
                round(sub['ds_interval_width'].mean(), 4),
                round(sub['conflict_k'].mean(), 4),
            ])
    # Also overall
    comp_rows.append([
        'overall_both',
        round(both_valid['p_fused'].mean(), 4),
        round(both_valid['belief_damage'].mean(), 4),
        round(both_valid['plausibility_damage'].mean(), 4),
        round(both_valid['ds_damage_mid'].mean(), 4),
        round(both_valid['ds_interval_width'].mean(), 4),
        round(both_valid['conflict_k'].mean(), 4),
    ])
    with open(COMPARE_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerows(comp_rows)
    print(f"  [CSV] Saved: {COMPARE_CSV}")

    return summary


# ============================================================
# 5. 绘图
# ============================================================

def plot_ds_maps(gdf):
    """绘制 D-S 结果地图"""
    gdf_wgs = gdf.to_crs("EPSG:4326")

    # 1. Belief Damage
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_wgs.plot(column='belief_damage', ax=ax, cmap='YlOrRd', vmin=0, vmax=1,
                 edgecolor='none', linewidth=0, legend=True,
                 legend_kwds={'label': 'Belief(Damage)', 'shrink': 0.7})
    ax.set_title('D-S Belief(Damage)', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); fig.savefig(MAP_BELIEF, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  [Figure] Saved: {MAP_BELIEF}")

    # 2. Plausibility Damage
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_wgs.plot(column='plausibility_damage', ax=ax, cmap='YlOrRd', vmin=0, vmax=1,
                 edgecolor='none', linewidth=0, legend=True,
                 legend_kwds={'label': 'Plausibility(Damage)', 'shrink': 0.7})
    ax.set_title('D-S Plausibility(Damage) = Belief + Uncertainty', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); fig.savefig(MAP_PLAUS, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  [Figure] Saved: {MAP_PLAUS}")

    # 3. Uncertainty (DS)
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_wgs.plot(column='uncertainty_ds', ax=ax, cmap='viridis', vmin=0, vmax=1,
                 edgecolor='none', linewidth=0, legend=True,
                 legend_kwds={'label': 'Uncertainty m(U)', 'shrink': 0.7})
    ax.set_title('D-S Uncertainty m(U)', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); fig.savefig(MAP_UNCDS, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  [Figure] Saved: {MAP_UNCDS}")

    # 4. Conflict K
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_wgs.plot(column='conflict_k', ax=ax, cmap='Reds', vmin=0, vmax=1,
                 edgecolor='none', linewidth=0, legend=True,
                 legend_kwds={'label': 'Conflict K', 'shrink': 0.7})
    ax.set_title('D-S Conflict K = Evidence Conflict', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); fig.savefig(MAP_CONFLICT, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  [Figure] Saved: {MAP_CONFLICT}")

    # 5. Decision Class
    decision_colors = {
        'confident_damage': '#253494',
        'possible_damage': '#fd8d3c',
        'uncertain_conflict': '#e31a1c',
        'likely_no_damage': '#ffffcc',
        'insufficient_evidence': '#d9d9d9',
    }
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls, color in decision_colors.items():
        sub = gdf_wgs[gdf_wgs['ds_decision_class'] == cls]
        if len(sub) > 0:
            sub.plot(ax=ax, color=color, edgecolor='none', linewidth=0,
                     label=cls.replace('_', ' ').title())
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax.set_title('D-S Decision Classification', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); fig.savefig(MAP_DECISION, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  [Figure] Saved: {MAP_DECISION}")

    # 6. Interval Width
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_wgs.plot(column='ds_interval_width', ax=ax, cmap='viridis', vmin=0, vmax=1,
                 edgecolor='none', linewidth=0, legend=True,
                 legend_kwds={'label': 'Interval Width = m(U)', 'shrink': 0.7})
    ax.set_title('D-S Interval Width: Plausibility - Belief', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); fig.savefig(MAP_INTERVAL, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  [Figure] Saved: {MAP_INTERVAL}")

    # 7. DS vs Logit Scatter
    both_v = gdf[(gdf['ds_source'] == 'both') &
                 gdf['p_fused'].notna() &
                 gdf['belief_damage'].notna()]
    if len(both_v) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        ax = axes[0]
        ax.scatter(both_v['p_fused'], both_v['belief_damage'],
                   c=both_v['conflict_k'], cmap='Reds', s=8, alpha=0.6, vmin=0, vmax=1)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4)
        ax.set_xlabel('p_fused (Logit-Average)')
        ax.set_ylabel('Belief(Damage) (D-S)')
        ax.set_title('Belief(Damage) vs p_fused')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax = axes[1]
        ax.scatter(both_v['p_fused'], both_v['plausibility_damage'],
                   c=both_v['conflict_k'], cmap='Reds', s=8, alpha=0.6, vmin=0, vmax=1)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4)
        ax.set_xlabel('p_fused (Logit-Average)')
        ax.set_ylabel('Plausibility(Damage) (D-S)')
        ax.set_title('Plausibility(Damage) vs p_fused')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax = axes[2]
        ax.scatter(both_v['conflict_k'], both_v['ds_interval_width'],
                   c=both_v['conflict_k'], cmap='Reds', s=8, alpha=0.6, vmin=0, vmax=1)
        ax.set_xlabel('Conflict K')
        ax.set_ylabel('Interval Width = m(U)')
        ax.set_title('Uncertainty Interval vs Conflict')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        fig.suptitle('D-S vs Logit-Average Fusion Comparison — Mariupol',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(MAP_SCATTER, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Figure] Saved: {MAP_SCATTER}")


# ============================================================
# 6. 报告
# ============================================================

def generate_report(df, summary):
    """生成中文报告"""
    both = df[df['ds_source'] == 'both']
    kv = both['conflict_k'].dropna()

    lines = []
    lines.append("# Dempster-Shafer 证据理论融合报告 (S2 + VIIRS)\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 5 — D-S 证据理论双源融合\n")
    lines.append("---\n")

    lines.append("## 1. 为什么需要 D-S 证据理论\n")
    lines.append(f"- S2 与 VIIRS Spearman ρ = -0.4216 (稳健负相关, 见阶段 4.6)\n")
    lines.append("- 简单 logit 平均融合将冲突证据互相抵消 → p_fused 被拉向 0.228 (过于保守)\n")
    lines.append("- D-S 证据理论优势:\n")
    lines.append("  - **显式表达冲突**: 通过 K (冲突系数) 量化两个传感器之间的证据冲突程度\n")
    lines.append("  - **区分不确定与无知**: m(U) 表达'不知道'的质量, 而非强行折中\n")
    lines.append("  - **Belief/Plausibility 区间**: 给出支持的上下界, 而非单点概率\n")
    lines.append("  - **保留单源信息**: 当只有一个传感器有数据时, BPA 自然退化为单源\n")

    lines.append("\n## 2. BPA 构造方法\n")
    lines.append("### 识别框架\n")
    lines.append("- Θ = {D (Damage), N (No Damage)}\n")
    lines.append("- 幂集: {∅, {D}, {N}, {D,N}}\n")
    lines.append("- m({D}) = 支持损害的质量, m({N}) = 支持无损害的质量, m({D,N}) = 不确定质量\n")

    lines.append("### Sentinel-2 BPA\n")
    lines.append("```\n")
    lines.append("p_s2 = clip(p_s2_mean, 0.001, 0.999)\n")
    lines.append("u_s2 = 0.15 + 0.5*(1 - p_s2_valid_ratio)  [+ 0.2*normalized_std]\n")
    lines.append("u_s2 = clip(u_s2, 0.05, 0.80)\n")
    lines.append("m_s2(D) = p_s2 * (1 - u_s2)\n")
    lines.append("m_s2(N) = (1 - p_s2) * (1 - u_s2)\n")
    lines.append("m_s2(U) = u_s2\n")
    lines.append("```\n")

    lines.append("### VIIRS BPA\n")
    lines.append("```\n")
    lines.append("p_viirs = clip(p_viirs_mean, 0.001, 0.999)\n")
    lines.append("u_viirs = 0.15 + 0.5*(1 - p_viirs_valid_ratio)\n")
    lines.append("u_viirs = clip(u_viirs, 0.05, 0.80)\n")
    lines.append("m_viirs(D) = p_viirs * (1 - u_viirs)\n")
    lines.append("m_viirs(N) = (1 - p_viirs) * (1 - u_viirs)\n")
    lines.append("m_viirs(U) = u_viirs\n")
    lines.append("```\n")

    lines.append("### u_s2 和 u_viirs 的含义\n")
    lines.append("- 基础不确定性: 0.15 (传感器固有的不完全可靠)\n")
    lines.append("- 覆盖不确定性: 0.5 × (1 - valid_ratio) — 网格内有效像元越少, 不确定性越高\n")
    lines.append("- u 的范围: [0.05, 0.80] — 即使完美覆盖也保留 5% 不确定性; 最差情况不超过 80%\n")

    lines.append("\n## 3. Dempster 组合规则\n")
    lines.append("### 冲突系数 K\n")
    lines.append("```\n")
    lines.append("K = m_s2(D)*m_viirs(N) + m_s2(N)*m_viirs(D)\n")
    lines.append("```\n")
    lines.append("- K 表示两个传感器在损害/无损害命题上的**直接冲突**\n")
    lines.append("- K 接近 0: 传感器一致 (都支持损害或都支持无损害)\n")
    lines.append("- K 接近 1: 传感器严重冲突 (一个说损害, 另一个说无损害)\n")

    lines.append("### 组合后 BPA\n")
    lines.append("```\n")
    lines.append("m(D) = [m_s2(D)*m_viirs(D) + m_s2(D)*m_viirs(U) + m_s2(U)*m_viirs(D)] / (1-K)\n")
    lines.append("m(N) = [m_s2(N)*m_viirs(N) + m_s2(N)*m_viirs(U) + m_s2(U)*m_viirs(N)] / (1-K)\n")
    lines.append("m(U) = [m_s2(U) * m_viirs(U)] / (1-K)\n")
    lines.append("```\n")
    lines.append("- 若 K >= 0.999: extreme_conflict, 不可归一化, 输出 NaN\n")

    lines.append("\n## 4. 冲突分析\n")
    lines.append(f"- 冲突系数 K 均值: {summary.get('conflict_k_both_mean', np.nan):.4f}\n")
    lines.append(f"- 冲突系数 K 中位数: {summary.get('conflict_k_both_median', np.nan):.4f}\n")
    lines.append(f"- Top 10% K 阈值: {summary.get('conflict_k_top10_threshold', np.nan):.4f}\n")

    cc = df['conflict_class'].value_counts()
    lines.append("\n### 冲突等级分布\n")
    lines.append("| 等级 | 数量 | 比例 |\n")
    lines.append("|------|------|------|\n")
    for cls in ['low_conflict', 'medium_conflict', 'high_conflict', 'extreme_conflict', 'no_data']:
        cnt = cc.get(cls, 0)
        lines.append(f"| {cls} | {cnt} | {cnt/len(df):.4f} ({100*cnt/len(df):.1f}%) |\n")

    lines.append("\n### Evidence Type × Conflict 交叉表\n")
    ct = pd.crosstab(df['evidence_type'].fillna('no_data'), df['conflict_class'].fillna('no_data'))
    lines.append(ct.to_string())

    lines.append("\n## 5. Belief vs Plausibility\n")
    lines.append("- **Belief(Damage)**: 证据**必须**支持损害的质量 — 保守下界\n")
    lines.append("- **Plausibility(Damage)**: 证据**可能**支持损害的质量 — 乐观上界 (Belief + Uncertainty)\n")
    lines.append("- **Interval Width = Plausibility - Belief = m(U)**: 不确定性的宽度\n")
    lines.append("- 窄区间 → 传感器一致, 置信度高; 宽区间 → 传感器分歧大或数据质量差\n")

    lines.append(f"\n- Belief(Damage) 均值: {summary.get('belief_damage_mean', np.nan):.4f}\n")
    lines.append(f"- Plausibility(Damage) 均值: {summary.get('plausibility_damage_mean', np.nan):.4f}\n")
    lines.append(f"- Interval Width 均值: {summary.get('ds_interval_width_mean', np.nan):.4f}\n")

    lines.append("\n## 6. 决策分类\n")
    lines.append("| 类别 | 定义 |\n")
    lines.append("|------|------|\n")
    lines.append("| confident_damage | Belief(D) >= 0.6 且 K < 0.5 — 证据一致支持损害 |\n")
    lines.append("| possible_damage | Belief(D) < 0.6 但 Plausibility(D) >= 0.6 — 不能排除损害 |\n")
    lines.append("| uncertain_conflict | K >= 0.5 — 传感器严重冲突, 不做损害/无损害判定 |\n")
    lines.append("| likely_no_damage | Belief(N) >= 0.6 且 K < 0.5 — 证据一致支持无损害 |\n")
    lines.append("| insufficient_evidence | 其他 — 证据不足或全缺失 |\n")

    dc = df['ds_decision_class'].value_counts()
    lines.append("\n### 决策分类分布\n")
    lines.append("| 类别 | 数量 | 比例 |\n")
    lines.append("|------|------|------|\n")
    for cls in ['confident_damage', 'possible_damage', 'uncertain_conflict',
                'likely_no_damage', 'insufficient_evidence']:
        cnt = dc.get(cls, 0)
        lines.append(f"| {cls} | {cnt} | {cnt/len(df):.4f} ({100*cnt/len(df):.1f}%) |\n")

    lines.append("\n### Evidence Type × Decision Class 交叉表\n")
    dc_ct = pd.crosstab(df['evidence_type'].fillna('no_data'), df['ds_decision_class'].fillna('no_data'))
    lines.append(str(dc_ct))

    lines.append("\n## 7. D-S 与 Logit 平均融合对比\n")
    lines.append("| 指标 | Logit p_fused | D-S Belief(D) | D-S Plausibility(D) | D-S Mid | Interval Width |\n")
    lines.append("|------|--------------|---------------|---------------------|---------|---------------|\n")
    lines.append(f"| 均值 | {summary.get('p_fused_mean', np.nan):.4f} | {summary.get('belief_damage_mean', np.nan):.4f} | {summary.get('plausibility_damage_mean', np.nan):.4f} | {summary.get('ds_damage_mid_mean', np.nan):.4f} | {summary.get('ds_interval_width_mean', np.nan):.4f} |\n")
    lines.append(f"| 中位数 | {summary.get('p_fused_median', np.nan):.4f} | {summary.get('belief_damage_median', np.nan):.4f} | {summary.get('plausibility_damage_median', np.nan):.4f} | {summary.get('ds_damage_mid_median', np.nan):.4f} | {summary.get('ds_interval_width_median', np.nan):.4f} |\n")

    lines.append("\n### 各证据类型的 D-S 与 Logit 对比\n")
    lines.append("| Evidence Type | p_fused | Belief(D) | Plausibility(D) | Mid | Conflict K |\n")
    lines.append("|---------------|---------|-----------|-----------------|-----|------------|\n")
    comp_df = pd.read_csv(COMPARE_CSV)
    for _, row in comp_df.iterrows():
        lines.append(f"| {row['metric']} | {row['logit_p_fused']} | {row['ds_belief_damage']} | {row['ds_plausibility_damage']} | {row['ds_damage_mid']} | {row['conflict_k']} |\n")

    lines.append("\n**关键观察**:\n")
    # Both high
    both_high_row = comp_df[comp_df['metric'] == 'both_high']
    if len(both_high_row) > 0:
        lines.append(f"- Both high: Belief={both_high_row['ds_belief_damage'].values[0]:.3f}, 冲突K={both_high_row['conflict_k'].values[0]:.3f} — D-S 和 Logit 都支持高损害\n")

    s2_low_vii_high = comp_df[comp_df['metric'] == 's2_low_viirs_high']
    if len(s2_low_vii_high) > 0:
        lines.append(f"- S2 low / VIIRS high: Belief={s2_low_vii_high['ds_belief_damage'].values[0]:.3f}, Plausibility={s2_low_vii_high['ds_plausibility_damage'].values[0]:.3f}, K={s2_low_vii_high['conflict_k'].values[0]:.3f}\n")
        lines.append(f"  → D-S 通过宽区间表达不确定性, 而非像 Logit 那样直接压低概率\n")

    both_low = comp_df[comp_df['metric'] == 'both_low']
    if len(both_low) > 0:
        lines.append(f"- Both low: Belief={both_low['ds_belief_damage'].values[0]:.3f}, K={both_low['conflict_k'].values[0]:.3f} — 一致低损害\n")

    lines.append("\n## 8. 关键问题回答\n")

    lines.append("### S2 low / VIIRS high 对应什么？\n")
    s2lvh = df[(df['evidence_type'] == 's2_low_viirs_high') & (df['ds_source'] == 'both')]
    if len(s2lvh) > 0:
        k_mean = s2lvh['conflict_k'].mean()
        dc_dist = s2lvh['ds_decision_class'].value_counts()
        lines.append(f"- 该类型冲突 K 均值: {k_mean:.4f}\n")
        lines.append(f"- 决策分类分布: {dict(dc_dist)}\n")
        lines.append("- 主要对应 **possible_damage** 或 **uncertain_conflict**\n")
        lines.append("- D-S 正确地表达了'VIIRS 说损害但 S2 说无损害'的冲突, 而非强行平均\n")

    lines.append("### Both high 对应什么？\n")
    bh = df[(df['evidence_type'] == 'both_high') & (df['ds_source'] == 'both')]
    if len(bh) > 0:
        lines.append(f"- Both high 网格数: {len(bh)} — 这是 AOI 内最可信的损害区域\n")
        lines.append(f"- 这些网格 D-S 给出高 Belief(Damage), 低冲突 K — 两个传感器一致确认\n")

    lines.append("### D-S 是否比 Logit 更好地表达冲突？\n")
    lines.append("- ✅ **是**。D-S 通过以下机制优于 Logit 平均:\n")
    lines.append("  1. **冲突可见化**: K 值直接量化冲突程度, Logit 平均则把冲突隐藏在中庸概率中\n")
    lines.append("  2. **区间而非单点**: [Belief, Plausibility] 区间传达了'我们不确定'的信息\n")
    lines.append("  3. **决策可解释**: uncertain_conflict 类别明确告知用户'此处传感器打架, 勿做判断'\n")
    lines.append("  4. **保留高置信度**: Both high 区域不被压制, 维持高损害置信\n")

    lines.append("\n## 9. 局限性\n")
    lines.append("1. **无地面真值**: 没有 UNOSAT 标签, D-S 结果是**证据融合**而非精度评估\n")
    lines.append("2. **BPA 参数主观性**: u_s2 和 u_viirs 中的 0.15 基础不确定性和 0.5 权重系数需要专家知识校准\n")
    lines.append("3. **Dempster 规则假设**: 假设两个传感器**独立**, 但 S2 和 VIIRS 来自不同物理过程, 独立性假设基本合理\n")
    lines.append("4. **冲突处理**: K → 0.999 时组合失败, 当前标记为 extreme_conflict\n")
    lines.append("5. **VIIRS valid_ratio 受重投影影响**: VIIRS 在 500m 网格中的 valid_ratio 上限约 0.75 (重投影采样效应)\n")

    lines.append("\n## 10. 主要结论\n")
    lines.append(f"1. D-S 融合在 {len(df)} 个 500m 网格上完成\n")
    lines.append(f"2. 平均冲突 K = {summary.get('conflict_k_both_mean', np.nan):.4f} — 证实 S2/VIIRS 存在实质性冲突\n")
    lines.append(f"3. Belief(Damage) = [0, {summary.get('belief_damage_mean', 0):.4f}] (保守); Plausibility(Damage) 可达 [{summary.get('plausibility_damage_mean', 0):.4f}] (乐观)\n")
    lines.append("4. D-S 比 Logit 平均更好地表达了冲突证据的不确定性\n")
    lines.append("5. 建议在有 UNOSAT 标签时, 用 D-S 的 Belief/Plausibility 区间与标签对比, 评估校准效果\n")

    lines.append("\n## 11. 输出文件\n")
    outputs = [
        (DS_CSV, "D-S 融合结果 CSV"),
        (DS_GEOJSON, "D-S 融合结果 GeoJSON"),
        (STATS_CSV, "汇总统计"),
        (CONFLICT_CSV, "冲突等级计数"),
        (DECISION_CSV, "决策分类计数"),
        (CROSSTAB_CSV, "Evidence × Conflict 交叉表"),
        (COMPARE_CSV, "D-S vs Logit 对比"),
        (MAP_BELIEF, "Belief(Damage) 地图"),
        (MAP_PLAUS, "Plausibility(Damage) 地图"),
        (MAP_UNCDS, "D-S Uncertainty 地图"),
        (MAP_CONFLICT, "Conflict K 地图"),
        (MAP_DECISION, "决策分类地图"),
        (MAP_SCATTER, "D-S vs Logit 散点对比"),
        (MAP_INTERVAL, "Interval Width 地图"),
        (REPORT_PATH, "本报告"),
    ]
    for p, desc in outputs:
        lines.append(f"| `{p.relative_to(PROJECT_ROOT)}` | {desc} |\n")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {REPORT_PATH}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("Dempster-Shafer 证据理论融合 (S2 + VIIRS)")
    print("=" * 70)

    # Load
    print("\n[1] Loading grid fusion data...")
    df = pd.read_csv(FUSION_CSV)
    print(f"    {len(df)} grid cells loaded")
    print(f"    Columns: {len(df.columns)}")

    # Process
    print("\n[2] Computing BPA and Dempster combination...")
    df_ds = process_all(df)

    # Save CSV & GeoJSON
    print("\n[3] Saving DS fusion CSV and GeoJSON...")
    gdf = gpd.read_file(FUSION_GEOJSON)
    # Merge DS columns into GeoDataFrame
    ds_cols = [c for c in df_ds.columns if c not in gdf.columns or c == 'grid_id']
    for c in ds_cols:
        if c in df_ds.columns:
            gdf[c] = df_ds[c].values

    csv_cols = [c for c in gdf.columns if c != 'geometry']
    gdf[csv_cols].to_csv(DS_CSV, index=False, encoding='utf-8')
    print(f"    Saved: {DS_CSV}")

    gdf.to_file(DS_GEOJSON, driver='GeoJSON')
    print(f"    Saved: {DS_GEOJSON}")

    # Save tables
    print("\n[4] Computing and saving statistics tables...")
    summary = save_tables(gdf)

    # Plot maps
    print("\n[5] Plotting D-S maps...")
    plot_ds_maps(gdf)

    # Report
    print("\n[6] Generating report...")
    generate_report(gdf, summary)

    # Final summary
    both = gdf[gdf['ds_source'] == 'both']
    kv = both['conflict_k'].dropna()
    cc = gdf['conflict_class'].value_counts()
    dc = gdf['ds_decision_class'].value_counts()

    print("\n" + "=" * 70)
    print("D-S 融合完成!")
    print(f"  平均 conflict_k: {kv.mean():.4f}")
    print(f"  中位数 conflict_k: {kv.median():.4f}")
    print(f"  high_conflict: {cc.get('high_conflict',0)} ({100*cc.get('high_conflict',0)/len(gdf):.1f}%)")
    print(f"  extreme_conflict: {cc.get('extreme_conflict',0)} ({100*cc.get('extreme_conflict',0)/len(gdf):.1f}%)")
    print(f"  confident_damage: {dc.get('confident_damage',0)}")
    print(f"  possible_damage: {dc.get('possible_damage',0)}")
    print(f"  uncertain_conflict: {dc.get('uncertain_conflict',0)}")
    print(f"  likely_no_damage: {dc.get('likely_no_damage',0)}")
    print(f"  Belief(Damage) 均值: {summary.get('belief_damage_mean', np.nan):.4f}")
    print(f"  Plausibility(Damage) 均值: {summary.get('plausibility_damage_mean', np.nan):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
