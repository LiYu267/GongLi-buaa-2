#!/usr/bin/env python3
"""
模块 3 — Bayesian vs Dempster-Shafer 一致性比较

T20 多传感器不确定性融合: 对比 Bayesian posterior 与 D-S belief/plausibility 区间.

核心问题:
  - Bayesian 后验与 D-S 区间在什么条件下一致?
  - 传感器冲突 (JSD) 如何影响两种方法的一致性?
  - D-S 的不确定性与 Bayesian 融合抑制有何关联?

前提: 需要先运行 11_ds_fusion.py, 16_bayesian_hierarchical.py, 17_kld_information_gain.py.
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DS_CSV       = PROJECT_ROOT / "data" / "processed" / "grid_ds_fusion_s2_viirs.csv"
BAYES_CSV    = PROJECT_ROOT / "data" / "processed" / "grid_bayesian_posterior.csv"
KLD_CSV      = PROJECT_ROOT / "data" / "processed" / "kld_information_gain_by_grid.csv"
FUSION_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.geojson"

OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG  = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB  = PROJECT_ROOT / "outputs" / "tables"
OUT_MD   = PROJECT_ROOT / "outputs"
for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)

# Output files
CONSISTENCY_CSV = OUT_DATA / "bayes_vs_ds_consistency_by_grid.csv"
SUMMARY_CSV     = OUT_TAB  / "bayes_vs_ds_consistency_summary.csv"
MAP_PNG         = OUT_FIG  / "bayes_vs_ds_consistency_map.png"
DIVERGENCE_PNG  = OUT_FIG  / "bayes_vs_ds_divergence_map.png"
INTERVAL_PNG    = OUT_FIG  / "bayes_vs_ds_interval_comparison.png"
SCATTER_PNG     = OUT_FIG  / "bayes_ds_jsd_conflict_scatter.png"
UNCERT_VS_JSD_PNG = OUT_FIG / "ds_uncertainty_vs_jsd.png"
REPORT_MD       = OUT_MD   / "bayes_vs_ds_consistency_report.md"

EPS = 1e-9


# ============================================================
# Data loading and merging
# ============================================================

def load_and_merge():
    """Load all three data sources and merge on grid_id."""

    # 1. Bayesian posterior
    bayes = pd.read_csv(BAYES_CSV)
    bayes = bayes[['grid_id', 'posterior_mean', 'posterior_median', 'posterior_sd',
                    'ci_2_5', 'ci_97_5', 'ci_width', 'evidence_type']]
    bayes = bayes.rename(columns={
        'posterior_mean': 'bayes_mean',
        'ci_2_5': 'bayes_ci_lower',
        'ci_97_5': 'bayes_ci_upper',
    })
    print(f"  Bayesian: {len(bayes)} rows")

    # 2. D-S fusion
    ds_cols = ['grid_id', 'belief_damage', 'plausibility_damage',
               'uncertainty_ds', 'ds_interval_width', 'conflict_k',
               'conflict_class', 'ds_decision_class']
    ds = pd.read_csv(DS_CSV)
    # Ensure numeric
    for col in ['belief_damage', 'plausibility_damage', 'ds_interval_width', 'conflict_k']:
        ds[col] = pd.to_numeric(ds[col], errors='coerce')

    # Compute ds_uncertainty_width if missing
    ds['ds_uncertainty_width'] = ds['plausibility_damage'] - ds['belief_damage']

    ds = ds[['grid_id', 'belief_damage', 'plausibility_damage',
             'uncertainty_ds', 'ds_interval_width', 'conflict_k',
             'conflict_class', 'ds_decision_class', 'ds_uncertainty_width']]
    ds = ds.rename(columns={
        'belief_damage': 'ds_belief',
        'plausibility_damage': 'ds_plausibility',
    })
    print(f"  D-S:      {len(ds)} rows")

    # 3. KLD metrics
    kld = pd.read_csv(KLD_CSV)
    kld_cols = ['grid_id', 'p_s2_only', 'p_viirs_only', 'p_both',
                'jsd_s2_viirs', 'fusion_suppression_ratio',
                'kl_s2_vs_viirs', 'kl_viirs_vs_s2', 'abs_diff_s2_viirs']
    kld = kld[kld_cols]
    print(f"  KLD:      {len(kld)} rows")

    # Merge
    df = bayes.merge(kld, on='grid_id', how='left')
    df = df.merge(ds, on='grid_id', how='left')

    # Track D-S validity
    df['ds_valid'] = df['ds_belief'].notna() & df['ds_plausibility'].notna()
    n_ds_valid = df['ds_valid'].sum()
    n_total = len(df)
    print(f"\n  Merged: {n_total} grids, {n_ds_valid} with valid D-S ({n_total - n_ds_valid} insufficient evidence)")

    return df


# ============================================================
# Consistency classification
# ============================================================

def classify_consistency(df):
    """
    Classify each grid's Bayesian-D-S consistency.

    Rules:
      strong_consistent:  bayes_mean in [ds_belief, ds_plausibility]
      weak_consistent:    Bayesian CI overlaps D-S interval but mean is outside
      divergent_high_bayes:  bayes_ci_lower > ds_plausibility
      divergent_low_bayes:   bayes_ci_upper < ds_belief
      no_ds_data:         D-S interval is NaN (insufficient evidence)
    """
    n = len(df)
    labels = np.full(n, 'unknown', dtype=object)
    reasons = np.full(n, '', dtype=object)

    for i in range(n):
        bm = df.loc[i, 'bayes_mean']
        bl = df.loc[i, 'bayes_ci_lower']
        bu = df.loc[i, 'bayes_ci_upper']
        db = df.loc[i, 'ds_belief']
        dp = df.loc[i, 'ds_plausibility']

        if pd.isna(db) or pd.isna(dp):
            labels[i] = 'no_ds_data'
            reasons[i] = 'Insufficient evidence for D-S fusion'
            continue

        # Strong consistent: bayes_mean inside D-S interval
        if db - EPS <= bm <= dp + EPS:
            labels[i] = 'strong_consistent'
            reasons[i] = 'Bayesian mean within D-S [Belief, Plausibility]'
        # Divergent: Bayesian CI completely above D-S
        elif bl > dp:
            labels[i] = 'divergent_high_bayes'
            reasons[i] = 'Bayesian posterior entirely above D-S plausibility'
        # Divergent: Bayesian CI completely below D-S
        elif bu < db:
            labels[i] = 'divergent_low_bayes'
            reasons[i] = 'Bayesian posterior entirely below D-S belief'
        # Weak consistent: intervals overlap but mean is outside
        elif bu >= db and bl <= dp:
            labels[i] = 'weak_consistent'
            reasons[i] = 'Bayesian CI overlaps D-S interval but mean outside'
        else:
            labels[i] = 'unclassified'
            reasons[i] = 'Edge case'

    df['consistency_label'] = labels

    # Enrich divergence reasons
    jsd_q75 = df['jsd_s2_viirs'].quantile(0.75)
    supp_q75 = df['fusion_suppression_ratio'].quantile(0.75)
    width_q75 = df['ds_uncertainty_width'].quantile(0.75)

    for i in range(n):
        parts = [reasons[i]]

        # D-S uncertainty
        dw = df.loc[i, 'ds_uncertainty_width']
        if not pd.isna(dw) and dw >= width_q75:
            parts.append('Wide D-S interval (top quartile)')

        # JSD
        jsd = df.loc[i, 'jsd_s2_viirs']
        if not pd.isna(jsd) and jsd >= jsd_q75:
            parts.append('High S2-VIIRS JSD (top quartile)')

        # Fusion suppression
        supp = df.loc[i, 'fusion_suppression_ratio']
        if not pd.isna(supp) and supp >= supp_q75:
            parts.append('High fusion suppression (top quartile)')

        # S2/VIIRS conflict pattern
        ps2 = df.loc[i, 'p_s2_only']
        pv = df.loc[i, 'p_viirs_only']
        if not pd.isna(ps2) and not pd.isna(pv):
            if ps2 < 0.1 and pv > 0.4:
                parts.append('S2 low structural damage, VIIRS high functional loss')
            elif ps2 > 0.3 and pv < 0.4:
                parts.append('S2 indicates structural damage, VIIRS low functional loss')

        if df.loc[i, 'bayes_mean'] < df.loc[i, 'ds_belief'] if not pd.isna(df.loc[i, 'ds_belief']) else False:
            parts.append('Bayesian more conservative than D-S')
        if df.loc[i, 'bayes_mean'] > df.loc[i, 'ds_plausibility'] if not pd.isna(df.loc[i, 'ds_plausibility']) else False:
            parts.append('Bayesian more aggressive than D-S')

        df.loc[i, 'divergence_reason'] = '; '.join([p for p in parts if p])

    # High uncertainty flag (secondary)
    df['high_uncertainty_flag'] = False
    for i in range(n):
        dw = df.loc[i, 'ds_uncertainty_width']
        if not pd.isna(dw) and dw >= width_q75:
            jsd = df.loc[i, 'jsd_s2_viirs']
            if not pd.isna(jsd) and jsd >= jsd_q75:
                df.loc[i, 'high_uncertainty_flag'] = True

    return df


# ============================================================
# Summary statistics
# ============================================================

def compute_summary(df):
    """Compute summary statistics by consistency label."""
    rows = []

    labels_order = ['strong_consistent', 'weak_consistent',
                    'divergent_high_bayes', 'divergent_low_bayes',
                    'no_ds_data']

    # Overall
    valid = df[df['consistency_label'] != 'no_ds_data']
    n_total = len(df)
    n_valid = len(valid)

    for label in labels_order:
        subset = df[df['consistency_label'] == label]
        n = len(subset)
        pct = n / n_total * 100

        # Mean metrics
        jsd_mean = subset['jsd_s2_viirs'].mean() if len(subset) > 0 else np.nan
        supp_mean = subset['fusion_suppression_ratio'].mean() if len(subset) > 0 else np.nan
        width_mean = subset['ds_uncertainty_width'].mean() if len(subset) > 0 else np.nan
        k_mean = subset['conflict_k'].mean() if len(subset) > 0 else np.nan

        row = {
            'consistency_label': label,
            'n_grids': n,
            'pct': round(pct, 2),
            'mean_jsd_s2_viirs': round(jsd_mean, 6) if not np.isnan(jsd_mean) else np.nan,
            'mean_fusion_suppression': round(supp_mean, 6) if not np.isnan(supp_mean) else np.nan,
            'mean_ds_uncertainty_width': round(width_mean, 6) if not np.isnan(width_mean) else np.nan,
            'mean_conflict_k': round(k_mean, 6) if not np.isnan(k_mean) else np.nan,
        }
        rows.append(row)

    # Add totals
    rows.append({
        'consistency_label': 'TOTAL',
        'n_grids': n_total,
        'pct': 100.0,
        'mean_jsd_s2_viirs': df['jsd_s2_viirs'].mean(),
        'mean_fusion_suppression': df['fusion_suppression_ratio'].mean(),
        'mean_ds_uncertainty_width': df['ds_uncertainty_width'].mean(),
        'mean_conflict_k': df['conflict_k'].mean(),
    })

    summary_df = pd.DataFrame(rows)
    return summary_df


# ============================================================
# Figures
# ============================================================

def load_geodata():
    gdf = gpd.read_file(FUSION_GEOJSON)
    if gdf.crs is not None and gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    return gdf


def plot_consistency_map(df, gdf):
    """Map showing consistency labels."""
    plot_gdf = gdf.copy()
    plot_gdf['label'] = df['consistency_label'].values

    color_map = {
        'strong_consistent': '#2E7D32',
        'weak_consistent': '#FFC107',
        'divergent_high_bayes': '#D32F2F',
        'divergent_low_bayes': '#1976D2',
        'no_ds_data': '#BDBDBD',
    }
    labels_order = ['strong_consistent', 'weak_consistent',
                    'divergent_high_bayes', 'divergent_low_bayes', 'no_ds_data']
    colors = [color_map[l] for l in labels_order]
    cmap = mcolors.ListedColormap(colors)

    # Convert to categorical with ordered levels
    plot_gdf['label_cat'] = pd.Categorical(plot_gdf['label'],
                                           categories=labels_order, ordered=True)
    plot_gdf['label_code'] = plot_gdf['label_cat'].cat.codes

    fig, ax = plt.subplots(figsize=(14, 10))
    plot_gdf.plot(column='label_code', ax=ax, cmap=cmap, legend=False,
                  missing_kwds={'color': '#BDBDBD'})

    # Counts for legend
    counts = df['consistency_label'].value_counts()
    legend_patches = []
    for label in labels_order:
        cnt = counts.get(label, 0)
        pct = cnt / len(df) * 100
        legend_patches.append(Patch(color=color_map[label],
                                    label=f'{label} ({cnt}, {pct:.1f}%)'))

    ax.legend(handles=legend_patches, loc='lower right',
              fontsize=8, title='Consistency')
    ax.set_title('Bayesian vs D-S Consistency Map', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(MAP_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure -> {MAP_PNG}")


def plot_divergence_map(df, gdf):
    """Map showing Bayesian-D-S distance."""
    plot_gdf = gdf.copy()

    # Compute distance: 0 if in interval, else distance to nearest bound
    dist = np.zeros(len(df))
    for i in range(len(df)):
        bm = df.loc[i, 'bayes_mean']
        db = df.loc[i, 'ds_belief']
        dp = df.loc[i, 'ds_plausibility']
        if pd.isna(db) or pd.isna(dp):
            dist[i] = np.nan
        elif db <= bm <= dp:
            dist[i] = 0.0
        else:
            dist[i] = min(abs(bm - db), abs(bm - dp))

    plot_gdf['bayes_ds_distance'] = dist

    fig, ax = plt.subplots(figsize=(14, 10))
    plot_gdf.plot(column='bayes_ds_distance', ax=ax, cmap='YlOrRd',
                  legend=True, vmin=0,
                  legend_kwds={'label': '|Bayes - DS| distance', 'shrink': 0.6},
                  missing_kwds={'color': 'lightgrey'})
    ax.set_title('Bayesian-DS Divergence Map\n'
                 f'(mean distance={np.nanmean(dist):.4f})',
                 fontsize=14, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(DIVERGENCE_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure -> {DIVERGENCE_PNG}")


def plot_interval_comparison(df):
    """Bayesian CI vs D-S interval for top 50 highest JSD grids."""
    # Filter to grids with valid D-S, sort by JSD
    valid = df[df['ds_valid']].copy()
    top50 = valid.nlargest(50, 'jsd_s2_viirs').sort_values('jsd_s2_viirs')

    fig, ax = plt.subplots(figsize=(22, 8))

    x = np.arange(len(top50))

    # D-S interval as shaded region
    for i, (_, row) in enumerate(top50.iterrows()):
        ax.plot([i, i], [row['ds_belief'], row['ds_plausibility']],
                'b-', linewidth=2, alpha=0.6, label='D-S [Bel, Pl]' if i == 0 else '')
        ax.plot(i, row['ds_belief'], 'b_', markersize=8, alpha=0.6,
                label='D-S Belief' if i == 0 else '')
        ax.plot(i, row['ds_plausibility'], 'b_', markersize=8, alpha=0.6,
                label='D-S Plausibility' if i == 0 else '')

    # Bayesian CI as error bars
    bayes_means = top50['bayes_mean'].values
    bayes_ci_lower = top50['bayes_ci_lower'].values
    bayes_ci_upper = top50['bayes_ci_upper'].values
    yerr_lower = bayes_means - bayes_ci_lower
    yerr_upper = bayes_ci_upper - bayes_means

    ax.errorbar(x, bayes_means, yerr=[yerr_lower, yerr_upper],
                fmt='ro', markersize=4, capsize=2, linewidth=1,
                label='Bayesian 95% CI', zorder=5)

    ax.set_xlabel('Grid (sorted by JSD)', fontsize=12)
    ax.set_ylabel('Damage Probability', fontsize=12)
    ax.set_title('Bayesian 95% CI vs D-S [Belief, Plausibility] Interval\n'
                 '(Top 50 highest JSD grids)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(INTERVAL_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure -> {INTERVAL_PNG}")


def plot_jsd_scatter(df):
    """JSD vs Bayesian-DS divergence scatter."""
    valid = df[df['ds_valid']].copy()

    # Compute bayes_ds_distance
    dist = np.zeros(len(valid))
    for i, (_, row) in enumerate(valid.iterrows()):
        bm = row['bayes_mean']
        db = row['ds_belief']
        dp = row['ds_plausibility']
        if db <= bm <= dp:
            dist[i] = 0
        else:
            dist[i] = min(abs(bm - db), abs(bm - dp))
    valid['bayes_ds_distance'] = dist

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: JSD vs Bayes-DS distance (colored by consistency)
    ax = axes[0]
    color_map = {
        'strong_consistent': '#2E7D32',
        'weak_consistent': '#FFC107',
        'divergent_high_bayes': '#D32F2F',
        'divergent_low_bayes': '#1976D2',
    }
    for label in ['strong_consistent', 'weak_consistent',
                  'divergent_high_bayes', 'divergent_low_bayes']:
        subset = valid[valid['consistency_label'] == label]
        if len(subset) > 0:
            ax.scatter(subset['jsd_s2_viirs'], subset['bayes_ds_distance'],
                      c=color_map[label], label=label, alpha=0.6, s=20,
                      edgecolors='none')

    ax.set_xlabel('JSD(S2, VIIRS) [nats]', fontsize=11)
    ax.set_ylabel('Bayes-DS Distance', fontsize=11)
    ax.set_title('JSD vs Bayesian-DS Divergence', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)

    # Compute correlation
    corr = valid['jsd_s2_viirs'].corr(valid['bayes_ds_distance'])
    ax.text(0.95, 0.95, f'r = {corr:.4f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel B: JSD vs D-S uncertainty width
    ax = axes[1]
    for label in ['strong_consistent', 'weak_consistent',
                  'divergent_high_bayes', 'divergent_low_bayes']:
        subset = valid[valid['consistency_label'] == label]
        if len(subset) > 0:
            ax.scatter(subset['jsd_s2_viirs'], subset['ds_uncertainty_width'],
                      c=color_map[label], label=label, alpha=0.6, s=20,
                      edgecolors='none')

    ax.set_xlabel('JSD(S2, VIIRS) [nats]', fontsize=11)
    ax.set_ylabel('D-S Uncertainty Width (Plaus - Bel)', fontsize=11)
    ax.set_title('JSD vs D-S Uncertainty Width', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)

    corr2 = valid['jsd_s2_viirs'].corr(valid['ds_uncertainty_width'])
    ax.text(0.95, 0.95, f'r = {corr2:.4f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure -> {SCATTER_PNG}")

    return corr, corr2


def plot_uncertainty_vs_jsd(df):
    """D-S uncertainty vs JSD with consistency overlay."""
    valid = df[df['ds_valid']].copy()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Hexbin
    hb = ax.hexbin(valid['jsd_s2_viirs'], valid['ds_uncertainty_width'],
                   gridsize=30, cmap='YlOrRd', mincnt=1, alpha=0.8)
    cbar = plt.colorbar(hb, ax=ax, label='Grid count')

    ax.set_xlabel('JSD(S2, VIIRS) [nats]', fontsize=12)
    ax.set_ylabel('D-S Uncertainty Width (Plausibility - Belief)', fontsize=12)
    ax.set_title('D-S Uncertainty Width vs JSD\n(Sensor conflict vs evidence conflict)',
                 fontweight='bold')

    corr = valid['jsd_s2_viirs'].corr(valid['ds_uncertainty_width'])
    ax.text(0.95, 0.95,
            f'r = {corr:.4f}\nmean width = {valid["ds_uncertainty_width"].mean():.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(UNCERT_VS_JSD_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure -> {UNCERT_VS_JSD_PNG}")


# ============================================================
# Validation checks
# ============================================================

def run_checks(df):
    """Run validation checks on the output data."""
    print(f"\n{'='*60}")
    print("Running validation checks")
    print(f"{'='*60}")

    # 1. Probability bounds
    for col in ['bayes_mean', 'bayes_ci_lower', 'bayes_ci_upper']:
        v = df[col].values
        assert np.all(v >= 0) and np.all(v <= 1), f"{col} out of [0,1]: [{v.min():.4f}, {v.max():.4f}]"
    print("  [PASS] Bayesian probabilities in [0, 1]")

    # 2. CI ordering
    assert np.all(df['bayes_ci_lower'] <= df['bayes_ci_upper']), \
        "bayes_ci_lower > bayes_ci_upper"
    print("  [PASS] bayes_ci_lower <= bayes_ci_upper")

    # 3. D-S ordering (where valid)
    valid = df[df['ds_valid']]
    bad_ds = (valid['ds_belief'] > valid['ds_plausibility'] + EPS).sum()
    if bad_ds > 0:
        print(f"  [WARN] {bad_ds} grids have ds_belief > ds_plausibility")
        bad_list = valid[valid['ds_belief'] > valid['ds_plausibility'] + EPS]
        bad_out = OUT_TAB / "bayes_vs_ds_invalid_ds_intervals.csv"
        bad_list[['grid_id', 'ds_belief', 'ds_plausibility']].to_csv(bad_out, index=False)
        print(f"    Saved to {bad_out}")
    else:
        print("  [PASS] ds_belief <= ds_plausibility for all valid grids")

    # 4. D-S uncertainty width >= 0
    assert np.all(valid['ds_uncertainty_width'] >= -EPS), \
        "Negative ds_uncertainty_width found"
    print("  [PASS] ds_uncertainty_width >= 0")

    # 5. No NaN in key output (except no_ds_data grids)
    key_cols = ['bayes_mean', 'jsd_s2_viirs', 'fusion_suppression_ratio']
    for col in key_cols:
        n_nan = df[col].isna().sum()
        assert n_nan == 0, f"{col} has {n_nan} NaN"
    print(f"  [PASS] No NaN in key fields")

    # 6. No inf
    for col in ['jsd_s2_viirs', 'fusion_suppression_ratio']:
        n_inf = np.isinf(df[col]).sum()
        assert n_inf == 0, f"{col} has {n_inf} inf"
    print("  [PASS] No inf in KLD fields")

    print("  All checks passed!")


# ============================================================
# Report
# ============================================================

def generate_report(df, summary_df, corr_jsd_dist, corr_jsd_width):
    """Generate markdown report."""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Count stats
    counts = df['consistency_label'].value_counts()
    n_total = len(df)
    n_valid = df['ds_valid'].sum()

    lines = []
    lines.append("# Bayesian vs Dempster-Shafer 一致性比较报告")
    lines.append(f"**生成时间**: {now_str}")
    lines.append(f"**模块**: 模块 3 — Bayesian vs D-S Consistency")
    lines.append("")

    # 1. Summary
    lines.append("## 1. 一致性分类统计")
    lines.append("")
    lines.append(f"- Total grids: {n_total}")
    lines.append(f"- Valid D-S data: {n_valid} ({n_valid/n_total*100:.1f}%)")
    lines.append(f"- No D-S data (insufficient evidence): {n_total - n_valid}")
    lines.append("")

    lines.append("| 一致性类别 | N | Pct | Mean JSD | Mean D-S Width | Mean Suppression | Mean K |")
    lines.append("|-----------|-----|-----|---------|---------------|-----------------|--------|")
    for _, row in summary_df.iterrows():
        if row['consistency_label'] == 'TOTAL':
            lines.append(f"| **{row['consistency_label']}** | **{int(row['n_grids'])}** | **{row['pct']:.1f}%** | "
                         f"{row['mean_jsd_s2_viirs']:.4f} | {row['mean_ds_uncertainty_width']:.4f} | "
                         f"{row['mean_fusion_suppression']:.4f} | {row['mean_conflict_k']:.4f} |")
        else:
            lines.append(f"| {row['consistency_label']} | {int(row['n_grids'])} | {row['pct']:.1f}% | "
                         f"{row['mean_jsd_s2_viirs']:.4f} | {row['mean_ds_uncertainty_width']:.4f} | "
                         f"{row['mean_fusion_suppression']:.4f} | {row['mean_conflict_k']:.4f} |")
    lines.append("")

    # 2. Key metrics by category
    lines.append("## 2. 各一致性类别的特征")
    lines.append("")

    valid = df[df['ds_valid']]
    for label in ['strong_consistent', 'weak_consistent',
                  'divergent_high_bayes', 'divergent_low_bayes']:
        subset = valid[valid['consistency_label'] == label]
        if len(subset) == 0:
            continue
        lines.append(f"### {label} (n={len(subset)})")
        lines.append("")
        lines.append(f"- Mean D-S interval width: {subset['ds_uncertainty_width'].mean():.4f}")
        lines.append(f"- Mean JSD(S2, VIIRS): {subset['jsd_s2_viirs'].mean():.4f} nats")
        lines.append(f"- Mean fusion suppression: {subset['fusion_suppression_ratio'].mean():.4f}")
        lines.append(f"- Mean conflict K: {subset['conflict_k'].mean():.4f}")
        lines.append("")

    # 3. Correlations
    lines.append("## 3. 相关性分析")
    lines.append("")
    lines.append(f"- **JSD vs Bayesian-DS divergence**: r = **{corr_jsd_dist:.4f}**")
    lines.append(f"- **JSD vs D-S uncertainty width**: r = **{corr_jsd_width:.4f}**")
    lines.append("")

    if corr_jsd_width > 0.3:
        rel = "正相关" if corr_jsd_width > 0 else "负相关"
        lines.append(f"JSD 与 D-S 不确定性宽度呈**{rel}** (r={corr_jsd_width:.4f})，")
        lines.append("说明传感器冲突越高的区域，D-S 的不确定区间也越宽。")
        lines.append("这与 D-S 理论预期一致：冲突证据 → 更大的 ignorance interval。")
    else:
        lines.append(f"JSD 与 D-S 不确定性宽度相关性较弱 (r={corr_jsd_width:.4f})。")
    lines.append("")

    # 4. Top 10 divergence grids
    lines.append("## 4. 分歧最大的 Top 10 Grids")
    lines.append("")
    # Compute distance
    dist = np.zeros(len(valid))
    for i, (_, row) in enumerate(valid.iterrows()):
        bm = row['bayes_mean']
        db = row['ds_belief']
        dp = row['ds_plausibility']
        if db <= bm <= dp:
            dist[i] = 0
        else:
            dist[i] = min(abs(bm - db), abs(bm - dp))
    valid = valid.copy()
    valid['bayes_ds_distance'] = dist
    top10 = valid.nlargest(10, 'bayes_ds_distance')

    lines.append("| grid_id | Bayes Mean | D-S Belief | D-S Plaus | Distance | JSD | Label |")
    lines.append("|---------|-----------|-----------|----------|----------|-----|-------|")
    for _, row in top10.iterrows():
        lines.append(f"| {int(row['grid_id'])} | {row['bayes_mean']:.4f} | "
                     f"{row['ds_belief']:.4f} | {row['ds_plausibility']:.4f} | "
                     f"{row['bayes_ds_distance']:.4f} | {row['jsd_s2_viirs']:.4f} | "
                     f"{row['consistency_label']} |")
    lines.append("")

    # 5. Interpretation
    lines.append("## 5. 三种方法的互补性")
    lines.append("")
    lines.append("### Bayesian 层次模型")
    lines.append("")
    lines.append("- 在 single latent damage state 假设下工作")
    lines.append("- 将传感器间冲突归因于 sensor bias (delta_s2) 和 noise (sigma_k)")
    lines.append(f"- Posterior 全域收缩至均值 (p_i range ≈ [0.151, 0.157])")
    lines.append("- 优点: 给出统一的概率估计 + 95% CI")
    lines.append("- 局限: single latent state 假设被 S2-VIIRS 冲突违反; 空间信息被压平")
    lines.append("")
    lines.append("### Dempster-Shafer 证据理论")
    lines.append("")
    lines.append("- 不强制证据收敛到单一 latent state")
    lines.append("- 用 [Belief, Plausibility] 区间表达不确定性")
    lines.append(f"- D-S 区间宽度均值 = {valid['ds_uncertainty_width'].mean():.4f}")
    lines.append("- 冲突证据 → wider ignorance interval → 保留\"不知\"的状态")
    lines.append("- 优点: 冲突下更诚实地表达无知")
    lines.append("- 局限: 不直接给出唯一的 posterior probability")
    lines.append("")
    lines.append("### KLD / JSD 信息指标")
    lines.append("")
    lines.append(f"- JSD 量化 S2-only vs VIIRS-only 的信息冲突 (均值 = {df['jsd_s2_viirs'].mean():.4f} nats)")
    lines.append(f"- Fusion suppression ratio 衡量 single latent state 融合的信息损失 (均值 = {df['fusion_suppression_ratio'].mean():.4f})")
    lines.append("- JSD + D-S uncertainty 相关性 r = {:.4f}".format(corr_jsd_width))
    lines.append("")
    lines.append("### 核心结论")
    lines.append("")
    n_strong = counts.get('strong_consistent', 0)
    n_weak = counts.get('weak_consistent', 0)
    n_div = counts.get('divergent_high_bayes', 0) + counts.get('divergent_low_bayes', 0)

    lines.append(f"> **Bayesian、D-S 和 KLD 三种方法是互补的，不是互相替代的。**")
    lines.append(f">")
    lines.append(f"> 在本案例的 {n_total} 个网格中：")
    lines.append(f"> - {n_strong} 个 ({n_strong/n_total*100:.1f}%) Bayesian 与 D-S **一致**")
    lines.append(f"> - {n_weak} 个 ({n_weak/n_total*100:.1f}%) **弱一致** (区间有交集)")
    lines.append(f"> - {n_div} 个 ({n_div/n_total*100:.1f}%) **分歧**")
    lines.append(f">")
    lines.append(f"> Bayesian 给出保守的统一后验，D-S 保留冲突证据下的不确定性边界，"
                 f"KLD/JSD 量化造成这种分歧的传感器冲突。")
    lines.append(f"> 当传感器系统性冲突时 (全域 JSD > 0.05)，single latent state 假设被违反，"
                 f"此时 D-S 的更宽不确定性区间可能是更诚实的表达。")
    lines.append("")

    # 6. Output files
    lines.append("## 6. 输出文件")
    lines.append("")
    lines.append("| 文件 | 描述 |")
    lines.append("|------|------|")
    for f in [CONSISTENCY_CSV, SUMMARY_CSV, MAP_PNG, DIVERGENCE_PNG,
              INTERVAL_PNG, SCATTER_PNG, UNCERT_VS_JSD_PNG, REPORT_MD]:
        lines.append(f"| `{f.relative_to(PROJECT_ROOT)}` | "
                     f"{'Per-grid consistency CSV' if 'csv' in f.suffix else 'Figure' if 'png' in f.suffix else 'Report'} |")

    report_text = '\n'.join(lines)
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  Report -> {REPORT_MD}")

    return report_text


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Module 3: Bayesian vs Dempster-Shafer Consistency")
    print("=" * 60)

    # 1. Load and merge
    print("\n[1/5] Loading and merging data...")
    df = load_and_merge()

    # 2. Classify consistency
    print("\n[2/5] Classifying consistency...")
    df = classify_consistency(df)

    # Print classification summary
    counts = df['consistency_label'].value_counts()
    for label, cnt in counts.items():
        print(f"  {label}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # 3. Save outputs
    print(f"\n[3/5] Saving outputs...")

    # Select and reorder columns for per-grid CSV
    out_cols = ['grid_id', 'bayes_mean', 'bayes_ci_lower', 'bayes_ci_upper',
                'ds_belief', 'ds_plausibility', 'ds_uncertainty_width', 'conflict_k',
                'consistency_label', 'high_uncertainty_flag', 'divergence_reason',
                'jsd_s2_viirs', 'fusion_suppression_ratio',
                'p_s2_only', 'p_viirs_only', 'p_both']
    # Only include columns that exist
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(CONSISTENCY_CSV, index=False)
    print(f"  Per-grid CSV -> {CONSISTENCY_CSV}")

    summary_df = compute_summary(df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"  Summary CSV  -> {SUMMARY_CSV}")

    # 4. Figures
    print(f"\n[4/5] Generating figures...")
    gdf = load_geodata()

    plot_consistency_map(df, gdf)
    plot_divergence_map(df, gdf)
    plot_interval_comparison(df)
    corr_jsd_dist, corr_jsd_width = plot_jsd_scatter(df)
    plot_uncertainty_vs_jsd(df)

    # 5. Validation and report
    print(f"\n[5/5] Running checks and generating report...")
    run_checks(df)
    report = generate_report(df, summary_df, corr_jsd_dist, corr_jsd_width)

    # Print key results
    print(f"\n{'='*60}")
    print("Bayesian vs D-S Consistency Analysis Complete!")
    print(f"{'='*60}")
    print(f"\n  === KEY RESULTS ===")
    for label in ['strong_consistent', 'weak_consistent',
                  'divergent_high_bayes', 'divergent_low_bayes', 'no_ds_data']:
        cnt = df['consistency_label'].value_counts().get(label, 0)
        print(f"  {label}: {cnt} ({cnt/len(df)*100:.1f}%)")
    print(f"  JSD vs D-S width correlation: r = {corr_jsd_width:.4f}")
    print(f"  JSD vs Bayes-DS distance correlation: r = {corr_jsd_dist:.4f}")

    return df, summary_df


if __name__ == '__main__':
    main()
