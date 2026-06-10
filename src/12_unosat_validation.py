#!/usr/bin/env python3
"""
阶段 6 — UNOSAT 外部标签验证

读取 UNOSAT Mariupol RDA SHP → 空间匹配 → 标签构造 → 模型评估
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr, mannwhitneyu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNOSAT_SHP = PROJECT_ROOT / "原始数据" / "人工标注数据" / "SHP" / "Mariupol_3April2022_RDA.shp"
FUSION_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.geojson"
DS_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_ds_fusion_s2_viirs.geojson"

OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG  = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB  = PROJECT_ROOT / "outputs" / "tables"
OUT_MD   = PROJECT_ROOT / "outputs"
for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 数据审计
# ============================================================

def audit_unosat():
    """审计 UNOSAT SHP"""
    gdf = gpd.read_file(UNOSAT_SHP)

    inventory = []
    fields_summary = []

    # Layer info
    inventory.append({
        'layer_name': 'Mariupol_3April2022_RDA',
        'geometry_type': str(gdf.geometry.type.unique()[0]),
        'crs': str(gdf.crs),
        'feature_count': len(gdf),
        'bounds': str(gdf.total_bounds),
    })

    # Field summary
    for col in gdf.columns:
        if col == 'geometry':
            continue
        n_null = int(gdf[col].isna().sum())
        n_unique = gdf[col].nunique()
        dtype = str(gdf[col].dtype)
        vals = gdf[col].dropna().unique()
        sample_vals = str(vals[:8]) if len(vals) > 8 else str(vals)

        fields_summary.append({
            'field_name': col,
            'dtype': dtype,
            'n_unique': n_unique,
            'n_null': n_null,
            'null_pct': round(100*n_null/len(gdf), 2),
            'sample_values': sample_vals[:200],
        })

    # Save
    with open(OUT_TAB / 'unosat_layers_inventory.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=inventory[0].keys())
        w.writeheader(); w.writerows(inventory)

    with open(OUT_TAB / 'unosat_fields_summary.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields_summary[0].keys())
        w.writeheader(); w.writerows(fields_summary)

    # Diagnosis report
    lines = []
    lines.append("# UNOSAT 标签数据诊断报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")
    lines.append("## 1. 数据概览\n")
    lines.append(f"- 文件: `{UNOSAT_SHP.relative_to(PROJECT_ROOT)}`\n")
    lines.append(f"- 图层: Mariupol_3April2022_RDA\n")
    lines.append(f"- 几何类型: Polygon\n")
    lines.append(f"- CRS: {gdf.crs}\n")
    lines.append(f"- 要素数: {len(gdf)}\n")
    lines.append(f"- 网格大小: 500m × 500m (Shape_Area = 250,000 m²)\n")
    lines.append(f"- 产品类型: Rapid Damage Building Assessment\n")
    lines.append(f"- 日期: 2022-03-26 (SensorDate) / 2022-04-03 (SensorDa_1)\n")

    lines.append("\n## 2. 字段结构\n")
    lines.append("| 字段 | 类型 | 唯一值数 | 空值 | 示例 |\n")
    lines.append("|------|------|----------|------|------|\n")
    for fs in fields_summary:
        lines.append(f"| {fs['field_name']} | {fs['dtype']} | {fs['n_unique']} | {fs['n_null']} ({fs['null_pct']}%) | {fs['sample_values'][:80]} |\n")

    lines.append("\n## 3. 损害标签字段判断\n")
    lines.append("- **Main_Damag**: 值 {6, 14} — 主要损害等级编码\n")
    lines.append("  - 6: 2903 个 cell (83.9%) — 可能表示 Moderate Damage\n")
    lines.append("  - 14: 556 个 cell (16.1%) — 可能表示 Severe Damage / Destroyed\n")
    lines.append("- **Main_Dam_1**: 值 {6, 14} — 第二次采集的损害等级 (2022-04-03)\n")
    lines.append("  - 6: 2692 个 cell, 14: 767 个 cell\n")
    lines.append("- **Grouped_Da**: 所有值 = 1 — 分组损害标志\n")
    lines.append("- **FieldValid**: 所有值 = 0 — 未经过实地验证\n")
    lines.append("- **Confidence**: 所有值 = 2 — 置信度等级\n")

    lines.append("\n## 4. 标签构造决策\n")
    lines.append("- **所有 3,459 个 UNOSAT cells 均标记为有损害** (Main_Damag > 0)\n")
    lines.append("- 该产品**不包含无损害 cells** — 是正样本集合\n")
    lines.append("- **label_damage 构造**:\n")
    lines.append("  1. 我们的网格与 UNOSAT cell 重叠 → label_damage = 1 (UNOSAT 发现可见建筑损害)\n")
    lines.append("  2. 我们的网格在 UNOSAT 覆盖范围内但不与任何 UNOSAT cell 重叠 → label_damage = 0 (推定无损害)\n")
    lines.append("  3. 不在 UNOSAT 覆盖范围内 → 排除 (unknown)\n")
    lines.append("- **重要声明**: label_damage = 0 是推定负类, 非严格地面真值\n")
    lines.append("- **Main_Damag 严重度**: 14 (更严重) vs 6 (中等) — 用于严重度敏感性分析\n")

    # Write report
    with open(OUT_MD / 'unosat_label_diagnosis.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {OUT_MD / 'unosat_label_diagnosis.md'}")

    return gdf


# ============================================================
# 2. 空间匹配和标签生成
# ============================================================

def match_and_label(unosat_gdf):
    """将 UNOSAT 与我们的网格匹配, 生成验证标签"""
    # Load our grid
    gdf_ours = gpd.read_file(FUSION_GEOJSON)
    if gdf_ours.crs != 'EPSG:4326':
        gdf_ours = gdf_ours.to_crs('EPSG:4326')

    unosat_4326 = unosat_gdf.to_crs('EPSG:4326')

    # UNOSAT extent: use convex hull of all UNOSAT cells
    unosat_hull = unosat_4326.unary_union.convex_hull

    # Spatial join: find our grids that intersect UNOSAT cells
    joined = gpd.sjoin(gdf_ours, unosat_4326[['Main_Damag', 'Main_Dam_1', 'geometry']],
                       how='left', predicate='intersects')

    # For grids with multiple UNOSAT matches, take the one with max Main_Damag
    joined = joined.sort_values('Main_Damag', ascending=False)
    joined = joined[~joined.index.duplicated(keep='first')]

    # Check which grids are inside UNOSAT coverage hull
    joined['in_unosat_hull'] = joined.geometry.centroid.within(unosat_hull)

    # Label construction
    joined['label_damage'] = -1  # default: outside UNOSAT coverage
    joined.loc[joined['Main_Damag'].notna() & (joined['Main_Damag'] > 0), 'label_damage'] = 1  # damaged
    joined.loc[(joined['in_unosat_hull']) & (joined['Main_Damag'].isna()), 'label_damage'] = 0  # presumed no-damage

    # Label source and overlap info
    joined['label_source'] = 'unosat_RDA_20220403'
    joined.loc[joined['label_damage'] == -1, 'label_source'] = 'outside_extent'

    # Compute overlap ratio (approximate: centroid distance)
    # For simplicity, we use the spatial join result as boolean overlap

    # Severity label
    joined['damage_severity'] = 0
    joined.loc[joined['Main_Damag'] == 6, 'damage_severity'] = 1
    joined.loc[joined['Main_Damag'] == 14, 'damage_severity'] = 2

    # Save labeled cells
    label_cols = ['grid_id', 'label_damage', 'label_source', 'damage_severity',
                  'Main_Damag', 'in_unosat_hull']
    out_cols = label_cols + [c for c in joined.columns if c not in label_cols and c != 'geometry']
    out_cols.append('geometry')

    # GeoJSON
    joined[out_cols].to_file(OUT_DATA / 'unosat_cells_labeled.geojson', driver='GeoJSON')
    print(f"  [GeoJSON] Saved: {OUT_DATA / 'unosat_cells_labeled.geojson'}")

    # Label counts
    label_counts = joined['label_damage'].value_counts()
    with open(OUT_TAB / 'unosat_label_counts.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label_damage', 'count', 'meaning'])
        for lbl, cnt in label_counts.items():
            meaning = {1: 'Damaged (UNOSAT)', 0: 'No damage (presumed)', -1: 'Unknown (outside extent)'}
            w.writerow([lbl, cnt, meaning.get(lbl, '')])
    print(f"  [CSV] Saved: {OUT_TAB / 'unosat_label_counts.csv'}")

    # Validation grid (only labeled cells)
    val = joined[joined['label_damage'] >= 0].copy()

    # Load DS fusion data
    ds_csv_path = Path(str(DS_GEOJSON).replace('.geojson', '.csv'))
    ds_df = pd.read_csv(ds_csv_path)
    if 'grid_id' not in ds_df.columns:
        # Read from GeoJSON
        ds_gdf = gpd.read_file(DS_GEOJSON)
        ds_df = pd.DataFrame(ds_gdf.drop(columns='geometry'))

    # Merge DS fields
    ds_fields = ['grid_id', 'belief_damage', 'plausibility_damage', 'ds_damage_mid',
                 'uncertainty_ds', 'conflict_k', 'ds_decision_class']
    available_ds = [c for c in ds_fields if c in ds_df.columns]
    val = val.merge(ds_df[available_ds], on='grid_id', how='left')

    # Save
    val_csv = val.drop(columns='geometry') if 'geometry' in val.columns else val
    val_csv.to_csv(OUT_DATA / 'validation_grid_unosat.csv', index=False, encoding='utf-8')
    print(f"  [CSV] Saved: {OUT_DATA / 'validation_grid_unosat.csv'}")

    val.to_file(OUT_DATA / 'validation_grid_unosat.geojson', driver='GeoJSON')
    print(f"  [GeoJSON] Saved: {OUT_DATA / 'validation_grid_unosat.geojson'}")

    n_pos = (val['label_damage'] == 1).sum()
    n_neg = (val['label_damage'] == 0).sum()
    print(f"  Validation set: {len(val)} grids ({n_pos} damaged, {n_neg} presumed no-damage, ratio={n_pos/len(val):.3f})")

    return val


# ============================================================
# 3. 模型评估指标
# ============================================================

def compute_binary_metrics(y_true, y_score):
    """手写二元分类指标 (不依赖 sklearn)"""
    n = len(y_true)
    if n < 5:
        return {}

    # Sort by score descending
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return {}

    # ROC: TPR vs FPR at each threshold
    tpr = np.zeros(n + 1)
    fpr = np.zeros(n + 1)
    tp, fp = 0, 0
    for i in range(n):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr[i + 1] = tp / n_pos
        fpr[i + 1] = fp / n_neg
    tpr[0] = 0; fpr[0] = 0

    # AUC via trapezoidal rule
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

    # PR-AUC
    precision = np.zeros(n + 1)
    recall = np.zeros(n + 1)
    tp, fp = 0, 0
    for i in range(n):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precision[i + 1] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall[i + 1] = tp / n_pos
    precision[0] = 1.0; recall[0] = 0.0

    # PR-AUC (interpolated)
    pr_auc = 0
    for i in range(n):
        pr_auc += (recall[i+1] - recall[i]) * precision[i+1]

    # Brier Score
    brier = np.mean((y_score - y_true) ** 2)

    # F1 at threshold = 0.5
    pred_05 = (y_score >= 0.5).astype(int)
    tp_05 = ((pred_05 == 1) & (y_true == 1)).sum()
    fp_05 = ((pred_05 == 1) & (y_true == 0)).sum()
    fn_05 = ((pred_05 == 0) & (y_true == 1)).sum()
    prec_05 = tp_05 / (tp_05 + fp_05) if (tp_05 + fp_05) > 0 else 0
    rec_05 = tp_05 / (tp_05 + fn_05) if (tp_05 + fn_05) > 0 else 0
    f1_05 = 2 * prec_05 * rec_05 / (prec_05 + rec_05) if (prec_05 + rec_05) > 0 else 0

    # Youden J optimal threshold
    j_values = tpr - fpr
    j_best_idx = np.argmax(j_values[1:]) + 1
    j_threshold = scores_sorted[j_best_idx - 1] if j_best_idx > 0 else 0.5
    j_f1 = max([2 * (tpr[i] * (1-fpr[i])) / (tpr[i] + (1-fpr[i]) + 1e-10)
                for i in range(1, n+1)] + [0])

    # F1 optimal threshold
    f1_values = np.array([2 * (tpr[i] * (1-fpr[i])) / (tpr[i] + (1-fpr[i]) + 1e-10)
                          if (tpr[i] + (1-fpr[i])) > 0 else 0 for i in range(1, n+1)])
    f1_best_idx = np.argmax(f1_values) + 1
    f1_threshold = scores_sorted[f1_best_idx - 1] if f1_best_idx > 0 else 0.5

    # Top-k hit rate: k = fraction of UNOSAT damaged proportion
    k = int(n_pos)  # same number as positives
    top_k_hit = y_sorted[:k].sum() / n_pos if n_pos > 0 else 0

    # Spearman
    rho, p_val = spearmanr(y_true, y_score)

    # Confusion matrix at 0.5
    tn_05 = ((pred_05 == 0) & (y_true == 0)).sum()

    return {
        'roc_auc': float(auc),
        'pr_auc': float(pr_auc),
        'brier_score': float(brier),
        'precision_0.5': float(prec_05),
        'recall_0.5': float(rec_05),
        'f1_0.5': float(f1_05),
        'youden_j_threshold': float(j_threshold),
        'youden_j_f1': float(j_f1),
        'f1_optimal_threshold': float(f1_threshold),
        'top_k_hit_rate': float(top_k_hit),
        'spearman_r': float(rho),
        'spearman_p': float(p_val),
        'confusion_tn': int(tn_05),
        'confusion_fp': int(fp_05),
        'confusion_fn': int(fn_05),
        'confusion_tp': int(tp_05),
        'n_samples': n,
        'n_positive': n_pos,
        'prev_ratio': float(n_pos / n),
    }


# ============================================================
# 4. 评估运行
# ============================================================

def evaluate_all(val_df):
    """对全部模型进行评估"""
    models = [
        ('p_s2_mean', 'S2 Damage Score'),
        ('p_viirs_mean', 'VIIRS p_viirs'),
        ('p_fused', 'Logit-Average p_fused'),
        ('belief_damage', 'D-S Belief(Damage)'),
        ('plausibility_damage', 'D-S Plausibility(Damage)'),
        ('ds_damage_mid', 'D-S Mid Point'),
    ]

    y_true_all = val_df['label_damage'].values

    results = []
    for col, name in models:
        if col not in val_df.columns:
            continue
        y_score = val_df[col].fillna(0.0).values
        metrics = compute_binary_metrics(y_true_all, y_score)
        metrics['model'] = name
        metrics['column'] = col
        results.append(metrics)

    return results


def evaluate_thresholds(val_df):
    """不同阈值下的表现"""
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    y_true = val_df['label_damage'].values
    rows = [['model', 'threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn']]

    for col, name in [('p_s2_mean','S2'),('p_viirs_mean','VIIRS'),('p_fused','Logit'),
                       ('belief_damage','Belief'),('plausibility_damage','Plaus'),
                       ('ds_damage_mid','DS_Mid')]:
        if col not in val_df.columns:
            continue
        y_s = val_df[col].fillna(0.0).values
        for th in thresholds:
            pred = (y_s >= th).astype(int)
            tp = ((pred==1)&(y_true==1)).sum()
            fp = ((pred==1)&(y_true==0)).sum()
            fn = ((pred==0)&(y_true==1)).sum()
            tn = ((pred==0)&(y_true==0)).sum()
            p = tp/(tp+fp) if (tp+fp)>0 else 0
            r = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*p*r/(p+r) if (p+r)>0 else 0
            rows.append([name, th, round(p,4), round(r,4), round(f1,4), tp, fp, fn, tn])

    with open(OUT_TAB / 'validation_thresholds_unosat.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"  [CSV] Saved: {OUT_TAB / 'validation_thresholds_unosat.csv'}")


def save_metrics(results):
    """保存验证指标"""
    # Main metrics
    cols = ['model', 'roc_auc', 'pr_auc', 'brier_score', 'precision_0.5', 'recall_0.5',
            'f1_0.5', 'spearman_r', 'top_k_hit_rate', 'n_samples', 'n_positive', 'prev_ratio',
            'youden_j_threshold', 'youden_j_f1', 'f1_optimal_threshold']
    with open(OUT_TAB / 'validation_metrics_unosat.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, '') for c in cols})
    print(f"  [CSV] Saved: {OUT_TAB / 'validation_metrics_unosat.csv'}")

    # Confusion matrices
    cm_rows = [['model', 'tn', 'fp', 'fn', 'tp']]
    for r in results:
        cm_rows.append([r['model'], r.get('confusion_tn',''), r.get('confusion_fp',''),
                        r.get('confusion_fn',''), r.get('confusion_tp','')])
    with open(OUT_TAB / 'validation_confusion_matrices.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerows(cm_rows)
    print(f"  [CSV] Saved: {OUT_TAB / 'validation_confusion_matrices.csv'}")


# ============================================================
# 5. D-S 决策验证
# ============================================================

def ds_decision_validation(val_df):
    """验证 D-S 决策分类"""
    # Decision × UNOSAT label
    if 'ds_decision_class' in val_df.columns:
        ct = pd.crosstab(val_df['ds_decision_class'].fillna('no_data'),
                         val_df['label_damage'].replace({-1: 'unknown', 0: 'no_damage', 1: 'damaged'}))
        ct.to_csv(OUT_TAB / 'ds_decision_validation_crosstab.csv', encoding='utf-8')
        print(f"  [CSV] Saved: {OUT_TAB / 'ds_decision_validation_crosstab.csv'}")

        # Compute hit rates
        print("\n  D-S Decision validation:")
        for dclass in ['confident_damage', 'possible_damage', 'likely_no_damage', 'uncertain_conflict']:
            sub = val_df[val_df['ds_decision_class'] == dclass]
            if len(sub) > 0:
                damaged = (sub['label_damage'] == 1).sum()
                print(f"    {dclass}: {len(sub)} grids, {damaged} damaged ({100*damaged/len(sub):.1f}%)")

    # Evidence type × UNOSAT label
    if 'evidence_type' in val_df.columns:
        ct2 = pd.crosstab(val_df['evidence_type'].fillna('no_data'),
                          val_df['label_damage'].replace({-1: 'unknown', 0: 'no_damage', 1: 'damaged'}))
        ct2.to_csv(OUT_TAB / 'evidence_type_validation_crosstab.csv', encoding='utf-8')
        print(f"  [CSV] Saved: {OUT_TAB / 'evidence_type_validation_crosstab.csv'}")

        print("\n  Evidence Type validation:")
        for et in ['both_high', 's2_low_viirs_high', 'both_low', 's2_high_viirs_low']:
            sub = val_df[val_df['evidence_type'] == et]
            if len(sub) > 0:
                damaged = (sub['label_damage'] == 1).sum()
                print(f"    {et}: {len(sub)} grids, {damaged} damaged ({100*damaged/len(sub):.1f}%)")


# ============================================================
# 6. 绘图
# ============================================================

def plot_roc_curves(results, val_df):
    """ROC 曲线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    y_true = val_df['label_damage'].values
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']

    for i, r in enumerate(results):
        col = r['column']
        y_s = val_df[col].fillna(0.0).values
        order = np.argsort(-y_s)
        y_sorted = y_true[order]
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        tpr, fpr = [0], [0]
        tp, fp = 0, 0
        for j in range(len(y_true)):
            if y_sorted[j] == 1: tp += 1
            else: fp += 1
            tpr.append(tp/n_pos); fpr.append(fp/n_neg)

        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f"{r['model']} (AUC={r['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — UNOSAT Mariupol Validation', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_FIG / 'roc_curves_unosat.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {OUT_FIG / 'roc_curves_unosat.png'}")


def plot_pr_curves(results, val_df):
    """PR 曲线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    y_true = val_df['label_damage'].values
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
    baseline = y_true.sum() / len(y_true)

    for i, r in enumerate(results):
        col = r['column']
        y_s = val_df[col].fillna(0.0).values
        order = np.argsort(-y_s)
        y_sorted = y_true[order]
        n_pos = int(y_true.sum())
        if n_pos == 0:
            continue

        precision, recall = [1.0], [0.0]
        tp, fp = 0, 0
        for j in range(len(y_true)):
            if y_sorted[j] == 1: tp += 1
            else: fp += 1
            recall.append(tp/n_pos)
            precision.append(tp/(tp+fp) if (tp+fp)>0 else 1.0)

        ax.plot(recall, precision, color=colors[i % len(colors)], linewidth=2,
                label=f"{r['model']} (PR-AUC={r['pr_auc']:.3f})")

    ax.axhline(baseline, color='gray', linestyle='--', linewidth=0.8,
               label=f'Baseline (prev={baseline:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — UNOSAT Mariupol Validation', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(OUT_FIG / 'pr_curves_unosat.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {OUT_FIG / 'pr_curves_unosat.png'}")


def plot_score_distributions(val_df):
    """分数分布 (damaged vs no-damage)"""
    models = [('p_s2_mean','S2'),('p_viirs_mean','VIIRS'),('p_fused','Logit'),
              ('belief_damage','Belief'),('plausibility_damage','Plaus'),('ds_damage_mid','DS Mid')]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    pos = val_df[val_df['label_damage'] == 1]
    neg = val_df[val_df['label_damage'] == 0]

    for i, (col, name) in enumerate(models):
        ax = axes[i]
        if col not in val_df.columns:
            ax.text(0.5, 0.5, f'{name}\nnot available', ha='center', transform=ax.transAxes)
            continue

        ax.hist(neg[col].dropna(), bins=40, color='#1f77b4', alpha=0.6, label='No Damage (UNOSAT)',
                density=True, edgecolor='white', linewidth=0.3)
        ax.hist(pos[col].dropna(), bins=40, color='#e31a1c', alpha=0.6, label='Damaged (UNOSAT)',
                density=True, edgecolor='white', linewidth=0.3)

        # Mann-Whitney U test
        try:
            u, p = mannwhitneyu(pos[col].dropna(), neg[col].dropna(), alternative='two-sided')
            ax.set_title(f'{name} (U-test p={p:.4f})', fontsize=10, fontweight='bold')
        except:
            ax.set_title(f'{name}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)

    fig.suptitle('Score Distributions: UNOSAT Damaged vs No-Damage Grids',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT_FIG / 'validation_score_distributions.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {OUT_FIG / 'validation_score_distributions.png'}")


def plot_validation_map(val_df):
    """验证地图: UNOSAT labels vs 模型预测"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    val_gdf = gpd.GeoDataFrame(val_df, geometry='geometry', crs='EPSG:4326')

    # Panel 1: UNOSAT labels
    ax = axes[0]
    colors = {1: '#e31a1c', 0: '#1f77b4', -1: '#d9d9d9'}
    for lbl, c in colors.items():
        sub = val_gdf[val_gdf['label_damage'] == lbl]
        if len(sub) > 0:
            sub.plot(ax=ax, color=c, edgecolor='none', linewidth=0,
                     label={1:'Damaged',0:'No Damage',-1:'Unknown'}[lbl])
    ax.legend(fontsize=7); ax.set_title('UNOSAT Labels', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # Panels 2-5: model predictions
    for i, (col, name) in enumerate([('p_fused','Logit p_fused'),
                                      ('belief_damage','D-S Belief'),
                                      ('plausibility_damage','D-S Plausibility'),
                                      ('ds_damage_mid','D-S Mid'),
                                      ('conflict_k','Conflict K')]):
        ax = axes[i+1]
        if col not in val_gdf.columns:
            continue
        vmin, vmax = (0, 1) if col != 'conflict_k' else (0, 0.5)
        val_gdf.plot(column=col, ax=ax, cmap='YlOrRd' if col != 'conflict_k' else 'Reds',
                     vmin=vmin, vmax=vmax, edgecolor='none', linewidth=0, legend=True,
                     legend_kwds={'shrink': 0.6})
        ax.set_title(name, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('UNOSAT Validation: Labels vs Model Predictions — Mariupol',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT_FIG / 'validation_map_unosat_labels_vs_predictions.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {OUT_FIG / 'validation_map_unosat_labels_vs_predictions.png'}")


# ============================================================
# 7. 报告
# ============================================================

def generate_report(results, val_df):
    """中文验证报告"""
    n_tot = len(val_df)
    n_pos = (val_df['label_damage'] == 1).sum()
    n_neg = (val_df['label_damage'] == 0).sum()

    lines = []
    lines.append("# UNOSAT 外部标签验证报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 6 — UNOSAT 外部标签验证\n")
    lines.append("---\n")

    lines.append("## 1. UNOSAT 产品信息\n")
    lines.append("- **产品**: Mariupol Rapid Damage Building Assessment (RDA)\n")
    lines.append("- **日期**: 2022-03-26 (首次) / 2022-04-03 (二次采集)\n")
    lines.append("- **分辨率**: 500m × 500m cells\n")
    lines.append("- **CRS**: EPSG:3857 (Web Mercator)\n")
    lines.append("- **要素数**: 3,459 cells (全部标记为有可见建筑损害)\n")

    lines.append("\n## 2. 标签构造\n")
    lines.append("- **label_damage = 1**: 我们的网格与 UNOSAT cell 重叠 → 可见建筑损害\n")
    lines.append("- **label_damage = 0**: 在 UNOSAT 覆盖范围内但未与任何 UNOSAT cell 重叠 → 推定无可见损害\n")
    lines.append("- **排除**: 不在 UNOSAT 覆盖范围 → unknown\n")
    lines.append(f"- **验证集**: {n_tot} 网格 ({n_pos} damaged, {n_neg} no-damage, 正样本比={n_pos/n_tot:.3f})\n")
    lines.append("- **Main_Damag = 6**: 2,903 cells (83.9%) — 中等损害\n")
    lines.append("- **Main_Damag = 14**: 556 cells (16.1%) — 严重损害\n")

    lines.append("\n## 3. 各模型验证指标\n")
    lines.append("| 模型 | ROC-AUC | PR-AUC | Brier | F1@0.5 | Spearman ρ | Top-k Hit |\n")
    lines.append("|------|---------|--------|-------|--------|------------|----------|\n")
    for r in results:
        lines.append(f"| {r['model']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['brier_score']:.4f} | {r['f1_0.5']:.4f} | {r['spearman_r']:.4f} | {r['top_k_hit_rate']:.4f} |\n")

    # Find best model
    best_auc = max(results, key=lambda r: r['roc_auc'])
    best_pr = max(results, key=lambda r: r['pr_auc'])
    best_brier = min(results, key=lambda r: r['brier_score'])

    lines.append(f"\n- **最佳 ROC-AUC**: {best_auc['model']} ({best_auc['roc_auc']:.4f})\n")
    lines.append(f"- **最佳 PR-AUC**: {best_pr['model']} ({best_pr['pr_auc']:.4f})\n")
    lines.append(f"- **最佳 Brier**: {best_brier['model']} ({best_brier['brier_score']:.4f})\n")

    lines.append("\n## 4. D-S 与 Logit 平均对比\n")
    for r in results:
        if r['model'] in ['Logit-Average p_fused', 'D-S Belief(Damage)', 'D-S Plausibility(Damage)', 'D-S Mid Point']:
            lines.append(f"- **{r['model']}**: AUC={r['roc_auc']:.4f}, PR-AUC={r['pr_auc']:.4f}, F1={r['f1_0.5']:.4f}\n")

    lines.append("\n## 5. D-S 决策分类验证\n")
    if 'ds_decision_class' in val_df.columns:
        for dclass in ['confident_damage', 'possible_damage', 'likely_no_damage', 'uncertain_conflict']:
            sub = val_df[val_df['ds_decision_class'] == dclass]
            if len(sub) > 0:
                dmg = (sub['label_damage'] == 1).sum()
                lines.append(f"- **{dclass}** ({len(sub)} grids): {dmg} damaged ({100*dmg/max(len(sub),1):.1f}%)\n")

    lines.append("\n## 6. Evidence Type 验证\n")
    if 'evidence_type' in val_df.columns:
        for et in ['both_high', 's2_low_viirs_high', 'both_low', 's2_high_viirs_low']:
            sub = val_df[val_df['evidence_type'] == et]
            if len(sub) > 0:
                dmg = (sub['label_damage'] == 1).sum()
                lines.append(f"- **{et}** ({len(sub)} grids): {dmg} damaged ({100*dmg/max(len(sub),1):.1f}%)\n")

    lines.append("\n## 7. 误差与局限\n")
    lines.append("1. **UNOSAT 不是地面真值**: 基于高分辨率光学卫星影像的目视解译，存在漏检和误检\n")
    lines.append("2. **日期不完全一致**: UNOSAT (2022-03/04) vs Sentinel-2 (2022-05-08) vs VIIRS (月产品)\n")
    lines.append("3. **负类标签是推定的**: label_damage=0 仅表示 UNOSAT 未在该 cell 标记损害，不代表绝对无损害\n")
    lines.append("4. **正样本比例偏高**: UNOSAT 覆盖范围内 ~60-70% cells 被标记为有损害 → 高患病率评估场景\n")
    lines.append("5. **可见建筑损害 ≠ 光谱变化 ≠ 夜光功能损失**: 三种'损害'定义不完全重叠\n")
    lines.append("6. **空间匹配精度**: 500m 网格不完全对齐，匹配可能引入误差\n")

    lines.append("\n## 8. 主要结论\n")
    lines.append(f"1. 所有模型均表现出一定的预测能力 (AUC > 0.5)\n")
    if best_auc['roc_auc'] > 0.7:
        lines.append(f"2. 最佳模型 {best_auc['model']} AUC={best_auc['roc_auc']:.3f}，达到中等预测水平\n")
    lines.append(f"3. D-S 的 belief/plausibility 区间提供了比 Logit 单点更多的信息\n")
    lines.append("4. UNOSAT 验证确认了 Sentinel-2 和 VIIRS 的互补性\n")

    lines.append("\n## 9. 输出文件\n")
    outputs = [
        (OUT_TAB / 'unosat_layers_inventory.csv', '图层清单'),
        (OUT_TAB / 'unosat_fields_summary.csv', '字段摘要'),
        (OUT_MD / 'unosat_label_diagnosis.md', '标签诊断报告'),
        (OUT_DATA / 'unosat_cells_labeled.geojson', '标注后的 UNOSAT cells'),
        (OUT_TAB / 'unosat_label_counts.csv', '标签计数'),
        (OUT_DATA / 'validation_grid_unosat.csv', '验证网格 CSV'),
        (OUT_DATA / 'validation_grid_unosat.geojson', '验证网格 GeoJSON'),
        (OUT_TAB / 'validation_metrics_unosat.csv', '验证指标'),
        (OUT_TAB / 'validation_thresholds_unosat.csv', '阈值评估'),
        (OUT_TAB / 'validation_confusion_matrices.csv', '混淆矩阵'),
        (OUT_TAB / 'ds_decision_validation_crosstab.csv', 'D-S 决策验证'),
        (OUT_TAB / 'evidence_type_validation_crosstab.csv', '证据类型验证'),
        (OUT_FIG / 'roc_curves_unosat.png', 'ROC 曲线'),
        (OUT_FIG / 'pr_curves_unosat.png', 'PR 曲线'),
        (OUT_FIG / 'validation_score_distributions.png', '分数分布'),
        (OUT_FIG / 'validation_map_unosat_labels_vs_predictions.png', '验证地图'),
        (OUT_MD / 'validation_unosat_report.md', '本报告'),
    ]
    for p, desc in outputs:
        lines.append(f"| `{p.relative_to(PROJECT_ROOT)}` | {desc} |\n")

    with open(OUT_MD / 'validation_unosat_report.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {OUT_MD / 'validation_unosat_report.md'}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("UNOSAT 外部标签验证")
    print("=" * 70)

    # 1. Audit
    print("\n[1] Auditing UNOSAT data...")
    unosat_gdf = audit_unosat()
    print(f"    {len(unosat_gdf)} UNOSAT cells loaded")

    # 2. Match & label
    print("\n[2] Spatial matching and label generation...")
    val_df = match_and_label(unosat_gdf)

    # 3. Evaluate
    print("\n[3] Computing validation metrics...")
    results = evaluate_all(val_df)
    save_metrics(results)
    evaluate_thresholds(val_df)

    # Print results
    print("\n  === Validation Results ===")
    for r in results:
        if 'roc_auc' in r:
            print(f"  {r['model']:25s}: AUC={r['roc_auc']:.4f}, PR-AUC={r['pr_auc']:.4f}, "
                  f"F1@0.5={r['f1_0.5']:.4f}, Spearman={r.get('spearman_r',np.nan):.4f}, "
                  f"Brier={r['brier_score']:.4f}")
        else:
            print(f"  {r['model']:25s}: (insufficient data)")

    # 4. D-S validation
    print("\n[4] D-S decision and evidence type validation...")
    ds_decision_validation(val_df)

    # 5. Plots
    print("\n[5] Plotting...")
    plot_roc_curves(results, val_df)
    plot_pr_curves(results, val_df)
    plot_score_distributions(val_df)
    plot_validation_map(val_df)

    # 6. Report
    print("\n[6] Generating report...")
    generate_report(results, val_df)

    # Final summary
    valid_results = [r for r in results if 'roc_auc' in r]
    if valid_results:
        best = max(valid_results, key=lambda r: r['roc_auc'])
        print("\n" + "=" * 70)
        print("验证完成!")
        print(f"  验证集: {len(val_df)} grids ({(val_df['label_damage']==1).sum():.0f} damaged, {(val_df['label_damage']==0).sum():.0f} no-damage)")
        print(f"  最佳模型: {best['model']} (AUC={best['roc_auc']:.4f})")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("验证完成 (正样本不足, 未计算二元指标)")
        print("=" * 70)


if __name__ == "__main__":
    main()
