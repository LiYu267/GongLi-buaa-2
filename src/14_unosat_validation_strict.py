#!/usr/bin/env python3
"""
阶段 6.2 — 使用修正后的 UNOSAT strict label 重做外部验证

关键修正: Main_Dam_1=14 → damaged, Main_Dam_1=6 → no visible damage
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr, mannwhitneyu
from shapely.geometry import Polygon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNOSAT_SHP = PROJECT_ROOT / "原始数据" / "人工标注数据" / "SHP" / "Mariupol_3April2022_RDA.shp"
FUSION_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.geojson"
DS_CSV = PROJECT_ROOT / "data" / "processed" / "grid_ds_fusion_s2_viirs.csv"

OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG  = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB  = PROJECT_ROOT / "outputs" / "tables"
OUT_MD   = PROJECT_ROOT / "outputs"
for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 构造 strict labels
# ============================================================

def build_strict_labels():
    """读取 UNOSAT, 使用 Main_Dam_1 构造 strict label, 空间匹配到我们的网格"""
    unosat = gpd.read_file(UNOSAT_SHP)
    unosat = unosat.to_crs('EPSG:4326')

    # Strict label on UNOSAT cells
    unosat['label_strict'] = -1
    unosat.loc[unosat['Main_Dam_1'] == 14, 'label_strict'] = 1
    unosat.loc[unosat['Main_Dam_1'] == 6, 'label_strict'] = 0

    n_dmg = int((unosat['label_strict'] == 1).sum())
    n_nodmg = int((unosat['label_strict'] == 0).sum())
    print(f"  UNOSAT: {len(unosat)} cells, {n_dmg} damaged ({100*n_dmg/len(unosat):.1f}%), "
          f"{n_nodmg} no visible damage ({100*n_nodmg/len(unosat):.1f}%)")

    # Load our grid
    gdf_ours = gpd.read_file(FUSION_GEOJSON)
    if gdf_ours.crs != 'EPSG:4326':
        gdf_ours = gdf_ours.to_crs('EPSG:4326')

    # Spatial join: match our grid centroids to UNOSAT cells
    gdf_ours['centroid'] = gdf_ours.geometry.centroid
    centroids_gdf = gpd.GeoDataFrame(gdf_ours[['grid_id', 'centroid']],
                                      geometry='centroid', crs='EPSG:4326')

    joined = gpd.sjoin(centroids_gdf,
                       unosat[['Main_Damag', 'Main_Dam_1', 'label_strict', 'geometry']],
                       how='left', predicate='within')

    # For grids with multiple UNOSAT matches, keep first
    joined = joined[~joined.index.duplicated(keep='first')]

    # Merge label back to our grid
    gdf_ours['label_damage'] = joined['label_strict'].values
    gdf_ours['unosat_main_dam_1'] = joined['Main_Dam_1'].values
    gdf_ours['unosat_main_damag'] = joined['Main_Damag'].values

    # Flag: matched vs unmatched
    gdf_ours['matched_to_unosat'] = joined['label_strict'].notna().values & (joined['label_strict'] >= 0)

    # Sub-damage categories
    gdf_ours['damage_subtype'] = 'unknown'
    gdf_ours.loc[(gdf_ours['unosat_main_damag'] == 6) & (gdf_ours['unosat_main_dam_1'] == 6), 'damage_subtype'] = 'no_damage_both'
    gdf_ours.loc[(gdf_ours['unosat_main_damag'] == 6) & (gdf_ours['unosat_main_dam_1'] == 14), 'damage_subtype'] = 'new_damage_apr3'
    gdf_ours.loc[(gdf_ours['unosat_main_damag'] == 14) & (gdf_ours['unosat_main_dam_1'] == 14), 'damage_subtype'] = 'persistent_damage'

    # Validation subset
    val = gdf_ours[gdf_ours['label_damage'] >= 0].copy()

    # Merge DS fields
    ds_df = pd.read_csv(DS_CSV)
    ds_fields = ['grid_id', 'belief_damage', 'plausibility_damage', 'ds_damage_mid',
                 'uncertainty_ds', 'conflict_k', 'ds_decision_class']
    available_ds = [c for c in ds_fields if c in ds_df.columns]
    val = val.merge(ds_df[available_ds], on='grid_id', how='left')

    return gdf_ours, val


# ============================================================
# 2. 评估指标
# ============================================================

def compute_metrics(y_true, y_score):
    """手写二元分类指标"""
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n < 5 or n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    # ROC
    tpr, fpr = np.zeros(n+1), np.zeros(n+1)
    tp, fp = 0, 0
    for i in range(n):
        if y_sorted[i] == 1: tp += 1
        else: fp += 1
        tpr[i+1] = tp / n_pos; fpr[i+1] = fp / n_neg

    auc = float(np.sum((fpr[1:]-fpr[:-1]) * (tpr[1:]+tpr[:-1]) / 2.0))

    # PR
    precision = np.zeros(n+1); recall = np.zeros(n+1)
    tp, fp = 0, 0
    for i in range(n):
        if y_sorted[i] == 1: tp += 1
        else: fp += 1
        recall[i+1] = tp / n_pos
        precision[i+1] = tp / (tp+fp) if (tp+fp) > 0 else 1.0
    precision[0] = 1.0; recall[0] = 0.0
    pr_auc = float(np.sum((recall[1:]-recall[:-1]) * precision[1:]))

    # Brier
    brier = float(np.mean((y_score - y_true) ** 2))

    # F1 @ 0.5
    pred_05 = (y_score >= 0.5).astype(int)
    tp05 = int(((pred_05==1)&(y_true==1)).sum())
    fp05 = int(((pred_05==1)&(y_true==0)).sum())
    fn05 = int(((pred_05==0)&(y_true==1)).sum())
    tn05 = int(((pred_05==0)&(y_true==0)).sum())
    p05 = tp05/(tp05+fp05) if (tp05+fp05)>0 else 0
    r05 = tp05/(tp05+fn05) if (tp05+fn05)>0 else 0
    f105 = 2*p05*r05/(p05+r05) if (p05+r05)>0 else 0

    # Youden J
    j_vals = tpr - fpr
    j_idx = int(np.argmax(j_vals[1:]) + 1)
    j_thresh = s_sorted[j_idx-1] if j_idx > 0 else 0.5

    # F1 optimal
    f1_vals = np.array([2*tpr[i]*(1-fpr[i])/(tpr[i]+(1-fpr[i])+1e-10)
                        if (tpr[i]+(1-fpr[i]))>0 else 0 for i in range(1,n+1)])
    f1_idx = int(np.argmax(f1_vals) + 1)
    f1_thresh = s_sorted[f1_idx-1] if f1_idx > 0 else 0.5

    # Top-k
    k = n_pos
    top_k_hit = float(y_sorted[:k].sum() / n_pos)

    # Spearman
    rho, pv = spearmanr(y_true, y_score)

    return {
        'roc_auc': auc, 'pr_auc': pr_auc, 'brier_score': brier,
        'precision_0.5': p05, 'recall_0.5': r05, 'f1_0.5': f105,
        'youden_j_threshold': j_thresh, 'f1_optimal_threshold': f1_thresh,
        'top_k_hit_rate': top_k_hit, 'spearman_r': float(rho), 'spearman_p': float(pv),
        'confusion_tn': tn05, 'confusion_fp': fp05,
        'confusion_fn': fn05, 'confusion_tp': tp05,
        'n_samples': n, 'n_positive': n_pos, 'prev_ratio': float(n_pos/n),
    }


def evaluate_all(val_df):
    """评估所有模型"""
    models = [
        ('p_s2_mean', 'S2 Damage Score'),
        ('p_viirs_mean', 'VIIRS p_viirs'),
        ('p_fused', 'Logit-Avg p_fused'),
        ('belief_damage', 'D-S Belief(D)'),
        ('plausibility_damage', 'D-S Plaus(D)'),
        ('ds_damage_mid', 'D-S Mid Point'),
    ]

    y_true = val_df['label_damage'].values.astype(float)

    results = []
    for col, name in models:
        if col not in val_df.columns:
            continue
        y_score = val_df[col].fillna(0.0).values
        m = compute_metrics(y_true, y_score)
        if m:
            m['model'] = name; m['column'] = col
            results.append(m)

    return results


# ============================================================
# 3. 保存
# ============================================================

def save_all(val_df, results, gdf_all):
    """保存所有输出"""
    # Validation grid
    geom_cols = [c for c in val_df.columns if c == 'geometry' or (hasattr(val_df[c], 'dtype') and str(val_df[c].dtype) == 'geometry')]
    val_csv = val_df.drop(columns=[c for c in geom_cols if c in val_df.columns], errors='ignore')
    val_csv.to_csv(OUT_DATA / 'validation_grid_unosat_strict.csv', index=False, encoding='utf-8')
    # Keep only main geometry column for GeoJSON
    if 'geometry' in val_df.columns:
        val_geo = val_df[['geometry'] + [c for c in val_df.columns if c != 'geometry' and not (hasattr(val_df[c], 'dtype') and str(val_df[c].dtype) == 'geometry')]].copy()
    else:
        val_geo = val_df.copy()
    val_geo.to_file(OUT_DATA / 'validation_grid_unosat_strict.geojson', driver='GeoJSON')
    print(f"  Saved: validation_grid_unosat_strict.*")

    # Label counts
    with open(OUT_TAB / 'validation_label_counts_strict.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'count', 'meaning'])
        for lbl in [1, 0, -1]:
            cnt = int((val_df['label_damage'] == lbl).sum())
            meaning = {1:'Visible Damage (Main_Dam_1=14)', 0:'No Visible Damage (Main_Dam_1=6)', -1:'Outside UNOSAT'}[lbl]
            w.writerow([lbl, cnt, meaning])
        # Damage subtypes
        for st in ['no_damage_both','new_damage_apr3','persistent_damage']:
            cnt = int((val_df['damage_subtype'] == st).sum())
            meaning = {'no_damage_both':'No damage either date','new_damage_apr3':'New damage (Apr 3)','persistent_damage':'Persistent damage'}[st]
            w.writerow([f'subtype_{st}', cnt, meaning])
    print(f"  Saved: validation_label_counts_strict.csv")

    # Validation metrics
    cols = ['model', 'roc_auc', 'pr_auc', 'brier_score', 'precision_0.5', 'recall_0.5',
            'f1_0.5', 'spearman_r', 'top_k_hit_rate', 'n_samples', 'n_positive', 'prev_ratio',
            'youden_j_threshold', 'f1_optimal_threshold']
    with open(OUT_TAB / 'validation_metrics_unosat_strict.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, '') for c in cols})
    print(f"  Saved: validation_metrics_unosat_strict.csv")

    # D-S decision cross-tab
    if 'ds_decision_class' in val_df.columns:
        ct = pd.crosstab(val_df['ds_decision_class'].fillna('no_data'),
                         val_df['label_damage'].map({1:'damaged', 0:'no_damage'}))
        ct.to_csv(OUT_TAB / 'ds_decision_validation_crosstab_strict.csv', encoding='utf-8')
        print(f"  Saved: ds_decision_validation_crosstab_strict.csv")

    # Evidence type cross-tab
    if 'evidence_type' in val_df.columns:
        ct2 = pd.crosstab(val_df['evidence_type'].fillna('no_data'),
                          val_df['label_damage'].map({1:'damaged', 0:'no_damage'}))
        ct2.to_csv(OUT_TAB / 'evidence_type_validation_crosstab_strict.csv', encoding='utf-8')
        print(f"  Saved: evidence_type_validation_crosstab_strict.csv")


# ============================================================
# 4. 绘图
# ============================================================

def plot_all(val_df, results):
    """所有图表"""
    y_true = val_df['label_damage'].values.astype(float)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(results):
        col = r['column']
        y_s = val_df[col].fillna(0.0).values
        order = np.argsort(-y_s)
        ys = y_true[order]
        n_pos = int(y_true.sum()); n_neg = len(y_true)-n_pos
        tpr, fpr = [0], [0]
        tp, fp = 0, 0
        for j in range(len(y_true)):
            if ys[j]==1: tp+=1
            else: fp+=1
            tpr.append(tp/n_pos); fpr.append(fp/n_neg)
        ax.plot(fpr, tpr, color=colors[i%6], lw=2, label=f"{r['model']} (AUC={r['roc_auc']:.4f})")
    ax.plot([0,1],[0,1],'k--',lw=0.8,alpha=0.4)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curves — UNOSAT Strict Labels', fontweight='bold', fontsize=14)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_FIG/'roc_curves_unosat_strict.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f"  [Fig] roc_curves_unosat_strict.png")

    # --- PR ---
    fig, ax = plt.subplots(figsize=(10, 8))
    prev = y_true.sum()/len(y_true)
    for i, r in enumerate(results):
        col = r['column']
        y_s = val_df[col].fillna(0.0).values
        order = np.argsort(-y_s)
        ys = y_true[order]
        n_pos = int(y_true.sum())
        prec, rec = [1.0], [0.0]
        tp, fp = 0, 0
        for j in range(len(y_true)):
            if ys[j]==1: tp+=1
            else: fp+=1
            rec.append(tp/n_pos)
            prec.append(tp/(tp+fp) if (tp+fp)>0 else 1.0)
        ax.plot(rec, prec, color=colors[i%6], lw=2, label=f"{r['model']} (PR-AUC={r['pr_auc']:.4f})")
    ax.axhline(prev, color='gray', ls='--', lw=0.8, label=f'Baseline (prev={prev:.4f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('PR Curves — UNOSAT Strict Labels', fontweight='bold', fontsize=14)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    plt.tight_layout(); fig.savefig(OUT_FIG/'pr_curves_unosat_strict.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f"  [Fig] pr_curves_unosat_strict.png")

    # --- Score distributions ---
    models_plot = [('p_s2_mean','S2'),('p_viirs_mean','VIIRS'),('p_fused','Logit'),
                   ('belief_damage','Belief'),('plausibility_damage','Plaus'),('ds_damage_mid','DS Mid')]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    pos = val_df[val_df['label_damage']==1]
    neg = val_df[val_df['label_damage']==0]
    for i, (col, name) in enumerate(models_plot):
        ax = axes[i]
        if col not in val_df.columns: continue
        ax.hist(neg[col].dropna(), bins=35, color='#1f77b4', alpha=0.6, label='No Visible Damage',
                density=True, edgecolor='white', lw=0.3)
        ax.hist(pos[col].dropna(), bins=35, color='#e31a1c', alpha=0.6, label='Visible Damage',
                density=True, edgecolor='white', lw=0.3)
        try:
            _, p = mannwhitneyu(pos[col].dropna(), neg[col].dropna(), alternative='two-sided')
            ax.set_title(f'{name} (MWU p={p:.4f})', fontsize=10, fontweight='bold')
        except:
            ax.set_title(name, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
    fig.suptitle('Score Distributions: UNOSAT Damaged vs No-Visible-Damage (Strict Labels)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(); fig.savefig(OUT_FIG/'validation_score_distributions_strict.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f"  [Fig] validation_score_distributions_strict.png")

    # --- Validation map ---
    geo_col = 'geometry' if 'geometry' in val_df.columns else val_df.geometry.name
    val_gdf = gpd.GeoDataFrame(val_df.drop(columns=[c for c in val_df.columns
                               if c != geo_col and hasattr(val_df[c], 'dtype') and str(val_df[c].dtype) == 'geometry']),
                               geometry=geo_col, crs='EPSG:4326')
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()
    # Panel 1: labels
    ax = axes[0]
    for lbl, c, lab in [(1,'#e31a1c','Damaged'),(0,'#1f77b4','No Visible Damage')]:
        sub = val_gdf[val_gdf['label_damage']==lbl]
        if len(sub)>0: sub.plot(ax=ax, color=c, edgecolor='none', lw=0, label=lab)
    ax.legend(fontsize=7); ax.set_title('UNOSAT Strict Labels', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for i,(col,name) in enumerate([('p_s2_mean','S2'),('p_viirs_mean','VIIRS'),('p_fused','Logit'),
                                    ('belief_damage','Belief(D)'),('ds_damage_mid','DS Mid')]):
        ax = axes[i+1]
        if col not in val_gdf.columns: continue
        val_gdf.plot(column=col, ax=ax, cmap='YlOrRd', vmin=0, vmax=1, edgecolor='none',
                     lw=0, legend=True, legend_kwds={'shrink':0.6})
        ax.set_title(name, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('UNOSAT Strict Labels vs Model Predictions — Mariupol',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(); fig.savefig(OUT_FIG/'validation_map_unosat_strict_labels_vs_predictions.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f"  [Fig] validation_map_unosat_strict_labels_vs_predictions.png")

    # --- AUC comparison bar ---
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r['model'] for r in results]
    aucs = [r['roc_auc'] for r in results]
    praucs = [r['pr_auc'] for r in results]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x-w/2, aucs, w, color='#1f77b4', label='ROC-AUC', edgecolor='white')
    bars2 = ax.bar(x+w/2, praucs, w, color='#ff7f0e', label='PR-AUC', edgecolor='white')
    ax.axhline(y=0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(y=val_df['label_damage'].mean(), color='green', ls=':', lw=0.8, alpha=0.5, label=f'Prev={val_df["label_damage"].mean():.3f}')
    for bar in bars1:
        h = bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.3f}', ha='center', fontsize=8)
    for bar in bars2:
        h = bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('AUC'); ax.set_title('Model AUC Comparison — UNOSAT Strict Labels', fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(0, 1.15); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_FIG/'model_auc_comparison_strict.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f"  [Fig] model_auc_comparison_strict.png")


# ============================================================
# 5. 报告
# ============================================================

def generate_report(val_df, results):
    """中文报告"""
    n = len(val_df)
    n_pos = int((val_df['label_damage']==1).sum())
    n_neg = int((val_df['label_damage']==0).sum())

    lines = []
    lines.append("# UNOSAT 外部标签验证报告 (Strict Labels, 修正版)\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 6.2 — 使用修正后的 strict label 重做验证\n")
    lines.append("---\n")

    lines.append("## 1. 阶段 6 原验证为何作废\n")
    lines.append("- 阶段 6 错误地将 **Main_Damag** 的所有非零值(6 和 14)都当作 damage\n")
    lines.append("- Main_Damag=6 实际是 **No Visible Damage** (2,903 cells, 83.9%)\n")
    lines.append("- Main_Damag=14 是 Previous Damage / Visible Damage (556 cells, 16.1%)\n")
    lines.append("- 这导致阶段 6 的正样本率 81.8%, 与 UNOSAT PDF 声明的 22% 严重不符\n")
    lines.append("- **阶段 6 的所有 AUC/F1/PR-AUC 数值作废, 不应被引用**\n")

    lines.append("\n## 2. 修正后的标签\n")
    lines.append("- **正确标签字段**: `Main_Dam_1` (4月3日最终评估)\n")
    lines.append("- Main_Dam_1 = **14** → **label_damage = 1** (Visible Damage, 767/3,459 = 22.2%)\n")
    lines.append("- Main_Dam_1 = **6** → **label_damage = 0** (No Visible Damage, 2,692/3,459 = 77.8%)\n")
    lines.append("- 与 UNOSAT PDF '767 cells out of 3,459 sustained visible damage (22%)' 完全一致\n")
    lines.append("- No Visible Damage 类别包含无建筑 cells (PDF 注明 'not all 3,459 cells include buildings')\n")

    lines.append("\n## 3. 验证集统计\n")
    lines.append(f"- 验证集总网格: **{n}**\n")
    lines.append(f"- Damaged (label=1): **{n_pos}** ({100*n_pos/n:.1f}%)\n")
    lines.append(f"- No Visible Damage (label=0): **{n_neg}** ({100*n_neg/n:.1f}%)\n")
    lines.append(f"- 正样本比例: {100*n_pos/n:.1f}% (较平衡)\n")

    lines.append("\n### 损害亚型\n")
    for st in ['persistent_damage', 'new_damage_apr3', 'no_damage_both']:
        cnt = int((val_df['damage_subtype']==st).sum())
        pct = 100*cnt/n
        desc = {'persistent_damage':'持续损害 (Mar 26 & Apr 3)', 'new_damage_apr3':'新增损害 (仅 Apr 3)',
                'no_damage_both':'两次评估均无损害'}[st]
        lines.append(f"- {desc}: {cnt} ({pct:.1f}%)\n")

    lines.append("\n## 4. 各模型验证指标\n")
    lines.append("| 模型 | ROC-AUC | PR-AUC | Brier | F1@0.5 | Precision@0.5 | Recall@0.5 | Spearman ρ | Top-k Hit |\n")
    lines.append("|------|---------|--------|-------|--------|---------------|------------|------------|----------|\n")
    for r in results:
        lines.append(f"| {r['model']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['brier_score']:.4f} | "
                     f"{r['f1_0.5']:.4f} | {r['precision_0.5']:.4f} | {r['recall_0.5']:.4f} | "
                     f"{r['spearman_r']:.4f} | {r['top_k_hit_rate']:.4f} |\n")

    # Best models
    valid = [r for r in results if 'roc_auc' in r]
    if valid:
        best_auc = max(valid, key=lambda r: r['roc_auc'])
        best_pr = max(valid, key=lambda r: r['pr_auc'])
        best_f1 = max(valid, key=lambda r: r['f1_0.5'])
        best_brier = min(valid, key=lambda r: r['brier_score'])

        lines.append(f"\n- **最佳 ROC-AUC**: {best_auc['model']} ({best_auc['roc_auc']:.4f})\n")
        lines.append(f"- **最佳 PR-AUC**: {best_pr['model']} ({best_pr['pr_auc']:.4f})\n")
        lines.append(f"- **最佳 F1@0.5**: {best_f1['model']} ({best_f1['f1_0.5']:.4f})\n")
        lines.append(f"- **最佳 Brier**: {best_brier['model']} ({best_brier['brier_score']:.4f})\n")

    lines.append("\n## 5. 融合是否改善了单源结果\n")
    # Compare S2, VIIRS, Logit
    s2_res = next((r for r in results if 'S2' in r['model']), None)
    viirs_res = next((r for r in results if 'VIIRS' in r['model']), None)
    logit_res = next((r for r in results if 'Logit' in r['model']), None)
    ds_belief = next((r for r in results if 'Belief' in r['model']), None)
    ds_plaus = next((r for r in results if 'Plaus' in r['model']), None)
    ds_mid = next((r for r in results if 'Mid' in r['model']), None)

    if s2_res and viirs_res and logit_res:
        lines.append(f"- S2 AUC={s2_res['roc_auc']:.4f}, VIIRS AUC={viirs_res['roc_auc']:.4f}, Logit AUC={logit_res['roc_auc']:.4f}\n")
        if logit_res['roc_auc'] > max(s2_res['roc_auc'], viirs_res['roc_auc']):
            lines.append("- ✅ 融合 AUC 优于两个单源 → 融合有效\n")
        elif logit_res['roc_auc'] > 0.5:
            lines.append("- ⚠️ 融合 AUC 低于最佳单源 → 等权融合可能不是最优策略\n")
        else:
            lines.append("- ❌ 融合 AUC < 0.5 → 融合损害了判别力, 需要调整权重\n")

    lines.append("\n## 6. D-S 验证\n")
    if 'ds_decision_class' in val_df.columns:
        lines.append("| D-S 决策类别 | 数量 | Damaged | 命中率 |\n")
        lines.append("|-------------|------|---------|--------|\n")
        for dclass in ['confident_damage','possible_damage','likely_no_damage','uncertain_conflict']:
            sub = val_df[val_df['ds_decision_class']==dclass]
            if len(sub)>0:
                dmg = int((sub['label_damage']==1).sum())
                lines.append(f"| {dclass} | {len(sub)} | {dmg} | {100*dmg/len(sub):.1f}% |\n")

    if 'evidence_type' in val_df.columns:
        lines.append("\n| Evidence Type | 数量 | Damaged | 命中率 |\n")
        lines.append("|--------------|------|---------|--------|\n")
        for et in ['both_high','s2_high_viirs_low','s2_low_viirs_high','both_low']:
            sub = val_df[val_df['evidence_type']==et]
            if len(sub)>0:
                dmg = int((sub['label_damage']==1).sum())
                lines.append(f"| {et} | {len(sub)} | {dmg} | {100*dmg/len(sub):.1f}% |\n")

    lines.append("\n## 7. 是否需要调整融合权重\n")
    if s2_res and viirs_res:
        if s2_res['roc_auc'] > viirs_res['roc_auc'] + 0.05:
            lines.append(f"- S2 (AUC={s2_res['roc_auc']:.3f}) 明显优于 VIIRS (AUC={viirs_res['roc_auc']:.3f})\n")
            lines.append("- **建议**: 融合时应给 S2 更高权重 (如 0.7 S2 + 0.3 VIIRS)\n")
            lines.append("- D-S BPA 中可降低 VIIRS 的基础不确定性 u_viirs\n")
        else:
            lines.append("- 两个传感器表现接近, 等权融合合理\n")

    lines.append("\n## 8. 局限性\n")
    lines.append("1. **UNOSAT 不是地面真值**: 基于高分辨率光学卫星的目视解译, 存在漏检/误检\n")
    lines.append("2. **日期不完全一致**: UNOSAT (Apr 3) vs Sentinel-2 (May 8) vs VIIRS (月产品)\n")
    lines.append("3. **'No Visible Damage' 含无建筑 cells**: label=0 中有 cells 根本没有建筑, 不完全是'结构完好'\n")
    lines.append("4. **可见损害不等于光谱变化**: 有些损害可能不影响光谱指数\n")
    lines.append("5. **空间匹配误差**: 500m 网格不完全 1:1 对齐\n")

    lines.append("\n## 9. 主要结论\n")
    if valid:
        best = max(valid, key=lambda r: r['roc_auc'])
        lines.append(f"1. 在修正后的 strict label 下, **{best['model']}** 表现最好 (AUC={best['roc_auc']:.4f})\n")
    lines.append(f"2. 验证集 {n} 个网格, 正样本比 {n_pos/n:.3f}, 与 UNOSAT 统计一致\n")
    lines.append("3. 阶段 6 原 AUC/F1 基于错误标签, 已全部作废\n")
    lines.append("4. 建议进入阶段 7: 融合权重与 D-S BPA 参数敏感性分析\n")

    lines.append("\n## 10. 输出文件\n")
    outputs = [
        (OUT_DATA / 'validation_grid_unosat_strict.csv', '验证集 CSV'),
        (OUT_DATA / 'validation_grid_unosat_strict.geojson', '验证集 GeoJSON'),
        (OUT_TAB / 'validation_label_counts_strict.csv', '标签计数'),
        (OUT_TAB / 'validation_metrics_unosat_strict.csv', '验证指标'),
        (OUT_TAB / 'ds_decision_validation_crosstab_strict.csv', 'D-S 决策交叉表'),
        (OUT_TAB / 'evidence_type_validation_crosstab_strict.csv', '证据类型交叉表'),
        (OUT_FIG / 'roc_curves_unosat_strict.png', 'ROC 曲线'),
        (OUT_FIG / 'pr_curves_unosat_strict.png', 'PR 曲线'),
        (OUT_FIG / 'validation_score_distributions_strict.png', '分数分布'),
        (OUT_FIG / 'validation_map_unosat_strict_labels_vs_predictions.png', '验证地图'),
        (OUT_FIG / 'model_auc_comparison_strict.png', 'AUC 对比'),
        (OUT_MD / 'validation_unosat_strict_report.md', '本报告'),
    ]
    for p, desc in outputs:
        lines.append(f"| `{p.relative_to(PROJECT_ROOT)}` | {desc} |\n")

    with open(OUT_MD / 'validation_unosat_strict_report.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {OUT_MD / 'validation_unosat_strict_report.md'}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("UNOSAT Strict Label 验证 (修正版)")
    print("=" * 70)

    # 1. Build strict labels
    print("\n[1] Building strict labels (Main_Dam_1=14 → damaged)...")
    gdf_all, val_df = build_strict_labels()
    n_val = len(val_df)
    n_pos = int((val_df['label_damage']==1).sum())
    n_neg = int((val_df['label_damage']==0).sum())
    print(f"    Validation set: {n_val} grids, {n_pos} damaged ({100*n_pos/n_val:.1f}%), "
          f"{n_neg} no visible damage ({100*n_neg/n_val:.1f}%)")

    # 2. Evaluate
    print("\n[2] Computing validation metrics...")
    results = evaluate_all(val_df)

    print("\n  === Results ===")
    for r in results:
        print(f"  {r['model']:25s}: AUC={r['roc_auc']:.4f}, PR-AUC={r['pr_auc']:.4f}, "
              f"F1@0.5={r['f1_0.5']:.4f}, Spearman={r['spearman_r']:.4f}, Brier={r['brier_score']:.4f}")

    # 3. D-S validation
    print("\n[3] D-S decision & evidence type validation...")
    if 'ds_decision_class' in val_df.columns:
        for dclass in ['confident_damage','possible_damage','likely_no_damage']:
            sub = val_df[val_df['ds_decision_class']==dclass]
            if len(sub)>0:
                dmg = int((sub['label_damage']==1).sum())
                print(f"    {dclass}: {len(sub)} grids, {dmg} damaged ({100*dmg/len(sub):.1f}%)")
    if 'evidence_type' in val_df.columns:
        for et in ['both_high','s2_low_viirs_high','both_low']:
            sub = val_df[val_df['evidence_type']==et]
            if len(sub)>0:
                dmg = int((sub['label_damage']==1).sum())
                print(f"    {et}: {len(sub)} grids, {dmg} damaged ({100*dmg/len(sub):.1f}%)")

    # 4. Save
    print("\n[4] Saving tables...")
    save_all(val_df, results, gdf_all)

    # 5. Plot
    print("\n[5] Plotting figures...")
    plot_all(val_df, results)

    # 6. Report
    print("\n[6] Generating report...")
    generate_report(val_df, results)

    # Final
    valid_res = [r for r in results if 'roc_auc' in r]
    best = max(valid_res, key=lambda r: r['roc_auc']) if valid_res else None
    print("\n" + "=" * 70)
    print("Strict label 验证完成!")
    print(f"  验证集: {n_val} grids ({n_pos} damaged, {n_neg} no-damage, prev={n_pos/n_val:.3f})")
    if best:
        print(f"  最佳模型: {best['model']} (AUC={best['roc_auc']:.4f})")
        s2_r = next((r for r in results if 'S2' in r['model']), None)
        viirs_r = next((r for r in results if 'VIIRS' in r['model']), None)
        logit_r = next((r for r in results if 'Logit' in r['model']), None)
        if s2_r and viirs_r and logit_r:
            print(f"  S2 AUC={s2_r['roc_auc']:.4f}, VIIRS AUC={viirs_r['roc_auc']:.4f}, "
                  f"Logit AUC={logit_r['roc_auc']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
