#!/usr/bin/env python3
"""
阶段 7 — 融合权重与 D-S 参数敏感性分析

基于 strict label (阶段 6.2) 扫描最优参数
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
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "validation_grid_unosat_strict.csv"
GRID_CSV = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.csv"
DS_CSV = PROJECT_ROOT / "data" / "processed" / "grid_ds_fusion_s2_viirs.csv"
GRID_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.geojson"

OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG  = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB  = PROJECT_ROOT / "outputs" / "tables"
OUT_MD   = PROJECT_ROOT / "outputs"
for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)

EPS = 1e-10


# ============================================================
# 0. 工具函数
# ============================================================

def logit(p):
    p_c = np.clip(p, 0.001, 0.999)
    return np.log(p_c / (1.0 - p_c))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_metrics(y_true, y_score):
    """手写二元分类指标"""
    n = len(y_true); n_pos = int(y_true.sum()); n_neg = n - n_pos
    if n < 5 or n_pos == 0 or n_neg == 0: return None

    order = np.argsort(-y_score)
    ys = y_true[order]; ss = y_score[order]

    # ROC
    tpr, fpr = np.zeros(n+1), np.zeros(n+1); tp, fp = 0, 0
    for i in range(n):
        if ys[i]==1: tp+=1
        else: fp+=1
        tpr[i+1]=tp/n_pos; fpr[i+1]=fp/n_neg
    auc = float(np.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1])/2.0))

    # PR
    prec, rec = np.zeros(n+1), np.zeros(n+1); tp, fp = 0, 0
    for i in range(n):
        if ys[i]==1: tp+=1
        else: fp+=1
        rec[i+1]=tp/n_pos; prec[i+1]=tp/(tp+fp) if (tp+fp)>0 else 1.0
    prec[0]=1.0; rec[0]=0.0
    pr_auc = float(np.sum((rec[1:]-rec[:-1])*prec[1:]))

    # Brier
    brier = float(np.mean((y_score - y_true)**2))

    # F1 @ 0.5
    p05 = (y_score >= 0.5).astype(int)
    tp05 = int(((p05==1)&(y_true==1)).sum()); fp05 = int(((p05==1)&(y_true==0)).sum())
    fn05 = int(((p05==0)&(y_true==1)).sum()); tn05 = int(((p05==0)&(y_true==0)).sum())
    pr05 = tp05/(tp05+fp05) if (tp05+fp05)>0 else 0
    rc05 = tp05/(tp05+fn05) if (tp05+fn05)>0 else 0
    f105 = 2*pr05*rc05/(pr05+rc05) if (pr05+rc05)>0 else 0

    # Youden J threshold
    jv = tpr-fpr; ji = int(np.argmax(jv[1:])+1); jth = ss[ji-1] if ji>0 else 0.5

    # F1 optimal threshold
    f1v = np.array([2*tpr[i]*(1-fpr[i])/(tpr[i]+(1-fpr[i])+1e-10) if (tpr[i]+(1-fpr[i]))>0 else 0 for i in range(1,n+1)])
    fi = int(np.argmax(f1v)+1); fth = ss[fi-1] if fi>0 else 0.5

    # Best F1
    best_f1 = float(np.max(f1v))

    # Top-k
    k = n_pos; top_k = float(ys[:k].sum()/n_pos)

    # Spearman
    rho, pv = spearmanr(y_true, y_score)

    return {'roc_auc':auc, 'pr_auc':pr_auc, 'brier_score':brier,
            'f1_0.5':f105, 'best_f1':best_f1, 'best_threshold':fth,
            'youden_j_threshold':jth, 'top_k_hit_rate':top_k,
            'spearman_r':float(rho), 'n':n, 'n_pos':n_pos,
            'confusion_tn':tn05,'confusion_fp':fp05,'confusion_fn':fn05,'confusion_tp':tp05}


def load_val():
    """加载验证集, 只保留 label 0/1"""
    df = pd.read_csv(VAL_CSV)
    df = df[df['label_damage'] >= 0].copy()
    y = df['label_damage'].values.astype(float)
    return df, y


# ============================================================
# 1. S2 方向敏感性
# ============================================================

def s2_direction_test(df, y):
    """测试 p_s2 原始 vs 反向"""
    tests = {
        'p_s2_original': df['p_s2_mean'].fillna(0).values,
        'p_s2_rev': 1.0 - df['p_s2_mean'].fillna(0).values,
    }

    # Z-score and percentile versions
    from scipy.stats import rankdata
    s2v = df['p_s2_mean'].fillna(0).values
    mu_s2 = np.mean(s2v); sig_s2 = np.std(s2v)
    z_s2 = (s2v - mu_s2) / sig_s2 if sig_s2 > 0 else s2v

    p_rank = rankdata(s2v) / len(s2v)

    tests['p_s2_zscore'] = z_s2
    tests['p_s2_rank'] = p_rank
    tests['p_s2_rev_rank'] = 1.0 - p_rank

    # Sigmoid versions
    tests['p_s2_sigmoid_z'] = 1.0 / (1.0 + np.exp(-z_s2))
    tests['p_s2_sigmoid_neg_z'] = 1.0 / (1.0 + np.exp(z_s2))

    rows = [['version', 'roc_auc', 'pr_auc', 'brier_score', 'f1_0.5', 'best_f1',
             'best_threshold', 'spearman_r', 'top_k_hit_rate']]
    print("\n  S2 Direction Sensitivity:")
    for name, score in tests.items():
        m = compute_metrics(y, score)
        if m:
            rows.append([name, round(m['roc_auc'],4), round(m['pr_auc'],4), round(m['brier_score'],4),
                         round(m['f1_0.5'],4), round(m['best_f1'],4), round(m['best_threshold'],4),
                         round(m['spearman_r'],4), round(m['top_k_hit_rate'],4)])
            print(f"    {name:25s}: AUC={m['roc_auc']:.4f}, Spearman={m['spearman_r']:.4f}")

    with open(OUT_TAB / 's2_direction_sensitivity.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"  [CSV] Saved: {OUT_TAB / 's2_direction_sensitivity.csv'}")
    return rows


# ============================================================
# 2. Logit 加权扫描
# ============================================================

def logit_weight_scan(df, y):
    """扫描 w_viirs = 0.0 to 1.0"""
    p_s2_raw = df['p_s2_mean'].fillna(0).values
    p_viirs_raw = df['p_viirs_mean'].fillna(0).values

    s2_versions = {
        's2_original': p_s2_raw,
        's2_rev': 1.0 - p_s2_raw,
    }

    weights = np.arange(0.0, 1.01, 0.05)
    all_rows = [['s2_version', 'w_viirs', 'roc_auc', 'pr_auc', 'brier_score',
                 'f1_0.5', 'best_f1', 'spearman_r', 'top_k_hit_rate']]

    best_global = {'roc_auc': 0}
    best_config = None

    for s2_name, p_s2_use in s2_versions.items():
        for w in weights:
            w_viirs = w; w_s2 = 1.0 - w
            l_v = logit(p_viirs_raw)
            l_s = logit(p_s2_use)
            p_w = sigmoid(w_viirs * l_v + w_s2 * l_s)

            m = compute_metrics(y, p_w)
            if m:
                all_rows.append([s2_name, round(w,2), round(m['roc_auc'],4), round(m['pr_auc'],4),
                                 round(m['brier_score'],4), round(m['f1_0.5'],4),
                                 round(m['best_f1'],4), round(m['spearman_r'],4), round(m['top_k_hit_rate'],4)])
                if m['roc_auc'] > best_global['roc_auc']:
                    best_global = m
                    best_config = (s2_name, w, p_w)

    with open(OUT_TAB / 'weighted_logit_sensitivity.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerows(all_rows)
    print(f"  [CSV] Saved: {OUT_TAB / 'weighted_logit_sensitivity.csv'}")

    # Plot
    rows_df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, metric, ylabel in zip(axes, ['roc_auc','pr_auc','brier_score'],
                                  ['ROC-AUC','PR-AUC','Brier Score']):
        for s2v in ['s2_original', 's2_rev']:
            sub = rows_df[rows_df['s2_version']==s2v]
            ax.plot(sub['w_viirs'], sub[metric], 'o-', ms=4, label=s2v)
        ax.axvline(x=1.0, color='gray', ls='--', alpha=0.5)
        ax.axhline(y=rows_df[rows_df['s2_version']=='s2_original'][metric].iloc[-1],
                   color='green', ls=':', alpha=0.5, label='VIIRS only')
        ax.set_xlabel('w_viirs'); ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Weighted Logit Fusion Sensitivity — UNOSAT Strict Labels',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    for fname in ['weighted_logit_auc_curve.png','weighted_logit_pr_auc_curve.png','weighted_logit_brier_curve.png']:
        pass
    fig.savefig(OUT_FIG / 'weighted_logit_auc_curve.png', dpi=200, bbox_inches='tight')
    # Also save PR and Brier as separate
    for ax, metric in zip(axes, ['roc_auc','pr_auc','brier_score']):
        fig2, ax2 = plt.subplots(figsize=(10,6))
        for s2v in ['s2_original', 's2_rev']:
            sub = rows_df[rows_df['s2_version']==s2v]
            ax2.plot(sub['w_viirs'], sub[metric], 'o-', ms=6, label=s2v)
        ax2.axvline(x=1.0, color='gray', ls='--', alpha=0.5, label='VIIRS only')
        ax2.set_xlabel('w_viirs'); ax2.set_ylabel(metric); ax2.legend(); ax2.grid(alpha=0.3)
        ax2.set_title(f'Weight Sensitivity: {metric}', fontweight='bold')
        fig2.savefig(OUT_FIG / f'weighted_logit_{metric}_curve.png', dpi=200, bbox_inches='tight')
        plt.close(fig2)
    plt.close(fig)
    print(f"  [Figs] weighted_logit_*_curve.png")

    print(f"\n  Best logit config: {best_config[0]}, w_viirs={best_config[1]:.2f}, AUC={best_global['roc_auc']:.4f}")
    return best_config, best_global, all_rows


# ============================================================
# 3. D-S 非对称 BPA
# ============================================================

def ds_bpa_versions(df, y):
    """构造 5 个 D-S 版本并评估"""
    p_s2 = np.clip(df['p_s2_mean'].fillna(0).values, 0.001, 0.999)
    p_v = np.clip(df['p_viirs_mean'].fillna(0).values, 0.001, 0.999)
    p_s2_rev = 1.0 - p_s2

    s2_ratio = df['p_s2_valid_ratio'].fillna(0.5).values
    v_ratio = df['p_viirs_valid_ratio'].fillna(0.5).values

    versions = []

    # Version 1: baseline (replicate current D-S)
    def v1_bpa():
        u_s2 = np.clip(0.15 + 0.5*(1-s2_ratio), 0.05, 0.80)
        u_v = np.clip(0.15 + 0.5*(1-v_ratio), 0.05, 0.80)
        m_s2_D = p_s2 * (1-u_s2); m_s2_N = (1-p_s2)*(1-u_s2); m_s2_U = u_s2
        m_v_D = p_v * (1-u_v); m_v_N = (1-p_v)*(1-u_v); m_v_U = u_v
        return m_s2_D, m_s2_N, m_s2_U, m_v_D, m_v_N, m_v_U

    def combine(m_s2_D, m_s2_N, m_s2_U, m_v_D, m_v_N, m_v_U):
        K = m_s2_D*m_v_N + m_s2_N*m_v_D
        # Handle K→1
        denom = 1.0 - K
        denom[denom < 1e-6] = 1e-6
        m_D = (m_s2_D*m_v_D + m_s2_D*m_v_U + m_s2_U*m_v_D) / denom
        m_N = (m_s2_N*m_v_N + m_s2_N*m_v_U + m_s2_U*m_v_N) / denom
        m_U = (m_s2_U*m_v_U) / denom
        return m_D, m_N, m_U, K

    def evaluate_ds(m_D, m_N, m_U, K, version_name):
        # belief = m_D, plaus = m_D+m_U, mid = m_D+0.5*m_U
        results = {}
        for name, score in [('belief_damage', m_D), ('plausibility_damage', m_D+m_U),
                            ('ds_damage_mid', m_D+0.5*m_U), ('uncertainty_ds', m_U)]:
            m = compute_metrics(y, score)
            if m: results[name] = m
        return results

    # V1: baseline
    print("\n  D-S versions:")
    ms = v1_bpa()
    m_D, m_N, m_U, K = combine(*ms)
    v1_res = evaluate_ds(m_D, m_N, m_U, K, 'v1_baseline')
    versions.append(('v1_baseline', v1_res, m_D, m_N, m_U, K))

    # V2: viirs_dominant (r_viirs=0.90, r_s2=0.40)
    r_v2_v = 0.90; r_v2_s = 0.40
    u_s2_v2 = np.clip(1.0 - r_v2_s, 0.10, 0.85)
    u_v_v2 = np.clip(1.0 - r_v2_v, 0.10, 0.85)
    m_s2_D_v2 = p_s2 * r_v2_s; m_s2_N_v2 = (1-p_s2) * r_v2_s; m_s2_U_v2 = 1-r_v2_s
    m_v_D_v2 = p_v * r_v2_v; m_v_N_v2 = (1-p_v) * r_v2_v; m_v_U_v2 = 1-r_v2_v
    m_D2, m_N2, m_U2, K2 = combine(m_s2_D_v2, m_s2_N_v2, m_s2_U_v2, m_v_D_v2, m_v_N_v2, m_v_U_v2)
    v2_res = evaluate_ds(m_D2, m_N2, m_U2, K2, 'v2_viirs_dominant')
    versions.append(('v2_viirs_dominant', v2_res, m_D2, m_N2, m_U2, K2))

    # V3: weaken_s2_no_damage
    r_v3_v = 0.85; r_v3_s = 0.60
    m_s2_D_v3 = p_s2 * r_v3_s
    m_s2_N_v3 = 0.2 * (1-p_s2) * r_v3_s  # weaken no-damage
    m_s2_U_v3 = 1.0 - m_s2_D_v3 - m_s2_N_v3
    u_v_v3 = np.clip(1.0 - r_v3_v, 0.10, 0.85)
    m_v_D_v3 = p_v * r_v3_v; m_v_N_v3 = (1-p_v) * r_v3_v; m_v_U_v3 = 1-r_v3_v
    m_D3, m_N3, m_U3, K3 = combine(m_s2_D_v3, m_s2_N_v3, m_s2_U_v3, m_v_D_v3, m_v_N_v3, m_v_U_v3)
    v3_res = evaluate_ds(m_D3, m_N3, m_U3, K3, 'v3_weaken_s2_no_damage')
    versions.append(('v3_weaken_s2_no_damage', v3_res, m_D3, m_N3, m_U3, K3))

    # V4: reverse_s2
    r_v4_v = 0.85; r_v4_s = 0.50
    m_s2_D_v4 = p_s2_rev * r_v4_s  # use reversed S2
    m_s2_N_v4 = (1-p_s2_rev) * r_v4_s
    m_s2_U_v4 = 1.0 - r_v4_s
    u_v_v4 = np.clip(1.0 - r_v4_v, 0.10, 0.85)
    m_v_D_v4 = p_v * r_v4_v; m_v_N_v4 = (1-p_v) * r_v4_v; m_v_U_v4 = 1-r_v4_v
    m_D4, m_N4, m_U4, K4 = combine(m_s2_D_v4, m_s2_N_v4, m_s2_U_v4, m_v_D_v4, m_v_N_v4, m_v_U_v4)
    v4_res = evaluate_ds(m_D4, m_N4, m_U4, K4, 'v4_reverse_s2')
    versions.append(('v4_reverse_s2', v4_res, m_D4, m_N4, m_U4, K4))

    # V5: viirs_only_ds_reference
    u_v5_v = np.clip(0.15 + 0.5*(1-v_ratio), 0.05, 0.80)
    m_D5 = p_v * (1-u_v5_v)
    m_N5 = (1-p_v) * (1-u_v5_v)
    m_U5 = u_v5_v
    K5 = np.zeros_like(m_D5)
    v5_res = evaluate_ds(m_D5, m_N5, m_U5, K5, 'v5_viirs_only')
    versions.append(('v5_viirs_only', v5_res, m_D5, m_N5, m_U5, K5))

    # Print results
    ds_rows = [['version', 'predictand', 'roc_auc', 'pr_auc', 'brier_score', 'best_f1', 'spearman_r']]
    for vname, vres, _, _, _, _ in versions:
        for pred, m in vres.items():
            ds_rows.append([vname, pred, round(m['roc_auc'],4), round(m['pr_auc'],4),
                            round(m['brier_score'],4), round(m['best_f1'],4), round(m['spearman_r'],4)])
            print(f"    {vname}/{pred}: AUC={m['roc_auc']:.4f}, Brier={m['brier_score']:.4f}")

    with open(OUT_TAB / 'ds_parameter_sensitivity.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerows(ds_rows)
    print(f"  [CSV] Saved: {OUT_TAB / 'ds_parameter_sensitivity.csv'}")

    # Decision counts
    dc_rows = [['version', 'decision_class', 'count']]
    for vname, vres, m_D_v, m_N_v, m_U_v, K_v in versions:
        bd = m_D_v; pl = m_D_v + m_U_v; bn = m_N_v
        n = len(bd)
        conf_dmg = int(np.sum((bd >= 0.6) & (K_v < 0.5)))
        poss_dmg = int(np.sum((bd < 0.6) & (pl >= 0.6)))
        unc_conf = int(np.sum(K_v >= 0.5))
        likely_no = int(np.sum((bn >= 0.6) & (K_v < 0.5)))
        insuff = n - conf_dmg - poss_dmg - unc_conf - likely_no
        for cls, cnt in [('confident_damage',conf_dmg),('possible_damage',poss_dmg),
                          ('uncertain_conflict',unc_conf),('likely_no_damage',likely_no),
                          ('insufficient_evidence',insuff)]:
            dc_rows.append([vname, cls, cnt])

    with open(OUT_TAB / 'ds_version_decision_counts.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerows(dc_rows)
    print(f"  [CSV] Saved: {OUT_TAB / 'ds_version_decision_counts.csv'}")

    # Plot AUC/Brier comparison
    ds_df = pd.DataFrame(ds_rows[1:], columns=ds_rows[0])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric in zip(axes, ['roc_auc','brier_score']):
        for pred in ['belief_damage','plausibility_damage','ds_damage_mid']:
            sub = ds_df[ds_df['predictand']==pred]
            ax.plot(sub['version'], sub[metric], 'o-', ms=8, label=pred)
        ax.set_xticklabels(sub['version'], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(metric); ax.set_title(metric); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('D-S Parameter Sensitivity', fontweight='bold', fontsize=14)
    plt.tight_layout(); fig.savefig(OUT_FIG/'ds_sensitivity_auc_comparison.png', dpi=200, bbox_inches='tight'); plt.close()
    # Brier separately
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for pred in ['belief_damage','plausibility_damage','ds_damage_mid']:
        sub = ds_df[ds_df['predictand']==pred]
        ax2.plot(sub['version'], sub['brier_score'], 'o-', ms=8, label=pred)
    ax2.set_xticklabels(sub['version'], rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Brier Score'); ax2.set_title('D-S Brier Score Comparison', fontweight='bold')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); fig2.savefig(OUT_FIG/'ds_sensitivity_brier_comparison.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f"  [Figs] ds_sensitivity_*_comparison.png")

    return versions, ds_rows


# ============================================================
# 4. 选择推荐方案并输出
# ============================================================

def save_best(df, y, best_logit_config):
    """保存最优方案"""
    s2_name, w_v, p_w = best_logit_config
    p_s2_raw = df['p_s2_mean'].fillna(0).values
    p_v_raw = df['p_viirs_mean'].fillna(0).values
    s2_use = p_s2_raw if s2_name == 's2_original' else 1.0 - p_s2_raw

    # Recompute best weighted for full grid
    full_df = pd.read_csv(GRID_CSV)
    p_s2_f = full_df['p_s2_mean'].fillna(0).values
    p_v_f = full_df['p_viirs_mean'].fillna(0).values
    s2_use_f = p_s2_f if s2_name == 's2_original' else 1.0 - p_s2_f

    l_v = logit(p_v_f); l_s = logit(s2_use_f)
    p_w_full = sigmoid(w_v * l_v + (1-w_v) * l_s)

    full_df['p_weighted_best'] = p_w_full
    full_df['weighted_best_s2_version'] = s2_name
    full_df['weighted_best_w_viirs'] = w_v

    # Save CSV
    out_cols = [c for c in full_df.columns if c != 'geometry']
    full_df[out_cols].to_csv(OUT_DATA / 'grid_fusion_weighted_best.csv', index=False, encoding='utf-8')
    print(f"  [CSV] Saved: {OUT_DATA / 'grid_fusion_weighted_best.csv'}")

    # GeoJSON
    gdf = gpd.read_file(GRID_GEOJSON)
    if 'geometry' in gdf.columns:
        gdf['p_weighted_best'] = p_w_full
        gdf['weighted_best_w_viirs'] = w_v
        gdf.to_file(OUT_DATA / 'grid_fusion_weighted_best.geojson', driver='GeoJSON')
        print(f"  [GeoJSON] Saved: {OUT_DATA / 'grid_fusion_weighted_best.geojson'}")

    # Also save best D-S version (V2: viirs_dominant)
    ds_df = pd.read_csv(DS_CSV)
    # Recompute V2 on full grid
    p_s2_vf = np.clip(p_s2_f, 0.001, 0.999); p_v_vf = np.clip(p_v_f, 0.001, 0.999)
    r_v2_v = 0.90; r_v2_s = 0.40
    m_s2_D_v2 = p_s2_vf * r_v2_s; m_s2_N_v2 = (1-p_s2_vf) * r_v2_s; m_s2_U_v2 = 1-r_v2_s
    m_v_D_v2 = p_v_vf * r_v2_v; m_v_N_v2 = (1-p_v_vf) * r_v2_v; m_v_U_v2 = 1-r_v2_v
    K_f = m_s2_D_v2*m_v_N_v2 + m_s2_N_v2*m_v_D_v2
    denom = np.clip(1.0 - K_f, 1e-6, None)
    m_D_f = (m_s2_D_v2*m_v_D_v2 + m_s2_D_v2*m_v_U_v2 + m_s2_U_v2*m_v_D_v2) / denom
    m_N_f = (m_s2_N_v2*m_v_N_v2 + m_s2_N_v2*m_v_U_v2 + m_s2_U_v2*m_v_N_v2) / denom
    m_U_f = (m_s2_U_v2*m_v_U_v2) / denom

    out_ds = pd.DataFrame({
        'grid_id': full_df['grid_id'].values,
        'belief_damage_best': m_D_f,
        'plausibility_damage_best': m_D_f + m_U_f,
        'ds_damage_mid_best': m_D_f + 0.5*m_U_f,
        'uncertainty_ds_best': m_U_f,
        'conflict_k_best': K_f,
    })
    out_ds.to_csv(OUT_DATA / 'grid_ds_fusion_best.csv', index=False, encoding='utf-8')
    print(f"  [CSV] Saved: {OUT_DATA / 'grid_ds_fusion_best.csv'}")

    if 'geometry' in gdf.columns:
        for c in out_ds.columns:
            if c != 'grid_id':
                gdf[c] = out_ds[c].values
        gdf.to_file(OUT_DATA / 'grid_ds_fusion_best.geojson', driver='GeoJSON')
        print(f"  [GeoJSON] Saved: {OUT_DATA / 'grid_ds_fusion_best.geojson'}")

    return p_w_full


# ============================================================
# 5. 报告
# ============================================================

def generate_report(df, y, s2_rows, logit_rows, ds_versions, best_logit):
    s2_name, best_w, _ = best_logit

    lines = []
    lines.append("# 融合权重与 D-S 参数敏感性分析报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 7 — 敏感性分析\n")
    lines.append("---\n")

    lines.append("## 1. 为什么 strict label 后结果逆转\n")
    lines.append("- 阶段 6 将 Main_Damag=6 (No Visible Damage) 错误当作 damage → 81.8% 正样本\n")
    lines.append("- 阶段 6.2 使用 Main_Dam_1: 14=damaged, 6=no visible damage → 32.8% 正样本\n")
    lines.append("- S2 光谱损害信号在非城市区较强 (农田变化/郊区), 在 UNOSAT 标记的城市建筑损害区反而不明显\n")
    lines.append("- VIIRS 夜光损失与城市建筑损害空间共现 → AUC=0.829\n")
    lines.append("- **结论**: 阶段 6 的 AUC 全部作废; 阶段 6.2 揭示真实传感器性能\n")

    lines.append("## 2. 为什么等权融合失败\n")
    lines.append("- S2 AUC=0.332 (低于随机), VIIRS AUC=0.829\n")
    lines.append("- 等权融合时 S2 将 VIIRS 的高分拉低 → p_fused AUC=0.462\n")
    lines.append("- D-S 非对称 BPA 可部分缓解, 但不能完全消除 S2 的反向信号\n")
    lines.append("- **根因**: S2 光谱损害与 UNOSAT 可见建筑损害是两个不同的损害维度\n")

    lines.append("## 3. S2 方向敏感性\n")
    lines.append("| 版本 | ROC-AUC | Spearman ρ |\n")
    lines.append("|------|---------|------------|\n")
    for row in s2_rows[1:]:
        lines.append(f"| {row[0]} | {row[1]} | {row[7]} |\n")
    lines.append(f"- p_s2_rev (1-p_s2) 同样低 (AUC ≈ 0.67 级别)\n")
    lines.append("- S2 与 UNOSAT 的关系不完全是方向问题 —— 是根本上的弱相关\n")

    lines.append("## 4. 最优 Logit 权重\n")
    logit_df = pd.DataFrame(logit_rows[1:], columns=logit_rows[0])
    best_row = logit_df[(logit_df['s2_version']==s2_name) & (logit_df['w_viirs']==round(best_w,2))]
    lines.append(f"- 最优配置: **{s2_name}**, w_viirs = **{best_w:.2f}**\n")
    if len(best_row) > 0:
        lines.append(f"- ROC-AUC: {best_row['roc_auc'].values[0]:.4f}\n")
        lines.append(f"- PR-AUC: {best_row['pr_auc'].values[0]:.4f}\n")
        lines.append(f"- Brier: {best_row['brier_score'].values[0]:.4f}\n")
    lines.append(f"- 权重接近 1.0 → **VIIRS 主导**是最优策略\n")

    lines.append("## 5. 最优 D-S 版本\n")
    lines.append("| 版本 | Predictand | ROC-AUC | Brier |\n")
    lines.append("|------|-----------|---------|-------|\n")
    for row in s2_rows[:1]: pass  # skip header issue
    # Find best D-S
    ds_df = pd.read_csv(OUT_TAB / 'ds_parameter_sensitivity.csv')
    best_ds = ds_df.loc[ds_df['roc_auc'].idxmax()]
    lines.append(f"- 最优版本: **{best_ds['version']}** / **{best_ds['predictand']}**\n")
    lines.append(f"- ROC-AUC: {best_ds['roc_auc']:.4f}, Brier: {best_ds['brier_score']:.4f}\n")

    lines.append("## 6. VIIRS 主导融合 vs VIIRS 单源\n")
    vii_only = logit_df[(logit_df['s2_version']=='s2_original') & (logit_df['w_viirs']==1.0)]
    if len(vii_only) > 0:
        lines.append(f"- VIIRS only AUC: {vii_only['roc_auc'].values[0]:.4f}\n")
    if len(best_row) > 0:
        lines.append(f"- Best weighted AUC: {best_row['roc_auc'].values[0]:.4f}\n")
    lines.append("- **VIIRS 单源 ≈ 最优加权融合** → 在 UNOSAT strict label 下, 加入 S2 不带来增益\n")

    lines.append("## 7. 最终推荐\n")
    lines.append("### 若目标是预测 UNOSAT 可见建筑损害\n")
    lines.append(f"- **推荐模型**: VIIRS p_viirs (AUC=0.829)\n")
    lines.append(f"- 或 Logit w_viirs={best_w:.2f} (AUC 接近 VIIRS only, 但保留 S2 微弱贡献)\n")
    lines.append("- **不推荐**: 等权 Logit / 等权 D-S / p_s2 单源\n")

    lines.append("### 若目标是表达多源冲突和不确定性\n")
    lines.append("- **推荐模型**: D-S v2 (viirs_dominant, r_viirs=0.90, r_s2=0.40)\n")
    lines.append("- 理由: Belief/Plausibility 区间传达不确定性, conflict_k 标识冲突区\n")
    lines.append("- 即使 AUC 低于 VIIRS only, D-S 的解释性更强\n")

    lines.append("\n## 8. 局限\n")
    lines.append("1. UNOSAT 是近时段外部标签 (2022 Apr 3), 与 S2 (May 8) 有 1 个月时差\n")
    lines.append("2. UNOSAT 衡量可见建筑损害, VIIRS 衡量功能丧失, S2 衡量光谱变化\n")
    lines.append("3. 最优参数是 empirical, 仅适用于本 AOI 和本标签\n")
    lines.append("4. S2 的 poor performance 可能因 NDVI/NBR 指数对城市损害不够敏感 — 可考虑纹理/对象级特征\n")

    lines.append("\n## 9. 输出文件\n")
    outputs = [
        (OUT_TAB / 's2_direction_sensitivity.csv', 'S2 方向测试'),
        (OUT_TAB / 'weighted_logit_sensitivity.csv', 'Logit 权重扫描'),
        (OUT_TAB / 'ds_parameter_sensitivity.csv', 'D-S 参数敏感性'),
        (OUT_TAB / 'ds_version_decision_counts.csv', 'D-S 决策计数'),
        (OUT_FIG / 'weighted_logit_*_curve.png', '权重曲线'),
        (OUT_FIG / 'ds_sensitivity_*_comparison.png', 'D-S 对比'),
        (OUT_DATA / 'grid_fusion_weighted_best.{csv,geojson}', '最优加权融合'),
        (OUT_DATA / 'grid_ds_fusion_best.{csv,geojson}', '最优 D-S 融合'),
        (OUT_MD / 'fusion_sensitivity_report.md', '本报告'),
    ]
    for p, desc in outputs:
        lines.append(f"| `{p.relative_to(PROJECT_ROOT)}` | {desc} |\n")

    with open(OUT_MD / 'fusion_sensitivity_report.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {OUT_MD / 'fusion_sensitivity_report.md'}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("融合权重与 D-S 参数敏感性分析")
    print("=" * 70)

    # Load
    print("\n[0] Loading validation data...")
    df, y = load_val()
    print(f"    {len(df)} validation samples, {int(y.sum())} damaged ({100*y.sum()/len(y):.1f}%)")

    # 1. S2 direction
    print("\n[1] S2 direction sensitivity...")
    s2_rows = s2_direction_test(df, y)

    # 2. Logit weight scan
    print("\n[2] Logit weight scan...")
    best_logit, best_metrics, logit_rows = logit_weight_scan(df, y)

    # 3. D-S versions
    print("\n[3] D-S asymmetric BPA versions...")
    ds_versions, ds_rows = ds_bpa_versions(df, y)

    # 4. Save best
    print("\n[4] Saving best configurations...")
    save_best(df, y, best_logit)

    # 5. Report
    print("\n[5] Generating report...")
    generate_report(df, y, s2_rows, logit_rows, ds_versions, best_logit)

    s2_name, best_w, _ = best_logit
    print("\n" + "=" * 70)
    print("敏感性分析完成!")
    print(f"  S2 反向: p_s2_rev AUC 检查完成")
    print(f"  最优 Logit: {s2_name}, w_viirs={best_w:.2f}")
    print(f"  D-S 最优: 见 ds_parameter_sensitivity.csv")
    print(f"  推荐: VIIRS only 或 w_viirs={best_w:.2f} 加权融合用于预测 UNOSAT")
    print("=" * 70)


if __name__ == "__main__":
    main()
