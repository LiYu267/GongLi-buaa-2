"""
实验 v3：可视化与评估
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
import json

BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs_v3'

print("生成 v3 可视化...")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

damage_cmap = LinearSegmentedColormap.from_list('damage', [(1,1,1),(1,0.95,0.5),(1,0.5,0),(0.8,0,0)], N=256)

# Load
names = ['VIIRS','S2','SAR','InSAR']
P, S = {}, {}
for n in names:
    P[n] = np.load(OUT/f'exp_v3_p_{n}.npy') if (OUT/f'exp_v3_p_{n}.npy').exists() else np.full((100,150),np.nan)
    try: S[n] = np.load(OUT/f'exp_v3_s_{n}.npy')
    except: S[n] = np.full_like(P[n], np.nan)

fused = np.load(OUT/'exp_v3_bayes_fused_mean.npy')
fused_std = np.load(OUT/'exp_v3_bayes_fused_std.npy')
hdi_low = np.load(OUT/'exp_v3_bayes_hdi_low.npy')
hdi_high = np.load(OUT/'exp_v3_bayes_hdi_high.npy')

try:
    ds_bel = np.load(OUT/'exp_v3_ds_bel.npy')
    ds_pl = np.load(OUT/'exp_v3_ds_pl.npy')
    ds_ig = np.load(OUT/'exp_v3_ds_ig.npy')
    has_ds = True
except: has_ds = False

try:
    with open(OUT/'exp_v3_summary.json') as f:
        summary = json.load(f)
except: summary = {}

# ===== Figure 1: 8-Panel Comparison =====
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Experiment v3: Four-Sensor Bayesian Fusion with Improvements\nMariupol, Ukraine', fontsize=14, fontweight='bold')

mu_v = np.nanmean(P['VIIRS']); mu_s2 = np.nanmean(P['S2']); mu_sar = np.nanmean(P['SAR'])
mu_insar = np.nanmean(P['InSAR']); mu_f = np.nanmean(fused); mu_fs = np.nanmean(fused_std)
panels = [
    (axes[0,0], P['VIIRS'], 'VIIRS NTL', 'mu=%.3f'%mu_v),
    (axes[0,1], P['S2'], 'Sentinel-2 (May->May)', 'mu=%.3f'%mu_s2),
    (axes[0,2], P['SAR'], 'SAR (Hist-Matched)', 'mu=%.3f'%mu_sar),
    (axes[0,3], P['InSAR'], 'InSAR (tau-Normalized)', 'mu=%.3f'%mu_insar),
    (axes[1,0], fused, 'Bayesian Fusion', 'mu=%.3f'%mu_f),
    (axes[1,1], fused_std, 'Fusion Uncertainty sigma', 'mu=%.3f'%mu_fs),
]
if has_ds:
    mu_ds = np.nanmean(ds_bel); mu_ig = np.nanmean(ds_ig)
    panels.append((axes[1,2], ds_bel, 'D-S Bel(Damage)', 'mu=%.3f'%mu_ds))
    panels.append((axes[1,3], ds_ig, 'D-S Ignorance', 'mu=%.3f'%mu_ig))
else:
    mu_hdi = np.nanmean(hdi_high-hdi_low)
    panels.append((axes[1,2], hdi_high-hdi_low, '95% HDI Width', 'mu=%.3f'%mu_hdi))

for ax, data, title, sub in panels:
    im = ax.imshow(data, cmap=damage_cmap, vmin=0, vmax=1, aspect='equal', interpolation='nearest')
    ax.set_title(title, fontweight='bold'); ax.set_xlabel(sub, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(OUT/'exp_v3_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → exp_v3_comparison.png')

# ===== Figure 2: Method Comparison Bar =====
fig, ax = plt.subplots(figsize=(12, 5))
method_names = ['VIIRS','S2','SAR','InSAR','Vote\nMean','Vote\nW','D-S\nBel','Bayesian']
method_vals = [
    np.nanmean(P['VIIRS']), np.nanmean(P['S2']), np.nanmean(P['SAR']), np.nanmean(P['InSAR']),
    np.nanmean((P['VIIRS']+P['S2']+P['SAR']+P['InSAR'])/4),
]
# Compute weighted vote and D-S Bel if available
valid_all = ~np.isnan(P['VIIRS'])
for n in names[1:]: valid_all &= ~np.isnan(P[n])
if np.sum(valid_all) > 0:
    p_arr = np.column_stack([P[n][valid_all] for n in names])
    s_arr = np.column_stack([S[n][valid_all] for n in names])
    prec = 1.0/(s_arr**2+0.005)
    method_vals.append(np.mean(np.sum(p_arr*prec,axis=1)/np.sum(prec,axis=1)))
else:
    method_vals.append(0)
method_vals.append(np.nanmean(ds_bel) if has_ds else 0)
method_vals.append(np.nanmean(fused))

colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#9E9E9E','#607D8B','#795548','#F44336']
bars = ax.bar(method_names, method_vals, color=colors)
for bar, val in zip(bars, method_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{val:.3f}',
            ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Damage Probability', fontsize=12)
ax.set_title('Experiment v3: Damage Probability by Method (4-Sensor Fusion with Improvements)', fontsize=13, fontweight='bold')
ax.set_ylim(0,1.1); ax.grid(True,alpha=0.2,axis='y')
plt.tight_layout(); fig.savefig(OUT/'exp_v3_methods.png', dpi=150, bbox_inches='tight'); plt.close()
print('  → exp_v3_methods.png')

# ===== Figure 3: Synthetic Validation =====
if summary.get('synthetic_results'):
    syn = summary['synthetic_results']
    fig, ax = plt.subplots(figsize=(12, 5))
    scenarios = [s['scenario'] for s in syn]
    x = np.arange(len(scenarios))
    w = 0.2
    for i, (label, key, color) in enumerate([
        ('Best Single', 'auc_single', '#2196F3'),
        ('Vote Mean', 'auc_vote', '#9E9E9E'),
        ('D-S Bel', 'auc_ds', '#795548'),
        ('Bayesian', 'auc_bayes', '#F44336'),
    ]):
        if key == 'auc_single':
            vals = [max(s[key]) for s in syn]
        else:
            vals = [s[key] for s in syn]
        ax.bar(x+i*w, vals, w, label=label, color=color, alpha=0.8)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel('ROC AUC'); ax.set_title('Synthetic Validation: Multi-Scenario Damage Injection', fontweight='bold')
    ax.legend(); ax.set_ylim(0.5,1.0); ax.grid(True,alpha=0.2,axis='y')
    plt.tight_layout(); fig.savefig(OUT/'exp_v3_synthetic.png', dpi=150, bbox_inches='tight'); plt.close()
    print('  → exp_v3_synthetic.png')

# ===== Figure 4: Improvement comparison v2 vs v3 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('v2 → v3: Key Improvements', fontsize=13, fontweight='bold')

# SAR comparison (v2 vs v3)
v2_sar_mean = 0.9936  # from v2 output
v3_sar_mean = np.nanmean(P['SAR'])
ax = axes[0]
ax.bar(['v2 (no calibration)', 'v3 (histogram matched)'], [v2_sar_mean, v3_sar_mean],
       color=['#F44336','#4CAF50'])
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Expected if balanced')
ax.set_ylabel('SAR p mean'); ax.set_title('SAR Calibration Fix'); ax.legend()
ax.set_ylim(0,1.1)

# InSAR comparison
v2_insar_mean = 0.2204
v3_insar_mean = np.nanmean(P['InSAR'])
ax = axes[1]
ax.bar(['v2 (raw ΔCoh)', 'v3 (τ-normalized)'], [v2_insar_mean, v3_insar_mean],
       color=['#F44336','#4CAF50'])
ax.set_ylabel('InSAR p mean'); ax.set_title('InSAR Temporal Normalization')
ax.set_ylim(0,1.1)

plt.tight_layout(); fig.savefig(OUT/'exp_v3_improvements.png', dpi=150, bbox_inches='tight'); plt.close()
print('  → exp_v3_improvements.png')

print('\n可视化完成。所有输出在 exp_outputs_v3/')
