"""
实验 v2：可视化与评估
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path

BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs_v2'

print("生成 v2 可视化...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results
data = {}
for name in ['VIIRS','S2','SAR','InSAR']:
    data[f'p_{name}'] = np.load(OUT / f'exp_v2_p_{name}.npy')
data['fused_mean'] = np.load(OUT / 'exp_v2_bayes_fused_mean.npy')
data['fused_std'] = np.load(OUT / 'exp_v2_bayes_fused_std.npy')
data['hdi_low'] = np.load(OUT / 'exp_v2_bayes_fused_hdi_low.npy')
data['hdi_high'] = np.load(OUT / 'exp_v2_bayes_fused_hdi_high.npy')

# ===== Figure 1: 6-panel comparison =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Experiment v2: Four-Sensor Damage Probability Fusion\nMariupol, Ukraine', fontsize=14, fontweight='bold')

panels = [
    (axes[0,0], data['p_VIIRS'], 'VIIRS NTL', 'GMM on ΔNTL'),
    (axes[0,1], data['p_S2'], 'Sentinel-2 (May→May)', 'GMM on |PC1|'),
    (axes[0,2], data['p_SAR'], 'Sentinel-1 SAR', 'GMM on Δσ⁰(dB)'),
    (axes[1,0], data['p_InSAR'], 'InSAR Coherence', 'ΔCoherence → p'),
    (axes[1,1], data['fused_mean'], 'Bayesian Fusion', '4-sensor posterior mean'),
    (axes[1,2], data['fused_std'], 'Fusion Uncertainty', 'Posterior std σ'),
]

for ax, d, title, sub in panels:
    valid = ~np.isnan(d)
    im = ax.imshow(d, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal', interpolation='nearest')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel(f'{sub}  |  mean={np.nanmean(d):.3f}', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(OUT / 'exp_v2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → exp_v2_comparison.png')

# ===== Figure 2: Scatter matrix =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Experiment v2: Sensor Agreement Analysis', fontsize=13, fontweight='bold')

valid_all = ~np.isnan(data['p_VIIRS'])
for n in ['S2','SAR','InSAR']:
    valid_all &= ~np.isnan(data[f'p_{n}'])

pairs = [
    (0, 1, 'VIIRS vs S2'), (0, 2, 'VIIRS vs SAR'), (0, 3, 'VIIRS vs InSAR'),
    (1, 2, 'S2 vs SAR'), (1, 3, 'S2 vs InSAR'), (2, 3, 'SAR vs InSAR'),
]
sensor_names = ['VIIRS','S2','SAR','InSAR']
for (i, j, title), ax in zip(pairs, axes.flat):
    pi = data[f'p_{sensor_names[i]}'][valid_all].ravel()
    pj = data[f'p_{sensor_names[j]}'][valid_all].ravel()
    rho = np.corrcoef(pi, pj)[0,1]
    # Subsample for speed
    n_plot = min(3000, len(pi))
    idx = np.random.RandomState(42).choice(len(pi), n_plot, replace=False)
    ax.scatter(pi[idx], pj[idx], s=3, alpha=0.4, c='#2196F3')
    ax.set_xlabel(sensor_names[i]); ax.set_ylabel(sensor_names[j])
    ax.set_title(f'{title}  (ρ={rho:.3f})')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(OUT / 'exp_v2_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → exp_v2_scatter.png')

# ===== Figure 3: Fusion input vs output =====
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle('Experiment v2: Single-Sensor vs Bayesian Fusion', fontsize=13, fontweight='bold')

p_f = data['fused_mean'][valid_all].ravel()
for k, (ax, name) in enumerate(zip(axes, sensor_names)):
    pk = data[f'p_{name}'][valid_all].ravel()
    rho = np.corrcoef(pk, p_f)[0,1]
    ax.scatter(pk[::5], p_f[::5], s=2, alpha=0.3, c='#FF5722')
    ax.plot([0,1], [0,1], 'k--', alpha=0.2, lw=0.5)
    ax.set_xlabel(f'{name} input')
    ax.set_ylabel('Bayesian fused')
    ax.set_title(f'{name} → Fusion\nρ={rho:.3f}')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(OUT / 'exp_v2_input_vs_output.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → exp_v2_input_vs_output.png')

# ===== Figure 4: Method comparison bar chart =====
fig, ax = plt.subplots(figsize=(10, 5))

methods = ['VIIRS', 'S2', 'SAR', 'InSAR', 'Vote\nMean', 'Vote\nWeighted', 'Bayesian\nFusion']
means = [
    np.nanmean(data['p_VIIRS']),
    np.nanmean(data['p_S2']),
    np.nanmean(data['p_SAR']),
    np.nanmean(data['p_InSAR']),
    np.nanmean((data['p_VIIRS'] + data['p_S2'] + data['p_SAR'] + data['p_InSAR'])/4),
    0.1392,  # from output
    0.4617,  # from output
]
colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#9E9E9E','#607D8B','#F44336']

bars = ax.bar(methods, means, color=colors)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Damage Probability', fontsize=12)
ax.set_title('Experiment v2: Damage Probability by Method\n(Mariupol, 4-sensor fusion)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
fig.savefig(OUT / 'exp_v2_method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → exp_v2_method_comparison.png')

print('\n可视化完成。')
