"""
Generate publication-quality figures for ACML 2026 paper
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
import json, warnings
warnings.filterwarnings('ignore')

BASE = Path('d:/数理统计大作业数据')
OUT_V4 = BASE / 'exp_outputs_v4'
FIG_DIR = BASE / 'paper_figures'
FIG_DIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'figure.dpi': 300,
})

# Custom colormaps
damage_cmap = LinearSegmentedColormap.from_list('damage', [(1,1,1),(.95,.9,.6),(1,.6,.2),(.85,.1,.1)], N=256)
uncert_cmap = plt.cm.viridis

names = ['VIIRS','S2','SAR','InSAR']
P = {}
for n in names:
    fp = OUT_V4 / f'exp_v4_p_{n}.npy'
    if fp.exists(): P[n] = np.load(fp)
    else: P[n] = np.full((100,150), np.nan)

fused = np.load(OUT_V4/'exp_v4_bayes_fused_mean.npy')
fused_std = np.load(OUT_V4/'exp_v4_bayes_fused_std.npy')

# ============================================================
# Figure 1: Four-sensor damage probability + fusion (5 panels)
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
fig.suptitle('Damage Probability Maps from Four Sensors and Bayesian Fusion', fontweight='bold', fontsize=13)

titles = ['(a) VIIRS NTL', '(b) Sentinel-2 MSI', '(c) Sentinel-1 SAR', '(d) InSAR Coherence', '(e) Bayesian Fusion']
datas = [P['VIIRS'], P['S2'], P['SAR'], P['InSAR'], fused]
for ax, title, data in zip(axes, titles, datas):
    mu = np.nanmean(data)
    im = ax.imshow(data, cmap=damage_cmap, vmin=0, vmax=1, aspect='equal', interpolation='nearest')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(f'$\mu$={mu:.3f}', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.008, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Damage Probability')
plt.tight_layout(rect=[0,0,0.92,0.95])
fig.savefig(FIG_DIR/'fig1_damage_maps.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR/'fig1_damage_maps.png', dpi=200, bbox_inches='tight')
plt.close()
print('  -> fig1_damage_maps.pdf')

# ============================================================
# Figure 2: Fusion uncertainty + HDI
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
valid = ~np.isnan(fused)
hdi_w = np.load(OUT_V4/'exp_v4_bayes_hdi_high.npy') - np.load(OUT_V4/'exp_v4_bayes_hdi_low.npy')

ax=axes[0]; im=ax.imshow(fused, cmap=damage_cmap, vmin=0, vmax=1, aspect='equal', interpolation='nearest')
ax.set_title('(a) Posterior Mean', fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046)

ax=axes[1]; im=ax.imshow(fused_std, cmap=uncert_cmap, aspect='equal', interpolation='nearest')
ax.set_title('(b) Posterior Std $\sigma$', fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046)

ax=axes[2]; im=ax.imshow(hdi_w, cmap=uncert_cmap, aspect='equal', interpolation='nearest')
ax.set_title('(c) 95% HDI Width', fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
fig.savefig(FIG_DIR/'fig2_fusion_uncertainty.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR/'fig2_fusion_uncertainty.png', dpi=200, bbox_inches='tight')
plt.close()
print('  -> fig2_fusion_uncertainty.pdf')

# ============================================================
# Figure 3: Method comparison bar chart
# ============================================================
vall = ~np.isnan(P['VIIRS'])
for n in names[1:]: vall &= ~np.isnan(P[n])
if np.sum(vall)>0:
    p_arr = np.column_stack([P[n][vall] for n in names])
    methods = ['VIIRS','Sentinel-2','SAR','InSAR','Vote\nMean','Vote\nWeighted','Median','Bayesian\nFusion']
    vals = [np.mean(p_arr[:,k]) for k in range(4)]
    vals.append(np.mean(np.mean(p_arr,axis=1)))
    prec = 1.0/(0.03**2 * np.ones_like(p_arr))
    vals.append(np.mean(np.sum(p_arr*prec,axis=1)/np.sum(prec,axis=1)))
    vals.append(np.median(np.mean(p_arr,axis=1)))
    vals.append(np.nanmean(fused))

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#9E9E9E','#607D8B','#795548','#F44336']
bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor='white', linewidth=0.5)
for i, (bar, val) in enumerate(zip(bars, vals)):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{val:.3f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Mean Damage Probability', fontsize=11)
ax.set_title('Damage Probability Estimates by Method', fontweight='bold')
ax.set_ylim(0, 1.15); ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
fig.savefig(FIG_DIR/'fig3_method_comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR/'fig3_method_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print('  -> fig3_method_comparison.pdf')

# ============================================================
# Figure 4: Sensor agreement scatter
# ============================================================
if np.sum(vall)>0:
    pv = np.column_stack([P[n][vall].ravel() for n in names])
    n_plot = min(3000, len(pv))
    idx = np.random.RandomState(42).choice(len(pv), n_plot, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for (i,j), ax in zip(pairs, axes.flat):
        rho = np.corrcoef(pv[idx,i], pv[idx,j])[0,1]
        ax.scatter(pv[idx,i], pv[idx,j], s=1, alpha=0.3, c='#2196F3')
        ax.plot([0,1],[0,1],'k--',alpha=0.2,lw=0.5)
        ax.set_xlabel(names[i]); ax.set_ylabel(names[j])
        ax.set_title(f'$\\rho$={rho:.3f}'); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(True,alpha=0.2)
    plt.tight_layout()
    fig.savefig(FIG_DIR/'fig4_sensor_agreement.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR/'fig4_sensor_agreement.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  -> fig4_sensor_agreement.pdf')

# ============================================================
# Figure 5: v2->v4 improvement tracking
# ============================================================
versions = ['v2\n(baseline)','v3\n(+SAR calib\n+InSAR norm)','v4\n(+urban VIIRS\n+directional S2)']
vii_means = [0.031, 0.021, 0.002]
s2_means  = [0.579, 0.579, 0.433]
sar_means = [0.994, 0.006, 0.006]
ins_means = [0.220, 0.215, 0.215]
fuse_means = [0.462, 0.210, 0.176]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(versions)); w = 0.15
ax.bar(x-2*w, vii_means, w, label='VIIRS', color='#2196F3')
ax.bar(x-w, s2_means, w, label='Sentinel-2', color='#4CAF50')
ax.bar(x, sar_means, w, label='SAR', color='#FF9800')
ax.bar(x+w, ins_means, w, label='InSAR', color='#9C27B0')
ax.bar(x+2*w, fuse_means, w, label='Bayesian Fusion', color='#F44336')
ax.set_xticks(x); ax.set_xticklabels(versions)
ax.set_ylabel('Mean Damage Probability'); ax.set_title('Improvement Tracking Across Versions', fontweight='bold')
ax.legend(fontsize=8, ncol=5, loc='upper center')
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
fig.savefig(FIG_DIR/'fig5_version_tracking.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR/'fig5_version_tracking.png', dpi=200, bbox_inches='tight')
plt.close()
print('  -> fig5_version_tracking.pdf')

# Save summary stats for the paper
stats = {
    'p_viirs': float(np.nanmean(P['VIIRS'])),
    'p_s2': float(np.nanmean(P['S2'])),
    'p_sar': float(np.nanmean(P['SAR'])),
    'p_insar': float(np.nanmean(P['InSAR'])),
    'p_fused': float(np.nanmean(fused)),
    'std_fused': float(np.nanmean(fused_std)),
    'hdi_width': float(np.nanmean(hdi_w)),
    'corr_matrix': [[float(np.corrcoef(p_arr[:,i],p_arr[:,j])[0,1]) for j in range(4)] for i in range(4)],
    'dNDVI': -0.018, 'dNBR': -0.026, 'dNDBI': 0.016,
    'urban_pct': 0.8, 'urban_dNTL': -9.92,
    's2_damage_dir_pct': 42.7,
}
with open(FIG_DIR/'paper_stats.json','w') as f:
    json.dump(stats, f, indent=2)

print('\nAll figures saved to paper_figures/')
