"""Generate improved figures for paper v3."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

FIG_DIR = Path('paper/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# NEW Fig 1: Overall evidence-chain workflow
# =====================================================
fig, ax = plt.subplots(figsize=(18, 5))
ax.set_xlim(0, 18)
ax.set_ylim(0, 5.5)
ax.axis('off')

box_style = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#333', linewidth=2, alpha=0.95)
arrow_props = dict(arrowstyle='->', color='#555', lw=2.5, connectionstyle='arc3,rad=0')

# Colors per phase
c_data = '#2196F3'
c_disc = '#FF9800'
c_bayes = '#4CAF50'
c_kld = '#9C27B0'
c_ds = '#F44336'
c_conc = '#00897B'

# Row: 7 nodes, equally spaced
nodes = [
    (1.2, 3.8, 'A. Sensor\nDamage Scores', c_data,
     'S2 spectral index\nVIIRS NTL loss'),
    (3.7, 3.8, 'B. Sensor\nDisagreement', c_disc,
     'r = -0.33\nNon-equivalent signals'),
    (6.2, 3.8, 'C1. Bayesian\nPosterior Collapse', c_bayes,
     'tau=0.063\npi in [0.151,0.157]'),
    (8.7, 3.8, 'C2. KLD/JSD\nFusion Suppression', c_kld,
     '46.8% suppression\nJSD=0.146 nats'),
    (11.2, 3.8, 'C3. Bayesian\nvs D-S Comparison', c_ds,
     '49.9% genuine\n28.1% divergent'),
    (13.7, 3.8, 'D. Multi-Dim\nDamage Model', c_conc,
     'Structural vs\nFunctional damage'),
]

for x, y, title, color, detail in nodes:
    box = mpatches.FancyBboxPatch((x-0.95, y-1.3), 1.9, 2.6,
                                   boxstyle='round,pad=0.1',
                                   facecolor=color, edgecolor='#333',
                                   alpha=0.12, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y+1.05, title, ha='center', fontsize=9, fontweight='bold', color='#222')
    ax.text(x, y-0.7, detail, ha='center', fontsize=7.5, color='#444',
            linespacing=1.3)

# Arrows between nodes
for i in range(len(nodes)-1):
    x1 = nodes[i][0] + 0.95
    x2 = nodes[i+1][0] - 0.95
    ax.annotate('', xy=(x2, 3.8), xytext=(x1, 3.8),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2.5))

# Top annotation
ax.text(7.35, 5.2, 'UNOSAT External Validation (AUC: VIIRS=0.829, S2=0.332)',
        ha='center', fontsize=10, fontweight='bold', color='#666',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', edgecolor='#FF9800', alpha=0.7))
ax.annotate('', xy=(3.7, 4.2), xytext=(7.35, 5.0),
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5, connectionstyle='arc3,rad=-0.3'))

# Bottom annotation
ax.text(7.35, 0.2, 'Evidence Chain: A -> B -> C1 -> C2 -> C3 -> D',
        ha='center', fontsize=12, fontweight='bold', color='#333')

plt.tight_layout(pad=0.2)
fig.savefig(FIG_DIR / 'fig1_workflow.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 1 (workflow) done')

# =====================================================
# IMPROVED Fig 4: Bayesian Collapse (larger, better layout)
# =====================================================
fig = plt.figure(figsize=(22, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 0.9])

ax0 = fig.add_subplot(gs[0])
img0 = mpimg.imread('outputs/figures/bayesian_posterior_map.png')
ax0.imshow(img0)
ax0.set_title('A. Joint Posterior: pi in [0.151, 0.157]', fontsize=11, fontweight='bold')
ax0.axis('off')

ax1 = fig.add_subplot(gs[1])
img1 = mpimg.imread('outputs/figures/sensor_only_comparison.png')
ax1.imshow(img1)
ax1.set_title('B. Sensor-Only vs Both: Each retains spatial structure alone', fontsize=11, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[2])
img2 = mpimg.imread('outputs/figures/bayesian_sensor_precision.png')
ax2.imshow(img2)
ax2.set_title('C. Sensor Noise: sigma_S2=2.685, sigma_VIIRS=0.904', fontsize=11, fontweight='bold')
ax2.axis('off')

plt.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / 'fig4_bayesian_collapse_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 4 (Bayesian collapse) done')

# =====================================================
# IMPROVED Fig 5a: KLD summary (large barplot)
# =====================================================
fig, ax = plt.subplots(figsize=(14, 5))
img_bar = mpimg.imread('outputs/figures/kld_summary_barplot.png')
ax.imshow(img_bar)
ax.set_title('Figure 5a: KLD Information Gain Summary', fontsize=13, fontweight='bold')
ax.axis('off')
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig5a_kld_summary_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 5a (KLD summary) done')

# =====================================================
# IMPROVED Fig 5b: JSD conflict + Fusion suppression maps (side by side)
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))

img_jsd = mpimg.imread('outputs/figures/kld_conflict_jsd_map.png')
axes[0].imshow(img_jsd)
axes[0].set_title('B. JSD(S2, VIIRS): Mean 0.146 nats\nAll 1,380 grids > 0.05 nats', fontsize=12, fontweight='bold')
axes[0].axis('off')

img_supp = mpimg.imread('outputs/figures/kld_fusion_suppression_map.png')
axes[1].imshow(img_supp)
axes[1].set_title('C. Fusion Suppression Ratio: Mean 46.8%\nSpatially pervasive across the AOI', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / 'fig5b_kld_maps_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 5b (KLD maps) done')

# =====================================================
# IMPROVED Fig 6: Bayesian vs D-S (larger panels)
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

img_cons = mpimg.imread('outputs/figures/bayes_vs_ds_consistency_map.png')
axes[0].imshow(img_cons)
axes[0].set_title('A. Bayesian vs D-S Consistency Map\n49.9% Genuine | 10.2% Uncertainty-Compatible | 28.1% Divergent',
                  fontsize=11, fontweight='bold')
axes[0].axis('off')

img_int = mpimg.imread('outputs/figures/bayes_vs_ds_interval_comparison.png')
axes[1].imshow(img_int)
axes[1].set_title('B. Bayesian 95% CI vs D-S [Belief, Plausibility]\nTop 50 highest JSD grids',
                  fontsize=11, fontweight='bold')
axes[1].axis('off')

plt.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / 'fig6_bayes_ds_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 6 (Bayes vs DS) done')

# =====================================================
# IMPROVED Fig 2: Sensor Disagreement (keep same, verify size)
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
img_a = mpimg.imread('outputs/figures/s2_main_damage_score.png')
axes[0].imshow(img_a)
axes[0].set_title('A. S2 Spectral Damage Score', fontsize=12, fontweight='bold')
axes[0].axis('off')
img_b = mpimg.imread('outputs/figures/viirs_ntl_loss_rate.png')
axes[1].imshow(img_b)
axes[1].set_title('B. VIIRS Nightlight Loss Rate', fontsize=12, fontweight='bold')
axes[1].axis('off')
img_c = mpimg.imread('outputs/figures/p_s2_vs_p_viirs_scatter.png')
axes[2].imshow(img_c)
axes[2].set_title('C. p(S2) vs p(VIIRS): r = -0.33', fontsize=12, fontweight='bold')
axes[2].axis('off')
plt.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / 'fig2_sensor_disagreement_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 2 (sensor disagreement) done')

# =====================================================
# IMPROVED Fig 3: UNOSAT Validation
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
img_roc = mpimg.imread('outputs/figures/roc_curves_unosat_strict.png')
axes[0].imshow(img_roc)
axes[0].set_title('A. ROC Curves (UNOSAT Strict Labels, n=995)', fontsize=12, fontweight='bold')
axes[0].axis('off')
img_auc = mpimg.imread('outputs/figures/model_auc_comparison_strict.png')
axes[1].imshow(img_auc)
axes[1].set_title('B. AUC Comparison', fontsize=12, fontweight='bold')
axes[1].axis('off')
plt.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / 'fig3_unosat_validation_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 3 (UNOSAT) done')

print('\nAll v3 figures generated!')
