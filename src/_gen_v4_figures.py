"""Generate all figures for paper v4 — final visual polish edition."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

FIG_DIR = Path('paper/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 250

# =====================================================
# Fig 1: Workflow — enlarged, taller, all text readable
# =====================================================
fig, ax = plt.subplots(figsize=(20, 6.5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 7)
ax.axis('off')

arrow_props = dict(arrowstyle='->', color='#555', lw=3, connectionstyle='arc3,rad=0')

nodes = [
    (1.5, 4.5, 'A. Sensor\nDamage Scores', '#2196F3',
     'S2 spectral index\nVIIRS NTL loss'),
    (4.5, 4.5, 'B. Sensor\nDisagreement', '#FF9800',
     'r = -0.33\nNon-equivalent signals'),
    (7.5, 4.5, 'C1. Bayesian\nPosterior Collapse', '#4CAF50',
     'tau = 0.063\np_i in [0.151, 0.157]'),
    (10.5, 4.5, 'C2. KLD / JSD\nFusion Suppression', '#9C27B0',
     '46.8% suppression\nJSD = 0.146 nats'),
    (13.5, 4.5, 'C3. Bayesian\nvs D-S Comparison', '#F44336',
     '49.9% genuine\n28.1% divergent'),
    (16.5, 4.5, 'D. Multi-Dim.\nDamage Model', '#00897B',
     'Structural vs\nFunctional damage'),
]

for x, y, title, color, detail in nodes:
    box = mpatches.FancyBboxPatch((x-1.2, y-1.6), 2.4, 3.2,
                                   boxstyle='round,pad=0.15',
                                   facecolor=color, edgecolor='#333',
                                   alpha=0.12, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x, y+1.3, title, ha='center', fontsize=10.5, fontweight='bold', color='#222')
    ax.text(x, y-0.6, detail, ha='center', fontsize=9, color='#444', linespacing=1.4)

for i in range(len(nodes)-1):
    x1 = nodes[i][0] + 1.2
    x2 = nodes[i+1][0] - 1.2
    ax.annotate('', xy=(x2, 4.5), xytext=(x1, 4.5), arrowprops=arrow_props)

# UNOSAT annotation
ax.text(10, 6.4, 'UNOSAT External Validation (AUC: VIIRS = 0.829, S2 = 0.332)',
        ha='center', fontsize=11.5, fontweight='bold', color='#666',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#FF9800', alpha=0.8, linewidth=1.5))
ax.annotate('', xy=(4.5, 5.0), xytext=(10, 6.1),
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2, connectionstyle='arc3,rad=-0.25'))

ax.text(10, 0.3, 'Evidence Chain: A  ->  B  ->  C1  ->  C2  ->  C3  ->  D',
        ha='center', fontsize=13, fontweight='bold', color='#333')

plt.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / 'fig1_workflow_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 1 done')

# =====================================================
# Fig 2: Sensor Disagreement — larger, clearer maps
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
for ax in axes:
    ax.axis('off')

img_a = mpimg.imread('outputs/figures/s2_main_damage_score.png')
axes[0].imshow(img_a, aspect='auto')
axes[0].set_title('A. S2 Spectral Damage Score', fontsize=13, fontweight='bold', pad=8)

img_b = mpimg.imread('outputs/figures/viirs_ntl_loss_rate.png')
axes[1].imshow(img_b, aspect='auto')
axes[1].set_title('B. VIIRS Nightlight Loss Rate', fontsize=13, fontweight='bold', pad=8)

img_c = mpimg.imread('outputs/figures/p_s2_vs_p_viirs_scatter.png')
axes[2].imshow(img_c, aspect='auto')
axes[2].set_title('C. p(S2) vs p(VIIRS): r = -0.33', fontsize=13, fontweight='bold', pad=8)

plt.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / 'fig2_sensor_disagreement_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 2 done')

# =====================================================
# Fig 4: Bayesian Collapse — full page width, full height
# =====================================================
fig = plt.figure(figsize=(24, 8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.85])
for i, (idx, img_path, title) in enumerate([
    (0, 'outputs/figures/bayesian_posterior_map.png',
     'A. Joint Posterior: p_i in [0.151, 0.157]'),
    (1, 'outputs/figures/sensor_only_comparison.png',
     'B. Sensor-Only vs Joint: Each sensor alone retains spatial structure'),
    (2, 'outputs/figures/bayesian_sensor_precision.png',
     'C. Sensor Noise: sigma_S2=2.685, sigma_VIIRS=0.904'),
]):
    ax = fig.add_subplot(gs[i])
    img = mpimg.imread(img_path)
    ax.imshow(img, aspect='auto')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
    ax.axis('off')

plt.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / 'fig4_bayesian_collapse_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 4 done')

# =====================================================
# Fig 5a: KLD barplot — keep as is, already clean
# =====================================================
fig, ax = plt.subplots(figsize=(15, 5.5))
img_bar = mpimg.imread('outputs/figures/kld_summary_barplot.png')
ax.imshow(img_bar, aspect='auto')
ax.set_title('Figure 5a: KLD Information Gain Decomposition', fontsize=14, fontweight='bold', pad=6)
ax.axis('off')
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig5a_kld_summary_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 5a done')

# =====================================================
# Fig 5b: JSD + Suppression maps — larger
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
for ax in axes:
    ax.axis('off')

img_jsd = mpimg.imread('outputs/figures/kld_conflict_jsd_map.png')
axes[0].imshow(img_jsd, aspect='auto')
axes[0].set_title('B. JSD(S2, VIIRS): Mean = 0.146 nats\nAll 1,380 grids exceed 0.05 nats',
                  fontsize=13, fontweight='bold', pad=6)

img_supp = mpimg.imread('outputs/figures/kld_fusion_suppression_map.png')
axes[1].imshow(img_supp, aspect='auto')
axes[1].set_title('C. Fusion Suppression Ratio: Mean = 46.8%\nSpatially pervasive across the entire AOI',
                  fontsize=13, fontweight='bold', pad=6)

plt.tight_layout(pad=0.4)
fig.savefig(FIG_DIR / 'fig5b_kld_maps_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 5b done')

# =====================================================
# Fig 6: Bayesian vs D-S — consistency map + interval comparison
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(24, 10))
for ax in axes:
    ax.axis('off')

img_cons = mpimg.imread('outputs/figures/bayes_vs_ds_consistency_map.png')
axes[0].imshow(img_cons, aspect='auto')
axes[0].set_title('A. Consistency Map: 49.9% Genuine | 10.2% Uncertainty-Compatible | 28.1% Divergent',
                  fontsize=12, fontweight='bold', pad=6)

img_int = mpimg.imread('outputs/figures/bayes_vs_ds_interval_comparison.png')
axes[1].imshow(img_int, aspect='auto')
axes[1].set_title('B. Bayesian 95% CI vs D-S [Belief, Plausibility] — Top 50 Highest JSD Grids',
                  fontsize=12, fontweight='bold', pad=6)

plt.tight_layout(pad=0.4)
fig.savefig(FIG_DIR / 'fig6_bayes_ds_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 6 done')

# =====================================================
# Fig 7: UNOSAT Validation — larger
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(22, 7))
for ax in axes:
    ax.axis('off')

img_roc = mpimg.imread('outputs/figures/roc_curves_unosat_strict.png')
axes[0].imshow(img_roc, aspect='auto')
axes[0].set_title('A. ROC Curves (UNOSAT Strict Labels, n = 995)', fontsize=13, fontweight='bold', pad=6)

img_auc = mpimg.imread('outputs/figures/model_auc_comparison_strict.png')
axes[1].imshow(img_auc, aspect='auto')
axes[1].set_title('B. AUC Comparison', fontsize=13, fontweight='bold', pad=6)

plt.tight_layout(pad=0.4)
fig.savefig(FIG_DIR / 'fig7_unosat_validation_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 7 (UNOSAT) done')

# =====================================================
# Fig 8: Conceptual Model — enlarged, less whitespace
# =====================================================
fig, ax = plt.subplots(figsize=(18, 7.5))
ax.set_xlim(0, 18)
ax.set_ylim(0, 8)
ax.axis('off')

s2_c = '#2196F3'; viirs_c = '#FF9800'; conflict_c = '#F44336'; future_c = '#4CAF50'

ax.text(9, 7.7, 'Conceptual Model: Multi-Sensor Damage Assessment Framework',
        ha='center', fontsize=17, fontweight='bold')

# Sensors
s2_b = mpatches.FancyBboxPatch((1, 5.8), 5.5, 1.2, boxstyle='round,pad=0.1',
                                 facecolor=s2_c, edgecolor='#1565C0', alpha=0.15, linewidth=2.5)
ax.add_patch(s2_b)
ax.text(3.75, 6.7, 'Sentinel-2', ha='center', fontsize=15, fontweight='bold', color='#1565C0')
ax.text(3.75, 6.2, 'Spectral Indices (NDVI, NBR, ...)', ha='center', fontsize=9.5, color='#333')
ax.text(3.75, 5.95, 'Pre / Post Difference', ha='center', fontsize=9.5, color='#333')

v_b = mpatches.FancyBboxPatch((11.5, 5.8), 5.5, 1.2, boxstyle='round,pad=0.1',
                                facecolor=viirs_c, edgecolor='#E65100', alpha=0.15, linewidth=2.5)
ax.add_patch(v_b)
ax.text(14.25, 6.7, 'VIIRS DNB', ha='center', fontsize=15, fontweight='bold', color='#E65100')
ax.text(14.25, 6.2, 'Nighttime Light Radiance', ha='center', fontsize=9.5, color='#333')
ax.text(14.25, 5.95, 'NTL Loss Rate', ha='center', fontsize=9.5, color='#333')

# Damage dimensions
d1 = mpatches.FancyBboxPatch((1, 4.3), 5.5, 0.9, boxstyle='round,pad=0.05',
                               facecolor=s2_c, edgecolor='#1565C0', alpha=0.3, linewidth=2)
ax.add_patch(d1)
ax.text(3.75, 4.75, 'Structural / Surface Alteration', ha='center', fontsize=12, fontweight='bold', color='#1565C0')

d2 = mpatches.FancyBboxPatch((11.5, 4.3), 5.5, 0.9, boxstyle='round,pad=0.05',
                               facecolor=viirs_c, edgecolor='#E65100', alpha=0.3, linewidth=2)
ax.add_patch(d2)
ax.text(14.25, 4.75, 'Functional / Activity Loss', ha='center', fontsize=12, fontweight='bold', color='#E65100')

# Converging arrows
ax.annotate('', xy=(9, 3.7), xytext=(4.5, 4.2),
            arrowprops=dict(arrowstyle='->', color='#666', lw=2, connectionstyle='arc3,rad=0.3'))
ax.annotate('', xy=(9, 3.7), xytext=(13.5, 4.2),
            arrowprops=dict(arrowstyle='->', color='#666', lw=2, connectionstyle='arc3,rad=-0.3'))

# Conflict box
cf = mpatches.FancyBboxPatch((3, 2.3), 12, 1.3, boxstyle='round,pad=0.1',
                               facecolor=conflict_c, edgecolor='#B71C1C', alpha=0.12, linewidth=2.5)
ax.add_patch(cf)
ax.text(9, 3.3, 'Forced into Single Latent Damage State', ha='center', fontsize=14, fontweight='bold', color='#B71C1C')
ax.text(9, 2.85, 'Systematic Conflict: JSD = 0.146 nats  |  Posterior Collapse: tau = 0.063  |  Fusion Suppression: 46.8%',
        ha='center', fontsize=10.5, color='#C62828')

# Two paradigms
ax.text(4.5, 1.85, 'Bayesian Hierarchical Model', ha='center', fontsize=11, fontweight='bold', color='#1565C0')
ax.text(4.5, 1.45, 'Conservative unified posterior, p_i near 0.155', ha='center', fontsize=8.5, color='#555')

ax.text(13.5, 1.85, 'Dempster-Shafer Theory', ha='center', fontsize=11, fontweight='bold', color='#E65100')
ax.text(13.5, 1.45, 'Belief / Plausibility intervals: Bel = 0.145, Pl = 0.369', ha='center', fontsize=8.5, color='#555')

# Future
future_b = mpatches.FancyBboxPatch((3, 0.15), 12, 0.85, boxstyle='round,pad=0.1',
                                     facecolor=future_c, edgecolor='#1B5E20', alpha=0.15, linewidth=2.5)
ax.add_patch(future_b)
ax.text(9, 0.7, 'Future: Multi-Dimensional Latent Damage Model', ha='center', fontsize=14, fontweight='bold', color='#1B5E20')
ax.text(9, 0.35, 'Structural Damage  <->  Functional Loss: correlated but distinct latent dimensions',
        ha='center', fontsize=9.5, color='#2E7D32')

plt.tight_layout(pad=0.2)
fig.savefig(FIG_DIR / 'fig8_conceptual_model_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 8 done')

# =====================================================
# Appendix: scatter plots
# =====================================================
fig, ax = plt.subplots(figsize=(14, 6))
img_scat = mpimg.imread('outputs/figures/bayes_ds_jsd_conflict_scatter.png')
ax.imshow(img_scat, aspect='auto')
ax.set_title('JSD vs Bayesian-DS Divergence (r = -0.35) and D-S Width (r = -0.24)', fontsize=12, fontweight='bold')
ax.axis('off')
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_appendix_scatter_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Appendix scatter done')

# Appendix: PR curves
fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
for ax in axes:
    ax.axis('off')
img_pr = mpimg.imread('outputs/figures/pr_curves_unosat_strict.png')
axes[0].imshow(img_pr, aspect='auto')
axes[0].set_title('A. PR Curves (UNOSAT Strict)', fontsize=13, fontweight='bold', pad=6)
img_vs = mpimg.imread('outputs/figures/validation_score_distributions_strict.png')
axes[1].imshow(img_vs, aspect='auto')
axes[1].set_title('B. Score Distributions by UNOSAT Label', fontsize=13, fontweight='bold', pad=6)
plt.tight_layout(pad=0.4)
fig.savefig(FIG_DIR / 'fig_appendix_pr_v4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Appendix PR done')

print('\nAll v4 figures generated successfully!')
