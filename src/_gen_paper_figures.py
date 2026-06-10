"""Generate 6 composite figures for paper v2."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pathlib import Path

FIG_DIR = Path('paper/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# Figure 5: Bayesian vs D-S Comparison (3 panels)
# ========================================
fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 2)

ax_a = fig.add_subplot(gs[:, 0])
img_cons = mpimg.imread('outputs/figures/bayes_vs_ds_consistency_map.png')
ax_a.imshow(img_cons)
ax_a.set_title('A. Bayesian vs D-S Consistency Map\n50% Genuine | 28% Divergent', fontsize=11, fontweight='bold')
ax_a.axis('off')

ax_b = fig.add_subplot(gs[0, 1])
img_int = mpimg.imread('outputs/figures/bayes_vs_ds_interval_comparison.png')
ax_b.imshow(img_int)
ax_b.set_title('B. Bayesian CI vs D-S [Belief, Plausibility]\nTop 50 Highest JSD Grids', fontsize=11, fontweight='bold')
ax_b.axis('off')

ax_c = fig.add_subplot(gs[1, 1])
img_scat = mpimg.imread('outputs/figures/bayes_ds_jsd_conflict_scatter.png')
ax_c.imshow(img_scat)
ax_c.set_title('C. JSD vs Divergence (r=-0.35) and D-S Width (r=-0.24)', fontsize=11, fontweight='bold')
ax_c.axis('off')

plt.tight_layout(pad=0.5)
fig.savefig(FIG_DIR / 'fig5_bayes_ds_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 5 done')

# ========================================
# Figure 6: Conceptual Model
# ========================================
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

s2_color = '#2196F3'
viirs_color = '#FF9800'
conflict_color = '#F44336'
future_color = '#4CAF50'

# Title
ax.text(8, 7.5, 'Conceptual Model: Multi-Sensor Damage Assessment Framework',
        ha='center', fontsize=16, fontweight='bold')

# ---- Row 1: Sensors ----
ax.text(8, 6.8, 'Observed Signals', ha='center', fontsize=10, color='gray', style='italic')

s2_box = mpatches.FancyBboxPatch((1, 5.5), 5, 1, boxstyle='round,pad=0.1',
                                  facecolor=s2_color, edgecolor='#1565C0', alpha=0.15, linewidth=2)
ax.add_patch(s2_box)
ax.text(3.5, 6.3, 'Sentinel-2', ha='center', fontsize=14, fontweight='bold', color='#1565C0')
ax.text(3.5, 5.9, 'Spectral Indices (NDVI, NBR, ...)', ha='center', fontsize=9, color='#333')
ax.text(3.5, 5.6, 'Pre/Post Difference', ha='center', fontsize=9, color='#333')

viirs_box = mpatches.FancyBboxPatch((10, 5.5), 5, 1, boxstyle='round,pad=0.1',
                                     facecolor=viirs_color, edgecolor='#E65100', alpha=0.15, linewidth=2)
ax.add_patch(viirs_box)
ax.text(12.5, 6.3, 'VIIRS DNB', ha='center', fontsize=14, fontweight='bold', color='#E65100')
ax.text(12.5, 5.9, 'Nighttime Light Radiance', ha='center', fontsize=9, color='#333')
ax.text(12.5, 5.6, 'NTL Loss Rate', ha='center', fontsize=9, color='#333')

# Arrows down
ax.annotate('', xy=(3.5, 5.4), xytext=(3.5, 5.3),
            arrowprops=dict(arrowstyle='->', color=s2_color, lw=2))
ax.annotate('', xy=(12.5, 5.4), xytext=(12.5, 5.3),
            arrowprops=dict(arrowstyle='->', color=viirs_color, lw=2))

# ---- Row 2: Damage Dimensions ----
ax.text(8, 5.1, 'Damage Dimensions Captured', ha='center', fontsize=10, color='gray', style='italic')

dim1 = mpatches.FancyBboxPatch((1, 4.3), 5, 0.7, boxstyle='round,pad=0.05',
                                 facecolor=s2_color, edgecolor='#1565C0', alpha=0.3, linewidth=2)
ax.add_patch(dim1)
ax.text(3.5, 4.65, 'Structural / Surface Alteration', ha='center', fontsize=11, fontweight='bold', color='#1565C0')

dim2 = mpatches.FancyBboxPatch((10, 4.3), 5, 0.7, boxstyle='round,pad=0.05',
                                 facecolor=viirs_color, edgecolor='#E65100', alpha=0.3, linewidth=2)
ax.add_patch(dim2)
ax.text(12.5, 4.65, 'Functional / Activity Loss', ha='center', fontsize=11, fontweight='bold', color='#E65100')

# Arrows converging to conflict
ax.annotate('', xy=(8, 3.8), xytext=(4, 4.2),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5, connectionstyle='arc3,rad=0.3'))
ax.annotate('', xy=(8, 3.8), xytext=(12, 4.2),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5, connectionstyle='arc3,rad=-0.3'))

# ---- Row 3: Single Latent State Conflict ----
conflict_box = mpatches.FancyBboxPatch((3, 2.6), 10, 1.1, boxstyle='round,pad=0.1',
                                        facecolor=conflict_color, edgecolor='#B71C1C', alpha=0.12, linewidth=2)
ax.add_patch(conflict_box)
ax.text(8, 3.5, 'Forced into Single Latent Damage State', ha='center', fontsize=13, fontweight='bold', color='#B71C1C')
ax.text(8, 3.1, 'Systematic Conflict: JSD = 0.146 nats', ha='center', fontsize=10, color='#C62828')
ax.text(8, 2.75, 'Posterior Collapse: tau=0.063 | Fusion Suppression: 46.8%', ha='center', fontsize=10, color='#C62828')

# Arrow down
ax.annotate('', xy=(8, 2.5), xytext=(8, 2.3),
            arrowprops=dict(arrowstyle='->', color='#666', lw=2))

# ---- Row 4: Two Paradigms ----
ax.text(3.5, 2.1, 'Bayesian Hierarchical Model', ha='center', fontsize=10, fontweight='bold', color='#1565C0')
ax.text(3.5, 1.85, 'Conservative unified posterior', ha='center', fontsize=8, color='#555')
ax.text(3.5, 1.65, 'p_i near 0.155 (spatially uniform)', ha='center', fontsize=8, color='#555')

ax.text(12.5, 2.1, 'Dempster-Shafer Theory', ha='center', fontsize=10, fontweight='bold', color='#E65100')
ax.text(12.5, 1.85, 'Belief/Plausibility intervals', ha='center', fontsize=8, color='#555')
ax.text(12.5, 1.65, 'Bel=0.145, Pl=0.369 (preserves uncertainty)', ha='center', fontsize=8, color='#555')

# Arrows to future
ax.annotate('', xy=(8, 0.9), xytext=(4, 1.5),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1, connectionstyle='arc3,rad=0.2'))
ax.annotate('', xy=(8, 0.9), xytext=(12, 1.5),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1, connectionstyle='arc3,rad=-0.2'))

# ---- Row 5: Future Direction ----
future_box = mpatches.FancyBboxPatch((3, 0.1), 10, 0.7, boxstyle='round,pad=0.1',
                                      facecolor=future_color, edgecolor='#1B5E20', alpha=0.15, linewidth=2)
ax.add_patch(future_box)
ax.text(8, 0.55, 'Future: Multi-Dimensional Latent Damage Model', ha='center', fontsize=13, fontweight='bold', color='#1B5E20')
ax.text(8, 0.25, 'Structural Damage and Functional Loss: correlated but distinct latent dimensions',
        ha='center', fontsize=9, color='#2E7D32')

plt.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / 'fig6_conceptual_model.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 6 done')

print('All 6 figures generated successfully!')
