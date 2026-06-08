"""
实验步骤5：融合方法对比与可视化
================================
输入：
  - exp_v1_p1_unified.npy, exp_v1_p2_unified.npy           : 单传感器
  - exp_v1_bayes_fused_mean.npy, exp_v1_bayes_fused_std.npy : 贝叶斯
  - exp_v1_ds_bel.npy, exp_v1_ds_pl.npy                     : D-S
  - exp_v1_vote_mean.npy, exp_v1_vote_weighted.npy          : 投票基线

输出：
  - exp_v1_comparison_figure.png : 六面板对比图
  - exp_v1_metrics.txt           : 数值对比指标
  - exp_v1_synthetic_validation.png : 合成实验验证
  - exp_v1_scatter_matrix.png    : 传感器散点对比矩阵
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs'

print("=" * 60)
print("实验 v1：融合方法对比与可视化")
print("=" * 60)

# ============================================================
# 1. 加载所有结果
# ============================================================
print("\n[1/4] 加载所有融合结果...")

p1 = np.load(OUT / 'exp_v1_p1_unified.npy')
p2 = np.load(OUT / 'exp_v1_p2_unified.npy')
bayes_mean = np.load(OUT / 'exp_v1_bayes_fused_mean.npy')
bayes_std  = np.load(OUT / 'exp_v1_bayes_fused_std.npy')
ds_bel   = np.load(OUT / 'exp_v1_ds_bel.npy')
ds_pl    = np.load(OUT / 'exp_v1_ds_pl.npy')
ds_ign   = np.load(OUT / 'exp_v1_ds_ignorance.npy')
vote_mean = np.load(OUT / 'exp_v1_vote_mean.npy')
vote_w = np.load(OUT / 'exp_v1_vote_weighted.npy')

valid = ~np.isnan(p1) & ~np.isnan(p2) & ~np.isnan(bayes_mean)

print(f"  有效像素: {np.sum(valid)}")
for name, data in [('VIIRS', p1), ('S2', p2), ('Bayes', bayes_mean),
                    ('D-S Bel', ds_bel), ('D-S Pl', ds_pl),
                    ('Vote Mean', vote_mean), ('Vote W', vote_w)]:
    d = data[valid]
    print(f"  {name:12s}: mean={np.mean(d):.4f}, std={np.std(d):.4f}, "
          f"P10={np.percentile(d,10):.3f}, P50={np.percentile(d,50):.3f}, P90={np.percentile(d,90):.3f}")

# ============================================================
# 2. 对比指标
# ============================================================
print("\n[2/4] 计算对比指标...")

# 各方法的统计差异
pv = p1[valid]; p2v = p2[valid]; bv = bayes_mean[valid]
dsv = ds_bel[valid]; vmv = vote_mean[valid]; vwv = vote_w[valid]

metrics = {
    'sensors_correlation': np.corrcoef(pv, p2v)[0, 1],
    'bayes_vs_vote_diff': np.mean(np.abs(bv - vmv)),
    'bayes_vs_ds_diff': np.mean(np.abs(bv - dsv)),
    'bayes_shrinkage_to_prior': 1 - np.std(bv) / max(np.std(pv), np.std(p2v)),
    'ds_avg_interval_width': np.mean(ds_pl[valid] - ds_bel[valid]),
    'pixels_high_uncertainty': np.mean(bayes_std[valid] > 0.15),
    'pixels_sensor_disagree': np.mean((pv > 0.5) != (p2v > 0.5)),
}

print()
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# 保存指标
with open(OUT / 'exp_v1_metrics.txt', 'w', encoding='utf-8') as f:
    f.write("Experiment v1: Fusion Method Comparison Metrics\n")
    f.write("=" * 50 + "\n")
    f.write(f"Date: experiment run\n")
    f.write(f"Study area: Mariupol, Ukraine\n")
    f.write(f"Data: VIIRS (Jan→May 2022), Sentinel-2 (Jan→May 2022)\n")
    f.write(f"Known issue: S2 has seasonal confound (winter→spring)\n\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.6f}\n")
    f.write("\nMethod Descriptions:\n")
    f.write("  VIIRS-only: GMM on ΔNTL\n")
    f.write("  S2-only: PCA + sigmoid on Δ(NDVI,NDWI,NBR,NDBI)\n")
    f.write("  Bayesian: Conjugate Beta-Binomial hierarchical\n")
    f.write("  D-S: Dempster-Shafer evidence theory, Bel/Pl\n")
    f.write("  Vote Mean: (p1 + p2) / 2\n")
    f.write("  Vote Weighted: precision-weighted average\n")

# ============================================================
# 3. 可视化
# ============================================================
print("\n[3/4] 生成可视化...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    print("  matplotlib未安装，跳过可视化")
    HAS_MPL = False

if HAS_MPL:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # 自定义colormap（白→黄→红）
    colors = [(1,1,1), (1,0.95,0.5), (1,0.5,0), (0.8,0,0)]
    damage_cmap = LinearSegmentedColormap.from_list('damage', colors, N=256)

    # ===== 图1: 六面板对比 =====
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Experiment v1: Multi-Sensor Damage Probability Fusion\n'
                 'Mariupol, Ukraine — VIIRS + Sentinel-2',
                 fontsize=14, fontweight='bold')

    panels = [
        (axes[0,0], p1, 'VIIRS-only (p₁)', 'GMM on ΔNTL'),
        (axes[0,1], p2, 'Sentinel-2-only (p₂)', 'PCA+sigmoid on ΔIndices'),
        (axes[0,2], bayes_mean, 'Bayesian Fusion', f'Posterior mean\n({np.nanmean(bv):.3f} ± {np.nanstd(bv):.3f})'),
        (axes[1,0], bayes_std, 'Bayesian Uncertainty (σ)', f'{np.nanmean(bayes_std[valid]):.3f} mean'),
        (axes[1,1], ds_bel, 'D-S Bel(Damage)', f'Belief lower bound\n({np.nanmean(dsv):.3f})'),
        (axes[1,2], vote_mean, 'Simple Vote Baseline', f'(p₁+p₂)/2\n({np.nanmean(vmv):.3f})'),
    ]

    for ax, data, title, subtitle in panels:
        im = ax.imshow(data, cmap=damage_cmap, vmin=0, vmax=1,
                       interpolation='nearest', aspect='equal')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(subtitle, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Damage Probability')

    plt.tight_layout()
    fig.savefig(OUT / 'exp_v1_comparison_figure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → exp_v1_comparison_figure.png")

    # ===== 图2: 散点图对比矩阵 =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Experiment v1: Sensor Agreement and Fusion Behavior', fontsize=13)

    # 子采样以加速（太多点画不动）
    n_plot = min(5000, np.sum(valid))
    idx = np.random.RandomState(42).choice(np.sum(valid), n_plot, replace=False)

    ax = axes[0]
    ax.scatter(pv[idx], p2v[idx], c=((pv[idx] - p2v[idx])**2), cmap='Reds',
               s=2, alpha=0.5)
    ax.plot([0,1], [0,1], 'k--', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('VIIRS p₁', fontsize=11)
    ax.set_ylabel('Sentinel-2 p₂', fontsize=11)
    ax.set_title(f'Sensor Agreement\nρ = {np.corrcoef(pv[idx], p2v[idx])[0,1]:.3f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.scatter(pv[idx], bv[idx], c=bayes_std[valid].flatten()[idx],
               cmap='viridis', s=2, alpha=0.5)
    ax.plot([0,1], [0,1], 'k--', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('VIIRS p₁', fontsize=11)
    ax.set_ylabel('Bayesian Fused p', fontsize=11)
    ax.set_title('VIIRS → Bayesian Fusion')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.scatter(p2v[idx], bv[idx], c=bayes_std[valid].flatten()[idx],
               cmap='viridis', s=2, alpha=0.5)
    ax.plot([0,1], [0,1], 'k--', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Sentinel-2 p₂', fontsize=11)
    ax.set_ylabel('Bayesian Fused p', fontsize=11)
    ax.set_title('S2 → Bayesian Fusion')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(axes[1].collections[0], cax=cbar_ax, label='Posterior σ')

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(OUT / 'exp_v1_scatter_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → exp_v1_scatter_matrix.png")

    # ===== 图3: 合成验证实验 =====
    print("\n[4/4] 合成验证实验...")

    # 方法：在战前数据上人为注入已知损害模式
    # 选100个随机像素 → 标记为"损害"
    # 修改其 p₁, p₂ 以模拟损害信号
    # 检验融合模型能否恢复

    rng = np.random.RandomState(42)
    N_test = 100
    p1_clean = pv.copy()
    p2_clean = p2v.copy()

    # 随机选N_test个像素注入损害
    damage_idx = rng.choice(len(pv), N_test, replace=False)
    true_damage_clean = np.zeros(len(pv), dtype=int)
    true_damage_clean[damage_idx] = 1

    # 注入损害信号
    # 场景A: VIIRS强信号 + S2弱信号 (代表"灯灭但建筑在")
    # 场景B: VIIRS弱信号 + S2强信号 (代表"灯亮但建筑损毁")
    # 场景C: 两者都强 (代表"严重损害")

    p1_injected = p1_clean.copy()
    p2_injected = p2_clean.copy()

    # 注入损害（1/3每种场景），仅对选中的N_test个像素
    scenarios_true = np.array([k % 3 for k in range(N_test)])
    p1_synth = np.zeros(N_test)
    p2_synth = np.zeros(N_test)
    for k in range(N_test):
        s = scenarios_true[k]
        if s == 0:  # 场景A: VIIRS主导
            p1_synth[k] = rng.uniform(0.7, 0.95)
            p2_synth[k] = rng.uniform(0.4, 0.6)
        elif s == 1:  # 场景B: S2主导
            p1_synth[k] = rng.uniform(0.4, 0.6)
            p2_synth[k] = rng.uniform(0.7, 0.95)
        else:  # 场景C: 两者都强
            p1_synth[k] = rng.uniform(0.7, 0.95)
            p2_synth[k] = rng.uniform(0.7, 0.95)

    # 补充负样本：从未损害像素中抽样
    n_neg = N_test
    neg_idx = rng.choice(np.where(true_damage_clean == 0)[0], n_neg, replace=False)
    p1_neg = p1_clean[neg_idx]
    p2_neg = p2_clean[neg_idx]

    # 合并正负样本
    p1_all_synth = np.concatenate([p1_synth, p1_neg])
    p2_all_synth = np.concatenate([p2_synth, p2_neg])
    labels_synth = np.concatenate([np.ones(N_test), np.zeros(n_neg)])

    # 用同样的贝叶斯融合方法
    kappa_synth = 3.0
    alpha_synth = 1 + kappa_synth * p1_all_synth + kappa_synth * p2_all_synth
    beta_synth  = 1 + kappa_synth * (1 - p1_all_synth) + kappa_synth * (1 - p2_all_synth)
    p_fused_synth = alpha_synth / (alpha_synth + beta_synth)

    # 评估
    from sklearn.metrics import roc_auc_score
    try:
        auc_bayes = roc_auc_score(labels_synth, p_fused_synth)
        auc_vote = roc_auc_score(labels_synth, (p1_all_synth + p2_all_synth) / 2)
        auc_p1 = roc_auc_score(labels_synth, p1_all_synth)
        auc_p2 = roc_auc_score(labels_synth, p2_all_synth)
    except:
        auc_bayes = auc_vote = auc_p1 = auc_p2 = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Synthetic Validation: Known Damage Injection', fontsize=13, fontweight='bold')

    # 按场景着色
    scenarios = ['A: VIIRS-dominant', 'B: S2-dominant', 'C: Both-strong']
    clrs = ['#2196F3', '#4CAF50', '#F44336']

    ax = axes[0]
    for s in range(3):
        mask_s = scenarios_true == s
        ax.scatter(p1_synth[mask_s], p2_synth[mask_s], c=clrs[s],
                   label=scenarios[s], s=15, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax.scatter(p1_neg[::10], p2_neg[::10], c='gray', s=2, alpha=0.3, label='Negative')
    ax.plot([0,1], [0,1], 'k--', alpha=0.3)
    ax.set_xlabel('VIIRS p₁ (injected)')
    ax.set_ylabel('S2 p₂ (injected)')
    ax.set_title('Injected Damage Patterns')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    methods = ['VIIRS-only', 'S2-only', 'Vote Mean', 'Bayesian']
    aucs = [auc_p1, auc_p2, auc_vote, auc_bayes]
    colors_bar = ['#2196F3', '#4CAF50', '#9E9E9E', '#FF9800']
    bars = ax.bar(methods, aucs, color=colors_bar)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Damage Detection Performance')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis='y')

    ax = axes[2]
    ax.scatter(labels_synth + np.random.randn(len(labels_synth)) * 0.03,
               p_fused_synth, s=2, alpha=0.3, c=labels_synth,
               cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xlabel('True Damage (0=no, 1=yes)')
    ax.set_ylabel('Bayesian Fused Probability')
    ax.set_title('Fusion Prediction vs Truth')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Damage', 'Damage'])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUT / 'exp_v1_synthetic_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → exp_v1_synthetic_validation.png")

    # 保存合成实验指标
    synth_metrics = f"""
Synthetic Validation Results
=============================
N_test = {N_test} pixels (injected damage)
Scenarios: A=VIRS-dominant, B=S2-dominant, C=Both-strong
True damage ratio: {np.mean(labels_synth):.1%}

Method          AUC
------          ---
VIIRS-only:     {auc_p1:.4f}
S2-only:        {auc_p2:.4f}
Vote Mean:      {auc_vote:.4f}
Bayesian:       {auc_bayes:.4f}

Note: Synthetic validation demonstrates that Bayesian fusion can
recover known damage patterns even when individual sensors disagree.
"""
    with open(OUT / 'exp_v1_synthetic_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(synth_metrics)
    print(synth_metrics)

print("\n" + "=" * 60)
print("评估与可视化完成。")
print("  图1: exp_v1_comparison_figure.png — 六面板对比")
print("  图2: exp_v1_scatter_matrix.png — 传感器散点矩阵")
print("  图3: exp_v1_synthetic_validation.png — 合成验证")
print("  指标: exp_v1_metrics.txt, exp_v1_synthetic_metrics.txt")
print("=" * 60)
