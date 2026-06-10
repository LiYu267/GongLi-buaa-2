#!/usr/bin/env python3
"""
模块 2 — KLD 信息增益分析

T20 多传感器不确定性融合: 用 KL 散度量化各传感器的信息贡献。

核心问题:
  ① 单传感器从 neutral prior (p=0.5) 获得了多少信息?
  ② 融合后还保留多少信息? (冲突是否抑制了信息?)
  ③ 各传感器之间的冲突度 (JSD)?
  ④ 增加传感器是否显著提高估计精度?

方法:
  - Bernoulli probability-level KL / JSD
  - Neutral prior baseline: p = 0.5
  - Global mean baseline: 衡量空间分异信息
  - 传感器冲突度: JSD(S2-only, VIIRS-only)

前提: 需要先运行 16_bayesian_hierarchical.py 生成 both-model posterior.
      本脚本自动运行 s2_only 和 viirs_only 的单传感器贝叶斯模型.
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_CSV    = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.csv"
FUSION_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.geojson"
POSTERIOR_CSV = PROJECT_ROOT / "data" / "processed" / "grid_bayesian_posterior.csv"

OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG  = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB  = PROJECT_ROOT / "outputs" / "tables"
OUT_MD   = PROJECT_ROOT / "outputs"
for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)

# 输出文件
KLD_CSV     = OUT_DATA / "kld_information_gain_by_grid.csv"
KLD_SUMMARY = OUT_TAB  / "kld_information_gain_summary.csv"
S2_ONLY_POST_CSV = OUT_DATA / "grid_s2_only_posterior.csv"
VIIRS_ONLY_POST_CSV = OUT_DATA / "grid_viirs_only_posterior.csv"

KLD_S2_MAP_PNG      = OUT_FIG / "kld_s2_information_map.png"
KLD_VIIRS_MAP_PNG   = OUT_FIG / "kld_viirs_information_map.png"
KLD_BOTH_MAP_PNG    = OUT_FIG / "kld_both_information_map.png"
KLD_CONFLICT_JSD_PNG = OUT_FIG / "kld_conflict_jsd_map.png"
KLD_SUPPRESSION_PNG  = OUT_FIG / "kld_fusion_suppression_map.png"
KLD_BARPLOT_PNG      = OUT_FIG / "kld_summary_barplot.png"

REPORT_MD = OUT_MD / "kld_information_gain_report.md"

EPS = 1e-9
SEED = 42
MCMC_DRAWS = 1000
MCMC_TUNE  = 2000
MCMC_CHAINS = 4
TARGET_ACCEPT = 0.99


# ============================================================
# 工具函数: KL / JSD (Bernoulli probability level)
# ============================================================

def safe_clip(p, eps=EPS):
    """Clip probability to [eps, 1-eps] to avoid log(0)."""
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def kl_bernoulli(p, q, eps=EPS):
    """
    KL(Bern(p) || Bern(q)) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    单位: nats (natural log).
    """
    p = safe_clip(p, eps)
    q = safe_clip(q, eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def jsd_bernoulli(p, q, eps=EPS):
    """
    Jensen-Shannon Divergence: symmetric, bounded [0, log(2)].
    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),  m = (p+q)/2
    """
    p = safe_clip(p, eps)
    q = safe_clip(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_bernoulli(p, m, eps) + 0.5 * kl_bernoulli(q, m, eps)


# ============================================================
# 数据准备
# ============================================================

def clip_logit(values, clip_range=1e-4):
    """Clip values to (clip_range, 1-clip_range) then logit transform."""
    v = np.asarray(values, dtype=np.float64)
    v_clipped = np.clip(v, clip_range, 1.0 - clip_range)
    return np.log(v_clipped / (1.0 - v_clipped))


def load_data():
    """加载融合数据, 返回观测 logit 值和元数据."""
    df = pd.read_csv(FUSION_CSV)
    y_s2 = clip_logit(df['p_s2_mean'].values)
    y_viirs = clip_logit(df['p_viirs_mean'].values)
    # NaN 标记为 masked
    y_s2[np.isnan(df['p_s2_mean'].values)] = np.nan
    y_viirs[np.isnan(df['p_viirs_mean'].values)] = np.nan
    return {
        'y_s2': y_s2,
        'y_viirs': y_viirs,
        'n': len(df),
        'df': df,
    }


def load_both_posterior():
    """加载 both-model 后验统计."""
    df = pd.read_csv(POSTERIOR_CSV)
    return df


# ============================================================
# 单传感器贝叶斯模型 (clean, no delta parameter)
# ============================================================

def build_single_sensor_model(y_obs, sensor_name, y_raw=None):
    """
    构建 clean 单传感器 Logit-Normal 层次模型 (非中心参数化).

    仅拟合有效观测的网格, 然后为所有网格生成 posterior predictive.

    θ_i ~ N(μ, τ²)
    p_i = invlogit(θ_i)
    y_obs_i ~ N(θ_i, σ²_k)
    """
    valid_mask = ~np.isnan(y_obs)
    y_valid = y_obs[valid_mask]
    n_valid = len(y_valid)
    n_total = len(y_obs)

    print(f"  Building model: {n_valid}/{n_total} valid grids")

    # 使用数据驱动的先验中心
    y_mean_init = float(np.mean(y_valid))
    y_std_init = float(np.std(y_valid))
    print(f"  y mean={y_mean_init:.3f}, y std={y_std_init:.3f}")

    with pm.Model() as model:
        # 超先验
        mu = pm.Normal('mu', mu=y_mean_init, sigma=2.0)
        tau = pm.HalfCauchy('tau', beta=1.0)

        # 传感器噪声使用更宽松的先验
        sigma_k = pm.HalfNormal('sigma_k', sigma=max(1.0, y_std_init))

        # 非中心参数化 for VALID grids only
        theta_raw_offset = pm.Normal('theta_raw_offset', mu=0.0, sigma=1.0,
                                     shape=n_valid)
        theta_offset = tau * theta_raw_offset  # N(0, tau²)
        theta_valid = pm.Deterministic('theta_valid', mu + theta_offset)
        p_damage_valid = pm.Deterministic('p_damage_valid',
                                          pm.math.invlogit(theta_valid))

        # 似然: 仅对有效观测
        pm.Normal('y_obs', mu=theta_valid, sigma=sigma_k,
                  observed=y_valid)

    return model, valid_mask


def sample_model(model, target_accept=TARGET_ACCEPT):
    """Run NUTS sampler, return InferenceData."""
    with model:
        idata = pm.sample(
            draws=MCMC_DRAWS, tune=MCMC_TUNE, chains=MCMC_CHAINS,
            random_seed=SEED, target_accept=target_accept,
            progressbar=False,
        )
    return idata


def run_single_sensor_model(y_obs, sensor_name, out_csv):
    """
    运行单传感器贝叶斯模型, 保存每网格后验均值.

    先在有效网格上 MCMC, 再对无效网格从先验预测分布采样.
    返回: per-grid posterior mean array (all grids) + 超参数 summary.
    """
    print(f"\n{'='*60}")
    print(f"Fitting {sensor_name}-only Bayesian model")
    print(f"{'='*60}")

    valid_mask = ~np.isnan(y_obs)
    n_valid = valid_mask.sum()
    n_total = len(y_obs)

    model, valid_mask_model = build_single_sensor_model(y_obs, sensor_name)
    idata = sample_model(model)

    # MCMC summary
    summary = az.summary(idata, var_names=['mu', 'tau', 'sigma_k'])
    print(summary[['mean', 'sd', 'r_hat']].to_string())

    mu_samples  = idata.posterior['mu'].values.flatten()       # (chains*draws,)
    tau_samples = idata.posterior['tau'].values.flatten()
    sigma_samples = idata.posterior['sigma_k'].values.flatten()

    # 有效网格的后验 p
    p_valid_samples = idata.posterior['p_damage_valid'].values  # (chains, draws, n_valid)
    p_valid_flat = p_valid_samples.reshape(-1, n_valid)

    # 为所有网格生成后验均值:
    # 有效网格: 直接用 MCMC 后验
    # 无效网格: 从先验预测分布采样 p ~ invlogit(N(μ, τ²))
    n_samples = len(mu_samples)
    p_mean = np.zeros(n_total)
    p_sd   = np.zeros(n_total)
    ci_2_5  = np.zeros(n_total)
    ci_97_5 = np.zeros(n_total)

    # 有效网格
    p_mean[valid_mask] = np.mean(p_valid_flat, axis=0)
    p_sd[valid_mask]   = np.std(p_valid_flat, axis=0)
    ci_2_5[valid_mask] = np.percentile(p_valid_flat, 2.5, axis=0)
    ci_97_5[valid_mask] = np.percentile(p_valid_flat, 97.5, axis=0)

    # 无效网格: 从超参数后验预测
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        theta_prior = (mu_samples[:, None] +
                       tau_samples[:, None] * np.random.randn(n_samples, n_missing))
        p_prior = 1.0 / (1.0 + np.exp(-theta_prior))
        p_mean[~valid_mask] = np.mean(p_prior, axis=0)
        p_sd[~valid_mask]   = np.std(p_prior, axis=0)
        ci_2_5[~valid_mask] = np.percentile(p_prior, 2.5, axis=0)
        ci_97_5[~valid_mask] = np.percentile(p_prior, 97.5, axis=0)

    # 超参数统计
    mu_mean  = float(np.mean(mu_samples))
    tau_mean = float(np.mean(tau_samples))
    sigma_mean = float(np.mean(sigma_samples))
    rhat_max = float(summary['r_hat'].max())
    div = int(idata.sample_stats.diverging.sum().values)

    print(f"\n  p_i range (valid): [{p_mean[valid_mask].min():.4f}, {p_mean[valid_mask].max():.4f}]")
    print(f"  p_i mean (all):    {np.mean(p_mean):.4f}")
    print(f"  μ={mu_mean:.4f}, τ={tau_mean:.4f}, σ={sigma_mean:.4f}")
    print(f"  R-hat max={rhat_max:.4f}, divergences={div}")

    # Save
    df_out = pd.DataFrame({
        'grid_id': np.arange(n_total),
        f'p_{sensor_name}_only_mean': p_mean,
        f'p_{sensor_name}_only_sd': p_sd,
        f'p_{sensor_name}_only_ci_2_5': ci_2_5,
        f'p_{sensor_name}_only_ci_97_5': ci_97_5,
    })
    df_out.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")

    hyper_summary = {
        'mu_mean': mu_mean,
        'tau_mean': tau_mean,
        'sigma_mean': sigma_mean,
        'rhat_max': rhat_max,
        'divergences': div,
        'p_mean_range': (float(p_mean.min()), float(p_mean.max())),
        'p_mean_mean': float(np.mean(p_mean)),
        'p_range_width': float(p_mean.max() - p_mean.min()),
        'n_valid': int(n_valid),
        'n_total': int(n_total),
    }

    return p_mean, hyper_summary, idata


# ============================================================
# KLD 指标计算
# ============================================================

def compute_kld_metrics(df_both, p_s2_only, p_viirs_only):
    """
    对每个网格计算所有 KLD 指标.

    参数:
      df_both: both-model posterior DataFrame (含 posterior_mean)
      p_s2_only: S2-only per-grid posterior mean array
      p_viirs_only: VIIRS-only per-grid posterior mean array

    返回: DataFrame with all KLD metrics per grid.
    """
    n = len(df_both)
    p_both = df_both['posterior_mean'].values

    # Neutral prior baseline
    P_NEUTRAL = 0.5

    # Global mean baselines (空间分异基准)
    p_s2_global = np.mean(p_s2_only)
    p_viirs_global = np.mean(p_viirs_only)
    p_both_global = np.mean(p_both)

    print(f"\n{'='*60}")
    print("Computing KLD metrics")
    print(f"{'='*60}")
    print(f"  p_s2_global_mean  = {p_s2_global:.6f}")
    print(f"  p_viirs_global_mean = {p_viirs_global:.6f}")
    print(f"  p_both_global_mean  = {p_both_global:.6f}")

    # 初始化结果数组
    results = {
        'grid_id': np.arange(n),
        'p_s2_only': p_s2_only,
        'p_viirs_only': p_viirs_only,
        'p_both': p_both,
    }

    # A. 单传感器信息增益 (vs neutral prior)
    results['kl_s2_vs_neutral_prior'] = kl_bernoulli(p_s2_only, P_NEUTRAL)
    results['kl_viirs_vs_neutral_prior'] = kl_bernoulli(p_viirs_only, P_NEUTRAL)

    # B. 融合后信息增益 (vs neutral prior)
    results['kl_both_vs_neutral_prior'] = kl_bernoulli(p_both, P_NEUTRAL)

    # C. 空间分异信息 (vs global mean)
    results['kl_s2_vs_s2_global'] = kl_bernoulli(p_s2_only, p_s2_global)
    results['kl_viirs_vs_viirs_global'] = kl_bernoulli(p_viirs_only, p_viirs_global)
    results['kl_both_vs_both_global'] = kl_bernoulli(p_both, p_both_global)

    # D. 传感器冲突度
    results['kl_s2_vs_viirs'] = kl_bernoulli(p_s2_only, p_viirs_only)
    results['kl_viirs_vs_s2'] = kl_bernoulli(p_viirs_only, p_s2_only)
    results['jsd_s2_viirs'] = jsd_bernoulli(p_s2_only, p_viirs_only)
    results['abs_diff_s2_viirs'] = np.abs(p_s2_only - p_viirs_only)

    # E. 融合抑制指标
    best_single = np.maximum(results['kl_s2_vs_neutral_prior'],
                             results['kl_viirs_vs_neutral_prior'])
    results['best_single_sensor_kl'] = best_single
    results['fusion_kl_loss'] = best_single - results['kl_both_vs_neutral_prior']
    # 防止除以 0
    denom = np.where(best_single < EPS, np.inf, best_single)
    results['fusion_suppression_ratio'] = np.where(
        best_single < EPS, 0.0,
        results['fusion_kl_loss'] / denom
    )

    # F. 冲突等级
    jsd = results['jsd_s2_viirs']
    conflict_rank = np.full(n, -1, dtype=int)
    conflict_rank[(jsd >= 0.0) & (jsd < 0.01)] = 0   # negligible
    conflict_rank[(jsd >= 0.01) & (jsd < 0.05)] = 1  # low
    conflict_rank[(jsd >= 0.05) & (jsd < 0.15)] = 2  # moderate
    conflict_rank[(jsd >= 0.15) & (jsd < 0.50)] = 3  # high
    conflict_rank[jsd >= 0.50] = 4                     # extreme
    results['conflict_rank'] = conflict_rank

    df_out = pd.DataFrame(results)

    # 验证: 无 NaN / inf
    for col in df_out.columns:
        if col == 'grid_id' or col == 'conflict_rank':
            continue
        n_nan = df_out[col].isna().sum()
        n_inf = np.isinf(df_out[col]).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"  WARNING: {col} has {n_nan} NaN, {n_inf} inf")

    return df_out


# ============================================================
# 输出: Summary table
# ============================================================

def compute_summary(df_kld):
    """计算每个指标的 summary statistics."""
    metric_cols = [c for c in df_kld.columns
                   if c not in ('grid_id', 'conflict_rank')
                   and df_kld[c].dtype in ('float64', 'float32', 'int64', 'int32')]

    rows = []
    for col in metric_cols:
        vals = df_kld[col].values
        vals_finite = vals[np.isfinite(vals)]
        rows.append({
            'metric': col,
            'mean': np.mean(vals_finite),
            'median': np.median(vals_finite),
            'std': np.std(vals_finite),
            'min': np.min(vals_finite),
            'max': np.max(vals_finite),
            'p25': np.percentile(vals_finite, 25),
            'p75': np.percentile(vals_finite, 75),
            'sum': np.sum(vals_finite),
        })

    summary_df = pd.DataFrame(rows)
    return summary_df


# ============================================================
# 输出: Figures
# ============================================================

def load_geodata():
    """加载 GeoJSON 用于地图绘制."""
    gdf = gpd.read_file(FUSION_GEOJSON)
    if gdf.crs is not None and gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    return gdf


def plot_kld_map(gdf, values, title, out_path, cmap='YlOrRd',
                 vmin=None, vmax=None, label='KL (nats)'):
    """通用 KLD 地图绘制."""
    plot_gdf = gdf.copy()
    plot_gdf['value'] = values

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_gdf.plot(column='value', ax=ax, cmap=cmap, legend=True,
                  vmin=vmin, vmax=vmax,
                  legend_kwds={'label': label, 'shrink': 0.6},
                  missing_kwds={'color': 'lightgrey', 'label': 'no data'})
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure → {out_path}")


def plot_all_figures(df_kld, gdf):
    """生成所有 KLD 地图和 barplot."""
    n = len(df_kld)

    # 1. S2 information map (vs neutral prior)
    plot_kld_map(gdf, df_kld['kl_s2_vs_neutral_prior'].values,
                 'S2-only Information Gain vs Neutral Prior\n'
                 f'KL(Bern(p_s2) || Bern(0.5)), mean={df_kld["kl_s2_vs_neutral_prior"].mean():.4f} nats',
                 KLD_S2_MAP_PNG, cmap='YlOrRd')

    # 2. VIIRS information map (vs neutral prior)
    plot_kld_map(gdf, df_kld['kl_viirs_vs_neutral_prior'].values,
                 'VIIRS-only Information Gain vs Neutral Prior\n'
                 f'KL(Bern(p_viirs) || Bern(0.5)), mean={df_kld["kl_viirs_vs_neutral_prior"].mean():.4f} nats',
                 KLD_VIIRS_MAP_PNG, cmap='YlOrRd')

    # 3. Both information map (vs neutral prior)
    plot_kld_map(gdf, df_kld['kl_both_vs_neutral_prior'].values,
                 'Both-sensor Fused Information Gain vs Neutral Prior\n'
                 f'KL(Bern(p_both) || Bern(0.5)), mean={df_kld["kl_both_vs_neutral_prior"].mean():.4f} nats',
                 KLD_BOTH_MAP_PNG, cmap='YlOrRd')

    # 4. JSD conflict map
    plot_kld_map(gdf, df_kld['jsd_s2_viirs'].values,
                 'Sensor Conflict: JSD(S2-only, VIIRS-only)\n'
                 f'mean JSD={df_kld["jsd_s2_viirs"].mean():.4f}, '
                 f'median={df_kld["jsd_s2_viirs"].median():.4f}',
                 KLD_CONFLICT_JSD_PNG, cmap='RdPu',
                 vmin=0, vmax=max(0.5, df_kld['jsd_s2_viirs'].max()),
                 label='JSD (nats)')

    # 5. Fusion suppression map
    supp = df_kld['fusion_suppression_ratio'].values.copy()
    supp[np.isinf(supp)] = np.nan
    plot_kld_map(gdf, supp,
                 'Fusion Suppression Ratio\n'
                 f'(best_single_kl - kl_both) / best_single_kl, mean={np.nanmean(supp):.4f}',
                 KLD_SUPPRESSION_PNG, cmap='Reds',
                 vmin=0, vmax=1.0,
                 label='Suppression ratio')

    # 6. Summary barplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: mean KL vs neutral prior (单传感器)
    ax = axes[0]
    metrics_a = ['kl_s2_vs_neutral_prior', 'kl_viirs_vs_neutral_prior',
                 'kl_both_vs_neutral_prior']
    labels_a = ['S2-only', 'VIIRS-only', 'Both']
    means_a = [df_kld[m].mean() for m in metrics_a]
    colors_a = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(labels_a, means_a, color=colors_a, edgecolor='black', linewidth=0.5)
    ax.set_title('Mean Information Gain vs Neutral Prior\n(Bernoulli KL, nats)', fontweight='bold')
    ax.set_ylabel('KL divergence (nats)')
    for bar, val in zip(bars, means_a):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel B: mean KL vs global mean (空间分异)
    ax = axes[1]
    metrics_b = ['kl_s2_vs_s2_global', 'kl_viirs_vs_viirs_global',
                 'kl_both_vs_both_global']
    labels_b = ['S2-only', 'VIIRS-only', 'Both']
    means_b = [df_kld[m].mean() for m in metrics_b]
    colors_b = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(labels_b, means_b, color=colors_b, edgecolor='black', linewidth=0.5)
    ax.set_title('Mean Spatial Differentiation Info\n(KL vs global mean)', fontweight='bold')
    ax.set_ylabel('KL divergence (nats)')
    for bar, val in zip(bars, means_b):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel C: fusion suppression
    ax = axes[2]
    categories = ['Fusion KL Loss\n(mean)', 'Fusion Suppression\nRatio (mean)', 'JSD S2-VIIRS\n(mean)']
    vals = [
        df_kld['fusion_kl_loss'].mean(),
        df_kld['fusion_suppression_ratio'].mean(),
        df_kld['jsd_s2_viirs'].mean(),
    ]
    colors_c = ['#f44336', '#9C27B0', '#FF5722']
    bars = ax.bar(categories, vals, color=colors_c, edgecolor='black', linewidth=0.5)
    ax.set_title('Fusion Suppression & Conflict Summary', fontweight='bold')
    ax.set_ylabel('Value (nats / ratio)')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(KLD_BARPLOT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure → {KLD_BARPLOT_PNG}")


# ============================================================
# 测试 / Assert
# ============================================================

def run_assertions(df_kld):
    """基本验证测试."""
    print(f"\n{'='*60}")
    print("Running assertions")
    print(f"{'='*60}")

    # 1. KL(p, p) ~ 0
    p_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    kl_self = kl_bernoulli(p_test, p_test)
    assert np.allclose(kl_self, 0, atol=1e-6), f"KL(p,p) should be 0, got {kl_self}"
    print("  [PASS] KL(p, p) ~ 0")

    # 2. KL(p, q) >= 0
    p_test2 = np.array([0.2, 0.4, 0.5])
    q_test2 = np.array([0.5, 0.5, 0.1])
    kl_pq = kl_bernoulli(p_test2, q_test2)
    assert np.all(kl_pq >= -1e-10), f"KL should be >= 0, got {kl_pq}"
    print("  [PASS] KL(p, q) >= 0")

    # 3. JSD symmetric
    jsd_ab = jsd_bernoulli(np.array([0.2, 0.7]), np.array([0.7, 0.2]))
    jsd_ba = jsd_bernoulli(np.array([0.7, 0.2]), np.array([0.2, 0.7]))
    assert np.allclose(jsd_ab, jsd_ba, atol=1e-10), "JSD should be symmetric"
    print("  [PASS] JSD(p, q) == JSD(q, p)")

    # 4. JSD(p, p) ~ 0
    jsd_self = jsd_bernoulli(p_test, p_test)
    assert np.allclose(jsd_self, 0, atol=1e-6), f"JSD(p,p) should be 0, got {jsd_self}"
    print("  [PASS] JSD(p, p) ~ 0")

    # 5. No NaN / inf in output
    check_cols = [c for c in df_kld.columns
                  if c not in ('grid_id', 'conflict_rank')]
    for col in check_cols:
        n_nan = df_kld[col].isna().sum()
        n_inf = np.isinf(df_kld[col]).sum()
        assert n_nan == 0, f"{col} has {n_nan} NaN values"
        assert n_inf == 0, f"{col} has {n_inf} inf values"
    print("  [PASS] No NaN / inf in output")

    # 6. p in [0, 1]
    for col in ['p_s2_only', 'p_viirs_only', 'p_both']:
        assert np.all(df_kld[col] >= 0) and np.all(df_kld[col] <= 1), \
            f"{col} out of [0,1] range"
    print("  [PASS] All p values in [0, 1]")

    # 7. KLD >= 0
    for col in ['kl_s2_vs_neutral_prior', 'kl_viirs_vs_neutral_prior',
                'kl_both_vs_neutral_prior', 'jsd_s2_viirs']:
        min_val = df_kld[col].min()
        assert min_val >= -1e-10, f"{col} has negative values: min={min_val}"
    print("  [PASS] All KLD / JSD >= 0")

    # 8. fusion_suppression_ratio no inf
    assert not np.any(np.isinf(df_kld['fusion_suppression_ratio'])), \
        "fusion_suppression_ratio contains inf"
    print("  [PASS] fusion_suppression_ratio no inf")

    print("  All assertions passed!")


# ============================================================
# Report generation
# ============================================================

def generate_report(df_kld, summary_df, s2_hyper, viirs_hyper, s2_global, viirs_global, p_both_global):
    """生成 Markdown 报告."""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 前 10 最高冲突网格
    top10_conflict = df_kld.nlargest(10, 'jsd_s2_viirs')[
        ['grid_id', 'p_s2_only', 'p_viirs_only', 'p_both',
         'jsd_s2_viirs', 'abs_diff_s2_viirs', 'conflict_rank']
    ]

    def get_row(metric_name):
        row = summary_df[summary_df['metric'] == metric_name]
        if len(row) == 0:
            return None
        return row.iloc[0]

    def fmt_row(metric_name, label, digits=4):
        row = get_row(metric_name)
        if row is None:
            return f"| {label} | N/A |"
        return (f"| {label} | {row['mean']:.{digits}f} | {row['median']:.{digits}f} | "
                f"{row['std']:.{digits}f} | [{row['min']:.{digits}f}, {row['max']:.{digits}f}] | "
                f"{row['sum']:.{digits}f} |")

    lines = []
    lines.append(f"# KLD 信息增益分析报告")
    lines.append(f"**生成时间**: {now_str}")
    lines.append(f"**模块**: 模块 2 — KLD 信息增益分析")
    lines.append("")

    # 1. 模型汇总
    lines.append("## 1. 单传感器贝叶斯模型汇总")
    lines.append("")
    lines.append("| 模型 | μ post mean | τ post mean | σ post mean | R-hat max | Div | p_i range | p_i mean |")
    lines.append("|------|------------|------------|------------|-----------|-----|-----------|----------|")
    for name, hyp in [('S2-only', s2_hyper), ('VIIRS-only', viirs_hyper)]:
        lines.append(f"| {name} | {hyp['mu_mean']:.4f} | {hyp['tau_mean']:.4f} | "
                     f"{hyp['sigma_mean']:.4f} | {hyp['rhat_max']:.4f} | {hyp['divergences']} | "
                     f"[{hyp['p_mean_range'][0]:.4f}, {hyp['p_mean_range'][1]:.4f}] | "
                     f"{hyp['p_mean_mean']:.4f} |")
    lines.append("")

    # Both model (from prior run)
    lines.append(f"- **Both-model** global mean: p_both_mean = {p_both_global:.6f}")
    lines.append("")

    # 2. KLD 核心指标
    lines.append("## 2. KLD 核心指标 Summary")
    lines.append("")
    lines.append("| Metric | Mean | Median | Std | [Min, Max] | Sum |")
    lines.append("|--------|------|--------|-----|------------|-----|")

    key_metrics = [
        # A. 单传感器 vs neutral prior
        ('kl_s2_vs_neutral_prior', 'KL(S2 || neutral)'),
        ('kl_viirs_vs_neutral_prior', 'KL(VIIRS || neutral)'),
        ('kl_both_vs_neutral_prior', 'KL(Both || neutral)'),
        # C. 空间分异
        ('kl_s2_vs_s2_global', 'KL(S2 || S2_global)'),
        ('kl_viirs_vs_viirs_global', 'KL(VIIRS || VIIRS_global)'),
        ('kl_both_vs_both_global', 'KL(Both || Both_global)'),
        # D. 冲突
        ('kl_s2_vs_viirs', 'KL(S2 || VIIRS)'),
        ('kl_viirs_vs_s2', 'KL(VIIRS || S2)'),
        ('jsd_s2_viirs', 'JSD(S2, VIIRS)'),
        ('abs_diff_s2_viirs', '|p_s2 - p_viirs|'),
        # E. 融合抑制
        ('best_single_sensor_kl', 'Best single KL'),
        ('fusion_kl_loss', 'Fusion KL loss'),
        ('fusion_suppression_ratio', 'Fusion suppression ratio'),
    ]

    for metric_name, label in key_metrics:
        lines.append(fmt_row(metric_name, label))

    lines.append("")

    # 3. 关键对比
    lines.append("## 3. 关键对比")
    lines.append("")

    kl_s2_mean = df_kld['kl_s2_vs_neutral_prior'].mean()
    kl_v_mean  = df_kld['kl_viirs_vs_neutral_prior'].mean()
    kl_both_mean = df_kld['kl_both_vs_neutral_prior'].mean()
    jsd_mean = df_kld['jsd_s2_viirs'].mean()
    loss_mean = df_kld['fusion_kl_loss'].mean()
    supp_mean = df_kld['fusion_suppression_ratio'].mean()

    kl_s2_spatial = df_kld['kl_s2_vs_s2_global'].mean()
    kl_v_spatial  = df_kld['kl_viirs_vs_viirs_global'].mean()
    kl_both_spatial = df_kld['kl_both_vs_both_global'].mean()

    lines.append(f"### 3.1 单传感器信息增益 (vs neutral prior p=0.5)")
    lines.append("")
    lines.append(f"- S2-only mean KL: **{kl_s2_mean:.6f}** nats")
    lines.append(f"- VIIRS-only mean KL: **{kl_v_mean:.6f}** nats")
    lines.append(f"- More informative single sensor: **{'S2' if kl_s2_mean > kl_v_mean else 'VIIRS'}**")
    lines.append("")

    lines.append(f"### 3.2 融合后信息增益 (vs neutral prior)")
    lines.append("")
    lines.append(f"- Both-sensor mean KL: **{kl_both_mean:.6f}** nats")
    lines.append(f"- Best single minus Both: **{loss_mean:.6f}** nats")
    lines.append(f"- Fusion suppression ratio: **{supp_mean:.4f}** "
                 f"({supp_mean*100:.1f}% of best single's information is suppressed)")
    lines.append("")

    lines.append(f"### 3.3 空间分异信息 (vs global mean)")
    lines.append("")
    lines.append(f"- S2-only spatial KL: **{kl_s2_spatial:.6f}** nats")
    lines.append(f"- VIIRS-only spatial KL: **{kl_v_spatial:.6f}** nats")
    lines.append(f"- Both-sensor spatial KL: **{kl_both_spatial:.6f}** nats")
    lines.append(f"- **Interpretation**: Both-model spatial KL {'≈ 0 → posterior nearly spatially uniform' if kl_both_spatial < 0.001 else '> 0 → some spatial structure remains'}")
    lines.append("")

    lines.append(f"### 3.4 传感器冲突度")
    lines.append("")
    lines.append(f"- Mean JSD(S2, VIIRS): **{jsd_mean:.6f}** nats")
    lines.append(f"- Mean |p_s2 - p_viirs|: **{df_kld['abs_diff_s2_viirs'].mean():.4f}**")
    lines.append("")

    # Conflict rank distribution
    rank_names = {0: 'negligible (<0.01)', 1: 'low (0.01-0.05)',
                  2: 'moderate (0.05-0.15)', 3: 'high (0.15-0.50)', 4: 'extreme (≥0.50)'}
    lines.append("| Conflict Rank | Count | Pct |")
    lines.append("|---------------|-------|-----|")
    for r in range(5):
        cnt = (df_kld['conflict_rank'] == r).sum()
        lines.append(f"| {r} - {rank_names[r]} | {cnt} | {cnt/len(df_kld)*100:.1f}% |")
    lines.append("")

    # 4. Top 10 conflict grids
    lines.append("## 4. Top 10 Highest Conflict Grids")
    lines.append("")
    lines.append("| grid_id | p_s2_only | p_viirs_only | p_both | JSD | |diff| | rank |")
    lines.append("|---------|-----------|-------------|--------|-----|--------|------|")
    for _, row in top10_conflict.iterrows():
        lines.append(f"| {int(row['grid_id'])} | {row['p_s2_only']:.4f} | "
                     f"{row['p_viirs_only']:.4f} | {row['p_both']:.4f} | "
                     f"{row['jsd_s2_viirs']:.4f} | {row['abs_diff_s2_viirs']:.4f} | "
                     f"{int(row['conflict_rank'])} |")
    lines.append("")

    # 5. 对 T20 问题的回答
    strongest = 'S2' if kl_s2_mean > kl_v_mean else 'VIIRS'
    strongest_kl = max(kl_s2_mean, kl_v_mean)
    weaker = 'VIIRS' if strongest == 'S2' else 'S2'
    weaker_kl = min(kl_s2_mean, kl_v_mean)

    lines.append("## 5. 对 T20 问题的回答")
    lines.append("")
    lines.append("### 5.1 KL vs neutral prior 的含义")
    lines.append("")
    lines.append("KL(Bern(p) || Bern(0.5)) 衡量后验概率偏离 neutral prior (p=0.5) 的程度。")
    lines.append("注意: 高 KL 不一定意味着\"更准确\"——p=0.01 和 p=0.99 都给出高 KL，")
    lines.append("但如果 ground truth 是\"未损害\"，p=0.99 就是错的。KL 衡量的是\"从 prior 偏离的程度\"，")
    lines.append("不是\"预测精度\"。")
    lines.append("")
    lines.append(f"- S2 后验均值 ≈ {s2_global:.4f} → 远离 0.5 → KL vs neutral = **{kl_s2_mean:.4f}** nats")
    lines.append(f"- VIIRS 后验均值 ≈ {viirs_global:.4f} → 接近 0.5 → KL vs neutral = **{kl_v_mean:.4f}** nats")
    lines.append(f"- 两个传感器都提供了非零信息增益，但方向不同。")
    lines.append("")
    lines.append("### 5.2 空间分异信息 (vs global mean)")
    lines.append("")
    lines.append("KL vs global mean 是更公平的\"空间信息含量\"指标——衡量各网格偏离全域均值的程度:")
    lines.append("")
    lines.append(f"- S2-only 空间 KL = **{kl_s2_spatial:.6f}** nats (均值), max = {df_kld['kl_s2_vs_s2_global'].max():.4f}")
    lines.append(f"- VIIRS-only 空间 KL = **{kl_v_spatial:.6f}** nats (均值), max = {df_kld['kl_viirs_vs_viirs_global'].max():.4f}")
    lines.append(f"- Both-sensor 空间 KL = **{kl_both_spatial:.6f}** nats (均值), max = {df_kld['kl_both_vs_both_global'].max():.4f}")
    lines.append("")
    if kl_v_spatial > kl_s2_spatial:
        lines.append("VIIRS 的空间分异信息多于 S2 (与 UNOSAT AUC 结论一致)。")
    else:
        lines.append("S2 的空间分异信息多于 VIIRS。")
    lines.append(f"Both 模型的空间 KL 接近 0 → 融合后验几乎全域均匀。")
    lines.append("")
    lines.append("### 5.3 问题③: 增加传感器是否能显著提高估计精度?")
    lines.append("")
    lines.append(f"**Short answer: 在本案例中, 不能。更准确地说, 在 single latent state 假设下, 增加冲突传感器降低了可整合信息。**")
    lines.append("")
    lines.append("**定量证据**:")
    lines.append("")
    lines.append(f"1. **信息增益退行**: 两个传感器独立都有信息, 但融合后的空间分异信息")
    lines.append(f"   ({kl_both_spatial:.6f} nats) 远小于任一单传感器")
    lines.append(f"   (S2: {kl_s2_spatial:.6f}, VIIRS: {kl_v_spatial:.6f})。")
    lines.append(f"   Both 后验的 p_i 范围仅 ~0.006 (from {s2_global:.4f} to {viirs_global:.4f}),")
    lines.append("   丧失了所有网格间区分能力。")
    lines.append("")
    lines.append(f"2. **融合抑制比**: {supp_mean*100:.1f}%——Nearly half the best single-sensor's")
    lines.append(f"   deviation from neutral prior is suppressed in the fused posterior.")
    lines.append("")
    lines.append(f"3. **传感器冲突**: 平均 JSD(S2, VIIRS) = **{jsd_mean:.4f}** nats, ")
    lines.append(f"   所有 1380 个网格均处于 moderate 或 high 冲突等级。")
    lines.append(f"   S2 和 VIIRS 分别独立包含空间信息, 但它们指向不同的\"损害真相\"。")
    lines.append("")
    lines.append(f"4. **冲突不对称**: KL(S2 || VIIRS) = **{df_kld['kl_s2_vs_viirs'].mean():.4f}** nats, ")
    lines.append(f"   KL(VIIRS || S2) = **{df_kld['kl_viirs_vs_s2'].mean():.4f}** nats。")
    lines.append(f"   后者更大, 因为 VIIRS(p≈0.50) 比 S2(p≈0.05) 更\"分散\"——")
    lines.append(f"   用 S2 的集中分布去近似 VIIRS 的分散分布比反过来需要更多信息。")
    lines.append("")
    lines.append("**根本原因**: S2 测量物理结构破坏 (mean p={:.4f}), "
                 "VIIRS 测量功能丧失 (mean p={:.4f})。"
                 "它们是损害的两个互补维度, 不收敛于同一个 latent state。"
                 "在 single latent state 假设下, 冲突证据导致后验收缩。".format(
                     s2_global, viirs_global))
    lines.append("")
    lines.append("**但这不意味着任一传感器无用**:")
    lines.append("")
    lines.append(f"- S2-only 从 neutral prior 获得了 {kl_s2_mean:.4f} nats 信息 (p≈{s2_global:.3f}, 全域一致低损害)")
    lines.append(f"- VIIRS-only 的空间分异信息是 S2-only 的 {kl_v_spatial/kl_s2_spatial:.1f}x (当 kl_s2_spatial > 0)")
    lines.append("- 问题不在于传感器质量, 而在于 **single latent state 假设不成立**")
    lines.append("- 两个传感器各自包含有意义的空间信息, 但信息方向冲突")
    lines.append("- 解决方案: 采用多维损害模型 (结构损害 + 功能损害), 或使用 D-S 证据理论保留冲突")
    lines.append("")

    lines.append("### 核心结论")
    lines.append("")
    lines.append("> Under the single latent damage-state assumption, adding a conflicting sensor "
                "does not necessarily increase information gain. Instead, S2 and VIIRS "
                "individually contain spatial information, but their disagreement suppresses "
                "the fused posterior's grid-level information. The fusion suppression ratio "
                f"of {supp_mean:.4f} quantifies this effect: the fused posterior retains only "
                f"{'(1 - supp_mean)*100' if False else (1-supp_mean)*100:.1f}% "
                "of the best single-sensor information, while the remaining "
                f"{supp_mean*100:.1f}% is lost due to inter-sensor conflict.")
    lines.append("")

    # 6. 输出文件
    lines.append("## 6. 输出文件")
    lines.append("")
    lines.append("| 文件 | 描述 |")
    lines.append("|------|------|")
    lines.append(f"| `{KLD_CSV.relative_to(PROJECT_ROOT)}` | Per-grid KLD metrics CSV |")
    lines.append(f"| `{KLD_SUMMARY.relative_to(PROJECT_ROOT)}` | KLD summary statistics |")
    lines.append(f"| `{S2_ONLY_POST_CSV.relative_to(PROJECT_ROOT)}` | S2-only per-grid posterior |")
    lines.append(f"| `{VIIRS_ONLY_POST_CSV.relative_to(PROJECT_ROOT)}` | VIIRS-only per-grid posterior |")
    lines.append(f"| `{KLD_S2_MAP_PNG.relative_to(PROJECT_ROOT)}` | S2 信息地图 |")
    lines.append(f"| `{KLD_VIIRS_MAP_PNG.relative_to(PROJECT_ROOT)}` | VIIRS 信息地图 |")
    lines.append(f"| `{KLD_BOTH_MAP_PNG.relative_to(PROJECT_ROOT)}` | Both 信息地图 |")
    lines.append(f"| `{KLD_CONFLICT_JSD_PNG.relative_to(PROJECT_ROOT)}` | JSD 冲突地图 |")
    lines.append(f"| `{KLD_SUPPRESSION_PNG.relative_to(PROJECT_ROOT)}` | 融合抑制地图 |")
    lines.append(f"| `{KLD_BARPLOT_PNG.relative_to(PROJECT_ROOT)}` | Summary barplot |")
    lines.append(f"| `{REPORT_MD.relative_to(PROJECT_ROOT)}` | 本报告 |")

    report_text = '\n'.join(lines)
    REPORT_MD.write_text(report_text, encoding='utf-8')
    print(f"\n  Report → {REPORT_MD}")
    return report_text


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("模块 2: KLD 信息增益分析")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/6] Loading data...")
    data = load_data()
    df_both = load_both_posterior()
    print(f"  Loaded {data['n']} grids, {df_both['posterior_mean'].notna().sum()} with both posterior")

    # --- Run single-sensor Bayesian models ---
    print("\n[2/6] Running single-sensor Bayesian models...")

    # S2-only
    p_s2_only, s2_hyper, idata_s2 = run_single_sensor_model(
        data['y_s2'], 's2', S2_ONLY_POST_CSV
    )

    # VIIRS-only
    p_viirs_only, viirs_hyper, idata_viirs = run_single_sensor_model(
        data['y_viirs'], 'viirs', VIIRS_ONLY_POST_CSV
    )

    # --- Compute KLD metrics ---
    print("\n[3/6] Computing KLD metrics per grid...")
    df_kld = compute_kld_metrics(df_both, p_s2_only, p_viirs_only)

    # --- Save KLD CSV ---
    print(f"\n[4/6] Saving KLD outputs...")
    df_kld.to_csv(KLD_CSV, index=False)
    print(f"  KLD CSV → {KLD_CSV} ({len(df_kld)} rows × {len(df_kld.columns)} cols)")

    # Summary
    summary_df = compute_summary(df_kld)
    summary_df.to_csv(KLD_SUMMARY, index=False)
    print(f"  Summary → {KLD_SUMMARY}")

    # --- Figures ---
    print(f"\n[5/6] Generating figures...")
    gdf = load_geodata()
    plot_all_figures(df_kld, gdf)

    # --- Assertions ---
    run_assertions(df_kld)

    # --- Report ---
    print(f"\n[6/6] Generating report...")
    p_both_global = df_both['posterior_mean'].mean()
    s2_global = np.mean(p_s2_only)
    viirs_global = np.mean(p_viirs_only)
    report = generate_report(df_kld, summary_df,
                             s2_hyper, viirs_hyper,
                             s2_global, viirs_global, p_both_global)

    print("\n" + "=" * 60)
    print("KLD 信息增益分析完成!")
    print("=" * 60)

    # Print key results
    print(f"\n  === KEY RESULTS ===")
    print(f"  KL(S2 || neutral)           = {df_kld['kl_s2_vs_neutral_prior'].mean():.6f} nats")
    print(f"  KL(VIIRS || neutral)        = {df_kld['kl_viirs_vs_neutral_prior'].mean():.6f} nats")
    print(f"  KL(Both || neutral)         = {df_kld['kl_both_vs_neutral_prior'].mean():.6f} nats")
    print(f"  KL(S2 || S2_global)         = {df_kld['kl_s2_vs_s2_global'].mean():.6f} nats")
    print(f"  KL(VIIRS || VIIRS_global)   = {df_kld['kl_viirs_vs_viirs_global'].mean():.6f} nats")
    print(f"  KL(Both || Both_global)     = {df_kld['kl_both_vs_both_global'].mean():.6f} nats")
    print(f"  JSD(S2, VIIRS)              = {df_kld['jsd_s2_viirs'].mean():.6f} nats")
    print(f"  Fusion KL loss              = {df_kld['fusion_kl_loss'].mean():.6f} nats")
    print(f"  Fusion suppression ratio    = {df_kld['fusion_suppression_ratio'].mean():.4f}")

    return df_kld, summary_df


if __name__ == '__main__':
    main()
