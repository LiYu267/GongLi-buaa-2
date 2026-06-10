#!/usr/bin/env python3
"""
模块 1 审计 — Bayesian 模型全面验证

1. Posterior Predictive Check (PPC)
2. Prior Sensitivity (tau)
3. Sensor-only Baseline
4. MCMC 诊断复查
5. 残差空间结构

使用前需先运行 16_bayesian_hierarchical.py 生成后验数据.
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
FUSION_CSV = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.csv"
FUSION_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.geojson"

OUT_DATA = PROJECT_ROOT / "data" / "processed"
OUT_FIG  = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB  = PROJECT_ROOT / "outputs" / "tables"
OUT_MD   = PROJECT_ROOT / "outputs"
for d in [OUT_DATA, OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)

EPS = 1e-8
SEED = 42
MCMC_DRAWS = 1000
MCMC_TUNE = 1000
MCMC_CHAINS = 4
TARGET_ACCEPT = 0.95


# ============================================================
# 数据准备 (复用)
# ============================================================

def clip_logit(values):
    v = np.asarray(values, dtype=np.float64)
    v_clipped = np.clip(v, EPS, 1.0 - EPS)
    return np.log(v_clipped / (1.0 - v_clipped))


def prepare_data():
    df = pd.read_csv(FUSION_CSV)
    y_s2 = clip_logit(df['p_s2_mean'].values)
    y_viirs = clip_logit(df['p_viirs_mean'].values)
    y_s2[np.isnan(df['p_s2_mean'].values)] = np.nan
    y_viirs[np.isnan(df['p_viirs_mean'].values)] = np.nan
    return {'y_s2': y_s2, 'y_viirs': y_viirs, 'n': len(df), 'df': df}


# ============================================================
# 模型构建 (参数化)
# ============================================================

def build_model_full(data, tau_beta=1.0, use_s2=True, use_viirs=True):
    """构建 Logit-Normal 层次模型, 可选传感器."""
    y_s2 = data['y_s2']
    y_viirs = data['y_viirs']
    n = data['n']

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0.0, sigma=2.0)
        tau = pm.HalfCauchy('tau', beta=tau_beta)
        sigma_s2 = pm.HalfCauchy('sigma_s2', beta=0.5)
        sigma_viirs = pm.HalfCauchy('sigma_viirs', beta=0.5)
        delta_s2 = pm.Normal('delta_s2', mu=0.0, sigma=1.0)
        delta_viirs = pm.Deterministic('delta_viirs', -delta_s2)

        theta_raw = pm.Normal('theta_raw', mu=0.0, sigma=1.0, shape=n)
        theta = pm.Deterministic('theta', mu + tau * theta_raw)
        p_damage = pm.Deterministic('p_damage', pm.math.invlogit(theta))

        if use_s2:
            pm.Normal('y_s2_obs', mu=theta + delta_s2, sigma=sigma_s2,
                      observed=np.ma.masked_invalid(y_s2))
        if use_viirs:
            pm.Normal('y_viirs_obs', mu=theta + delta_viirs, sigma=sigma_viirs,
                      observed=np.ma.masked_invalid(y_viirs))

    return model


def sample_model(model, target_accept=TARGET_ACCEPT):
    with model:
        idata = pm.sample(
            draws=MCMC_DRAWS, tune=MCMC_TUNE, chains=MCMC_CHAINS,
            random_seed=SEED, target_accept=target_accept,
            progressbar=False,
        )
    return idata


# ============================================================
# 1. Posterior Predictive Check
# ============================================================

def run_ppc(data):
    """PPC: 从已有后验生成 predictive samples, 对比观测."""
    print("=" * 60)
    print("审计 1: Posterior Predictive Check")
    print("=" * 60)

    # 从主模型 fit 一次获取后验
    print("  Fitting model...")
    model = build_model_full(data)
    idata = sample_model(model)

    # PPC samples
    print("  Sampling PPC...")
    with model:
        ppc = pm.sample_posterior_predictive(
            idata, random_seed=SEED, progressbar=False
        )

    # 提取观测和预测
    obs_s2 = data['y_s2'][~np.isnan(data['y_s2'])]
    obs_viirs = data['y_viirs'][~np.isnan(data['y_viirs'])]

    # y_s2_obs PPC
    ppc_s2_all = ppc.posterior_predictive['y_s2_obs'].values
    n_s2 = (~np.isnan(data['y_s2'])).sum()
    ppc_s2 = ppc_s2_all[:, :, :n_s2].reshape(-1, n_s2)  # flatten chains

    # y_viirs_obs PPC
    ppc_viirs_all = ppc.posterior_predictive['y_viirs_obs'].values
    n_viirs = (~np.isnan(data['y_viirs'])).sum()
    ppc_viirs = ppc_viirs_all[:, :, :n_viirs].reshape(-1, n_viirs)

    # 汇总统计
    rows = []
    for label, obs, ppc_mat in [
        ('S2 (logit)', obs_s2, ppc_s2),
        ('VIIRS (logit)', obs_viirs, ppc_viirs),
    ]:
        # Predicted mean across draws
        pred_mean = ppc_mat.mean(axis=0)
        pred_std = ppc_mat.std(axis=0)
        # Overall summaries
        rows.append({
            'sensor': label,
            'obs_mean': float(np.mean(obs)),
            'obs_std': float(np.std(obs)),
            'pred_mean_mean': float(np.mean(pred_mean)),
            'pred_mean_std': float(np.std(pred_mean)),
            'pred_std_mean': float(np.mean(pred_std)),
            'mean_error': float(np.mean(obs) - np.mean(pred_mean)),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_TAB / "ppc_summary.csv", index=False)
    print("  Summary → ppc_summary.csv")
    print(summary_df.to_string(index=False))

    # 图: observed vs predicted 分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, label, obs, ppc_mat in [
        (axes[0], 'S2', obs_s2, ppc_s2),
        (axes[1], 'VIIRS', obs_viirs, ppc_viirs),
    ]:
        # Histogram of observed
        ax.hist(obs, bins=50, density=True, alpha=0.6, color='black',
                label='Observed', edgecolor='none')
        # Overlay a few predictive draws
        for i in range(min(20, len(ppc_mat))):
            ax.hist(ppc_mat[i], bins=50, density=True, alpha=0.05,
                    color='steelblue', edgecolor='none')
        ax.hist(ppc_mat[0], bins=50, density=True, alpha=0.4,
                color='steelblue', edgecolor='none', label='Predicted (1 draw)')
        ax.set_title(f'{label} — Observed vs Posterior Predictive')
        ax.set_xlabel('logit(damage)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_FIG / "ppc_s2_viirs.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figure → ppc_s2_viirs.png")

    return idata, ppc


# ============================================================
# 2. Prior Sensitivity
# ============================================================

def run_prior_sensitivity(data):
    """对 τ 使用不同先验, 比较后验行为."""
    print("\n" + "=" * 60)
    print("审计 2: Prior Sensitivity (tau)")
    print("=" * 60)

    # Prior settings: tighter, default, wider
    # HalfCauchy(β) — smaller β = tighter prior
    prior_settings = {
        'tau_HalfCauchy_0.3': 0.3,   # very tight
        'tau_HalfCauchy_1.0': 1.0,   # default
        'tau_HalfCauchy_3.0': 3.0,   # wide
        'tau_Exponential_2.0': 'exp2', # Exponential(λ=2) ≈ HalfCauchy-ish
        'tau_HalfNormal_1.0': 'hn1',  # HalfNormal(σ=1)
    }

    results = []
    p_range_evol = {}

    for label, tau_setting in prior_settings.items():
        print(f"\n  Prior: {label}")

        if tau_setting == 'exp2':
            # Build model manually with Exponential prior
            y_s2 = data['y_s2']
            y_viirs = data['y_viirs']
            n = data['n']
            with pm.Model() as model:
                mu = pm.Normal('mu', mu=0.0, sigma=2.0)
                tau = pm.Exponential('tau', lam=2.0)  # E[τ] = 0.5
                sigma_s2 = pm.HalfCauchy('sigma_s2', beta=0.5)
                sigma_viirs = pm.HalfCauchy('sigma_viirs', beta=0.5)
                delta_s2 = pm.Normal('delta_s2', mu=0.0, sigma=1.0)
                delta_viirs = pm.Deterministic('delta_viirs', -delta_s2)
                theta_raw = pm.Normal('theta_raw', mu=0.0, sigma=1.0, shape=n)
                theta = pm.Deterministic('theta', mu + tau * theta_raw)
                p_damage = pm.Deterministic('p_damage', pm.math.invlogit(theta))
                pm.Normal('y_s2_obs', mu=theta + delta_s2, sigma=sigma_s2,
                          observed=np.ma.masked_invalid(y_s2))
                pm.Normal('y_viirs_obs', mu=theta + delta_viirs, sigma=sigma_viirs,
                          observed=np.ma.masked_invalid(y_viirs))
        elif tau_setting == 'hn1':
            y_s2 = data['y_s2']
            y_viirs = data['y_viirs']
            n = data['n']
            with pm.Model() as model:
                mu = pm.Normal('mu', mu=0.0, sigma=2.0)
                tau = pm.HalfNormal('tau', sigma=1.0)
                sigma_s2 = pm.HalfCauchy('sigma_s2', beta=0.5)
                sigma_viirs = pm.HalfCauchy('sigma_viirs', beta=0.5)
                delta_s2 = pm.Normal('delta_s2', mu=0.0, sigma=1.0)
                delta_viirs = pm.Deterministic('delta_viirs', -delta_s2)
                theta_raw = pm.Normal('theta_raw', mu=0.0, sigma=1.0, shape=n)
                theta = pm.Deterministic('theta', mu + tau * theta_raw)
                p_damage = pm.Deterministic('p_damage', pm.math.invlogit(theta))
                pm.Normal('y_s2_obs', mu=theta + delta_s2, sigma=sigma_s2,
                          observed=np.ma.masked_invalid(y_s2))
                pm.Normal('y_viirs_obs', mu=theta + delta_viirs, sigma=sigma_viirs,
                          observed=np.ma.masked_invalid(y_viirs))
        else:
            model = build_model_full(data, tau_beta=tau_setting)

        idata = sample_model(model)

        # Extract key metrics
        p_samples = idata.posterior['p_damage'].values
        p_flat = p_samples.reshape(-1, p_samples.shape[-1])
        p_mean_per_grid = np.mean(p_flat, axis=0)

        tau_samples = idata.posterior['tau'].values.flatten()
        sigma_s2_samples = idata.posterior['sigma_s2'].values.flatten()
        sigma_viirs_samples = idata.posterior['sigma_viirs'].values.flatten()
        delta_s2_samples = idata.posterior['delta_s2'].values.flatten()

        # MCMC diagnostics
        summary = az.summary(idata)
        rhat_max = float(summary['r_hat'].max())
        divergences = int(idata.sample_stats.diverging.sum().values)

        results.append({
            'prior_label': label,
            'p_mean_range': f"[{p_mean_per_grid.min():.4f}, {p_mean_per_grid.max():.4f}]",
            'p_mean_mean': float(np.mean(p_mean_per_grid)),
            'p_mean_range_width': float(p_mean_per_grid.max() - p_mean_per_grid.min()),
            'p_ci_width_mean': float(np.mean(np.percentile(p_flat, 97.5, axis=0) - np.percentile(p_flat, 2.5, axis=0))),
            'tau_post_mean': float(np.mean(tau_samples)),
            'tau_post_ci': f"[{np.percentile(tau_samples, 2.5):.4f}, {np.percentile(tau_samples, 97.5):.4f}]",
            'sigma_s2_post_mean': float(np.mean(sigma_s2_samples)),
            'sigma_viirs_post_mean': float(np.mean(sigma_viirs_samples)),
            'delta_s2_post_mean': float(np.mean(delta_s2_samples)),
            'rhat_max': rhat_max,
            'divergences': divergences,
        })

        p_range_evol[label] = p_mean_per_grid.copy()

        print(f"    p_i range: [{p_mean_per_grid.min():.4f}, {p_mean_per_grid.max():.4f}]")
        print(f"    tau post mean: {np.mean(tau_samples):.4f}")
        print(f"    R-hat max: {rhat_max:.4f}, div: {divergences}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_TAB / "prior_sensitivity_summary.csv", index=False)
    print(f"\n  Summary → prior_sensitivity_summary.csv")

    # 图: 各 prior 的 p_i 分布对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    for label, p_vals in p_range_evol.items():
        ax1.hist(p_vals, bins=50, alpha=0.4, label=label, density=True)
    ax1.set_xlabel('Posterior Mean p(damage)')
    ax1.set_ylabel('Density')
    ax1.set_title('Prior Sensitivity: p_i Distribution')
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    labels = list(p_range_evol.keys())
    widths = [r['p_mean_range_width'] for r in results]
    ax2.bar(range(len(labels)), widths, color='steelblue')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('p_i Range Width')
    ax2.set_title('Prior Sensitivity: p_i Range Width')

    plt.tight_layout()
    fig.savefig(OUT_FIG / "prior_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figure → prior_sensitivity.png")

    return result_df


# ============================================================
# 3. Sensor-only Baseline
# ============================================================

def run_sensor_only_baseline(data):
    """分别只用一个传感器拟合, 对比."""
    print("\n" + "=" * 60)
    print("审计 3: Sensor-only Baseline")
    print("=" * 60)

    configs = [
        ('both', True, True),
        ('s2_only', True, False),
        ('viirs_only', False, True),
    ]

    results = {}
    rows = []

    for label, use_s2, use_viirs in configs:
        print(f"\n  Model: {label}")
        model = build_model_full(data, use_s2=use_s2, use_viirs=use_viirs)
        idata = sample_model(model)

        p_samples = idata.posterior['p_damage'].values
        p_flat = p_samples.reshape(-1, p_samples.shape[-1])
        p_mean_per_grid = np.mean(p_flat, axis=0)

        # Extract posterior for tau, mu
        posterior = idata.posterior
        mu_mean = float(posterior['mu'].values.mean())
        tau_mean = float(posterior['tau'].values.mean())

        summary = az.summary(idata)
        rhat_max = float(summary['r_hat'].max())
        div = int(idata.sample_stats.diverging.sum().values)

        rows.append({
            'model': label,
            'mu_post_mean': mu_mean,
            'tau_post_mean': tau_mean,
            'p_mean_range': f"[{p_mean_per_grid.min():.4f}, {p_mean_per_grid.max():.4f}]",
            'p_mean_mean': float(np.mean(p_mean_per_grid)),
            'p_range_width': float(p_mean_per_grid.max() - p_mean_per_grid.min()),
            'rhat_max': rhat_max,
            'divergences': div,
        })

        results[label] = {
            'idata': idata,
            'p_mean': p_mean_per_grid,
            'p_flat': p_flat,
        }

        print(f"    p_i range: [{p_mean_per_grid.min():.4f}, {p_mean_per_grid.max():.4f}]")
        print(f"    mu={mu_mean:.3f}, tau={tau_mean:.3f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_TAB / "sensor_only_comparison.csv", index=False)
    print(f"\n  Summary → sensor_only_comparison.csv")

    # 图: 三模型 p_i 地图对比
    gdf = gpd.read_file(FUSION_GEOJSON)
    if gdf.crs is not None and gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (label, res) in zip(axes, results.items()):
        plot_gdf = gdf.copy()
        plot_gdf['p_mean'] = res['p_mean']
        plot_gdf.plot(column='p_mean', ax=ax, cmap='RdYlGn_r',
                      legend=True, vmin=0, vmax=1,
                      legend_kwds={'label': 'p(damage)', 'shrink': 0.5})
        ax.set_title(f'{label}\nrange=[{res["p_mean"].min():.4f}, {res["p_mean"].max():.4f}]')
        ax.set_aspect('auto')

    plt.tight_layout()
    fig.savefig(OUT_FIG / "sensor_only_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figure → sensor_only_comparison.png")

    return result_df


# ============================================================
# 4. MCMC 诊断复查
# ============================================================

def run_mcmc_diagnostics(data):
    """详细 MCMC 诊断: ESS, R-hat, target_accept 对比."""
    print("\n" + "=" * 60)
    print("审计 4: MCMC 诊断复查")
    print("=" * 60)

    # 用 target_accept=0.95 和 0.99 各跑一次
    rows = []
    for ta in [0.90, 0.95, 0.99]:
        print(f"\n  target_accept = {ta}")
        model = build_model_full(data)
        idata = sample_model(model, target_accept=ta)

        summary = az.summary(idata)

        # 关键参数诊断
        key_params = ['mu', 'tau', 'sigma_s2', 'sigma_viirs', 'delta_s2']
        # Also get p_damage min/max R-hat
        p_damage_rows = summary[summary.index.str.startswith('p_damage')]
        p_rhat_max = float(p_damage_rows['r_hat'].max()) if len(p_damage_rows) > 0 else np.nan

        row = {'target_accept': ta}
        for p in key_params:
            param_rows = summary[summary.index == p]
            if len(param_rows) > 0:
                row[f'{p}_rhat'] = float(param_rows['r_hat'].values[0])
                row[f'{p}_ess_bulk'] = float(param_rows['ess_bulk'].values[0])
                row[f'{p}_ess_tail'] = float(param_rows['ess_tail'].values[0])

        row['p_damage_rhat_max'] = p_rhat_max
        row['divergences'] = int(idata.sample_stats.diverging.sum().values)
        # elapsed_time might be under 'sample_stats' or not present in newer ArviZ
        try:
            row['total_time_s'] = float(idata.sample_stats.elapsed_time.values.sum())
        except AttributeError:
            row['total_time_s'] = np.nan

        rows.append(row)

        print(f"    div={row['divergences']}, "
              f"tau_rhat={row.get('tau_rhat', np.nan):.4f}, "
              f"tau_ess_bulk={row.get('tau_ess_bulk', np.nan):.0f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_TAB / "mcmc_diagnostics.csv", index=False)
    print(f"\n  Summary → mcmc_diagnostics.csv")
    print(result_df.to_string(index=False))

    return result_df


# ============================================================
# 5. 残差空间结构
# ============================================================

def run_residual_analysis(data):
    """计算残差, 检查空间自相关."""
    print("\n" + "=" * 60)
    print("审计 5: 残差空间结构")
    print("=" * 60)

    # 用主模型 fit
    print("  Fitting main model...")
    model = build_model_full(data)
    idata = sample_model(model)

    posterior = idata.posterior
    theta_samples = posterior['theta'].values  # (chain, draw, n)
    delta_s2_samples = posterior['delta_s2'].values  # (chain, draw)
    delta_viirs_samples = posterior['delta_viirs'].values  # (chain, draw)

    # 计算每个网格的期望残差
    n_chains, n_draws, n_grids = theta_samples.shape

    # Mean posterior theta + delta
    theta_mean = np.mean(theta_samples, axis=(0, 1))  # (n,)
    delta_s2_mean = np.mean(delta_s2_samples)
    delta_viirs_mean = np.mean(delta_viirs_samples)

    # Residuals: observed - expected
    obs_s2 = data['y_s2']
    obs_viirs = data['y_viirs']

    resid_s2 = np.full(n_grids, np.nan)
    resid_viirs = np.full(n_grids, np.nan)

    mask_s2 = ~np.isnan(obs_s2)
    mask_viirs = ~np.isnan(obs_viirs)

    resid_s2[mask_s2] = obs_s2[mask_s2] - (theta_mean[mask_s2] + delta_s2_mean)
    resid_viirs[mask_viirs] = obs_viirs[mask_viirs] - (theta_mean[mask_viirs] + delta_viirs_mean)

    print(f"  S2 residual: mean={np.nanmean(resid_s2):.4f}, std={np.nanstd(resid_s2):.4f}, "
          f"range=[{np.nanmin(resid_s2):.2f}, {np.nanmax(resid_s2):.2f}]")
    print(f"  VIIRS residual: mean={np.nanmean(resid_viirs):.4f}, std={np.nanstd(resid_viirs):.4f}, "
          f"range=[{np.nanmin(resid_viirs):.2f}, {np.nanmax(resid_viirs):.2f}]")

    # 保存残差到 CSV
    resid_df = pd.DataFrame({
        'grid_id': np.arange(n_grids),
        'resid_s2': resid_s2,
        'resid_viirs': resid_viirs,
        'theta_mean': theta_mean,
    })
    resid_df.to_csv(OUT_TAB / "bayesian_residuals.csv", index=False)

    # 图: 残差地图
    gdf = gpd.read_file(FUSION_GEOJSON)
    if gdf.crs is not None and gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # S2 残差
    ax1 = axes[0]
    plot_gdf = gdf.copy()
    plot_gdf['resid'] = resid_s2
    vmax = max(abs(np.nanmin(resid_s2)), abs(np.nanmax(resid_s2)))
    plot_gdf.plot(column='resid', ax=ax1, cmap='RdBu_r',
                  legend=True, vmin=-vmax, vmax=vmax,
                  legend_kwds={'label': 'Residual (logit)', 'shrink': 0.6})
    ax1.set_title(f'S2 Residual (logit scale)\nmean={np.nanmean(resid_s2):.3f}, std={np.nanstd(resid_s2):.3f}')
    ax1.set_aspect('auto')

    # VIIRS 残差
    ax2 = axes[1]
    plot_gdf['resid'] = resid_viirs
    vmax = max(abs(np.nanmin(resid_viirs)), abs(np.nanmax(resid_viirs)))
    plot_gdf.plot(column='resid', ax=ax2, cmap='RdBu_r',
                  legend=True, vmin=-vmax, vmax=vmax,
                  legend_kwds={'label': 'Residual (logit)', 'shrink': 0.6})
    ax2.set_title(f'VIIRS Residual (logit scale)\nmean={np.nanmean(resid_viirs):.3f}, std={np.nanstd(resid_viirs):.3f}')
    ax2.set_aspect('auto')

    plt.tight_layout()
    fig.savefig(OUT_FIG / "residual_spatial_map.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figure → residual_spatial_map.png")

    # 残差 vs 观测散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, obs, resid, label in [
        (axes[0], obs_s2, resid_s2, 'S2'),
        (axes[1], obs_viirs, resid_viirs, 'VIIRS'),
    ]:
        valid = ~np.isnan(obs) & ~np.isnan(resid)
        ax.scatter(obs[valid], resid[valid], alpha=0.3, s=3, c='#333333')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'Observed {label} (logit)')
        ax.set_ylabel('Residual (logit)')
        ax.set_title(f'{label}: Observed vs Residual')

    plt.tight_layout()
    fig.savefig(OUT_FIG / "residual_vs_observed.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Figure → residual_vs_observed.png")

    return resid_df


# ============================================================
# 审计总结
# ============================================================

def write_audit_report(ppc_df, prior_df, baseline_df, mcmc_df, resid_df):
    """生成审计报告."""
    lines = [
        f"# Bayesian 模型审计报告",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. PPC 结果",
        "",
        "| Sensor | Obs Mean | Obs Std | Pred Mean | Mean Error |",
        "|--------|----------|---------|-----------|------------|",
    ]
    for _, r in ppc_df.iterrows():
        lines.append(f"| {r['sensor']} | {r['obs_mean']:.4f} | {r['obs_std']:.4f} | "
                     f"{r['pred_mean_mean']:.4f} | {r['mean_error']:.4f} |")

    lines += [
        "",
        "## 2. Prior Sensitivity",
        "",
        "| Prior | p_i Range Width | τ Post Mean | σ_s2 Post Mean | σ_viirs Post Mean |",
        "|-------|----------------|-------------|----------------|-------------------|",
    ]
    for _, r in prior_df.iterrows():
        lines.append(f"| {r['prior_label']} | {r['p_mean_range_width']:.6f} | "
                     f"{r['tau_post_mean']:.4f} | {r['sigma_s2_post_mean']:.3f} | "
                     f"{r['sigma_viirs_post_mean']:.3f} |")

    lines += [
        "",
        "## 3. Sensor-only Baseline",
        "",
        "| Model | μ Post Mean | p_i Range Width | R-hat Max | Div |",
        "|-------|------------|----------------|-----------|-----|",
    ]
    for _, r in baseline_df.iterrows():
        lines.append(f"| {r['model']} | {r['mu_post_mean']:.4f} | "
                     f"{r['p_range_width']:.6f} | {r['rhat_max']:.4f} | "
                     f"{r['divergences']} |")

    lines += [
        "",
        "## 4. MCMC 诊断",
        "",
        "| target_accept | Div | tau R-hat | tau ESS bulk |",
        "|---------------|-----|-----------|-------------|",
    ]
    for _, r in mcmc_df.iterrows():
        lines.append(f"| {r['target_accept']} | {r['divergences']} | "
                     f"{r.get('tau_rhat', 'N/A')} | {r.get('tau_ess_bulk', 'N/A')} |")

    lines += [
        "",
        "## 5. 残差分析",
        "",
        f"- S2 残差: mean={resid_df['resid_s2'].mean():.4f}, std={resid_df['resid_s2'].std():.4f}",
        f"- VIIRS 残差: mean={resid_df['resid_viirs'].mean():.4f}, std={resid_df['resid_viirs'].std():.4f}",
        "",
        "## 6. 结论",
        "",
        "### p_i 极窄是数据支持还是模型收缩过度?",
        "",
        "待审计完成后填写.",
        "",
        "### 是否可将当前 posterior 用于模块 2?",
        "",
        "待审计完成后填写.",
        "",
        "## 7. 输出文件",
        "",
        f"| `outputs/tables/ppc_summary.csv` | PPC 汇总 |",
        f"| `outputs/tables/prior_sensitivity_summary.csv` | Prior 敏感性 |",
        f"| `outputs/tables/sensor_only_comparison.csv` | 传感器基线 |",
        f"| `outputs/tables/mcmc_diagnostics.csv` | MCMC 诊断 |",
        f"| `outputs/tables/bayesian_residuals.csv` | 残差 |",
        f"| `outputs/figures/ppc_s2_viirs.png` | PPC 图 |",
        f"| `outputs/figures/prior_sensitivity.png` | Prior 敏感性图 |",
        f"| `outputs/figures/sensor_only_comparison.png` | 传感器对比图 |",
        f"| `outputs/figures/residual_spatial_map.png` | 残差地图 |",
        f"| `outputs/figures/residual_vs_observed.png` | 残差散点图 |",
        f"| `outputs/bayesian_audit_report.md` | 本报告 |",
    ]

    with open(OUT_MD / "bayesian_audit_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  Report → bayesian_audit_report.md")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("贝叶斯模型审计")
    print("=" * 60)

    data = prepare_data()
    print(f"  数据: {data['n']} grids, S2={np.sum(~np.isnan(data['y_s2']))}, "
          f"VIIRS={np.sum(~np.isnan(data['y_viirs']))}")

    # 1. PPC
    _, _ = run_ppc(data)

    # 2. Prior Sensitivity
    prior_df = run_prior_sensitivity(data)

    # 3. Sensor-only Baseline
    baseline_df = run_sensor_only_baseline(data)

    # 4. MCMC 诊断
    mcmc_df = run_mcmc_diagnostics(data)

    # 5. 残差
    resid_df = run_residual_analysis(data)

    # 报告
    ppc_summary = pd.read_csv(OUT_TAB / "ppc_summary.csv")
    write_audit_report(ppc_summary, prior_df, baseline_df, mcmc_df, resid_df)

    print("\n" + "=" * 60)
    print("审计完成.")
    print("=" * 60)


if __name__ == '__main__':
    main()
