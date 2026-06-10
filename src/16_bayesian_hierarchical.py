#!/usr/bin/env python3
"""
模块 1 — 贝叶斯层次模型 + MCMC 后验推断

T20 多传感器不确定性融合: 无监督 Logit-Normal 层次模型.

  Level 1 (观测):  logit(y_ik) ~ N(θ_i , σ²_k)     k ∈ {S2, VIIRS}
  Level 2 (潜在):  θ_i = logit(p_i) ~ N(μ , τ²)
  Level 3 (超先验): μ ~ N(0, 2²), τ ~ HalfCauchy(1), σ_k ~ HalfCauchy(0.5)

- 不需要 ground truth, 传感器噪声 σ_k 从数据估计
- 缺失观测自然处理 (单源网格 CI 更宽)
- 先验收缩: 数据少/冲突大时向 μ 回归
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

# 输出文件
POSTERIOR_CSV = OUT_DATA / "grid_bayesian_posterior.csv"
PARAM_CSV = OUT_TAB / "bayesian_parameter_summary.csv"
TRACE_PNG = OUT_FIG / "bayesian_trace_diagnostics.png"
POST_MAP_PNG = OUT_FIG / "bayesian_posterior_map.png"
SENSOR_PREC_PNG = OUT_FIG / "bayesian_sensor_precision.png"
REPORT_MD = OUT_MD / "bayesian_hierarchical_report.md"

MCMC_DRAWS = 1000
MCMC_TUNE = 1000
MCMC_CHAINS = 4
RANDOM_SEED = 42
EPS = 1e-8


# ============================================================
# 1. 数据准备
# ============================================================

def prepare_data(csv_path):
    """
    读取融合 CSV, 提取观测矩阵并做 clip + logit 变换.

    Returns
    -------
    data : dict
        y_s2, y_viirs : ndarray (N,)  已 logit 变换, NaN 表示缺失
        n_grids : int
        df : DataFrame  原始数据
    """
    df = pd.read_csv(csv_path)
    n = len(df)

    def clip_logit(values):
        """将 [0,1] 概率 clip 到安全范围后做 logit"""
        v = np.asarray(values, dtype=np.float64)
        v_clipped = np.clip(v, EPS, 1.0 - EPS)
        return np.log(v_clipped / (1.0 - v_clipped))

    y_s2 = clip_logit(df['p_s2_mean'].values)
    y_viirs = clip_logit(df['p_viirs_mean'].values)

    # 标记缺失
    y_s2[np.isnan(df['p_s2_mean'].values)] = np.nan
    y_viirs[np.isnan(df['p_viirs_mean'].values)] = np.nan

    print(f"  总网格: {n}")
    print(f"  S2 有效: {np.sum(~np.isnan(y_s2))}")
    print(f"  VIIRS 有效: {np.sum(~np.isnan(y_viirs))}")
    print(f"  双源: {np.sum(~np.isnan(y_s2) & ~np.isnan(y_viirs))}")
    print(f"  无数据: {np.sum(np.isnan(y_s2) & np.isnan(y_viirs))}")

    return {'y_s2': y_s2, 'y_viirs': y_viirs, 'n_grids': n, 'df': df}


# ============================================================
# 2. 构建 PyMC 模型
# ============================================================

def build_model(data):
    """
    构建 Logit-Normal 层次模型.

    θ_i ~ N(μ, τ²)      网格级 logit 损害概率
    y_ik ~ N(θ_i, σ²_k) 传感器观测

    缺失观测用 np.ma.masked_invalid 处理.
    """
    y_s2 = data['y_s2']
    y_viirs = data['y_viirs']
    n = data['n_grids']

    # 构建 masked array (PyMC 自动忽略 NaN)
    y_s2_masked = np.ma.masked_invalid(y_s2)
    y_viirs_masked = np.ma.masked_invalid(y_viirs)

    with pm.Model() as model:
        # --- 超先验 ---
        mu = pm.Normal('mu', mu=0.0, sigma=2.0)           # 全域平均 logit 损害
        tau = pm.HalfCauchy('tau', beta=1.0)               # 网格间标准差

        sigma_s2 = pm.HalfCauchy('sigma_s2', beta=0.5)     # S2 传感器噪声
        sigma_viirs = pm.HalfCauchy('sigma_viirs', beta=0.5)  # VIIRS 传感器噪声

        # --- 传感器偏置 (S2 vs VIIRS 系统性差异) ---
        # δ_s2: S2 偏置, VIIRS 偏置 = -δ_s2 (sum-to-zero)
        delta_s2 = pm.Normal('delta_s2', mu=0.0, sigma=1.0)
        delta_viirs = pm.Deterministic('delta_viirs', -delta_s2)

        # --- 网格级随机效应 (非中心化参数化, 避免 Neal's Funnel) ---
        theta_raw = pm.Normal('theta_raw', mu=0.0, sigma=1.0, shape=n)
        theta = pm.Deterministic('theta', mu + tau * theta_raw)

        # --- 观测似然 (含传感器偏置) ---
        pm.Normal('y_s2_obs', mu=theta + delta_s2, sigma=sigma_s2,
                  observed=y_s2_masked)
        pm.Normal('y_viirs_obs', mu=theta + delta_viirs, sigma=sigma_viirs,
                  observed=y_viirs_masked)

        # --- 导出量: 后验概率 p_i = inv_logit(θ_i) ---
        p_damage = pm.Deterministic('p_damage', pm.math.invlogit(theta))

    return model


# ============================================================
# 3. MCMC 采样
# ============================================================

def run_mcmc(model, draws=MCMC_DRAWS, tune=MCMC_TUNE, chains=MCMC_CHAINS, seed=RANDOM_SEED):
    """NUTS 采样并返回 InferenceData."""
    print(f"  开始 MCMC: {chains} chains × {tune} tune + {draws} draws ...")
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            random_seed=seed,
            target_accept=0.95,  # 非中心化 + 高 accept 减少 divergences
            progressbar=True,
        )

    # 收敛诊断
    summary = az.summary(idata)
    rhat_max = float(summary['r_hat'].max())
    divergences = int(idata.sample_stats.diverging.sum().values)

    print(f"  R-hat max: {rhat_max:.4f}")
    print(f"  Divergences: {divergences}")

    if rhat_max > 1.05:
        print(f"  WARNING: R-hat > 1.05, consider increasing tune/draws")
    if divergences > 0:
        print(f"  WARNING: {divergences} divergences, consider increasing target_accept")

    return idata


# ============================================================
# 4. 后验提取
# ============================================================

def extract_posterior(idata, df):
    """
    从 InferenceData 提取网格级后验统计.

    Returns pd.DataFrame: grid_id + posterior_mean/median/sd/ci2.5/ci97.5/r_hat
    """
    posterior = idata.posterior

    # 网格级 p_damage
    p_samples = posterior['p_damage'].values  # (chain, draw, n_grids)
    # Stack chains
    p_chain_draw = p_samples.reshape(-1, p_samples.shape[-1])  # (chain*draw, n_grids)

    result = pd.DataFrame({
        'grid_id': df['grid_id'].values,
        'posterior_mean': np.mean(p_chain_draw, axis=0),
        'posterior_median': np.median(p_chain_draw, axis=0),
        'posterior_sd': np.std(p_chain_draw, axis=0),
        'ci_2_5': np.percentile(p_chain_draw, 2.5, axis=0),
        'ci_97_5': np.percentile(p_chain_draw, 97.5, axis=0),
        'ci_width': np.percentile(p_chain_draw, 97.5, axis=0) - np.percentile(p_chain_draw, 2.5, axis=0),
    })

    # R-hat per grid
    rhat_vals = az.rhat(idata, var_names=['p_damage'])
    if hasattr(rhat_vals, 'to_dataframe'):
        result['r_hat'] = rhat_vals.to_dataframe()['p_damage'].values
    else:
        # Fallback for older ArviZ
        result['r_hat'] = float(rhat_vals['p_damage'].values.mean())

    # Theta posterior
    theta_samples = posterior['theta'].values.reshape(-1, posterior['theta'].shape[-1])
    result['theta_mean'] = np.mean(theta_samples, axis=0)
    result['theta_sd'] = np.std(theta_samples, axis=0)

    # 合并原始字段（便于下游使用）
    for col in ['grid_row', 'grid_col', 'lon_center', 'lat_center',
                'p_s2_mean', 'p_s2_std', 'p_s2_valid_ratio',
                'p_viirs_mean', 'p_viirs_std', 'p_viirs_valid_ratio',
                'p_fused', 'evidence_type', 'fusion_source']:
        if col in df.columns:
            result[col] = df[col].values

    print(f"  后验均值范围: [{result['posterior_mean'].min():.4f}, {result['posterior_mean'].max():.4f}]")
    print(f"  后验中位数范围: [{result['posterior_median'].min():.4f}, {result['posterior_median'].max():.4f}]")
    print(f"  CI 宽度均值: {result['ci_width'].mean():.4f}")
    print(f"  R-hat max (grid): {result['r_hat'].max():.4f}")

    return result


def extract_hyperparams(idata):
    """提取超参数后验统计."""
    params = ['mu', 'tau', 'sigma_s2', 'sigma_viirs', 'delta_s2']
    rows = []
    posterior = idata.posterior
    summary = az.summary(idata)

    for p in params:
        samples = posterior[p].values.flatten()
        # Get R-hat from summary
        param_rows = summary[summary.index == p]
        r_hat_val = float(param_rows['r_hat'].values[0]) if len(param_rows) > 0 else np.nan
        rows.append({
            'parameter': p,
            'mean': float(np.mean(samples)),
            'median': float(np.median(samples)),
            'sd': float(np.std(samples)),
            'ci_2_5': float(np.percentile(samples, 2.5)),
            'ci_97_5': float(np.percentile(samples, 97.5)),
            'r_hat': r_hat_val,
        })

    param_df = pd.DataFrame(rows)
    print("\n  超参数后验:")
    for _, r in param_df.iterrows():
        print(f"    {r['parameter']:12s}  mean={r['mean']:.3f}  "
              f"ci=[{r['ci_2_5']:.3f}, {r['ci_97_5']:.3f}]  R-hat={r['r_hat']:.3f}")

    return param_df


# ============================================================
# 5. 可视化
# ============================================================

def make_trace_plot(idata):
    """生成 trace 诊断图: 超参数后验密度 + trace 手动绘制."""
    posterior = idata.posterior
    params = ['mu', 'tau', 'sigma_s2', 'sigma_viirs', 'delta_s2']
    n_params = len(params)

    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8))

    for j, param in enumerate(params):
        samples = posterior[param].values  # (chain, draw)

        # Top row: trace (overlay chains)
        ax_trace = axes[0, j]
        n_chains = samples.shape[0]
        for c in range(n_chains):
            ax_trace.plot(samples[c, :], alpha=0.5, lw=0.5)
        ax_trace.set_title(param)
        ax_trace.set_xlabel('draw')

        # Bottom row: posterior density
        ax_dens = axes[1, j]
        ax_dens.hist(samples.flatten(), bins=50, density=True, alpha=0.7,
                     color='#333333')
        # Add vertical lines for mean and 95% CI
        flat = samples.flatten()
        ax_dens.axvline(np.mean(flat), color='red', lw=1.5, linestyle='-',
                        label=f'mean={np.mean(flat):.3f}')
        ax_dens.axvline(np.percentile(flat, 2.5), color='blue', lw=1, linestyle='--')
        ax_dens.axvline(np.percentile(flat, 97.5), color='blue', lw=1, linestyle='--')
        ax_dens.set_xlabel('value')
        ax_dens.legend(fontsize=7)

    fig.suptitle('Bayesian Hierarchical Model — Hyperparameter Diagnostics',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(TRACE_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Trace plot → {TRACE_PNG.name}")


def make_posterior_map(posterior_df, geojson_path):
    """生成后验均值 + CI 宽度地图."""
    gdf = gpd.read_file(geojson_path)

    # Merge
    plot_gdf = gdf.merge(
        posterior_df[['grid_id', 'posterior_mean', 'posterior_sd', 'ci_width']],
        on='grid_id', how='left'
    )
    # 转为 EPSG:4326 用于 matplotlib
    if plot_gdf.crs is not None and plot_gdf.crs != 'EPSG:4326':
        plot_gdf = plot_gdf.to_crs('EPSG:4326')

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 左: 后验均值 (p_damage)
    ax1 = axes[0]
    plot_gdf.plot(column='posterior_mean', ax=ax1, cmap='RdYlGn_r',
                  legend=True, vmin=0, vmax=1,
                  legend_kwds={'label': 'Posterior Mean p(damage)', 'shrink': 0.6})

    # 右: CI 宽度
    ax2 = axes[1]
    plot_gdf.plot(column='ci_width', ax=ax2, cmap='viridis',
                  legend=True,
                  legend_kwds={'label': '95% CI Width', 'shrink': 0.6})

    for ax, title in zip(axes, ['Bayesian Posterior Mean (p_damage)', '95% Credible Interval Width']):
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('auto')

    plt.tight_layout()
    fig.savefig(POST_MAP_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Posterior map → {POST_MAP_PNG.name}")


def make_sensor_precision_plot(idata, posterior_df):
    """传感器噪声 vs 数据质量对照."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: σ_s2 vs σ_viirs 联合后验散点
    ax1 = axes[0]
    sigma_s2 = idata.posterior['sigma_s2'].values.flatten()
    sigma_viirs = idata.posterior['sigma_viirs'].values.flatten()
    # 随机取 2000 个点
    idx = np.random.choice(len(sigma_s2), min(2000, len(sigma_s2)), replace=False)
    ax1.scatter(sigma_s2[idx], sigma_viirs[idx], alpha=0.3, s=3, c='#333333')
    ax1.set_xlabel('σ_s2 (S2 sensor noise)')
    ax1.set_ylabel('σ_viirs (VIIRS sensor noise)')
    ax1.set_title('Joint Posterior: Sensor Noise')
    ax1.axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5,
               label='σ_s2 = σ_viirs')
    ax1.legend()

    # 右: evidence_type 分组的后验均值分布
    ax2 = axes[1]
    valid = posterior_df.dropna(subset=['evidence_type', 'posterior_mean'])
    types = ['both_high', 'both_low', 's2_low_viirs_high', 's2_high_viirs_low',
             'single_source']
    type_labels = []
    data_groups = []
    for et in types:
        subset = valid[valid['evidence_type'] == et]['posterior_mean'].values
        if len(subset) > 0:
            data_groups.append(subset)
            type_labels.append(f'{et}\n(n={len(subset)})')

    bp = ax2.boxplot(data_groups, labels=type_labels, patch_artist=True,
                     showfliers=False)
    for patch, color in zip(bp['boxes'],
                            ['#2E7D32', '#BDBDBD', '#FF9800', '#2196F3', '#9C27B0']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('Posterior Mean p(damage)')
    ax2.set_title('Bayesian Posterior by Evidence Type')
    plt.setp(ax2.get_xticklabels(), rotation=15, ha='right', fontsize=8)

    plt.tight_layout()
    fig.savefig(SENSOR_PREC_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sensor precision plot → {SENSOR_PREC_PNG.name}")


# ============================================================
# 6. 报告
# ============================================================

def write_report(idata, posterior_df, param_df):
    """生成 Markdown 报告."""
    p = posterior_df

    # 计算一些汇总
    both_mask = (~p['p_s2_mean'].isna()) & (~p['p_viirs_mean'].isna())
    both_df = p[both_mask]

    lines = [
        f"# 贝叶斯层次模型报告",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**模块**: 模块 1 — 贝叶斯层次模型 + MCMC",
        "",
        "## 1. 模型规格",
        "",
        "```",
        "Level 1 (观测):   logit(y_ik) ~ N(θ_i , σ²_k)      k ∈ {S2, VIIRS}",
        "Level 2 (潜在):   θ_i = logit(p_i) ~ N(μ , τ²)",
        "Level 3 (超先验): μ ~ N(0, 2²),  τ ~ HalfCauchy(1)",
        "                  σ_k ~ HalfCauchy(0.5)",
        "```",
        "",
        f"- MCMC: {MCMC_CHAINS} chains × {MCMC_TUNE} tune + {MCMC_DRAWS} draws (NUTS)",
        f"- 总网格: {len(p)}, 双源: {both_mask.sum()}, S2 only: {(~p['p_s2_mean'].isna() & p['p_viirs_mean'].isna()).sum()}, VIIRS only: {(p['p_s2_mean'].isna() & ~p['p_viirs_mean'].isna()).sum()}",
        "",
        "## 2. 超参数后验",
        "",
        "| Parameter | Mean | Median | SD | 2.5% | 97.5% | R-hat |",
        "|-----------|------|--------|-----|------|-------|-------|",
    ]

    for _, r in param_df.iterrows():
        lines.append(f"| {r['parameter']} | {r['mean']:.4f} | {r['median']:.4f} | "
                     f"{r['sd']:.4f} | {r['ci_2_5']:.4f} | {r['ci_97_5']:.4f} | "
                     f"{r['r_hat']:.4f} |")

    lines += [
        "",
        "**关键解读**:",
        f"- σ_s2 = {param_df[param_df['parameter']=='sigma_s2']['mean'].values[0]:.4f} "
        f"  → S2 传感器在 logit 尺度的噪声水平",
        f"- σ_viirs = {param_df[param_df['parameter']=='sigma_viirs']['mean'].values[0]:.4f} "
        f"  → VIIRS 传感器在 logit 尺度的噪声水平",
        f"- τ = {param_df[param_df['parameter']=='tau']['mean'].values[0]:.4f} "
        f"  → 网格间损害的异质性 (大于传感器噪声 → 网格间有实质差异)",
        "",
        "## 3. 后验概率汇总",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 后验均值 mean(p_i) | {p['posterior_mean'].mean():.4f} |",
        f"| 后验均值 median(p_i) | {p['posterior_mean'].median():.4f} |",
        f"| 95% CI 平均宽度 | {p['ci_width'].mean():.4f} |",
        f"| 后验 SD 均值 | {p['posterior_sd'].mean():.4f} |",
        f"| R-hat max (grid) | {p['r_hat'].max():.4f} |",
        "",
        "### 按 Evidence Type 分组",
        "",
        "| Evidence Type | 数量 | Post. Mean | Post. SD | CI Width |",
        "|---------------|------|-----------|----------|----------|",
    ]

    for et in ['both_high', 'both_low', 's2_low_viirs_high', 's2_high_viirs_low',
               'single_source']:
        sub = p[p['evidence_type'] == et]
        if len(sub) > 0:
            lines.append(f"| {et} | {len(sub)} | {sub['posterior_mean'].mean():.4f} | "
                         f"{sub['posterior_sd'].mean():.4f} | {sub['ci_width'].mean():.4f} |")

    lines += [
        "",
        "## 4. 收敛诊断",
        "",
        f"- R-hat max (超参数): {param_df['r_hat'].max():.4f}",
        f"- R-hat max (网格 p_i): {p['r_hat'].max():.4f}",
        f"- Divergences: {int(idata.sample_stats.diverging.sum().values)}",
        "",
        "## 5. 与 D-S / Logit 融合对比",
        "",
        f"| 方法 | 均值 | 中位数 | 标准差 |",
        f"|------|------|--------|--------|",
        f"| Logit p_fused | {p['p_fused'].mean():.4f} | {p['p_fused'].median():.4f} | {p['p_fused'].std():.4f} |",
        f"| Bayesian post. mean | {p['posterior_mean'].mean():.4f} | {p['posterior_mean'].median():.4f} | {p['posterior_mean'].std():.4f} |",
        "",
        "## 6. 输出文件",
        "",
        f"| `{POSTERIOR_CSV.relative_to(PROJECT_ROOT)}` | 后验统计 CSV |",
        f"| `{PARAM_CSV.relative_to(PROJECT_ROOT)}` | 超参数后验 |",
        f"| `{TRACE_PNG.relative_to(PROJECT_ROOT)}` | Trace 诊断图 |",
        f"| `{POST_MAP_PNG.relative_to(PROJECT_ROOT)}` | 后验地图 |",
        f"| `{SENSOR_PREC_PNG.relative_to(PROJECT_ROOT)}` | 传感器精度对比 |",
        f"| `{REPORT_MD.relative_to(PROJECT_ROOT)}` | 本报告 |",
    ]

    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report → {REPORT_MD.name}")


# ============================================================
# 7. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("模块 1: 贝叶斯层次模型 + MCMC")
    print("=" * 60)

    # 7.1 准备数据
    print("\n[1/5] 准备数据...")
    data = prepare_data(FUSION_CSV)

    # 7.2 构建模型
    print("\n[2/5] 构建 PyMC 模型...")
    model = build_model(data)
    print(f"  模型变量: {[v.name for v in model.free_RVs]}")
    print(f"  观测变量: {[v.name for v in model.observed_RVs]}")

    # 7.3 MCMC 采样
    print("\n[3/5] MCMC 采样...")
    idata = run_mcmc(model)

    # 7.4 提取后验
    print("\n[4/5] 提取后验...")
    posterior_df = extract_posterior(idata, data['df'])
    param_df = extract_hyperparams(idata)

    # 保存
    posterior_df.to_csv(POSTERIOR_CSV, index=False)
    print(f"  后验 CSV → {POSTERIOR_CSV.name}")
    param_df.to_csv(PARAM_CSV, index=False)
    print(f"  参数 CSV → {PARAM_CSV.name}")

    # 7.5 可视化 + 报告
    print("\n[5/5] 可视化 + 报告...")
    make_trace_plot(idata)
    make_posterior_map(posterior_df, FUSION_GEOJSON)
    make_sensor_precision_plot(idata, posterior_df)
    write_report(idata, posterior_df, param_df)

    print("\n" + "=" * 60)
    print("模块 1 完成.")
    print("=" * 60)


if __name__ == '__main__':
    main()
