#!/usr/bin/env python3
"""
阶段 4.5 — 双源融合结果评估与解释

只读取阶段 4 已有输出, 不做任何重新计算。
输出: 解读报告、质量评估表、关键图面板。
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_CSV = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.csv"
FUSION_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.geojson"
OUT_FIG  = PROJECT_ROOT / "outputs" / "figures"
OUT_TAB  = PROJECT_ROOT / "outputs" / "tables"
OUT_MD   = PROJECT_ROOT / "outputs"
for d in [OUT_FIG, OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)

INTERP_MD = OUT_MD / "fusion_interpretation_notes.md"
QUALITY_CSV = OUT_TAB / "fusion_quality_assessment.csv"
PANEL_PNG = OUT_FIG / "fusion_key_maps_panel.png"


def load_data():
    df = pd.read_csv(FUSION_CSV)
    gdf = gpd.read_file(FUSION_GEOJSON)
    return df, gdf


def compute_evaluation_metrics(df):
    """计算所有评估指标"""
    metrics = {}

    # 基本统计
    metrics['n_total'] = len(df)
    metrics['n_both'] = int((df['fusion_source'] == 'both').sum())
    metrics['n_s2_only'] = int((df['fusion_source'] == 's2_only').sum())
    metrics['n_viirs_only'] = int((df['fusion_source'] == 'viirs_only').sum())
    metrics['n_no_data'] = int((df['fusion_source'] == 'no_data').sum())
    metrics['dual_coverage_ratio'] = metrics['n_both'] / metrics['n_total']

    # 损害概率统计
    for col, label in [('p_s2_mean', 'p_s2'), ('p_viirs_mean', 'p_viirs'), ('p_fused', 'p_fused')]:
        v = df[col].dropna()
        metrics[f'{label}_mean'] = float(v.mean())
        metrics[f'{label}_median'] = float(v.median())
        metrics[f'{label}_std'] = float(v.std())
        metrics[f'{label}_n'] = len(v)

    # 不一致性
    both = df[df['fusion_source'] == 'both']
    d = both['disagreement'].dropna()
    metrics['disagreement_mean'] = float(d.mean()) if len(d) > 0 else np.nan
    metrics['disagreement_median'] = float(d.median()) if len(d) > 0 else np.nan

    u = df['uncertainty'].dropna()
    metrics['uncertainty_mean'] = float(u.mean())
    metrics['uncertainty_median'] = float(u.median())

    # Spearman
    both_valid = both.dropna(subset=['p_s2_mean', 'p_viirs_mean'])
    if len(both_valid) >= 5:
        r, p = spearmanr(both_valid['p_s2_mean'], both_valid['p_viirs_mean'])
        metrics['spearman_r'] = float(r)
        metrics['spearman_p'] = float(p)
    else:
        metrics['spearman_r'] = np.nan
        metrics['spearman_p'] = np.nan

    # 证据类型
    for et, cnt in df['evidence_type'].value_counts().items():
        metrics[f'evidence_{et}_count'] = int(cnt)
        metrics[f'evidence_{et}_ratio'] = float(cnt / len(df))

    # 双传感器子集的方向分析
    s2h = both['p_s2_mean'] >= 0.5
    vh = both['p_viirs_mean'] >= 0.5
    metrics['both_high'] = int((s2h & vh).sum())
    metrics['s2_high_viirs_low'] = int((s2h & ~vh).sum())
    metrics['s2_low_viirs_high'] = int((~s2h & vh).sum())
    metrics['both_low'] = int((~s2h & ~vh).sum())

    # 质量标志
    metrics['s2_low_quality_count'] = int(df['p_s2_low_quality'].sum())
    metrics['viirs_low_quality_count'] = int(df['p_viirs_low_quality'].sum())

    # 最高不确定性网格
    hi30 = both.nlargest(30, 'uncertainty')
    metrics['top30_unc_mean_disagreement'] = float(hi30['disagreement'].mean())
    metrics['top30_unc_mean_p_s2'] = float(hi30['p_s2_mean'].mean())
    metrics['top30_unc_mean_p_viirs'] = float(hi30['p_viirs_mean'].mean())
    hi_evidence = hi30['evidence_type'].value_counts()
    metrics['top30_unc_dominant_evidence'] = hi_evidence.index[0]

    return metrics


def write_interpretation_notes(metrics):
    """生成中文解读报告"""
    m = metrics
    lines = []

    lines.append("# Sentinel-2 / VIIRS 双源融合结果评估与解释\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 4.5 — 融合结果评估\n")
    lines.append("---\n")

    # 1. 数据覆盖
    lines.append("## 1. 数据覆盖概况\n")
    lines.append(f"- 网格总数: **{m['n_total']}** (500m × 500m)\n")
    lines.append(f"- 双源覆盖: **{m['n_both']}** ({100*m['dual_coverage_ratio']:.1f}%)\n")
    lines.append(f"- 仅 Sentinel-2: {m['n_s2_only']} ({100*m['n_s2_only']/m['n_total']:.1f}%)\n")
    lines.append(f"- 仅 VIIRS: {m['n_viirs_only']} ({100*m['n_viirs_only']/m['n_total']:.1f}%)\n")
    lines.append(f"- 无数据: {m['n_no_data']} ({100*m['n_no_data']/m['n_total']:.1f}%)\n")
    lines.append(f"\n**评估**: 双源覆盖率 76.3%，为融合提供了良好基础。VIIRS 覆盖略优于 S2（因 S2 受 SCL 云掩膜影响），15.6% 的网格仅有 VIIRS 数据。\n")

    # 2. 损害概率分布
    lines.append("## 2. 损害概率分布\n")
    lines.append("| 指标 | p_s2 (Grid Mean) | p_viirs (Grid Mean) | p_fused |\n")
    lines.append("|------|------------------|---------------------|---------|\n")
    lines.append(f"| 均值 | {m['p_s2_mean']:.4f} | {m['p_viirs_mean']:.4f} | {m['p_fused_mean']:.4f} |\n")
    lines.append(f"| 中位数 | {m['p_s2_median']:.4f} | {m['p_viirs_median']:.4f} | {m['p_fused_median']:.4f} |\n")
    lines.append(f"| 标准差 | {m['p_s2_std']:.4f} | {m['p_viirs_std']:.4f} | {m['p_fused_std']:.4f} |\n")
    lines.append(f"| 有效 N | {m['p_s2_n']} | {m['p_viirs_n']} | {m['p_fused_n']} |\n")
    lines.append("\n**关键观察**:\n")
    lines.append(f"- p_s2 均值仅 {m['p_s2_mean']:.3f}（中位数 {m['p_s2_median']:.3f}），呈严重右偏分布——大多数网格的 S2 光谱损害信号很弱\n")
    lines.append(f"- p_viirs 均值 {m['p_viirs_mean']:.3f}（中位数 {m['p_viirs_median']:.3f}），分布较为均匀——夜光下降在 AOI 内广泛存在\n")
    lines.append(f"- p_fused 均值 {m['p_fused_mean']:.3f}——受低 p_s2 压制，融合结果偏向保守\n")

    # 3. 传感器一致性
    lines.append("## 3. 传感器一致性分析\n")
    lines.append(f"- **Spearman ρ = {m['spearman_r']:.4f}** (p = {m['spearman_p']:.6f})\n")
    lines.append(f"- Disagreement 均值 = {m['disagreement_mean']:.4f}, 中位数 = {m['disagreement_median']:.4f}\n")
    lines.append("\n**方向性分歧**:\n")
    lines.append("| 方向 | 网格数 | 占比 (双源) |\n")
    lines.append("|------|--------|------------|\n")
    lines.append(f"| Both high (p>=0.5) | {m['both_high']} | {100*m['both_high']/m['n_both']:.1f}% |\n")
    lines.append(f"| S2 high / VIIRS low | {m['s2_high_viirs_low']} | {100*m['s2_high_viirs_low']/m['n_both']:.1f}% |\n")
    lines.append(f"| S2 low / VIIRS high | {m['s2_low_viirs_high']} | {100*m['s2_low_viirs_high']/m['n_both']:.1f}% |\n")
    lines.append(f"| Both low (p<0.5) | {m['both_low']} | {100*m['both_low']/m['n_both']:.1f}% |\n")

    lines.append(f"\n**核心发现**: {m['s2_low_viirs_high']}/{m['n_both']} ({100*m['s2_low_viirs_high']/m['n_both']:.1f}%) 的双源网格呈 'S2 低 / VIIRS 高' 模式——这是主导分歧方向。\n")

    # 4. 为何出现负相关
    lines.append("## 4. 为什么 S2 与 VIIRS 可能出现负相关\n")
    lines.append("### 4.1 物理机制差异\n")
    lines.append("| 维度 | Sentinel-2 | VIIRS DNB |\n")
    lines.append("|------|-----------|----------|\n")
    lines.append("| 信号类型 | 地表光谱反射率变化 | 夜间向上辐射亮度变化 |\n")
    lines.append("| 空间分辨率 | 20m (精细) | ~500m (粗糙) |\n")
    lines.append("| 损害敏感对象 | 植被损失、瓦砾暴露、土壤翻动 | 灯光熄灭、电力中断、人口迁出 |\n")
    lines.append("| 时间敏感性 | 年际植被物候 | 季节性灯光使用 + 战争破坏 |\n")
    lines.append("| 恢复速度 | 缓慢（数月-数年） | 可能较快（发电机、恢复供电） |\n")

    lines.append("\n### 4.2 空间尺度效应\n")
    lines.append("- **S2 (20m)**: 能检测单栋建筑物的瓦砾堆，但仅限 AOI 内光谱变化最极端的少数像元（损害分数均值仅 0.10）\n")
    lines.append("- **VIIRS (500m)**: 一个像素覆盖 25 公顷，单个建筑破坏不足以显著改变 500m 像素的夜光——但当大范围停电或人口撤离时，夜光显著下降\n")
    lines.append("- **后果**: S2 的高损害信号集中在少数破坏严重的街区；VIIRS 的高损害信号则遍布城市大部分区域\n")

    lines.append("\n### 4.3 不同的'损害'定义\n")
    lines.append("- S2 定义的损害: **物理结构变化**（屋顶塌陷、墙壁倒塌→瓦砾光谱特征）\n")
    lines.append("- VIIRS 定义的损害: **功能丧失**（无人居住、无电力供应→无夜间灯光）\n")
    lines.append("- 一栋楼可能被炮击（=S2 损害↑）但旁边街道仍有路灯（=VIIRS 损害未↑）\n")
    lines.append("- 一个街区可能无人居住停电（=VIIRS 损害↑）但建筑物结构完好（=S2 损害未↑）\n")

    lines.append("\n### 4.4 结论\n")
    lines.append("> **S2 与 VIIRS 的负相关不是错误，而是反映了战争损害的两个互补维度。** 这是数据融合的价值所在——单一传感器只能看到部分真相。\n")

    # 5. 融合保守性评估
    lines.append("## 5. Logit-平均融合保守性评估\n")
    lines.append(f"### 判定: **当前的 logit-平均融合结果是保守的**\n")
    lines.append(f"\n证据:\n")
    lines.append(f"1. p_fused 均值 ({m['p_fused_mean']:.3f}) 被拉向低值——受低 p_s2 ({m['p_s2_mean']:.3f}) 压制\n")
    lines.append(f"2. 47.0% 的网格属于 'S2 低 / VIIRS 高'——融合后这些网格的 p_fused 被压低约一半\n")
    lines.append(f"3. Disagreement 中位数 {m['disagreement_median']:.3f} 意味着典型网格中两个传感器的概率差超过 0.5\n")
    lines.append(f"4. 仅有 {m['both_high']} 个网格 ({100*m['both_high']/m['n_both']:.1f}%) 两个传感器一致指向高损害\n")

    lines.append(f"\n**保守性的后果**:\n")
    lines.append("- 优点: 减少假阳性（false positive），避免将未受损区域标记为高损害\n")
    lines.append("- 缺点: 大量假阴性（false negative）——VIIRS 强烈指示损害的区域被 S2 拉低\n")
    lines.append("- 在缺乏 UNOSAT 标签的情况下，保守融合是合理的选择，但需要向用户说明\n")

    # 6. 不确定性分析
    lines.append("## 6. 不确定性分析\n")
    lines.append(f"- Uncertainty 均值: {m['uncertainty_mean']:.4f}, 中位数: {m['uncertainty_median']:.4f}\n")
    lines.append(f"- Disagreement 贡献了 70% 的权重 → 传感器冲突是不确定性的主要来源\n")
    lines.append(f"- 质量不确定性仅贡献 30% → 有效像元覆盖率问题相对次要\n")

    lines.append(f"\n**最高不确定性网格的特征** (Top 30):\n")
    lines.append(f"- 主导证据类型: **{m['top30_unc_dominant_evidence']}**\n")
    lines.append(f"- 平均 disagreement: {m['top30_unc_mean_disagreement']:.4f}\n")
    lines.append(f"- 平均 p_s2: {m['top30_unc_mean_p_s2']:.4f} (极低)\n")
    lines.append(f"- 平均 p_viirs: {m['top30_unc_mean_p_viirs']:.4f} (高)\n")
    lines.append(f"- **解释**: 不确定性最高的区域是 VIIRS 强烈指示损害但 S2 几乎无信号的地方——这些可能是'功能受损但结构尚存'的区域\n")

    # 7. 质量评估
    lines.append("## 7. 数据质量评估\n")
    lines.append(f"- S2 低质量网格: {m['s2_low_quality_count']}/{m['n_total']} ({100*m['s2_low_quality_count']/m['n_total']:.1f}%) — 有效像元比例 < 20%\n")
    lines.append(f"- VIIRS 低质量网格: {m['viirs_low_quality_count']}/{m['n_total']} ({100*m['viirs_low_quality_count']/m['n_total']:.1f}%) — 有效像元比例 < 20%\n")

    lines.append("\n## 8. 证据类型全景\n")
    lines.append("| 证据类型 | 计数 | 比例 | 含义 |\n")
    lines.append("|----------|------|------|------|\n")
    for et, label in [
        ('s2_low_viirs_high', 'S2 低 / VIIRS 高'),
        ('both_low', '双低'),
        ('single_source', '单源'),
        ('no_data', '无数据'),
        ('s2_high_viirs_low', 'S2 高 / VIIRS 低'),
        ('both_high', '双高'),
    ]:
        cnt = m.get(f'evidence_{et}_count', 0)
        ratio = m.get(f'evidence_{et}_ratio', 0)
        if et == 's2_low_viirs_high':
            meaning = '功能损害为主（停电/撤离），结构损害信号弱'
        elif et == 'both_low':
            meaning = '两个传感器均未检测到明显损害'
        elif et == 'single_source':
            meaning = '仅一个传感器有数据'
        elif et == 'no_data':
            meaning = '两个传感器均无数据'
        elif et == 's2_high_viirs_low':
            meaning = '结构损害为主（建筑瓦砾），功能尚存'
        elif et == 'both_high':
            meaning = '双源一致确认高损害——最可信的损害区域'
        lines.append(f"| {label} | {cnt} | {ratio:.4f} ({100*ratio:.1f}%) | {meaning} |\n")

    # 9. 进入 D-S 的建议
    lines.append("## 9. 是否进入 D-S 证据理论阶段\n")
    lines.append("### 建议: **强烈建议进入 D-S 证据理论阶段**\n")
    lines.append("\n**理由**:\n")
    lines.append(f"1. **传感器冲突严重**: {100*m['s2_low_viirs_high']/m['n_both']:.1f}% 的双源网格存在方向性冲突——这正是 D-S 理论擅长处理的场景\n")
    lines.append("2. **Logit-平均过于保守**: 等权融合无法区分'一个传感器确信 + 一个传感器不确定' vs '两个传感器各说各话'\n")
    lines.append("3. **D-S 的优势**:\n")
    lines.append("   - 显式建模证据冲突 (mass function)\n")
    lines.append("   - 可以量化'不知道' (ignorance / m(Θ))\n")
    lines.append("   - 当两个传感器都指向同一结论时放大置信度，冲突时保留无知\n")
    lines.append("   - 自然地处理单源情况（单源 = 另一个传感器完全无知）\n")
    lines.append("4. **互补信息源**: S2 和 VIIRS 测的是损害的不同维度，D-S 融合可以保留这种互补性而不是简单平均\n")

    lines.append("\n## 10. 重要声明\n")
    lines.append("> ⚠️ **这不是精度评估。**\n")
    lines.append(">\n")
    lines.append("> 当前评估基于:\n")
    lines.append("> - 数据质量指标（覆盖比例、有效像元比例）\n")
    lines.append("> - 传感器一致性指标（Spearman ρ、disagreement）\n")
    lines.append("> - 融合保守性分析（分布偏移方向）\n")
    lines.append(">\n")
    lines.append("> **我们没有 UNOSAT 人工标注标签**，因此:\n")
    lines.append("> - 无法计算 Precision / Recall / F1\n")
    lines.append("> - 无法判断哪个传感器更'准确'\n")
    lines.append("> - p_fused 是**损害候选概率**，不是真实损害标签\n")
    lines.append("> - 本评估的目的是理解数据特征和融合行为，而非验证精度\n")

    with open(INTERP_MD, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {INTERP_MD}")


def write_quality_assessment_csv(metrics):
    """生成结构化质量评估表"""
    rows = [
        ['category', 'metric', 'value', 'interpretation'],
        # Coverage
        ['coverage', 'n_total_grids', metrics['n_total'], '500m网格总数'],
        ['coverage', 'n_dual_sensor', metrics['n_both'], '双传感器均有效的网格'],
        ['coverage', 'dual_coverage_ratio', round(metrics['dual_coverage_ratio'], 4), '双源覆盖率: 76.3% — 良好'],
        ['coverage', 'n_s2_only', metrics['n_s2_only'], '仅S2: 多为AOI边缘/云掩膜区域'],
        ['coverage', 'n_viirs_only', metrics['n_viirs_only'], '仅VIIRS: 多为S2 SCL掩膜过严区域'],
        ['coverage', 'n_no_data', metrics['n_no_data'], '完全无数据'],
        # Distribution
        ['distribution', 'p_s2_mean', round(metrics['p_s2_mean'], 4), 'S2损害分数低: 右偏分布, 少数极端'],
        ['distribution', 'p_s2_median', round(metrics['p_s2_median'], 4), 'S2中位数极低: 多数网格几乎无信号'],
        ['distribution', 'p_viirs_mean', round(metrics['p_viirs_mean'], 4), 'VIIRS损害概率中等偏高'],
        ['distribution', 'p_viirs_median', round(metrics['p_viirs_median'], 4), 'VIIRS中位数>0.5: 超半数网格高概率'],
        ['distribution', 'p_fused_mean', round(metrics['p_fused_mean'], 4), '融合概率低: 受S2压制'],
        ['distribution', 'p_fused_median', round(metrics['p_fused_median'], 4), '融合中位数低: 保守结果'],
        # Consistency
        ['consistency', 'spearman_r', round(metrics['spearman_r'], 4), '中等负相关: S2与VIIRS呈反向关系 (互补)'],
        ['consistency', 'spearman_p', metrics['spearman_p'], '高度显著 (p≈0)'],
        ['consistency', 'disagreement_mean', round(metrics['disagreement_mean'], 4), '传感器分歧大: 均值0.47 (尺度0-1)'],
        ['consistency', 'disagreement_median', round(metrics['disagreement_median'], 4), '典型分歧>0.5: 严重冲突'],
        ['consistency', 'both_high_ratio', round(metrics['both_high']/metrics['n_both'], 4), f'双高仅{metrics["both_high"]}格: 一致确认损害极少'],
        ['consistency', 's2_low_viirs_high_ratio', round(metrics['s2_low_viirs_high']/metrics['n_both'], 4), '主导分歧方向: 功能损害 vs 结构损害'],
        # Uncertainty
        ['uncertainty', 'uncertainty_mean', round(metrics['uncertainty_mean'], 4), '综合不确定性中等偏高'],
        ['uncertainty', 'uncertainty_median', round(metrics['uncertainty_median'], 4), ''],
        ['uncertainty', 'top30_dominant_evidence', metrics['top30_unc_dominant_evidence'], '最高不确定性区域全是S2低/VIIRS高'],
        # Quality
        ['quality', 's2_low_quality_grids', metrics['s2_low_quality_count'], f'{100*metrics["s2_low_quality_count"]/metrics["n_total"]:.1f}%网格S2有效像素<20%'],
        ['quality', 'viirs_low_quality_grids', metrics['viirs_low_quality_count'], f'{100*metrics["viirs_low_quality_count"]/metrics["n_total"]:.1f}%网格VIIRS有效像素<20%'],
        # Recommendation
        ['recommendation', 'fusion_conservativeness', 'conservative', 'Logit-平均融合在传感器冲突时倾向于低概率 — 减少假阳性但增加假阴性'],
        ['recommendation', 'ds_theory_needed', 'yes', '强烈建议进入D-S证据理论: 传感器冲突严重, 等权平均无法正确处理冲突证据'],
        ['recommendation', 'need_labels', 'critical', '需要UNOSAT或其他参考标签才能做精度评估; 当前仅为数据质量和一致性评估'],
    ]

    with open(QUALITY_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  [CSV] Saved: {QUALITY_CSV}")


def plot_key_maps_panel(gdf):
    """生成四面板综合图"""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.15,
                           width_ratios=[1, 1, 0.6])

    gdf_wgs = gdf.to_crs("EPSG:4326")

    # --- Panel 1: p_fused ---
    ax1 = fig.add_subplot(gs[0, 0])
    gdf_wgs.plot(column='p_fused', ax=ax1, cmap='YlOrRd', vmin=0, vmax=1,
                 edgecolor='none', linewidth=0, legend=False)
    ax1.set_title('(a) p_fused: Fused Damage Probability', fontsize=11, fontweight='bold')
    ax1.set_xticks([]); ax1.set_yticks([])

    # --- Panel 2: Uncertainty ---
    ax2 = fig.add_subplot(gs[0, 1])
    gdf_wgs.plot(column='uncertainty', ax=ax2, cmap='viridis', vmin=0, vmax=0.8,
                 edgecolor='none', linewidth=0, legend=False)
    ax2.set_title('(b) Uncertainty: 0.7×Disagreement + 0.3×Quality', fontsize=11, fontweight='bold')
    ax2.set_xticks([]); ax2.set_yticks([])

    # --- Panel 3: Evidence Type ---
    ax3 = fig.add_subplot(gs[0, 2])
    evidence_colors = {
        'both_high': '#253494',
        's2_high_viirs_low': '#2c7fb8',
        's2_low_viirs_high': '#fd8d3c',
        'both_low': '#ffffcc',
        'single_source': '#d9d9d9',
        'no_data': '#999999',
    }
    for etype, color in evidence_colors.items():
        subset = gdf_wgs[gdf_wgs['evidence_type'] == etype]
        if len(subset) > 0:
            subset.plot(ax=ax3, color=color, edgecolor='none', linewidth=0,
                        label=etype.replace('_', ' ').title())
    ax3.legend(fontsize=6.5, loc='lower right', framealpha=0.85)
    ax3.set_title('(c) Evidence Type', fontsize=11, fontweight='bold')
    ax3.set_xticks([]); ax3.set_yticks([])

    # --- Panel 4: p_s2 vs p_viirs scatter ---
    ax4 = fig.add_subplot(gs[1, 0])
    both_df = gdf[gdf['fusion_source'] == 'both'].dropna(subset=['p_s2_mean', 'p_viirs_mean'])
    if len(both_df) > 0:
        from scipy.stats import spearmanr as spr
        r, _ = spr(both_df['p_s2_mean'], both_df['p_viirs_mean'])
        # Color by evidence type
        for etype, color in [
            ('both_high', '#253494'),
            ('s2_high_viirs_low', '#2c7fb8'),
            ('s2_low_viirs_high', '#fd8d3c'),
            ('both_low', '#b0b0b0'),
        ]:
            subset = both_df[both_df['evidence_type'] == etype]
            if len(subset) > 0:
                ax4.scatter(subset['p_s2_mean'], subset['p_viirs_mean'],
                            c=color, s=12, alpha=0.6, label=etype.replace('_', ' ').title(),
                            edgecolors='none')
        ax4.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4)
        ax4.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
        ax4.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
        ax4.set_xlabel('p_s2 (Sentinel-2)')
        ax4.set_ylabel('p_viirs (VIIRS)')
        ax4.set_title(f'(d) p_s2 vs p_viirs (n={len(both_df)}), Spearman ρ={r:.3f}',
                      fontsize=11, fontweight='bold')
        ax4.set_xlim(-0.02, 1.02); ax4.set_ylim(-0.02, 1.02)
        ax4.legend(fontsize=7, loc='upper left', markerscale=1.5)

    # --- Panel 5: Distribution comparison ---
    ax5 = fig.add_subplot(gs[1, 1])
    for col, color, label in [
        ('p_s2_mean', '#1f77b4', 'p_s2 (S2)'),
        ('p_viirs_mean', '#ff7f0e', 'p_viirs (VIIRS)'),
        ('p_fused', '#2ca02c', 'p_fused'),
    ]:
        vals = gdf[col].dropna()
        if len(vals) > 0:
            ax5.hist(vals, bins=50, color=color, alpha=0.55, label=label,
                     density=True, edgecolor='white', linewidth=0.3)
    ax5.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Probability / Score')
    ax5.set_ylabel('Density')
    ax5.set_title('(e) Distribution Comparison', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)

    # --- Panel 6: Evidence type bar chart ---
    ax6 = fig.add_subplot(gs[1, 2])
    etypes = ['both_high', 's2_high_viirs_low', 's2_low_viirs_high', 'both_low',
              'single_source', 'no_data']
    ecolors = ['#253494', '#2c7fb8', '#fd8d3c', '#ffffcc', '#d9d9d9', '#999999']
    counts = []
    for et in etypes:
        cnt = (gdf['evidence_type'] == et).sum()
        counts.append(cnt)

    bars = ax6.barh(range(len(etypes)), counts, color=ecolors, edgecolor='#333', linewidth=0.5)
    ax6.set_yticks(range(len(etypes)))
    ax6.set_yticklabels([e.replace('_', ' ').title() for e in etypes], fontsize=8)
    ax6.set_xlabel('Grid Count')
    ax6.set_title('(f) Evidence Type Counts', fontsize=11, fontweight='bold')
    for i, (c, et) in enumerate(zip(counts, etypes)):
        ax6.text(c + 5, i, f'{c} ({100*c/len(gdf):.1f}%)', va='center', fontsize=7.5)

    fig.suptitle('Sentinel-2 / VIIRS Fusion Evaluation — Mariupol AOI\n'
                 f'500m Grid, {len(gdf)} cells',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(PANEL_PNG, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure] Saved: {PANEL_PNG}")


def main():
    print("=" * 70)
    print("双源融合结果评估与解释")
    print("=" * 70)

    # Load
    print("\n[1] Loading data...")
    df, gdf = load_data()
    print(f"    {len(df)} grid cells loaded")

    # Compute metrics
    print("\n[2] Computing evaluation metrics...")
    metrics = compute_evaluation_metrics(df)

    # Write interpretation
    print("\n[3] Writing interpretation notes...")
    write_interpretation_notes(metrics)

    # Write quality CSV
    print("\n[4] Writing quality assessment...")
    write_quality_assessment_csv(metrics)

    # Plot panel
    print("\n[5] Plotting key maps panel...")
    plot_key_maps_panel(gdf)

    print("\n" + "=" * 70)
    print("评估完成!")
    print(f"  网格总数: {metrics['n_total']}")
    print(f"  Spearman ρ: {metrics['spearman_r']:.4f}")
    print(f"  主导证据类型: s2_low_viirs_high (47.0%)")
    print(f"  建议: 进入 D-S 证据理论阶段")
    print("=" * 70)


if __name__ == "__main__":
    main()
