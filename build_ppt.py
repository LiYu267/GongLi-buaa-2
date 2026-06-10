"""
Build 10-page PPT structure:
  ppt_slides/page_01/ ... page_10/
Each folder: 1-2 images + description.txt (≤20 words if 2 images)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os, shutil
from pathlib import Path

BASE = Path('d:/数理统计大作业数据')
S2 = BASE / 'GongLi-buaa-2-main'  # Student 2: 2-sensor project
US = BASE  # Our: 4-sensor project

PPT = BASE / 'ppt_slides'
if PPT.exists(): shutil.rmtree(PPT)
PPT.mkdir()

# ============================================================
# Page 01 — 研究问题与动机
# ============================================================
d = PPT / 'page_01'; d.mkdir()
# Use our workflow diagram or damage maps
shutil.copy(US / 'paper_figures/fig5_version_tracking.png', d / '01_improvement_tracking.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('研究问题: 如何用多源遥感统计融合评估冲突损害?\n')
    f.write('单一传感器存在固有盲区:\n')
    f.write('  VIIRS: 测灯光=功能性损害, 无法感知建筑结构\n')
    f.write('  Sentinel-2: 测光谱, 受季节和云层影响\n')
    f.write('  SAR/InSAR: 测雷达回波/地表稳定度\n')
    f.write('四个传感器测不同物理量, 需统计融合')

# ============================================================
# Page 02 — 两套方案总览
# ============================================================
d = PPT / 'page_02'; d.mkdir()
# Use side-by-side: student2's workflow + our workflow
# Student 2 workflow
shutil.copy(S2 / 'paper/figures/fig1_workflow_v4.png', d / '02a_workflow_2sensor.png')
# No direct equivalent for ours, use the framework fig from our paper context
# Copy our fig1 damage maps as proxy for our framework
shutil.copy(US / 'paper_figures/fig1_damage_maps.png', d / '02b_workflow_4sensor.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('左: 方案A — 两传感器(S2+VIIRS)完整流程\n')
    f.write('右: 方案B — 四传感器(VIIRS+S2+SAR+InSAR)简易对比流程')

# ============================================================
# Page 03 — 数据源对比
# ============================================================
d = PPT / 'page_03'; d.mkdir()
# Could use a table image - let's use sensor comparison figures
shutil.copy(S2 / 'outputs/figures/viirs_ntl_pre_post.png', d / '03a_viirs_raw.png')
shutil.copy(S2 / 'outputs/figures/s2_main_damage_score.png', d / '03b_s2_damage_score.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('左: VIIRS 战前/战后夜间灯光对比 (500m)\n')
    f.write('右: Sentinel-2 方向性损毁评分 (10m→500m)\n')
    f.write('方案B额外使用SAR和InSAR')

# ============================================================
# Page 04 — 同学1: 两传感器核心方法
# ============================================================
d = PPT / 'page_04'; d.mkdir()
shutil.copy(S2 / 'outputs/figures/s2_main_delta_indices.png', d / '04a_s2_indices.png')
shutil.copy(S2 / 'paper/figures/fig6_conceptual_model.png', d / '04b_conceptual_model.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('方案A核心: S2三维光谱指数+PCA降维+Beta-Binomial层次模型')

# ============================================================
# Page 05 — 同学1: 两传感器关键结果
# ============================================================
d = PPT / 'page_05'; d.mkdir()
shutil.copy(S2 / 'outputs/figures/sensor_only_comparison.png', d / '05a_sensor_compare.png')
shutil.copy(S2 / 'paper/figures/fig1_unosat_validation.png', d / '05b_unosat_validation.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('核心发现: 两传感器信号冲突, 融合退化至均匀分布\n'
            'p≈0.155, 融合信息抑制率46.8%')

# ============================================================
# Page 06 — 我们: 四传感器核心方法
# ============================================================
d = PPT / 'page_06'; d.mkdir()
shutil.copy(US / 'paper_figures/fig1_damage_maps.png', d / '06a_4sensor_maps.png')
shutil.copy(US / 'paper_figures/fig5_version_tracking.png', d / '06b_version_improvement.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('方案B: 四传感器简易流程\n'
            'SAR直方图匹配+InSAR归一化+S2方向感知+VIIRS城市聚焦')

# ============================================================
# Page 07 — 我们: 传感器独立性验证
# ============================================================
d = PPT / 'page_07'; d.mkdir()
shutil.copy(US / 'paper_figures/fig4_sensor_agreement.png', d / '07a_sensor_correlation.png')
shutil.copy(US / 'paper_figures/fig3_method_comparison.png', d / '07b_method_comparison.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('四传感器两两相关 |ρ|<0.11, 验证融合必要性\n'
            '贝叶斯融合p=0.176')

# ============================================================
# Page 08 — 核心对比: 两方案融合行为
# ============================================================
d = PPT / 'page_08'; d.mkdir()
shutil.copy(S2 / 'paper/figures/fig2_bayesian_posterior_map.png', d / '08a_2sensor_fusion.png')
shutil.copy(US / 'paper_figures/fig2_fusion_uncertainty.png', d / '08b_4sensor_fusion.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('左: 2传感器融合 右: 4传感器融合\n'
            '方案A更完整(UNOSAT验证+MCMC), 方案B更全面(传感器更多)')

# ============================================================
# Page 09 — 信息论对比: KLD & D-S
# ============================================================
d = PPT / 'page_09'; d.mkdir()
shutil.copy(S2 / 'outputs/figures/kld_summary_barplot.png', d / '09a_kld_barplot.png')
shutil.copy(S2 / 'outputs/figures/bayes_vs_ds_interval_comparison.png', d / '09b_bayes_vs_ds.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('信息论分析: KLD量化传感器贡献, D-S vs Bayesian行为对比')

# ============================================================
# Page 10 — 结论与展望
# ============================================================
d = PPT / 'page_10'; d.mkdir()
shutil.copy(S2 / 'outputs/figures/grid_p_s2_map.png', d / '10a_s2_map.png')
shutil.copy(S2 / 'outputs/figures/grid_p_viirs_map.png', d / '10b_viirs_map.png')
with open(d/'description.txt', 'w', encoding='utf-8') as f:
    f.write('结论: 两方案共同揭示传感器信号近正交性\n'
            '未来: 方案A+MCMC+UNOSAT 与 方案B+SAR/InSAR 互补整合\n'
            '更多传感器→更完整损害维度覆盖→更诚实的后验分布')

print("PPT structure created:")
for d in sorted(PPT.glob('page_*')):
    files = list(d.glob('*'))
    desc = (d / 'description.txt').read_text(encoding='utf-8').split('\n')[0]
    print(f'  {d.name}: {len(files)-1} images | {desc[:80]}')
