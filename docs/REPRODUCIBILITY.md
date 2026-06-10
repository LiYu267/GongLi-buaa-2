# 复现说明 (Reproducibility Guide)

本文档说明如何从原始数据出发，复现论文中的所有分析和图表。

## 推荐运行环境

### 操作系统

- Linux (Ubuntu 20.04+, Debian 11+) — 推荐
- macOS 12+ (Apple Silicon / Intel)
- Windows 10/11 (通过 WSL2 或原生 Python)

### 硬件

| 需求 | 最低 | 推荐 |
|------|------|------|
| RAM | 16 GB | 32 GB |
| 磁盘 | 50 GB 空闲 | 100 GB SSD |
| CPU | 4 核 | 8+ 核（MCMC 可并行） |
| GPU | 不需要 | — |

### Python 版本

- Python 3.10 或更高版本（已测试：3.13）

## 依赖安装

### 方案 A：pip

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/macOS
# 或 venv\Scripts\activate  (Windows)

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 方案 B：conda（推荐，尤其 Windows 用户）

```bash
conda create -n t20_mariupol python=3.10
conda activate t20_mariupol

# 优先走 conda-forge 安装空间分析核心库
conda install -c conda-forge numpy pandas rasterio geopandas shapely pyproj

# 安装 pymc 和 arviz（conda-forge 编译更稳定）
conda install -c conda-forge pymc arviz

# 安装其余依赖
pip install h5py xarray rioxarray matplotlib scipy
```

### 验证

```bash
python src/00_check_environment.py
```

所有依赖应显示为 ✅ available。如果 `pymc` 或 `arviz` 缺失，请参考 PyMC [官方安装文档](https://www.pymc.io/projects/docs/en/latest/installation.html)。

## 配置文件设置

### 步骤 1：创建本地配置

```bash
# 复制模板配置
cp config/project_config.example.yaml config/project_config.yaml
cp config/aoi_mariupol.example.geojson config/aoi_mariupol.geojson
```

### 步骤 2：编辑 project_config.yaml

修改以下路径以匹配本地环境：

```yaml
# 示例：如原始数据在不同位置，修改脚本中的 RAW_DATA 路径
# 大多数脚本使用 PROJECT_ROOT / "原始数据" 作为默认路径
# 可在各脚本顶部的 "配置" 区块中自定义路径
```

> 大部分脚本的路径配置位于脚本顶部的 `PROJECT_ROOT` 和相关常量中，默认假设原始数据位于项目根目录下的 `原始数据/`。如需自定义路径，请编辑对应脚本。

## 从原始数据到最终图表的完整流程

以下为从头复现的完整步骤，按依赖关系排序：

### 阶段 A：数据诊断与预处理（~1–2 小时）

```bash
# A1. 清查文件清单，确认原始数据完整
python src/01_inventory_files.py

# A2. 检查 S2 波段完整性
python src/02_check_sentinel2_bands.py

# A3. S1 GRD 快速诊断（可选，不影响主流程）
python src/03_check_sentinel1_grd.py

# A4. S2 主预处理（最耗时步骤，~30–90 分钟）
#     输出：data/processed/s2_main_*_indices.tif
python src/04_sentinel2_main_preprocessing.py

# A5. S2 损毁评分
#     输出：data/processed/s2_main_damage_score.tif
python src/05_sentinel2_damage_score.py
```

### 阶段 B：VIIRS 处理（~10 分钟）

```bash
# B1. VIIRS H5 诊断
python src/06_viirs_h5_diagnosis.py

# B2. VIIRS NTL 处理：质量掩膜 → 网格聚合 → NTL 损失率
#     输出：data/processed/viirs_ntl_*.tif, viirs_damage_score.tif
python src/07_viirs_processing.py
```

### 阶段 C：网格融合与初步验证（~5 分钟）

```bash
# C1. S2 + VIIRS 融合到 500m 网格
#     输出：data/processed/grid_fusion_s2_viirs.csv, .geojson
python src/08_grid_fusion_s2_viirs.py

# C2. 融合评估（与中立先验比较）
python src/09_fusion_evaluation.py

# C3. UNOSAT 严格验证（ROC/PR 曲线）
python src/14_unosat_validation_strict.py
```

### 阶段 D：贝叶斯层次模型 + MCMC（核心分析，~20–60 分钟）

```bash
# D1. 贝叶斯层次 Logit-Normal 模型 + NUTS MCMC
#     输出：data/processed/grid_bayesian_posterior.csv
#          outputs/tables/bayesian_parameter_summary.csv
#          outputs/figures/bayesian_posterior_map.png
python src/16_bayesian_hierarchical.py

# D2. (可选) 贝叶斯模型诊断审计
python src/16a_bayesian_audit.py
```

### 阶段 E：KLD 信息增益与融合抑制分析（~20–40 分钟）

```bash
# E1. KLD/JSD 分析（自动运行单传感器贝叶斯模型作为基线）
#     输出：data/processed/kld_information_gain_by_grid.csv
#          outputs/tables/kld_information_gain_summary.csv
#          outputs/figures/kld_*.png
python src/17_kld_information_gain.py
```

### 阶段 F：Dempster-Shafer 证据理论（~5 分钟）

```bash
# F1. D-S BPA 构造 + Dempster 组合 + 决策分类
#     输出：data/processed/grid_ds_fusion_s2_viirs.csv, .geojson
python src/11_ds_fusion.py
```

### 阶段 G：Bayes vs. D-S 一致性比较（~5 分钟）

```bash
# G1. 网格级逐对比：贝叶斯后验 vs. D-S Belief/Plausibility
#     输出：outputs/figures/bayes_vs_ds_*.png
python src/18_bayes_vs_ds_consistency.py
```

### 阶段 H：论文图表生成（~20 分钟）

```bash
# H1. 生成论文最终图表集
python src/_gen_v4_figures.py
```

### 阶段 I：论文编译

```bash
cd paper/
latexmk -pdf main_v4.tex
# 或: pdflatex main_v4.tex && bibtex main_v4 && pdflatex main_v4.tex && pdflatex main_v4.tex
```

## 如何重新生成论文中的 Figures 和 Tables

### 论文插图对应关系

论文 v4 (`main_v4.tex`) 中的插图生成来源：

| 论文 Figure | 源脚本 | 输出文件 |
|-------------|--------|----------|
| Fig 1: Workflow + UNOSAT Validation | `14_unosat_validation_strict.py`, `_gen_v4_figures.py` | `fig1_workflow_v4.png`, `fig1_unosat_validation.png` |
| Fig 2: Bayesian posterior map | `16_bayesian_hierarchical.py` | `fig2_bayesian_posterior_map.png` |
| Fig 3: Sensor comparison | `16_bayesian_hierarchical.py`, `_gen_v4_figures.py` | `fig3_sensor_only_comparison.png` |
| Fig 4: KLD barplot + Information conflict | `17_kld_information_gain.py`, `_gen_v4_figures.py` | `fig4_kld_barplot.png`, `fig4_information_conflict.png` |
| Fig 5: JSD conflict maps | `17_kld_information_gain.py`, `_gen_v4_figures.py` | `fig5_jsd_conflict_map.png` (或 `fig5a_kld_summary_v4.png`, `fig5b_kld_maps_v4.png`) |
| Fig 6: Fusion suppression + Bayes-DS | `17_kld_information_gain.py`, `_gen_v4_figures.py` | `fig6_fusion_suppression_map.png`, `fig6_bayes_ds_v4.png` |
| Fig 7: Consistency map | `18_bayes_vs_ds_consistency.py` | `fig7_consistency_map.png` |
| Fig 8: Interval comparison | `18_bayes_vs_ds_consistency.py`, `_gen_v4_figures.py` | `fig8_interval_comparison.png` |
| Appendix | `14_unosat_validation_strict.py`, `_gen_v4_figures.py` | `fig_appendix_pr_v4.png`, `fig_appendix_scatter_v4.png` |

### 论文表格对应关系

论文中的 5 张表格主要来自以下数据：

| 论文 Table | 源 CSV | 源脚本 |
|------------|--------|--------|
| Data sources summary | 手工编写 (LaTeX) | — |
| Bayesian hyperparameters | `outputs/tables/bayesian_parameter_summary.csv` | `16_bayesian_hierarchical.py` |
| KLD/JSD summary | `outputs/tables/kld_information_gain_summary.csv` | `17_kld_information_gain.py` |
| Bayes vs. D-S consistency | `outputs/tables/bayes_vs_ds_consistency_summary.csv` | `18_bayes_vs_ds_consistency.py` |
| Method comparison (Appendix) | 手工编写 (LaTeX) | — |

> **注意**：表格内容是嵌入在 LaTeX 源文件中的（由脚本生成的数值手动填入或通过脚本输出 CSV 后手工整理）。请检查 `paper/main_v4.tex` 中的表格数据是否与最新脚本输出一致。

## 已知限制

1. **MCMC 随机性**：贝叶斯模型使用 MCMC (NUTS) 采样，虽已设置随机种子 (`RANDOM_SEED=42`)，不同平台的浮点运算可能导致微小的数值差异。关键数值（如融合抑制率 46.8%）应在不同运行间保持一致（±1%）。

2. **单传感器贝叶斯模型收敛**：S2-only 和 VIIRS-only 的单传感器贝叶斯模型存在 MCMC 发散问题（S2-only: ~263 divergences, $\hat{R}_{\max}=1.21$），其后验区间应视为诊断性参考而非精确估计。

3. **UNOSAT 标签覆盖率**：UNOSAT 标注仅覆盖 Mariupol 的 Livoberezhnyi 区，而非整个 AOI。验证结果反映的是有标注区域内的相对对比，不代表全 AOI 精度。

4. **500m 网格尺度**：所有分析在 500m 网格上进行。不同空间分辨率下的冲突度和信息增益可能不同（聚合效应）。

5. **时间不匹配**：S2 和 VIIRS 的获取时间不完全一致。VIIRS 为月合成、S2 为单日获取，损伤过程在两个时相之间持续演化。

6. **Windows 路径**：脚本使用 `pathlib.Path` 处理路径，Windows 用户应注意反斜杠与正斜杠的兼容性。推荐在 WSL2 环境下运行。

7. **未使用 scikit-learn**：本项目的 AUC、PR-AUC、Brier Score 等指标为手工实现，不依赖 scikit-learn。`00_check_environment.py` 中的 sklearn 检查为非必需项。

8. **PyMC 版本兼容性**：PyMC v5.x 的 API 与 v4.x 有差异（如 `pm.HalfCauchy` 的参数名从 `beta` 变更为 `beta` 统一）。本项目以 PyMC v5 为目标版本。如使用 v4，请修改 `pm.HalfCauchy('tau', beta=1.0)` 为 `pm.HalfCauchy('tau', beta=1.0)`（v4/v5 的该 API 相同，但内部实现有差异）。

9. **栅格数据重新生成**：`data/processed/sample/` 中不含 TIF 栅格文件（体积和授权原因），栅格需通过运行阶段 A–B 的预处理脚本从原始数据重新生成。后续分析脚本（`16_bayesian_hierarchical.py` 起）主要依赖 CSV/GeoJSON 中间产物，不直接读取 TIF。
