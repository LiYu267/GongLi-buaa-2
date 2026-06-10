# T20 Mariupol 多源遥感损毁评估 — 贝叶斯不确定性融合

> **Bayesian Uncertainty Fusion for Multi-Sensor Remote Sensing Damage Assessment: A Case Study of Sentinel-2 and VIIRS in Mariupol**

本项目利用贝叶斯层次模型、KLD/JSD 信息增益分析和 Dempster-Shafer 证据理论，融合 Sentinel-2 光谱指数与 VIIRS 夜间灯光数据，对马里乌波尔（Mariupol, Ukraine）2022 年冲突期间的建筑损毁进行网格级概率估计与不确定性量化。

## 研究背景

快速、准确的损毁评估对武装冲突和自然灾害后的救援与重建至关重要。卫星遥感具有大面积、可重复观测的优势：

- **Sentinel-2（光学）**：通过战前/战后光谱变化（NDVI、NBR、NDBI 等）检测地表结构变化。
- **VIIRS DNB（夜间灯光）**：通过夜间灯光辐射亮度损失检测功能性中断（供电中断、人员撤离等）。
- **UNOSAT**：基于高分辨率卫星影像目视解译的建筑物损毁标注，作为外部参考。

当前研究的核心发现是：**S2 和 VIIRS 测量的是不同的损毁维度（结构变化 vs. 功能丧失），在单隐变量贝叶斯模型下，两者信号相互冲突，导致融合后验几乎退化为空间均一分布（$p \approx 0.155$），融合信息抑制率达 46.8%。**

## 数据来源说明

| 数据源 | 产品 | 战前日期 | 战后日期 | 空间分辨率 |
|--------|------|----------|----------|------------|
| Sentinel-2 MSI | L2A 地表反射率 | 2021-05-23 | 2022-05-08 | 10–20 m（聚合至 500 m 网格） |
| VIIRS DNB | VNP46A3 月合成 | 2021-05 | 2022-05 | 500 m（原生） |
| UNOSAT | 损毁评估（目视解译） | — | 2022-05-12 | 建筑级 |

### Sentinel-1 GRD（未用于主实验）

Sentinel-1 GRD 数据已下载并进行了基础诊断（`src/03_check_sentinel1_grd.py`），但由于 SAR 幅度图像的损毁信号提取需要额外处理（辐射定标、散斑滤波、极化分解等），**未纳入论文主实验**。SAR/InSAR 可在后续工作中作为互补数据源使用。

### 原始数据不在仓库中

**本仓库不包含任何卫星遥感原始数据。** 所有 `.SAFE`、`.tif`、`.jp2`、`.h5` 等遥感数据文件需要用户自行从以下来源下载：

- **Sentinel-2**: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) 或 [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- **VIIRS VNP46A3**: [LAADS DAAC (NASA)](https://ladsweb.modaps.eosdis.nasa.gov/)
- **UNOSAT**: [UNOSAT Humanitarian Data Exchange](https://unosat.org/)

## 仓库结构说明

```
.
├── README.md                 # 项目说明（本文件）
├── LICENSE                   # 许可证
├── .gitignore                # Git 忽略规则
├── requirements.txt          # Python 依赖
├── config/
│   ├── project_config.example.yaml  # 项目配置文件模板
│   └── aoi_mariupol.example.geojson # AOI 边界模板
├── src/                      # 源代码（按运行顺序编号）
│   ├── 00_check_environment.py         # 环境检查
│   ├── 01_inventory_files.py           # 文件清查
│   ├── 02_check_sentinel2_bands.py     # S2 波段检查
│   ├── 03_check_sentinel1_grd.py       # S1 GRD 诊断
│   ├── 04_sentinel2_main_preprocessing.py  # S2 预处理与指数计算
│   ├── 05_sentinel2_damage_score.py    # S2 损毁评分
│   ├── 06_viirs_h5_diagnosis.py        # VIIRS H5 诊断
│   ├── 07_viirs_processing.py          # VIIRS 处理
│   ├── 08_grid_fusion_s2_viirs.py      # S2+VIIRS 网格融合
│   ├── 09_fusion_evaluation.py         # 融合评估
│   ├── 10_aoi_robustness.py            # AOI 稳健性分析
│   ├── 11_ds_fusion.py                 # D-S 证据理论融合
│   ├── 12_unosat_validation.py         # UNOSAT 验证
│   ├── 13_unosat_label_audit.py        # UNOSAT 标签审计
│   ├── 14_unosat_validation_strict.py  # UNOSAT 严格验证
│   ├── 15_fusion_sensitivity.py        # 融合敏感性分析
│   ├── 16_bayesian_hierarchical.py     # 贝叶斯层次模型 (MCMC)
│   ├── 16a_bayesian_audit.py           # 贝叶斯诊断审计
│   ├── 17_kld_information_gain.py      # KLD/JSD 信息增益分析
│   ├── 18_bayes_vs_ds_consistency.py   # Bayes vs. D-S 一致性比较
│   ├── _gen_paper_figures.py           # 论文图表生成（v1/v2）
│   ├── _gen_v3_figures.py              # 论文图表生成（v3）
│   └── _gen_v4_figures.py              # 论文图表生成（v4）
├── data/
│   └── processed/
│       └── sample/             # 小样例输出数据（CSV + GeoJSON）
├── outputs/
│   ├── figures/                # 输出图表（PNG）
│   └── tables/                 # 输出表格（CSV）
├── paper/
│   ├── main.tex                # 主论文 LaTeX 源文件（ACML 2026 / JMLR 格式）
│   ├── main_v2.tex             # 论文 v2
│   ├── main_v3.tex             # 论文 v3
│   ├── main_v4.tex             # 论文 v4（最新）
│   ├── references.bib          # 参考文献
│   ├── acml26.bib              # ACML 模板样例参考文献
│   ├── jmlr.cls                # JMLR 文档类
│   ├── README_build.md         # 论文编译说明
│   ├── README_v2_changes.md    # v2 修改说明
│   ├── README_v3_changes.md    # v3 修改说明
│   ├── figures/                # 论文插图（PNG）
│   └── tables/                 # 论文表格
└── docs/
    ├── DATA.md                 # 数据说明
    └── REPRODUCIBILITY.md      # 复现说明
```

## 环境安装方法

### 前置条件

- Python 3.10+
- 推荐使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 管理 Python 环境
- TeX Live 2024+ 或 MiKTeX（用于编译论文 PDF）

### 安装步骤

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd <repo-name>

# 2. 创建并激活 conda 环境
conda create -n t20_mariupol python=3.10
conda activate t20_mariupol

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证环境
python src/00_check_environment.py
```

### 依赖说明

主要 Python 依赖：

| 类别 | 库 | 用途 |
|------|-----|------|
| 数据处理 | `numpy`, `pandas` | 数值计算与表格处理 |
| 地理空间 | `rasterio`, `geopandas`, `shapely`, `pyproj` | 栅格/矢量遥感数据处理 |
| 科学数据 | `h5py` | VIIRS HDF5 文件读取 |
| 可视化 | `matplotlib` | 所有图表生成 |
| 统计推断 | `scipy` | Spearman 相关性等统计检验 |
| 贝叶斯建模 | `pymc`, `arviz` | MCMC (NUTS) 后验推断与诊断 |

> **注意**：`pymc` 和 `arviz` 的安装在不同平台可能需要额外配置。如在 Windows 上遇到问题，建议使用 conda-forge：
> ```bash
> conda install -c conda-forge pymc arviz
> ```

## 数据准备方法

### 原始数据下载

本仓库**不包含任何原始遥感数据**。用户需自行下载以下数据并组织为推荐结构：

| 数据 | 来源 | 说明 |
|------|------|------|
| Sentinel-2 L2A | [Copernicus Data Space](https://dataspace.copernicus.eu/) | 瓦片 T37TCN、T37TDN；战前(2021-05-23)、战后(2022-05-08) |
| VIIRS VNP46A3 | [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/) | 2021 年 5 月与 2022 年 5 月月合成产品 |
| Sentinel-1 GRD（可选） | Copernicus Data Space | 战前/战后 IW GRDH 产品 |
| UNOSAT 标注 | [UNOSAT / HDX](https://data.humdata.org/) | Mariupol Livoberezhnyi District, 2022-05-12 |

### 推荐数据目录结构

将下载的原始数据按以下结构放置：

```
原始数据/
├── Sentinel-2/
│   ├── 战前1（2021.5）/     # T37TCN, 2021-05-23
│   ├── 战前2（2021.5）/     # T37TDN, 2021-05-23
│   ├── 战后1(2022.5)/       # T37TCN, 2022-05-08
│   └── 战后2(2022.5)/       # T37TDN, 2022-05-08
├── VIIRS/
│   ├── VNP46A3.A2021121.h19v03.001.*.h5  # 2021-05
│   └── VNP46A3.A2022121.h19v03.001.*.h5  # 2022-05
├── Sentinel-1 GRD/
│   ├── S1A_IW_GRDH_1SDV_20210601T...SAFE
│   └── S1A_IW_GRDH_1SDV_20220616T...SAFE
└── 人工标注数据/
    ├── GDB/
    └── SHP/
```

> 详细说明见 [docs/DATA.md](docs/DATA.md)。

## 运行流程

脚本按编号顺序运行，各阶段之间存在输入/输出依赖。以下为主要流程：

### 1. 数据预处理

```bash
# 检查环境
python src/00_check_environment.py

# 清查现有文件
python src/01_inventory_files.py

# S2 波段检查与 GRD 诊断
python src/02_check_sentinel2_bands.py
python src/03_check_sentinel1_grd.py

# S2 主预处理：波段读取 → DN→反射率 → 20m重采样 → SCL掩膜 → 镶嵌 → AOI裁剪 → 指数计算
python src/04_sentinel2_main_preprocessing.py

# S2 损毁评分
python src/05_sentinel2_damage_score.py

# VIIRS 诊断与处理
python src/06_viirs_h5_diagnosis.py
python src/07_viirs_processing.py
```

### 2. 特征提取与网格融合

```bash
# S2 + VIIRS 网格融合（生成统一 500m 网格特征表）
python src/08_grid_fusion_s2_viirs.py
```

### 3. 模型训练与分析

```bash
# 融合评估
python src/09_fusion_evaluation.py

# UNOSAT 验证（多种版本）
python src/12_unosat_validation.py
python src/13_unosat_label_audit.py
python src/14_unosat_validation_strict.py

# 贝叶斯层次模型 (MCMC, 核心分析)
python src/16_bayesian_hierarchical.py

# KLD/JSD 信息增益分析（依赖贝叶斯后验）
python src/17_kld_information_gain.py

# D-S 证据理论融合
python src/11_ds_fusion.py

# Bayes vs. D-S 一致性比较
python src/18_bayes_vs_ds_consistency.py

# 敏感性分析（可选）
python src/10_aoi_robustness.py
python src/15_fusion_sensitivity.py
python src/16a_bayesian_audit.py
```

### 4. 图表生成

```bash
# 生成论文插图（v4 为最终版本）
python src/_gen_v4_figures.py
```

### 5. 论文结果复现

论文中的 8 张主图 + 附录图均来自以上脚本的输出。图表 → 论文的对应关系详见 `paper/README_build.md` 和 `paper/main_v4.tex`。

## 输出结果说明

### outputs/figures/ — 关键图表

| 文件 | 内容 |
|------|------|
| `roc_curves_unosat_strict.png` | S2 / VIIRS / 融合 / D-S 的 ROC 与 PR 曲线 |
| `bayesian_posterior_map.png` | 贝叶斯后验均值的空间分布图 |
| `sensor_only_comparison.png` | 单传感器 vs. 联合模型后验范围对比 |
| `kld_summary_barplot.png` | KLD 信息增益汇总柱状图 |
| `kld_conflict_jsd_map.png` | 传感器冲突 JSD 空间分布图 |
| `kld_fusion_suppression_map.png` | 融合信息抑制率空间分布图 |
| `bayes_vs_ds_consistency_map.png` | Bayes vs. D-S 一致性分类空间图 |
| `bayes_vs_ds_interval_comparison.png` | 置信区间/信度区间逐网格对比 |
| `model_auc_comparison_strict.png` | 各方法 AUC 对比 |

### outputs/tables/ — 关键表格

| 文件 | 内容 |
|------|------|
| `bayesian_parameter_summary.csv` | 贝叶斯超参数后验摘要 |
| `kld_information_gain_summary.csv` | KLD/JSD 信息增益统计 |
| `validation_metrics_unosat_strict.csv` | UNOSAT 严格验证指标 |
| `bayes_vs_ds_consistency_summary.csv` | Bayes vs. D-S 一致性分类统计 |
| `ds_fusion_summary_statistics.csv` | D-S 融合汇总统计 |

## 注意事项

1. **MCMC 运行时间**：`16_bayesian_hierarchical.py` 和 `17_kld_information_gain.py` 包含 MCMC 采样（4 链 × 2000 迭代），预计运行时间 10–60 分钟（取决于 CPU 和网格数量）。

2. **内存需求**：Sentinel-2 预处理（`04_*`）需要同时加载 4 个 SAFE 产品的多波段数据，建议 ≥16 GB RAM。

3. **配置文件**：复制 `config/project_config.example.yaml` 为 `project_config.yaml` 并根据本地路径修改。同样复制 `config/aoi_mariupol.example.geojson` 为 `aoi_mariupol.geojson`。

4. **数据许可**：Sentinel 数据遵循 Copernicus 自由开放数据政策；VIIRS 数据遵循 NASA 开放数据政策；UNOSAT 数据使用需遵循其使用条款。**用户需自行确认数据使用许可**。

5. **外部参考 vs. 真值**：UNOSAT 标注是基于目视解译的外部参考，不是绝对真值。它主要捕捉可见建筑结构损毁，可能遗漏功能损毁或亚检测阈值结构变化。

6. **路径约定**：所有脚本使用相对于项目根目录的路径。请从项目根目录运行脚本，或确保 `PROJECT_ROOT` 能正确解析。

7. **脚本编号对应性**：脚本编号不代表严格线性依赖。建议按上文"运行流程"中的顺序执行，跳过不需要的阶段。

## 引用方式 Citation

如使用本研究的代码、方法或结果，请引用：

```bibtex
@unpublished{T20_Mariupol_2026,
  title = {Bayesian Uncertainty Fusion for Multi-Sensor Remote Sensing Damage Assessment:
           A Case Study of Sentinel-2 and VIIRS in Mariupol},
  author = {Author Name},
  note = {Manuscript in preparation},
  year = {2026}
}
```

> 引用信息待最终论文发表后更新。

## License 说明

本项目代码采用 [MIT License](LICENSE) 授权。

- 代码 (`src/`)：MIT License
- 论文源文件 (`paper/`)：CC BY 4.0（或根据目标期刊政策调整）
- 样例数据 (`data/processed/sample/`)：CC BY 4.0
- 原始遥感数据**不包含**在本仓库中，其使用遵循各自数据提供方的许可条款

## 联系方式

如有问题或合作意向，请联系：

- **作者**: [请填写姓名]
- **邮箱**: [请填写邮箱]
- **机构**: [请填写机构]
- **GitHub**: [请填写 GitHub 用户名]
