# 数据说明 (Data Documentation)

## 为什么原始数据不放入 GitHub

本仓库原则上**不包含任何卫星遥感原始数据**，原因如下：

1. **文件体积**：单个 Sentinel-2 SAFE 产品约 1–2 GB，全套原始数据超过 20 GB，远超 GitHub 推荐仓库大小（< 1 GB）和单文件大小限制（100 MB）。
2. **许可限制**：虽然 Copernicus 和 NASA 数据是自由开放的，但重新分发数据需遵守各数据提供方的条款。本仓库提供代码和配置，用户从官方来源自行下载数据。
3. **可复现性设计**：所有数据处理脚本通过配置文件引用数据路径，用户只需按推荐结构组织本地数据即可复现完整流程。

## 需要用户自行下载的数据

| # | 数据 | 来源 | 下载链接/说明 |
|---|------|------|---------------|
| 1 | Sentinel-2 L2A, T37TCN, 2021-05-23 | Copernicus Data Space | [dataspace.copernicus.eu](https://dataspace.copernicus.eu/) |
| 2 | Sentinel-2 L2A, T37TDN, 2021-05-23 | 同上 | 同上 |
| 3 | Sentinel-2 L2A, T37TCN, 2022-05-08 | 同上 | 同上 |
| 4 | Sentinel-2 L2A, T37TDN, 2022-05-08 | 同上 | 同上 |
| 5 | VIIRS VNP46A3, 2021-05 (h19v03) | LAADS DAAC | [ladsweb.modaps.eosdis.nasa.gov](https://ladsweb.modaps.eosdis.nasa.gov/) |
| 6 | VIIRS VNP46A3, 2022-05 (h19v03) | 同上 | 同上 |
| 7 | UNOSAT Mariupol 损毁评估 | HDX / UNOSAT | [data.humdata.org](https://data.humdata.org/) |
| 8 | Sentinel-1 GRD（可选） | Copernicus Data Space | 战前/战后 IW GRDH 产品 |

## 推荐数据目录结构

请将下载的数据按以下结构放置于项目父目录下（或修改 `config/project_config.yaml` 中的路径）：

```
<project_root>/
├── 原始数据/
│   ├── Sentinel-2/
│   │   ├── 战前1（2021.5）/          # T37TCN, 2021-05-23
│   │   │   └── S2A_MSIL2A_20210523T...SAFE/
│   │   ├── 战前2（2021.5）/          # T37TDN, 2021-05-23
│   │   │   └── S2A_MSIL2A_20210523T...SAFE/
│   │   ├── 战后1(2022.5)/            # T37TCN, 2022-05-08
│   │   │   └── S2A_MSIL2A_20220508T...SAFE/
│   │   └── 战后2(2022.5)/            # T37TDN, 2022-05-08
│   │       └── S2A_MSIL2A_20220508T...SAFE/
│   ├── VIIRS/
│   │   ├── VNP46A3.A2021121.h19v03.001.*.h5
│   │   └── VNP46A3.A2022121.h19v03.001.*.h5
│   ├── Sentinel-1 GRD/               # 可选
│   │   ├── S1A_IW_GRDH_1SDV_20210601T...SAFE/
│   │   └── S1A_IW_GRDH_1SDV_20220616T...SAFE/
│   └── 人工标注数据/
│       ├── GDB/                       # UNOSAT File Geodatabase
│       └── SHP/                       # 或 Shapefile 格式
```

## 小样例数据的用途

仓库中的 `data/processed/sample/` 目录包含了一组小体积样例输出数据（CSV + GeoJSON ≤ 10 MB），用途如下：

1. **验证环境**：新用户可以运行依赖这些中间文件的后续脚本（如 `17_kld_information_gain.py`、`18_bayes_vs_ds_consistency.py`），无需从头处理原始卫星数据。
2. **理解数据结构**：样例文件展示了各处理阶段的输出格式和字段，便于用户理解数据流。
3. **论文图表复现**：部分图表生成脚本（如 `_gen_v4_figures.py`）可能读取这些中间文件生成插图。

> **注意**：样例数据中的 TIF 栅格文件（如 `*.tif`）因体积和授权原因已排除。完整栅格数据需通过运行预处理脚本重新生成。

## 数据许可与引用注意事项

### Sentinel-2

- **许可**：遵循 [Copernicus Sentinel Data Terms and Conditions](https://scihub.copernicus.eu/twiki/do/view/SciHubWebPortal/TermsConditions) — 自由、完整、开放访问。
- **引用**：使用时可注明 "Contains modified Copernicus Sentinel data [Year]".

### VIIRS DNB (VNP46A3)

- **许可**：遵循 NASA Earth Science Data and Information Policy — 自由开放。
- **引用**：Román, M.O., et al. (2018). NASA's Black Marble nighttime lights product suite. *Remote Sensing of Environment*, 210, 113–143.

### UNOSAT

- **许可**：需遵循 UNOSAT 数据使用条款，通常允许研究使用并提供署名。
- **引用**：UNOSAT (2022). Damage Assessment in Livoberezhnyi District, Mariupol City, Donetska Oblast, Ukraine. 12 May 2022.

### 本仓库的样例数据

`data/processed/sample/` 中的 CSV 和 GeoJSON 文件是本研究的派生产品，与代码同样采用本项目 License 授权。
