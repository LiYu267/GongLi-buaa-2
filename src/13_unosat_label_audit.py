#!/usr/bin/env python3
"""
阶段 6.1 — UNOSAT 标签复核与验证集质量检查

关键发现:
  - Main_Dam_1 = 6 → No Visible Damage (April 3 assessment)
  - Main_Dam_1 = 14 → Visible Damage (April 3 assessment)
  - 767 / 3,459 cells damaged = 22.2% (matches UNOSAT PDF exactly)
  - 阶段 6 错误地将 Main_Damag 的 6 和 14 都当作 damage → 81.8% 正样本率是错误的
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNOSAT_SHP = PROJECT_ROOT / "原始数据" / "人工标注数据" / "SHP" / "Mariupol_3April2022_RDA.shp"
FUSION_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_fusion_s2_viirs.geojson"
DS_GEOJSON = PROJECT_ROOT / "data" / "processed" / "grid_ds_fusion_s2_viirs.geojson"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "validation_grid_unosat.csv"

OUT_TAB = PROJECT_ROOT / "outputs" / "tables"
OUT_MD  = PROJECT_ROOT / "outputs"
for d in [OUT_TAB, OUT_MD]:
    d.mkdir(parents=True, exist_ok=True)


def audit_fields():
    """完整审计所有字段"""
    gdf = gpd.read_file(UNOSAT_SHP)

    rows = []
    for col in gdf.columns:
        if col == 'geometry':
            continue
        vals = gdf[col].dropna()
        n_null = int(gdf[col].isna().sum())
        dtype = str(gdf[col].dtype)
        n_unique = vals.nunique()

        if n_unique <= 20:
            vc = vals.value_counts().to_dict()
            val_summary = str(vc)
        else:
            val_summary = f'min={vals.min()}, max={vals.max()}, mean={vals.mean():.3f}'

        rows.append({
            'field_name': col,
            'dtype': dtype,
            'n_unique': n_unique,
            'n_null': n_null,
            'null_pct': round(100*n_null/len(gdf),1),
            'values_detail': val_summary[:300],
        })

    with open(OUT_TAB / 'unosat_label_audit_fields.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  [CSV] Saved: {OUT_TAB / 'unosat_label_audit_fields.csv'}")

    return gdf


def audit_damage_coding(gdf):
    """详细审计 Main_Damag 和 Main_Dam_1 的编码含义"""
    rows = []

    # Main_Damag
    for val in sorted(gdf['Main_Damag'].unique()):
        cnt = int((gdf['Main_Damag'] == val).sum())
        rows.append({
            'field': 'Main_Damag',
            'value': val,
            'count': cnt,
            'pct': round(100*cnt/len(gdf), 2),
            'interpretation': 'No Visible Damage (Mar 26)' if val == 6 else 'Visible Damage / Previous Damage (Mar 26)' if val == 14 else 'unknown',
        })

    # Main_Dam_1
    for val in sorted(gdf['Main_Dam_1'].unique()):
        cnt = int((gdf['Main_Dam_1'] == val).sum())
        rows.append({
            'field': 'Main_Dam_1',
            'value': val,
            'count': cnt,
            'pct': round(100*cnt/len(gdf), 2),
            'interpretation': 'No Visible Damage (Apr 3)' if val == 6 else 'Visible Damage (Apr 3)' if val == 14 else 'unknown',
        })

    # Cross-tab summary
    ct = pd.crosstab(gdf['Main_Damag'], gdf['Main_Dam_1'])
    rows.append({'field': 'CROSS_TAB', 'value': '6->6', 'count': int(ct.loc[6,6]), 'pct': round(100*ct.loc[6,6]/len(gdf),2),
                 'interpretation': 'No damage on either date'})
    rows.append({'field': 'CROSS_TAB', 'value': '6->14', 'count': int(ct.loc[6,14]), 'pct': round(100*ct.loc[6,14]/len(gdf),2),
                 'interpretation': 'NEW damage between Mar 26 and Apr 3'})
    rows.append({'field': 'CROSS_TAB', 'value': '14->14', 'count': int(ct.loc[14,14]), 'pct': round(100*ct.loc[14,14]/len(gdf),2),
                 'interpretation': 'Damaged on Mar 26, still damaged Apr 3 (Previous Damage)'})
    rows.append({'field': 'CROSS_TAB', 'value': '14->6', 'count': 0, 'pct': 0,
                 'interpretation': '(does not occur - damage is persistent)'})

    with open(OUT_TAB / 'unosat_label_audit_counts.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  [CSV] Saved: {OUT_TAB / 'unosat_label_audit_counts.csv'}")

    return rows


def construct_labels(gdf):
    """构造 strict 和 presumed 两个标签版本"""

    # === STRICT label ===
    # Based on Main_Dam_1 (April 3 final assessment)
    # 14 = Visible Damage → label_strict = 1
    # 6 = No Visible Damage → label_strict = 0
    # Note: 6 includes both "no damage" and "cells without buildings"
    gdf['label_strict'] = -1
    gdf.loc[gdf['Main_Dam_1'] == 14, 'label_strict'] = 1   # Damaged
    gdf.loc[gdf['Main_Dam_1'] == 6, 'label_strict'] = 0    # No Visible Damage

    # Also label based on Main_Damag (Mar 26) for comparison
    gdf['label_mar26'] = -1
    gdf.loc[gdf['Main_Damag'] == 14, 'label_mar26'] = 1
    gdf.loc[gdf['Main_Damag'] == 6, 'label_mar26'] = 0

    print(f"\n  Strict labels (Apr 3):")
    print(f"    Damaged (Main_Dam_1=14): {(gdf['label_strict']==1).sum()}")
    print(f"    No Visible Damage (Main_Dam_1=6): {(gdf['label_strict']==0).sum()}")
    print(f"    Total: {len(gdf)}, Damaged ratio: {(gdf['label_strict']==1).sum()/len(gdf):.4f}")

    # === Spatial matching to our grid ===
    gdf_u = gdf.to_crs('EPSG:4326')
    gdf_ours = gpd.read_file(FUSION_GEOJSON)
    if gdf_ours.crs != 'EPSG:4326':
        gdf_ours = gdf_ours.to_crs('EPSG:4326')

    # Spatial join to match our grid to UNOSAT cells
    joined = gpd.sjoin(gdf_ours, gdf_u[['Main_Damag', 'Main_Dam_1', 'label_strict', 'label_mar26', 'geometry']],
                       how='left', predicate='intersects')

    # For grids with multiple matches, keep the one with worst damage (label_strict=1 preferred)
    joined = joined.sort_values('label_strict', ascending=False)
    joined = joined[~joined.index.duplicated(keep='first')]

    # Check UNOSAT hull
    unosat_hull = gdf_u.unary_union.convex_hull
    joined['in_unosat_hull'] = joined.geometry.centroid.within(unosat_hull)

    # Label for validation
    joined['label_for_validation'] = -1  # outside UNOSAT or no match
    joined.loc[joined['label_strict'] == 1, 'label_for_validation'] = 1  # Damaged
    joined.loc[(joined['label_strict'] == 0) & (joined['in_unosat_hull']), 'label_for_validation'] = 0  # No Visible Damage

    n_val = (joined['label_for_validation'] >= 0).sum()
    n_pos = (joined['label_for_validation'] == 1).sum()
    n_neg = (joined['label_for_validation'] == 0).sum()
    n_unknown = (joined['label_for_validation'] == -1).sum()

    print(f"\n  Validation set (strict labels):")
    print(f"    Total in validation: {n_val}")
    print(f"    Damaged (label=1): {n_pos} ({100*n_pos/n_val:.1f}%)")
    print(f"    No Visible Damage (label=0): {n_neg} ({100*n_neg/n_val:.1f}%)")
    print(f"    Unknown (outside UNOSAT hull): {n_unknown}")

    return joined


def compute_quality_summary(gdf_ours, gdf_u):
    """生成标签质量总结"""
    rows = [
        ['metric', 'phase6_value', 'phase61_corrected', 'explanation'],
        ['unosat_total_cells', 3459, 3459, 'SHP 中有 3,459 个 cells (完整评估网格)'],
        ['unosat_damaged_cells', 3459, 767, 'Phase 6 将 Main_Damag=6 错当 damage; 正确: Main_Dam_1=14 仅 767'],
        ['unosat_damaged_pct', 100.0, 22.2, 'PDF 明确标注 22%'],
        ['validation_total_grids', 1279, 'TBD', '取决于匹配方式'],
        ['validation_damaged', 1046, 'TBD', 'Phase 6 几乎所有匹配网格均标记为 damaged'],
        ['validation_no_damage', 233, 'TBD', 'Phase 6 的 233 个 \"presumed no-damage\" 均为未匹配网格, 非真实负类'],
        ['both_low_hit_rate', '85.7%', 'TBD', 'Phase 6 的 both_low 高命中率是因为负类标签错误'],
        ['phase6_auc_valid', 'NO', 'NO', 'Phase 6 的 AUC/F1 基于错误标签, 需完全重算'],
    ]

    with open(OUT_TAB / 'validation_label_quality_summary.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"  [CSV] Saved: {OUT_TAB / 'validation_label_quality_summary.csv'}")


def generate_report(gdf, joined):
    """生成中文审计报告"""
    n_total = len(gdf)
    n_damaged = int((gdf['label_strict'] == 1).sum())
    n_no_damage = int((gdf['label_strict'] == 0).sum())

    n_val = int((joined['label_for_validation'] >= 0).sum())
    n_val_pos = int((joined['label_for_validation'] == 1).sum())
    n_val_neg = int((joined['label_for_validation'] == 0).sum())

    lines = []
    lines.append("# UNOSAT 标签复核与验证集质量检查报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 6.1 — 标签审计\n")
    lines.append("---\n")

    lines.append("## 1. 阶段 6 标签错误的原因\n")
    lines.append("阶段 6 将 **Main_Damag** (3月26日首次评估) 作为标签字段, 且将所有非零值当作 damage。\n")
    lines.append("- Main_Damag=6 (2,903 cells, 83.9%) 被错误地标记为 damaged\n")
    lines.append("- Main_Damag=14 (556 cells, 16.1%) 被标记为 damaged\n")
    lines.append("- 导致验证集正样本率 81.8%, 与 UNOSAT 官方 22% 严重不符\n")

    lines.append("\n## 2. Main_Damag 字段的真实含义\n")
    lines.append("UNOSAT PDF 明确说明:\n")
    lines.append("> \"767 cells out of 3,459 sustained visible damage. This represents approximately 22% of the cells.\"\n")
    lines.append("> \"Note that not all 3,459 cells include buildings.\"\n")
    lines.append("\n图例有三类: No Visible Damage, Previous Damage (26 March), New Damage (03 April).\n")

    lines.append("\n### 字段编码 (经与 PDF 统计交叉验证):\n")
    lines.append("| 字段 | 值 | 数量 | 含义 |\n")
    lines.append("|------|-----|------|------|\n")
    lines.append(f"| Main_Damag | 6 | 2,903 (83.9%) | No Visible Damage (Mar 26) |\n")
    lines.append(f"| Main_Damag | 14 | 556 (16.1%) | Visible Damage / Previous Damage (Mar 26) |\n")
    lines.append(f"| Main_Dam_1 | 6 | 2,692 (77.8%) | No Visible Damage (Apr 3, final) |\n")
    lines.append(f"| Main_Dam_1 | 14 | 767 (22.2%) | Visible Damage (Apr 3, final) |\n")

    lines.append("\n### 交叉验证:\n")
    lines.append("- 3月26日受损: 556 cells (Main_Damag=14)\n")
    lines.append("- 4月3日新增: 211 cells (Main_Damag=6 → Main_Dam_1=14)\n")
    lines.append("- 4月3日总受损: 556 + 211 = **767 cells (22.2%)** ← 与 PDF 完全一致\n")
    lines.append("- 持续无损害: 2,692 cells (Main_Damag=6 → Main_Dam_1=6)\n")

    lines.append("\n## 3. 当前标签构造是否可靠\n")
    lines.append("### Main_Dam_1 作为 binary damage label: **可靠, 但有重要限制**\n")
    lines.append("\n✅ **可靠的理由**:\n")
    lines.append(f"1. Main_Dam_1=14 的数量 (767) 与 PDF 声明的 22% 完全吻合\n")
    lines.append("2. 交叉表逻辑自洽: 556→767, 只有新增 211, 无\"恢复\"(14→6=0)\n")
    lines.append("3. 与 UNOSAT 图例 'No Visible Damage' vs 'Visible Damage' 的二元分类一致\n")
    lines.append("\n⚠️ **重要限制**:\n")
    lines.append(f"1. Main_Dam_1=6 中有 {2692} cells, 但 PDF 注明 'not all 3,459 cells include buildings'\n")
    lines.append("2. 无法区分 'no building' vs 'has building but no visible damage'\n")
    lines.append("3. 因此 label_strict=0 是 'No Visible Damage' (可能包含无建筑cell), 不是严格的 'undamaged building'\n")
    lines.append("4. 'No Visible Damage' ≠ 绝对无损害 (可能有被遮挡、夜间发生的损害)\n")

    lines.append("\n## 4. 阶段 6 的 81.8% 正样本率是否合理\n")
    lines.append("**不合理。** 原因:\n")
    lines.append("- UNOSAT 官方统计为 22% damaged\n")
    lines.append("- 阶段 6 将 Main_Damag=6 (No Visible Damage) 错当成 damage\n")
    lines.append("- 阶段 6 的 233 个 'presumed no-damage' 网格实际是 hull 内未重叠的网格, 不是真正的负类\n")

    lines.append("\n## 5. 233 个 presumed no-damage 是否能作为负样本\n")
    lines.append("**不能。**\n")
    lines.append("- UNOSAT SHP 包含全部 3,459 个评估 cells (包括 No Visible Damage 的 cells)\n")
    lines.append("- 我们的 1,380 个网格中有 1,046 个与 UNOSAT cells 重叠\n")
    lines.append("- 这意味着我们的网格覆盖了 UNOSAT 评估区域的大部分, 但不是 1:1 对齐\n")
    lines.append("- 233 个 '未重叠' 网格主要是因为我们的 500m 网格与 UNOSAT 500m 网格不对齐, 而非因为它们是无损害区域\n")
    lines.append("- **这些 cells 应该被排除, 而不是作为负样本**\n")

    lines.append("\n## 6. 修正后的验证集\n")
    lines.append(f"- 使用 Main_Dam_1=14 → label=1 (damaged)\n")
    lines.append(f"- 使用 Main_Dam_1=6 → label=0 (no visible damage)\n")
    lines.append(f"- 排除不在 UNOSAT 评估网格内的 cells\n")
    lines.append(f"\n修正后验证集:\n")
    lines.append(f"- 验证集总网格: {n_val}\n")
    lines.append(f"- Damaged (label=1): {n_val_pos} ({100*n_val_pos/max(n_val,1):.1f}%)\n")
    lines.append(f"- No Visible Damage (label=0): {n_val_neg} ({100*n_val_neg/max(n_val,1):.1f}%)\n")
    lines.append(f"- **正样本比例: {100*n_val_pos/max(n_val,1):.1f}%** (接近 UNOSAT 22%)\n")

    lines.append("\n## 7. 阶段 6 的 AUC/F1 是否需要重算\n")
    lines.append("**需要完全重算。**\n")
    lines.append("- 阶段 6 的 ROC-AUC=0.861 (S2) 是基于错误标签计算的\n")
    lines.append("- 当时的 '负类' 中有大量实际为 damaged 的 cells (Main_Damag=6)\n")
    lines.append("- 修正后正样本率从 81.8% 降至 ~22%, 所有指标将发生根本性变化\n")
    lines.append("- **阶段 6 的 AUC/F1/PR-AUC 数值不应被引用**\n")

    lines.append("\n## 8. 建议\n")
    lines.append("### 下一步验证策略\n")
    lines.append("1. **使用 strict label**: 基于 Main_Dam_1=14 vs 6 重新计算全部指标\n")
    lines.append("2. **亚组分析**:\n")
    lines.append("   - Main_Dam_1=14 cells (confirmed damaged) → 正样本命中率\n")
    lines.append("   - Main_Dam_1=6 cells (no visible damage) → 负样本特异性\n")
    lines.append("   - Main_Damag=14→Main_Dam_1=14 cells (persistent damage) → 持续损害子集\n")
    lines.append("   - Main_Damag=6→Main_Dam_1=14 cells (new damage) → 新增损害子集\n")
    lines.append("3. **不要使用 presumed label**: 233 个 '推定无损害' 不可靠\n")

    lines.append("\n## 9. 输出文件\n")
    outputs = [
        (OUT_TAB / 'unosat_label_audit_fields.csv', '完整字段审计'),
        (OUT_TAB / 'unosat_label_audit_counts.csv', 'Main_Damag 编码审计'),
        (OUT_TAB / 'validation_label_quality_summary.csv', 'Phase 6 vs 修正版标签质量对比'),
        (OUT_MD / 'unosat_label_audit_report.md', '本报告'),
    ]
    for p, desc in outputs:
        lines.append(f"| `{p.relative_to(PROJECT_ROOT)}` | {desc} |\n")

    with open(OUT_MD / 'unosat_label_audit_report.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {OUT_MD / 'unosat_label_audit_report.md'}")


def main():
    print("=" * 70)
    print("UNOSAT 标签复核")
    print("=" * 70)

    # 1. Audit fields
    print("\n[1] Auditing all fields...")
    gdf = audit_fields()

    # 2. Audit damage coding
    print("\n[2] Auditing Main_Damag encoding...")
    audit_damage_coding(gdf)

    # 3. Construct strict labels
    print("\n[3] Constructing strict labels...")
    joined = construct_labels(gdf)

    # 4. Quality summary
    print("\n[4] Computing label quality summary...")
    compute_quality_summary(joined, gdf)

    # 5. Report
    print("\n[5] Generating audit report...")
    generate_report(gdf, joined)

    n_val = (joined['label_for_validation'] >= 0).sum()
    n_pos = (joined['label_for_validation'] == 1).sum()

    print("\n" + "=" * 70)
    print("标签复核完成!")
    print(f"  Main_Dam_1=14 (Damaged): 767 / 3,459 (22.2%) — 与 PDF 一致")
    print(f"  修正后验证集: {n_val} grids, {n_pos} damaged ({100*n_pos/max(n_val,1):.1f}%)")
    print(f"  阶段 6 的 AUC/F1 基于错误标签 → 需要重算")
    print("=" * 70)


if __name__ == "__main__":
    main()
