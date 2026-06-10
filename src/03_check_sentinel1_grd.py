#!/usr/bin/env python3
"""
阶段 0 — Sentinel-1 GRD 结构检查
只检查 .SAFE 文件夹结构，不处理 SAR 数据。
"""

import sys
import os
import csv
import re
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# ============================================================
# Sentinel-1 GRD .SAFE 文件夹
# ============================================================
s1_grd_dir = project_root / "原始数据" / "Sentinel-1 GRD"

s1_folders = [
    {
        "safe_name": "S1A_IW_GRDH_1SDV_20210601T033952_20210601T034017_038141_048069_5417.SAFE",
        "expected_role": "s1_grd_pre_20210601",
    },
    {
        "safe_name": "S1A_IW_GRDH_1SDV_20220616T152025_20220616T152050_043690_053749_2778.SAFE",
        "expected_role": "s1_grd_post_20220616",
    },
]


def parse_s1_filename(fname):
    """从 Sentinel-1 文件名解析元数据"""
    # Pattern: S1A_IW_GRDH_1SDV_20210601T033952_...
    pattern = r"^(S1[AB])_(IW|EW|SM)_(GRDH|GRDM|GRDF|SLC)_(\dS[DH]V?)_(\d{8})T(\d{6})_(\d{8})T(\d{6})_(\d+)_(\d+)_([A-F0-9]+)"
    m = re.match(pattern, fname)
    if m:
        return {
            "satellite": m.group(1),
            "mode": m.group(2),
            "product_type": m.group(3),
            "polarization": m.group(4),
            "acquisition_date": m.group(5),
            "acquisition_time": m.group(6),
        }
    return None


def check_safe_structure(safe_path):
    """检查 .SAFE 文件夹结构"""
    checks = {
        "manifest.safe": os.path.exists(safe_path / "manifest.safe"),
        "measurement": os.path.isdir(safe_path / "measurement"),
        "annotation": os.path.isdir(safe_path / "annotation"),
        "annotation/calibration": os.path.isdir(safe_path / "annotation" / "calibration"),
    }

    # Check measurement TIFFs
    measurement_dir = safe_path / "measurement"
    measurement_files = []
    if checks["measurement"]:
        measurement_files = [f for f in os.listdir(measurement_dir) if f.endswith(".tiff")]

    # Check annotation files
    annotation_dir = safe_path / "annotation"
    annotation_files = []
    if checks["annotation"]:
        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith(".xml")]

    # Check calibration files
    cal_dir = safe_path / "annotation" / "calibration"
    cal_files = []
    if checks["annotation/calibration"]:
        cal_files = [f for f in os.listdir(cal_dir) if f.endswith(".xml")]

    # Check RFI files (may not exist in all products)
    rfi_dir = safe_path / "annotation" / "rfi"
    has_rfi = os.path.isdir(rfi_dir)

    return {
        **checks,
        "measurement_files": measurement_files,
        "annotation_files": annotation_files,
        "calibration_files": cal_files,
        "has_rfi": has_rfi,
        "structure_ok": all(checks.values()),
    }


def main():
    results = []
    md_lines = []

    md_lines.append("# Sentinel-1 GRD 结构诊断报告\n")
    md_lines.append("## 产品类型确认\n")
    md_lines.append("这两个产品是 **GRDH (Ground Range Detected, High resolution)** 产品，\n")
    md_lines.append("属于 **Sentinel-1 GRD 后向散射幅度数据**，不是 InSAR 干涉相位数据。\n")
    md_lines.append("GRDH 可以用于后向散射变化检测，但不能直接用于干涉相位分析。\n")

    for folder_info in s1_folders:
        safe_name = folder_info["safe_name"]
        safe_path = s1_grd_dir / safe_name
        expected_role = folder_info["expected_role"]

        print(f"\nChecking: {safe_name}")
        print(f"  Path exists: {safe_path.exists()}")

        if not safe_path.exists():
            results.append({
                "safe_name": safe_name,
                "expected_role": expected_role,
                "exists": False,
                "structure_ok": False,
                "manifest_safe": False,
                "measurement_dir": False,
                "annotation_dir": False,
                "annotation_cal_dir": False,
                "has_rfi": False,
                "measurement_files": "",
                "satellite": "unknown",
                "mode": "unknown",
                "product_type_s1": "unknown",
                "acquisition_date": "unknown",
                "polarization": "unknown",
                "notes": "SAFE 文件夹不存在",
            })
            md_lines.append(f"\n## {safe_name}\n")
            md_lines.append("[FAIL] **文件夹不存在**\n")
            continue

        # Parse filename
        parsed = parse_s1_filename(safe_name.replace(".SAFE", ""))
        structure = check_safe_structure(safe_path)

        print(f"  Structure OK: {structure['structure_ok']}")
        print(f"  manifest.safe: {structure['manifest.safe']}")
        print(f"  measurement/: {structure['measurement']} ({len(structure['measurement_files'])} TIFFs)")
        print(f"  annotation/: {structure['annotation']} ({len(structure['annotation_files'])} XMLs)")
        print(f"  annotation/calibration/: {structure['annotation/calibration']} ({len(structure['calibration_files'])} XMLs)")
        print(f"  RFI: {structure['has_rfi']}")

        if parsed:
            print(f"  Satellite: {parsed['satellite']}")
            print(f"  Mode: {parsed['mode']}")
            print(f"  Product: {parsed['product_type']}")
            print(f"  Polarization: {parsed['polarization']}")
            print(f"  Date: {parsed['acquisition_date']}")

        notes = "GRDH 后向散射幅度产品，可用于 Sentinel-1 GRD 后向散射变化检测模块"
        if not structure["structure_ok"]:
            missing = [k for k, v in structure.items() if isinstance(v, bool) and not v and k not in ("has_rfi", "structure_ok")]
            notes += f" | 警告: 缺少 {', '.join(missing)}"

        results.append({
            "safe_name": safe_name,
            "expected_role": expected_role,
            "exists": True,
            "structure_ok": structure["structure_ok"],
            "manifest_safe": structure["manifest.safe"],
            "measurement_dir": structure["measurement"],
            "annotation_dir": structure["annotation"],
            "annotation_cal_dir": structure["annotation/calibration"],
            "has_rfi": structure["has_rfi"],
            "measurement_files": "; ".join(structure["measurement_files"]),
            "satellite": parsed["satellite"] if parsed else "unknown",
            "mode": parsed["mode"] if parsed else "unknown",
            "product_type_s1": parsed["product_type"] if parsed else "unknown",
            "acquisition_date": parsed["acquisition_date"] if parsed else "unknown",
            "polarization": parsed["polarization"] if parsed else "unknown",
            "notes": notes,
        })

        # MD section
        md_lines.append(f"\n## {safe_name}\n")
        if parsed:
            md_lines.append(f"- **卫星**: {parsed['satellite']}\n")
            md_lines.append(f"- **模式**: {parsed['mode']}\n")
            md_lines.append(f"- **产品类型**: {parsed['product_type']} (Ground Range Detected, High resolution)\n")
            md_lines.append(f"- **极化方式**: {parsed['polarization']}\n")
            md_lines.append(f"- **采集日期**: {parsed['acquisition_date']}\n")
        md_lines.append(f"- **角色**: {expected_role}\n")
        status = "[OK]" if structure["structure_ok"] else "[FAIL]"
        md_lines.append(f"- **结构完整**: {status}\n")
        md_lines.append(f"- **manifest.safe**: {'[OK]' if structure['manifest.safe'] else '[FAIL]'}\n")
        md_lines.append(f"- **measurement/**: {'[OK]' if structure['measurement'] else '[FAIL]'} ({len(structure['measurement_files'])} TIFFs)\n")
        md_lines.append(f"- **annotation/**: {'[OK]' if structure['annotation'] else '[FAIL]'} ({len(structure['annotation_files'])} XMLs)\n")
        md_lines.append(f"- **annotation/calibration/**: {'[OK]' if structure['annotation/calibration'] else '[FAIL]'} ({len(structure['calibration_files'])} XMLs)\n")

    # Write CSV
    out_dir = project_root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sentinel1_grd_check.csv"

    fieldnames = [
        "safe_name", "expected_role", "exists", "structure_ok",
        "manifest_safe", "measurement_dir", "annotation_dir", "annotation_cal_dir",
        "has_rfi", "measurement_files",
        "satellite", "mode", "product_type_s1", "acquisition_date", "polarization",
        "notes",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWritten: {csv_path}")

    # Write MD
    md_path = project_root / "outputs" / "sentinel1_grd_diagnosis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)

    print(f"Written: {md_path}")

    # Summary
    all_ok = all(r["structure_ok"] for r in results)
    if all_ok:
        print("\n[OK] All Sentinel-1 GRD .SAFE structures complete, suitable for subsequent advanced modules.")
    else:
        print("\n[FAIL] Some .SAFE structures incomplete.")

    print("\nSentinel-1 GRD 检查完成。")


if __name__ == "__main__":
    main()
