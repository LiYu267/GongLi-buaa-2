#!/usr/bin/env python3
"""
阶段 0 — 文件清点脚本
只扫描文件路径和大小，不读取大影像内容。
扫描范围: papers/ 和 原始数据/
"""

import os
import sys
import csv
import re
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
scan_roots = [
    project_root / "papers",
    project_root / "原始数据",
]

# ============================================================
# 识别规则
# ============================================================
S2_ROLE_MAP = {
    "战前1（2021.5）": "main_pre_20210523_T37TCN",
    "战前2（2021.5）": "main_pre_20210523_T37TDN",
    "战后1(2022.5)": "main_post_20220508_T37TCN",
    "战后2(2022.5)": "main_post_20220508_T37TDN",
}

S1_GRD_DATE_MAP = {
    "20210601": "s1_grd_pre_20210601",
    "20220616": "s1_grd_post_20220616",
}


def detect_sensor_and_product(path_str, fname):
    """根据路径和文件名检测传感器和产品类型"""
    sensor = "unknown"
    product_type = "unknown"
    role = "unknown"
    date_str = "unknown"
    tile = "unknown"
    notes = ""

    # Sentinel-2
    if any(kw in path_str for kw in ["Sentinel-2", "L2A", "GRANULE", "IMG_DATA"]):
        sensor = "Sentinel-2"
        product_type = "L2A"
        # Detect role from folder
        for folder_key, role_val in S2_ROLE_MAP.items():
            if folder_key in path_str:
                role = role_val
                break
        # Detect tile
        m_tile = re.search(r"(T37[TCD]N)", path_str)
        if m_tile:
            tile = m_tile.group(1)
        # Detect date
        m_date = re.search(r"(\d{8})T", path_str)
        if m_date:
            date_str = m_date.group(1)
        # Detect .jp2 band
        if fname.endswith(".jp2"):
            m_band = re.search(r"_(B\d{2}|B\dA|SCL|TCI|AOT|WVP|PVI)_", fname)
            if m_band:
                product_type = f"L2A_{m_band.group(1)}"
        elif fname.endswith(".xml"):
            product_type = "L2A_metadata"
        elif fname.endswith(".kml"):
            product_type = "L2A_kml"
        else:
            product_type = "L2A_support"

    # Sentinel-1 GRD
    elif "Sentinel-1 GRD" in path_str or "S1A_IW_GRDH" in fname:
        sensor = "Sentinel-1"
        product_type = "GRD_backscatter"
        m_s1 = re.match(r"(S1[AB])_(IW|EW|SM)_(GRDH|GRDM|GRDF|SLC)_(\dS[DH]V?)_(\d{8})T", fname)
        if m_s1:
            for date_key, role_val in S1_GRD_DATE_MAP.items():
                if date_key in fname:
                    role = role_val
                    date_str = date_key
                    break
            notes = "GRDH 后向散射幅度产品，非 InSAR 干涉相位"
        elif "SAFE" in fname:
            product_type = "SAFE_folder"
        elif fname.endswith(".tiff"):
            product_type = "GRD_measurement_tiff"
        elif fname.endswith(".xml"):
            product_type = "GRD_annotation_xml"
        elif fname.endswith(".xsd"):
            product_type = "GRD_schema_xsd"
        elif fname.endswith(".pdf"):
            product_type = "GRD_report_pdf"
        else:
            product_type = "GRD_support"

    # Sentinel-1 干涉相位
    elif "干涉相位" in path_str:
        sensor = "Sentinel-1"
        product_type = "InSAR_interferogram"
        role = "insar_optional"
        if "diff_pha" in fname:
            product_type = "InSAR_diff_phase"
        elif "cc" in fname and "geo" in fname:
            product_type = "InSAR_coherence"
        elif "diff_unfiltered" in fname:
            product_type = "InSAR_diff_unfiltered_phase"
        m_date = re.match(r"(\d{8})_(\d{8})", fname)
        if m_date:
            date_str = f"{m_date.group(1)}_{m_date.group(2)}"

    # VIIRS
    elif any(kw in path_str.lower() or kw in fname.lower() for kw in
             ["viirs", "vnp46", "blackmarble", "black_marble", "ntl", "nighttime", "nightlight", "夜光"]):
        sensor = "VIIRS"
        product_type = "NTL_H5"
        if "战前" in fname or "pre" in fname.lower():
            role = "viirs_pre"
        elif "战后" in fname or "post" in fname.lower():
            role = "viirs_post"
        notes = "VIIRS Nighttime Lights / Black Marble HDF5"

    # Papers
    elif "papers" in path_str.lower():
        sensor = "N/A"
        product_type = "reference_paper"

    return sensor, product_type, date_str, tile, role, notes


def get_file_size_mb(filepath):
    """获取文件大小(MB)"""
    try:
        return round(os.path.getsize(filepath) / (1024 * 1024), 4)
    except OSError:
        return 0.0


def main():
    records = []

    # Also check if VIIRS folder is empty
    viirs_dir = project_root / "原始数据" / "VIIRS"
    viirs_has_files = False

    for root in scan_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fname in filenames:
                full_path = Path(dirpath) / fname
                path_str = str(full_path)
                size_mb = get_file_size_mb(full_path)
                parent_folder = str(Path(dirpath).relative_to(project_root))
                ext = full_path.suffix.lower() if full_path.suffix else "(none)"

                sensor, product_type, date_str, tile, role, notes = detect_sensor_and_product(
                    path_str, fname
                )

                # Check VIIRS
                if "VIIRS" in parent_folder or sensor == "VIIRS":
                    viirs_has_files = True

                records.append({
                    "path": path_str.replace(str(project_root) + os.sep, ""),
                    "filename": fname,
                    "extension": ext,
                    "size_mb": size_mb,
                    "parent_folder": parent_folder,
                    "detected_sensor": sensor,
                    "detected_product_type": product_type,
                    "detected_date": date_str,
                    "detected_tile": tile,
                    "detected_role": role,
                    "notes": notes,
                })

    # Also check empty directories
    empty_dirs = []
    for root in scan_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if not dirnames and not filenames:
                empty_dirs.append(str(Path(dirpath).relative_to(project_root)))

    # Special: check VIIRS folder
    viirs_dir_path = project_root / "原始数据" / "VIIRS"
    if viirs_dir_path.exists() and not viirs_has_files:
        records.append({
            "path": str(viirs_dir_path.relative_to(project_root)),
            "filename": "(empty)",
            "extension": "",
            "size_mb": 0,
            "parent_folder": "原始数据/VIIRS",
            "detected_sensor": "VIIRS",
            "detected_product_type": "unknown",
            "detected_date": "unknown",
            "detected_tile": "unknown",
            "detected_role": "unknown",
            "notes": "当前未检测到 VIIRS 文件 — 文件夹为空",
        })

    # Write CSV
    out_dir = project_root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "file_inventory.csv"

    fieldnames = [
        "path", "filename", "extension", "size_mb", "parent_folder",
        "detected_sensor", "detected_product_type", "detected_date",
        "detected_tile", "detected_role", "notes"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Written: {csv_path}")
    print(f"Total records: {len(records)}")

    # Summary
    sensors = {}
    for r in records:
        s = r["detected_sensor"]
        sensors[s] = sensors.get(s, 0) + 1
    print("\nSensor summary:")
    for s, c in sorted(sensors.items()):
        print(f"  {s}: {c} files")

    if viirs_has_files:
        print("\nVIIRS: 检测到文件")
    else:
        print("\nVIIRS: 当前未检测到文件")

    print("\n文件清点完成。")


if __name__ == "__main__":
    main()
