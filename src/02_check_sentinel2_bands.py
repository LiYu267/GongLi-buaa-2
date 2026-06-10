#!/usr/bin/env python3
"""
阶段 0 — Sentinel-2 波段文件检查
只检查文件路径是否存在，不读取像元数据。
"""

import sys
import os
import csv
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# ============================================================
# 四个主实验文件夹配置
# ============================================================
s2_folders = [
    {
        "role": "main_pre_20210523_T37TCN",
        "folder": "战前1（2021.5）",
        "date": "20210523",
        "tile": "T37TCN",
        "path": project_root / "原始数据/Sentinel-2/战前1（2021.5）/GRANULE/L2A_T37TCN_A030906_20210523T083300/IMG_DATA",
    },
    {
        "role": "main_pre_20210523_T37TDN",
        "folder": "战前2（2021.5）",
        "date": "20210523",
        "tile": "T37TDN",
        "path": project_root / "原始数据/Sentinel-2/战前2（2021.5）/GRANULE/L2A_T37TDN_A030906_20210523T083300/IMG_DATA",
    },
    {
        "role": "main_post_20220508_T37TCN",
        "folder": "战后1(2022.5)",
        "date": "20220508",
        "tile": "T37TCN",
        "path": project_root / "原始数据/Sentinel-2/战后1(2022.5)/GRANULE/L2A_T37TCN_A035911_20220508T083304/IMG_DATA",
    },
    {
        "role": "main_post_20220508_T37TDN",
        "folder": "战后2(2022.5)",
        "date": "20220508",
        "tile": "T37TDN",
        "path": project_root / "原始数据/Sentinel-2/战后2(2022.5)/GRANULE/L2A_T37TDN_A035911_20220508T083304/IMG_DATA",
    },
]

# 需要查找的波段以及预期所在分辨率
bands_to_check = [
    ("B03", "R10m"),
    ("B04", "R10m"),
    ("B08", "R10m"),
    ("B11", "R20m"),
    ("B12", "R20m"),
    ("SCL", "R20m"),
]


def find_band_file(img_data_path, band, resolution, tile, date):
    """查找指定波段的 .jp2 文件"""
    res_dir = img_data_path / resolution
    if not res_dir.exists():
        return None

    # 查找匹配的文件
    for fname in os.listdir(res_dir):
        if fname.endswith(".jp2") and band in fname:
            # 确认 tile 和 date 匹配
            if tile in fname or band == "SCL":
                return str(res_dir / fname)

    return None


def main():
    results = []

    for folder_info in s2_folders:
        role = folder_info["role"]
        folder_name = folder_info["folder"]
        date = folder_info["date"]
        tile = folder_info["tile"]
        img_data_path = folder_info["path"]

        row = {
            "role": role,
            "folder": folder_name,
            "date": date,
            "tile": tile,
            "B03_path": "",
            "B04_path": "",
            "B08_path": "",
            "B11_path": "",
            "B12_path": "",
            "SCL_path": "",
            "missing_bands": "",
            "ready_for_processing": "yes",
        }

        missing = []
        for band, resolution in bands_to_check:
            col_name = f"{band}_path"
            file_path = find_band_file(img_data_path, band, resolution, tile, date)
            if file_path:
                row[col_name] = file_path
            else:
                row[col_name] = "MISSING"
                missing.append(f"{band}({resolution})")

        if missing:
            row["missing_bands"] = "; ".join(missing)
            row["ready_for_processing"] = "NO — 缺少必要波段"
        else:
            row["missing_bands"] = ""
            row["ready_for_processing"] = "yes"

        # Check if folder exists at all
        if not img_data_path.exists():
            row["ready_for_processing"] = "NO — 文件夹不存在"
            row["missing_bands"] = "文件夹不存在"
            for band, _ in bands_to_check:
                row[f"{band}_path"] = "FOLDER_NOT_FOUND"

        results.append(row)

    # ============================================================
    # 写入 CSV
    # ============================================================
    out_dir = project_root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sentinel2_band_check.csv"

    fieldnames = [
        "role", "folder", "date", "tile",
        "B03_path", "B04_path", "B08_path",
        "B11_path", "B12_path", "SCL_path",
        "missing_bands", "ready_for_processing",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Written: {csv_path}")

    # ============================================================
    # 打印摘要
    # ============================================================
    print("\n" + "=" * 70)
    print("Sentinel-2 波段检查摘要")
    print("=" * 70)
    all_ready = True
    for r in results:
        status = "[OK]" if r["ready_for_processing"] == "yes" else "[FAIL]"
        print(f"  {status} {r['role']}")
        if r["missing_bands"]:
            print(f"     Missing: {r['missing_bands']}")
            all_ready = False
        else:
            print(f"     All bands present")

    if all_ready:
        print("\n[OK] All four Sentinel-2 tiles have complete bands, ready for preprocessing.")
    else:
        print("\n[FAIL] Some tiles missing bands, need to supplement data.")

    print("\nSentinel-2 波段检查完成。")


if __name__ == "__main__":
    main()
