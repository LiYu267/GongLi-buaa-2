#!/usr/bin/env python3
"""
阶段 0 — 环境探测脚本
只检查 Python 环境和已安装库，不安装任何新库。
"""

import sys
import os
from pathlib import Path

# ============================================================
# 1. Python 可执行路径与版本
# ============================================================
python_path = sys.executable
python_version = sys.version

print(f"Python executable: {python_path}")
print(f"Python version: {python_version}")

# ============================================================
# 2. 库检查
# ============================================================
libs_to_check = [
    "os",
    "pathlib",
    "json",
    "csv",
    "zipfile",
    "xml.etree.ElementTree",
    "numpy",
    "pandas",
    "rasterio",
    "geopandas",
    "shapely",
    "pyproj",
    "h5py",
    "xarray",
    "rioxarray",
    "matplotlib",
    "scipy",
    "sklearn",
]

results = []

for lib in libs_to_check:
    status = "available"
    note = ""
    try:
        if lib == "xml.etree.ElementTree":
            import xml.etree.ElementTree
        elif lib == "sklearn":
            import sklearn
        else:
            __import__(lib)
    except ImportError as e:
        status = "missing"
        note = str(e)
    results.append((lib, status, note))
    print(f"  {lib:30s} -> {status}")

# ============================================================
# 3. 输出目录
# ============================================================
project_root = Path(__file__).resolve().parent.parent
out_dir = project_root / "outputs"
tables_dir = out_dir / "tables"
out_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. 写入 outputs/environment_check.md
# ============================================================
md_path = out_dir / "environment_check.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# 环境检查报告\n\n")
    f.write(f"- **Python 路径**: `{python_path}`\n")
    f.write(f"- **Python 版本**: `{python_version.strip()}`\n\n")
    f.write("| 库 | 状态 | 备注 |\n")
    f.write("|----|------|------|\n")
    for lib, status, note in results:
        emoji = "✅" if status == "available" else "❌"
        f.write(f"| {lib} | {emoji} {status} | {note} |\n")

print(f"\nWritten: {md_path}")

# ============================================================
# 5. 写入 outputs/tables/environment_packages.csv
# ============================================================
csv_path = tables_dir / "environment_packages.csv"
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    f.write("package,status,note\n")
    for lib, status, note in results:
        f.write(f"{lib},{status},{note}\n")

print(f"Written: {csv_path}")
print("\n环境检查完成。")
