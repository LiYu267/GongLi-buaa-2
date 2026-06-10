#!/usr/bin/env python3
"""
阶段 3a — VIIRS H5 文件诊断
检查两个 VIIRS .h5 文件结构, 识别夜光亮度层。
"""

import os, sys, csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIIRS_DIR = PROJECT_ROOT / "原始数据" / "VIIRS"
AOI_PATH = PROJECT_ROOT / "config" / "aoi_mariupol.geojson"
OUT_TAB = PROJECT_ROOT / "outputs" / "tables"
OUT_MD = PROJECT_ROOT / "outputs"
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_MD.mkdir(parents=True, exist_ok=True)

# Load AOI
with open(AOI_PATH, 'r') as f:
    aoi_data = json.load(f)
aoi_coords = aoi_data['features'][0]['geometry']['coordinates'][0]
aoi_lon_min = min(p[0] for p in aoi_coords)
aoi_lon_max = max(p[0] for p in aoi_coords)
aoi_lat_min = min(p[1] for p in aoi_coords)
aoi_lat_max = max(p[1] for p in aoi_coords)

def diagnose_h5(filepath, label):
    """全面诊断一个 H5 文件"""
    result = {
        'filepath': str(filepath),
        'label': label,
        'exists': filepath.exists(),
        'size_mb': round(filepath.stat().st_size / 1024 / 1024, 2) if filepath.exists() else 0,
        'datasets': [],
        'attributes': {},
    }

    if not filepath.exists():
        return result

    with h5py.File(filepath, 'r') as h5:
        # 全局属性
        for key in h5.attrs:
            val = h5.attrs[key]
            result['attributes'][key] = str(val)[:500]

        # 递归列出所有 dataset
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                info = {
                    'path': name,
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size_mb': round(np.prod(obj.shape) * obj.dtype.itemsize / 1024 / 1024, 4),
                    'attrs': {},
                }
                for k in obj.attrs:
                    v = obj.attrs[k]
                    info['attrs'][k] = str(v)[:300]
                result['datasets'].append(info)

        h5.visititems(visit)

    return result


def identify_ntl_layer(datasets):
    """识别夜光亮度层, 按优先级匹配关键词"""
    keywords_priority = [
        # 高优先级: BRDF 校正后的无雪 NTL
        ['BRDF-Corrected_NTL', 'Snow_Free'],
        ['Gap_Filled_DNB_BRDF-Corrected_NTL', 'Snow_Free'],
        ['DNB_BRDF-Corrected_NTL'],
        ['Gap_Filled_DNB_BRDF-Corrected_NTL'],
        # 中优先级
        ['All_Angles', 'Snow_Free', 'NTL'],
        ['Near_Nadir', 'Snow_Free', 'NTL'],
        ['Off_Nadir', 'Snow_Free', 'NTL'],
        # 低优先级: 任何 NTL 或 Radiance
        ['NTL'],
        ['DNB'],
        ['Radiance'],
        ['Corrected'],
    ]

    for kw_list in keywords_priority:
        candidates = []
        for ds in datasets:
            name = ds['path']
            if all(kw.lower() in name.lower().replace(' ', '_').replace('-', '_')
                   for kw in kw_list):
                candidates.append(ds)
        if candidates:
            # 优选 float32, 其次 float64, 排除 uint/int
            float_candidates = [d for d in candidates if 'float' in d['dtype'].lower()]
            if float_candidates:
                return float_candidates[0], kw_list
            return candidates[0], kw_list

    return None, None


def check_coverage(ds_info):
    """尝试从 H5 dataset 中获取地理信息"""
    # VNP46A3 月产品通常包含 lat/lon 数组
    shape = ds_info['shape']
    attrs = ds_info['attrs']

    # 寻找 lat/lon 信息
    lat_info = None
    lon_info = None
    for k, v in attrs.items():
        v_lower = (k + ' ' + v).lower()
        if 'lat' in k.lower() and ('min' in v_lower or 'max' in v_lower or 'range' in v_lower):
            lat_info = v
        if 'lon' in k.lower() and ('min' in v_lower or 'max' in v_lower or 'range' in v_lower):
            lon_info = v

    return {
        'shape': shape,
        'lat_info': lat_info,
        'lon_info': lon_info,
    }


def main():
    h5_files = [
        (VIIRS_DIR / "战前夜光.h5", "战前 (VNP46A3)"),
        (VIIRS_DIR / "战后夜光.h5", "战后 (VNP46A3)"),
    ]

    print("=" * 70)
    print("VIIRS H5 文件诊断")
    print("=" * 70)

    all_results = []
    for fpath, label in h5_files:
        print(f"\n--- {label} ---")
        print(f"    File: {fpath}")
        res = diagnose_h5(fpath, label)
        print(f"    Exists: {res['exists']}, Size: {res['size_mb']} MB")
        print(f"    Datasets found: {len(res['datasets'])}")
        for ds in res['datasets']:
            print(f"      {ds['path']}: shape={ds['shape']}, dtype={ds['dtype']}")
        all_results.append(res)

    # 识别 NTL 层
    print("\n--- 识别夜光亮度层 ---")
    ntl_choices = {}
    for res in all_results:
        if not res['exists']:
            continue
        ntl_ds, keywords = identify_ntl_layer(res['datasets'])
        ntl_choices[res['label']] = (ntl_ds, keywords)
        if ntl_ds:
            print(f"  {res['label']}: [OK] {ntl_ds['path']} (matched: {keywords})")
        else:
            print(f"  {res['label']}: ❌ 未找到合适的 NTL 层")

    # 写 CSV
    csv_path = OUT_TAB / "viirs_h5_layers.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['h5_file', 'dataset_path', 'shape', 'dtype', 'size_mb', 'attrs'])
        for res in all_results:
            for ds in res['datasets']:
                writer.writerow([
                    res['label'],
                    ds['path'],
                    str(ds['shape']),
                    ds['dtype'],
                    ds['size_mb'],
                    json.dumps(ds['attrs'], ensure_ascii=False),
                ])
    print(f"\n  [CSV] Saved: {csv_path}")

    # 生成诊断报告
    lines = []
    lines.append("# VIIRS H5 文件诊断报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**阶段**: 阶段 3a — VIIRS H5 诊断\n")
    lines.append("---\n")

    lines.append("## 1. 输入文件\n")
    for res in all_results:
        lines.append(f"- **{res['label']}**: `{res['filepath']}`\n")
        lines.append(f"  - 存在: {'✅' if res['exists'] else '❌'}\n")
        lines.append(f"  - 大小: {res['size_mb']} MB\n")

    lines.append("\n## 2. Dataset 清单\n")
    for res in all_results:
        lines.append(f"### {res['label']}\n")
        lines.append("| Dataset | Shape | Dtype | Size (MB) |\n")
        lines.append("|---------|-------|-------|-----------|\n")
        for ds in res['datasets']:
            lines.append(f"| `{ds['path']}` | {ds['shape']} | {ds['dtype']} | {ds['size_mb']:.4f} |\n")

    lines.append("\n## 3. 夜光亮度层识别\n")
    for res in all_results:
        if not res['exists']:
            continue
        ntl_ds, kw = ntl_choices.get(res['label'], (None, None))
        if ntl_ds:
            lines.append(f"### {res['label']}\n")
            lines.append(f"- ✅ **选中**: `{ntl_ds['path']}`\n")
            lines.append(f"- 匹配关键词: {kw}\n")
            lines.append(f"- Shape: {ntl_ds['shape']}\n")
            lines.append(f"- Dtype: {ntl_ds['dtype']}\n")
            lines.append(f"- 属性: {json.dumps(ntl_ds['attrs'], ensure_ascii=False)}\n")
        else:
            lines.append(f"### {res['label']}\n")
            lines.append(f"- ❌ 未找到合适的 NTL 层\n")

    # 地理参考信息
    lines.append("\n## 4. 地理参考与 AOI 覆盖\n")
    lines.append(f"- Mariupol AOI (EPSG:4326): lon=[{aoi_lon_min}, {aoi_lon_max}], lat=[{aoi_lat_min}, {aoi_lat_max}]\n")

    # 尝试读取 lat/lon 数组
    for res in all_results:
        if not res['exists']:
            continue
        lines.append(f"### {res['label']}\n")
        with h5py.File(res['filepath'], 'r') as h5:
            lat_key = None
            lon_key = None
            for ds in res['datasets']:
                if ds['path'].endswith('/lat') or ds['path'] == 'lat':
                    lat_key = ds['path']
                if ds['path'].endswith('/lon') or ds['path'] == 'lon':
                    lon_key = ds['path']

            if lat_key and lon_key:
                lats = h5[lat_key][:]
                lons = h5[lon_key][:]
                lines.append(f"- Grid lat: [{lats.min():.4f}, {lats.max():.4f}] (from `{lat_key}`)\n")
                lines.append(f"- Grid lon: [{lons.min():.4f}, {lons.max():.4f}] (from `{lon_key}`)\n")

                # Check overlap with AOI
                lat_overlap = not (lats.max() < aoi_lat_min or lats.min() > aoi_lat_max)
                lon_overlap = not (lons.max() < aoi_lon_min or lons.min() > aoi_lon_max)
                covers = lat_overlap and lon_overlap
                lines.append(f"- **覆盖 Mariupol AOI**: {'✅ 是' if covers else '❌ 否'}\n")
                if not covers:
                    lines.append(f"  - 经度重叠: {lon_overlap}, 纬度重叠: {lat_overlap}\n")
            else:
                lines.append("- ⚠️ 未找到 lat/lon 数据, 无法直接判断 AOI 覆盖\n")
                lines.append("- 将尝试通过 VIIRS grid 参数推断\n")

    # 全局属性
    lines.append("\n## 5. H5 全局属性\n")
    for res in all_results:
        if not res['exists']:
            continue
        lines.append(f"### {res['label']}\n")
        for k, v in res['attributes'].items():
            lines.append(f"- **{k}**: `{v[:200]}`\n")

    # 结论
    lines.append("\n## 6. 诊断结论\n")
    all_identified = all(ntl_choices.get(r['label'], (None,))[0] is not None
                         for r in all_results if r['exists'])
    if all_identified:
        lines.append("> ✅ **两个 H5 文件的夜光亮度层均已成功识别, 可以进入 VIIRS NTL 提取阶段。**\n")
    else:
        lines.append("> ❌ **部分文件未能识别夜光亮度层, 请人工检查 dataset 清单。**\n")

    report_path = OUT_MD / "viirs_h5_diagnosis.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  [Report] Saved: {report_path}")

    print("\n诊断完成.")
    return 0 if all_identified else 1


if __name__ == "__main__":
    sys.exit(main())
