"""
实验 v2：四传感器冲突损害贝叶斯不确定性融合
============================================
传感器清单:
  1. VIIRS 夜间灯光 (VNP46A2, 500m) — 战前2022.01.08 → 战后2022.05.08
  2. Sentinel-2 多光谱 (MSI L2A, 10m) — 战前2021.05.23 → 战后2022.05.08 (同季节!)
  3. Sentinel-1 SAR 幅度 (GRD IW, ~10m) — 战前2021.06.01 → 战后2022.06.16
  4. InSAR 相干性 (Interferometric Coherence) — 战前/战后干涉对

输出目录: exp_outputs_v2/
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os, json, warnings
warnings.filterwarnings('ignore')

BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs_v2'
OUT.mkdir(exist_ok=True)
RNG = np.random.RandomState(42)

print("=" * 65)
print("实验 v2：四传感器冲突损害贝叶斯融合 — 马里乌波尔")
print("=" * 65)

# ============================================================
# 传感器 1: VIIRS 夜间灯光
# ============================================================
print("\n[1/4] VIIRS 夜间灯光 (500m)")

import h5py
f_pre = h5py.File(BASE / 'VNP46A2.A2022008.h21v04.002.2025114204956.h5', 'r')
f_post = h5py.File(BASE / 'VNP46A2.A2022128.h21v04.002.2025119063131.h5', 'r')
ntl_pre = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL'][:]
ntl_post = f_post['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL'][:]
lat_v = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lat'][:]
lon_v = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lon'][:]
qf_pre = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Mandatory_Quality_Flag'][:]
qf_post = f_post['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Mandatory_Quality_Flag'][:]
f_pre.close(); f_post.close()

lat_2d = lat_v[:, np.newaxis]; lon_2d = lon_v[np.newaxis, :]
aoi_mask = ((lat_2d >= 46.85) & (lat_2d <= 47.86) &
            (lon_2d >= 37.66) & (lon_2d <= 39.14))
quality_mask = (qf_pre <= 2) & (qf_post <= 2)
full_mask = aoi_mask & quality_mask
rows = np.any(full_mask, axis=1); cols = np.any(full_mask, axis=0)
dNTL = ntl_post[rows,:][:,cols] - ntl_pre[rows,:][:,cols]
qsub = quality_mask[rows,:][:,cols]
dNTL_v = np.where(qsub, dNTL, np.nan)

valid1 = ~np.isnan(dNTL_v)
gmm1 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(dNTL_v[valid1].reshape(-1,1))
dm = np.argmin(gmm1.means_.flatten())
p1 = np.full_like(dNTL_v, np.nan, dtype=np.float32)
p1[valid1] = gmm1.predict_proba(dNTL_v[valid1].reshape(-1,1))[:, dm].astype(np.float32)

# Bootstrap σ²
n1 = np.sum(valid1); p1_boot = np.zeros((100, n1))
for b in range(100):
    ib = RNG.choice(n1, n1, replace=True)
    gb = GaussianMixture(n_components=2, random_state=b, n_init=3)
    try:
        gb.fit(dNTL_v[valid1].reshape(-1,1)[ib])
        p1_boot[b] = gb.predict_proba(dNTL_v[valid1].reshape(-1,1))[:, np.argmin(gb.means_.flatten())]
    except: p1_boot[b] = p1[valid1]
s1 = np.full_like(dNTL_v, np.nan, dtype=np.float32)
s1[valid1] = np.std(p1_boot, axis=0).astype(np.float32)

print(f"  ΔNTL mean={np.nanmean(dNTL_v):.2f}, 下降{np.sum(dNTL_v<0)}/{np.sum(qsub)}像素")
print(f"  GMM: 正常={gmm1.means_.flatten()[1-dm]:.2f}, 损害={gmm1.means_.flatten()[dm]:.2f}, 权重损害={gmm1.weights_[dm]:.1%}")
print(f"  p₁: mean={np.nanmean(p1):.4f}, σ₁: mean={np.nanmean(s1[valid1]):.4f}")

# ============================================================
# 传感器 2: Sentinel-2 多光谱 (May→May 同季节!)
# ============================================================
print("\n[2/4] Sentinel-2 多光谱 (10m, May 2021 → May 2022)")

def find_s2_jp2(s2_dir, band_pat):
    for root, dirs, files in os.walk(s2_dir):
        for f in files:
            if band_pat in f and f.endswith('.jp2') and 'MSK_' not in f and 'QI_DATA' not in root:
                return os.path.join(root, f)
    return None

def load_s2(s2_dir, band_pat):
    import rasterio
    fp = find_s2_jp2(s2_dir, band_pat)
    with rasterio.open(fp) as src:
        return src.read(1).astype(np.float32)

def calc_delta(pre_d, post_d, idx_name):
    if idx_name == 'NDVI':
        pre = (load_s2(pre_d, 'B08_10m') - load_s2(pre_d, 'B04_10m')) / (load_s2(pre_d, 'B08_10m') + load_s2(pre_d, 'B04_10m') + 1e-10)
        post = (load_s2(post_d, 'B08_10m') - load_s2(post_d, 'B04_10m')) / (load_s2(post_d, 'B08_10m') + load_s2(post_d, 'B04_10m') + 1e-10)
    elif idx_name == 'NBR':
        pre = (load_s2(pre_d, 'B8A_20m') - load_s2(pre_d, 'B12_20m')) / (load_s2(pre_d, 'B8A_20m') + load_s2(pre_d, 'B12_20m') + 1e-10)
        post = (load_s2(post_d, 'B8A_20m') - load_s2(post_d, 'B12_20m')) / (load_s2(post_d, 'B8A_20m') + load_s2(post_d, 'B12_20m') + 1e-10)
    elif idx_name == 'NDBI':
        pre = (load_s2(pre_d, 'B11_20m') - load_s2(pre_d, 'B8A_20m')) / (load_s2(pre_d, 'B11_20m') + load_s2(pre_d, 'B8A_20m') + 1e-10)
        post = (load_s2(post_d, 'B11_20m') - load_s2(post_d, 'B8A_20m')) / (load_s2(post_d, 'B11_20m') + load_s2(post_d, 'B8A_20m') + 1e-10)
    return post - pre

# Pre-war: s2_may2021, Post-war: 战后1(T37TCN) / 战后2(T37TDN)
pre_s2 = {
    'T37TDN': str(BASE / 's2_may2021/T37TDN'),
    'T37TCN': str(BASE / 's2_may2021/T37TCN'),
}
post_s2 = {
    'T37TDN': str(BASE / '战后2'),
    'T37TCN': str(BASE / '战后1'),
}

p2_tiles = {}
for tile in ['T37TDN', 'T37TCN']:
    pre_d = pre_s2[tile]; post_d = post_s2[tile]
    print(f"  {tile}: 计算Δ指数...")
    dNDVI = calc_delta(pre_d, post_d, 'NDVI')
    dNBR  = calc_delta(pre_d, post_d, 'NBR')
    dNDBI = calc_delta(pre_d, post_d, 'NDBI')
    print(f"    ΔNDVI={np.nanmean(dNDVI):.4f}, ΔNBR={np.nanmean(dNBR):.4f}, ΔNDBI={np.nanmean(dNDBI):.4f}")

    # SCL mask (20m)
    scl_pre = load_s2(pre_d, 'SCL_20m'); scl_post = load_s2(post_d, 'SCL_20m')
    mask = (scl_pre >= 2) & (scl_pre <= 7) & (scl_post >= 2) & (scl_post <= 7)
    H20, W20 = mask.shape

    # Resample NDVI 10m→20m
    dNDVI_20 = dNDVI[:H20*2:2, :W20*2:2]
    valid_idx = np.where(mask.ravel())[0]
    n_samp = min(50000, len(valid_idx))
    idx_s = RNG.choice(valid_idx, n_samp, replace=False)

    X = np.column_stack([dNDVI_20.ravel()[idx_s], dNBR.ravel()[idx_s], dNDBI.ravel()[idx_s]])
    X_s = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3).fit(X_s)
    pc1 = pca.transform(X_s)[:, 0]
    print(f"    PCA: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

    # 使用 GMM 于 PC1 来区分"正常变化"与"极端变化（损害）"
    # May→May 消除了季节效应，大部分像素变化≈0，极端值=潜在损害
    pc1_abs = np.abs(pc1)
    gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(pc1_abs.reshape(-1,1))
    dm2 = np.argmax(gmm2.means_.flatten())  # 损害=变化幅度更大的分量
    p2_vals = gmm2.predict_proba(pc1_abs.reshape(-1,1))[:, dm2]
    print(f"    GMM on |PC1|: 正常={gmm2.means_.flatten()[1-dm2]:.3f}, 损害={gmm2.means_.flatten()[dm2]:.3f}, 损害权重={gmm2.weights_[dm2]:.1%}")

    p2_full = np.full(mask.shape, np.nan, dtype=np.float32)
    for k, vi in enumerate(idx_s):
        i, j = vi // W20, vi % W20
        p2_full[i, j] = p2_vals[k]

    p2_tiles[tile] = {'p2': p2_full, 'mask': mask, 'p2_vals': p2_vals}

p2_pooled = np.concatenate([p2_tiles[t]['p2_vals'] for t in ['T37TDN','T37TCN']])
print(f"  p₂ pooled: mean={np.mean(p2_pooled):.4f}, std={np.std(p2_pooled):.4f}")
p2_main = p2_tiles['T37TDN']['p2']  # 主tile
s2_main = np.full_like(p2_main, np.nan)
s2_main[p2_tiles['T37TDN']['mask']] = 0.08

# ============================================================
# 传感器 3: Sentinel-1 SAR 幅度
# ============================================================
print("\n[3/4] Sentinel-1 SAR GRD (VV极化)")

def find_s1_tiffs(s1_dir):
    for root, dirs, files in os.walk(s1_dir):
        tiffs = sorted([f for f in files if f.endswith('.tiff')])
        if tiffs:
            return [os.path.join(root, f) for f in tiffs]
    return []

pre_tiffs = find_s1_tiffs(BASE / 's1_pre')
post_tiffs = find_s1_tiffs(BASE / 's1_post')

if pre_tiffs and post_tiffs:
    vv_pre = [f for f in pre_tiffs if '-002.' in f or 'vv' in f.lower()][0]
    vv_post = [f for f in post_tiffs if '-002.' in f or 'vv' in f.lower()][0]
    print(f"  VV pre: {os.path.basename(vv_pre)}")
    print(f"  VV post: {os.path.basename(vv_post)}")

    import rasterio
    with rasterio.open(vv_pre) as src:
        sar_pre = src.read(1).astype(np.float32)
    with rasterio.open(vv_post) as src:
        sar_post = src.read(1).astype(np.float32)

    # Crop to common dimensions
    H_pre, W_pre = sar_pre.shape
    H_post, W_post = sar_post.shape
    Hc = min(H_pre, H_post); Wc = min(W_pre, W_post)
    sar_pre = sar_pre[:Hc, :Wc]
    sar_post = sar_post[:Hc, :Wc]

    # dB conversion
    sar_pre_db = 10 * np.log10(np.maximum(sar_pre, 0.001))
    sar_post_db = 10 * np.log10(np.maximum(sar_post, 0.001))
    dSAR = sar_post_db - sar_pre_db
    print(f"  ΔSAR(dB): mean={np.mean(dSAR):.3f}, std={np.std(dSAR):.3f}")

    # Subsampled GMM
    stride = 8; dSAR_sub = dSAR[::stride, ::stride]
    Hsub, Wsub = dSAR_sub.shape
    v_sar = np.isfinite(dSAR_sub)
    dSAR_vals = dSAR_sub[v_sar].reshape(-1,1)
    print(f"    SAR subsample: {dSAR_sub.shape}, valid={len(dSAR_vals)}")

    if len(dSAR_vals) > 50000:
        idx_sar = RNG.choice(len(dSAR_vals), 50000, replace=False)
        X_sar = dSAR_vals[idx_sar]
    else:
        X_sar = dSAR_vals

    gmm_sar = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(X_sar)
    dm_sar = np.argmax(np.abs(gmm_sar.means_.flatten()))  # 极端变化=损害
    probs_all = gmm_sar.predict_proba(dSAR_vals)[:, dm_sar].astype(np.float32)

    p3_full = np.full(dSAR.shape, np.nan, dtype=np.float32)
    # Fill only the subsampled grid positions
    p3_sub = np.full((Hsub, Wsub), np.nan, dtype=np.float32)
    p3_sub[v_sar] = probs_all
    p3_full[::stride, ::stride] = p3_sub
    print(f"    GMM SAR: 正常={gmm_sar.means_.flatten()[1-dm_sar]:.2f}, 损害={gmm_sar.means_.flatten()[dm_sar]:.2f}, 损害权重={gmm_sar.weights_[dm_sar]:.1%}")
    print(f"  p₃: mean={np.nanmean(p3_full):.4f}")
    s3_full = np.full_like(p3_full, np.nan)
    s3_full[~np.isnan(p3_full)] = 0.12
else:
    print("  ⚠ SAR tiff 未找到，使用占位数据")
    p3_full = np.full((100,150), np.nan, dtype=np.float32)
    s3_full = np.full((100,150), np.nan, dtype=np.float32)

# ============================================================
# 传感器 4: InSAR 相干性
# ============================================================
print("\n[4/4] InSAR 相干性 (Interferometric Coherence)")

import rasterio
for label, fp in [('战前(Dec-Feb)', BASE/'干涉/20211230_20220204.geo.cc.tif'),
                   ('战后(Feb-Mar)', BASE/'干涉/20220228_20220312.geo.cc.tif')]:
    with rasterio.open(fp) as src:
        data = src.read(1).astype(np.float32) / 255.0
    print(f"  {label}: mean coherence={np.nanmean(data):.4f}")

with rasterio.open(BASE/'干涉/20211230_20220204.geo.cc.tif') as src:
    coh_pre = src.read(1).astype(np.float32) / 255.0
    insar_shape = src.shape
with rasterio.open(BASE/'干涉/20220228_20220312.geo.cc.tif') as src:
    coh_post = src.read(1).astype(np.float32) / 255.0

dCoh = coh_post - coh_pre  # <0 = decorrelation = damage
print(f"  ΔCoherence: mean={np.nanmean(dCoh):.4f}, {np.sum(dCoh<0)}/{dCoh.size} 像素下降")

# Damage prob from coherence drop
dCoh_clip = np.clip(dCoh, -0.5, 0.5)
p4 = np.clip(-dCoh_clip / 0.5 * 0.5 + 0.5, 0, 1).astype(np.float32)
p4[(coh_pre < 0.15) & (coh_post < 0.15)] = np.nan  # always low coherence
s4 = np.full_like(p4, np.nan)
valid4 = ~np.isnan(p4)
s4[valid4] = 0.10
print(f"  p₄: mean={np.nanmean(p4):.4f}")

# ============================================================
# 空间对齐到统一网格
# ============================================================
print("\n" + "=" * 65)
print("空间对齐 → 200m 统一网格")
print("=" * 65)

def coarsen(data, Ht, Wt):
    if data is None or np.all(np.isnan(data)):
        return np.full((Ht, Wt), np.nan, dtype=np.float32)
    H, W = data.shape
    bh, bw = max(1, H//Ht), max(1, W//Wt)
    out = np.full((Ht, Wt), np.nan, dtype=np.float32)
    for i in range(Ht):
        for j in range(Wt):
            block = data[i*bh:min((i+1)*bh,H), j*bw:min((j+1)*bw,W)]
            v = block[~np.isnan(block)]
            if len(v) > 0: out[i,j] = np.mean(v)
    return out

UH, UW = 100, 150
P = {}; S = {}
P['VIIRS'], S['VIIRS'] = coarsen(p1, UH, UW), coarsen(s1, UH, UW)
P['S2'], S['S2']     = coarsen(p2_main, UH, UW), coarsen(s2_main, UH, UW)
P['SAR'], S['SAR']   = coarsen(p3_full, UH, UW), coarsen(s3_full, UH, UW)
P['InSAR'], S['InSAR'] = coarsen(p4, UH, UW), coarsen(s4, UH, UW)

names = ['VIIRS','S2','SAR','InSAR']
for n in names:
    v = ~np.isnan(P[n])
    print(f"  {n:6s}: mean={np.nanmean(P[n]):.4f}, valid={np.sum(v)}/{UH*UW}")

# ============================================================
# 贝叶斯层次融合
# ============================================================
print("\n" + "=" * 65)
print("四传感器贝叶斯层次融合 (Beta-Binomial 共轭)")
print("=" * 65)

# 四传感器均有效的像素
valid_all = ~np.isnan(P['VIIRS'])
for n in names[1:]:
    valid_all &= ~np.isnan(P[n])
n_fuse = np.sum(valid_all)
print(f"四传感器均有效: {n_fuse}/{UH*UW} 像素")

if n_fuse > 10:
    p_arr = np.column_stack([P[n][valid_all] for n in names])
    s_arr = np.column_stack([S[n][valid_all] for n in names])

    # 传感器相关性
    print("\n传感器相关性矩阵:")
    print(f"        {'VIIRS':>8} {'S2':>8} {'SAR':>8} {'InSAR':>8}")
    for i, n1 in enumerate(names):
        row_str = f"  {n1:6s}"
        for j, n2 in enumerate(names):
            row_str += f" {np.corrcoef(p_arr[:,i], p_arr[:,j])[0,1]:8.4f}"
        print(row_str)

    # Beta-Binomial conjugate
    eps = 1e-6; p_c = np.clip(p_arr, eps, 1-eps)
    a0, b0 = 1.0, 1.0

    # κ: precision from 1/σ²
    kappa = np.clip(1.0 / (s_arr.mean(axis=0)**2 + 0.001), 1, 20)
    print(f"\n传感器精度 κ: {dict(zip(names, kappa.round(2)))}")

    a_post = a0 + sum(kappa[k] * p_c[:,k] for k in range(4))
    b_post = b0 + sum(kappa[k] * (1 - p_c[:,k]) for k in range(4))

    p_fused = a_post / (a_post + b_post)
    std_f = np.sqrt(a_post * b_post / ((a_post + b_post)**2 * (a_post + b_post + 1)))
    hdi_l = np.array([stats.beta.ppf(0.025, a, b) if (a>0 and b>0) else 0 for a,b in zip(a_post, b_post)])
    hdi_h = np.array([stats.beta.ppf(0.975, a, b) if (a>0 and b>0) else 1 for a,b in zip(a_post, b_post)])

    print(f"\n贝叶斯融合结果:")
    print(f"  p_fused: mean={np.mean(p_fused):.4f}, std={np.std(p_fused):.4f}")
    print(f"  σ_fused: mean={np.mean(std_f):.4f}")
    print(f"  95% HDI 宽度: mean={np.mean(hdi_h-hdi_l):.4f}")

    # 单传感器 vs 融合
    print(f"\n  {'传感器':12s} {'输入均值':>10s} {'与融合的相关性':>16s}")
    for k in range(4):
        corr = np.corrcoef(p_c[:,k], p_fused)[0,1]
        print(f"  {names[k]:12s} {np.mean(p_c[:,k]):10.4f} {corr:16.4f}")

    # Save
    fused_map = np.full((UH, UW), np.nan, dtype=np.float32)
    std_map = np.full((UH, UW), np.nan, dtype=np.float32)
    hdi_l_map = np.full((UH, UW), np.nan, dtype=np.float32)
    hdi_h_map = np.full((UH, UW), np.nan, dtype=np.float32)

    vi_all = np.where(valid_all.ravel())[0]
    for k, vi in enumerate(vi_all):
        i, j = vi // UW, vi % UW
        fused_map[i,j] = p_fused[k]
        std_map[i,j] = std_f[k]
        hdi_l_map[i,j] = hdi_l[k]
        hdi_h_map[i,j] = hdi_h[k]

    np.save(OUT / 'exp_v2_bayes_fused_mean.npy', fused_map)
    np.save(OUT / 'exp_v2_bayes_fused_std.npy', std_map)
    np.save(OUT / 'exp_v2_bayes_fused_hdi_low.npy', hdi_l_map)
    np.save(OUT / 'exp_v2_bayes_fused_hdi_high.npy', hdi_h_map)

    # 单传感器 maps
    for n in names:
        np.save(OUT / f'exp_v2_p_{n}.npy', P[n])
        np.save(OUT / f'exp_v2_s_{n}.npy', S[n])

    # ============================================================
    # 基线对比
    # ============================================================
    print(f"\n{'='*65}")
    print("基线方法对比")
    print("="*65)

    vote_simple = np.mean(p_c, axis=1)
    prec = 1.0/(s_arr**2 + 0.001)
    vote_w = np.sum(p_c * prec, axis=1) / np.sum(prec, axis=1)
    vote_median = np.median(p_c, axis=1)

    print(f"  简单平均:     {np.mean(vote_simple):.4f}")
    print(f"  精度加权:     {np.mean(vote_w):.4f}")
    print(f"  中位数投票:   {np.mean(vote_median):.4f}")
    print(f"  贝叶斯融合:   {np.mean(p_fused):.4f}")

    # KLD 信息增益
    from scipy.special import betaln, digamma
    def kl_beta(a1,b1,a2,b2):
        return (betaln(a2,b2)-betaln(a1,b1) + (a1-a2)*digamma(a1)
                + (b1-b2)*digamma(b1) + (a2-a1+b2-b1)*digamma(a1+b1))

    print(f"\n  KLD 信息增益 (先验 → 后验):");
    for k in range(4):
        gain = np.mean([kl_beta(a0+kappa[k]*p_c[i,k], b0+kappa[k]*(1-p_c[i,k]), a0, b0)
                         for i in range(min(1000, n_fuse))])
        print(f"    {names[k]:6s}: {gain:.4f}")

    # ============================================================
    # Summary
    # ============================================================
    summary = {
        'experiment': 'v2',
        'n_sensors': 4,
        'sensors': names,
        'p_input_means': [float(np.nanmean(P[n])) for n in names],
        'correlation_matrix': [[float(np.corrcoef(p_arr[:,i], p_arr[:,j])[0,1])
                                for j in range(4)] for i in range(4)],
        'fusion_mean': float(np.mean(p_fused)),
        'fusion_std': float(np.mean(std_f)),
        'hdi_width': float(np.mean(hdi_h - hdi_l)),
        'n_pixels': int(n_fuse),
        'kappa': [float(x) for x in kappa],
    }
    with open(OUT / 'exp_v2_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*65}")
print("v2 流水线完成")
print(f"输出: {OUT}/")
print("="*65)
