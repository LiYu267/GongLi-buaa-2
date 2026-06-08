"""
实验 v3：四传感器完整改进版
===========================
改进项（vs v2）：
  1. SAR: 直方图匹配归一化 → 消除系统性辐射偏差
  2. InSAR: 时间去相干指数模型归一化 → 修正时间基线差异
  3. 不确定性: 预测级 Bootstrap + 先验校准
  4. D-S 证据理论融合 + 对比
  5. 合成验证升级: 多场景系统性实验

输出: exp_outputs_v3/
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import betaln, digamma
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os, json, warnings
warnings.filterwarnings('ignore')

BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs_v3'
OUT.mkdir(exist_ok=True)
RNG = np.random.RandomState(42)
N_BOOT = 150

print("=" * 65)
print("实验 v3：四传感器改进版 — 马里乌波尔")
print("=" * 65)

# ============================================================
# 传感器 1: VIIRS 夜间灯光
# ============================================================
print("\n[1/4] VIIRS 夜间灯光 (500m)")

import h5py
f_pre = h5py.File(BASE/'VNP46A2.A2022008.h21v04.002.2025114204956.h5','r')
f_post = h5py.File(BASE/'VNP46A2.A2022128.h21v04.002.2025119063131.h5','r')
ntl_pre = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL'][:]
ntl_post = f_post['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_BRDF-Corrected_NTL'][:]
lat_v = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lat'][:]
lon_v = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lon'][:]
qf_pre = f_pre['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Mandatory_Quality_Flag'][:]
qf_post = f_post['HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Mandatory_Quality_Flag'][:]
f_pre.close(); f_post.close()

lat2, lon2 = lat_v[:,None], lon_v[None,:]
aoi = ((lat2>=46.85)&(lat2<=47.86)&(lon2>=37.66)&(lon2<=39.14))
qm = (qf_pre<=2)&(qf_post<=2); fm = aoi & qm
rows = np.any(fm,axis=1); cols = np.any(fm,axis=0)
dNTL = ntl_post[rows,:][:,cols] - ntl_pre[rows,:][:,cols]
qsub = qm[rows,:][:,cols]
dNTL_v = np.where(qsub, dNTL, np.nan)

v1 = ~np.isnan(dNTL_v)
gmm1 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(dNTL_v[v1].reshape(-1,1))
dm = np.argmin(gmm1.means_.flatten())
p1_f = gmm1.predict_proba(dNTL_v[v1].reshape(-1,1))[:,dm]

# Bootstrap σ (预测级: 重采样像素而非模型参数)
n1 = np.sum(v1)
p1_boot = np.zeros((N_BOOT, n1))
for b in range(N_BOOT):
    ib = RNG.choice(n1, n1, replace=True)
    gb = GaussianMixture(n_components=2, random_state=b, n_init=3)
    try:
        gb.fit(dNTL_v[v1].reshape(-1,1)[ib])
        p1_boot[b] = gb.predict_proba(dNTL_v[v1].reshape(-1,1))[:,np.argmin(gb.means_.flatten())]
    except: p1_boot[b] = p1_f

p1 = np.full_like(dNTL_v, np.nan, dtype=np.float32); p1[v1] = p1_f.astype(np.float32)
s1 = np.full_like(dNTL_v, np.nan, dtype=np.float32); s1[v1] = np.std(p1_boot, axis=0).astype(np.float32)
# 校准: 避免σ过小
s1_min = 0.02; s1[s1 < s1_min] = s1_min
print(f"  p₁: {np.nanmean(p1):.4f}, σ₁: {np.nanmean(s1[v1]):.4f} (min_clamped={s1_min})")

# ============================================================
# 传感器 2: Sentinel-2 多光谱 (May→May)
# ============================================================
print("\n[2/4] Sentinel-2 (May 2021 → May 2022)")

def find_s2_jp2(d, pat):
    for r,_,fs in os.walk(d):
        for f in fs:
            if pat in f and f.endswith('.jp2') and 'MSK_' not in f and 'QI_DATA' not in r:
                return os.path.join(r,f)
    return None

def load_s2_band(d, pat):
    import rasterio
    with rasterio.open(find_s2_jp2(d,pat)) as src:
        return src.read(1).astype(np.float32)

def s2_delta(pre_d, post_d, idx):
    if idx=='NDVI':
        pre=(load_s2_band(pre_d,'B08_10m')-load_s2_band(pre_d,'B04_10m'))/(load_s2_band(pre_d,'B08_10m')+load_s2_band(pre_d,'B04_10m')+1e-10)
        post=(load_s2_band(post_d,'B08_10m')-load_s2_band(post_d,'B04_10m'))/(load_s2_band(post_d,'B08_10m')+load_s2_band(post_d,'B04_10m')+1e-10)
    elif idx=='NBR':
        pre=(load_s2_band(pre_d,'B8A_20m')-load_s2_band(pre_d,'B12_20m'))/(load_s2_band(pre_d,'B8A_20m')+load_s2_band(pre_d,'B12_20m')+1e-10)
        post=(load_s2_band(post_d,'B8A_20m')-load_s2_band(post_d,'B12_20m'))/(load_s2_band(post_d,'B8A_20m')+load_s2_band(post_d,'B12_20m')+1e-10)
    elif idx=='NDBI':
        pre=(load_s2_band(pre_d,'B11_20m')-load_s2_band(pre_d,'B8A_20m'))/(load_s2_band(pre_d,'B11_20m')+load_s2_band(pre_d,'B8A_20m')+1e-10)
        post=(load_s2_band(post_d,'B11_20m')-load_s2_band(post_d,'B8A_20m'))/(load_s2_band(post_d,'B11_20m')+load_s2_band(post_d,'B8A_20m')+1e-10)
    return post-pre

pre_s2 = {'T37TDN': str(BASE/'s2_may2021/T37TDN'), 'T37TCN': str(BASE/'s2_may2021/T37TCN')}
post_s2 = {'T37TDN': str(BASE/'战后2'), 'T37TCN': str(BASE/'战后1')}

p2_vals_all = []
p2_tiles = {}
for tile in ['T37TDN','T37TCN']:
    pre_d, post_d = pre_s2[tile], post_s2[tile]
    dNDVI = s2_delta(pre_d, post_d, 'NDVI')
    dNBR  = s2_delta(pre_d, post_d, 'NBR')
    dNDBI = s2_delta(pre_d, post_d, 'NDBI')
    print(f"  {tile}: ΔNDVI={np.nanmean(dNDVI):.4f}, ΔNBR={np.nanmean(dNBR):.4f}, ΔNDBI={np.nanmean(dNDBI):.4f}")

    scl_pre = load_s2_band(pre_d,'SCL_20m'); scl_post = load_s2_band(post_d,'SCL_20m')
    mask = (scl_pre>=2)&(scl_pre<=7)&(scl_post>=2)&(scl_post<=7)
    H20, W20 = mask.shape
    dNDVI_20 = dNDVI[:H20*2:2,:W20*2:2]
    vidx = np.where(mask.ravel())[0]
    ns = min(50000, len(vidx)); idx_s = RNG.choice(vidx, ns, replace=False)

    X = np.column_stack([dNDVI_20.ravel()[idx_s], dNBR.ravel()[idx_s], dNDBI.ravel()[idx_s]])
    X_s = StandardScaler().fit_transform(X)
    pc1 = PCA(n_components=3).fit_transform(X_s)[:,0]
    print(f"    PCA PC1 var={PCA(n_components=3).fit(X_s).explained_variance_ratio_[0]:.1%}")

    pc1_abs = np.abs(pc1)
    gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(pc1_abs.reshape(-1,1))
    dm2 = np.argmax(gmm2.means_.flatten())
    p2_vals = gmm2.predict_proba(pc1_abs.reshape(-1,1))[:,dm2]

    # Bootstrap
    p2_boot = np.zeros((min(N_BOOT,50), ns))
    for b in range(min(N_BOOT,50)):
        ib = RNG.choice(ns, ns, replace=True)
        gb = GaussianMixture(n_components=2, random_state=b, n_init=3)
        try:
            gb.fit(pc1_abs[ib].reshape(-1,1))
            p2_boot[b] = gb.predict_proba(pc1_abs.reshape(-1,1))[:,np.argmax(gb.means_.flatten())]
        except: p2_boot[b] = p2_vals
    s2_vals = np.std(p2_boot, axis=0)
    s2_vals = np.clip(s2_vals, 0.03, None)

    p2_vals_all.append(p2_vals)
    p2_tiles[tile] = {'p2': p2_vals, 's2': s2_vals, 'mask': mask, 'idx': idx_s}

p2_pooled = np.concatenate(p2_vals_all)
s2_pooled = np.concatenate([p2_tiles[t]['s2'] for t in ['T37TDN','T37TCN']])
print(f"  p₂ pooled: {np.mean(p2_pooled):.4f}, σ₂: {np.mean(s2_pooled):.4f}")
# Use main tile p2 for map
p2_map = p2_tiles['T37TDN']['p2']
s2_map = p2_tiles['T37TDN']['s2']

# ============================================================
# 传感器 3: Sentinel-1 SAR (直方图匹配归一化)
# ============================================================
print("\n[3/4] Sentinel-1 SAR (直方图匹配归一化)")

def find_s1_tiffs(d):
    for r,_,fs in os.walk(d):
        tiffs = sorted([f for f in fs if f.endswith('.tiff')])
        if tiffs: return [os.path.join(r,f) for f in tiffs]
    return []

pre_tiffs = find_s1_tiffs(BASE/'s1_pre')
post_tiffs = find_s1_tiffs(BASE/'s1_post')

if pre_tiffs and post_tiffs:
    vv_pre = [f for f in pre_tiffs if '-002.' in f or 'vv' in f.lower()][0]
    vv_post = [f for f in post_tiffs if '-002.' in f or 'vv' in f.lower()][0]

    import rasterio
    with rasterio.open(vv_pre) as src:
        sar_pre = src.read(1).astype(np.float32)
    with rasterio.open(vv_post) as src:
        sar_post = src.read(1).astype(np.float32)

    # 裁剪到共同尺寸
    Hc = min(sar_pre.shape[0], sar_post.shape[0])
    Wc = min(sar_pre.shape[1], sar_post.shape[1])
    sar_pre, sar_post = sar_pre[:Hc,:Wc], sar_post[:Hc,:Wc]

    # === 改进1: 直方图匹配归一化 ===
    # 将post SAR的直方图映射到pre SAR的分布
    # 这消除了系统性辐射偏差（轨道、季节差异），保留局部变化信号
    pre_flat = sar_pre.ravel()
    post_flat = sar_post.ravel()

    # 对有效像素做histogram matching
    v_pre = pre_flat > 0
    v_post = post_flat > 0

    # 计算CDF
    pre_sorted = np.sort(pre_flat[v_pre])
    post_sorted = np.sort(post_flat[v_post])

    # 映射函数: post值 → pre分布中相同百分位的值
    from scipy.interpolate import interp1d
    post_cdf = np.linspace(0, 1, len(post_sorted))
    pre_cdf = np.linspace(0, 1, len(pre_sorted))

    # 插值: post_value → pre_value_at_same_percentile
    mapper = interp1d(post_cdf, post_sorted, bounds_error=False, fill_value='extrapolate')
    # 反过来: 给定post值, 找它在post分布中的百分位, 再映射到pre中的对应值
    # 简化做法: 用rank-based matching
    post_matched = np.zeros_like(post_flat)

    # 快速直方图匹配: subsample→分位数映射→全图应用
    # Step 1: 子采样估计映射函数
    n_sample = 200000
    pre_samp = RNG.choice(pre_flat[v_pre], min(n_sample, np.sum(v_pre)), replace=False)
    post_samp = RNG.choice(post_flat[v_post], min(n_sample, np.sum(v_post)), replace=False)

    # Step 2: 在100个分位数点建立映射
    pcts = np.linspace(0, 100, 101)
    pre_pct_vals = np.percentile(pre_samp, pcts)
    post_pct_vals = np.percentile(post_samp, pcts)

    # Step 3: 对全图post值做插值映射
    # post_value → 它在post分位数中的位置 → 对应pre分位数的值
    sar_post_norm = np.zeros_like(sar_post)
    vp = sar_post > 0
    # 使用np.interp: 给定post值, 先映射到[0,100], 再映射到pre值
    post_to_pct = np.interp(sar_post[vp], post_pct_vals, pcts)
    sar_post_norm[vp] = np.interp(post_to_pct, pcts, pre_pct_vals)
    del post_to_pct, pre_samp, post_samp

    # 现在两幅图在相同的辐射标度上
    sar_pre_db = 10 * np.log10(np.maximum(sar_pre, 0.001))
    sar_post_norm_db = 10 * np.log10(np.maximum(sar_post_norm, 0.001))
    dSAR = sar_post_norm_db - sar_pre_db

    print(f"  ΔSAR(dB) 归一化后: mean={np.mean(dSAR):.4f}, std={np.std(dSAR):.4f}")
    print(f"  (v2 中为 mean=-1.61, 系统性偏差已消除)")

    # GMM on matched difference
    stride = 8
    dSAR_sub = dSAR[::stride,::stride]
    v_sar = np.isfinite(dSAR_sub)
    X_sar = dSAR_sub[v_sar].reshape(-1,1)
    if len(X_sar) > 50000:
        X_sar = X_sar[RNG.choice(len(X_sar), 50000, replace=False)]

    gmm_sar = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(X_sar)
    dm_sar = np.argmax(np.abs(gmm_sar.means_.flatten()))
    p3_vals = gmm_sar.predict_proba(dSAR_sub[v_sar].reshape(-1,1))[:,dm_sar].astype(np.float32)

    # Bootstrap
    p3_boot = np.zeros((min(N_BOOT,50), len(p3_vals)))
    for b in range(min(N_BOOT,50)):
        ib_s = RNG.choice(len(p3_vals), len(p3_vals), replace=True)
        gb = GaussianMixture(n_components=2, random_state=b, n_init=3)
        try:
            gb.fit(dSAR_sub[v_sar].reshape(-1,1)[ib_s])
            p3_boot[b] = gb.predict_proba(dSAR_sub[v_sar].reshape(-1,1))[:,np.argmax(np.abs(gb.means_.flatten()))]
        except: p3_boot[b] = p3_vals
    s3_vals = np.clip(np.std(p3_boot, axis=0), 0.03, None)

    # 构建全图
    p3_full = np.full(dSAR.shape, np.nan, dtype=np.float32)
    p3_sub = np.full(dSAR_sub.shape, np.nan, dtype=np.float32)
    p3_sub[v_sar] = p3_vals
    p3_full[::stride,::stride] = p3_sub

    s3_full = np.full(dSAR.shape, np.nan, dtype=np.float32)
    s3_sub = np.full(dSAR_sub.shape, np.nan, dtype=np.float32)
    s3_sub[v_sar] = s3_vals
    s3_full[::stride,::stride] = s3_sub

    print(f"  GMM SAR: 正常={gmm_sar.means_.flatten()[1-dm_sar]:.2f}dB, 损害={gmm_sar.means_.flatten()[dm_sar]:.2f}dB, 损害权重={gmm_sar.weights_[dm_sar]:.1%}")
    print(f"  p₃: {np.nanmean(p3_full):.4f}, σ₃: {np.nanmean(s3_vals):.4f}")
else:
    print("  ⚠ SAR 数据不可用")
    p3_full = np.full((100,150), np.nan, dtype=np.float32)
    s3_full = np.full((100,150), np.nan, dtype=np.float32)

# ============================================================
# 传感器 4: InSAR 相干性 (时间基线归一化)
# ============================================================
print("\n[4/4] InSAR 相干性 (指数去相干模型归一化)")

import rasterio
with rasterio.open(BASE/'干涉/20211230_20220204.geo.cc.tif') as src:
    coh_pre_raw = src.read(1).astype(np.float32)/255.0
with rasterio.open(BASE/'干涉/20220228_20220312.geo.cc.tif') as src:
    coh_post_raw = src.read(1).astype(np.float32)/255.0

# === 改进2: 时间去相干指数模型归一化 ===
# γ(Δt) = γ₀ · exp(-Δt/τ)
# 目标: 将两个干涉对归一化到相同的时间基线 Δt_ref
#
# 时间基线:
Δt_pre = 36   # days: 2021-12-30 → 2022-02-04
Δt_post = 12  # days: 2022-02-28 → 2022-03-12

# 时间去相干常数: C波段在城市/农田的典型值
# 文献: τ ≈ 30-80 days for C-band in urban areas
# Mariupol混合地表: 取 τ=50 days
tau = 50.0

print(f"  Δt_pre={Δt_pre}d, Δt_post={Δt_post}d, τ={tau:.0f}d (文献值)")
print(f"  战前原始相干性: mean={np.nanmean(coh_pre_raw):.4f}")
print(f"  战后原始相干性: mean={np.nanmean(coh_post_raw):.4f}")

# 归一化到 Δt=0 (零基线):
# γ(0) = γ(Δt) · exp(Δt/τ)
# 限制相干性不超过1.0
coh_pre_norm = np.minimum(coh_pre_raw * np.exp(Δt_pre/tau), 1.0)
coh_post_norm = np.minimum(coh_post_raw * np.exp(Δt_post/tau), 1.0)

print(f"  战前归一化相干性 (→Δt=0): mean={np.nanmean(coh_pre_norm):.4f}")
print(f"  战后归一化相干性 (→Δt=0): mean={np.nanmean(coh_post_norm):.4f}")

# 归一化后的差值
dCoh_norm = coh_post_norm - coh_pre_norm
print(f"  ΔCoherence_norm: mean={np.nanmean(dCoh_norm):.4f}, {np.sum(dCoh_norm<0)}/{dCoh_norm.size} 像素下降")

# 损害概率: 相干性下降 → 损害
# 使用 sigmoid 映射: p = 1/(1+exp(k*dCoh))
# k 控制灵敏度, k越大越陡
k_slope = 10.0  # 使 dCoh=-0.2 → p≈0.88
p4 = (1.0 / (1.0 + np.exp(k_slope * dCoh_norm))).astype(np.float32)

# 排除永远低相干的区域
low_coh = (coh_pre_raw < 0.10) & (coh_post_raw < 0.10)
p4[low_coh] = np.nan
valid4 = ~np.isnan(p4)

# σ₄: 基于相干性本身的信噪比
s4 = np.full_like(p4, np.nan)
s4[valid4] = np.clip(0.05 + 0.15 * (1.0 - coh_pre_norm[valid4]), 0.03, 0.3)

print(f"  p₄: mean={np.nanmean(p4):.4f}, σ₄: mean={np.nanmean(s4[valid4]):.4f}")
print(f"  (v2 中 p₄≈0.22，现在基于归一化相干性)")

# ============================================================
# 空间对齐
# ============================================================
print("\n" + "=" * 65)
print("空间对齐 → 200m 统一网格")
print("=" * 65)

def coarsen(data, Ht, Wt):
    if data is None or np.all(np.isnan(data)):
        return np.full((Ht,Wt), np.nan, dtype=np.float32)
    H,W = data.shape
    bh,bw = max(1,H//Ht), max(1,W//Wt)
    out = np.full((Ht,Wt), np.nan, dtype=np.float32)
    for i in range(Ht):
        for j in range(Wt):
            blk = data[i*bh:min((i+1)*bh,H), j*bw:min((j+1)*bw,W)]
            v = blk[~np.isnan(blk)]
            if len(v)>0: out[i,j] = np.mean(v)
    return out

UH, UW = 100, 150
# For S2/SAR/InSAR that are per-pixel maps, coarsen to unified grid
p1_u = coarsen(p1, UH, UW); s1_u = coarsen(s1, UH, UW)

# For S2 sampled results, interpolate to unified grid directly
p2_u = np.full((UH,UW), np.nan, dtype=np.float32)
s2_u = np.full((UH,UW), np.nan, dtype=np.float32)
# Fill with pooled mean values since S2 is sampled
p2_u[...] = np.mean(p2_pooled); s2_u[...] = np.mean(s2_pooled)
# Add spatial structure from main tile
p2_tdn = p2_tiles['T37TDN']
Hmask, Wmask = p2_tdn['mask'].shape
for k, vi in enumerate(p2_tdn['idx'][:5000]):
    i_map = int((vi//Wmask)/Hmask * UH); j_map = int((vi%Wmask)/Wmask * UW)
    if 0<=i_map<UH and 0<=j_map<UW:
        p2_u[i_map,j_map] = p2_tdn['p2'][k] if not np.isnan(p2_u[i_map,j_map]) or k==0 else np.nanmean([p2_u[i_map,j_map], p2_tdn['p2'][k]])
        s2_u[i_map,j_map] = p2_tdn['s2'][k]

p3_u = coarsen(p3_full, UH, UW); s3_u = coarsen(s3_full, UH, UW)
p4_u = coarsen(p4, UH, UW); s4_u = coarsen(s4, UH, UW)

names = ['VIIRS','S2','SAR','InSAR']
P = {'VIIRS':p1_u, 'S2':p2_u, 'SAR':p3_u, 'InSAR':p4_u}
S = {'VIIRS':s1_u, 'S2':s2_u, 'SAR':s3_u, 'InSAR':s4_u}

for n in names:
    v = ~np.isnan(P[n])
    print(f"  {n:6s}: mean={np.nanmean(P[n]):.4f}, σ={np.nanmean(S[n][v]):.4f}, valid={np.sum(v)}/{UH*UW}")

# Define KL divergence for Beta distributions
def kl_beta(a1,b1,a2,b2):
    return (betaln(a2,b2)-betaln(a1,b1)+(a1-a2)*digamma(a1)+(b1-b2)*digamma(b1)+(a2-a1+b2-b1)*digamma(a1+b1))

# ============================================================
# 贝叶斯层次融合
# ============================================================
print("\n" + "=" * 65)
print("贝叶斯层次融合 (Beta-Binomial 共轭)")
print("=" * 65)

valid_all = ~np.isnan(P['VIIRS'])
for n in names[1:]: valid_all &= ~np.isnan(P[n])
n_fuse = np.sum(valid_all)
print(f"四传感器均有效: {n_fuse}/{UH*UW}")

if n_fuse > 10:
    p_arr = np.column_stack([P[n][valid_all] for n in names])
    s_arr = np.column_stack([S[n][valid_all] for n in names])
    eps = 1e-6; p_c = np.clip(p_arr, eps, 1-eps)

    # 传感器相关性
    print("\n传感器相关性:")
    print(f"        {'VIIRS':>8} {'S2':>8} {'SAR':>8} {'InSAR':>8}")
    for i,n1 in enumerate(names):
        row = f"  {n1:6s}"
        for j in range(4): row += f" {np.corrcoef(p_c[:,i],p_c[:,j])[0,1]:8.4f}"
        print(row)

    # κ: 精度加权 (考虑σ估计的不确定性, 加正则化)
    sigma_mean = np.clip(s_arr.mean(axis=0), 0.02, 1.0)
    kappa = np.clip(1.0/(sigma_mean**2 + 0.005), 1.0, 30.0)
    print(f"\nσ_mean: {dict(zip(names, sigma_mean.round(4)))}")
    print(f"κ: {dict(zip(names, kappa.round(2)))}")

    a0,b0 = 1.0, 1.0
    a_post = a0 + sum(kappa[k]*p_c[:,k] for k in range(4))
    b_post = b0 + sum(kappa[k]*(1-p_c[:,k]) for k in range(4))
    p_fused = a_post/(a_post+b_post)
    std_f = np.sqrt(a_post*b_post/((a_post+b_post)**2*(a_post+b_post+1)))
    hdi_l = np.array([stats.beta.ppf(0.025,a,b) if (a>0 and b>0) else 0 for a,b in zip(a_post,b_post)])
    hdi_h = np.array([stats.beta.ppf(0.975,a,b) if (a>0 and b>0) else 1 for a,b in zip(a_post,b_post)])

    print(f"\n贝叶斯融合:")
    print(f"  p_fused: {np.mean(p_fused):.4f}±{np.std(p_fused):.4f}")
    print(f"  σ_fused: {np.mean(std_f):.4f}")
    print(f"  95%HDI宽度: {np.mean(hdi_h-hdi_l):.4f}")
    for k in range(4):
        print(f"  {names[k]:6s}→fusion corr={np.corrcoef(p_c[:,k],p_fused)[0,1]:.4f}, KLD={np.mean([kl_beta(a0+kappa[k]*p_c[i,k],b0+kappa[k]*(1-p_c[i,k]),a0,b0) for i in range(min(1000,n_fuse))]):.4f}")

    # Save
    for label, arr in [('fused_mean',p_fused),('fused_std',std_f),('hdi_low',hdi_l),('hdi_high',hdi_h)]:
        m = np.full((UH,UW),np.nan,dtype=np.float32)
        for k,vi in enumerate(np.where(valid_all.ravel())[0]):
            i,j = vi//UW, vi%UW; m[i,j]=arr[k]
        np.save(OUT/f'exp_v3_bayes_{label}.npy', m)

# ============================================================
# D-S 证据理论融合
# ============================================================
print("\n" + "=" * 65)
print("Dempster-Shafer 证据理论融合")
print("=" * 65)

if n_fuse > 10:
    # 构建 BPA
    # σ_normalized → m(Θ): 归一化后的"无知度"
    sigma_norm = np.clip(s_arr/0.3, 0.001, 0.99)  # 校准: σ=0.3 → 完全无知
    m_D = p_c*(1-sigma_norm)       # 支持损害
    m_U = (1-p_c)*(1-sigma_norm)   # 支持未损害
    m_Theta = sigma_norm            # 无知

    # D-S 组合 (4个源按序组合)
    K_total = np.zeros(n_fuse)
    m12_D = np.zeros(n_fuse); m12_U = np.zeros(n_fuse); m12_T = np.ones(n_fuse)

    for k in range(4):
        # m_combined ⊕ m_k
        K = m12_D*m_U[:,k] + m12_U*m_D[:,k]
        one_minus_K = 1.0 - np.clip(K, 0, 0.9999)
        new_D = (m12_D*m_D[:,k] + m12_D*m_Theta[:,k] + m12_T*m_D[:,k])/one_minus_K
        new_U = (m12_U*m_U[:,k] + m12_U*m_Theta[:,k] + m12_T*m_U[:,k])/one_minus_K
        new_T = (m12_T*m_Theta[:,k])/one_minus_K
        m12_D, m12_U, m12_T = new_D, new_U, new_T
        K_total += K

    bel = m12_D; pl = m12_D + m12_T
    ignorance = pl - bel

    print(f"  D-S Bel(D): {np.mean(bel):.4f}")
    print(f"  D-S Pl(D):  {np.mean(pl):.4f}")
    print(f"  Ignorance (Pl-Bel): {np.mean(ignorance):.4f}")
    print(f"  累计冲突 K: {np.mean(K_total):.4f}")

    # Save D-S
    for label, arr in [('ds_bel',bel),('ds_pl',pl),('ds_ig',ignorance)]:
        m = np.full((UH,UW),np.nan,dtype=np.float32)
        for k,vi in enumerate(np.where(valid_all.ravel())[0]):
            i,j = vi//UW, vi%UW; m[i,j]=arr[k]
        np.save(OUT/f'exp_v3_{label}.npy', m)

# ============================================================
# 投票基线
# ============================================================
print("\n" + "=" * 65)
print("方法对比")
print("=" * 65)

if n_fuse > 10:
    vote_s = np.mean(p_c, axis=1)
    prec = 1.0/(s_arr**2+0.005)
    vote_w = np.sum(p_c*prec, axis=1)/np.sum(prec, axis=1)
    vote_m = np.median(p_c, axis=1)

    methods = {
        'VIIRS-only': np.mean(p_c[:,0]),
        'S2-only': np.mean(p_c[:,1]),
        'SAR-only': np.mean(p_c[:,2]),
        'InSAR-only': np.mean(p_c[:,3]),
        'Vote Mean': np.mean(vote_s),
        'Vote Weighted': np.mean(vote_w),
        'Vote Median': np.mean(vote_m),
        'D-S Bel': np.mean(bel),
        'D-S Pl': np.mean(pl),
        'Bayesian': np.mean(p_fused),
    }
    for name, val in methods.items():
        print(f"  {name:15s}: {val:.4f}")

# ============================================================
# 合成验证升级
# ============================================================
print("\n" + "=" * 65)
print("合成验证: 多场景系统性实验")
print("=" * 65)

if n_fuse > 10:
    N_test = 200
    rng_syn = np.random.RandomState(123)
    results_syn = []

    # 基础数据（取未损害像素作为背景）
    bg_mask = p_c[:,1] < 0.4  # S2低损害 = 背景
    if np.sum(bg_mask) < N_test*3:
        bg_mask = np.ones(n_fuse, dtype=bool)  # fallback

    for scenario_name, p1_strength, p2_strength, p3_strength, p4_strength in [
        ('A:VIIRS-dom', (0.75,0.95), (0.35,0.55), (0.45,0.65), (0.45,0.65)),
        ('B:S2-dom',    (0.35,0.55), (0.75,0.95), (0.45,0.65), (0.45,0.65)),
        ('C:SAR-dom',   (0.35,0.55), (0.35,0.55), (0.75,0.95), (0.45,0.65)),
        ('D:All-strong',(0.70,0.90), (0.70,0.90), (0.70,0.90), (0.70,0.90)),
        ('E:Mixed',     (0.50,0.70), (0.50,0.70), (0.50,0.70), (0.50,0.70)),
    ]:
        # 采样背景+注入损害
        pos_idx = rng_syn.choice(np.where(bg_mask)[0], N_test, replace=False)
        neg_idx = rng_syn.choice(np.where(bg_mask)[0], N_test, replace=False)

        p_inj = p_c.copy()
        for k, (lo, hi) in enumerate([p1_strength, p2_strength, p3_strength, p4_strength]):
            p_inj[pos_idx, k] = rng_syn.uniform(lo, hi, N_test)

        labels = np.concatenate([np.ones(N_test), np.zeros(N_test)])
        p_all_test = np.vstack([p_inj[pos_idx], p_inj[neg_idx]])

        # 四种融合方法
        auc_single = [roc_auc_score(labels, p_all_test[:,k]) for k in range(4)]

        # 贝叶斯
        a_test = a0 + sum(kappa[k]*p_all_test[:,k] for k in range(4))
        b_test = b0 + sum(kappa[k]*(1-p_all_test[:,k]) for k in range(4))
        p_bayes = a_test/(a_test+b_test)
        auc_bayes = roc_auc_score(labels, p_bayes)

        # 投票
        auc_vote = roc_auc_score(labels, np.mean(p_all_test, axis=1))

        # D-S (简化)
        s_test = np.clip(s_arr.mean(axis=0)*np.ones((2*N_test,4)), 0.001, 0.99)
        mD = p_all_test*(1-s_test); mU = (1-p_all_test)*(1-s_test); mT = s_test
        m12d, m12u, m12t = np.zeros(2*N_test), np.zeros(2*N_test), np.ones(2*N_test)
        for k in range(4):
            Kk = m12d*mU[:,k] + m12u*mD[:,k]
            omk = 1.0-np.clip(Kk,0,0.9999)
            m12d, m12u, m12t = (m12d*mD[:,k]+m12d*mT[:,k]+m12t*mD[:,k])/omk, (m12u*mU[:,k]+m12u*mT[:,k]+m12t*mU[:,k])/omk, (m12t*mT[:,k])/omk
        auc_ds = roc_auc_score(labels, m12d)

        results_syn.append({
            'scenario': scenario_name,
            'auc_single': [round(x,4) for x in auc_single],
            'auc_bayes': round(auc_bayes,4),
            'auc_vote': round(auc_vote,4),
            'auc_ds': round(auc_ds,4),
        })
        print(f"  {scenario_name:16s}: Best-single={max(auc_single):.3f}, Vote={auc_vote:.3f}, D-S={auc_ds:.3f}, Bayes={auc_bayes:.3f}")

# ============================================================
# Save summary
# ============================================================
summary = {
    'experiment': 'v3',
    'improvements': ['SAR_histogram_matching','InSAR_temporal_normalization','prediction_bootstrap_sigma','DS_fusion','upgraded_synthetic'],
    'n_sensors': 4,
    'sensors': names,
    'p_means': [float(np.nanmean(P[n])) for n in names],
    'fusion_mean': float(np.mean(p_fused)) if n_fuse>10 else None,
    'fusion_std': float(np.mean(std_f)) if n_fuse>10 else None,
    'hdi_width': float(np.mean(hdi_h-hdi_l)) if n_fuse>10 else None,
    'ds_bel': float(np.mean(bel)) if n_fuse>10 else None,
    'ds_pl': float(np.mean(pl)) if n_fuse>10 else None,
    'n_pixels': int(n_fuse),
    'synthetic_results': results_syn,
}
with open(OUT/'exp_v3_summary.json','w',encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*65}")
print("v3 完成 — 输出: exp_outputs_v3/")
print("="*65)
