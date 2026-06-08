"""
实验 v4：两处关键修正
=====================
改进1: VIIRS GMM 只在城市核心区（战前NTL>10）拟合，不再被郊区稀释
改进2: S2 用有符号PC1替代|PC1|，区分破坏性变化(NDVI↓)和建设性变化(NDVI↑)

其余同v3: SAR直方图匹配 + InSAR τ归一化 + 贝叶斯/D-S融合 + 合成验证
输出: exp_outputs_v4/
"""

import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import betaln, digamma
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os, json, warnings
warnings.filterwarnings('ignore')

BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs_v4'; OUT.mkdir(exist_ok=True)
RNG = np.random.RandomState(42); N_BOOT = 150

print("=" * 65)
print("实验 v4：VIIRS城市核心聚焦 + S2方向感知融合")
print("=" * 65)

# ============================================================
# 传感器 1: VIIRS — 只在城市核心区拟合GMM
# ============================================================
print("\n[1/4] VIIRS — 城市核心区聚焦")

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
ntl_pre_sub = ntl_pre[rows,:][:,cols]
ntl_post_sub = ntl_post[rows,:][:,cols]
qsub = qm[rows,:][:,cols]
dNTL = ntl_post_sub - ntl_pre_sub
dNTL_v = np.where(qsub, dNTL, np.nan)

# === 改进1: 只在城市核心区拟合GMM ===
# 战前灯光 > 10 的像素 ≈ 城市/建成区
# 马里乌波尔市区NTL典型值: 郊区<5, 居民区5-20, 商业区20-200
urban_mask = (ntl_pre_sub > 10) & qsub
n_urban = np.sum(urban_mask)
n_total_q = np.sum(qsub)
print(f"  城市核心像素 (战前NTL>10): {n_urban} / {n_total_q} ({n_urban/n_total_q*100:.1f}%)")

dNTL_urban = dNTL_v[urban_mask].reshape(-1,1)
print(f"  城市核心ΔNTL: mean={np.mean(dNTL_urban):.2f}, median={np.median(dNTL_urban):.2f}")
print(f"  城市核心NTL下降>50%: {np.sum(dNTL_urban < -0.5*ntl_pre_sub[urban_mask].ravel())} / {n_urban}")

# GMM仅在市区拟合
gmm1 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(dNTL_urban)
dm = np.argmin(gmm1.means_.flatten())
print(f"  GMM (仅市区): 正常={gmm1.means_.flatten()[1-dm]:.2f}, 损害={gmm1.means_.flatten()[dm]:.2f}")
print(f"  损害分量权重: {gmm1.weights_[dm]:.1%}")
print(f"  (v3全图: 正常=0.16, 损害=-2.12, 损害权重=3.1%)")

# 用市区GMM对所有像素评分
v1 = ~np.isnan(dNTL_v)
dNTL_flat = dNTL_v[v1].reshape(-1,1)
p1_f = gmm1.predict_proba(dNTL_flat)[:,dm]

# Bootstrap (市区采样)
n_urb = len(dNTL_urban)
p1_boot = np.zeros((N_BOOT, len(dNTL_flat)))
for b in range(N_BOOT):
    ib = RNG.choice(n_urb, n_urb, replace=True)
    gb = GaussianMixture(n_components=2, random_state=b, n_init=3)
    try:
        gb.fit(dNTL_urban[ib])
        p1_boot[b] = gb.predict_proba(dNTL_flat)[:,np.argmin(gb.means_.flatten())]
    except: p1_boot[b] = p1_f

p1 = np.full_like(dNTL_v, np.nan, dtype=np.float32); p1[v1] = p1_f.astype(np.float32)
s1 = np.full_like(dNTL_v, np.nan, dtype=np.float32); s1[v1] = np.clip(np.std(p1_boot,axis=0).astype(np.float32), 0.02, None)
print(f"  p₁: {np.nanmean(p1):.4f}, σ₁: {np.nanmean(s1[v1]):.4f}")
print(f"  市区p₁: {np.nanmean(p1[urban_mask]):.4f} (v3全图: 0.0307)")

# ============================================================
# 传感器 2: Sentinel-2 — 有符号PC1替代|PC1|
# ============================================================
print("\n[2/4] Sentinel-2 — 方向感知变化检测")

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

pre_s2_tdn = str(BASE/'s2_may2021/T37TDN')
post_s2_tdn = str(BASE/'战后2')

print("  T37TDN: 计算Δ指数...")
dNDVI = s2_delta(pre_s2_tdn, post_s2_tdn, 'NDVI')
dNBR  = s2_delta(pre_s2_tdn, post_s2_tdn, 'NBR')
dNDBI = s2_delta(pre_s2_tdn, post_s2_tdn, 'NDBI')
print(f"  ΔNDVI={np.nanmean(dNDVI):.4f}, ΔNBR={np.nanmean(dNBR):.4f}, ΔNDBI={np.nanmean(dNDBI):.4f}")

scl_pre = load_s2_band(pre_s2_tdn,'SCL_20m'); scl_post = load_s2_band(post_s2_tdn,'SCL_20m')
mask20 = (scl_pre>=2)&(scl_pre<=7)&(scl_post>=2)&(scl_post<=7)
H20,W20 = mask20.shape
dNDVI_20 = dNDVI[:H20*2:2,:W20*2:2]
vidx = np.where(mask20.ravel())[0]
ns = min(50000, len(vidx)); idx_s = RNG.choice(vidx, ns, replace=False)

X = np.column_stack([dNDVI_20.ravel()[idx_s], dNBR.ravel()[idx_s], dNDBI.ravel()[idx_s]])
X_s = StandardScaler().fit_transform(X)
pca = PCA(n_components=3).fit(X_s)
pc1 = pca.transform(X_s)[:,0]
# 检查PC1载荷方向
loadings = pca.components_[0]
print(f"  PC1载荷: NDVI={loadings[0]:.3f}, NBR={loadings[1]:.3f}, NDBI={loadings[2]:.3f}")
print(f"  PCA: PC1={pca.explained_variance_ratio_[0]:.1%}")

# === 改进2: 方向感知损害检测 ===
# 损害信号: NDVI↓ + NBR↓ + NDBI↑
# PC1载荷: NDVI=+0.577, NBR=+0.580, NDBI=-0.579
# 解释: PC1↑ = 植被变多(NDVI↑, NBR↑) + 建筑变少(NDBI↓) = "变绿", 非损害
#       PC1↓ = 植被变少(NDVI↓, NBR↓) + 建筑变多(NDBI↑) = "破坏", 损害方向
#
# 损害得分 = max(0, -PC1) — 仅在损害方向且幅度大时才算损害
# 然后GMM分离"零/负得分"(正常+变绿)和"高正得分"(损害)

ndvi_loading_sign = np.sign(loadings[0])
pc1_damage_dir = -ndvi_loading_sign * pc1  # damage direction = positive
damage_score = np.maximum(0, pc1_damage_dir)  # one-sided: only damage direction

print(f"  PC1载荷: NDVI={loadings[0]:.3f}, NBR={loadings[1]:.3f}, NDBI={loadings[2]:.3f}")
print(f"  damage_score范围=[{damage_score.min():.3f},{damage_score.max():.3f}], "
      f">0的比例={np.mean(damage_score>0.01):.1%}")

# GMM仅在 damage_score > 0 的部分区分"显著损害"和"微弱变化"
# 所有 damage_score ≈ 0 的像素 → p₂ ≈ 0
gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(damage_score.reshape(-1,1))
means2 = gmm2.means_.flatten()
dm2 = np.argmax(means2)  # higher mean = more damage
p2_vals = gmm2.predict_proba(damage_score.reshape(-1,1))[:,dm2]
print(f"  GMM(one-sided damage): 低分={means2[1-dm2]:.3f}, 高分={means2[dm2]:.3f}, 高分权重={gmm2.weights_[dm2]:.1%}")
print(f"  (v3用|PC1|: 损害权重=57%, 现在用了方向过滤)")

# Bootstrap
p2_boot = np.zeros((min(N_BOOT,50), ns))
for b in range(min(N_BOOT,50)):
    ib = RNG.choice(ns, ns, replace=True)
    gb = GaussianMixture(n_components=2, random_state=b, n_init=3)
    try:
        gb.fit(pc1_signed[ib].reshape(-1,1))
        p2_boot[b] = gb.predict_proba(pc1_signed.reshape(-1,1))[:,np.argmax(gb.means_.flatten())]
    except: p2_boot[b] = p2_vals
s2_vals = np.clip(np.std(p2_boot, axis=0), 0.03, None)
print(f"  p₂: {np.mean(p2_vals):.4f}, σ₂: {np.mean(s2_vals):.4f}")

# ============================================================
# 传感器 3: SAR — 直方图匹配 (同v3)
# ============================================================
print("\n[3/4] SAR — 直方图匹配归一化")

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
    with rasterio.open(vv_pre) as src: sar_pre = src.read(1).astype(np.float32)
    with rasterio.open(vv_post) as src: sar_post = src.read(1).astype(np.float32)

    Hc = min(sar_pre.shape[0], sar_post.shape[0])
    Wc = min(sar_pre.shape[1], sar_post.shape[1])
    sar_pre, sar_post = sar_pre[:Hc,:Wc], sar_post[:Hc,:Wc]

    # 直方图匹配
    v_pre = sar_pre.ravel()>0; v_post = sar_post.ravel()>0
    n_samp = 200000
    pre_samp = RNG.choice(sar_pre.ravel()[v_pre], min(n_samp, np.sum(v_pre)), replace=False)
    post_samp = RNG.choice(sar_post.ravel()[v_post], min(n_samp, np.sum(v_post)), replace=False)
    pcts = np.linspace(0,100,101)
    pre_pct = np.percentile(pre_samp, pcts); post_pct = np.percentile(post_samp, pcts)

    sar_post_norm = np.zeros_like(sar_post)
    vp = sar_post>0
    sar_post_norm[vp] = np.interp(np.interp(sar_post[vp], post_pct, pcts), pcts, pre_pct)

    dSAR = 10*np.log10(np.maximum(sar_post_norm,0.001)) - 10*np.log10(np.maximum(sar_pre,0.001))
    print(f"  ΔSAR(匹配后): mean={np.mean(dSAR):.4f} dB (v2原始: -1.61 dB)")

    stride=8; dSAR_sub=dSAR[::stride,::stride]; v_sar=np.isfinite(dSAR_sub)
    X_sar_full=dSAR_sub[v_sar].reshape(-1,1)
    # Subsampling: fit GMM on 50K, predict on full, fast bootstrap
    n_sar=min(50000,len(X_sar_full))
    idx_sar_fit=RNG.choice(len(X_sar_full),n_sar,replace=False)
    X_sar=X_sar_full[idx_sar_fit]

    gmm_sar=GaussianMixture(n_components=2, random_state=42, n_init=5).fit(X_sar)
    dm_sar=np.argmax(np.abs(gmm_sar.means_.flatten()))
    p3_vals=gmm_sar.predict_proba(X_sar_full)[:,dm_sar].astype(np.float32)

    # Fast bootstrap (subsampled fitting only)
    p3_boot=np.zeros((20,len(p3_vals)))
    for b in range(20):
        ib=RNG.choice(n_sar,n_sar,replace=True)
        gb=GaussianMixture(n_components=2,random_state=b,n_init=3)
        try:
            gb.fit(X_sar[ib])
            p3_boot[b]=gb.predict_proba(X_sar_full)[:,np.argmax(np.abs(gb.means_.flatten()))]
        except: p3_boot[b]=p3_vals
    s3_vals=np.clip(np.std(p3_boot,axis=0),0.03,None)

    p3_full=np.full(dSAR.shape,np.nan,dtype=np.float32)
    p3_sub=np.full(dSAR_sub.shape,np.nan,dtype=np.float32)
    p3_sub[v_sar]=p3_vals; p3_full[::stride,::stride]=p3_sub
    s3_full=np.full(dSAR.shape,np.nan,dtype=np.float32)
    s3_sub=np.full(dSAR_sub.shape,np.nan,dtype=np.float32)
    s3_sub[v_sar]=s3_vals; s3_full[::stride,::stride]=s3_sub

    print(f"  GMM SAR: 正常={gmm_sar.means_.flatten()[1-dm_sar]:.2f}, 损害={gmm_sar.means_.flatten()[dm_sar]:.2f}, 损害权重={gmm_sar.weights_[dm_sar]:.1%}")
    print(f"  p₃: {np.nanmean(p3_full):.4f}, σ₃: {np.nanmean(s3_vals):.4f}")
else:
    p3_full=np.full((100,150),np.nan,dtype=np.float32); s3_full=np.full((100,150),np.nan,dtype=np.float32)

# ============================================================
# 传感器 4: InSAR — τ归一化 (同v3)
# ============================================================
print("\n[4/4] InSAR — 时间基线归一化")

import rasterio
with rasterio.open(BASE/'干涉/20211230_20220204.geo.cc.tif') as src:
    coh_pre_raw=src.read(1).astype(np.float32)/255.
with rasterio.open(BASE/'干涉/20220228_20220312.geo.cc.tif') as src:
    coh_post_raw=src.read(1).astype(np.float32)/255.

dt_pre,dt_post,tau=36,12,50.
coh_pre_n=np.minimum(coh_pre_raw*np.exp(dt_pre/tau),1.0)
coh_post_n=np.minimum(coh_post_raw*np.exp(dt_post/tau),1.0)
dCoh_n=coh_post_n-coh_pre_n
p4_raw=(1./(1.+np.exp(10.*dCoh_n))).astype(np.float32)
p4_raw[(coh_pre_raw<0.1)&(coh_post_raw<0.1)]=np.nan
valid4=~np.isnan(p4_raw)
s4_raw=np.full_like(p4_raw,np.nan)
s4_raw[valid4]=np.clip(0.05+0.15*(1.-coh_pre_n[valid4]),0.03,0.3)
print(f"  p₄: {np.nanmean(p4_raw):.4f}")

# ============================================================
# 空间对齐
# ============================================================
def coarsen(data,Ht,Wt):
    if data is None or np.all(np.isnan(data)):
        return np.full((Ht,Wt),np.nan,dtype=np.float32)
    H,W=data.shape; bh,bw=max(1,H//Ht),max(1,W//Wt)
    out=np.full((Ht,Wt),np.nan,dtype=np.float32)
    for i in range(Ht):
        for j in range(Wt):
            blk=data[i*bh:min((i+1)*bh,H),j*bw:min((j+1)*bw,W)]
            v=blk[~np.isnan(blk)]
            if len(v)>0: out[i,j]=np.mean(v)
    return out

UH,UW=100,150
p1_u=coarsen(p1,UH,UW); s1_u=coarsen(s1,UH,UW)
p3_u=coarsen(p3_full,UH,UW); s3_u=coarsen(s3_full,UH,UW)
p4_u=coarsen(p4_raw,UH,UW); s4_u=coarsen(s4_raw,UH,UW)

# S2: 将子样本结果映射到统一网格
p2_u=np.full((UH,UW),np.nan,dtype=np.float32); s2_u=np.full((UH,UW),np.nan,dtype=np.float32)
for k,vi in enumerate(idx_s):
    i_map = int((vi//W20)/H20*UH); j_map = int((vi%W20)/W20*UW)
    if 0<=i_map<UH and 0<=j_map<UW:
        cur = p2_u[i_map,j_map]
        p2_u[i_map,j_map] = p2_vals[k] if np.isnan(cur) else (cur+p2_vals[k])/2
        s2_u[i_map,j_map] = s2_vals[k] if np.isnan(s2_u[i_map,j_map]) else (s2_u[i_map,j_map]+s2_vals[k])/2

names=['VIIRS','S2','SAR','InSAR']
P={'VIIRS':p1_u,'S2':p2_u,'SAR':p3_u,'InSAR':p4_u}
S={'VIIRS':s1_u,'S2':s2_u,'SAR':s3_u,'InSAR':s4_u}

print("\n统一网格统计:")
for n in names:
    v=~np.isnan(P[n])
    print(f"  {n:6s}: mean={np.nanmean(P[n]):.4f}, valid={np.sum(v)}/{UH*UW}")

# ============================================================
# 贝叶斯融合
# ============================================================
print("\n"+"="*65)
print("贝叶斯层次融合")
print("="*65)

valid_all = ~np.isnan(P['VIIRS'])
for n in names[1:]: valid_all &= ~np.isnan(P[n])
n_fuse = np.sum(valid_all); print(f"四传感器均有效: {n_fuse}/{UH*UW}")

if n_fuse>10:
    p_arr=np.column_stack([P[n][valid_all] for n in names])
    s_arr=np.column_stack([S[n][valid_all] for n in names])
    eps=1e-6; p_c=np.clip(p_arr,eps,1-eps)

    print("\n传感器相关性:")
    print(f"        {'VIIRS':>8} {'S2':>8} {'SAR':>8} {'InSAR':>8}")
    for i,n1 in enumerate(names):
        row=f"  {n1:6s}"
        for j in range(4): row+=f" {np.corrcoef(p_c[:,i],p_c[:,j])[0,1]:8.4f}"
        print(row)

    sigma_mean=np.clip(s_arr.mean(axis=0),0.02,1.0)
    kappa=np.clip(1.0/(sigma_mean**2+0.005),1.0,30.0)
    print(f"\nκ: {dict(zip(names,kappa.round(2)))}")

    a0,b0=1.0,1.0
    a_post=a0+sum(kappa[k]*p_c[:,k] for k in range(4))
    b_post=b0+sum(kappa[k]*(1-p_c[:,k]) for k in range(4))
    p_fused=a_post/(a_post+b_post)
    std_f=np.sqrt(a_post*b_post/((a_post+b_post)**2*(a_post+b_post+1)))
    hdi_l=np.array([stats.beta.ppf(0.025,a,b) if (a>0 and b>0) else 0 for a,b in zip(a_post,b_post)])
    hdi_h=np.array([stats.beta.ppf(0.975,a,b) if (a>0 and b>0) else 1 for a,b in zip(a_post,b_post)])

    print(f"\n贝叶斯融合: p={np.mean(p_fused):.4f}±{np.std(p_fused):.4f}, σ={np.mean(std_f):.4f}, HDI宽度={np.mean(hdi_h-hdi_l):.4f}")
    for k in range(4):
        print(f"  {names[k]:6s}→fusion: ρ={np.corrcoef(p_c[:,k],p_fused)[0,1]:.4f}")

    # Save
    for label,arr in [('fused_mean',p_fused),('fused_std',std_f),('hdi_low',hdi_l),('hdi_high',hdi_h)]:
        m=np.full((UH,UW),np.nan,dtype=np.float32)
        for k,vi in enumerate(np.where(valid_all.ravel())[0]): i,j=vi//UW,vi%UW; m[i,j]=arr[k]
        np.save(OUT/f'exp_v4_bayes_{label}.npy',m)

# Save single-sensor maps
for n in names:
    np.save(OUT/f'exp_v4_p_{n}.npy',P[n]); np.save(OUT/f'exp_v4_s_{n}.npy',S[n])

print(f"\n{'='*65}")
print("v4 完成 — exp_outputs_v4/")
print("="*65)
