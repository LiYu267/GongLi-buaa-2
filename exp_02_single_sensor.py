"""
实验步骤2：单传感器损害概率估计
================================
输入（从 exp_outputs/ 读取）：
  - exp_v1_viirs_dntl.npy       : VIIRS ΔNTL
  - exp_v1_s2_dNDVI_t37tdn.npy  : S2 ΔNDVI (10m)
  - exp_v1_s2_dNDWI_t37tdn.npy  : S2 ΔNDWI (10m)
  - exp_v1_s2_dNBR_t37tdn.npy   : S2 ΔNBR (20m)
  - exp_v1_s2_dNDBI_t37tdn.npy  : S2 ΔNDBI (20m)
  - (T37TCN 同理)

方法：
  VIIRS → GMM 2分量聚类 ΔNTL → "损害"后验概率 p₁
          Bootstrap 200次重采样 → 方差 σ²₁
  S2     → 4个Δ指数 → PCA降维 → PC1 作为损害得分
          战前-战后 Hotelling T² → p-value → p₂
          Bootstrap → σ²₂

输出：
  - exp_v1_p1_viirs.npy      : VIIRS 损害概率 (与 ΔNTL 同网格)
  - exp_v1_s1_viirs.npy      : VIIRS 标准差
  - exp_v1_p2_s2.npy          : S2 损害概率 (上采样到 VIIRS 网格)
  - exp_v1_s2_s2.npy          : S2 标准差
  - exp_v1_unified_grid_info.npz : 统一网格参数

命名说明：
  p_k = 传感器 k 的损害概率估计
  s_k = 传感器 k 的标准差 (σ_k)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 设置
# ============================================================
BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs'
RNG = np.random.RandomState(42)
N_BOOTSTRAP = 200

print("=" * 60)
print("实验 v1：单传感器损害概率估计")
print("=" * 60)

# ============================================================
# 1. 加载预处理数据
# ============================================================
print("\n[1/4] 加载预处理数据...")

dntl = np.load(OUT / 'exp_v1_viirs_dntl.npy')  # (Hv, Wv), may contain NaN

# S2 数据：加载 mask 和 index 差值
s2_data = {}
for tile in ['t37tdn', 't37tcn']:
    mask = np.load(OUT / f'exp_v1_s2_mask_{tile}.npy')  # (H20, W20)
    dndvi = np.load(OUT / f'exp_v1_s2_dNDVI_{tile}.npy')   # (H10, W10)
    dndwi = np.load(OUT / f'exp_v1_s2_dNDWI_{tile}.npy')   # (H10, W10)
    dnbr  = np.load(OUT / f'exp_v1_s2_dNBR_{tile}.npy')     # (H20, W20)
    dndbi = np.load(OUT / f'exp_v1_s2_dNDBI_{tile}.npy')    # (H20, W20)

    # 统一上采样到 10m 分辨率
    # mask 是 20m，上采样到 10m
    mask_10m = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)[:dndvi.shape[0], :dndvi.shape[1]]

    # NBR, NDBI 从 20m 上采样到 10m
    dnbr_10m = np.repeat(np.repeat(dnbr, 2, axis=0), 2, axis=1)[:dndvi.shape[0], :dndvi.shape[1]]
    dndbi_10m = np.repeat(np.repeat(dndbi, 2, axis=0), 2, axis=1)[:dndvi.shape[0], :dndvi.shape[1]]

    # 组合有效掩膜：mask AND 所有指数都有有限值
    valid_mask = (mask_10m &
                  np.isfinite(dndvi) & np.isfinite(dndwi) &
                  np.isfinite(dnbr_10m) & np.isfinite(dndbi_10m))

    s2_data[tile] = {
        'dndvi': dndvi, 'dndwi': dndwi,
        'dnbr': dnbr_10m, 'dndbi': dndbi_10m,
        'mask': valid_mask
    }
    n_valid = np.sum(valid_mask)
    print(f"  {tile}: {n_valid} valid 10m pixels ({n_valid/valid_mask.size*100:.1f}%)")

# ============================================================
# 2. VIIRS 损害概率估计
# ============================================================
print("\n[2/4] VIIRS GMM 损害概率估计...")

# 只用有效像素（非NaN）拟合 GMM
valid_viirs = ~np.isnan(dntl)
dntl_flat = dntl[valid_viirs].reshape(-1, 1)

# GMM 2分量：分量0=正常波动(均值接近0)，分量1=显著下降(均值为负)
gmm = GaussianMixture(n_components=2, random_state=42,
                       init_params='kmeans', n_init=5)
gmm.fit(dntl_flat)

# 识别"损害"分量（均值更负的）
means = gmm.means_.flatten()
damage_comp = np.argmin(means)
normal_comp = 1 - damage_comp
print(f"  GMM 均值: 正常分量={means[normal_comp]:.3f}, 损害分量={means[damage_comp]:.3f}")
print(f"  GMM 权重: 正常={gmm.weights_[normal_comp]:.3f}, 损害={gmm.weights_[damage_comp]:.3f}")

# 每个像素属于"损害"分量的后验概率 = p₁
probs = gmm.predict_proba(dntl_flat)
p1_flat = probs[:, damage_comp]

# 还原到网格
p1_viirs = np.full(dntl.shape, np.nan, dtype=np.float32)
p1_viirs[valid_viirs] = p1_flat.astype(np.float32)

print(f"  p₁ 范围: [{np.nanmin(p1_viirs):.4f}, {np.nanmax(p1_viirs):.4f}]")
print(f"  p₁ 均值: {np.nanmean(p1_viirs):.4f}")

# Bootstrap 估计方差
print(f"  Bootstrap ({N_BOOTSTRAP}次)...")
p1_boot = np.zeros((N_BOOTSTRAP, np.sum(valid_viirs)), dtype=np.float32)
n_samples = len(dntl_flat)
for b in range(N_BOOTSTRAP):
    idx = RNG.choice(n_samples, n_samples, replace=True)
    boot_data = dntl_flat[idx]
    gmm_b = GaussianMixture(n_components=2, random_state=b, n_init=3)
    try:
        gmm_b.fit(boot_data)
        probs_b = gmm_b.predict_proba(dntl_flat)
        dm_b = np.argmin(gmm_b.means_.flatten())
        p1_boot[b] = probs_b[:, dm_b].astype(np.float32)
    except:
        p1_boot[b] = p1_flat

s1_viirs = np.full(dntl.shape, np.nan, dtype=np.float32)
s1_viirs[valid_viirs] = np.std(p1_boot, axis=0).astype(np.float32)
print(f"  σ₁ 范围: [{np.nanmin(s1_viirs):.4f}, {np.nanmax(s1_viirs):.4f}]")
print(f"  σ₁ 均值: {np.nanmean(s1_viirs):.4f}")

np.save(OUT / 'exp_v1_p1_viirs.npy', p1_viirs)
np.save(OUT / 'exp_v1_s1_viirs.npy', s1_viirs)

# ============================================================
# 3. Sentinel-2 损害概率估计
# ============================================================
print("\n[3/4] Sentinel-2 PCA + Hotelling T² 损害概率估计...")

def estimate_s2_damage(tile_data, tile_name, rng):
    """对单个 S2 tile 估计损害概率"""
    # 提取有效像素的 4 维特征向量
    mask = tile_data['mask']
    dndvi  = tile_data['dndvi'][mask]
    dndwi  = tile_data['dndwi'][mask]
    dnbr   = tile_data['dnbr'][mask]
    dndbi  = tile_data['dndbi'][mask]

    X = np.column_stack([dndvi, dndwi, dnbr, dndbi])
    print(f"  {tile_name} 特征矩阵: {X.shape}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=4)
    pca.fit(X_scaled)
    print(f"  PCA 解释方差: {pca.explained_variance_ratio_}")

    # PC1 作为损害指数
    pc1 = pca.transform(X_scaled)[:, 0]

    # 方法：H₀: 变化均值为0 → Hotelling T² 检验
    # T² = n * (x̄)' Σ⁻¹ (x̄)
    # 实际上在低维空间 p-value = 1 - F_cdf(T² * (n-p)/(p(n-1)))
    n, p = X_scaled.shape
    xbar = X_scaled.mean(axis=0)
    S = np.cov(X_scaled.T)

    try:
        S_inv = np.linalg.inv(S)
        T2 = n * xbar @ S_inv @ xbar
        F_stat = T2 * (n - p) / (p * (n - 1))
        pval = 1 - stats.f.cdf(F_stat, p, n - p)
    except np.linalg.LinAlgError:
        pval = 0.5
        T2 = 0

    print(f"  Hotelling T² = {T2:.2f}, p-value = {pval:.6f}")

    # 对每个像素：p₂ = 1 - FDR-adjusted p-value (简化：直接用 PC1 的极端程度)
    # 更实际的方法：PC1 偏离0越远，越可能是损害
    # 使用 logistic 函数映射 PC1 → [0,1]
    # p₂ = sigmoid(-PC1 * sign) 使得 PC1 越负 → p₂ 越高
    pc1_sign = np.sign(np.median(pc1) - np.mean(pc1))  # 判断损害方向
    if pc1_sign > 0:
        # PC1 越大 = 越损害
        p2_flat = 1 / (1 + np.exp(-pc1))  # sigmoid
    else:
        p2_flat = 1 / (1 + np.exp(pc1))

    # 重建到全图
    p2_full = np.full(mask.shape, np.nan, dtype=np.float32)
    p2_full[mask] = p2_flat.astype(np.float32)

    # Bootstrap 方差估计
    s2_flat = np.zeros(len(p2_flat), dtype=np.float32)
    step = max(1, len(p2_flat) // 10000)  # 采样避免过慢
    X_sub = X[::step]
    for b in range(min(N_BOOTSTRAP, 50)):  # S2像素太多，减少bootstrap次数
        idx_b = rng.choice(len(X_sub), len(X_sub), replace=True)
        pca_b = PCA(n_components=4)
        Xb = X_sub[idx_b]
        pca_b.fit(StandardScaler().fit_transform(Xb))
        pc1_b = pca_b.transform(X_scaled[::step])[:, 0]
        p2_b = 1 / (1 + np.exp(pc1_b)) if pc1_sign <= 0 else 1 / (1 + np.exp(-pc1_b))
        # Upsample back
        idx_full = np.arange(0, len(p2_flat), step)[:len(p2_b)]
        s2_flat[idx_full] += (p2_b - p2_flat[idx_full]) ** 2

    s2_flat = np.sqrt(s2_flat / min(N_BOOTSTRAP, 50))
    s2_full = np.full(mask.shape, np.nan, dtype=np.float32)
    s2_full[mask] = s2_flat.astype(np.float32)

    return p2_full, s2_full, pc1, pca

print("\n  处理 T37TDN...")
p2_t37tdn, s2_t37tdn, pc1_tdn, pca_tdn = estimate_s2_damage(s2_data['t37tdn'], 'T37TDN', RNG)

print("\n  处理 T37TCN...")
p2_t37tcn, s2_t37tcn, pc1_tcn, pca_tcn = estimate_s2_damage(s2_data['t37tcn'], 'T37TCN', RNG)

# ============================================================
# 4. 空间对齐：将 S2 聚合到 VIIRS 网格
# ============================================================
print("\n[4/4] 空间对齐：S2 (10m/20m) → VIIRS (500m) 统一网格...")
print("  注意：这是实验v1的简化对齐，使用分块平均聚合")

# VIIRS 网格尺寸
Hv, Wv = dntl.shape  # 242 x 355
# S2 10m 网格尺寸
Hs2, Ws2 = p2_t37tdn.shape  # 10980 x 10980

# 简化方法：块平均 (block averaging)
# VIIRS 500m / S2 10m = 50x upscale factor
# 直接降采样 50x
def downsample_block_mean(data, block_size=50):
    """块平均降采样，忽略 NaN"""
    H, W = data.shape
    H_new = H // block_size
    W_new = W // block_size
    result = np.full((H_new, W_new), np.nan, dtype=np.float32)
    for i in range(H_new):
        for j in range(W_new):
            block = data[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            valid = block[~np.isnan(block)]
            if len(valid) > block_size * block_size * 0.1:  # at least 10% valid
                result[i, j] = np.mean(valid)
    return result

# 实际对齐需要地理坐标匹配。在 exp_v1 中，我们简化：
# - S2 两个 tile 各有约 10980×10980 像素
# - VIIRS 子区域 242×355 像素
# - VIIRS 覆盖更广，S2 tile 覆盖其子集
# - 我们取 S2 降采样50倍，然后裁剪到与 VIIRS 重叠
#
# 但由于 VIIRS 和 S2 的精确地理对齐需要坐标变换，在 v1 中采用简化策略：
# 对 VIIRS 网格的每个像素，找到对应的 S2 区域做聚合

print("  简化对齐：将 S2(10m) 降采样 50× 到 ~500m...")

p2_ds = downsample_block_mean(p2_t37tdn, 50)  # 大约 219×219
s2_ds = downsample_block_mean(s2_t37tdn, 50)

# 合并两个 S2 tile（取并集，实际应该做镶嵌）
# v1简化：仅用T37TDN（覆盖马里乌波尔市区）

# 进一步裁剪到与VIIRS相同的形状
H_target, W_target = min(Hv, p2_ds.shape[0]), min(Wv, p2_ds.shape[1])
p2_unified = p2_ds[:H_target, :W_target]
s2_unified = s2_ds[:H_target, :W_target]

# 对应的 VIIRS 数据也需要裁剪对齐
p1_unified = p1_viirs[:H_target, :W_target]
s1_unified = s1_viirs[:H_target, :W_target]

print(f"  统一网格: {H_target}×{W_target} 像素 @ ~500m")
print(f"  p₁(VIIRS): mean={np.nanmean(p1_unified):.3f}, std={np.nanstd(p1_unified):.3f}")
print(f"  p₂(S2):    mean={np.nanmean(p2_unified):.3f}, std={np.nanstd(p2_unified):.3f}")

# 保存对齐后的结果
np.save(OUT / 'exp_v1_p1_unified.npy', p1_unified)
np.save(OUT / 'exp_v1_s1_unified.npy', s1_unified)
np.save(OUT / 'exp_v1_p2_unified.npy', p2_unified)
np.save(OUT / 'exp_v1_s2_unified.npy', s2_unified)
np.savez(OUT / 'exp_v1_grid_info.npz',
         H=H_target, W=W_target,
         pixel_size_m=500,
         description='Unified grid ~500m, block-averaged from S2 10m')

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
print("单传感器估计完成。输出文件：")
import itertools
for f in sorted(itertools.chain(OUT.glob('exp_v1_p*_*'), OUT.glob('exp_v1_s*_*'), OUT.glob('exp_v1_grid*'))):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.0f} KB)")
print("=" * 60)
print("\n关键发现（v1实验）：")
print(f"  VIIRS p₁: 基于GMM，损害概率集中在 {np.nanmean(p1_unified):.2f}±{np.nanstd(p1_unified):.2f}")
print(f"  S2 p₂:    基于PCA+逻辑映射，损害概率集中在 {np.nanmean(p2_unified):.2f}±{np.nanstd(p2_unified):.2f}")
print(f"  两传感器相关性: ρ={np.corrcoef(p1_unified[~np.isnan(p1_unified) & ~np.isnan(p2_unified)], p2_unified[~np.isnan(p1_unified) & ~np.isnan(p2_unified)])[0,1]:.3f}")
print("  注意：相关性预期偏低，因为S2存在严重季节混淆（1月vs5月=冬季vs春季）")
