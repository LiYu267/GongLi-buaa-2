"""
实验步骤3：贝叶斯层次融合（解析共轭近似）
==========================================
输入（从 exp_outputs/ 读取）：
  - exp_v1_p1_unified.npy  : VIIRS 损害概率 p₁ (统一网格)
  - exp_v1_s1_unified.npy  : VIIRS 标准差 σ₁
  - exp_v1_p2_unified.npy  : S2 损害概率 p₂
  - exp_v1_s2_unified.npy  : S2 标准差 σ₂

方法（解析贝叶斯共轭近似）:
  层次模型结构:
    Level 3: p_i_true ~ Beta(α₀, β₀)          ← 全局先验
    Level 2: p_i_obs_k ~ Beta(κₖ·p_i_true, κₖ·(1-p_i_true)) ← 传感器似然
    Level 1: p_i_true|obs ~ Beta(α₀+Σκₖpₖ, β₀+Σκₖ(1-pₖ))   ← 后验（解析）

  其中 κₖ = 1/σ²ₖ（传感器精度）
  注意：由于Windows环境PyMC MCMC在无C编译器下极慢，v1使用解析Beta-Binomial共轭近似。
  这是层次贝叶斯的有效近似——完整MCMC实现保留给后续版本。

输出：
  - exp_v1_bayes_fused_mean.npy       : 融合后验均值
  - exp_v1_bayes_fused_std.npy        : 融合后验标准差
  - exp_v1_bayes_fused_hdi_low.npy    : 95% HDI 下界
  - exp_v1_bayes_fused_hdi_high.npy   : 95% HDI 上界
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy import stats as sp_stats
from pathlib import Path

BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs'

print("=" * 60)
print("实验 v1：贝叶斯层次融合（解析共轭近似）")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/4] 加载单传感器概率估计...")

p1 = np.load(OUT / 'exp_v1_p1_unified.npy')
s1 = np.load(OUT / 'exp_v1_s1_unified.npy')
p2 = np.load(OUT / 'exp_v1_p2_unified.npy')
s2 = np.load(OUT / 'exp_v1_s2_unified.npy')
grid_info = dict(np.load(OUT / 'exp_v1_grid_info.npz'))

H, W = int(grid_info['H']), int(grid_info['W'])
print(f"  统一网格: {H}×{W} = {H*W} 像素")

# 有效像素
valid = ~np.isnan(p1) & ~np.isnan(p2) & ~np.isnan(s1) & ~np.isnan(s2)
valid &= (s1 > 0) | (s2 > 0)  # 至少一个传感器有正精度
n_valid = np.sum(valid)
print(f"  有效像素: {n_valid} / {H*W}")

# 截断
eps = 1e-6
p1_v = np.clip(p1[valid].flatten(), eps, 1 - eps)
p2_v = np.clip(p2[valid].flatten(), eps, 1 - eps)
s1_v = s1[valid].flatten()
s2_v = s2[valid].flatten()

print(f"  p₁ range: [{p1_v.min():.4f}, {p1_v.max():.4f}], mean={p1_v.mean():.4f}")
print(f"  p₂ range: [{p2_v.min():.4f}, {p2_v.max():.4f}], mean={p2_v.mean():.4f}")
print(f"  σ₁ mean: {s1_v.mean():.6f}, σ₂ mean: {s2_v.mean():.6f}")
print(f"  Corr(p₁, p₂) = {np.corrcoef(p1_v, p2_v)[0,1]:.4f}")

# ============================================================
# 2. 构建贝叶斯层次模型（解析共轭）
# ============================================================
print("\n[2/4] 构建贝叶斯层次模型...")
print("""
  Level 3（全局先验）:
    p_true ~ Beta(α₀, β₀)
    使用弱信息先验: α₀ = 1, β₀ = 1（等价于均匀分布）

  Level 2（传感器观测似然）:
    p_k | p_true ~ Beta(κₖ·p_true, κₖ·(1-p_true))
    其中 κₖ 是传感器的"有效样本量"，由精度 1/σ²ₖ 决定

  Level 1（后验推断 - Beta-Binomial 共轭）:
    p_true | p₁, p₂ ~ Beta(
      α = α₀ + κ₁·p₁ + κ₂·p₂,
      β = β₀ + κ₁·(1-p₁) + κ₂·(1-p₂)
    )

  物理意义：
  - κ 越大 = 传感器越可靠 → 后验更靠近该传感器的估计
  - 两传感器一致时 → 后验方差 < 任一单传感器方差
  - 两传感器分歧时 → 后验在两者之间，方差增大
""")

# 先验参数
alpha0 = 1.0  # Beta(1,1) = Uniform(0,1)
beta0  = 1.0

# 传感器精度（有效样本量）
# 注意：v1 Bootstrap σ 估计偏小，这里做校准：
# 将 σ 映射到 κ ∈ [1, 20] 范围
# κ = max(1, min(20, base_kappa * (1/σ²)))
# 这样即使 σ≈0也能得到合理的 κ

# 方法：κ = clamp(τ / (σ² + ε), 1, 50)
# τ 是调节参数，使得中位精度约为 5
tau = np.median(s1_v[s1_v > 0])**2 * 5 if np.any(s1_v > 0) else 5.0

kappa1_v = np.clip(tau / (s1_v**2 + 0.001), 1.0, 50.0)
kappa2_v = np.clip(tau / (s2_v**2 + 0.001), 1.0, 50.0)

print(f"  κ₁ range: [{kappa1_v.min():.1f}, {kappa1_v.max():.1f}], median={np.median(kappa1_v):.1f}")
print(f"  κ₂ range: [{kappa2_v.min():.1f}, {kappa2_v.max():.1f}], median={np.median(kappa2_v):.1f}")

# ============================================================
# 3. 计算后验
# ============================================================
print("\n[3/4] 计算后验分布...")

alpha_post = alpha0 + kappa1_v * p1_v + kappa2_v * p2_v
beta_post  = beta0 + kappa1_v * (1 - p1_v) + kappa2_v * (1 - p2_v)

p_fused = alpha_post / (alpha_post + beta_post + 1e-10)
std_fused = np.sqrt(alpha_post * beta_post /
                    ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1) + 1e-10))

# 95% HDI
hdi_low = np.zeros_like(p_fused)
hdi_high = np.zeros_like(p_fused)
for i in range(len(p_fused)):
    a, b = alpha_post[i], beta_post[i]
    if a > 0 and b > 0:
        hdi_low[i] = sp_stats.beta.ppf(0.025, a, b)
        hdi_high[i] = sp_stats.beta.ppf(0.975, a, b)

print(f"  融合后验均值:  {np.mean(p_fused):.4f} (±{np.std(p_fused):.4f})")
print(f"  融合后验标准差: {np.mean(std_fused):.4f}")
print(f"  95% HDI 平均宽度: {np.mean(hdi_high - hdi_low):.4f}")
print(f"  对比: 输入p₁均值={np.mean(p1_v):.4f}, 输入p₂均值={np.mean(p2_v):.4f}")

# ============================================================
# 4. 还原到全图并保存
# ============================================================
print("\n[4/4] 还原全图并保存...")

fused_mean_map = np.full((H, W), np.nan, dtype=np.float32)
fused_std_map  = np.full((H, W), np.nan, dtype=np.float32)
fused_hdi_low_map  = np.full((H, W), np.nan, dtype=np.float32)
fused_hdi_high_map = np.full((H, W), np.nan, dtype=np.float32)

valid_idx = np.where(valid.ravel())[0]
for k, vi in enumerate(valid_idx):
    i, j = vi // W, vi % W
    fused_mean_map[i, j] = p_fused[k]
    fused_std_map[i, j]  = std_fused[k]
    fused_hdi_low_map[i, j]  = hdi_low[k]
    fused_hdi_high_map[i, j] = hdi_high[k]

np.save(OUT / 'exp_v1_bayes_fused_mean.npy', fused_mean_map)
np.save(OUT / 'exp_v1_bayes_fused_std.npy', fused_std_map)
np.save(OUT / 'exp_v1_bayes_fused_hdi_low.npy', fused_hdi_low_map)
np.save(OUT / 'exp_v1_bayes_fused_hdi_high.npy', fused_hdi_high_map)

# 保存超参数用于报告
hyper_text = f"""
Bayesian Hierarchical Fusion (Conjugate Approximation)
======================================================
Prior: Beta({alpha0}, {beta0})  [uniform]
Sensor 1 (VIIRS): kappa median = {np.median(kappa1_v):.2f}
Sensor 2 (S2):    kappa median = {np.median(kappa2_v):.2f}
Posterior mean of p_fused: {np.nanmean(fused_mean_map):.4f}
Posterior std  of p_fused: {np.nanmean(fused_std_map):.4f}
Note: Analytical Beta-Binomial conjugate used. Full MCMC pending.
"""
with open(OUT / 'exp_v1_bayes_hyperparams.txt', 'w', encoding='utf-8') as fp:
    fp.write(hyper_text)

print("\n" + "=" * 60)
print("贝叶斯融合完成（解析Beta-Binomial共轭近似）")
print(f"  → exp_v1_bayes_fused_mean.npy  ({fused_mean_map.nbytes/1024:.0f} KB)")
print(f"  → exp_v1_bayes_fused_std.npy   ({fused_std_map.nbytes/1024:.0f} KB)")
print(f"  → exp_v1_bayes_fused_hdi_low.npy / high.npy")
print("=" * 60)
