"""
实验步骤4：Dempster-Shafer 证据理论融合
========================================
输入（从 exp_outputs/ 读取）：
  - exp_v1_p1_unified.npy  : VIIRS 损害概率 p₁
  - exp_v1_s1_unified.npy  : VIIRS 标准差 σ₁
  - exp_v1_p2_unified.npy  : S2 损害概率 p₂
  - exp_v1_s2_unified.npy  : S2 标准差 σ₂

方法：
  Dempster-Shafer 证据理论
  识别框架 Θ = {D(损害), U(未损害)}
  基本概率分配 (BPA):
    m_k({D}) = p_k · (1 - σ_k)      ← 支持"损害"的证据
    m_k({U}) = (1 - p_k) · (1 - σ_k) ← 支持"未损害"的证据
    m_k(Θ)   = σ_k                   ← 分配给"全局无知"的证据
  Dempster 组合规则: m₁₂ = m₁ ⊕ m₂

输出：
  - exp_v1_ds_bel.npy    : Bel(损害) — 信任下界
  - exp_v1_ds_pl.npy     : Pl(损害) — 信任上界
  - exp_v1_ds_ignorance.npy : Pl - Bel — 认知不确定性区间宽度
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path

BASE = Path('d:/数理统计大作业数据')
OUT = BASE / 'exp_outputs'

print("=" * 60)
print("实验 v1：Dempster-Shafer 证据理论融合")
print("=" * 60)

# ============================================================
# 1. 加载数据 & 构建 BPA
# ============================================================
print("\n[1/3] 加载传感器概率估计，构建基本概率分配...")

p1 = np.load(OUT / 'exp_v1_p1_unified.npy')
s1 = np.load(OUT / 'exp_v1_s1_unified.npy')
p2 = np.load(OUT / 'exp_v1_p2_unified.npy')
s2 = np.load(OUT / 'exp_v1_s2_unified.npy')

H, W = p1.shape

# 有效像素掩膜
valid = ~np.isnan(p1) & ~np.isnan(s1) & ~np.isnan(p2) & ~np.isnan(s2)
valid &= (s1 >= 0) & (s2 >= 0)

# 将标准差归一化到 [0, 1] 作为"无知度"
# 原始 σ₁, σ₂ 在 [0, ~0.3] 范围，做 min-max 归一化
all_s1 = s1[valid].flatten()
all_s2 = s2[valid].flatten()
# 使用95百分位作为上界，避免异常值影响
s1_95 = np.percentile(all_s1, 95) if len(all_s1) > 0 else 0.3
s2_95 = np.percentile(all_s2, 95) if len(all_s2) > 0 else 0.3
s1_norm = np.clip(s1 / max(s1_95, 0.01), 0, 0.99)
s2_norm = np.clip(s2 / max(s2_95, 0.01), 0, 0.99)

print(f"  σ₁ 归一化参考值 (P95): {s1_95:.4f}")
print(f"  σ₂ 归一化参考值 (P95): {s2_95:.4f}")
print(f"  σ₁_norm range: [{np.nanmin(s1_norm[valid]):.3f}, {np.nanmax(s1_norm[valid]):.3f}]")
print(f"  σ₂_norm range: [{np.nanmin(s2_norm[valid]):.3f}, {np.nanmax(s2_norm[valid]):.3f}]")

# 构建 BPA
# m({D}) = p · (1 - σ_norm)
# m({U}) = (1 - p) · (1 - σ_norm)
# m(Θ)   = σ_norm
m1_D  = p1 * (1 - s1_norm)
m1_U  = (1 - p1) * (1 - s1_norm)
m1_Theta = s1_norm
m1_D[~valid] = np.nan; m1_U[~valid] = np.nan; m1_Theta[~valid] = np.nan

m2_D  = p2 * (1 - s2_norm)
m2_U  = (1 - p2) * (1 - s2_norm)
m2_Theta = s2_norm
m2_D[~valid] = np.nan; m2_U[~valid] = np.nan; m2_Theta[~valid] = np.nan

print(f"  VIIRS BPA: m(D)={np.nanmean(m1_D):.4f}, m(U)={np.nanmean(m1_U):.4f}, m(Θ)={np.nanmean(m1_Theta):.4f}")
print(f"  S2 BPA:    m(D)={np.nanmean(m2_D):.4f}, m(U)={np.nanmean(m2_U):.4f}, m(Θ)={np.nanmean(m2_Theta):.4f}")

# ============================================================
# 2. Dempster 组合规则
# ============================================================
print("\n[2/3] 执行 Dempster 组合...")

# 对每个像素:
# K = m1({D})·m2({U}) + m1({U})·m2({D})   ← 冲突质量
# m12({D}) = (m1({D})·m2({D}) + m1({D})·m2(Θ) + m1(Θ)·m2({D})) / (1-K)
# m12({U}) 同理
# m12(Θ) = m1(Θ)·m2(Θ) / (1-K)

# 逐个像素计算 D-S 组合 (矢量化运算)
K = m1_D * m2_U + m1_U * m2_D  # 冲突

# 处理 K=1 的情况（完全冲突，组合无定义）
K_safe = np.clip(K, 0, 0.9999)
one_minus_K = 1 - K_safe

m12_D = (m1_D * m2_D + m1_D * m2_Theta + m1_Theta * m2_D) / one_minus_K
m12_U = (m1_U * m2_U + m1_U * m2_Theta + m1_Theta * m2_U) / one_minus_K
m12_Theta = (m1_Theta * m2_Theta) / one_minus_K

# 标记完全冲突的像素
total_conflict = (K > 0.999)
m12_D[total_conflict] = np.nan
m12_U[total_conflict] = np.nan
m12_Theta[total_conflict] = 1.0

# Bel(损害) = m12({D})   （因为没有真子集）
# Pl(损害) = m12({D}) + m12(Θ) = 1 - m12({U})
bel = m12_D
pl  = m12_D + m12_Theta
ignorance = pl - bel  # = m12(Θ)

print(f"  D-S 融合结果:")
print(f"    Bel(D): mean={np.nanmean(bel):.4f}, std={np.nanstd(bel):.4f}")
print(f"    Pl(D):  mean={np.nanmean(pl):.4f}, std={np.nanstd(pl):.4f}")
print(f"    Ignorance (Pl-Bel): mean={np.nanmean(ignorance):.4f}")
print(f"    冲突质量 K: mean={np.nanmean(K):.4f}")
print(f"    完全冲突像素 (K>0.999): {np.sum(total_conflict)}")

# ============================================================
# 3. 保存结果
# ============================================================
print("\n[3/3] 保存 D-S 融合结果...")

np.save(OUT / 'exp_v1_ds_bel.npy', bel.astype(np.float32))
np.save(OUT / 'exp_v1_ds_pl.npy', pl.astype(np.float32))
np.save(OUT / 'exp_v1_ds_ignorance.npy', ignorance.astype(np.float32))
np.save(OUT / 'exp_v1_ds_conflict.npy', K.astype(np.float32))

# 简单投票基线（用于后续对比）
vote_mean = np.full_like(p1, np.nan)
vote_mean[valid] = (p1[valid] + p2[valid]) / 2

# 加权投票（按精度倒数加权）
precision1 = 1.0 / (s1**2 + 0.001)
precision2 = 1.0 / (s2**2 + 0.001)
vote_weighted = np.full_like(p1, np.nan)
vote_weighted[valid] = (p1[valid] * precision1[valid] + p2[valid] * precision2[valid]) / (precision1[valid] + precision2[valid])

np.save(OUT / 'exp_v1_vote_mean.npy', vote_mean.astype(np.float32))
np.save(OUT / 'exp_v1_vote_weighted.npy', vote_weighted.astype(np.float32))

print("  → exp_v1_ds_bel.npy, exp_v1_ds_pl.npy, exp_v1_ds_ignorance.npy")
print("  → exp_v1_ds_conflict.npy")
print("  → exp_v1_vote_mean.npy, exp_v1_vote_weighted.npy (基线方法)")

print("\n" + "=" * 60)
print("D-S 融合完成。")
print(f"  Bel(损害) = {np.nanmean(bel):.3f} [信任下界]")
print(f"  Pl(损害)  = {np.nanmean(pl):.3f} [信任上界]")
print(f"  不确定性 = {np.nanmean(ignorance):.3f} [区间宽度]")
print(f"  对比: 输入 p₁={np.nanmean(p1[valid]):.3f}, p₂={np.nanmean(p2[valid]):.3f}")
print("=" * 60)
