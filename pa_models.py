"""Synthetic PA forward models used as ground-truth for experiments.

We supply three models, in increasing realism:
  * Saleh  — memoryless AM/AM + AM/PM (TWT-like)
  * Rapp   — memoryless soft-clip (SSPA-like)
  * GMP-based PA with memory — generalized memory polynomial with fixed
    "ground-truth" coefficients (gives a realistic PA with memory and
    cross-terms for evaluating DPD methods).
"""
import numpy as np


def saleh(x: np.ndarray, alpha_a: float = 2.1587, beta_a: float = 1.1517,
          alpha_p: float = 4.0033, beta_p: float = 9.1040) -> np.ndarray:
    r = np.abs(x)
    theta = np.angle(x)
    A = alpha_a * r / (1.0 + beta_a * r ** 2)
    P = alpha_p * r ** 2 / (1.0 + beta_p * r ** 2)
    return (A * np.exp(1j * (theta + P))).astype(np.complex64)


def rapp(x: np.ndarray, v_sat: float = 1.0, p: float = 2.5) -> np.ndarray:
    r = np.abs(x)
    g = r / (1.0 + (r / v_sat) ** (2 * p)) ** (1.0 / (2 * p))
    return (g * np.exp(1j * np.angle(x))).astype(np.complex64)


def _build_gmp_pa_coeffs(seed: int = 7):
    """构建 GMP-PA 的固定复数系数集合。 在模块加载时执行一次。

    返回:
        aligned_coefs: list of (k, q, coef)        —— 对齐项 (MP 项)
        cross_coefs:   list of (k, q, m, coef)     —— 滞后交叉项
    """
    rng = np.random.default_rng(seed)
    aligned = []
    for k in [1, 3, 5, 7]:
        for q in [0, 1, 2]:
            mag = (0.6 if k == 1 else 0.18) * (0.7 ** q)
            c = mag * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
            aligned.append((k, q, c))
    cross = []
    for k in [3, 5]:
        for q in [0, 1]:
            for m in [1, 2]:
                c = 0.05 * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
                cross.append((k, q, m, c))
    return aligned, cross


# 全局固定的 GMP-PA 系数。 关键:加载一次,所有调用共享 -> PA 是真正的固定函数。
_GMP_PA_ALIGNED, _GMP_PA_CROSS = _build_gmp_pa_coeffs(seed=7)
# 把第一项(线性)的相位旋转和增益吸收掉,使 PA 的小信号增益恒等于 1
_GMP_PA_LIN_GAIN = _GMP_PA_ALIGNED[0][2]   # 对齐项 k=1,q=0 的复系数


def gmp_pa(x: np.ndarray) -> np.ndarray:
    """带记忆 PA(固定复数 GMP 系数,小信号增益恒为 1)。

    模拟带记忆 + 交叉项的真实 PA 非线性行为。 系数在模块加载时一次性生成,
    不再随输入信号变化,从而确保 PA 是真正的固定函数。
    """
    N = len(x)
    y = np.zeros(N, dtype=np.complex64)

    # 对齐(MP) 项
    for k, q, coef in _GMP_PA_ALIGNED:
        if q == 0:
            xq = x
        else:
            xq = np.zeros_like(x); xq[q:] = x[:-q]
        y += (coef * xq * np.abs(xq) ** (k - 1)).astype(np.complex64)

    # 滞后交叉项
    for k, q, m, coef in _GMP_PA_CROSS:
        if q == 0:
            xq = x
        else:
            xq = np.zeros_like(x); xq[q:] = x[:-q]
        xqm = np.zeros_like(x); xqm[q + m:] = x[:-(q + m)]
        y += (coef * xq * np.abs(xqm) ** (k - 1)).astype(np.complex64)

    # 用固定的线性增益归一化,使小信号增益 = 1
    return (y / _GMP_PA_LIN_GAIN).astype(np.complex64)


PA_REGISTRY = {
    'saleh': saleh,
    'rapp':  rapp,
    'gmp_pa': gmp_pa,
}
