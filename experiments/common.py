"""实验脚本的公共工具:信号准备、模型工厂、检查点路径。"""
import os
import numpy as np
import torch

import config
from signal_gen import gen_ofdm_like, apply_backoff
from pa_models import PA_REGISTRY
from data import make_features, complex_to_iq, split_indices, feature_size
from models.mp_gmp import MemoryPolynomial, GMP
from models.nn_baselines import MLP, GRUNet
from models.cnn import CNN1D
from models.kan import KAN
from models.hybrid import GMP_KAN_Residual


# ---- 信号准备 ----
def prepare_signal(pa_name: str, seed: int, n_samples: int | None = None):
    """生成 OFDM-like 信号并施加 PA。 返回 (x, y, pa_fn, alpha)。"""
    n_samples = n_samples or config.N_SAMPLES
    rng = np.random.default_rng(seed)
    x = gen_ofdm_like(n_samples, config.N_SUBCARRIERS, config.OSR, rng)
    x = apply_backoff(x, config.SIGNAL_BACKOFF_DB)
    # 驱动电平:无记忆 PA 选较低电平避免严重饱和扩展导致 ILA 外推失败
    # 驱动电平:保证 PA 工作在中等压缩区,使 ILA 后逆器在 x 与 y_norm 域足够接近
    target_rms = {'saleh': 0.18, 'rapp': 0.18, 'gmp_pa': 0.13}.get(pa_name, 0.18)
    x = (x / np.sqrt(np.mean(np.abs(x) ** 2)) * target_rms).astype(np.complex64)
    pa_fn = PA_REGISTRY[pa_name]
    y = pa_fn(x)
    # alpha = ILA 目标线性增益。 对不同 PA 选择最稳健的估计方式:
    #   * gmp_pa: 已在内部把小信号增益归一化为 1,直接取 1
    #   * 其它(saleh/rapp): 低幅样本 LS,在小信号区接近物理增益
    if pa_name == 'gmp_pa':
        alpha = complex(1.0)
    else:
        low = np.abs(x) < 0.2 * np.std(np.abs(x))
        alpha = complex(np.sum(np.conj(x[low]) * y[low])
                        / (np.sum(np.abs(x[low]) ** 2) + 1e-12))
    return x, y, pa_fn, alpha


# ---- 路径 ----
def results_dir(pa_name: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, 'results', pa_name)
    os.makedirs(p, exist_ok=True)
    return p


def ckpt_path(pa_name: str, exp_tag: str, model_name: str, seed: int) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, 'checkpoints', pa_name, exp_tag)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f'{model_name}_seed{seed}.pt')


def history_path(pa_name: str, exp_tag: str, model_name: str, seed: int) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, 'checkpoints', pa_name, exp_tag, 'history')
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f'{model_name}_seed{seed}.json')


# ---- 模型工厂 ----
def make_nn_model(name: str, in_dim: int, tap: int, *,
                  kan_hidden: int = None, kan_grid: int = None):
    """根据 name 返回新建的 NN 模型实例。"""
    kan_hidden = kan_hidden or config.KAN_HIDDEN
    kan_grid = kan_grid or config.KAN_GRID
    extra = in_dim - 2 * tap   # envelope + phase 特征数
    if name == 'MLP':
        return MLP(in_dim, config.MLP_HIDDEN)
    if name == 'CNN':
        return CNN1D(tap, config.CNN_CHANNELS, extra_dim=extra)
    if name == 'GRU':
        return GRUNet(tap, config.GRU_HIDDEN, extra_dim=extra)
    if name == 'KAN':
        return KAN([in_dim, kan_hidden, 2], grid_size=kan_grid,
                   spline_order=config.KAN_SPLINE_ORDER)
    if name == 'GMP+KAN':
        return GMP_KAN_Residual(in_dim, kan_hidden=kan_hidden,
                                grid_size=kan_grid,
                                spline_order=config.KAN_SPLINE_ORDER)
    raise ValueError(f'unknown model {name}')


# ---- 复杂度 ----
def model_param_count(name: str, in_dim: int, tap: int) -> int:
    """近似参数量(用于横向比较)。 注:MP/GMP 是复数,1 个复参 ≈ 2 个实参。"""
    if name == 'MP':
        return 2 * config.MP_K * (config.MP_Q + 1)
    if name == 'GMP':
        # 对齐 + 滞后 + 超前
        K, Q, L = config.GMP_K, config.GMP_Q, config.GMP_LAG
        n = (Q + 1) * K + 2 * (Q + 1) * L * (K - 1)
        return 2 * n
    m = make_nn_model(name, in_dim, tap)
    return sum(p.numel() for p in m.parameters())
