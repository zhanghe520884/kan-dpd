"""特征提取与 Dataset。

KAN 增强输入特征:
    [I[n],Q[n], I[n-1],Q[n-1], ..., I[n-T+1],Q[n-T+1],
     |x[n]|, |x[n]|^3, sin θ[n], cos θ[n]]
其中 θ[n] = angle(x[n])。 因此每样本特征维度 = 2*TAP + 4。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


# 特征维度: 2*TAP (I/Q 抽头) + 4 (|x|, |x|^3, sinθ, cosθ)
def feature_size(tap: int, with_envelope: bool = True,
                 with_phase: bool = True) -> int:
    d = 2 * tap
    if with_envelope:
        d += 2
    if with_phase:
        d += 2
    return d


def make_features(x: np.ndarray, tap: int,
                  with_envelope: bool = True,
                  with_phase: bool = True) -> np.ndarray:
    """构造 NN 模型的实值输入特征矩阵。

    Args:
        x: 复基带样本 [N]
        tap: 时延抽头数
        with_envelope: 是否拼接 |x|, |x|^3 (幅度特征)
        with_phase: 是否拼接 sinθ, cosθ (相位特征)
    Returns:
        features: float32 数组 [N, feature_size(...)]
    """
    N = len(x)
    d = feature_size(tap, with_envelope, with_phase)
    feats = np.zeros((N, d), dtype=np.float32)
    # 抽头 I/Q
    for q in range(tap):
        xq = np.zeros(N, dtype=np.complex64)
        if q == 0:
            xq[:] = x
        else:
            xq[q:] = x[:-q]
        feats[:, 2 * q] = xq.real
        feats[:, 2 * q + 1] = xq.imag
    col = 2 * tap
    if with_envelope:
        mag = np.abs(x).astype(np.float32)
        feats[:, col] = mag
        feats[:, col + 1] = mag ** 3
        col += 2
    if with_phase:
        # 用 x/|x| 计算 cos/sin,避免 angle() 在 |x|≈0 处不稳定
        eps = 1e-8
        ax = np.abs(x) + eps
        feats[:, col] = (x.real / ax).astype(np.float32)        # cosθ
        feats[:, col + 1] = (x.imag / ax).astype(np.float32)    # sinθ
    return feats


def complex_to_iq(y: np.ndarray) -> np.ndarray:
    """复数 -> [I,Q] 实值表示。"""
    return np.stack([y.real, y.imag], axis=-1).astype(np.float32)


def iq_to_complex(iq: np.ndarray) -> np.ndarray:
    return (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64)


def split_indices(N: int, train_frac: float, val_frac: float, tap: int):
    """顺序切分(避免 test 泄漏)。 跳过最前 tap 个样本(延迟线零填充区)。"""
    start = tap
    end = N
    total = end - start
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    train_idx = np.arange(start, start + n_train)
    val_idx = np.arange(start + n_train, start + n_train + n_val)
    test_idx = np.arange(start + n_train + n_val, end)
    return train_idx, val_idx, test_idx


class FeatureDataset(Dataset):
    """通用样本数据集。 extra 可选,用于 hybrid 模型的主路径输出输入。"""

    def __init__(self, feats: np.ndarray, labels: np.ndarray, idx: np.ndarray,
                 extra: np.ndarray | None = None):
        self.feats = feats
        self.labels = labels
        self.idx = idx
        self.extra = extra

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        if self.extra is None:
            return (torch.from_numpy(self.feats[k]),
                    torch.from_numpy(self.labels[k]))
        return (torch.from_numpy(self.feats[k]),
                torch.from_numpy(self.labels[k]),
                torch.from_numpy(self.extra[k]))
