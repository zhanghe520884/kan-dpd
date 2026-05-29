"""GMP 主路径 + KAN 残差("物理先验 + 可学习一元非线性"混合架构)。

GMP 在 LS 拟合后系数固定,KAN 仅学习 GMP 主路径对真值的残差。
"""
import torch
import torch.nn as nn

from .kan import KAN


class GMP_KAN_Residual(nn.Module):
    def __init__(self, in_dim: int, kan_hidden: int = 16, grid_size: int = 7,
                 spline_order: int = 3):
        super().__init__()
        self.kan = KAN([in_dim, kan_hidden, 2],
                       grid_size=grid_size, spline_order=spline_order)

    def forward(self, feats: torch.Tensor, gmp_pred: torch.Tensor) -> torch.Tensor:
        return gmp_pred + self.kan(feats)
