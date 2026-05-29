"""Compact B-spline KAN (Kolmogorov-Arnold Network) layer.

The implementation follows the "efficient KAN" reformulation: each edge holds
its own univariate spline, expressed as
    phi(x) = w_b * silu(x) + w_s * sum_i c_i * B_i(x)
where B_i are B-spline basis functions on a fixed grid. The grid is fixed
(not learned) for simplicity — sufficient for the PA/DPD setting where
inputs are normalized to a known range.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 7, spline_order: int = 3,
                 grid_range: tuple = (-1.5, 1.5),
                 scale_noise: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Fixed grid: shape [in_features, grid_size + 2*spline_order + 1]
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0])
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer('grid', grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))
        with torch.no_grad():
            noise = (torch.rand(grid_size + 1, in_features, out_features)
                     - 0.5) * scale_noise / grid_size
            self.spline_weight.data.copy_(
                self._curve2coeff(self.grid.T[spline_order:-spline_order], noise)
            )

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis. x: [B, in_features] -> [B, in_features, n_basis]"""
        grid = self.grid  # [in, G+2k+1]
        x = x.unsqueeze(-1)  # [B, in, 1]
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8) * bases[..., :-1]
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k] + 1e-8) * bases[..., 1:]
            bases = left + right
        return bases  # [B, in, G+k]

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Solve for spline coefficients matching curve samples (init helper)."""
        # x: [n_pts, in], y: [n_pts, in, out]  -> coeffs [out, in, G+k]
        A = self._b_splines(x).transpose(0, 1)  # [in, n_pts, G+k]
        B = y.transpose(0, 1)                   # [in, n_pts, out]
        sol = torch.linalg.lstsq(A, B).solution  # [in, G+k, out]
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base path
        base = F.linear(F.silu(x), self.base_weight)
        # spline path
        bases = self._b_splines(x)  # [B, in, G+k]
        spline = F.linear(
            bases.view(x.shape[0], -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base + spline


class KAN(nn.Module):
    """Stack of KAN layers ending in a linear (or KAN) head."""

    def __init__(self, layers: list[int], grid_size: int = 7,
                 spline_order: int = 3, grid_range: tuple = (-1.5, 1.5)):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            mods.append(KANLinear(layers[i], layers[i + 1],
                                  grid_size=grid_size,
                                  spline_order=spline_order,
                                  grid_range=grid_range))
        self.layers = nn.ModuleList(mods)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
