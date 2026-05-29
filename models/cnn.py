"""1D-CNN 基线,参考 De Silva et al., "A Modular 1D-CNN Architecture for
Real-time Digital Pre-distortion", RWW 2022 [17]。

输入 [B, 2*tap + extra]:
  前 2*tap 个特征 reshape 成 [B, 2, tap] 作为 1D 卷积输入(2 通道 I/Q);
  其余 extra 个特征(包络/相位)与卷积输出拼接后送全连接。
"""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, tap: int, channels: int = 16, extra_dim: int = 4):
        super().__init__()
        self.tap = tap
        self.extra_dim = extra_dim
        self.conv = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels + extra_dim, 2)

    def forward(self, x):                            # x: [B, 2*tap + extra]
        B = x.shape[0]
        iq = x[:, :2 * self.tap].view(B, self.tap, 2).transpose(1, 2)  # [B, 2, tap]
        h = self.conv(iq)                            # [B, C, tap]
        h = self.pool(h).squeeze(-1)                 # [B, C]
        extra = x[:, 2 * self.tap:]
        return self.fc(torch.cat([h, extra], dim=-1))
