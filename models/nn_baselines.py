"""神经网络基线: RVTDNN 风格 MLP 与小型 GRU。

输入特征由 data.make_features 给出,输入维度由 feature_size() 决定。
输出为 [I,Q] 二维实向量,代表复数样本的实部/虚部。
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """RVTDNN(real-valued time-delay neural network)风格的 MLP。"""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):                            # x: [B, in_dim]
        return self.net(x)


class GRUNet(nn.Module):
    """以时延抽头为时间序列的 GRU。 每个时间步为 (I,Q,|x|) 三通道。"""

    def __init__(self, tap: int, hidden: int = 24,
                 extra_dim: int = 4):
        """extra_dim: 抽头之外的额外特征维度(envelope/phase 总数)。"""
        super().__init__()
        self.tap = tap
        self.extra_dim = extra_dim
        self.gru = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        # 把 GRU 末态 + 额外特征拼接后送 FC
        self.fc = nn.Linear(hidden + extra_dim, 2)

    def forward(self, x):                            # x: [B, 2*tap + extra]
        B = x.shape[0]
        iq = x[:, :2 * self.tap].view(B, self.tap, 2)
        mag = torch.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2 + 1e-12).unsqueeze(-1)
        seq = torch.cat([iq, mag], dim=-1)            # [B, tap, 3]
        seq = torch.flip(seq, dims=[1])               # 时间从旧到新
        out, _ = self.gru(seq)
        h = out[:, -1]                                # [B, hidden]
        extra = x[:, 2 * self.tap:]
        return self.fc(torch.cat([h, extra], dim=-1))
