"""频域感知的 KAN-DPD 复合损失。

已发表工作通常把 ACPR 作为评测指标但不纳入训练损失(频段功率积分不可微)。
本模块实现一个可微的邻道功率代理,作为时域 MSE 的频域正则项:

    L_freq(e) = mean_b ||FFT(e_b) ⊙ adj_mask||^2

其中 e_b 是一个连续时间块的误差序列(复数),adj_mask 在主信道置 0、
邻信道置 1。 训练时这一项作为正则项与时域 MSE 加权求和。
"""
import torch
import torch.nn.functional as F


def time_mse(pred_iq: torch.Tensor, target_iq: torch.Tensor) -> torch.Tensor:
    """普通时域 MSE。 输入形状 [B, 2]。"""
    return F.mse_loss(pred_iq, target_iq)


def freq_acpr_surrogate(pred_iq: torch.Tensor,
                        target_iq: torch.Tensor,
                        block_size: int = 256,
                        main_bw: float = 0.20,
                        adj_bw: float = 0.20,
                        fs: float = 1.0) -> torch.Tensor:
    """可微的邻信道功率代理。

    Args:
        pred_iq:    [N, 2]   连续时间序列的预测输出(I,Q)
        target_iq:  [N, 2]   连续时间序列的目标(I,Q)
        block_size: FFT 长度;N 不是 block_size 整数倍则只取前 (N//block_size)*block_size 个样本
        main_bw, adj_bw, fs: 与 metrics.acpr_db 一致的归一化带宽

    Returns:
        scalar 损失:邻信道误差能量(取对数主信道能量做归一化)
    """
    N = pred_iq.shape[0]
    K = N // block_size
    if K == 0:
        # 数据不够一个块就退化到 MSE
        return time_mse(pred_iq, target_iq)
    n_use = K * block_size

    err = pred_iq[:n_use] - target_iq[:n_use]
    err_c = torch.complex(err[:, 0], err[:, 1])             # [n_use]
    err_c = err_c.view(K, block_size)                       # [K, block_size]
    # 加 Hann 窗减少泄漏
    win = torch.hann_window(block_size, periodic=False,
                            dtype=err_c.real.dtype,
                            device=err_c.device)
    spec = torch.fft.fftshift(torch.fft.fft(err_c * win, dim=-1),
                              dim=-1)                       # [K, block_size]
    psd = spec.abs() ** 2                                   # [K, block_size]

    # 频率轴 (归一化)
    freqs = torch.fft.fftshift(
        torch.fft.fftfreq(block_size, d=1.0 / fs)).to(psd.device)
    half_main = main_bw / 2.0
    # 邻信道掩码:|f|∈(half_main, half_main+adj_bw)
    adj_mask = ((freqs.abs() > half_main) &
                (freqs.abs() < half_main + adj_bw)).to(psd.dtype)
    # 主信道掩码:|f| ≤ half_main
    main_mask = (freqs.abs() <= half_main).to(psd.dtype)

    adj_power = (psd * adj_mask).sum(dim=-1).mean()
    main_power = (psd * main_mask).sum(dim=-1).mean() + 1e-12
    # 用相对功率比作为损失(数量级稳定)
    return adj_power / main_power


def combined_dpd_loss(pred_iq: torch.Tensor, target_iq: torch.Tensor,
                      lam: float = 0.0, **freq_kw) -> torch.Tensor:
    """L = MSE_time + lam * L_freq。 lam=0 退化为纯时域。"""
    L = time_mse(pred_iq, target_iq)
    if lam > 0:
        L = L + lam * freq_acpr_surrogate(pred_iq, target_iq, **freq_kw)
    return L
