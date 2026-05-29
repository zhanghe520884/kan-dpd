"""NMSE / ACPR / EVM metrics for complex-baseband PA / DPD evaluation."""
import numpy as np


def nmse_db(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized MSE in dB.

    NMSE = 10 log10( sum|y_true - y_pred|^2 / sum|y_true|^2 )
    """
    err = y_true - y_pred
    num = np.sum(np.abs(err) ** 2)
    den = np.sum(np.abs(y_true) ** 2) + 1e-20
    return float(10.0 * np.log10(num / den + 1e-20))


def acpr_db(y: np.ndarray, fs: float = 1.0,
            main_bw: float = 0.20, adj_bw: float = 0.20,
            gap: float = 0.0, nfft: int = 4096) -> tuple[float, float]:
    """Compute ACPR (lower and upper). Bandwidths are normalized to fs.

    Returns (acpr_lower_db, acpr_upper_db). Negative numbers = better
    linearity. PSD via Welch with Hann window.
    """
    n = len(y)
    seg = min(nfft, n)
    win = np.hanning(seg)
    # Welch
    n_overlap = seg // 2
    step = seg - n_overlap
    psd_acc = np.zeros(seg)
    n_seg = 0
    for start in range(0, n - seg + 1, step):
        chunk = y[start:start + seg] * win
        spec = np.fft.fftshift(np.fft.fft(chunk, n=seg))
        psd_acc += np.abs(spec) ** 2
        n_seg += 1
    psd = psd_acc / max(n_seg, 1)
    freqs = np.fft.fftshift(np.fft.fftfreq(seg, d=1.0 / fs))

    def band_power(f_lo, f_hi):
        mask = (freqs >= f_lo) & (freqs < f_hi)
        return float(np.sum(psd[mask]))

    half_main = main_bw / 2
    main_p = band_power(-half_main, half_main)
    lo_edge = -half_main - gap
    up_edge = half_main + gap
    lower_p = band_power(lo_edge - adj_bw, lo_edge)
    upper_p = band_power(up_edge, up_edge + adj_bw)
    acpr_low = 10.0 * np.log10(lower_p / (main_p + 1e-20) + 1e-20)
    acpr_up = 10.0 * np.log10(upper_p / (main_p + 1e-20) + 1e-20)
    return float(acpr_low), float(acpr_up)


def evm_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """EVM (%) defined as sqrt(mean|err|^2 / mean|y_true|^2) * 100.

    For comparison purposes, this is essentially sqrt(NMSE-linear)*100.
    """
    err = y_true - y_pred
    num = np.mean(np.abs(err) ** 2)
    den = np.mean(np.abs(y_true) ** 2) + 1e-20
    return float(np.sqrt(num / den) * 100.0)


def psd(y: np.ndarray, fs: float = 1.0, nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD in dB. Returns (freqs, psd_db)."""
    n = len(y)
    seg = min(nfft, n)
    win = np.hanning(seg)
    n_overlap = seg // 2
    step = seg - n_overlap
    psd_acc = np.zeros(seg)
    n_seg = 0
    for start in range(0, n - seg + 1, step):
        chunk = y[start:start + seg] * win
        spec = np.fft.fftshift(np.fft.fft(chunk, n=seg))
        psd_acc += np.abs(spec) ** 2
        n_seg += 1
    psd_lin = psd_acc / max(n_seg, 1)
    freqs = np.fft.fftshift(np.fft.fftfreq(seg, d=1.0 / fs))
    psd_db = 10.0 * np.log10(psd_lin / np.max(psd_lin + 1e-20) + 1e-20)
    return freqs, psd_db
