import numpy as np

def gen_ofdm_like(n_samples: int, n_subcarriers: int = 256, osr: int = 4,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate an OFDM-like complex-baseband signal with high PAPR.

    Returns complex64 array of length approximately n_samples. The signal
    is normalized to unit average power.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    block_len = n_subcarriers * osr
    n_blocks = int(np.ceil(n_samples / block_len))

    out = np.empty(n_blocks * block_len, dtype=np.complex64)
    for b in range(n_blocks):
        # QPSK symbols on active subcarriers, zero-pad to osr*N
        bits = rng.integers(0, 4, size=n_subcarriers)
        symbols = np.exp(1j * (np.pi / 4 + bits * np.pi / 2)).astype(np.complex64)
        # Place symbols on the central N_subcarriers bins of an osr*N FFT
        spec = np.zeros(block_len, dtype=np.complex64)
        half = n_subcarriers // 2
        spec[1:half + 1] = symbols[:half]
        spec[-half:] = symbols[half:]
        block = np.fft.ifft(spec) * np.sqrt(block_len)
        out[b * block_len:(b + 1) * block_len] = block.astype(np.complex64)

    out = out[:n_samples]
    # Normalize average power to 1
    out = out / np.sqrt(np.mean(np.abs(out) ** 2) + 1e-12)
    return out.astype(np.complex64)


def apply_backoff(x: np.ndarray, backoff_db: float) -> np.ndarray:
    """Scale signal so its average power is `backoff_db` below 0 dBFS."""
    scale = 10 ** (-backoff_db / 20.0)
    return (x * scale).astype(np.complex64)
