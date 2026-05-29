"""测量数据接入层 + "MeasuredPA-like" 合成代理。

设计原则:
  * 真实数据走 .npz(详见 data/measured/README.md)
  * 没有真实数据时,提供一个比 pa_models.gmp_pa 更复杂的合成代理:
      - 更高阶非线性(K 到 9)
      - 加入慢热漂移(blockwise gain 抖动)
      - 加入轻微 I/Q 不平衡
"""
import os
import numpy as np

from signal_gen import gen_ofdm_like, apply_backoff
import config


MEAS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'measured')


def list_available() -> list[str]:
    """列出 data/measured/ 下可用的 .npz 数据集名。"""
    if not os.path.isdir(MEAS_DIR):
        return []
    return sorted(f for f in os.listdir(MEAS_DIR)
                  if f.endswith('.npz') and not f.startswith('_'))


def load_measured(name: str) -> dict:
    """加载一个 .npz 测量数据。

    返回 dict:
        x:         complex64 [N]
        y:         complex64 [N]
        train_end: int       OpenDPD 官方 train 段终止 index(None 表示无)
        val_end:   int       OpenDPD 官方 val 段终止 index(None 表示无)
    """
    path = os.path.join(MEAS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'未找到 {path};请按 README 下载 OpenDPD 数据')
    z = np.load(path, allow_pickle=False)
    x = z['x'].astype(np.complex64)
    y = z['y'].astype(np.complex64)
    assert len(x) == len(y), 'x 和 y 长度不一致'
    out = {'x': x, 'y': y, 'train_end': None, 'val_end': None}
    if 'train_end' in z.files:
        out['train_end'] = int(z['train_end'])
    if 'val_end' in z.files:
        out['val_end'] = int(z['val_end'])
    return out


# ---------------------------------------------------------------------------
# 合成代理:模拟一个"难于仿真但易于评测"的 PA
# ---------------------------------------------------------------------------

def _build_measured_proxy_coeffs(seed: int = 31):
    """更复杂的 GMP 系数集,模拟真实 GaN PA 的高阶 + 强交叉项。"""
    rng = np.random.default_rng(seed)
    aligned = []
    for k in [1, 3, 5, 7, 9]:
        for q in [0, 1, 2, 3]:
            # 偶数阶不存在 (RF PA 复基带模型)
            mag = (0.6 if k == 1 else 0.22) * (0.65 ** q)
            c = mag * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
            aligned.append((k, q, c))
    cross = []
    for k in [3, 5, 7]:
        for q in [0, 1, 2]:
            for m in [1, 2, 3]:
                c = 0.08 * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
                cross.append((k, q, m, c))
    return aligned, cross


_MEAS_ALIGNED, _MEAS_CROSS = _build_measured_proxy_coeffs(seed=31)
_MEAS_LIN_GAIN = _MEAS_ALIGNED[0][2]
# I/Q 不平衡(幅度 5%,相位 3 度)
_IQ_AMP = 1.05
_IQ_PHASE = np.deg2rad(3.0)


def measured_proxy_pa(x: np.ndarray, *, with_drift: bool = True,
                      drift_amp_db: float = 0.3,
                      drift_period: int = 8000) -> np.ndarray:
    """合成代理 PA。 比 pa_models.gmp_pa 更复杂(高阶 + 漂移 + I/Q 不平衡)。"""
    N = len(x)
    y = np.zeros(N, dtype=np.complex64)

    # 对齐 + 交叉 GMP 项
    for k, q, coef in _MEAS_ALIGNED:
        xq = np.zeros_like(x)
        if q == 0:
            xq[:] = x
        else:
            xq[q:] = x[:-q]
        y += (coef * xq * np.abs(xq) ** (k - 1)).astype(np.complex64)
    for k, q, m, coef in _MEAS_CROSS:
        xq = np.zeros_like(x)
        xqm = np.zeros_like(x)
        if q == 0:
            xq[:] = x
        else:
            xq[q:] = x[:-q]
        xqm[q + m:] = x[:-(q + m)]
        y += (coef * xq * np.abs(xqm) ** (k - 1)).astype(np.complex64)

    y = y / _MEAS_LIN_GAIN

    # 慢热漂移(以 block 为单位)
    if with_drift:
        n_blocks = N // drift_period + 1
        rng = np.random.default_rng(7)
        drift_db = rng.normal(scale=drift_amp_db, size=n_blocks)
        drift = 10 ** (np.repeat(drift_db, drift_period)[:N] / 20.0)
        y = y * drift.astype(np.complex64)

    # I/Q 不平衡
    y_real = y.real * _IQ_AMP * np.cos(_IQ_PHASE) - y.imag * np.sin(_IQ_PHASE)
    y_imag = y.real * _IQ_AMP * np.sin(_IQ_PHASE) + y.imag * np.cos(_IQ_PHASE)
    return (y_real + 1j * y_imag).astype(np.complex64)


def synthesize_measured_proxy(n_samples: int | None = None,
                              target_rms: float = 0.12,
                              seed: int = 9999):
    """生成一段"实测代理"的 (x, y) 数据,默认 N_SAMPLES * 2 长度。"""
    n = n_samples or 2 * config.N_SAMPLES
    rng = np.random.default_rng(seed)
    x = gen_ofdm_like(n, config.N_SUBCARRIERS, config.OSR, rng)
    x = apply_backoff(x, config.SIGNAL_BACKOFF_DB)
    x = (x / np.sqrt(np.mean(np.abs(x) ** 2)) * target_rms).astype(np.complex64)
    y = measured_proxy_pa(x)
    return x, y


def _try_auto_download():
    """首次调用时尝试自动下载测量数据。 失败静默,由调用者决定回退。"""
    try:
        import importlib.util, sys
        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(here, 'scripts', 'download_data.py')
        if not os.path.exists(script):
            return
        spec = importlib.util.spec_from_file_location('download_data', script)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['download_data'] = mod
        spec.loader.exec_module(mod)
        print('[data_measured] data/measured/ 为空,尝试自动下载 OpenDPD 数据...')
        n = mod.fetch_all()
        if n == 0:
            print('[data_measured] 自动下载未成功,将使用合成代理 PA')
    except Exception as e:
        print(f'[data_measured] 自动下载流程异常:{e},退回合成代理')


def get_dataset(name: str | None = None, offline: bool = False) -> tuple[dict, str]:
    """统一入口:有真实数据用真实数据,否则回退到合成代理。

    Args:
        name:
            * 文件名(如 'DPA_200MHz.npz'):指定具体数据集
            * None:有真实数据就用第一个,无则触发自动下载,再无则用合成代理
            * '__proxy__':强制用合成代理(不联网,不读硬盘)
        offline:
            * True:不尝试自动下载,有本地数据用本地,无则用合成代理

    Returns:
        (info_dict, tag):
            info_dict = {x, y, train_end, val_end}
            tag       = 数据集名 (如 'DPA_200MHz.npz') 或 '__proxy__'
    """
    if name == '__proxy__':
        x, y = synthesize_measured_proxy()
        return {'x': x, 'y': y, 'train_end': None, 'val_end': None}, '__proxy__'

    avail = list_available()
    if not avail and not offline:
        _try_auto_download()
        avail = list_available()

    if not avail:
        x, y = synthesize_measured_proxy()
        return {'x': x, 'y': y, 'train_end': None, 'val_end': None}, '__proxy__'

    # 默认优先选 DPA_200MHz(e5 主基线),其次按字母序
    preferred = ['DPA_200MHz.npz', 'DPA_160MHz.npz', 'APA_200MHz.npz']
    if name in avail:
        pick = name
    else:
        pick = next((p for p in preferred if p in avail), avail[0])
    return load_measured(pick), pick
