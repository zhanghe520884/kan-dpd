"""把 OpenDPD 的 .mat 数据转换为本仓库约定的 .npz 格式。

用法:
    python scripts/convert_opendpd.py path/to/DPA_100MHz.mat
"""
import os
import sys
import numpy as np


def convert(path_mat: str):
    try:
        from scipy.io import loadmat
    except ImportError:
        print('请先安装 scipy:  pip install scipy')
        sys.exit(1)

    m = loadmat(path_mat)
    # OpenDPD 一般用 'x' / 'y' 或 'input_iq' / 'output_iq' 这类命名
    # 这里给一个鲁棒的查找器
    def find(key_candidates):
        for k in key_candidates:
            for mk in m:
                if k.lower() in mk.lower():
                    return np.asarray(m[mk]).squeeze()
        return None

    x = find(['x_in', 'input', 'tx', 'x'])
    y = find(['y_out', 'output', 'rx', 'y'])
    if x is None or y is None:
        print(f'未在 {path_mat} 中找到 x/y, 实际 keys: {list(m.keys())}')
        sys.exit(2)
    x = x.astype(np.complex64).ravel()
    y = y.astype(np.complex64).ravel()
    if len(x) != len(y):
        print(f'WARN: 长度不一致 len(x)={len(x)} len(y)={len(y)}, 取最短')
        L = min(len(x), len(y))
        x = x[:L]; y = y[:L]
    base = os.path.splitext(os.path.basename(path_mat))[0]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                       'data', 'measured', f'{base}.npz')
    np.savez(out, x=x, y=y)
    print(f'已写入 {out}  (N={len(x)})')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python scripts/convert_opendpd.py path/to/DPA_xxx.mat')
        sys.exit(1)
    for p in sys.argv[1:]:
        convert(p)
