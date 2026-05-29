"""自动下载 OpenDPD 公开测量数据。

OpenDPD 数据每个数据集是 6 个 CSV(train/val/test 的 input/output 各一个)
+ 1 个 spec.json。 本脚本会下载这些文件,然后合并成 1 个 .npz:

    <name>.npz: {
        x:         complex64[N]   完整复基带输入(train + val + test 顺序拼接)
        y:         complex64[N]   对应 PA 输出
        train_end: int             train 段终止 index (即 val 起始)
        val_end:   int             val 段终止 index (即 test 起始)
        spec:      dict (json bytes)  数据集元信息
    }

下载失败不抛异常,e5 会自动回退到合成代理。
"""
import os
import sys
import json
import io
import urllib.request
import urllib.error
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MEAS_DIR = os.path.join(ROOT, 'data', 'measured')
SOURCES = os.path.join(MEAS_DIR, 'SOURCES.json')


# ---------- 通用下载 ----------
def _human(n: int) -> str:
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f'{n:.1f}{u}'
        n /= 1024
    return f'{n:.1f}TB'


def _get(url: str, *, timeout: int = 90,
         max_retries: int = 2) -> bytes | None:
    """带重试的 HTTP GET。 失败返回 None。"""
    last_err = None
    for attempt in range(1, max_retries + 2):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'kan-dpd/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = int(resp.headers.get('Content-Length', 0) or 0)
                buf = io.BytesIO()
                chunk = 64 * 1024
                done = 0
                while True:
                    b = resp.read(chunk)
                    if not b:
                        break
                    buf.write(b); done += len(b)
                    if total:
                        pct = 100.0 * done / total
                        msg = f'\r    [{pct:5.1f}%] {_human(done)} / {_human(total)}'
                    else:
                        msg = f'\r    {_human(done)}'
                    sys.stdout.write(msg); sys.stdout.flush()
                sys.stdout.write('\n')
                return buf.getvalue()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            sys.stdout.write(f'\n    第 {attempt} 次失败:{e}\n')
    sys.stdout.write(f'    放弃,最后错误:{last_err}\n')
    return None


# ---------- CSV → IQ 数组 ----------
def _iq_csv_to_complex(csv_bytes: bytes) -> np.ndarray:
    """OpenDPD CSV 第一行 'I,Q',之后每行是两个浮点。"""
    arr = np.loadtxt(io.BytesIO(csv_bytes), delimiter=',',
                     skiprows=1, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f'unexpected CSV shape: {arr.shape}')
    return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)


def fetch_one(name: str, base_url: str, files: list[str]) -> bool:
    """下载并合并一个数据集。"""
    out_npz = os.path.join(MEAS_DIR, f'{name}.npz')
    if os.path.exists(out_npz):
        print(f'  [{name}] 已存在,跳过(删除 {out_npz} 可强制重下)')
        return True

    print(f'  [{name}] 开始下载 ({len(files)} 个文件)')
    blobs = {}
    for fn in files:
        url = f'{base_url}/{name}/{fn}'
        print(f'    {fn}')
        data = _get(url)
        if data is None:
            print(f'  [{name}] 中止(网络/URL 问题)')
            return False
        blobs[fn] = data

    try:
        spec = json.loads(blobs['spec.json'].decode('utf-8'))
        x_train = _iq_csv_to_complex(blobs['train_input.csv'])
        y_train = _iq_csv_to_complex(blobs['train_output.csv'])
        x_val   = _iq_csv_to_complex(blobs['val_input.csv'])
        y_val   = _iq_csv_to_complex(blobs['val_output.csv'])
        x_test  = _iq_csv_to_complex(blobs['test_input.csv'])
        y_test  = _iq_csv_to_complex(blobs['test_output.csv'])
    except Exception as e:
        print(f'  [{name}] 解析 CSV 失败:{e}')
        return False

    # 长度对齐
    if not (len(x_train) == len(y_train)
            and len(x_val) == len(y_val)
            and len(x_test) == len(y_test)):
        print(f'  [{name}] input/output 长度不一致')
        return False

    x = np.concatenate([x_train, x_val, x_test])
    y = np.concatenate([y_train, y_val, y_test])
    train_end = len(x_train)
    val_end = train_end + len(x_val)

    np.savez(out_npz, x=x, y=y,
             train_end=np.int64(train_end),
             val_end=np.int64(val_end),
             spec=json.dumps(spec).encode('utf-8'))
    print(f'  [{name}] 保存 {out_npz}  '
          f'(N={len(x)}, train_end={train_end}, val_end={val_end})')
    return True


def fetch_all(only: list[str] | None = None) -> int:
    if not os.path.exists(SOURCES):
        print(f'未找到 {SOURCES}'); return 0
    with open(SOURCES, encoding='utf-8') as fp:
        cfg = json.load(fp)
    base_url = cfg['base_url']
    files = cfg['files_per_dataset']
    os.makedirs(MEAS_DIR, exist_ok=True)
    ok = 0
    for rec in cfg.get('datasets', []):
        if only and rec['name'] not in only:
            continue
        try:
            if fetch_one(rec['name'], base_url, files):
                ok += 1
        except KeyboardInterrupt:
            print('\n  中断'); raise
        except Exception as e:
            print(f'  [{rec["name"]}] 异常:{e}')
    return ok


if __name__ == '__main__':
    only = sys.argv[1:] or None
    print(f'OpenDPD 数据下载  →  {MEAS_DIR}')
    n = fetch_all(only)
    print(f'\n完成:{n} 项成功')
    if n == 0:
        print('如果全部失败,实验五会自动使用合成代理 PA。')
