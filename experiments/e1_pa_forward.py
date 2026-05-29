"""实验一:PA 前向建模 + 单指标(NMSE) 排雷 + 多种子统计。

对每个 PA(saleh / rapp / gmp_pa),用 5 个随机种子训练
GMP / MLP / CNN / GRU / KAN / GMP+KAN(残差),
比较测试集 NMSE_dB,并保存最优 ckpt 与训练曲线。
"""
import os
import json
import numpy as np
import torch

import config
from data import make_features, complex_to_iq, split_indices, feature_size, FeatureDataset
from models.mp_gmp import MemoryPolynomial, GMP
from trainer import train_model, predict_full, count_params
from metrics import nmse_db, evm_pct
from stats import summarize_runs, mean_std
from plotting import (plot_am_am, plot_am_pm, plot_convergence,
                      plot_error_hist, plot_bar)
from experiments.common import (prepare_signal, results_dir, ckpt_path,
                                history_path, make_nn_model)


NN_MODELS = ['MLP', 'CNN', 'GRU', 'KAN', 'GMP+KAN']


def run_one_seed(pa_name: str, seed: int, n_samples: int | None = None):
    """单次 seed 训练并返回结果字典。"""
    torch.manual_seed(seed); np.random.seed(seed)
    x, y, _, alpha = prepare_signal(pa_name, seed, n_samples)
    tap = config.TAP
    feats = make_features(x, tap, with_envelope=True, with_phase=True)
    labels = complex_to_iq(y)
    in_dim = feats.shape[1]
    tr, va, te = split_indices(len(x), config.TRAIN_FRAC, config.VAL_FRAC, tap)

    results = {}
    histories = {}

    # ---- MP / GMP(LS) ----
    mp = MemoryPolynomial(config.MP_K, config.MP_Q).fit(x[:tr[-1] + 1], y[:tr[-1] + 1])
    y_pred_mp = mp.predict(x)
    results['MP'] = {
        'NMSE_dB': nmse_db(y[te], y_pred_mp[te]),
        'EVM_%':   evm_pct(y[te], y_pred_mp[te]),
        'params':  2 * mp.coeffs.size,
    }
    # MP 系数保存
    np.savez(ckpt_path(pa_name, 'pa_fwd', 'MP', seed).replace('.pt', '.npz'),
             coeffs=mp.coeffs)

    gmp = GMP(config.GMP_K, config.GMP_Q, config.GMP_LAG)
    gmp.fit(x[:tr[-1] + 1], y[:tr[-1] + 1])
    y_pred_gmp = gmp.predict(x)
    results['GMP'] = {
        'NMSE_dB': nmse_db(y[te], y_pred_gmp[te]),
        'EVM_%':   evm_pct(y[te], y_pred_gmp[te]),
        'params':  2 * gmp.coeffs.size,
    }
    np.savez(ckpt_path(pa_name, 'pa_fwd', 'GMP', seed).replace('.pt', '.npz'),
             coeffs=gmp.coeffs)

    # ---- NN 基线与 KAN ----
    train_ds = FeatureDataset(feats, labels, tr)
    val_ds = FeatureDataset(feats, labels, va)
    gmp_iq = complex_to_iq(y_pred_gmp)  # GMP+KAN 主路径

    for name in NN_MODELS:
        m = make_nn_model(name, in_dim, tap)
        use_extra = (name == 'GMP+KAN')
        if use_extra:
            tr_ds = FeatureDataset(feats, labels, tr, extra=gmp_iq)
            va_ds = FeatureDataset(feats, labels, va, extra=gmp_iq)
        else:
            tr_ds, va_ds = train_ds, val_ds
        cp = ckpt_path(pa_name, 'pa_fwd', name, seed)
        hp = history_path(pa_name, 'pa_fwd', name, seed)
        m, hist = train_model(m, tr_ds, va_ds, use_extra=use_extra,
                              ckpt_path=cp, history_path=hp)
        histories[name] = hist
        extra_np = gmp_iq if use_extra else None
        pred = predict_full(m, feats, te, extra_np=extra_np)
        results[name] = {
            'NMSE_dB': nmse_db(y[te], pred),
            'EVM_%':   evm_pct(y[te], pred),
            'params':  count_params(m),
        }

    return results, histories, (x, y, te, y_pred_gmp)


def run_pa_forward(pa_name: str, seeds: list[int] | None = None):
    seeds = seeds or config.SEEDS
    out_dir = results_dir(pa_name)
    print(f'\n=== Experiment 1: PA forward modeling on "{pa_name}"  '
          f'({len(seeds)} seeds) ===')

    all_runs: dict[str, dict[str, list[float]]] = {}
    last_hist = None
    last_extra = None

    for seed in seeds:
        print(f'  seed={seed}')
        results, hist, extra = run_one_seed(pa_name, seed)
        last_hist = hist
        last_extra = (results, extra)
        for name, r in results.items():
            for k, v in r.items():
                all_runs.setdefault(k, {}).setdefault(name, []).append(v)

    # 汇总(以 MP 为参照)
    summary_nmse = summarize_runs(all_runs['NMSE_dB'], reference='MP',
                                  alpha=config.ALPHA)
    summary_evm = summarize_runs(all_runs['EVM_%'], reference='MP',
                                 alpha=config.ALPHA)
    summary = {
        'NMSE_dB': summary_nmse,
        'EVM_%':   summary_evm,
        'params':  {n: int(np.mean(vs)) for n, vs in all_runs['params'].items()},
        'seeds':   list(seeds),
    }
    with open(os.path.join(out_dir, 'e1_pa_forward.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)

    # 屏幕汇总
    print(f'  -- summary (NMSE_dB, mean ± std over {len(seeds)} seeds) --')
    for name, s in summary_nmse.items():
        msg = f'    {name:10s}  NMSE={s["mean"]:7.2f} ± {s["std"]:.2f} dB'
        msg += f"   params≈{summary['params'][name]:6d}"
        if 'paired_vs_ref' in s:
            sig = '*' if s['paired_vs_ref']['significant_holm'] else ' '
            msg += f'   p={s["paired_vs_ref"]["p_value"]:.3g}{sig}'
        print(msg)

    # ---- 图(用最后一个 seed 的预测做可视化) ----
    results_last, (x, y, te, y_pred_gmp) = last_extra
    plot_am_am({'true': (x[te], y[te])},
               os.path.join(out_dir, 'e1_am_am'),
               title=f'AM/AM — {pa_name} (test)')
    plot_am_pm({'true': (x[te], y[te])},
               os.path.join(out_dir, 'e1_am_pm'),
               title=f'AM/PM — {pa_name} (test)')
    plot_convergence(last_hist, os.path.join(out_dir, 'e1_convergence'),
                     title=f'Validation MSE convergence — {pa_name}')

    # 误差直方图(对最后一个 seed)
    errs = {}
    # MP / GMP
    from models.mp_gmp import MemoryPolynomial as _MP, GMP as _GMP
    mp = _MP(config.MP_K, config.MP_Q).fit(x[:te[0]], y[:te[0]])
    errs['MP'] = y[te] - mp.predict(x)[te]
    errs['GMP'] = y[te] - y_pred_gmp[te]
    plot_error_hist(errs, os.path.join(out_dir, 'e1_error_hist_linear_baselines'),
                    title=f'Forward modeling error — {pa_name}')

    # NMSE 柱状图
    mean_d = {n: summary_nmse[n]['mean'] for n in summary_nmse}
    std_d = {n: summary_nmse[n]['std'] for n in summary_nmse}
    plot_bar(mean_d, os.path.join(out_dir, 'e1_nmse_bar'),
             ylabel='NMSE (dB)', err=std_d,
             title=f'PA forward NMSE — {pa_name}')

    return summary


def run_all(pas=('saleh', 'rapp', 'gmp_pa')):
    out = {}
    for pa in pas:
        out[pa] = run_pa_forward(pa)
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pa', nargs='+', default=['saleh', 'rapp', 'gmp_pa'])
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    args = p.parse_args()
    for pa in args.pa:
        run_pa_forward(pa, args.seeds)
