"""实验二:基于 ILA 的 DPD,在固定的(最佳) PA 上训练并联合评估 NMSE/EVM/ACPR。

ILA 流程:
  1. 估计小信号增益 alpha
  2. 训练后逆 G:  y_norm = y/alpha  ->  x   (post-distorter)
  3. 部署:  u = G(x)  ->  z = PA(u),目标 z ≈ alpha*x
"""
import os
import json
import numpy as np
import torch

import config
from data import make_features, complex_to_iq, split_indices, FeatureDataset
from models.mp_gmp import GMP
from trainer import train_model, predict_full, count_params
from metrics import nmse_db, acpr_db, evm_pct
from stats import summarize_runs
from plotting import (plot_psd, plot_am_am, plot_am_pm, plot_constellation,
                      plot_error_hist, plot_convergence, plot_bar)
from experiments.common import (prepare_signal, results_dir, ckpt_path,
                                history_path, make_nn_model)


DPD_MODELS = ['MLP', 'CNN', 'GRU', 'KAN', 'GMP+KAN']


def _train_dpd(name: str, feats_post, labels_post, tr, va,
               gmp_extra=None, pa_name='', seed=0):
    in_dim = feats_post.shape[1]
    tap = config.TAP
    m = make_nn_model(name, in_dim, tap)
    use_extra = (name == 'GMP+KAN')
    if use_extra:
        tr_ds = FeatureDataset(feats_post, labels_post, tr, extra=gmp_extra)
        va_ds = FeatureDataset(feats_post, labels_post, va, extra=gmp_extra)
    else:
        tr_ds = FeatureDataset(feats_post, labels_post, tr)
        va_ds = FeatureDataset(feats_post, labels_post, va)
    cp = ckpt_path(pa_name, 'dpd_ila', name, seed)
    hp = history_path(pa_name, 'dpd_ila', name, seed)
    m, hist = train_model(m, tr_ds, va_ds, use_extra=use_extra,
                          ckpt_path=cp, history_path=hp)
    return m, hist, use_extra


def run_one_seed(pa_name: str, seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    x, y, pa_fn, alpha = prepare_signal(pa_name, seed)
    print(f'    seed={seed}  alpha={alpha:.4f}')
    tap = config.TAP
    target = (alpha * x).astype(np.complex64)
    y_norm = (y / alpha).astype(np.complex64)

    # post-distorter 训练数据:输入 y_norm,标签 x
    feats_post = make_features(y_norm, tap)
    labels_post = complex_to_iq(x)
    tr, va, te = split_indices(len(x), config.TRAIN_FRAC, config.VAL_FRAC, tap)

    results = {}
    histories = {}
    z_traces = {}

    # ---- 0) No-DPD ----
    a_l, a_u = acpr_db(y[te])
    results['NoDPD'] = {
        'NMSE_dB': nmse_db(target[te], y[te]),
        'EVM_%':   evm_pct(target[te], y[te]),
        'ACPR_lower_dB': a_l, 'ACPR_upper_dB': a_u,
        'params': 0,
    }
    z_traces['NoDPD'] = y

    # ---- 1) GMP 作为 DPD(经典强基线) ----
    gmp = GMP(config.GMP_K, config.GMP_Q, config.GMP_LAG)
    gmp.fit(y_norm[:tr[-1] + 1], x[:tr[-1] + 1])
    u_gmp = gmp.predict(x)                        # 当作 predistorter 使用
    z = pa_fn(u_gmp)
    a_l, a_u = acpr_db(z[te])
    results['GMP'] = {
        'NMSE_dB': nmse_db(target[te], z[te]),
        'EVM_%':   evm_pct(target[te], z[te]),
        'ACPR_lower_dB': a_l, 'ACPR_upper_dB': a_u,
        'params': 2 * gmp.coeffs.size,
    }
    z_traces['GMP'] = z
    np.savez(ckpt_path(pa_name, 'dpd_ila', 'GMP', seed).replace('.pt', '.npz'),
             coeffs=gmp.coeffs)

    # GMP+KAN residual 需要的 GMP 主路径输出
    u_main_full = gmp.predict(y_norm)             # 训练域(y_norm)上的 GMP 输出
    gmp_extra = complex_to_iq(u_main_full)

    feats_pre = make_features(x, tap)             # 预失真部署用 x 作为输入
    idx_all = np.arange(len(x))
    u_main_pre = gmp.predict(x)
    gmp_extra_pre = complex_to_iq(u_main_pre)

    # ---- 2) NN DPD(MLP / CNN / GRU / KAN / GMP+KAN) ----
    for name in DPD_MODELS:
        m, hist, use_extra = _train_dpd(name, feats_post, labels_post, tr, va,
                                        gmp_extra=gmp_extra,
                                        pa_name=pa_name, seed=seed)
        histories[name] = hist
        if use_extra:
            u_pred = predict_full(m, feats_pre, idx_all, extra_np=gmp_extra_pre)
        else:
            u_pred = predict_full(m, feats_pre, idx_all)
        z = pa_fn(u_pred)
        a_l, a_u = acpr_db(z[te])
        results[name] = {
            'NMSE_dB': nmse_db(target[te], z[te]),
            'EVM_%':   evm_pct(target[te], z[te]),
            'ACPR_lower_dB': a_l, 'ACPR_upper_dB': a_u,
            'params': count_params(m),
        }
        z_traces[name] = z

    return results, histories, (x, target, te, z_traces)


def run_dpd(pa_name: str, seeds=None):
    seeds = seeds or config.SEEDS
    out_dir = results_dir(pa_name)
    print(f'\n=== Experiment 2: DPD (ILA) on "{pa_name}"  '
          f'({len(seeds)} seeds) ===')

    all_runs: dict[str, dict[str, list[float]]] = {}
    last_extra = None
    last_hist = None

    for seed in seeds:
        results, hist, extra = run_one_seed(pa_name, seed)
        last_extra = (results, extra)
        last_hist = hist
        for name, r in results.items():
            for k, v in r.items():
                all_runs.setdefault(k, {}).setdefault(name, []).append(v)

    # 以 GMP 为参考做配对检验(GMP 是工程上的强基线)
    summary = {}
    for metric in ('NMSE_dB', 'EVM_%', 'ACPR_lower_dB', 'ACPR_upper_dB'):
        summary[metric] = summarize_runs(all_runs[metric], reference='GMP',
                                         alpha=config.ALPHA)
    summary['params'] = {n: int(np.mean(vs)) for n, vs in all_runs['params'].items()}
    summary['seeds'] = list(seeds)
    with open(os.path.join(out_dir, 'e2_dpd_ila.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)

    print(f'  -- summary (mean ± std over {len(seeds)} seeds) --')
    for name in summary['NMSE_dB']:
        nm = summary['NMSE_dB'][name]
        al = summary['ACPR_lower_dB'][name]
        au = summary['ACPR_upper_dB'][name]
        ev = summary['EVM_%'][name]
        sig = ''
        if 'paired_vs_ref' in nm:
            sig = '*' if nm['paired_vs_ref']['significant_holm'] else ''
        print(f'    {name:10s}  NMSE={nm["mean"]:7.2f}±{nm["std"]:.2f} dB  '
              f'EVM={ev["mean"]:5.2f}±{ev["std"]:.2f} %  '
              f'ACPR(L/U)={al["mean"]:.1f}/{au["mean"]:.1f} dB  '
              f'params≈{summary["params"][name]:6d}  {sig}')

    # ---- 可视化(用最后一个 seed) ----
    results_last, (x, target, te, z_traces) = last_extra
    plot_psd({k: v[te] for k, v in z_traces.items()},
             os.path.join(out_dir, 'e2_psd'),
             title=f'PSD after DPD — {pa_name}')
    # AM/AM:对比 NoDPD 与 GMP+KAN 后的线性化结果
    plot_am_am({'NoDPD': (x[te], z_traces['NoDPD'][te]),
                'GMP':   (x[te], z_traces['GMP'][te]),
                'GMP+KAN': (x[te], z_traces['GMP+KAN'][te])},
               os.path.join(out_dir, 'e2_am_am_after_dpd'),
               title=f'AM/AM after DPD — {pa_name}')
    plot_am_pm({'NoDPD': (x[te], z_traces['NoDPD'][te]),
                'GMP+KAN': (x[te], z_traces['GMP+KAN'][te])},
               os.path.join(out_dir, 'e2_am_pm_after_dpd'),
               title=f'AM/PM after DPD — {pa_name}')
    plot_constellation({'target (linear)': target[te][:3000],
                        'NoDPD':           z_traces['NoDPD'][te][:3000],
                        'GMP+KAN':         z_traces['GMP+KAN'][te][:3000]},
                       os.path.join(out_dir, 'e2_constellation'),
                       title=f'Constellation — {pa_name}')
    plot_convergence(last_hist, os.path.join(out_dir, 'e2_convergence'),
                     title=f'DPD training convergence — {pa_name}')
    plot_error_hist({n: target[te] - z_traces[n][te]
                     for n in ('NoDPD', 'GMP', 'KAN', 'GMP+KAN')},
                    os.path.join(out_dir, 'e2_error_hist'),
                    title=f'Linearization error — {pa_name}')

    # NMSE / ACPR bar charts
    mean_n = {n: summary['NMSE_dB'][n]['mean'] for n in summary['NMSE_dB']}
    std_n = {n: summary['NMSE_dB'][n]['std'] for n in summary['NMSE_dB']}
    plot_bar(mean_n, os.path.join(out_dir, 'e2_nmse_bar'),
             ylabel='Linearization NMSE (dB)', err=std_n,
             title=f'NMSE — {pa_name}')
    mean_a = {n: summary['ACPR_lower_dB'][n]['mean'] for n in summary['ACPR_lower_dB']}
    std_a = {n: summary['ACPR_lower_dB'][n]['std'] for n in summary['ACPR_lower_dB']}
    plot_bar(mean_a, os.path.join(out_dir, 'e2_acpr_bar'),
             ylabel='ACPR lower (dB)', err=std_a,
             title=f'ACPR lower — {pa_name}')

    return summary


def run_all(pas=('saleh', 'rapp', 'gmp_pa')):
    out = {}
    for pa in pas:
        out[pa] = run_dpd(pa)
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pa', nargs='+', default=['saleh', 'rapp', 'gmp_pa'])
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    args = p.parse_args()
    for pa in args.pa:
        run_dpd(pa, args.seeds)
