"""实验四(创新点):频域感知 KAN-DPD 损失。

已发表工作虽以 ACPR 为核心评测,但不将 ACPR 纳入训练损失(频段功率积分
不可微)。 本实验在 KAN-DPD 的训练损失中加入可微的
邻信道功率代理项,扫描权重 λ ∈ {0, 0.1, 1.0, 5.0} 并报告 NMSE/EVM/ACPR
的折中。

为节约时间,在 gmp_pa 上、3 个种子内完成,与 e2 中的纯时域 KAN-DPD 形成
对照。
"""
import os
import json
import numpy as np
import torch

import config
from data import (make_features, complex_to_iq, split_indices,
                  FeatureDataset)
from models.kan import KAN
from trainer import train_model, predict_full, count_params
from metrics import nmse_db, acpr_db, evm_pct
from stats import mean_std
from plotting import plot_bar, plot_psd
from experiments.common import (prepare_signal, results_dir, ckpt_path,
                                history_path)


E4_PA = 'gmp_pa'
E4_SEEDS = config.SEEDS[:3]
LAMBDAS = [0.0, 0.1, 1.0, 5.0]
FREQ_BLOCK = 256


def _train_kan_dpd_with_freq(seed: int, lam: float):
    torch.manual_seed(seed); np.random.seed(seed)
    x, y, pa_fn, alpha = prepare_signal(E4_PA, seed)
    target = (alpha * x).astype(np.complex64)
    y_norm = (y / alpha).astype(np.complex64)

    tap = config.TAP
    feats_post = make_features(y_norm, tap)
    labels_post = complex_to_iq(x)
    in_dim = feats_post.shape[1]
    tr, va, te = split_indices(len(x), config.TRAIN_FRAC, config.VAL_FRAC, tap)

    train_ds = FeatureDataset(feats_post, labels_post, tr)
    val_ds = FeatureDataset(feats_post, labels_post, va)

    m = KAN([in_dim, config.KAN_HIDDEN, 2],
            grid_size=config.KAN_GRID,
            spline_order=config.KAN_SPLINE_ORDER)
    tag = f'freq_lam{lam:g}'
    cp = ckpt_path(E4_PA, 'freq_loss', f'KAN_{tag}', seed)
    hp = history_path(E4_PA, 'freq_loss', f'KAN_{tag}', seed)
    m, hist = train_model(m, train_ds, val_ds,
                          ckpt_path=cp, history_path=hp,
                          freq_lambda=lam, freq_block_size=FREQ_BLOCK)

    # 部署:用 x 作输入,送 PA
    feats_pre = make_features(x, tap)
    u = predict_full(m, feats_pre, np.arange(len(x)))
    z = pa_fn(u)
    a_l, a_u = acpr_db(z[te])
    return {
        'NMSE_dB': nmse_db(target[te], z[te]),
        'EVM_%':   evm_pct(target[te], z[te]),
        'ACPR_lower_dB': a_l, 'ACPR_upper_dB': a_u,
        'params': count_params(m),
    }, (x, target, te, z)


def run():
    out_dir = results_dir(E4_PA)
    print(f'\n=== Experiment 4: Spectrum-aware KAN-DPD on "{E4_PA}"  '
          f'({len(E4_SEEDS)} seeds × {len(LAMBDAS)} λ) ===')

    runs: dict[str, dict[str, list[float]]] = {}
    last_z = {}
    for lam in LAMBDAS:
        for seed in E4_SEEDS:
            r, traces = _train_kan_dpd_with_freq(seed, lam)
            for k, v in r.items():
                runs.setdefault(k, {}).setdefault(f'λ={lam:g}', []).append(v)
            if seed == E4_SEEDS[-1]:
                last_z[f'λ={lam:g}'] = traces  # 最后一个 seed 留作绘图

    # 汇总
    summary = {}
    for metric, by_lam in runs.items():
        summary[metric] = {}
        for tag, vals in by_lam.items():
            m, s = mean_std(vals)
            summary[metric][tag] = {'mean': m, 'std': s, 'values': vals}
    summary['seeds'] = list(E4_SEEDS)
    summary['lambdas'] = LAMBDAS
    summary['block_size'] = FREQ_BLOCK
    with open(os.path.join(out_dir, 'e4_freq_loss.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)

    # 屏幕汇总
    print(f'  -- summary (mean ± std over {len(E4_SEEDS)} seeds) --')
    print(f'    {"λ":8s}  {"NMSE_dB":>14s}  {"EVM_%":>10s}  {"ACPR_L":>8s}  {"ACPR_U":>8s}')
    for tag in [f'λ={l:g}' for l in LAMBDAS]:
        n = summary['NMSE_dB'][tag]
        e = summary['EVM_%'][tag]
        al = summary['ACPR_lower_dB'][tag]
        au = summary['ACPR_upper_dB'][tag]
        print(f'    {tag:8s}  {n["mean"]:6.2f}±{n["std"]:.2f}  '
              f'{e["mean"]:5.2f}±{e["std"]:.2f}  '
              f'{al["mean"]:6.2f}    {au["mean"]:6.2f}')

    # PSD 对比图(用最后一个 seed 的预测)
    if last_z:
        first_tag = list(last_z.keys())[0]
        _, _, te, _ = last_z[first_tag]
        plot_psd({tag: z[te] for tag, (x, t, te, z) in last_z.items()},
                 os.path.join(out_dir, 'e4_psd_vs_lambda'),
                 title=f'PSD: KAN-DPD vs frequency-loss weight λ ({E4_PA})')

    # NMSE / ACPR 折线对比柱状图
    nmse_mean = {tag: summary['NMSE_dB'][tag]['mean'] for tag in summary['NMSE_dB']}
    nmse_std = {tag: summary['NMSE_dB'][tag]['std'] for tag in summary['NMSE_dB']}
    plot_bar(nmse_mean, os.path.join(out_dir, 'e4_nmse_vs_lambda'),
             ylabel='Linearization NMSE (dB)', err=nmse_std,
             title=f'NMSE vs freq-loss weight ({E4_PA})')
    acpr_mean = {tag: summary['ACPR_lower_dB'][tag]['mean']
                 for tag in summary['ACPR_lower_dB']}
    acpr_std = {tag: summary['ACPR_lower_dB'][tag]['std']
                for tag in summary['ACPR_lower_dB']}
    plot_bar(acpr_mean, os.path.join(out_dir, 'e4_acpr_vs_lambda'),
             ylabel='ACPR lower (dB)', err=acpr_std,
             title=f'ACPR vs freq-loss weight ({E4_PA})')
    return summary


if __name__ == '__main__':
    run()
