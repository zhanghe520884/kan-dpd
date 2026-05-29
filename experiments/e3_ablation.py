"""实验三:消融实验(围绕 DPD 任务)。

至少 5 类消融:
  A. 样条网格数 grid ∈ {5,7,9}
  B. 时延抽头数 tap ∈ {3,5,7,9}
  C. 输入特征类型 (基础 I/Q  vs  +|x|,|x|^3  vs  +sinθ,cosθ)
  D. 纯 KAN vs hybrid (GMP+KAN residual)
  E. KAN 边函数 LUT 离散化(简易量化)
此外补充 GMP 阶/记忆深度的扫描(K∈{5,7,9}, Q∈{3,5,7})。
为节约时间,默认在单一 PA(gmp_pa) 上、2 个种子内完成。
"""
import os
import json
import numpy as np
import torch

import config
from data import make_features, complex_to_iq, split_indices, FeatureDataset
from models.mp_gmp import GMP
from models.kan import KAN, KANLinear
from models.hybrid import GMP_KAN_Residual
from trainer import train_model, predict_full, count_params
from metrics import nmse_db, acpr_db, evm_pct
from plotting import plot_bar
from experiments.common import prepare_signal, results_dir, ckpt_path, history_path


ABLATION_PA = 'gmp_pa'
ABLATION_SEEDS = config.SEEDS[:2]   # 节约时间


def _prepare(seed: int, tap: int, with_envelope=True, with_phase=True):
    """共用的数据准备步骤(post-distorter 视角)。"""
    x, y, pa_fn, alpha = prepare_signal(ABLATION_PA, seed)
    target = (alpha * x).astype(np.complex64)
    y_norm = (y / alpha).astype(np.complex64)
    feats_post = make_features(y_norm, tap, with_envelope, with_phase)
    feats_pre = make_features(x, tap, with_envelope, with_phase)
    labels_post = complex_to_iq(x)
    tr, va, te = split_indices(len(x), config.TRAIN_FRAC, config.VAL_FRAC, tap)
    return dict(x=x, y=y, y_norm=y_norm, target=target, pa_fn=pa_fn,
                feats_post=feats_post, feats_pre=feats_pre,
                labels_post=labels_post, tr=tr, va=va, te=te)


def _evaluate(model, dat, use_extra=False, extra_pre=None):
    idx_all = np.arange(len(dat['x']))
    u = predict_full(model, dat['feats_pre'], idx_all, extra_np=extra_pre)
    z = dat['pa_fn'](u)
    te = dat['te']
    a_l, a_u = acpr_db(z[te])
    return {
        'NMSE_dB': nmse_db(dat['target'][te], z[te]),
        'EVM_%':   evm_pct(dat['target'][te], z[te]),
        'ACPR_lower_dB': a_l, 'ACPR_upper_dB': a_u,
    }


# ---- A) 样条网格数 ----
def ablation_grid():
    print('\n[Ablation A] KAN spline grid size')
    out = {}
    tap = config.TAP
    for grid in [5, 7, 9]:
        nmse_list = []
        for seed in ABLATION_SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            dat = _prepare(seed, tap)
            in_dim = dat['feats_post'].shape[1]
            m = KAN([in_dim, config.KAN_HIDDEN, 2], grid_size=grid,
                    spline_order=config.KAN_SPLINE_ORDER)
            tr_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['tr'])
            va_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['va'])
            cp = ckpt_path(ABLATION_PA, 'ablation_grid', f'KAN_g{grid}', seed)
            hp = history_path(ABLATION_PA, 'ablation_grid', f'KAN_g{grid}', seed)
            m, _ = train_model(m, tr_ds, va_ds, ckpt_path=cp, history_path=hp)
            r = _evaluate(m, dat)
            nmse_list.append(r['NMSE_dB'])
        out[f'grid={grid}'] = {'NMSE_dB_mean': float(np.mean(nmse_list)),
                               'NMSE_dB_std': float(np.std(nmse_list, ddof=1)
                                                    if len(nmse_list) > 1 else 0.0),
                               'values': nmse_list}
        print(f'    grid={grid}  NMSE = {np.mean(nmse_list):.2f} ± '
              f'{out[f"grid={grid}"]["NMSE_dB_std"]:.2f} dB')
    return out


# ---- B) 时延抽头数 ----
def ablation_tap():
    print('\n[Ablation B] Time-delay tap count')
    out = {}
    for tap in [3, 5, 7, 9]:
        nmse_list = []
        for seed in ABLATION_SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            dat = _prepare(seed, tap)
            in_dim = dat['feats_post'].shape[1]
            m = KAN([in_dim, config.KAN_HIDDEN, 2],
                    grid_size=config.KAN_GRID,
                    spline_order=config.KAN_SPLINE_ORDER)
            tr_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['tr'])
            va_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['va'])
            cp = ckpt_path(ABLATION_PA, 'ablation_tap', f'KAN_t{tap}', seed)
            hp = history_path(ABLATION_PA, 'ablation_tap', f'KAN_t{tap}', seed)
            m, _ = train_model(m, tr_ds, va_ds, ckpt_path=cp, history_path=hp)
            r = _evaluate(m, dat)
            nmse_list.append(r['NMSE_dB'])
        out[f'tap={tap}'] = {'NMSE_dB_mean': float(np.mean(nmse_list)),
                             'NMSE_dB_std': float(np.std(nmse_list, ddof=1)
                                                  if len(nmse_list) > 1 else 0.0),
                             'values': nmse_list}
        print(f'    tap={tap}  NMSE = {np.mean(nmse_list):.2f} dB')
    return out


# ---- C) 输入特征类型 ----
def ablation_features():
    print('\n[Ablation C] Input feature variants')
    out = {}
    variants = [
        ('IQ_only',   dict(with_envelope=False, with_phase=False)),
        ('IQ+env',    dict(with_envelope=True, with_phase=False)),
        ('IQ+env+phase', dict(with_envelope=True, with_phase=True)),
    ]
    for vname, kw in variants:
        nmse_list = []
        for seed in ABLATION_SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            dat = _prepare(seed, config.TAP, **kw)
            in_dim = dat['feats_post'].shape[1]
            m = KAN([in_dim, config.KAN_HIDDEN, 2],
                    grid_size=config.KAN_GRID,
                    spline_order=config.KAN_SPLINE_ORDER)
            tr_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['tr'])
            va_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['va'])
            cp = ckpt_path(ABLATION_PA, 'ablation_feat', f'KAN_{vname}', seed)
            hp = history_path(ABLATION_PA, 'ablation_feat', f'KAN_{vname}', seed)
            m, _ = train_model(m, tr_ds, va_ds, ckpt_path=cp, history_path=hp)
            r = _evaluate(m, dat)
            nmse_list.append(r['NMSE_dB'])
        out[vname] = {'NMSE_dB_mean': float(np.mean(nmse_list)),
                      'NMSE_dB_std': float(np.std(nmse_list, ddof=1)
                                           if len(nmse_list) > 1 else 0.0),
                      'values': nmse_list}
        print(f'    {vname:18s} NMSE = {np.mean(nmse_list):.2f} dB')
    return out


# ---- D) 纯 KAN vs hybrid ----
def ablation_hybrid():
    print('\n[Ablation D] Pure KAN vs GMP+KAN residual')
    out = {}
    tap = config.TAP
    for tag in ['pure_KAN', 'GMP+KAN']:
        nmse_list = []
        for seed in ABLATION_SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            dat = _prepare(seed, tap)
            in_dim = dat['feats_post'].shape[1]
            tr_ds_base = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['tr'])
            va_ds_base = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['va'])
            if tag == 'pure_KAN':
                m = KAN([in_dim, config.KAN_HIDDEN, 2],
                        grid_size=config.KAN_GRID,
                        spline_order=config.KAN_SPLINE_ORDER)
                cp = ckpt_path(ABLATION_PA, 'ablation_hybrid', 'pure_KAN', seed)
                hp = history_path(ABLATION_PA, 'ablation_hybrid', 'pure_KAN', seed)
                m, _ = train_model(m, tr_ds_base, va_ds_base,
                                   ckpt_path=cp, history_path=hp)
                r = _evaluate(m, dat)
            else:
                gmp = GMP(config.GMP_K, config.GMP_Q, config.GMP_LAG)
                gmp.fit(dat['y_norm'][:dat['tr'][-1] + 1],
                        dat['x'][:dat['tr'][-1] + 1])
                main = gmp.predict(dat['y_norm'])
                extra = complex_to_iq(main)
                m = GMP_KAN_Residual(in_dim, kan_hidden=config.KAN_HIDDEN,
                                     grid_size=config.KAN_GRID,
                                     spline_order=config.KAN_SPLINE_ORDER)
                tr_ds = FeatureDataset(dat['feats_post'], dat['labels_post'],
                                       dat['tr'], extra=extra)
                va_ds = FeatureDataset(dat['feats_post'], dat['labels_post'],
                                       dat['va'], extra=extra)
                cp = ckpt_path(ABLATION_PA, 'ablation_hybrid', 'GMP+KAN', seed)
                hp = history_path(ABLATION_PA, 'ablation_hybrid', 'GMP+KAN', seed)
                m, _ = train_model(m, tr_ds, va_ds, use_extra=True,
                                   ckpt_path=cp, history_path=hp)
                main_pre = gmp.predict(dat['x'])
                extra_pre = complex_to_iq(main_pre)
                r = _evaluate(m, dat, use_extra=True, extra_pre=extra_pre)
            nmse_list.append(r['NMSE_dB'])
        out[tag] = {'NMSE_dB_mean': float(np.mean(nmse_list)),
                    'NMSE_dB_std': float(np.std(nmse_list, ddof=1)
                                         if len(nmse_list) > 1 else 0.0),
                    'values': nmse_list}
        print(f'    {tag:10s}  NMSE = {np.mean(nmse_list):.2f} dB')
    return out


# ---- E) KAN 边函数 LUT 量化 ----
def _quantize_kan_(model: torch.nn.Module, bits: int):
    """对每条 KAN 边的样条系数做均匀量化(模拟 LUT 离散化)。 in-place。"""
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, KANLinear):
                w = mod.spline_weight
                lo, hi = w.min().item(), w.max().item()
                if hi == lo:
                    continue
                levels = 2 ** bits - 1
                step = (hi - lo) / levels
                wq = torch.round((w - lo) / step) * step + lo
                w.copy_(wq)


def ablation_quant():
    print('\n[Ablation E] KAN spline LUT-quantization')
    out = {}
    tap = config.TAP
    base_nmse_per_seed = []
    for seed in ABLATION_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        dat = _prepare(seed, tap)
        in_dim = dat['feats_post'].shape[1]
        # 训练一次浮点 KAN
        m = KAN([in_dim, config.KAN_HIDDEN, 2],
                grid_size=config.KAN_GRID,
                spline_order=config.KAN_SPLINE_ORDER)
        tr_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['tr'])
        va_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['va'])
        cp = ckpt_path(ABLATION_PA, 'ablation_quant', 'KAN_fp', seed)
        hp = history_path(ABLATION_PA, 'ablation_quant', 'KAN_fp', seed)
        m, _ = train_model(m, tr_ds, va_ds, ckpt_path=cp, history_path=hp)
        base = _evaluate(m, dat)
        base_nmse_per_seed.append(base['NMSE_dB'])
        out.setdefault('fp', []).append(base['NMSE_dB'])
        # 不同位宽
        import copy
        for bits in [12, 10, 8, 6, 4]:
            mq = copy.deepcopy(m)
            _quantize_kan_(mq, bits)
            r = _evaluate(mq, dat)
            out.setdefault(f'q{bits}', []).append(r['NMSE_dB'])
    summary = {}
    for k, vs in out.items():
        summary[k] = {'NMSE_dB_mean': float(np.mean(vs)),
                      'NMSE_dB_std': float(np.std(vs, ddof=1) if len(vs) > 1 else 0.0),
                      'values': vs}
        print(f'    {k:5s}  NMSE = {summary[k]["NMSE_dB_mean"]:.2f} ± '
              f'{summary[k]["NMSE_dB_std"]:.2f} dB')
    return summary


# ---- F) GMP K/Q 扫描 ----
def sweep_gmp():
    print('\n[Sweep] GMP nonlinear order K and memory Q')
    out = {}
    for K in [5, 7, 9]:
        for Q in [3, 5, 7]:
            nmse_list = []
            for seed in ABLATION_SEEDS:
                np.random.seed(seed)
                dat = _prepare(seed, config.TAP)
                g = GMP(K, Q, config.GMP_LAG)
                g.fit(dat['y_norm'][:dat['tr'][-1] + 1],
                      dat['x'][:dat['tr'][-1] + 1])
                u = g.predict(dat['x'])
                z = dat['pa_fn'](u)
                nmse_list.append(nmse_db(dat['target'][dat['te']], z[dat['te']]))
            tag = f'K={K},Q={Q}'
            out[tag] = {'NMSE_dB_mean': float(np.mean(nmse_list)),
                        'NMSE_dB_std': float(np.std(nmse_list, ddof=1)
                                             if len(nmse_list) > 1 else 0.0),
                        'values': nmse_list}
            print(f'    {tag:10s}  NMSE = {np.mean(nmse_list):.2f} dB')
    return out


def run_all():
    out = {
        'A_grid':   ablation_grid(),
        'B_tap':    ablation_tap(),
        'C_feature': ablation_features(),
        'D_hybrid': ablation_hybrid(),
        'E_quant':  ablation_quant(),
        'F_gmp_sweep': sweep_gmp(),
        'pa':       ABLATION_PA,
        'seeds':    ABLATION_SEEDS,
    }
    rd = results_dir(ABLATION_PA)
    with open(os.path.join(rd, 'e3_ablation.json'), 'w') as fp:
        json.dump(out, fp, indent=2)

    # 绘制四张柱状图
    def _bar(d_key, tag, ylabel):
        d = out[d_key]
        means = {k: v['NMSE_dB_mean'] for k, v in d.items()}
        stds = {k: v['NMSE_dB_std'] for k, v in d.items()}
        plot_bar(means, os.path.join(rd, f'e3_{tag}'),
                 ylabel=ylabel, err=stds, title=f'Ablation {tag}',
                 horizontal=(d_key == 'F_gmp_sweep'))
    _bar('A_grid', 'A_grid', 'NMSE (dB)')
    _bar('B_tap', 'B_tap', 'NMSE (dB)')
    _bar('C_feature', 'C_feature', 'NMSE (dB)')
    _bar('D_hybrid', 'D_hybrid', 'NMSE (dB)')
    _bar('E_quant', 'E_quant', 'NMSE (dB)')
    _bar('F_gmp_sweep', 'F_gmp_sweep', 'NMSE (dB)')
    return out


if __name__ == '__main__':
    run_all()
