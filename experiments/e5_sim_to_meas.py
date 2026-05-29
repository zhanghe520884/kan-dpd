"""实验五:仿真 → 实测迁移。

"仿真→实测"迁移评估:在仿真 PA 上完成收敛与指标验证后,把训练好的
KAN-DPD 模型迁移到 OpenDPD 公开测量数据上,做 zero-shot / few-shot 微调,
检验仿真预训练对真实测量的可迁移性。

本实验在"实测代理"(data_measured.measured_proxy_pa)上比较三种策略,
都用 KAN-DPD 模型,统一架构与超参,公平对照:

  A. **From-scratch**:  仅用实测数据从零训练,作为上限对照
  B. **Zero-shot**:     用仿真训练的 KAN ckpt 直接迁移,不微调
  C. **Few-shot 微调**: 加载仿真 ckpt 后,只用 5% / 20% 实测数据微调

评估方式:由于实测数据没有可微的 PA 前向,我们用一个 GMP-forward
**代理 PA**(在实测数据上 LS 拟合)作为评测前向。 该代理同时也是 ILA
post-distorter 训练时的"y_norm 来源"。

如果用户在 data/measured/ 下放置了真实 OpenDPD .npz,会自动使用;
否则使用 data_measured.synthesize_measured_proxy() 生成的合成代理数据。
"""
import os
import json
import copy
import numpy as np
import torch

import config
from data import (make_features, complex_to_iq, split_indices,
                  FeatureDataset)
from models.mp_gmp import GMP
from models.kan import KAN
from trainer import train_model, predict_full, count_params
from metrics import nmse_db, acpr_db, evm_pct
from stats import mean_std
from plotting import plot_bar, plot_psd
from data_measured import get_dataset
from experiments.common import ckpt_path, history_path


SOURCE_PA = 'gmp_pa'        # 仿真预训练源(有记忆 PA 仿真)
FEW_SHOT_FRACS = [0.05, 0.20]
SEEDS = config.SEEDS[:3]    # 节约时间


def _prep_measured_split(info: dict):
    """对实测数据建立 post-distorter 训练域 + 切分。

    若 info 含 OpenDPD 官方 train_end/val_end,则使用官方分割,否则按 config 顺序切。
    """
    x = info['x']; y = info['y']
    tap = config.TAP
    alpha = complex(np.sum(np.conj(x) * y) / (np.sum(np.abs(x) ** 2) + 1e-12))
    y_norm = (y / alpha).astype(np.complex64)
    target = (alpha * x).astype(np.complex64)
    feats_post = make_features(y_norm, tap)
    feats_pre = make_features(x, tap)
    labels_post = complex_to_iq(x)

    if info.get('train_end') is not None and info.get('val_end') is not None:
        # OpenDPD 官方分割
        train_end = int(info['train_end']); val_end = int(info['val_end'])
        tr = np.arange(tap, train_end)
        va = np.arange(train_end, val_end)
        te = np.arange(val_end, len(x))
    else:
        tr, va, te = split_indices(len(x), config.TRAIN_FRAC, config.VAL_FRAC, tap)

    gmp_fwd = GMP(7, 4, 2)
    gmp_fwd.fit(x[:tr[-1] + 1], y[:tr[-1] + 1])

    def proxy_pa(u: np.ndarray) -> np.ndarray:
        return gmp_fwd.predict(u)

    return dict(x=x, y=y, alpha=alpha, target=target, y_norm=y_norm,
                feats_post=feats_post, feats_pre=feats_pre,
                labels_post=labels_post, tr=tr, va=va, te=te,
                proxy_pa=proxy_pa)


def _evaluate(model, dat):
    idx_all = np.arange(len(dat['x']))
    u = predict_full(model, dat['feats_pre'], idx_all)
    z = dat['proxy_pa'](u)
    te = dat['te']
    a_l, a_u = acpr_db(z[te])
    return {
        'NMSE_dB': nmse_db(dat['target'][te], z[te]),
        'EVM_%':   evm_pct(dat['target'][te], z[te]),
        'ACPR_lower_dB': a_l, 'ACPR_upper_dB': a_u,
    }, z


def _make_kan(in_dim: int) -> KAN:
    return KAN([in_dim, config.KAN_HIDDEN, 2],
               grid_size=config.KAN_GRID,
               spline_order=config.KAN_SPLINE_ORDER)


def run(measured_name: str | None = None,
        epochs_scratch: int | None = None,
        epochs_finetune: int = 15,
        offline: bool = False):
    epochs_scratch = epochs_scratch or config.EPOCHS

    info, data_tag = get_dataset(measured_name, offline=offline)
    # 不同数据集独立目录,避免互相覆盖
    safe_tag = data_tag.replace('.npz', '') if data_tag != '__proxy__' else 'proxy'
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results',
                           'sim_to_meas', safe_tag)
    os.makedirs(out_dir, exist_ok=True)
    x_meas, y_meas = info['x'], info['y']
    print(f'\n=== Experiment 5: Sim → Measured transfer  (measured = "{data_tag}") ===')
    print(f'  measured size: {len(x_meas)} samples')
    if info.get('train_end') is not None:
        print(f'  OpenDPD 官方分割: train_end={info["train_end"]}  '
              f'val_end={info["val_end"]}')

    runs: dict[str, dict[str, list[float]]] = {}
    last_z_per_strategy = {}

    for seed in SEEDS:
        print(f'  seed = {seed}')
        torch.manual_seed(seed); np.random.seed(seed)
        dat = _prep_measured_split(info)
        in_dim = dat['feats_post'].shape[1]

        # ---- A. From-scratch on measured ----
        m_scratch = _make_kan(in_dim)
        tr_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['tr'])
        va_ds = FeatureDataset(dat['feats_post'], dat['labels_post'], dat['va'])
        cp = ckpt_path('sim_to_meas', f'measured_{data_tag}',
                       'KAN_scratch', seed)
        hp = history_path('sim_to_meas', f'measured_{data_tag}',
                          'KAN_scratch', seed)
        m_scratch, _ = train_model(m_scratch, tr_ds, va_ds,
                                   epochs=epochs_scratch,
                                   ckpt_path=cp, history_path=hp)
        r, z = _evaluate(m_scratch, dat)
        for k, v in r.items():
            runs.setdefault(k, {}).setdefault('A_scratch', []).append(v)
        if seed == SEEDS[-1]:
            last_z_per_strategy['A_scratch'] = z

        # ---- B. Zero-shot from sim ckpt ----
        sim_cp = ckpt_path(SOURCE_PA, 'dpd_ila', 'KAN', seed)
        if not os.path.exists(sim_cp):
            print(f'    WARN: 仿真 ckpt 不存在 {sim_cp},跳过 B 与 C')
            continue
        m_zs = _make_kan(in_dim)
        m_zs.load_state_dict(torch.load(sim_cp, map_location='cpu',
                                        weights_only=True))
        r, z = _evaluate(m_zs, dat)
        for k, v in r.items():
            runs.setdefault(k, {}).setdefault('B_zero_shot', []).append(v)
        if seed == SEEDS[-1]:
            last_z_per_strategy['B_zero_shot'] = z

        # ---- C. Few-shot fine-tune ----
        for frac in FEW_SHOT_FRACS:
            tag = f'C_finetune_{int(frac*100)}pct'
            # 用前 frac 的 train 样本做微调
            n_use = max(int(frac * len(dat['tr'])), 64)
            sub_tr = dat['tr'][:n_use]
            sub_va = dat['va'][:max(int(frac * len(dat['va'])), 32)]
            tr_ds_s = FeatureDataset(dat['feats_post'], dat['labels_post'], sub_tr)
            va_ds_s = FeatureDataset(dat['feats_post'], dat['labels_post'], sub_va)

            m_ft = _make_kan(in_dim)
            m_ft.load_state_dict(torch.load(sim_cp, map_location='cpu',
                                            weights_only=True))
            cp = ckpt_path('sim_to_meas', f'measured_{data_tag}',
                           f'KAN_finetune_{int(frac*100)}pct', seed)
            hp = history_path('sim_to_meas', f'measured_{data_tag}',
                              f'KAN_finetune_{int(frac*100)}pct', seed)
            m_ft, _ = train_model(m_ft, tr_ds_s, va_ds_s,
                                  epochs=epochs_finetune,
                                  lr=config.LR * 0.3,        # 微调用更小 LR
                                  ckpt_path=cp, history_path=hp)
            r, z = _evaluate(m_ft, dat)
            for k, v in r.items():
                runs.setdefault(k, {}).setdefault(tag, []).append(v)
            if seed == SEEDS[-1]:
                last_z_per_strategy[tag] = z

    # ---- 汇总 + 落盘 ----
    summary = {}
    for metric, by_tag in runs.items():
        summary[metric] = {}
        for tag, vals in by_tag.items():
            m, s = mean_std(vals)
            summary[metric][tag] = {'mean': m, 'std': s, 'values': vals}
    summary['source_pa'] = SOURCE_PA
    summary['measured_data'] = data_tag
    summary['seeds'] = list(SEEDS)
    summary['few_shot_fracs'] = FEW_SHOT_FRACS

    # 每个数据集落盘到独立 JSON,便于多数据集对照
    tag_for_file = data_tag.replace('.npz', '')
    json_name = f'e5_sim_to_meas_{tag_for_file}.json'
    with open(os.path.join(out_dir, json_name), 'w') as fp:
        json.dump(summary, fp, indent=2)
    # 同时维护一个"最新一次"指针,F6 默认读这个
    with open(os.path.join(out_dir, 'e5_sim_to_meas.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)

    print(f'  -- summary (mean ± std over {len(SEEDS)} seeds) --')
    print(f'    {"strategy":24s}  {"NMSE_dB":>14s}  {"EVM_%":>10s}  '
          f'{"ACPR_L":>8s}')
    for tag in summary['NMSE_dB']:
        n = summary['NMSE_dB'][tag]
        e = summary['EVM_%'][tag]
        al = summary['ACPR_lower_dB'][tag]
        print(f'    {tag:24s}  {n["mean"]:6.2f}±{n["std"]:.2f}  '
              f'{e["mean"]:6.2f}±{e["std"]:.2f}  {al["mean"]:6.2f}')

    # ---- 绘图(用最后一个 seed) ----
    if last_z_per_strategy:
        te = dat['te']
        plot_psd({tag: z[te] for tag, z in last_z_per_strategy.items()},
                 os.path.join(out_dir, f'e5_psd_{tag_for_file}'),
                 title=f'PSD by transfer strategy ({data_tag})')

    # NMSE/ACPR 柱状图
    nmse_m = {t: summary['NMSE_dB'][t]['mean'] for t in summary['NMSE_dB']}
    nmse_s = {t: summary['NMSE_dB'][t]['std'] for t in summary['NMSE_dB']}
    plot_bar(nmse_m, os.path.join(out_dir, f'e5_nmse_{tag_for_file}'),
             ylabel='NMSE (dB)  [↓ better]', err=nmse_s,
             title=f'Sim → Measured NMSE ({data_tag})',
             horizontal=True)
    acpr_m = {t: summary['ACPR_lower_dB'][t]['mean']
              for t in summary['ACPR_lower_dB']}
    acpr_s = {t: summary['ACPR_lower_dB'][t]['std']
              for t in summary['ACPR_lower_dB']}
    plot_bar(acpr_m, os.path.join(out_dir, f'e5_acpr_{tag_for_file}'),
             ylabel='ACPR$_\\mathrm{lower}$ (dB)  [↓ better]', err=acpr_s,
             title=f'Sim → Measured ACPR ({data_tag})',
             horizontal=True)

    return summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=None,
                   help='测量数据集名(如 DPA_200MHz.npz / DPA_160MHz.npz);'
                        '不指定则用 SOURCES.json 默认偏好')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    run(measured_name=args.dataset, offline=args.offline)
