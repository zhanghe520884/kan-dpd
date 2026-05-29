"""Nature 风格 hero 图集生成脚本。

按 nature-figure skill 的 figure-contract 流程,把所有实验 JSON 与少量
中间复现数据组合为 5 张论文级 hero 图,输出到 results/figures/。

- F1: PA characterization (AM/AM 真值,三种 PA)
- F2: PA forward modeling NMSE + 参数效率
- F3: DPD 全指标 (PSD / AM-AM / NMSE bar / ACPR bar)
- F4: 5 类消融 + GMP K/Q sweep
- F5: 创新点 — 频域损失 NMSE↔ACPR Pareto + PSD

每张图同时输出 SVG / PDF / TIFF (600 dpi),文本可编辑。
"""
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- 让脚本独立可运行(任何位置 cd 进来都行) ----
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config
from signal_gen import gen_ofdm_like, apply_backoff
from pa_models import PA_REGISTRY
from metrics import psd

RES = os.path.join(ROOT, 'results')
FIGDIR = os.path.join(RES, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# =============================================================================
# Nature 全局样式
# =============================================================================
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'mathtext.fontset':  'dejavusans',
    'svg.fonttype':      'none',          # 可编辑 SVG 文本
    'pdf.fonttype':      42,              # TrueType,可编辑 PDF 文本
    'ps.fonttype':       42,
    'font.size':         7,
    'axes.labelsize':    7,
    'axes.titlesize':    7.5,
    'legend.fontsize':   6.5,
    'xtick.labelsize':   6.5,
    'ytick.labelsize':   6.5,
    'lines.linewidth':   0.9,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'axes.spines.right': False,
    'axes.spines.top':   False,
    'legend.frameon':    False,
    'figure.dpi':        110,
    'savefig.dpi':       600,
    'savefig.bbox':      'tight',
    'savefig.transparent': False,
})

# NMI pastel + 一对方向色(绿/红 留作 gain/loss 标注)
PAL = {
    'KAN':      '#4E79A7',
    'GMP':      '#76B7B2',
    'GMP+KAN':  '#59A14F',
    'MP':       '#9C755F',
    'MLP':      '#B07AA1',
    'CNN':      '#F28E2B',
    'GRU':      '#EDC948',
    'NoDPD':    '#999999',
    'target':   '#222222',
    # 方向色
    'gain':     '#2CA02C',
    'loss':     '#D62728',
}
PA_LIST = ['saleh', 'rapp', 'gmp_pa']
PA_LABEL = {'saleh': 'Saleh PA', 'rapp': 'Rapp PA', 'gmp_pa': 'GMP PA'}
MODEL_ORDER_FWD = ['MP', 'GMP', 'MLP', 'CNN', 'GRU', 'KAN', 'GMP+KAN']
MODEL_ORDER_DPD = ['NoDPD', 'MP', 'GMP', 'MLP', 'CNN', 'GRU', 'KAN', 'GMP+KAN']
# 注意 e2 里 DPD 没有 MP 项,在绘图时跳过

INCH_PER_MM = 1.0 / 25.4
def mm(*vals):
    return tuple(v * INCH_PER_MM for v in vals)

# 论文标准宽度
W_SINGLE = 89 * INCH_PER_MM       # ≈ 3.50"
W_DOUBLE = 183 * INCH_PER_MM      # ≈ 7.20"


def save_pub(fig, name, dpi=600):
    base = os.path.join(FIGDIR, name)
    fig.savefig(base + '.svg')
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.tiff', dpi=dpi)
    fig.savefig(base + '.png', dpi=dpi)
    plt.close(fig)


def panel_label(ax, txt, x=-0.20, y=1.05):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left')


def load_json(*parts):
    p = os.path.join(RES, *parts)
    if not os.path.exists(p):
        return None
    with open(p) as fp:
        return json.load(fp)


# =============================================================================
# F1 — PA characterization
# =============================================================================
def figure1():
    """三种 PA 的 AM/AM 散点(真值)。 单栏 1×3。"""
    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.30), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)

    target_rms_map = {'saleh': 0.18, 'rapp': 0.18, 'gmp_pa': 0.13}
    rng = np.random.default_rng(2026)
    x = gen_ofdm_like(8000, 256, 4, rng)
    x = apply_backoff(x, 7.0)

    for j, pa in enumerate(PA_LIST):
        ax = fig.add_subplot(gs[0, j])
        x_pa = (x / np.sqrt(np.mean(np.abs(x) ** 2))
                * target_rms_map[pa]).astype(np.complex64)
        y = PA_REGISTRY[pa](x_pa)
        ax.scatter(np.abs(x_pa), np.abs(y), s=0.6, alpha=0.35,
                   color=PAL['KAN'], rasterized=True, linewidth=0)
        # 理想线性参考(小信号增益)
        low = np.abs(x_pa) < 0.2 * np.std(np.abs(x_pa))
        alpha_lin = np.sum(np.conj(x_pa[low]) * y[low]) / \
                    (np.sum(np.abs(x_pa[low]) ** 2) + 1e-12)
        a = abs(alpha_lin)
        xs = np.linspace(0, np.abs(x_pa).max(), 50)
        ax.plot(xs, a * xs, '--', color='black', linewidth=0.6,
                label='ideal linear')
        ax.set_xlabel(r'$|x|$')
        ax.set_ylabel(r'$|y|$')
        ax.set_title(PA_LABEL[pa])
        if j == 0:
            ax.legend(loc='upper left')
        panel_label(ax, chr(ord('a') + j))

    save_pub(fig, 'F1_pa_characterization')


# =============================================================================
# F2 — Forward modeling
# =============================================================================
def figure2():
    """PA forward modeling NMSE 分组柱状图 + 参数效率散点。 双栏 1×2。"""
    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.35), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.7, 1.0])

    # ----- 面板 a: 分组柱状图 NMSE -----
    ax = fig.add_subplot(gs[0, 0])
    data = {pa: load_json(pa, 'e1_pa_forward.json') for pa in PA_LIST}
    models = MODEL_ORDER_FWD
    nP = len(PA_LIST)
    nM = len(models)
    width = 0.8 / nM
    xs = np.arange(nP)
    for k, m in enumerate(models):
        means = [data[pa]['NMSE_dB'][m]['mean'] for pa in PA_LIST]
        stds = [data[pa]['NMSE_dB'][m]['std'] for pa in PA_LIST]
        ax.bar(xs + (k - nM / 2 + 0.5) * width, means, width,
               yerr=stds, capsize=1.2,
               color=PAL.get(m, '#888'), label=m,
               edgecolor='black', linewidth=0.3,
               error_kw={'elinewidth': 0.5})
    ax.set_xticks(xs)
    ax.set_xticklabels([PA_LABEL[p] for p in PA_LIST])
    ax.set_ylabel('Forward NMSE (dB)')
    ax.invert_yaxis()  # NMSE 越负越好,翻转后柱子越高越好
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18),
              columnspacing=0.8, handletextpad=0.4)
    panel_label(ax, 'a', x=-0.12)
    ax.grid(True, axis='y', alpha=0.25, linestyle=':', linewidth=0.4)

    # ----- 面板 b: 参数效率(参数量 vs NMSE,gmp_pa) -----
    ax = fig.add_subplot(gs[0, 1])
    pa = 'gmp_pa'
    j = data[pa]
    for m in models:
        p = j['params'][m]
        nmse = j['NMSE_dB'][m]['mean']
        s = j['NMSE_dB'][m]['std']
        marker = 'o' if m in ('KAN', 'GMP+KAN') else 's' if m in ('MLP', 'CNN', 'GRU') else '^'
        ax.errorbar(p, nmse, yerr=s, marker=marker,
                    markersize=6 if m in ('KAN', 'GMP+KAN') else 4.5,
                    color=PAL.get(m, '#888'),
                    markeredgecolor='black', markeredgewidth=0.4,
                    capsize=1.5, linewidth=0,
                    elinewidth=0.6, label=m)
        # 直标
        ax.annotate(m, (p, nmse), textcoords='offset points',
                    xytext=(5, 2), fontsize=5.8, color=PAL.get(m, '#444'))
    ax.set_xscale('log')
    ax.set_xlabel('Parameters (log)')
    ax.set_ylabel(f'NMSE on {PA_LABEL[pa]} (dB)')
    panel_label(ax, 'b', x=-0.18)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)

    save_pub(fig, 'F2_pa_forward')


# =============================================================================
# F3 — DPD performance
# =============================================================================
def figure3():
    """DPD 全指标 hero。 双栏 2×2。 子面板:PSD / EVM bar / NMSE bar / ACPR bar。"""
    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.62), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    pa_focus = 'gmp_pa'

    # ----- 面板 a: PSD before/after on gmp_pa -----
    ax = fig.add_subplot(gs[0, 0])
    z_dict = _compute_dpd_outputs(pa_focus, seed=2026)
    plot_models = ['NoDPD', 'GMP', 'KAN', 'GMP+KAN']
    for name in plot_models:
        if name not in z_dict:
            continue
        f, p = psd(z_dict[name])
        ax.plot(f, p, label=name, color=PAL.get(name, '#888'),
                linewidth=0.7, rasterized=True)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('PSD (dB)')
    # 图例放在面板内上方,曲线主要在下方所以不会遮挡
    ax.legend(loc='lower center', ncol=4, fontsize=6,
              columnspacing=0.6, handletextpad=0.3,
              bbox_to_anchor=(0.5, -0.32))
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)
    panel_label(ax, 'a')
    ax.set_title(f'PSD after DPD — {PA_LABEL[pa_focus]}')

    # ----- 面板 b: EVM bar across PAs (log y) -----
    ax = fig.add_subplot(gs[0, 1])
    _grouped_bar(ax, metric='EVM_%', ylabel='EVM (%)  [↓ better]',
                 reverse_y=False, log_y=True)
    panel_label(ax, 'b')

    # ----- 面板 c: NMSE across PAs (not inverted, with arrow annotation) -----
    ax = fig.add_subplot(gs[1, 0])
    _grouped_bar(ax, metric='NMSE_dB',
                 ylabel='Linearization NMSE (dB)  [↓ better]',
                 reverse_y=False)
    panel_label(ax, 'c')

    # ----- 面板 d: ACPR_lower bar -----
    ax = fig.add_subplot(gs[1, 1])
    _grouped_bar(ax, metric='ACPR_lower_dB',
                 ylabel=r'ACPR$_{\mathrm{lower}}$ (dB)  [↓ better]',
                 reverse_y=False)
    panel_label(ax, 'd')

    save_pub(fig, 'F3_dpd_performance')


def _grouped_bar(ax, *, metric: str, ylabel: str, reverse_y: bool,
                 log_y: bool = False):
    data = {pa: load_json(pa, 'e2_dpd_ila.json') for pa in PA_LIST}
    # DPD 里没有 MP,过滤掉
    models = [m for m in MODEL_ORDER_DPD if m in data['saleh'][metric]]
    nP = len(PA_LIST); nM = len(models)
    width = 0.8 / nM
    xs = np.arange(nP)
    for k, m in enumerate(models):
        means = [data[pa][metric][m]['mean'] for pa in PA_LIST]
        stds = [data[pa][metric][m]['std'] for pa in PA_LIST]
        ax.bar(xs + (k - nM / 2 + 0.5) * width, means, width,
               yerr=stds, capsize=1.0,
               color=PAL.get(m, '#888'),
               edgecolor='black', linewidth=0.3,
               error_kw={'elinewidth': 0.4},
               label=m)
    ax.set_xticks(xs)
    ax.set_xticklabels([PA_LABEL[p] for p in PA_LIST])
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale('log')
    if reverse_y:
        ax.invert_yaxis()
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.22),
              columnspacing=0.6, handletextpad=0.3)
    ax.grid(True, axis='y', alpha=0.25, linestyle=':', linewidth=0.4)


def _compute_dpd_outputs(pa_name: str, seed: int = 2026):
    """从 ckpt 加载 DPD 模型并产出 z 信号,用于绘图。 不重训。"""
    import torch
    from experiments.common import prepare_signal, ckpt_path, make_nn_model
    from data import make_features, complex_to_iq, split_indices, FeatureDataset
    from models.mp_gmp import GMP
    from trainer import predict_full

    torch.manual_seed(seed); np.random.seed(seed)
    x, y, pa_fn, alpha = prepare_signal(pa_name, seed)
    tap = config.TAP
    target = (alpha * x).astype(np.complex64)
    y_norm = (y / alpha).astype(np.complex64)
    feats_post = make_features(y_norm, tap)
    feats_pre = make_features(x, tap)
    labels_post = complex_to_iq(x)
    tr, va, te = split_indices(len(x), config.TRAIN_FRAC, config.VAL_FRAC, tap)

    out = {'_x': x, '_alpha': alpha, '_te': te, '_target': target,
           'NoDPD': y}

    # GMP DPD (load coeffs)
    cp = ckpt_path(pa_name, 'dpd_ila', 'GMP', seed).replace('.pt', '.npz')
    if os.path.exists(cp):
        z = np.load(cp)
        gmp = GMP(config.GMP_K, config.GMP_Q, config.GMP_LAG)
        gmp.coeffs = z['coeffs']
        u = gmp.predict(x)
        out['GMP'] = pa_fn(u)
    # NN DPD
    in_dim = feats_post.shape[1]
    for name in ['MLP', 'CNN', 'GRU', 'KAN', 'GMP+KAN']:
        cp = ckpt_path(pa_name, 'dpd_ila', name, seed)
        if not os.path.exists(cp):
            continue
        m = make_nn_model(name, in_dim, tap)
        m.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True))
        idx_all = np.arange(len(x))
        if name == 'GMP+KAN':
            gmp_main = gmp.predict(x)
            u = predict_full(m, feats_pre, idx_all,
                             extra_np=complex_to_iq(gmp_main))
        else:
            u = predict_full(m, feats_pre, idx_all)
        out[name] = pa_fn(u)
    return out


# =============================================================================
# F4 — Ablations
# =============================================================================
def figure4():
    """5 类消融 + GMP K/Q 扫描。 双栏 2×3。"""
    j = load_json('gmp_pa', 'e3_ablation.json')
    if j is None:
        print('e3 ablation JSON missing — skipping F4')
        return

    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.55), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    def _bar(ax, key, xlabel, title, label):
        d = j[key]
        names = list(d.keys())
        means = [d[k]['NMSE_dB_mean'] for k in names]
        stds = [d[k]['NMSE_dB_std'] for k in names]
        # 选择柱子颜色:KAN-类用蓝,GMP-类用青绿
        colors = [PAL['KAN'] if 'KAN' in k.upper() or 'fp' in k or 'q' in k
                  else PAL['GMP'] if 'K=' in k else PAL['KAN']
                  for k in names]
        ax.bar(range(len(names)), means, yerr=stds, capsize=1.2,
               color=colors, edgecolor='black', linewidth=0.3,
               error_kw={'elinewidth': 0.5})
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel('NMSE (dB)')
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        panel_label(ax, label)
        ax.grid(True, axis='y', alpha=0.25, linestyle=':', linewidth=0.4)

    _bar(fig.add_subplot(gs[0, 0]), 'A_grid', 'KAN grid size',
         'Spline grid', 'a')
    _bar(fig.add_subplot(gs[0, 1]), 'B_tap', 'Delay taps',
         'Memory depth', 'b')
    _bar(fig.add_subplot(gs[0, 2]), 'C_feature', 'Input feature variant',
         'Feature ablation', 'c')
    _bar(fig.add_subplot(gs[1, 0]), 'D_hybrid', 'Architecture',
         'Pure vs Hybrid KAN', 'd')
    _bar(fig.add_subplot(gs[1, 1]), 'E_quant', 'Bit width',
         'LUT quantization', 'e')

    # GMP K/Q heatmap
    ax = fig.add_subplot(gs[1, 2])
    Ks = [5, 7, 9]; Qs = [3, 5, 7]
    M = np.zeros((len(Qs), len(Ks)))
    for i, Q in enumerate(Qs):
        for k_idx, K in enumerate(Ks):
            tag = f'K={K},Q={Q}'
            if tag in j['F_gmp_sweep']:
                M[i, k_idx] = j['F_gmp_sweep'][tag]['NMSE_dB_mean']
    im = ax.imshow(M, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(Ks))); ax.set_xticklabels([f'K={k}' for k in Ks])
    ax.set_yticks(range(len(Qs))); ax.set_yticklabels([f'Q={q}' for q in Qs])
    for i in range(M.shape[0]):
        for j2 in range(M.shape[1]):
            ax.text(j2, i, f'{M[i, j2]:.1f}', ha='center', va='center',
                    color='white' if M[i, j2] < M.mean() else 'black',
                    fontsize=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.06)
    cb.set_label('NMSE (dB)', rotation=270, labelpad=10, fontsize=6.5)
    cb.ax.tick_params(width=0.4, labelsize=6)
    ax.set_title('GMP order/memory sweep', pad=10)
    panel_label(ax, 'f', y=1.10)

    save_pub(fig, 'F4_ablations')


# =============================================================================
# F5 — Innovation
# =============================================================================
def figure5():
    """频域损失:NMSE↔ACPR Pareto + PSD。 单栏 2×1。"""
    j = load_json('gmp_pa', 'e4_freq_loss.json')
    if j is None:
        print('e4 freq loss JSON missing — skipping F5')
        return

    fig = plt.figure(figsize=(W_SINGLE, W_SINGLE * 1.55), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1])

    # ----- 面板 a: NMSE vs ACPR scatter (Pareto) -----
    ax = fig.add_subplot(gs[0])
    tags = list(j['NMSE_dB'].keys())
    nmse_mean = [j['NMSE_dB'][t]['mean'] for t in tags]
    nmse_std = [j['NMSE_dB'][t]['std'] for t in tags]
    acpr_mean = [j['ACPR_lower_dB'][t]['mean'] for t in tags]
    acpr_std = [j['ACPR_lower_dB'][t]['std'] for t in tags]
    # 渐变色按 λ
    lambdas = [float(t.split('=')[1]) for t in tags]
    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.Normalize(vmin=min(lambdas), vmax=max(lambdas))
    for k in range(len(tags)):
        c = cmap(norm(lambdas[k]))
        ax.errorbar(nmse_mean[k], acpr_mean[k],
                    xerr=nmse_std[k], yerr=acpr_std[k],
                    marker='o', markersize=6,
                    color=c, markeredgecolor='black', markeredgewidth=0.5,
                    capsize=1.5, elinewidth=0.6, linewidth=0)
        ax.annotate(tags[k], (nmse_mean[k], acpr_mean[k]),
                    textcoords='offset points', xytext=(5, 4),
                    fontsize=6, fontweight='bold')
    # 连接成 Pareto 折线
    order = np.argsort(nmse_mean)
    ax.plot([nmse_mean[i] for i in order],
            [acpr_mean[i] for i in order],
            '-', color='#666', linewidth=0.5, zorder=0)
    ax.set_xlabel('NMSE (dB)')
    ax.set_ylabel('ACPR$_\\mathrm{lower}$ (dB)')
    ax.set_title('NMSE ↔ ACPR trade-off via spectrum-aware loss')
    panel_label(ax, 'a')
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)
    # 方向标注
    ax.annotate('better\nspectrum', xy=(0.03, 0.05), xycoords='axes fraction',
                fontsize=6, color=PAL['gain'], fontweight='bold',
                ha='left', va='bottom')
    ax.annotate('better\nin-band', xy=(0.97, 0.95), xycoords='axes fraction',
                fontsize=6, color=PAL['gain'], fontweight='bold',
                ha='right', va='top')

    # ----- 面板 b: PSD overlay -----
    ax = fig.add_subplot(gs[1])
    z_by_lam = _compute_freq_dpd_outputs()
    if z_by_lam:
        te = z_by_lam['_te']
        for tag in z_by_lam:
            if tag.startswith('_'):
                continue
            f, p = psd(z_by_lam[tag][te])
            ax.plot(f, p, label=tag,
                    color=cmap(norm(float(tag.split('=')[1]))),
                    linewidth=0.9)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('PSD (dB)')
    ax.set_title('PSD vs spectrum-loss weight λ')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
              ncol=4, fontsize=6.5, columnspacing=0.8,
              handletextpad=0.3, frameon=False)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.4)
    panel_label(ax, 'b')

    save_pub(fig, 'F5_freq_loss_innovation')


def _compute_freq_dpd_outputs(seed=2026):
    """加载 e4 不同 λ 的 KAN ckpt,运行预失真链得到 z。"""
    import torch
    from experiments.common import (prepare_signal, ckpt_path,
                                    make_nn_model)
    from data import make_features, complex_to_iq, split_indices
    from trainer import predict_full

    j = load_json('gmp_pa', 'e4_freq_loss.json')
    if j is None:
        return {}
    lambdas = j.get('lambdas', [0.0, 0.1, 1.0, 5.0])
    torch.manual_seed(seed); np.random.seed(seed)
    x, y, pa_fn, alpha = prepare_signal('gmp_pa', seed)
    tap = config.TAP
    feats_pre = make_features(x, tap)
    in_dim = feats_pre.shape[1]
    tr, va, te = split_indices(len(x), config.TRAIN_FRAC, config.VAL_FRAC, tap)
    out = {'_te': te}
    for lam in lambdas:
        tag_short = f'KAN_freq_lam{lam:g}'
        cp = ckpt_path('gmp_pa', 'freq_loss', tag_short, seed)
        if not os.path.exists(cp):
            continue
        m = make_nn_model('KAN', in_dim, tap)
        m.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True))
        u = predict_full(m, feats_pre, np.arange(len(x)))
        out[f'λ={lam:g}'] = pa_fn(u)
    return out


# =============================================================================
# Entry
# =============================================================================
def figure6():
    """仿真 → 实测迁移,跨 2 个 OpenDPD 数据集对照。 双栏 2×2。

    行: DPA_200MHz, DPA_160MHz
    列: NMSE, ACPR_lower
    """
    pretty = {
        'A_scratch':         'From scratch (measured only)',
        'B_zero_shot':       'Zero-shot (sim ckpt)',
        'C_finetune_5pct':   'Few-shot 5% fine-tune',
        'C_finetune_20pct':  'Few-shot 20% fine-tune',
    }
    palette = ['#999999', '#D62728', '#F28E2B', PAL['KAN']]

    # 自动发现 e5 多数据集 JSON。 每个数据集在 results/sim_to_meas/<tag>/ 下
    # 有自己的 e5_sim_to_meas_<tag>.json
    sm_dir = os.path.join(RES, 'sim_to_meas')
    candidates = []
    if os.path.isdir(sm_dir):
        for entry in sorted(os.listdir(sm_dir)):
            sub = os.path.join(sm_dir, entry)
            if os.path.isdir(sub):
                for f in sorted(os.listdir(sub)):
                    if f.startswith('e5_sim_to_meas_') and f.endswith('.json'):
                        candidates.append(os.path.join(sub, f))
            elif entry.startswith('e5_sim_to_meas_') and entry.endswith('.json'):
                candidates.append(sub)
    # 没有多数据集 JSON 就回退到单数据集
    if not candidates:
        single = os.path.join(sm_dir, 'e5_sim_to_meas.json')
        if not os.path.exists(single):
            print('  e5 sim→meas JSON 不存在,跳过 F6')
            return
        candidates = [single]

    datasets = []
    for p in candidates:
        with open(p) as fp:
            j = json.load(fp)
        datasets.append((j.get('measured_data', '?'), j))

    n_rows = len(datasets)
    fig_h = W_DOUBLE * 0.30 * n_rows + 0.3
    fig = plt.figure(figsize=(W_DOUBLE, fig_h), constrained_layout=True)
    gs = gridspec.GridSpec(n_rows, 2, figure=fig)

    panel_letters = 'abcdefgh'
    for row, (tag, j) in enumerate(datasets):
        order = [t for t in pretty if t in j['NMSE_dB']]
        yvals = np.arange(len(order))

        # NMSE
        ax = fig.add_subplot(gs[row, 0])
        m = [j['NMSE_dB'][t]['mean'] for t in order]
        s = [j['NMSE_dB'][t]['std'] for t in order]
        ax.barh(yvals, m, xerr=s, color=palette[:len(order)],
                edgecolor='black', linewidth=0.3,
                error_kw={'elinewidth': 0.5}, capsize=1.5)
        ax.set_yticks(yvals)
        ax.set_yticklabels([pretty[t] for t in order])
        ax.invert_yaxis()
        ax.set_xlabel('NMSE (dB)  [↓ better]')
        ax.set_title(f'Sim → {tag}: NMSE  (source = {j["source_pa"]})',
                     fontsize=7.5)
        panel_label(ax, panel_letters[2 * row], x=-0.34)
        ax.grid(True, axis='x', alpha=0.25, linestyle=':', linewidth=0.4)

        # ACPR
        ax = fig.add_subplot(gs[row, 1])
        m = [j['ACPR_lower_dB'][t]['mean'] for t in order]
        s = [j['ACPR_lower_dB'][t]['std'] for t in order]
        ax.barh(yvals, m, xerr=s, color=palette[:len(order)],
                edgecolor='black', linewidth=0.3,
                error_kw={'elinewidth': 0.5}, capsize=1.5)
        ax.set_yticks(yvals)
        ax.set_yticklabels([pretty[t] for t in order])
        ax.invert_yaxis()
        ax.set_xlabel(r'ACPR$_{\mathrm{lower}}$ (dB)  [↓ better]')
        ax.set_title(f'Sim → {tag}: ACPR$_{{lower}}$', fontsize=7.5)
        panel_label(ax, panel_letters[2 * row + 1], x=-0.28)
        ax.grid(True, axis='x', alpha=0.25, linestyle=':', linewidth=0.4)

    save_pub(fig, 'F6_sim_to_meas')


def main():
    print('Generating Nature-style hero figures...')
    figure1(); print('  F1 done')
    figure2(); print('  F2 done')
    figure3(); print('  F3 done')
    figure4(); print('  F4 done')
    figure5(); print('  F5 done')
    figure6(); print('  F6 done')
    print(f'All hero figures saved to: {FIGDIR}')


if __name__ == '__main__':
    main()
