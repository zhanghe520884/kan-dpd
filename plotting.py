"""学术风格绘图工具。

约定:
  * 单栏 figsize ≈ (3.5, 2.6),双栏 ≈ (7.0, 3.0)
  * 字体使用 serif,size=8~10
  * 同时输出 PNG (dpi=300) 与 PDF (矢量),便于论文使用
  * 色盲友好 palette (基于 matplotlib 'tab10')
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from metrics import psd

# ---- 全局样式 ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.frameon': False,
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
           '#8c564b', '#17becf', '#e377c2']


def _save(fig, out_path: str):
    """同名同时保存 PNG 和 PDF。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + '.png')
    fig.savefig(base + '.pdf')
    plt.close(fig)


def plot_psd(signals: dict, out_path: str, title: str = '', xlim=(-0.5, 0.5)):
    """PSD 比较图。 图例放在图外底部,避免遮挡曲线。"""
    n = len(signals)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    for i, (name, s) in enumerate(signals.items()):
        f, p = psd(np.asarray(s))
        ax.plot(f, p, label=name, color=PALETTE[i % len(PALETTE)],
                linewidth=0.9, rasterized=True)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('PSD (dB)')
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
    ncol = min(n, 4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
              ncol=ncol, fontsize=7.5, columnspacing=1.0,
              handletextpad=0.4, frameon=False)
    _save(fig, out_path)


def plot_am_am(curves: dict, out_path: str, title: str = ''):
    """curves: {name: (x_complex, y_complex)}。 散点 AM/AM 比较图。"""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for i, (name, (xc, yc)) in enumerate(curves.items()):
        ax.scatter(np.abs(xc), np.abs(yc), s=1.5, alpha=0.35,
                   color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel(r'$|x|$')
    ax.set_ylabel(r'$|y|$')
    if title:
        ax.set_title(title)
    ax.legend(markerscale=4, loc='lower right')
    ax.grid(True, alpha=0.25, linestyle=':')
    _save(fig, out_path)


def plot_am_pm(curves: dict, out_path: str, title: str = ''):
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for i, (name, (xc, yc)) in enumerate(curves.items()):
        dphi = np.angle(yc * np.conj(xc))
        ax.scatter(np.abs(xc), dphi, s=1.5, alpha=0.35,
                   color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel(r'$|x|$')
    ax.set_ylabel(r'$\angle(y\,x^{*})$ [rad]')
    if title:
        ax.set_title(title)
    ax.legend(markerscale=4, loc='best')
    ax.grid(True, alpha=0.25, linestyle=':')
    _save(fig, out_path)


def plot_constellation(signals: dict, out_path: str, title: str = '',
                       max_points: int = 4000):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    for i, (name, s) in enumerate(signals.items()):
        s = np.asarray(s)
        if len(s) > max_points:
            idx = np.linspace(0, len(s) - 1, max_points).astype(int)
            s = s[idx]
        ax.scatter(s.real, s.imag, s=1.5, alpha=0.4,
                   color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    if title:
        ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(markerscale=4, loc='upper right')
    ax.grid(True, alpha=0.25, linestyle=':')
    _save(fig, out_path)


def plot_error_hist(errors: dict, out_path: str, title: str = '', bins: int = 80):
    """误差幅度的 log-x 直方图 + ECDF 组合图(误差量级跨多个数量级时更清晰)。

    errors: {name: complex_error_array}
    上图: log-x 阶梯直方图(stairs);下图: 经验 CDF。
    """
    # 收集所有 |err| 用以确定统一的 log-bin 范围
    eps = 1e-10
    all_mags = []
    for e in errors.values():
        mag = np.abs(np.asarray(e))
        mag = mag[mag > eps]
        if mag.size:
            all_mags.append(mag)
    if not all_mags:
        return
    lo = max(eps, min(m.min() for m in all_mags))
    hi = max(m.max() for m in all_mags)
    edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 4.0), sharex=True)
    for i, (name, e) in enumerate(errors.items()):
        mag = np.abs(np.asarray(e))
        # 阶梯密度直方图(stairs 比 hist 多线更清晰)
        h, _ = np.histogram(mag, bins=edges, density=True)
        ax1.stairs(h, edges, label=name,
                   color=PALETTE[i % len(PALETTE)], linewidth=1.4)
        # ECDF
        s = np.sort(mag)
        y = np.arange(1, len(s) + 1) / len(s)
        ax2.plot(s, y, color=PALETTE[i % len(PALETTE)], linewidth=1.2,
                 label=name)
    ax1.set_xscale('log')
    ax1.set_ylabel('Density')
    if title:
        ax1.set_title(title)
    ax1.legend(loc='upper left', fontsize=7)
    ax1.grid(True, alpha=0.25, linestyle=':', which='both')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'$|y_{\mathrm{true}} - \hat{y}|$')
    ax2.set_ylabel('ECDF')
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.25, linestyle=':', which='both')
    fig.tight_layout()
    _save(fig, out_path)


def plot_convergence(histories: dict, out_path: str, title: str = ''):
    """histories: {name: {'train_mse': [...], 'val_mse': [...]}}。"""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for i, (name, h) in enumerate(histories.items()):
        if not h.get('val_mse'):
            continue
        ep = np.arange(1, len(h['val_mse']) + 1)
        ax.semilogy(ep, h['val_mse'], color=PALETTE[i % len(PALETTE)],
                    label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MSE')
    if title:
        ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.25, linestyle=':', which='both')
    _save(fig, out_path)


def plot_bar(values: dict, out_path: str, *, ylabel: str = '',
             title: str = '', err: dict | None = None,
             horizontal: bool = False):
    """通用柱状图(用于 NMSE/ACPR 对比 + 误差条)。 values: {name: mean}。"""
    names = list(values.keys())
    means = [values[n] for n in names]
    errs = [err[n] for n in names] if err else None
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    if horizontal:
        ax.barh(names, means, xerr=errs, color=colors,
                edgecolor='black', linewidth=0.5, capsize=3)
        ax.set_xlabel(ylabel)
    else:
        ax.bar(names, means, yerr=errs, color=colors,
               edgecolor='black', linewidth=0.5, capsize=3)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=30, ha='right')
    if title:
        ax.set_title(title)
    ax.grid(True, axis='y' if not horizontal else 'x', alpha=0.25, linestyle=':')
    _save(fig, out_path)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
