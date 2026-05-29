"""显著性检验:配对 t / Wilcoxon / Holm-Bonferroni 校正。

实验默认配置:
  * 每个模型 ≥ 5 个随机种子,报告均值 ± 标准差
  * 优先做配对检验, α = 0.05
  * 同时比较多个模型时用 Holm 校正控制 family-wise error rate
"""
import numpy as np
from scipy import stats


def mean_std(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1) if len(a) > 1 else 0.0)


def paired_test(a: list[float], b: list[float], prefer: str = 'auto') -> dict:
    """对配对样本 a, b 做差异显著性检验。

    prefer:
        'auto'      —— 样本量≥8 且 Shapiro 正态时用 t,否则 Wilcoxon
        't'         —— 强制配对 t 检验
        'wilcoxon'  —— 强制 Wilcoxon signed-rank
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    d = a - b

    if prefer == 'auto':
        if len(d) >= 8:
            try:
                _, p_norm = stats.shapiro(d)
                use_t = p_norm > 0.05
            except Exception:
                use_t = False
        else:
            use_t = False
        kind = 't' if use_t else 'wilcoxon'
    else:
        kind = prefer

    if kind == 't':
        t, p = stats.ttest_rel(a, b)
        return {'test': 'paired_t', 'statistic': float(t), 'p_value': float(p),
                'mean_diff': float(np.mean(d))}
    else:
        try:
            w, p = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        except ValueError:
            # 所有差值为零等退化情形
            return {'test': 'wilcoxon', 'statistic': 0.0, 'p_value': 1.0,
                    'mean_diff': float(np.mean(d))}
        return {'test': 'wilcoxon', 'statistic': float(w), 'p_value': float(p),
                'mean_diff': float(np.mean(d))}


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni 校正。 返回每个对比是否仍显著(True/False)。"""
    n = len(p_values)
    order = np.argsort(p_values)
    significant = [False] * n
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if p_values[idx] <= threshold:
            significant[idx] = True
        else:
            break  # Holm:首次失败后即停
    return significant


def summarize_runs(runs: dict[str, list[float]], reference: str,
                   alpha: float = 0.05) -> dict:
    """对多模型多种子结果做均值/标准差 + 与 reference 模型的配对检验 + Holm。

    Args:
        runs: {model_name: [metric_seed_1, metric_seed_2, ...]}
        reference: 用作 baseline 的模型名(其它模型与之做配对检验)
    """
    summary = {}
    for name, vals in runs.items():
        m, s = mean_std(vals)
        summary[name] = {'mean': m, 'std': s, 'values': list(vals)}
    if reference not in runs:
        return summary
    ref_vals = runs[reference]
    others = [n for n in runs if n != reference]
    p_list = []
    raw = {}
    for n in others:
        r = paired_test(runs[n], ref_vals)
        raw[n] = r
        p_list.append(r['p_value'])
    sig = holm_correction(p_list, alpha=alpha)
    for n, s, r in zip(others, sig, raw.values()):
        summary[n]['paired_vs_ref'] = {**r, 'reference': reference,
                                       'significant_holm': bool(s)}
    return summary
