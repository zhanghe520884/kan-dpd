"""一键生成跨 PA × 模型 × 指标的横向汇总表。

读取 results/<pa>/e{1,2,3,4}_*.json,合成:
    results/SUMMARY.md   —— Markdown 表(论文/报告直接复制)
    results/SUMMARY.csv  —— 机读 CSV(便于二次分析)

无须额外训练,仅读取 JSON 落盘。
"""
import os
import json
import csv
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        return json.load(fp)


def fmt(v, prec=2):
    if v is None or isinstance(v, (str, list, dict)):
        return str(v)
    if isinstance(v, float):
        return f'{v:.{prec}f}'
    return str(v)


def section_e1(md, csv_rows):
    md.append('\n## 1. PA 前向建模 NMSE (dB)  ——  5 seeds, mean ± std\n')
    pas = ['saleh', 'rapp', 'gmp_pa']
    data = {}
    models_seen = []
    for pa in pas:
        j = load_json(os.path.join(RES, pa, 'e1_pa_forward.json'))
        if j is None:
            continue
        data[pa] = j['NMSE_dB']
        for m in j['NMSE_dB']:
            if m not in models_seen:
                models_seen.append(m)
    md.append('| Model | ' + ' | '.join(pas) + ' | params |')
    md.append('|---' + '|---:' * (len(pas) + 1) + '|')
    for m in models_seen:
        row = [m]
        for pa in pas:
            d = data.get(pa, {}).get(m)
            if d is None:
                row.append('—')
            else:
                row.append(f'**{d["mean"]:.2f}** ± {d["std"]:.2f}')
        # params(取任一 PA 的值,通常一致)
        params = next((load_json(os.path.join(RES, pa, 'e1_pa_forward.json'))
                       ['params'].get(m) for pa in pas
                       if load_json(os.path.join(RES, pa, 'e1_pa_forward.json')) and
                       m in load_json(os.path.join(RES, pa, 'e1_pa_forward.json'))['params']),
                      None)
        row.append(fmt(params, 0))
        md.append('| ' + ' | '.join(row) + ' |')
        csv_rows.append(['e1_NMSE_dB', m] + [
            data.get(pa, {}).get(m, {}).get('mean', '') for pa in pas
        ] + [params])
    return models_seen


def section_e2(md, csv_rows):
    md.append('\n## 2. DPD (ILA) 全指标  ——  5 seeds, mean ± std\n')
    md.append('### 2.1 线性化 NMSE (dB)\n')
    pas = ['saleh', 'rapp', 'gmp_pa']
    metrics = ['NMSE_dB', 'EVM_%', 'ACPR_lower_dB']
    nice_name = {'NMSE_dB': 'NMSE (dB)', 'EVM_%': 'EVM (%)',
                 'ACPR_lower_dB': 'ACPR_lower (dB)'}
    j_by_pa = {pa: load_json(os.path.join(RES, pa, 'e2_dpd_ila.json'))
               for pa in pas}
    models = []
    for pa in pas:
        if j_by_pa[pa] is None:
            continue
        for m in j_by_pa[pa]['NMSE_dB']:
            if m not in models:
                models.append(m)
    for metric in metrics:
        md.append(f'\n#### {nice_name[metric]}\n')
        md.append('| Model | ' + ' | '.join(pas) + ' |')
        md.append('|---' + '|---:' * len(pas) + '|')
        for m in models:
            row = [m]
            for pa in pas:
                j = j_by_pa[pa]
                if j is None or metric not in j or m not in j[metric]:
                    row.append('—')
                    continue
                d = j[metric][m]
                row.append(f'{d["mean"]:.2f} ± {d["std"]:.2f}')
            md.append('| ' + ' | '.join(row) + ' |')
            csv_rows.append([f'e2_{metric}', m] + [
                j_by_pa[pa][metric][m]['mean'] if (j_by_pa[pa] and metric in j_by_pa[pa]
                                                  and m in j_by_pa[pa][metric]) else ''
                for pa in pas
            ])


def section_e3(md, csv_rows):
    md.append('\n## 3. 消融实验(gmp_pa, 2 seeds)\n')
    j = load_json(os.path.join(RES, 'gmp_pa', 'e3_ablation.json'))
    if j is None:
        md.append('*(no e3 results)*\n')
        return
    keys = [('A_grid', 'A: KAN 样条网格'),
            ('B_tap', 'B: 时延抽头数'),
            ('C_feature', 'C: 输入特征'),
            ('D_hybrid', 'D: 纯 KAN vs Hybrid'),
            ('E_quant', 'E: KAN 边函数 LUT 量化'),
            ('F_gmp_sweep', 'F: GMP K/Q 扫描')]
    for k, title in keys:
        if k not in j:
            continue
        md.append(f'\n### {title}\n')
        md.append('| 设置 | NMSE (dB) |')
        md.append('|---|---:|')
        for tag, d in j[k].items():
            m = d.get('NMSE_dB_mean', d.get('mean'))
            s = d.get('NMSE_dB_std', d.get('std', 0.0))
            md.append(f'| {tag} | {m:.2f} ± {s:.2f} |')
            csv_rows.append([f'e3_{k}', tag, m, s])


def section_e4(md, csv_rows):
    p = os.path.join(RES, 'gmp_pa', 'e4_freq_loss.json')
    j = load_json(p)
    if j is None:
        md.append('\n## 4. 频域感知 KAN-DPD (尚未运行)\n')
        return
    md.append('\n## 4. 频域感知 KAN-DPD 损失(创新点)\n')
    md.append(f'在 gmp_pa 上,KAN-DPD 训练损失 = MSE_time + λ · ACPR_surrogate,'
              f'block_size = {j.get("block_size", "?")}, '
              f'seeds = {j.get("seeds", [])}\n')
    md.append('| λ | NMSE (dB) | EVM (%) | ACPR_L (dB) | ACPR_U (dB) |')
    md.append('|---|---:|---:|---:|---:|')
    tags = list(j['NMSE_dB'].keys())
    for tag in tags:
        n = j['NMSE_dB'][tag]
        e = j['EVM_%'][tag]
        al = j['ACPR_lower_dB'][tag]
        au = j['ACPR_upper_dB'][tag]
        md.append(f'| {tag} | {n["mean"]:.2f} ± {n["std"]:.2f} | '
                  f'{e["mean"]:.2f} ± {e["std"]:.2f} | '
                  f'{al["mean"]:.2f} ± {al["std"]:.2f} | '
                  f'{au["mean"]:.2f} ± {au["std"]:.2f} |')
        csv_rows.append(['e4_freq_loss', tag, n['mean'], e['mean'],
                         al['mean'], au['mean']])


def main():
    md = [
        '# 实验结果汇总表',
        '',
        '*自动生成,见 `scripts/summary_table.py`。 各 JSON 原始数据在 '
        '`results/<pa>/e*.json` 下。*',
    ]
    csv_rows = []
    section_e1(md, csv_rows)
    section_e2(md, csv_rows)
    section_e3(md, csv_rows)
    section_e4(md, csv_rows)

    os.makedirs(RES, exist_ok=True)
    md_path = os.path.join(RES, 'SUMMARY.md')
    csv_path = os.path.join(RES, 'SUMMARY.csv')
    with open(md_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(md) + '\n')
    with open(csv_path, 'w', encoding='utf-8', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['section', 'name_or_setting', 'v1', 'v2', 'v3', 'v4', 'v5'])
        for r in csv_rows:
            w.writerow(r + [''] * (7 - len(r)))
    print(f'写入 {md_path}')
    print(f'写入 {csv_path}')


if __name__ == '__main__':
    main()
