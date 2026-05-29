"""课程设计实验统一入口。

按以下顺序执行:
  1. 准备复基带信号 -> 应用 Saleh/Rapp/GMP-PA
  2. PA forward modeling (e1):  比较 NMSE,选最佳前向模型
  3. DPD ILA (e2): 在固定 PA 上训练后逆并部署评估 ACPR/EVM/NMSE
  4. 消融实验 (e3): 网格 / 抽头 / 特征 / 纯vs混合 / LUT / GMP 扫描

每个 NN 模型保存 state_dict 到 checkpoints/,重复运行可直接复用。
"""
import argparse
from experiments import (e1_pa_forward, e2_dpd_ila, e3_ablation,
                         e4_freq_loss, e5_sim_to_meas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pa', nargs='+', default=['saleh', 'rapp', 'gmp_pa'])
    p.add_argument('--stage', nargs='+',
                   default=['e1', 'e2', 'e3', 'e4', 'e5'],
                   choices=['e1', 'e2', 'e3', 'e4', 'e5'],
                   help='哪些阶段需要跑(默认全部;e4 频域损失,e5 仿真→实测)')
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    p.add_argument('--offline', action='store_true',
                   help='e5 阶段不尝试自动下载实测数据,直接使用合成代理')
    p.add_argument('--datasets', nargs='+',
                   default=['DPA_200MHz.npz', 'DPA_160MHz.npz'],
                   help='e5 在哪些实测数据集上跑(默认 DPA_200MHz + DPA_160MHz)')
    args = p.parse_args()

    if 'e1' in args.stage:
        for pa in args.pa:
            e1_pa_forward.run_pa_forward(pa, args.seeds)
    if 'e2' in args.stage:
        for pa in args.pa:
            e2_dpd_ila.run_dpd(pa, args.seeds)
    if 'e3' in args.stage:
        e3_ablation.run_all()
    if 'e4' in args.stage:
        e4_freq_loss.run()
    if 'e5' in args.stage:
        for ds in args.datasets:
            print(f'\n>>> e5 on dataset = {ds}')
            e5_sim_to_meas.run(measured_name=ds, offline=args.offline)


if __name__ == '__main__':
    main()
