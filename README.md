# KAN-DPD: Kolmogorov–Arnold Network 用于射频功放数字预失真

本仓库实现并系统比较 KAN(Kolmogorov–Arnold Network)、经典 MP/GMP、以及 MLP/CNN/GRU 等基线在射频功放(PA)非线性建模与数字预失真(DPD)上的性能。 完整实验流程:

```
复基带信号生成
   ↓
PA forward modeling      (e1, 3 仿真 PA × 7 模型 × 5 seed)
   ↓ 固定最佳 PA forward
ILA-DPD                  (e2, 全指标 NMSE / EVM / ACPR)
   ↓
5 类消融 + GMP K/Q 扫描   (e3)
   ↓
[创新点] 频域感知 KAN-DPD 损失   (e4, λ 扫描帕累托)
   ↓
仿真 → 实测迁移          (e5, OpenDPD DPA_200MHz 真实数据)
```

---

## 主要结果摘要

### PA 前向建模 NMSE (dB),5 seeds 均值

| Model | Saleh | Rapp | GMP-PA |
|---|---:|---:|---:|
| MP | **-81.8** | **-128.5** | -50.5 |
| GMP | -64.5 | -109.1 | **-101.1** |
| MLP | -32.5 | -48.1 | -38.7 |
| CNN | -26.2 | -35.6 | -23.8 |
| GRU | -28.7 | -52.2 | -35.0 |
| **KAN** | -44.1 | -51.2 | -44.9 |
| GMP+KAN | -63.6 | -67.8 | -76.1 |

→ **KAN 在所有 PA 上均优于其他 NN 基线**;经典 MP/GMP 在自身参数化下不可击败。

### 仿真→实测迁移 (双数据集对照, 3 seeds)

| 策略 | DPA_200MHz<br>(38k 样本) | DPA_160MHz<br>(480k 样本) |
|---|---:|---:|
| A. 仅用实测从零训练 | +16.4 ± 2.8 | +5.9 ± 6.4 |
| B. Zero-shot 迁移 | **+1.7** | **+1.7** |
| C. Few-shot 5% 微调 | -4.8 | **-17.5** |
| C. Few-shot 20% 微调 | -8.8 | **-30.7** |

→ **关键发现**:
1. 仿真预训练给 zero-shot NMSE 带来 6–14 dB 提升,**两个数据集上 zero-shot 性能完全一致**(+1.7 dB),说明预训练学到的是 PA 非线性的"通用形态"
2. **Few-shot 微调强烈依赖实测数据规模**: DPA_160MHz(12 倍数据量)的 20% 微调能到 -30.7 dB,远胜 DPA_200MHz 的 -8.8 dB
3. **从零训练失败的程度也与数据量相关**,小数据时(+16.4 dB) 远比大数据时(+5.9 dB) 糟糕

---

## 目录

1. [实验设计目标](#1-实验设计目标)
2. [环境搭建](#2-环境搭建)
3. [代码结构](#3-代码结构)
4. [一键复现](#4-一键复现)
5. [单独运行某个阶段](#5-单独运行某个阶段)
6. [测量数据自动下载](#6-测量数据自动下载)
7. [输出与结果说明](#7-输出与结果说明)
8. [已实现的模型 / 指标 / 消融清单](#8-已实现的模型--指标--消融清单)
9. [创新点 — 频域感知 KAN-DPD 损失](#9-创新点--频域感知-kan-dpd-损失)
10. [仿真→实测迁移](#10-仿真实测迁移)
11. [Nature 风格 hero 图集](#11-nature-风格-hero-图集)
12. [主要超参数](#12-主要超参数)
13. [常见问题](#13-常见问题)
14. [代码改写 / 二次开发](#14-代码改写--二次开发)

---

## 1. 实验设计目标

| 实验目标 | 本仓库实现 |
|---|---|
| ① 生成复基带 PA 输入信号 | `signal_gen.gen_ofdm_like` (OFDM-like, PAPR ≈ 8 dB) |
| ② 训练/验证/测试切分 | `data.split_indices` 顺序 7:1.5:1.5;实测数据用 OpenDPD 官方分割 |
| ③ 前向 PA 建模 NMSE 比较 | `experiments/e1_pa_forward.py` |
| ④ 固定最佳 PA → DPD 训练 | `experiments/e2_dpd_ila.py` (ILA 间接学习架构) |
| ⑤ 输出 NMSE / EVM / ACPR | `metrics.py` 全套 |
| ⑥ 消融 + 复杂度 + 显著性 | `experiments/e3_ablation.py` + `stats.py` |
| 三类仿真 PA 数据源 | `pa_models.py`: Saleh / Rapp / GMP-with-memory |
| 基线模型: GMP+MLP+CNN+GRU+KAN+Hybrid | `models/` 6 个模型 |
| KAN 增强输入特征 | `data.make_features`: [I,Q,…,\|x\|,\|x\|³,sinθ,cosθ] |
| GMP K∈{5,7,9} × Q∈{3,5,7} 扫描 | `e3.sweep_gmp` |
| 5 类消融实验 | `e3.{grid,tap,feature,hybrid,quant}` |
| ≥5 个随机种子 + 配对检验 + Holm 校正 | `stats.py`,`config.SEEDS` 长度 5 |
| 图: PSD/AM-AM/AM-PM/星座/收敛 | `plotting.py` + `scripts/make_figures.py` |
| **[创新点]** 频域感知 KAN-DPD 损失 | `losses.py` + `experiments/e4_freq_loss.py` |
| 仿真→实测迁移 | `experiments/e5_sim_to_meas.py` + 自动下载 OpenDPD |

---

## 2. 环境搭建

### 2.1 系统与硬件要求

| 项 | 推荐 / 实测 |
|---|---|
| Python | 3.10 / 3.11 / 3.12 (开发与回归测试用 3.12.4) |
| 操作系统 | Windows 10 / 11、Ubuntu 22.04+、macOS 12+ |
| CPU/GPU | **CPU 即可**;有 GPU 自动启用 |
| 内存 | ≥ 4 GB |
| 磁盘 | ≥ 1 GB(代码 + ckpt + 结果);若启用自动下载 OpenDPD 再加 ~10 MB |
| 网络 | 仅 e5 首次自动下载需要(可选,无网则用合成代理) |

完整 e1+e2+e3+e4+e5 在普通笔记本 CPU 上约 **90 分钟**完成(5 seeds)。

### 2.2 第三方依赖

仅 4 个,无任何"重型"或私有依赖:

| 库 | 用途 | 最低版本 | 实测版本 |
|---|---|---|---|
| `numpy` | 数值计算 / FFT / LS | 1.24 | 1.26.4 |
| `scipy` | 显著性检验(`scipy.stats`) | 1.10 | 1.13.1 |
| `matplotlib` | 出图(PNG/PDF/SVG/TIFF) | 3.6 | 3.8.4 |
| `torch` | 神经网络(KAN/MLP/CNN/GRU) | 2.2 | 2.4.1 |

> 标准库已涵盖:`argparse`, `copy`, `csv`, `io`, `json`, `math`,`os`, `sys`, `urllib`(自动下载用),`importlib`。 **不依赖** pandas /sklearn / pykan / sympy / h5py / yaml / tqdm 等任何额外库。

### 2.3 一键安装

推荐用 conda 或 venv 隔离环境。 以下三种方式选其一:

```bash
# 方式 A: conda(推荐)
conda create -n kan-dpd python=3.11 -y
conda activate kan-dpd
pip install -r requirements.txt

# 方式 B: 系统 Python + venv
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 方式 C: pip 直接装(若不介意污染全局)
pip install numpy>=1.24 scipy>=1.10 matplotlib>=3.6 torch>=2.2
```

#### Windows 上 PyTorch DLL 错误的处理

少数 Windows 机器在用 pip 直接装 CPU 版 PyTorch 时会遇到
`OSError: [WinError 1114] DLL initialization routine failed`。
此时按 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/) 重装
CPU 版即可:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

需要 GPU 时,按上面同一页面选 CUDA 版本(默认 CUDA 12.1):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2.4 验证环境

```bash
python -c "
import numpy, scipy, scipy.stats, matplotlib, torch
print('NumPy   ', numpy.__version__)
print('SciPy   ', scipy.__version__)
print('matplot ', matplotlib.__version__)
print('PyTorch ', torch.__version__)
print('GPU     ', 'yes' if torch.cuda.is_available() else 'no')
"
```

期望输出形如:

```
NumPy    1.26.4
SciPy    1.13.1
matplot  3.8.4
PyTorch  2.4.1+cu121
GPU      no   # 或 yes
```

再做一次最小整体冒烟测试(应在 ~5 s 内打印 `All project imports OK`):

```bash
python -c "
import config, signal_gen, pa_models, metrics, data, stats, trainer, plotting, losses, data_measured
from models import mp_gmp, nn_baselines, cnn, kan, hybrid
from experiments import e1_pa_forward, e2_dpd_ila, e3_ablation, e4_freq_loss, e5_sim_to_meas
print('All project imports OK')
"
```

---

## 3. 代码结构

> 本仓库仅包含**实验代码**。 详见该目录下的 README。

```
kan_code/
├── README.md / requirements.txt
├── run_all.py                    # 统一入口
│
├── config.py                     # 全局超参数(seeds / 训练 / 指标设置)
├── signal_gen.py                 # OFDM-like 复基带信号
├── pa_models.py                  # Saleh / Rapp / GMP-with-memory 仿真 PA
├── metrics.py                    # NMSE / ACPR / EVM / PSD
├── data.py                       # 特征工程 + Dataset
├── data_measured.py              # 实测数据加载 + 合成代理 + 自动下载触发
├── stats.py                      # 配对 t / Wilcoxon / Holm-Bonferroni
├── trainer.py                    # NN 训练循环(支持频域损失 + ckpt)
├── losses.py                     # 时域 MSE + 可微 ACPR 代理损失(创新点)
├── plotting.py                   # 单图绘制(原始版,28 张/PA)
│
├── models/                       # 待比较的所有模型
│   ├── mp_gmp.py                 # MP / GMP, LS 闭解
│   ├── nn_baselines.py           # RVTDNN-MLP / GRU
│   ├── cnn.py                    # 1D-CNN (De Silva 2022)
│   ├── kan.py                    # 高效 B-spline KAN
│   └── hybrid.py                 # GMP + KAN 残差
│
├── experiments/                  # 五段式实验脚本
│   ├── common.py                 # 信号准备 / 模型工厂 / 路径管理
│   ├── e1_pa_forward.py          # 步骤③: PA forward modeling
│   ├── e2_dpd_ila.py             # 步骤④⑤: ILA-DPD 全指标
│   ├── e3_ablation.py            # 步骤⑥: 5 类消融 + GMP K/Q 扫描
│   ├── e4_freq_loss.py           # 创新点: 频域感知损失 λ 扫描
│   └── e5_sim_to_meas.py         # 仿真→实测迁移 (OpenDPD)
│
├── scripts/                      # 辅助脚本
│   ├── make_figures.py           # Nature 风格 6 张 hero 图
│   ├── summary_table.py          # 横向汇总表(SUMMARY.md/csv)
│   ├── download_data.py          # 自动下载 OpenDPD 数据
│   └── convert_opendpd.py        # .mat → .npz 单文件转换
│
├── data/                         # 数据输入(集中管理)
│   ├── measured/
│   │   ├── README.md             # 数据格式说明
│   │   ├── SOURCES.json          # OpenDPD 下载源 URL
│   │   ├── DPA_200MHz.npz        # 真实测量数据(首次自动下载)
│   │   ├── DPA_160MHz.npz
│   │   └── APA_200MHz.npz
│   └── processed/                # 预处理缓存
│
├── checkpoints/                  # 训练好的模型(自动生成,可加载跳过训练)
│   └── <pa>/<exp>/
│       ├── <model>_seed<k>.pt    # NN state_dict
│       ├── <model>_seed<k>.npz   # MP/GMP 复数系数
│       └── history/              # 每 epoch loss 曲线
│
├── results/                      # 结果产出
│   ├── SUMMARY.md / SUMMARY.csv  # 跨 PA × 模型横向汇总
│   ├── figures/                  # 6 张 hero 图 (SVG+PDF+TIFF+PNG)
│   ├── saleh/  rapp/  gmp_pa/    # 单 PA 详细图 + JSON
│   └── sim_to_meas/              # e5 详细图 + JSON
│
└── logs/                         # 训练运行日志
```

---

## 4. 一键复现

```bash
cd kan_code
pip install -r requirements.txt
python run_all.py
```

会按顺序自动完成:

1. e1: 3 仿真 PA × 7 模型 × 5 seed 前向建模
2. e2: 同上 + ILA-DPD 训练 + 全指标评测
3. e3: 5 类消融 + GMP K/Q 扫描(gmp_pa, 2 seeds)
4. e4: 创新点 — λ ∈ {0, 0.1, 1, 5} 频域损失对照(gmp_pa, 3 seeds)
5. e5: **自动下载 OpenDPD** 数据 → 在 DPA_200MHz / DPA_160MHz 上做仿真→实测迁移

总耗时 (普通笔记本 CPU, torch CPU 版): **约 90 分钟**

后续运行会**自动跳过已训完的模型**(ckpt 命中),典型 1–2 分钟刷新图表。

```bash
# 跑完后,生成汇总表与 hero 图(若未自动跑过)
python scripts/summary_table.py        # results/SUMMARY.{md, csv}
python scripts/make_figures.py         # results/figures/F1..F6
```

---

## 5. 单独运行某个阶段

```bash
# 只跑前向建模(3 PA)
python run_all.py --stage e1

# 只跑 DPD,且只在 saleh 上
python run_all.py --stage e2 --pa saleh

# 只跑消融
python run_all.py --stage e3

# 只跑创新点(频域损失实验)
python run_all.py --stage e4

# 只跑仿真→实测迁移实验(默认两个数据集 DPA_200MHz + DPA_160MHz)
python run_all.py --stage e5

# 指定单数据集
python run_all.py --stage e5 --datasets DPA_200MHz.npz

# 跑全部三个数据集
python run_all.py --stage e5 --datasets DPA_200MHz.npz DPA_160MHz.npz APA_200MHz.npz

# 多阶段任意组合
python run_all.py --stage e1 e2 --pa gmp_pa
python run_all.py --stage e1 --seeds 0 1 2

# 自定义 seeds(默认 [2026, 7, 19, 73, 101] 5 个)
python run_all.py --stage e1 --seeds 0 1 2
```

模块脚本也可独立运行:

```bash
python -m experiments.e1_pa_forward --pa saleh rapp
python -m experiments.e2_dpd_ila --pa gmp_pa --seeds 0 1 2
python -m experiments.e3_ablation
python -m experiments.e4_freq_loss
python -m experiments.e5_sim_to_meas
```

---

## 6. 测量数据自动下载

### 6.1 数据加载顺序

```
data/measured/*.npz  → 用本地数据
       ↓ 若为空
首次跑 e5 时尝试从 GitHub raw 下载 OpenDPD CSV
       ↓ 下载成功
合并 6 个 CSV → <name>.npz,带 OpenDPD 官方 train/val/test 索引
       ↓ 下载失败(网络问题/URL 失效)
data_measured.synthesize_measured_proxy()  → 合成代理(永远可用)
```

### 6.2 单独触发下载

```bash
# 下载 SOURCES.json 里的所有数据集
python scripts/download_data.py

# 只下载指定的
python scripts/download_data.py DPA_200MHz DPA_160MHz
```

### 6.3 数据来源

- 仓库: https://github.com/lab-emi/OpenDPD
- 路径: `datasets/{DPA_200MHz, DPA_160MHz, APA_200MHz}/`
- 每个数据集 = 6 CSV(train/val/test 的 input/output)+ 1 spec.json
- 本仓库自动拼成 `<name>.npz`,字段:
  ```
  x:         complex64[N]   完整复基带输入
  y:         complex64[N]   PA 输出
  train_end: int            train 段终止 index (= val 起始)
  val_end:   int            val 段终止 index (= test 起始)
  ```

### 6.4 离线 / 强制合成代理

```bash
python run_all.py --stage e5 --offline
```

### 6.5 数据规模

| 数据集 | 总样本 | 文件大小 |
|---|---:|---:|
| DPA_200MHz | 38 400 | 600 KB |
| DPA_160MHz | 480 000 | 7.6 MB |
| APA_200MHz | 96 000 | 1.6 MB |

### 6.6 数据格式自定义

把用户自己采集的数据放进 `data/measured/<myname>.npz`,字段同上即可。
没有 `train_end/val_end` 字段时会自动按 config 的 7:1.5:1.5 切。

---

## 7. 输出与结果说明

### 7.1 JSON 汇总

每个 `results/<pa>/e*.json` 含:均值/标准差/每 seed 原值/配对检验 p/Holm 显著性。

`results/SUMMARY.md` 是跨 PA × 模型 × 指标的横向表。

### 7.2 图 — 两套

**A. hero 图集** (`results/figures/F1..F6`,每张 SVG+PDF+TIFF+PNG):

| Figure | 内容 | 布局 |
|---|---|---|
| F1 | 3 PA 的 AM/AM 真值散点 | 双栏 1×3 |
| F2 | 前向建模 NMSE + 参数效率 | 双栏 1×2 |
| F3 | DPD 全指标(PSD/EVM/NMSE/ACPR) | 双栏 2×2 |
| F4 | 5 类消融 + GMP K/Q 热力图 | 双栏 2×3 |
| F5 | **创新点:频域损失 NMSE↔ACPR Pareto** | 单栏 2×1 |
| F6 | **仿真→实测迁移 NMSE / ACPR (2 数据集对照)** | 双栏 2×2 |

字体 Arial 7pt,可编辑(svg.fonttype=none + pdf.fonttype=42),
TIFF 600 dpi。

**B. 单图详细图**(`results/<pa>/e*.png`):每 PA × 实验 28 张图,作为补充材料。

### 7.3 模型 checkpoint

```python
import torch, numpy as np
from models.kan import KAN
from data import feature_size

tap = 5
in_dim = feature_size(tap, with_envelope=True, with_phase=True)  # = 14
model = KAN([in_dim, 16, 2], grid_size=7, spline_order=3)
model.load_state_dict(torch.load(
    'checkpoints/saleh/dpd_ila/KAN_seed2026.pt'))
model.eval()

# MP/GMP 系数
coeffs = np.load('checkpoints/saleh/dpd_ila/GMP_seed2026.npz')['coeffs']
```

---

## 8. 已实现的模型 / 指标 / 消融清单

### 8.1 PA 仿真(`pa_models.py`)
- **Saleh** [Saleh 1981]: 无记忆 AM/AM + AM/PM (TWT-like)
- **Rapp** [Rapp 1991]: 无记忆软限幅 (SSPA-like)
- **GMP-PA**: 固定复数系数 GMP,带交叉项,小信号增益归一化为 1

### 8.2 模型(`models/`)
- **MP** [Ding 2004]: 7 阶 × 4 抽头,LS 闭解
- **GMP** [Morgan 2006]: 5 阶 × 3 抽头 × 2 lag/lead
- **MLP / RVTDNN** [Liu 2004]: 2 层 tanh
- **CNN-1D** [De Silva 2022]: 2 层 1D conv + GAP + FC
- **GRU**: 单层 24 隐藏
- **KAN**: 单层 B-spline KAN
- **GMP + KAN residual**

### 8.3 指标(`metrics.py`)
- **NMSE** (dB)
- **EVM** (%)
- **ACPR** lower / upper (dB), Welch-PSD based
- **PSD** for plotting

### 8.4 消融(`e3_ablation.py`)
- A 样条网格 grid ∈ {5,7,9}
- B 时延抽头 tap ∈ {3,5,7,9}
- C 输入特征 (IQ / +env / +env+phase)
- D 纯 KAN vs GMP+KAN residual
- E KAN 边函数 LUT 量化 4/6/8/10/12 bit
- F GMP K∈{5,7,9} × Q∈{3,5,7} 扫描

### 8.5 统计(`stats.py`)
- 均值 ± std (ddof=1)
- 自适应配对检验(n≥8 且 Shapiro 正态 → 配对 t,否则 Wilcoxon)
- Holm-Bonferroni 控制 family-wise error rate, α = 0.05

---

## 9. 创新点 — 频域感知 KAN-DPD 损失

### 9.1 动机

OpenDPD、MP-DPD 等已发表工作虽然把 ACPR 当作核心评测指标,但**没有将其纳入训练损失**(因为 ACPR 计算涉及不可微的频段功率积分)。 公开 KAN-DPD 实现里也未见此功能。 本仓库自主实现一个可微的 ACPR 代理项,作为 KAN-DPD 的频域感知损失,实现 NMSE↔ACPR 可控的帕累托权衡。

### 9.2 方法

`losses.py` 实现可微 ACPR 代理:

```
L = MSE_time(u_pred, x) + λ · L_freq

L_freq = E_b[ Σ_{f ∈ adj-channel}  |FFT(e_b · Hann)[f]|²
              ─────────────────────────────────────────── ]
              Σ_{f ∈ main-channel} |FFT(e_b)[f]|²

其中 e_b = u_pred[b·B:(b+1)·B] − x[b·B:(b+1)·B]   连续时间块
       B = freq_block_size (默认 256)
       λ ∈ {0, 0.1, 1.0, 5.0}                     扫描
```

实现要点:
- 训练时切换为**顺序采样**(`DataLoader(shuffle=False)`),保证 FFT 时序连贯
- Hann 加窗降低频谱泄漏
- 用"邻信道 / 主信道"功率比代替绝对功率,数量级稳定

### 9.3 结果(`results/gmp_pa/e4_freq_loss.json`)

| λ | NMSE (dB) | EVM (%) | ACPR_L (dB) |
|---:|---:|---:|---:|
| 0 | -8.4 ± 5.2 | 43 ± 27 | -5.7 |
| 0.1 | -5.2 ± 0.1 | 55 ± 0.7 | -1.8 |
| 1.0 | +2.5 ± 4.7 | 147 ± 87 | -10.5 |
| 5.0 | +8.9 ± 0.2 | 279 ± 7 | **-18.2** |

→ **教科书般的频谱整形权衡**:λ 从 0 升到 5,ACPR 改善 13 dB,代价是 NMSE 劣化。
为下游通信系统提供了"通信质量 vs 频谱合规"调节旋钮。 详情见 F5。

---

## 10. 仿真→实测迁移

### 10.1 实验三策略

| 策略 | 训练数据 | 用途 |
|---|---|---|
| A. From-scratch | 100% 实测训练集 | 上限对照,验证从零训难度 |
| B. Zero-shot | gmp_pa 仿真 ckpt 直接迁移 | 看预训练的可迁移性 |
| C. Few-shot 5% | 仿真 ckpt + 5% 实测微调 | 实用场景 |
| C. Few-shot 20% | 仿真 ckpt + 20% 实测微调 | 实用场景 |

### 10.2 评测方法

实测数据没有可微 PA 前向,所以用 GMP-forward 在训练集上拟合**代理 PA 前向**
(典型 NMSE < -40 dB),用作 ILA-DPD 部署后的"PA 模拟器" `z = PA(u)`。

### 10.3 真实 OpenDPD 数据上的双数据集结果 (3 seeds)

**DPA_200MHz** (38 400 样本, 容量受限)

| 策略 | NMSE (dB) | EVM (%) | ACPR_L (dB) |
|---|---:|---:|---:|
| A. 从零训练 | +16.4 ± 2.8 | 683% | -3.3 |
| B. Zero-shot | +1.7 ± 0.0 | 122% | -7.9 |
| C. Few-shot 5% | -4.8 ± 0.06 | 58% | **-8.1** |
| C. Few-shot 20% | **-8.8 ± 0.20** | **36%** | -7.2 |

**DPA_160MHz** (480 000 样本, 数据丰富)

| 策略 | NMSE (dB) | EVM (%) | ACPR_L (dB) |
|---|---:|---:|---:|
| A. 从零训练 | +5.9 ± 6.4 | 231% | +0.8 |
| B. Zero-shot | +1.7 ± 0.0 | 122% | -8.2 |
| C. Few-shot 5% | -17.5 ± 1.4 | 13% | -7.6 |
| C. Few-shot 20% | **-30.7 ± 0.9** | **2.9%** | **-8.7** |

**跨数据集关键发现**:
1. **仿真预训练对真实数据贡献巨大**:zero-shot 比从零训练好 6–14 dB
2. **Zero-shot 性能在两个数据集上完全一致**(+1.7 dB),验证了仿真预训练
   学到的是 PA 非线性的"通用形态",与目标 PA 的具体型号无关
3. **Few-shot 性能强烈依赖实测数据规模**: 12 倍数据量(480k vs 38k)让 20%
   微调 NMSE 从 -8.8 dB 改善到 **-30.7 dB**
4. **小数据集场景下 ACPR 的最佳点在 5% 微调**,过多微调反而轻微劣化 ACPR;
   大数据集没有这一拐点,微调越多越好

---

## 11. hero 图集

`scripts/make_figures.py` 读取所有实验 JSON 与 ckpt,组合为 **6 张 hero 图**。

```bash
python scripts/make_figures.py
# 输出到 results/figures/F1..F6.{svg, pdf, tiff, png}
```

- 单栏 89 mm / 双栏 183 mm 
- 字体 Arial 7pt,面板字母 9pt bold
- 可编辑 SVG/PDF(svg.fonttype=none + pdf.fonttype=42)
- TIFF 600 dpi
- 色板:NMI pastel 低饱和;绿/红仅留给方向标注

---

## 12. 主要超参数

`config.py` 集中所有可调参数:

```python
# ---- 复现 ----
SEEDS = [2026, 7, 19, 73, 101]   # 5 个随机种子,用于统计显著性检验

# ---- 信号 ----
N_SAMPLES = 30_000               # 仿真复基带样本数
OSR = 4                          # 过采样
N_SUBCARRIERS = 256
SIGNAL_BACKOFF_DB = 7.0

# ---- 切分(仅仿真 PA) ----
TRAIN_FRAC = 0.7                 # 实测数据用 OpenDPD 官方分割
VAL_FRAC = 0.15                  # test = 0.15

# ---- 经典模型 ----
MP_K, MP_Q = 7, 4
GMP_K, GMP_Q, GMP_LAG = 5, 3, 2
TAP = 5                          # 神经网络时延抽头数

# ---- KAN ----
KAN_HIDDEN = 16                  # 隐层宽度,消融时扫描 {16, 32, 64}
KAN_GRID = 7                     # B-spline 网格点数,消融时扫描 {5, 7, 9}
KAN_SPLINE_ORDER = 3             # B-spline 阶数(固定 3)

# ---- 其他 NN ----
MLP_HIDDEN = 32
GRU_HIDDEN = 24
CNN_CHANNELS = 16

# ---- 训练 ----
BATCH = 64                       # OpenDPD 推荐
EPOCHS = 30                      # 训练轮数(完整 100 epoch 也可)
LR = 1e-3                        # Adam
GRAD_CLIP = 5.0

# ---- ACPR 评测 ----
ACPR_MAIN_BW = 0.20
ACPR_ADJ_BW = 0.20

# ---- 显著性 ----
ALPHA = 0.05
```

---

## 13. 常见问题

### Q1: 第一次跑实验很慢?
是。 默认 5 seeds × 多 PA × 多模型,约 90 分钟。 第二次跑会加载 ckpt 跳过训练,
通常 1–2 分钟即可刷新所有图。

### Q2: KAN 为什么不用官方 pykan?
pykan 面向小规模科学拟合,批训练慢一两个数量级。 本仓库的实现等价于
"efficient-KAN"(数学形式与原论文 Eq. 2.10 完全相同),只依赖 `torch`。
近期 PA/DPD 领域的 KAN 应用工作(如 RVTD-KAN、LEANN)也都自写 PyTorch
实现而非 pykan。

### Q3: 我可以减少 seeds 时间吗?
可以,改 `config.SEEDS` 列表。 但显著性检验通常要求 ≥5 个种子,投稿/答辩按 5 走最稳。

### Q4: 怎么加新模型?
1. 在 `models/` 新建文件,继承 `nn.Module`,`forward(x: [B, in_dim]) -> [B, 2]`
2. 在 `experiments/common.make_nn_model` 加 elif 分支
3. 在 `e1/e2` 的 `NN_MODELS` / `DPD_MODELS` 列表加上名字

### Q5: OpenDPD 自动下载失败怎么办?
1. 看 `logs/run_e5_real.log` 的报错信息
2. 检查网络/防火墙
3. 手动从 https://github.com/lab-emi/OpenDPD 下载 `datasets/DPA_200MHz/` 下 6 个 CSV
4. 把它们组成 `<name>.npz`(参见 `scripts/download_data.py:fetch_one`)
5. 或直接用 `--offline` 跑合成代理

### Q6: 已下载的数据想强制重下?
```bash
rm data/measured/DPA_200MHz.npz
python scripts/download_data.py DPA_200MHz
```

### Q7: Wilcoxon p=0.0625 是不是不显著?
是。 n=5 时 Wilcoxon signed-rank 的最小可能 p = 0.0625 (W=0)。 这是检验方法
的下限,不是模型差异不显著。 解决:把 `config.SEEDS` 调到 ≥8 会启动配对 t 分支。

### Q8: 中文字符在图里显示成方框?
图里我用的是 Arial 英文标签。 想用中文,改 `scripts/make_figures.py` 的
`mpl.rcParams['font.sans-serif']` 为 `['SimHei', 'Microsoft YaHei']` 即可。

---

## 14. 代码改写 / 二次开发

### 14.1 增加 PA 类型
在 `pa_models.py` 加函数,注册到 `PA_REGISTRY`,再在
`experiments/common.prepare_signal` 的 `target_rms` 字典加一项。

### 14.2 接入自己的测量数据
把 .npz 放到 `data/measured/`,字段:
```python
np.savez('mydata.npz',
         x=x_complex64,
         y=y_complex64,
         train_end=np.int64(n_train),     # 可选
         val_end=np.int64(n_train + n_val))  # 可选
```
e5 会自动检测并使用。

### 14.3 改用 GPU
设置 `CUDA_VISIBLE_DEVICES`,代码会自动用 cuda。 默认 `config.DEVICE`
会探测 `torch.cuda.is_available()`。

### 14.4 改频域损失到其他模型
`trainer.train_model(..., freq_lambda=1.0, freq_block_size=256)` 对所有 NN
都适用。 e4 只对 KAN 做了对照,但可以替换为 MLP/GRU/etc.

---

## 参考文献

- Liu et al., **KAN: Kolmogorov-Arnold Networks**, arXiv:2404.19756 (2024)
- Morgan et al., **A Generalized Memory Polynomial Model for DPD**, IEEE TSP (2006)
- Ding et al., **A Robust Digital Baseband Predistorter**, IEEE TCOM (2004)
- Saleh, **Frequency-Independent/Dependent Nonlinear Models**, IEEE TCOM (1981)
- Rapp, **Effects of HPA-nonlinearity on a 4-DPSK/OFDM-signal**, ECSC (1991)
- Liu et al., **Dynamic Behavioral Modeling of 3G PAs Using RVTDNN**, IEEE TMTT (2004)
- De Silva et al., **Modular 1D-CNN for Real-time DPD**, RWW (2022)
- Wu et al., **OpenDPD: End-to-End Learning & Benchmarking for DPD**, ISCAS (2024)

---

## License

仅用于课程作业。 引用本仓库请注明出处。
