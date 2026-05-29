"""全局配置:数据规模、模型超参、训练与评测设置。"""
import torch

# ---- 复现与设备 ----
SEEDS = [2026, 7, 19, 73, 101]           # 5 个随机种子,用于配对显著性检验
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 信号生成 ----
N_SAMPLES = 30_000                       # 复基带采样点数
OSR = 4                                  # 过采样率
N_SUBCARRIERS = 256                      # OFDM 子载波数
SIGNAL_BACKOFF_DB = 7.0                  # 输入回退（PAPR 控制起点）
# 注:目标 RMS 在 main 中根据 PA 类型选择,使预失真器工作在可达域内

# ---- 切分(顺序切分,避免 test 泄漏) ----
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# ---- 经典模型超参 ----
# MP / GMP 的默认值,在消融实验中再做 K∈{5,7,9}, Q∈{3,5,7} 扫描
MP_K = 7
MP_Q = 4
GMP_K = 5
GMP_Q = 3
GMP_LAG = 2

# ---- 神经网络共享:时延抽头数 ----
TAP = 5                                  # 延迟线长度,消融中扫 {3,5,7,9}

# ---- KAN(B-spline) ----
KAN_HIDDEN = 16                          # 隐层宽度,扫 {16,32,64}
KAN_GRID = 7                             # 样条网格数,扫 {5,7,9}
KAN_SPLINE_ORDER = 3                     # 样条阶数,固定 3

# ---- MLP / GRU / CNN ----
MLP_HIDDEN = 32
GRU_HIDDEN = 24
CNN_CHANNELS = 16

# ---- 训练 ----
BATCH = 64                               # OpenDPD 推荐 batch=64
EPOCHS = 30                              # 训练轮数(完整 100 epoch 也可)
LR = 1e-3                                # Adam 起始学习率
GRAD_CLIP = 5.0

# ---- 指标:ACPR ----
ACPR_FS = 1.0                            # 归一化采样率
ACPR_MAIN_BW = 0.20                      # 主信道带宽
ACPR_ADJ_BW = 0.20                       # 邻信道带宽
ACPR_GAP = 0.0

# ---- 显著性检验 ----
ALPHA = 0.05                             # 显著性水平
