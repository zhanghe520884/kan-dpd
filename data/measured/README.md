# 测量数据目录

把 OpenDPD 公开测量数据放到这个目录下,文件名建议:
- `DPA_100MHz.npz`
- `DPA_160MHz.npz`
- `DPA_200MHz.npz`
- `APA_200MHz.npz`

每个 `.npz` 应包含两个 key:
- `x` — 复基带输入 (np.complex64, 1D)
- `y` — 对应的 PA 输出 (np.complex64, 1D, 长度同 x)

## 下载方式

参见 OpenDPD 官方仓库:https://github.com/lab-emi/OpenDPD

OpenDPD 给的原始数据通常是 `.mat`,可用 `scripts/convert_opendpd.py` 转成
本目录约定的 `.npz` 格式(如果用户提供了 `.mat`,该脚本会自动处理)。

## 没有真实数据怎么办?

`experiments/e5_sim_to_meas.py` 会自动生成一个**"MeasuredPA-like" 合成代理**
作为占位,使你能立刻跑通"仿真→实测"迁移流程。 拿到真实数据后,只需把
`.npz` 放到本目录下,无需改代码。
