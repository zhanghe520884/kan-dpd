"""NN 训练循环。

特性:
  * 记录 train/val 收敛曲线
  * 早停 + 最佳验证模型保存
  * checkpoint 落盘到 results/.../ckpt/  以便快速复现
  * 支持 hybrid 模型(extra 输入)
"""
import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

import config
from losses import combined_dpd_loss


def _eval(model, loader, device, use_extra):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if use_extra:
                f, y, e = batch
                f = f.to(device); y = y.to(device); e = e.to(device)
                p = model(f, e)
            else:
                f, y = batch
                f = f.to(device); y = y.to(device)
                p = model(f)
            preds.append(p.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def train_model(model, train_ds, val_ds, *, epochs=None, lr=None, batch=None,
                use_extra=False, device=None, verbose=False,
                ckpt_path: str | None = None, history_path: str | None = None,
                freq_lambda: float = 0.0, freq_block_size: int = 256):
    """通用训练循环。 返回(model, history)。

    Args:
        ckpt_path:  best state_dict 落盘路径(.pt);若存在则训练前直接加载。
        history_path:  每 epoch 的 loss 曲线落盘路径(.json)。
    """
    epochs = epochs or config.EPOCHS
    lr = lr or config.LR
    batch = batch or config.BATCH
    device = device or config.DEVICE
    model = model.to(device)

    # ---- 已有 ckpt 则跳过训练 ----
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            if history_path and os.path.exists(history_path):
                with open(history_path) as fp:
                    history = json.load(fp)
            else:
                history = {'train_mse': [], 'val_mse': []}
            return model, history
        except Exception:
            pass  # 加载失败就重训

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # 频域损失需要时序连贯,启用顺序采样并把 batch 调到 ≥ FFT 块长
    use_freq = freq_lambda > 0
    if use_freq:
        eff_batch = max(batch, freq_block_size)
        train_loader = DataLoader(train_ds, batch_size=eff_batch,
                                  shuffle=False, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                                  drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch * 4, shuffle=False)

    best_val = float('inf')
    best_state = None
    history = {'train_mse': [], 'val_mse': []}

    for ep in range(epochs):
        model.train()
        total = 0.0; n = 0
        for batch_data in train_loader:
            if use_extra:
                f, y, e = batch_data
                f = f.to(device); y = y.to(device); e = e.to(device)
                p = model(f, e)
            else:
                f, y = batch_data
                f = f.to(device); y = y.to(device)
                p = model(f)
            loss = combined_dpd_loss(p, y, lam=freq_lambda,
                                     block_size=freq_block_size)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
            total += loss.item() * f.shape[0]
            n += f.shape[0]
        sched.step()
        train_mse = total / max(n, 1)
        preds, targets = _eval(model, val_loader, device, use_extra)
        val_mse = float(np.mean((preds - targets) ** 2))
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d}  train={train_mse:.5f}  val={val_mse:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- 保存 ckpt / history ----
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    if history_path:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as fp:
            json.dump(history, fp)
    return model, history


def predict_full(model, feats_np, idx_np, device=None, batch=4096,
                 extra_np=None):
    """对 idx_np 指定位置批量预测,返回与之对齐的复值数组。"""
    device = device or config.DEVICE
    model = model.to(device).eval()
    preds = np.zeros((len(idx_np), 2), dtype=np.float32)
    use_extra = extra_np is not None
    with torch.no_grad():
        for start in range(0, len(idx_np), batch):
            sl = idx_np[start:start + batch]
            f = torch.from_numpy(feats_np[sl]).to(device)
            if use_extra:
                e = torch.from_numpy(extra_np[sl]).to(device)
                p = model(f, e)
            else:
                p = model(f)
            preds[start:start + batch] = p.cpu().numpy()
    return (preds[:, 0] + 1j * preds[:, 1]).astype(np.complex64)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
