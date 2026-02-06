#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST Stage-1 Membership Differentiation (Deterministic)

目标：尽快把共享 membership Z 分化出来（entropy 明显下降，z_usage 拉开），
并避免 S(=encoder输出) 尺度爆炸导致“靠幅度拟合”。

核心策略（为“快分化”而设计）：
1) Z 使用更大学习率 (lr_Z >> lr_enc)
2) 强化尖锐化：entropy 正则 + 较激进的 tau 退火
3) 控制 S 尺度：LayerNorm + s2 penalty + 可选的 s 范数裁剪
4) 使用率防塌缩：usage_kl（弱）

运行：
    python train_stage1_fast.py --amp

依赖：
- voxel_pick.npy (50000,)
- /root/autodl-tmp/time_series/voxel_sub*.mat 里含 voxel_result[field]

输出：
- out_dir/stage1_fast_log.jsonl
- out_dir/Z_stage1_fast.npy
- out_dir/ckpt_epXXX.pt
"""

import os, glob, re, json, argparse
import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


FIELDS = [
    "focus_clear", "focus_fuzzy",
    "inhibition_clear", "inhibition_fuzzy",
    "distancing_clear", "distancing_fuzzy",
]


# -----------------------------
# Data
# -----------------------------
def nat_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def pick_subjects(files, n=12):
    files = sorted(files, key=nat_key)
    if len(files) <= n:
        return files
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idx]

def load_TV(mat_path, field, pick):
    """
    Return X: (140, Vpick) float32
    per-voxel z-score within this sample.
    """
    with h5py.File(mat_path, "r") as f:
        x = np.array(f["voxel_result"][field])
    x = x.transpose()                          # (91,109,91,4,35)
    x = np.transpose(x, (0, 1, 2, 4, 3))       # (91,109,91,35,4)
    x = x.reshape(-1, x.shape[3] * x.shape[4]) # (V,140)
    x = x[pick, :].T                           # (140,Vpick)

    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mu) / sd
    return x.astype(np.float32)


# -----------------------------
# Model (Deterministic Stage1)
# -----------------------------
class FastMembershipStage1(nn.Module):
    """
    Z_logits: (K,V)
    Z = softmax(Z_logits/tau, dim=0)  # voxel-wise simplex membership

    encoder: x(B,V) -> s(B,K)
    s is normalized (LayerNorm) to stabilize scale.
    x_hat = s @ Z
    """
    def __init__(self, V, K, hidden=256, dropout=0.0, s_clip=None):
        super().__init__()
        self.V, self.K = V, K
        self.Z_logits = nn.Parameter(torch.zeros(K, V))
        nn.init.normal_(self.Z_logits, mean=0.0, std=0.01)

        self.fc1 = nn.Linear(V, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, K)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 关键：稳定 S 的尺度（极其重要）
        self.s_norm = nn.LayerNorm(K)

        # 可选：把每行 s 的 L2 范数裁剪到 s_clip（更“硬”的稳定）
        self.s_clip = s_clip

    def forward(self, x, tau=1.0):
        Z = F.softmax(self.Z_logits / tau, dim=0)  # (K,V)

        h = F.relu(self.fc1(x))
        h = self.drop(h)
        h = F.relu(self.fc2(h))
        h = self.drop(h)
        s = self.fc3(h)                # (B,K)
        s = self.s_norm(s)             # (B,K) normalized

        if self.s_clip is not None and self.s_clip > 0:
            # clip each row norm
            norms = torch.norm(s, dim=1, keepdim=True) + 1e-8
            scale = torch.clamp(self.s_clip / norms, max=1.0)
            s = s * scale

        x_hat = s @ Z                  # (B,V)
        return x_hat, s, Z


# -----------------------------
# Regularizers
# -----------------------------
def usage_kl_from_Z(Z, eps=1e-9):
    """
    KL(z_mean || uniform) to prevent collapse (weakly).
    """
    z_mean = Z.mean(dim=1)  # (K,)
    pi = torch.full_like(z_mean, 1.0 / Z.shape[0])
    return torch.sum(z_mean * (torch.log(z_mean + eps) - torch.log(pi + eps)))

def voxel_entropy(Z, eps=1e-9):
    """
    Mean over voxels of entropy over K (lower => sharper membership per voxel).
    """
    return -torch.mean(torch.sum(Z * torch.log(Z + eps), dim=0))


# -----------------------------
# Schedules (aggressive for fast differentiation)
# -----------------------------
def tau_schedule_fast(ep):
    """
    更激进的 tau 退火：尽快把 membership 拉尖。
    你也可以改得更激进，但先用这套一般比较稳：
      1-4:  1.0
      5-8:  0.7
      9-12: 0.5
      13+:  0.35
    """
    if ep <= 4:
        return 1.0
    if ep <= 8:
        return 0.7
    if ep <= 12:
        return 0.5
    return 0.35


def make_optimizer(model, lr_enc, lr_Z):
    enc_params, z_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("Z_logits"):
            z_params.append(p)
        else:
            enc_params.append(p)
    return torch.optim.Adam(
        [
            {"params": enc_params, "lr": lr_enc},
            {"params": z_params, "lr": lr_Z},
        ],
        betas=(0.9, 0.999)
    )


# -----------------------------
# Train
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/root/output/out_stage1_fast")
    ap.add_argument("--K", type=int, default=14)
    ap.add_argument("--V_target", type=int, default=50000)
    ap.add_argument("--n_subjects", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)

    # 强化“快分化”的关键超参
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr_enc", type=float, default=3e-4)
    ap.add_argument("--lr_Z", type=float, default=1e-2)          # 关键：Z 学得快
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)

    # 正则项（默认偏“快分化”）
    ap.add_argument("--lambda_sharp", type=float, default=5e-2)  # 关键：压熵（尖锐化）
    ap.add_argument("--lambda_usage", type=float, default=3e-4)  # 防塌缩（弱）
    ap.add_argument("--lambda_s2", type=float, default=5e-4)     # 关键：抑制 s 爆炸

    # 可选：硬裁剪 s 每行范数
    ap.add_argument("--s_clip", type=float, default=6.0)

    # 训练设置
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pick = np.load("voxel_pick.npy")
    assert pick.size == args.V_target, f"voxel_pick has V={pick.size}, expected {args.V_target}"
    V = int(pick.size)

    files = glob.glob("/root/autodl-tmp/time_series/voxel_sub*.mat")
    if len(files) == 0:
        raise FileNotFoundError("No voxel_sub*.mat found under /root/autodl-tmp/time_series/")
    subs = pick_subjects(files, n=args.n_subjects)
    pairs = [(fp, f) for fp in subs for f in FIELDS]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model = FastMembershipStage1(
        V=V, K=args.K, hidden=args.hidden, dropout=args.dropout, s_clip=args.s_clip
    ).to(device)

    opt = make_optimizer(model, lr_enc=args.lr_enc, lr_Z=args.lr_Z)
    log_path = os.path.join(args.out_dir, "stage1_fast_log.jsonl")

    print(f"[FAST-Stage1] device={device} V={V} K={args.K} pairs={len(pairs)}")
    print(f"[FAST-Stage1] batch={args.batch_size} epochs={args.epochs} lr_enc={args.lr_enc} lr_Z={args.lr_Z}")
    print(f"[FAST-Stage1] lambda_sharp={args.lambda_sharp} lambda_usage={args.lambda_usage} lambda_s2={args.lambda_s2} s_clip={args.s_clip}")
    print(f"[FAST-Stage1] tau schedule: 1.0 -> 0.7 -> 0.5 -> 0.35")

    for ep in range(1, args.epochs + 1):
        tau = tau_schedule_fast(ep)

        model.train()
        rng.shuffle(pairs)

        total_loss = total_rec = total_ent = total_usage = total_s2 = 0.0
        n_samples = 0

        for fp, field in tqdm(pairs, desc=f"Epoch {ep}/{args.epochs} (tau={tau:.3f})"):
            X = load_TV(fp, field, pick)      # (140,V)
            rows = np.arange(X.shape[0])
            rng.shuffle(rows)

            for i in range(0, len(rows), args.batch_size):
                b = rows[i:i + args.batch_size]
                xb = torch.from_numpy(X[b]).to(device)  # (B,V)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    x_hat, s, Z = model(xb, tau=tau)
                    rec = F.mse_loss(x_hat, xb)

                    ent = voxel_entropy(Z)
                    u = usage_kl_from_Z(Z)
                    s2 = torch.mean(torch.sum(s * s, dim=1))

                    loss = rec + args.lambda_sharp * ent + args.lambda_usage * u + args.lambda_s2 * s2

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                bs = xb.shape[0]
                total_loss += loss.item() * bs
                total_rec  += rec.item() * bs
                total_ent  += ent.item() * bs
                total_usage += u.item() * bs
                total_s2   += s2.item() * bs
                n_samples  += bs

        # epoch report
        model.eval()
        with torch.no_grad():
            Z_now = F.softmax(model.Z_logits / tau, dim=0)
            z_usage = Z_now.mean(dim=1).detach().cpu().numpy()

        metrics = {
            "epoch": ep,
            "tau": float(tau),
            "loss": total_loss / n_samples,
            "rec": total_rec / n_samples,
            "entropy": total_ent / n_samples,
            "usage_kl": total_usage / n_samples,
            "s2": total_s2 / n_samples,
            "z_usage_min": float(z_usage.min()),
            "z_usage_max": float(z_usage.max()),
        }
        print(metrics)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        torch.save(
            {"model": model.state_dict(), "K": args.K, "V": V, "epoch": ep, "tau": float(tau)},
            os.path.join(args.out_dir, f"ckpt_ep{ep:03d}.pt")
        )

    # save final Z at tau=1.0 for comparability
    model.eval()
    with torch.no_grad():
        Z_final = F.softmax(model.Z_logits / 1.0, dim=0).detach().cpu().numpy().astype(np.float32)
    np.save(os.path.join(args.out_dir, "Z_stage1_fast.npy"), Z_final)
    np.save(os.path.join(args.out_dir, "voxel_pick.npy"), pick)

    print("[FAST-Stage1] Saved:", os.path.join(args.out_dir, "Z_stage1_fast.npy"))
    print("[FAST-Stage1] Log:", log_path)


if __name__ == "__main__":
    main()
