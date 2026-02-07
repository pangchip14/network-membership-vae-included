#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2：VAE Fine-tune（在 Stage1 已分化出的 Z 基础上做概率化微调）
==============================================================

你已经在 Stage1 得到了“快速分化”的 Z（membership 已尖锐、z_usage 已拉开、s2 稳定）。
Stage2 的目标不是重新“猛分化”，而是：

1) 固化并微调共享的 Z（共享脑网络空间模式）
2) 用 VAE 的方式对每个样本/时间点的 S（时间动态系数）做概率化推断
3) 通过 KL 约束（+ free-bits）避免 posterior collapse，同时提升稳健性与可解释性

核心形式：
    X ≈ S @ Z
其中：
    - Z: (K, V) 共享，来自 Stage1 的初始化（非常关键）
    - S: (B, K) 每个 batch/timepoint 的隐变量，VAE encoder 输出分布参数

--------------------------------------------------------------
使用方法（推荐）：

# 1) 直接跑（会读取 stage1 ckpt 初始化 Z_logits）
python train_stage2_vae.py \
  --stage1_ckpt /output/out_stage1_fast/ckpt_ep006.pt \
  --out_dir /output/out_stage2_vae_from_fast \
  --amp

# 2) 你也可以指定 epochs / lr / beta_max 等
python train_stage2_vae.py --stage1_ckpt ... --epochs 20 --beta_max 0.3 --free_nats 2.0 --amp

--------------------------------------------------------------
输入要求：
- 当前目录有 voxel_pick.npy（50000 个 voxel index）
- /data/voxel_sub*.mat 存在，且含 voxel_result[field]
- Stage1 ckpt（必须包含 model.Z_logits）

输出：
- out_dir/stage2_log.jsonl
- out_dir/ckpt_epXXX.pt
- out_dir/Z.npy（最终共享 Z）
- out_dir/voxel_pick.npy
"""

import os, glob, re, json, argparse
import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 你关心的 6 个 field（与 Stage1 保持一致）
# -----------------------------
FIELDS = [
    "focus_clear", "focus_fuzzy",
    "inhibition_clear", "inhibition_fuzzy",
    "distancing_clear", "distancing_fuzzy",
]


# -----------------------------
# 一些工具：自然排序 / 选取被试
# -----------------------------
def nat_key(s: str):
    """用于自然排序：sub2 在 sub10 前面。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def pick_subjects(files, n=12):
    """从所有 sub*.mat 中均匀抽取 n 个（与你之前做法一致）。"""
    files = sorted(files, key=nat_key)
    if len(files) <= n:
        return files
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idx]


# -----------------------------
# 数据读取：把 mat 里的 5D voxel_result[field] 变成 (140, Vpick)
# -----------------------------
def load_TV(mat_path, field, pick):
    """
    返回：
        X: (140, Vpick) float32

    处理流程（与你旧代码一致）：
    1) 从 h5 读 voxel_result[field]
    2) transpose / reshape -> (V, 140)
    3) 选取 voxel_pick.npy 指定的 50000 个 voxel
    4) 转置到 (140, Vpick)
    5) 对每个 voxel 做 z-score 标准化（每个样本内部）
    """
    with h5py.File(mat_path, "r") as f:
        x = np.array(f["voxel_result"][field])

    x = x.transpose()                          # (91,109,91,4,35)
    x = np.transpose(x, (0, 1, 2, 4, 3))       # (91,109,91,35,4)
    x = x.reshape(-1, x.shape[3] * x.shape[4]) # (V,140)
    x = x[pick, :].T                           # (140, Vpick)

    # voxel-wise z-score（每个样本内部）
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mu) / sd
    return x.astype(np.float32)


# -----------------------------
# 正则项：usage_kl 与 voxel entropy（与 Stage1 一致）
# -----------------------------
def usage_kl_from_Z(Z, eps=1e-9):
    """
    Z: (K, V)，每列（每个 voxel）在 K 上 softmax 后归一（sum_k=1）

    用途：防止“某些网络彻底不用”（网络塌缩）
    做法：看每个网络平均占用率 z_mean，与均匀分布做 KL
    """
    z_mean = Z.mean(dim=1)  # (K,)
    pi = torch.full_like(z_mean, 1.0 / Z.shape[0])
    return torch.sum(z_mean * (torch.log(z_mean + eps) - torch.log(pi + eps)))

def voxel_entropy(Z, eps=1e-9):
    """
    voxel-wise entropy：
        对每个 voxel v，计算其在 K 个网络上的熵，然后对所有 voxel 求平均
    熵越小 -> membership 越尖锐（越接近 one-hot）。
    """
    return -torch.mean(torch.sum(Z * torch.log(Z + eps), dim=0))


# -----------------------------
# VAE 相关：KL(N(mu,sigma^2) || N(0,1))
# -----------------------------
def kl_normal_standard(mu, logvar):
    """
    返回每个样本（每行）的 KL：shape (B,)
    KL = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mu * mu - 1.0 - logvar, dim=-1)


# -----------------------------
# Stage2 模型：共享 Z + VAE encoder 输出 (mu, logvar)
# -----------------------------
class VAEEncoder(nn.Module):
    """
    输入：x (B, V)
    输出：mu, logvar (B, K)

    说明：
    - 用两层 MLP 做近似后验 q_phi(s|x)
    - logvar 做 clamp 防止数值爆炸
    """
    def __init__(self, V, K, hidden=256, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(V, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(hidden, 2 * K)  # mu, logvar

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.drop(h)
        h = F.relu(self.fc2(h))
        h = self.drop(h)
        out = self.out(h)
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mu, logvar


class MembershipVAE(nn.Module):
    """
    共享参数：
        Z_logits: (K, V)

    前向：
        Z = softmax(Z_logits / tau, dim=0)  # voxel-wise membership（每个 voxel 在 K 上归一）
        mu, logvar = encoder(x)
        s = mu + eps * exp(0.5*logvar)      # 重参数化
        x_hat = s @ Z
    """
    def __init__(self, V, K, hidden=256, dropout=0.0):
        super().__init__()
        self.V, self.K = V, K

        self.Z_logits = nn.Parameter(torch.zeros(K, V))
        nn.init.normal_(self.Z_logits, mean=0.0, std=0.01)

        self.encoder = VAEEncoder(V=V, K=K, hidden=hidden, dropout=dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, tau=1.0):
        Z = F.softmax(self.Z_logits / tau, dim=0)  # (K,V)
        mu, logvar = self.encoder(x)               # (B,K)
        s = self.reparameterize(mu, logvar)        # (B,K)
        x_hat = s @ Z                              # (B,V)
        return x_hat, s, Z, mu, logvar


# -----------------------------
# beta warmup：KL 权重从 0 逐步升到 beta_max
# -----------------------------
def beta_warmup(ep, beta_max=0.3, warmup_epochs=20):
    """
    ep 从 1 开始计数：
    - ep < warmup_epochs: beta 线性增长
    - ep >= warmup_epochs: beta = beta_max
    """
    if warmup_epochs <= 0:
        return beta_max
    if ep >= warmup_epochs:
        return beta_max
    return beta_max * (ep / float(warmup_epochs))


# -----------------------------
# tau 策略：Stage2 不需要太激进（避免把 Z 锁死）
# -----------------------------
def tau_schedule_stage2(ep):
    """
    推荐：先固定一段 tau=0.7（沿用你 Stage1 已分化的尖锐程度），再缓慢降低
    - 1-10: 0.7
    - 11+:  0.5
    """
    if ep <= 10:
        return 0.7
    return 0.5


# -----------------------------
# 优化器：Z 与 encoder 分组 lr（Z 通常稍小，避免破坏 Stage1 结构）
# -----------------------------
def make_optimizer(model, lr_enc=3e-4, lr_Z=2e-3):
    """
    为何要分组？
    - encoder 需要足够学习率去拟合 posterior
    - Z 已经有结构，lr_Z 不宜过大，否则可能“洗掉” Stage1 的好结构
    """
    enc_params, z_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("Z_logits"):
            z_params.append(p)
        else:
            enc_params.append(p)

    opt = torch.optim.Adam(
        [
            {"params": enc_params, "lr": lr_enc},
            {"params": z_params, "lr": lr_Z},
        ],
        betas=(0.9, 0.999)
    )
    return opt


# -----------------------------
# 主训练
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_ckpt", type=str, required=True, help="Stage1 ckpt path (must contain model.Z_logits).")
    ap.add_argument("--out_dir", type=str, default="/output/out_stage2_vae")
    ap.add_argument("--K", type=int, default=14)
    ap.add_argument("--V_target", type=int, default=50000)
    ap.add_argument("--n_subjects", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)

    # 学习率
    ap.add_argument("--lr_enc", type=float, default=3e-4)
    ap.add_argument("--lr_Z", type=float, default=2e-3)

    # VAE 相关：beta 与 free-bits（防 collapse）
    ap.add_argument("--beta_max", type=float, default=0.3)
    ap.add_argument("--kl_warmup_epochs", type=int, default=20)
    ap.add_argument("--free_nats", type=float, default=2.0)

    # 正则（Stage2 一般比 Stage1 更温和）
    ap.add_argument("--lambda_sharp", type=float, default=5e-3)
    ap.add_argument("--lambda_usage", type=float, default=1e-3)
    ap.add_argument("--lambda_s2", type=float, default=0.0)

    # 模型容量
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)

    # 训练设置
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 固定随机性（可复现）
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 读取 voxel_pick.npy
    pick = np.load("voxel_pick.npy")
    assert pick.size == args.V_target, f"voxel_pick has V={pick.size}, expected {args.V_target}"
    V = int(pick.size)

    # 读取并抽取被试
    files = glob.glob("/data/voxel_sub*.mat")
    if len(files) == 0:
        raise FileNotFoundError("No voxel_sub*.mat found under /data/")
    subs = pick_subjects(files, n=args.n_subjects)

    # 构造 (subject, field) 的 72 个样本对
    pairs = [(fp, f) for fp in subs for f in FIELDS]

    # 设备与 AMP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[Stage2] device={device} V={V} K={args.K} pairs={len(pairs)}")
    print(f"[Stage2] stage1_ckpt={args.stage1_ckpt}")
    print(f"[Stage2] batch={args.batch_size} epochs={args.epochs}")
    print(f"[Stage2] lr_enc={args.lr_enc} lr_Z={args.lr_Z}")
    print(f"[Stage2] beta_max={args.beta_max} warmup={args.kl_warmup_epochs} free_nats={args.free_nats}")
    print(f"[Stage2] lambda_sharp={args.lambda_sharp} lambda_usage={args.lambda_usage} lambda_s2={args.lambda_s2}")

    # 构建模型
    model = MembershipVAE(V=V, K=args.K, hidden=args.hidden, dropout=args.dropout).to(device)

    # -------------------------
    # 从 Stage1 ckpt 初始化 Z_logits（关键）
    # -------------------------
    if not os.path.isfile(args.stage1_ckpt):
        raise FileNotFoundError(f"Stage1 ckpt not found: {args.stage1_ckpt}")

    ck = torch.load(args.stage1_ckpt, map_location="cpu")

    # 兼容两种保存方式：
    # 1) ck = {"model": state_dict, ...}
    # 2) ck = state_dict 直接保存
    if isinstance(ck, dict) and "model" in ck:
        st1 = ck["model"]
    else:
        st1 = ck

    # 期望存在 "Z_logits"
    if "Z_logits" not in st1:
        # 有些保存会带前缀，例如 "model.Z_logits"
        # 这里做一次兜底查找
        key_candidates = [k for k in st1.keys() if k.endswith("Z_logits")]
        if len(key_candidates) == 0:
            raise KeyError("Stage1 state_dict does not contain Z_logits.")
        z_key = key_candidates[0]
        print(f"[Stage2] Warning: using '{z_key}' as Z_logits key")
    else:
        z_key = "Z_logits"

    with torch.no_grad():
        model.Z_logits.copy_(st1[z_key])

    # 优化器（分组学习率）
    opt = make_optimizer(model, lr_enc=args.lr_enc, lr_Z=args.lr_Z)

    # 日志文件（jsonl）
    log_path = os.path.join(args.out_dir, "stage2_log.jsonl")

    # -------------------------
    # 训练循环
    # -------------------------
    for ep in range(1, args.epochs + 1):
        # tau：控制 softmax 尖锐度（Stage2 不宜太激进）
        tau = tau_schedule_stage2(ep)

        # beta：KL 权重 warmup
        beta = beta_warmup(ep, beta_max=args.beta_max, warmup_epochs=args.kl_warmup_epochs)

        model.train()
        rng.shuffle(pairs)

        total_loss = total_rec = total_kl = total_ent = total_usage = total_s2 = 0.0
        n_samples = 0

        for fp, field in tqdm(pairs, desc=f"[Stage2] Epoch {ep}/{args.epochs} (tau={tau:.3f}, beta={beta:.3f})"):
            # 每个 (subject, field) 读取一次 X (140,V)，再按 batch 切分 140 个时间点
            X = load_TV(fp, field, pick)
            rows = np.arange(X.shape[0])
            rng.shuffle(rows)

            for i in range(0, len(rows), args.batch_size):
                b = rows[i:i + args.batch_size]
                xb = torch.from_numpy(X[b]).to(device)  # (B,V)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    # 前向
                    x_hat, s, Z, mu, logvar = model(xb, tau=tau)

                    # (1) 重建误差：MSE
                    rec = F.mse_loss(x_hat, xb)

                    # (2) KL：加 free-bits 防 collapse
                    # 直观解释：每个样本的 KL 至少为 free_nats，迫使 latent 携带一定信息
                    kl_per = kl_normal_standard(mu, logvar)        # (B,)
                    kl_per = torch.clamp(kl_per, min=args.free_nats)
                    kl = kl_per.mean()

                    # (3) Z 的正则：usage 与 sharpness
                    ent = voxel_entropy(Z)
                    u = usage_kl_from_Z(Z)

                    # (4) 可选：对 s 的幅度轻微约束（一般可设 0）
                    s2 = torch.mean(torch.sum(s * s, dim=1))

                    # 总损失：Stage2 不求过强尖锐化，而是稳定微调
                    loss = rec + beta * kl + args.lambda_sharp * ent + args.lambda_usage * u + args.lambda_s2 * s2

                # 反向与更新
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                bs = xb.shape[0]
                total_loss += loss.item() * bs
                total_rec  += rec.item() * bs
                total_kl   += kl.item() * bs
                total_ent  += ent.item() * bs
                total_usage += u.item() * bs
                total_s2   += s2.item() * bs
                n_samples  += bs

        # -------------------------
        # Epoch 级指标（看是否稳定）
        # -------------------------
        model.eval()
        with torch.no_grad():
            Z_now = F.softmax(model.Z_logits / tau, dim=0)
            z_usage = Z_now.mean(dim=1).detach().cpu().numpy()

        metrics = {
            "epoch": ep,
            "tau": float(tau),
            "beta": float(beta),
            "free_nats": float(args.free_nats),
            "loss": total_loss / n_samples,
            "rec": total_rec / n_samples,
            "kl": total_kl / n_samples,
            "entropy": total_ent / n_samples,
            "usage_kl": total_usage / n_samples,
            "s2": total_s2 / n_samples,
            "z_usage_min": float(z_usage.min()),
            "z_usage_max": float(z_usage.max()),
        }
        print(metrics)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        # 保存 checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "K": args.K,
                "V": V,
                "epoch": ep,
                "tau": float(tau),
                "beta": float(beta),
            },
            os.path.join(args.out_dir, f"ckpt_ep{ep:03d}.pt")
        )

    # -------------------------
    # 保存最终 Z（用 tau=1.0，便于对比与可视化）
    # -------------------------
    model.eval()
    with torch.no_grad():
        Z_final = F.softmax(model.Z_logits / 1.0, dim=0).detach().cpu().numpy().astype(np.float32)

    np.save(os.path.join(args.out_dir, "Z.npy"), Z_final)
    np.save(os.path.join(args.out_dir, "voxel_pick.npy"), pick)

    print("[Stage2] 保存完成：")
    print("  - log:", log_path)
    print("  - Z.npy:", os.path.join(args.out_dir, "Z.npy"))
    print("  - voxel_pick.npy:", os.path.join(args.out_dir, "voxel_pick.npy"))


if __name__ == "__main__":
    main()
