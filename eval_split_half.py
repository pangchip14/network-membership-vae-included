import os, glob, re
import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


FIELDS = [
    "focus_clear","focus_fuzzy",
    "inhibition_clear","inhibition_fuzzy",
    "distancing_clear","distancing_fuzzy",
]

def nat_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def load_TV(mat_path, field, pick):
    with h5py.File(mat_path, "r") as f:
        x = np.array(f["voxel_result"][field])
    x = x.transpose()                          # (91,109,91,4,35)
    x = np.transpose(x, (0,1,2,4,3))           # (91,109,91,35,4)
    x = x.reshape(-1, x.shape[3]*x.shape[4])   # (V,140)
    x = x[pick, :].T                           # (140, Vpick)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)  # (140, V)

class MembershipModel(nn.Module):
    def __init__(self, V, K, hidden=256):
        super().__init__()
        self.Z_logits = nn.Parameter(torch.zeros(K, V))
        self.net = nn.Sequential(nn.Linear(V, hidden), nn.ReLU(), nn.Linear(hidden, K))
    def forward(self, x):
        Z = F.softmax(self.Z_logits, dim=0)     # K x V
        s = self.net(x)                         # B x K
        x_hat = s @ Z                           # B x V
        return x_hat, Z

def train_on_X(X, K=14, epochs=6, batch_size=64, lr=3e-4, lambda_usage=3e-5,lambda_sharp=1e-2, amp=True, clip=1.0, seed=0):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V = X.shape[1]
    model = MembershipModel(V, K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    rng = np.random.default_rng(seed)
    rows = np.arange(X.shape[0])

    for _ in range(epochs):
        rng.shuffle(rows)
        for i in range(0, len(rows), batch_size):
            b = rows[i:i+batch_size]
            xb = torch.from_numpy(X[b]).to(device)

            with torch.amp.autocast('cuda',enabled=amp):
                x_hat, Z = model(xb)
                rec = F.mse_loss(x_hat, xb)
                eps = 1e-9
                # (a) usage KL: KL(z_mean || pi)  只防塌缩，不强迫均匀
                z_mean = Z.mean(dim=1)  # K
                pi = torch.full_like(z_mean, 1.0 / Z.shape[0])
                usage_kl = torch.sum(z_mean * (torch.log(z_mean + eps) - torch.log(pi + eps)))

                # (b) sharpness: voxel-wise entropy (minimize entropy -> peaky)
                entropy = -torch.mean(torch.sum(Z * torch.log(Z + eps), dim=0))  # scalar

                loss = rec + lambda_usage * usage_kl + lambda_sharp * entropy

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()

    model.eval()
    with torch.no_grad():
        Z = F.softmax(model.Z_logits, dim=0).detach().cpu().numpy().astype(np.float32)  # K x V
    return Z

def greedy_align(Z_ref, Z):
    # 对齐网络：用 cosine similarity 的贪心匹配（K 小时够用）
    K = Z_ref.shape[0]
    A = Z_ref / (np.linalg.norm(Z_ref, axis=1, keepdims=True) + 1e-9)
    B = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S = A @ B.T  # K x K

    perm = [-1]*K
    used = set()
    for k in range(K):
        j = int(np.argmax(S[k]))
        while j in used:
            S[k, j] = -1e9
            j = int(np.argmax(S[k]))
        perm[k] = j
        used.add(j)
    Z_aligned = Z[perm, :]
    return Z_aligned

def metrics(Z1, Z2):
    # voxel cosine similarity
    # centered cosine: remove per-voxel mean (≈1/K) to avoid saturation
    Z1c = Z1 - Z1.mean(axis=0, keepdims=True)
    Z2c = Z2 - Z2.mean(axis=0, keepdims=True)
    a = Z1c.T / (np.linalg.norm(Z1c.T, axis=1, keepdims=True) + 1e-9)
    b = Z2c.T / (np.linalg.norm(Z2c.T, axis=1, keepdims=True) + 1e-9)
    cos = np.sum(a * b, axis=1)
    mean_cos = float(np.mean(cos))
    top1_1 = np.argmax(Z1, axis=0)
    top1_2 = np.argmax(Z2, axis=0)
    top1_agree = float(np.mean(top1_1 == top1_2))

    top3_1 = np.argsort(Z1, axis=0)[-3:,:]
    top3_2 = np.argsort(Z2, axis=0)[-3:,:]
    hit = 0
    V = Z1.shape[1]
    for v in range(V):
        if len(set(top3_1[:,v]).intersection(set(top3_2[:,v]))) > 0:
            hit += 1
    top3_hit = hit / V
    return mean_cos, top1_agree, top3_hit

def main(n_pairs=6):
    pick = np.load("/root/output/out_membership_small_lr3e-4_test2-tau-sharp0/voxel_pick.npy")
    files = sorted(glob.glob("/root/autodl-tmp/time_series/voxel_sub*.mat"), key=nat_key)

    rng = np.random.default_rng(0)
    chosen_files = rng.choice(files, size=min(6, len(files)), replace=False)
    chosen_fields = rng.choice(FIELDS, size=n_pairs, replace=True)

    results = []
    for fp, field in zip(chosen_files, chosen_fields):
        X = load_TV(fp, field, pick)           # 140 x V
        X1, X2 = X[:70], X[70:]

        Z1 = train_on_X(X1, seed=1)
        Z2 = train_on_X(X2, seed=2)
        Z2a = greedy_align(Z1, Z2)

        mcos, top1, top3 = metrics(Z1, Z2a)
        results.append((os.path.basename(fp), field, mcos, top1, top3))

    for r in results:
        print({"file": r[0], "field": r[1], "mean_voxel_cos": r[2], "top1_agree": r[3], "top3_hit": r[4]})

if __name__ == "__main__":
    main()
