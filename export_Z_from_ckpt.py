import torch
import numpy as np
import torch.nn.functional as F
import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="ckpt_epXXX.pt path")
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--tau", type=float, default=1.0, help="softmax temperature for exporting Z")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 读取 checkpoint
    ck = torch.load(args.ckpt, map_location="cpu")

    if "model" in ck:
        state = ck["model"]
    else:
        state = ck

    # 2. 取出 Z_logits
    if "Z_logits" not in state:
        raise KeyError("Checkpoint does not contain Z_logits")

    Z_logits = state["Z_logits"]   # (K, V)

    # 3. softmax -> Z
    Z = F.softmax(Z_logits / args.tau, dim=0).numpy().astype(np.float32)

    # 4. 保存
    z_path = os.path.join(args.out_dir, "Z.npy")
    np.save(z_path, Z)
    print(f"Saved Z to {z_path}, shape={Z.shape}")

    # 5. 同时保存 voxel_pick.npy（直接拷贝）
    pick = np.load("voxel_pick.npy")
    pick_path = os.path.join(args.out_dir, "voxel_pick.npy")
    np.save(pick_path, pick)
    print(f"Saved voxel_pick to {pick_path}, shape={pick.shape}")

if __name__ == "__main__":
    main()
