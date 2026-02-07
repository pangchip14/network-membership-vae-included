import os, glob, re
import numpy as np
import h5py
from tqdm import tqdm
'''
基于前3个被试的focus_clear字段，从90万多voxel数据中：
1. 筛选出在所有3个被试中都有效（方差>1e-8）的体素
2. 从这些有效体素中随机选择5万个voxel
3. 保存其索引到voxel_pick.npy文件中，供后续使用
相当于保存的是voxel的索引
'''

BASE_FIELD = "focus_clear"
V_TARGET = 50000
MASK_SUBS = 3
SEED = 0

def nat_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def load_field(mat_path, field):
    with h5py.File(mat_path, "r") as f:
        x = np.array(f["voxel_result"][field])
    x = x.transpose()                         # (91,109,91,4,35)
    x = np.transpose(x, (0,1,2,4,3))          # (91,109,91,35,4)
    x = x.reshape(-1, x.shape[3]*x.shape[4])  # (V,140)
    return x                                  # V x 140

files = sorted(glob.glob("data/voxel_sub*.mat"), key=nat_key)
assert files, "No voxel_sub*.mat found"
files = files[:MASK_SUBS]
print("Mask files:", [os.path.basename(x) for x in files])

mask = None
# 筛选出在140这个维度视角上方差比较大的voxel
for fp in tqdm(files, desc="Build mask"):
    VT = load_field(fp, BASE_FIELD)
    v = VT.var(axis=1)
    m = np.isfinite(v) & (v > 1e-8)
    mask = m if mask is None else (mask & m)

idx = np.where(mask)[0]
print("Valid voxels:", idx.size, "/", mask.size)

rng = np.random.default_rng(SEED)
pick = rng.choice(idx, size=V_TARGET, replace=False)
pick.sort()

np.save("voxel_pick.npy", pick.astype(np.int64))
print("Saved voxel_pick.npy with V=", V_TARGET)
