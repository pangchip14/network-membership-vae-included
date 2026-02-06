import os, glob, re
import numpy as np
import h5py
from tqdm import tqdm

'''
计算每个sub的6个field的voxel time series数据的一些基本统计指标
使用之前生成的voxel_pick.npy来选取体素
'''

FIELDS = [
    "focus_clear","focus_fuzzy",
    "inhibition_clear","inhibition_fuzzy",
    "distancing_clear","distancing_fuzzy",
]

def nat_key(s):
    '''
    自然排序的 key 函数
    例如 "voxel_sub10.mat" 会排在 "voxel_sub2.mat" 后面
    方便按被试编号顺序处理文件
    '''
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def load_TV(mat_path, field, pick):
    '''
    读取指定被试的指定 field 的数据，并选取指定的 voxel 索引
    返回 shape=(T, Vpick)
    其中 T=140 是时间点数，Vpick 是选取的体素数
    读取后对每个 voxel 做 z-score 标准化
    以便后续统计分析
    这样可以避免不同 voxel 信号幅度差异过大影响统计结果
    例如 Vpick=50000
    结果 shape=(140, 50000)
    其中 pick 是一个一维整数数组，表示要选取的 voxel 索引
    例如 pick=np.load("voxel_pick.npy")
    这样可以保证所有被试选取的是同一批 voxel
    方便后续跨被试分析
    例如计算均值、方差等指标
    这些指标在 z-score 标准化后更具可比性
    '''
    with h5py.File(mat_path, "r") as f:
        x = np.array(f["voxel_result"][field]) # 找到指定sub的指定field的数据
    x = x.transpose()                          # (91,109,91,4,35)
    x = np.transpose(x, (0,1,2,4,3))           # (91,109,91,35,4)
    x = x.reshape(-1, x.shape[3]*x.shape[4])   # (V,140)
    x = x[pick, :].T                           # (140, Vpick)

    # z-score per voxel
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mu) / sd
    return x.astype(np.float32)                # zscore处理后的(140, Vpick)

def pick_subjects(files, n=12):
    files = sorted(files, key=nat_key)
    if len(files) <= n:
        return files
    idx = np.linspace(0, len(files)-1, n).round().astype(int)
    return [files[i] for i in idx]

def main():
    pick = np.load("voxel_pick.npy")
    files = glob.glob("/root/autodl-tmp/time_series/voxel_sub*.mat")
    assert files, "No voxel_sub*.mat found"
    subs = pick_subjects(files, n=12)

    print("Using subjects (n=12):")
    for s in subs:
        print("  ", os.path.basename(s))

    # 统计：逐个 sub×field 读取，汇总一些 sanity check 指标
    n_blocks = 0
    mean_list, std_list, nan_frac_list = [], [], []

    for fp in tqdm(subs, desc="Subjects"):
        for field in FIELDS:
            X = load_TV(fp, field, pick)              # 140 x 50k
            mean_list.append(float(np.mean(X)))
            std_list.append(float(np.std(X)))
            nan_frac_list.append(float(np.isnan(X).mean()))
            n_blocks += 1

    print("\nBlocks read（一个block：指定sub指定field的140 x 50k的z-score）:", n_blocks, "(should be 12*6=72)")
    print("Mean over blocks:  mean=%.4f, std=%.4f" % (np.mean(mean_list), np.std(mean_list)))
    print("Std  over blocks:  mean=%.4f, std=%.4f" % (np.mean(std_list), np.std(std_list)))
    print("NaN fraction:      max=%.6f" % (np.max(nan_frac_list)))

if __name__ == "__main__":
    main()
