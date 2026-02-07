# 检测 .mat 文件内容结构，确认维度，方便后续读取
import h5py
import numpy as np

mat_path = "data/voxel_sub4.mat"   # 换成目录里任意一个

with h5py.File(mat_path, "r") as f:
    print("Top-level keys:", list(f.keys()))

    vr = f["voxel_result"]
    print("Fields in voxel_result:", list(vr.keys()))

    # 取一个字段试读
    field = list(vr.keys())[0]
    dset = vr[field]

    print("Raw dataset shape in HDF5:", dset.shape)

    x = np.array(dset)
    print("Numpy array shape before transpose:", x.shape)

    # MATLAB v7.3 常见需要转置
    x = x.transpose()
    print("After transpose:", x.shape)
