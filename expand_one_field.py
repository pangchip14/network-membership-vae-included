import h5py
import numpy as np
'''
将 MATLAB 保存的单个 5D 数据恢复成 (V, T) 形状的 numpy 数组
其中 V 是体素数，T 是时间点数（4*35=140）
最终维度：(V, T)=(91*109*91, 140)= (902629, 140)
'''

mat_path = "/root/autodl-tmp/time_series/voxel_sub4.mat"
field = "focus_clear"

with h5py.File(mat_path, "r") as f:
    x = np.array(f["voxel_result"][field])

# 恢复 MATLAB 语义
x = x.transpose()   # -> (91,109,91,4,35)
print("After transpose:", x.shape)

# 折叠 (4,35) -> T=140
# 先换成 (91,109,91,35,4)
x = np.transpose(x, (0,1,2,4,3))
print("After swap:", x.shape)

# 再 reshape
V = x.shape[0] * x.shape[1] * x.shape[2]
T = x.shape[3] * x.shape[4]
x = x.reshape(V, T)

print("Final flattened shape (V, T):", x.shape)
