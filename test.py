import h5py

# 打开 .mat 文件
with h5py.File('/root/autodl-tmp/time_series/voxel_sub4.mat', 'r') as f:
    # 查看文件结构
    print(list(f.keys()))
    
    # 访问数据
    dataset = f['voxel_result']
    
    # 转换为 numpy 数组
    data_array = dataset[:]
    
    # 查看数据的形状和类型
    print(f"Shape: {data_array.shape}")
    print(f"Data type: {dataset.dtype}")
    
    # 如果是结构体数组，可能需要特殊处理
    #if isinstance(dataset, h5py.Group):
    #    for key in dataset.keys():
    #        print(f"{key}: {dataset[key][:]}")