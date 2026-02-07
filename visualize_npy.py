def visualize_npy(file_path, max_display=100):
    """
    智能可视化npy文件
    
    参数:
    - file_path: npy文件路径
    - max_display: 最大显示元素数
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 加载数据
    data = np.load(file_path, allow_pickle=True)
    
    print("="*50)
    print(f"文件: {file_path}")
    print(f"形状: {data.shape}")
    print(f"维度: {data.ndim}")
    print(f"数据类型: {data.dtype}")
    print("="*50)
    if data.ndim == 2:
        print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")
        
        fig = plt.figure(figsize=(12, 5))
        
        # 等高线
        ax1 = fig.add_subplot(121)
        X, Y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
        ax1.contourf(X, Y, data, 20, cmap='RdYlBu_r')
        ax1.set_title('contour plot')
        
        # 3D图
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X, Y, data, cmap='coolwarm', alpha=0.8)
        ax2.set_title('3D curve surface figure')
        
        plt.tight_layout()
        plt.savefig('visualization_2d.png')
        plt.show()
        plt.close()
        
        ##########################################################
        col_index = 2500
        col_data = data[:, col_index]

        plt.figure(figsize=(10, 4))
        bars = plt.barh(range(len(col_data)), col_data, height=0.6)

        # 在条形右侧显示数值
        for bar, value in zip(bars, col_data):
            plt.text(bar.get_width() + 0.1, 
                    bar.get_y() + bar.get_height()/2,
                    f'{value}', 
                    va='center',
                    fontsize=12)

        plt.title(f'{col_index}th row data (horizontal bar chart)', fontsize=14, pad=15)
        plt.ylabel('col data', fontsize=12)
        plt.xlabel('number', fontsize=12)
        plt.yticks(range(len(col_data)), [f'col {i}' for i in range(len(col_data))])
        plt.grid(True, alpha=0.3, axis='x')
        plt.savefig('visualization_col_barh.png')
        plt.show()
        plt.close()

# 使用示例
visualize_npy('/output/out_stage2_vae_from_fast/Z.npy')


