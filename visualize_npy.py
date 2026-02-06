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
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 热力图
        im1 = axes[0, 0].imshow(data, cmap='viridis')
        axes[0, 0].set_title('colormap heatmap')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 灰度图
        axes[0, 1].imshow(data, cmap='gray')
        axes[0, 1].set_title('grayscale heatmap')
        
        # 等高线
        X, Y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
        axes[1, 0].contourf(X, Y, data, 20, cmap='RdYlBu_r')
        axes[1, 0].set_title('contour plot')
        
        # 3D图
        from mpl_toolkits.mplot3d import Axes3D
        ax3d = fig.add_subplot(224, projection='3d')
        ax3d.plot_surface(X, Y, data, cmap='coolwarm', alpha=0.8)
        ax3d.set_title('3D curve surface figure')
        
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
visualize_npy('/root/output/out_stage2_vae_from_fast/export_ep006/Z.npy')


