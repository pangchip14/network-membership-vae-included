# 项目简介
本项目主要关注如何从原始的**若干被试者的全脑fMRI数据**中提取**特定voxel在不同的网络中的强度分布情况**。

数据转换与处理流程：

    n个被试 * 每个被试12个field情景* 每个情景（91 * 109 * 91 * 35 * 4）的fMRI原始数据
    ⬇️
    挑选12sub * 6个field/sub = 72个field的数据，并将每个field的5D数据整理成（91*109*91，35*4的形式）
    ⬇️
    对于90多万个voxel数据，在时序上过滤出方差较大的若干voxel，并从中随机挑选50000个voxel进行分析，最终获得72个（140，50000）的数据。
    ⬇️
    送入学习相关的规律分布。假定共有14个网络，最终由上面的数据得到一个共同的（14*50000）的矩阵Z，Z每一列代表一个voxel在14个网络上的分布。

具体的数学分析，参见项目中的/learn.md。
学习过程整体分成两个stage：
- stage1:一个线性分解，快速将X分解为SZ的矩阵乘积。
- stage2:引入变分自编码器VAE，在1的基础上适当微调。

最终呈现效果：
![50000voxel呈现效果](visualization_2d.png)


# 项目结构






/root/codes/inspect_one.py
->
/root/codes/expand_one_field.py
->
/root/codes/make_voxel_pick.py
->
/root/codes/inspect_small_dataset.py
->
/root/codes/train_membership_small.py
->
/root/codes/eval_split_half.py