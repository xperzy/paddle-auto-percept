# 从零开始学 DETR3D - DETR3D的整体架构

# 论文地址：

**DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries**

https://arxiv.org/pdf/2110.06922

# DETR3D整体结构：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20DETR3D%20-%20DETR3D%E7%9A%84%E6%95%B4%E4%BD%93%E6%9E%B6%E6%9E%84%201bd3135fac1780bba99cd8ff65b4e869/image.png)

- DETR3D的输入：环视图像，表示同一时刻（帧）主车各个方向的camera采集到的图像数据
- DETR3D的输出：3D障碍物，区别于DETR和DeformableDETR做2D图像的目标检测，这里输出的是3D空间中的障碍物bbox。

## 具体的步骤：

**首先是多视角图像特征提取：**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20DETR3D%20-%20DETR3D%E7%9A%84%E6%95%B4%E4%BD%93%E6%9E%B6%E6%9E%84%201bd3135fac1780bba99cd8ff65b4e869/image%201.png)

这一步的输入是多视角图像，但是对于图像backbone和neck来说仍然是2D的图像backbone和neck，那么这一步的计算可以理解为：

- 原本的2D图像backbone，单视角的输入格式是：[B, C, H, W]，其中B是batch size，C是通道数通常为3，HW是图像尺寸
- **多视角图像输入：**
    - **[B, N, C, H, W] → [B*N, C, H, W] →Backbone → [B*N, D, H’, W’] → [B, N, D, H’, W’]**
    - 就是把camera数量和batch维度结合，进入backbone之后相当于对每张图分别计算图像特征，输出再reshape回原来的维度即可。

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20DETR3D%20-%20DETR3D%E7%9A%84%E6%95%B4%E4%BD%93%E6%9E%B6%E6%9E%84%201bd3135fac1780bba99cd8ff65b4e869/image%202.png)

- 如上图，经过FPN的图像特征，不再是单层的图像特征，而是多尺度的多层特征，对于每个视角的图像，都会有多层特征，这些特征都会被用来进行下一步计算。

**其次是Decoder的Self-Attention部分：**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20DETR3D%20-%20DETR3D%E7%9A%84%E6%95%B4%E4%BD%93%E6%9E%B6%E6%9E%84%201bd3135fac1780bba99cd8ff65b4e869/image%203.png)

- 首先是self-attention，这部分是object queries之间的标准自注意力计算（如图中，虚线框里的黑色箭头）

**然后是Decoder的Cross-Attention部分：**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20DETR3D%20-%20DETR3D%E7%9A%84%E6%95%B4%E4%BD%93%E6%9E%B6%E6%9E%84%201bd3135fac1780bba99cd8ff65b4e869/image%204.png)

1. 首先是从object query的position embedding，经过线性层（sub-network）计算出3D参考点位置（如图中蓝色箭头所示）
2. 有了这些参考点位置，再经过3D到2D的投影计算，得到3D点到每一个View图像的2D坐标（如图中绿色箭头所示），这些坐标点表示特征采样点的位置。
3. 有了各个View上的采样点位置，进行特征采样，然后再将这些采样的特征进行聚合，并进行注意力加权等操作，最后用来更新object queries。（如图中红色箭头所示）
    1. 这里的特征采样，就相当于是拿到这些采样点位置的特征，如果是非整数坐标，那么需要进行双线性差值，根据坐标点临近位置的特征进行计算得到该点的特征，所以叫做特征采样。

**最后目标检测：**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20DETR3D%20-%20DETR3D%E7%9A%84%E6%95%B4%E4%BD%93%E6%9E%B6%E6%9E%84%201bd3135fac1780bba99cd8ff65b4e869/image%205.png)

经过decoder之后，object queries再接上detection head，包括分类头和框回归头，就可以以类似DETR的方式进行训练和预测了。

DETR3D的设计比较简洁，主要结构类似于DETR的Decoder部分，注意力计算是基于Deformable Attention，主要的区别是任务变成了3D目标检测，输入也变成了环视图像，需要进行投影变换得到各个视角下的采样点位置。主要的难点都在实现时候的细节处理，所以下一节开始，我们开始从零实现DETR3D，并详细讲解每一个计算过程。