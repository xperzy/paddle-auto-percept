# 从零开始学 BEVFormer (6) - Encoder - Spatial Cross Attention

SCA的核心是计算bev特征和多视角图像特征的交叉注意力，这里的图像特征是多层的、多视角的。SCA注意力计算的原理，是对于BEV特征上的每个点（实际空间位置的点），找到该位置在各个视角图像上对应的2D位置，然后进行特征采样和注意力计算。也就是说，我们在3D空间中预先定义了一些位置（BEV特征的各个位置，乘以pc_range就是实际的空间点的坐标），通过相机的内外参，可以找到这些3D点在各个2D图像上对应的像素位置。然后再根据可变注意力机制的方式，在各个图像上的参考点上采样若干点，并做特征采样，进而完成注意力的计算，最终完成BEV特征对图像信息的结合。

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(6)%20-%20Encoder%20-%20Spatial%20Cross%20Atte%201c13135fac1780e9be25c76fcb7922e4/image.png)

### SCA的位置：

- SCA是Encoder的一部分，Encoder是一个Self-Attn + Cross-Attn的Transformer结构；
- SCA是一个Cross-Attention结构
- SCA的位置是在TSA（Self-Attn） 和 LayerNorm 之后
- SCA之后还会接LayerNorm，再之后是FFN

### SCA的结构：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(6)%20-%20Encoder%20-%20Spatial%20Cross%20Atte%201c13135fac1780e9be25c76fcb7922e4/image%201.png)

- **输入：**
    - **flattened_feat**: 是包含有多层多视角的图像特征，经过了flattened操作，shape是`[num_cams, seq_l, bs, embed_dim],` 这里的`seq_l`表示多层特征的所有图像token转换成token序列的长度。例如，有4层图像特征，各层的feature map大小分别是: $(H_1 \times W_1)$,$(H_2 \times W_2),(H_3 \times W_3),(H_4 \times W_4)$:
        - $seq\_l = H_1 \times W_1 + H_2 \times W_2 + H_3 \times W_3 + H_4 \times W_4$
    - **bev_queries**: encoder前面经过TSA和LayerNorm计算的结果，shape是`[bs, bev_h * bev_w, embed_dim]`
    - **bev_pos_embeds**: 和Encoder的TSA的bev pos一样，主要用于保存bev特征的位置信息编码。
- **模型结构：**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(6)%20-%20Encoder%20-%20Spatial%20Cross%20Atte%201c13135fac1780e9be25c76fcb7922e4/image%202.png)

基本结构和Deformable Attention基本类似，主要的区别是这里计算的是图像特征和bev queries的交叉注意力。其中feature sampling和Deformable attention的计算方式基本一致：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(6)%20-%20Encoder%20-%20Spatial%20Cross%20Atte%201c13135fac1780e9be25c76fcb7922e4/image%203.png)

这里需要注意的是，reference_points_cam的生成方式：

1. 根据BEV特征图的尺寸生成3D空间中的参考点，shape是**[bs, num_points_in_pillar, h*w, 3]，** 其中，num_points_in_pillar 是在Z轴上采样的参考点的个数，通常为4
    1. 原理类似下图，除了BEV平面上的点之外，还在Z轴上采样了一些具有高度的点（蓝色的点），同样作为参考点
    
    ![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(4)%20-%20EncoderLayer%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac1780c2b9fecb29ff06be86/image%205.png)
    
2. point_sampling：根据相机的内外参，对于每个BEV空间的参考点位置，计算出其投影到各个图像上的2D位置。
    1. 这里会使用lidar2img的投影转换矩阵，进行计算
    2. 得到的2D图像参考点位置，需要过滤掉没投影到该视角下的那些点（这里使用mask记录）
    
    ![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(4)%20-%20EncoderLayer%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac1780c2b9fecb29ff06be86/image%206.png)