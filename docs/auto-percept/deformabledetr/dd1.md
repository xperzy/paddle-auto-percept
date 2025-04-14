# 从零开始学Deformable DETR (1) - Deformable DETR 的整体结构

# 论文地址：

**Deformable DETR: Deformable Transformers for End-to-End Object Detection**

https://arxiv.org/pdf/2010.04159

# Deformable DETR的整体结构：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6Deformable%20DETR%20(1)%20-%20Deformable%20DETR%20%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201ba3135fac17806ca12ae0db5182fdca/image.png)

**从上图中可以看到Deformable DETR的整体结构包含：**

1. **图像特征（Image Feature Maps）**：
    - 是由一个Image Backbone提取的多层多尺度图像特征
2. **编码器（Encoder）**：
    - 是多层EncoderLayer， 每个EncoderLayer是由Multi-Scale Deformable Self-Attention 组成的Transformer结构。
3. **解码器（Decoder）**：
    - 是由object queries和encoder输出计算，包含Self-Attention（object queries之间）和Cross-Attention（object queries 和 encoder输出之间，并且是Deformable Attention）
4. **Deformable Self-Attention**:
    - 图中Encoder里的紫色圆点，表示该特征图位置上，注意力计算后的特征，是由（紫色虚线箭头）上一层（Encoder层）的多尺度特征中不同的采样点进行特征采样加上注意力权重（是一个可学习的线性层）计算得到的。
5. **Deformable Cross-Attention：**
    - 图中Decoder里的红色箭头指向的部分，表示该该层的object queries，会先进行Deformable 注意力计算，是由（红色虚线箭头）Encoder层的多尺度特征中不同的采样点进行特征采样加上注意力权重（是一个可学习的线性层），最终和object query计算得到的。（细节下面章节会详细介绍）。

作者在论文中指出，这样的设计相较于DETR，Deformable DETR会大幅提升小目标物体的检测，同时能够大幅减少训练轮次。

# Deformable Attention Module：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6Deformable%20DETR%20(1)%20-%20Deformable%20DETR%20%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201ba3135fac17806ca12ae0db5182fdca/image%201.png)

**可以从上图中看到Deformable Attention的计算过程：**

1. query特征（左上橙色部分），进行线性变换得到sampling offsets，这个offset将用于对value特征的采样
2. query特征（左上橙色部分），进行线性变换和softmax，得到attention weights，区别于标准的self-attention使用 softmax(q*k’)，这里直接采用线性层得到attention weights
3. Value部分的特征图（左下灰色部分）进行线性变换，得到x_v = W_v * x（中间Values部分）
4. 特征采样：根据offset，获得每个特征图上的对应位置的特征，然后与attention weight进行聚合（可以简单理解为相乘），得到了当前点位置（图中左下角灰色特征图中的黄色小框）的特征，对每个位置都进行这样的计算，就可以得到value特征图经过Deformable attn的所有输出。
5. 输出：最后再经过输出的线性层，得到输出特征。

Deformable DETR相较于DETR最主要的改进是在计算Attention的方式上，从全局的注意力变为基于局部采样的可变形注意力计算，节省了计算的同时能够更好的学习到局部特征，从而获得更好的效果。接下来我们详细分析Deformable DETR的每个模块。