# 从零开始学DETR (3) - 捕捉全局关系：Transformer Encoder

![DETR Transformer的整体结构](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(3)%20-%20%E6%8D%95%E6%8D%89%E5%85%A8%E5%B1%80%E5%85%B3%E7%B3%BB%EF%BC%9ATransformer%20Encoder%201b93135fac178057880cf15cf7dede36/image.png)

DETR Transformer的整体结构

DETR 通过使用 Transformer 结构实现特征提取和目标检测。输入图像首先经过 Image Backbone 提取初步的视觉特征，随后进入 Transformer 的 Encoder，通过自注意力机制对全局特征进行进一步的建模和优化，为后续的目标检测奠定基础。本节我们来看Transformer的Encoder部分。

# DETR Transformer的输入

### 图像特征提取

- **输入图像**：
    - 原始图像维度 [N, C, H, W]，C表示图像通道数，通常为3（彩色图像）
- **Backbone 特征提取**：
    - 使用预训练的 CNN（如 ResNet）提取特征
    - 输出特征图 [N, C, H′, W′]，其中 H′,W′ 是特征图的空间维度，通常为原图的 1/32，C为特征维度，通常为512。为了便于理解，我们在此只考虑单层特征的情况。

### 图像特征展开成序列

- 将图像特征从[N, C,  H’, W’] 转换为 [N, L, C]，其中，L = H’ * W’
- 这样做相当于把图像分块（图像特征的形式）排在一起成为一个序列，这样图像特征就从2D的图像特征，转换为一个1D的图像特征的序列，也叫做图像Token序列，其中每个Token的特征维度是C。

### 特征线性变换

- 使用线性层（Linear），将图像Token的维度从C维变换为Transformer的输入特征维度embed_dim
- Token序列的维度从[N, L, C] → [N, L, embed_dim]，其中，L = H’ * W’
- 这里得到的Token序列可表示为： $X_{embed}$

### 添加位置编码

由于图像特征经过展平，原本的2D空间的位置信息丢失，为了保留这些位置信息，DETR 通过 **可学习的正弦位置编码** 为每个特征添加位置信息：

- $X_{pos} = X_{embed} + PE$
- $X_{embed}$为输入的特征，维度是 [L, embed_dim]
- $PE$是位置编码，通过二维度正弦函数生成，维度也是 [L, embed_dim]
    - PE位置编码的详细介绍： TODO（ADD link）

# Transformer Encoder

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(3)%20-%20%E6%8D%95%E6%8D%89%E5%85%A8%E5%B1%80%E5%85%B3%E7%B3%BB%EF%BC%9ATransformer%20Encoder%201b93135fac178057880cf15cf7dede36/image%201.png)

Encoder首先是一个多层的网络结构，其中主要包含：

- 多头自注意力模块（MultiHead Self-Attention）
- 前馈神经网络模块（FFN）
- 归一化模块（LayerNorm）
- 以及残差链接

主要的计算过程可以理解为，对于输入的 Token 序列，计算每个 Token 与所有其他 Token 之间的匹配程度，并用这个匹配权重对所有 Token 进行加权求和，从而生成新的 Token 特征。随后，这些特征经过归一化和前馈神经网络（FFN）处理，最终得到输出 Token。整个过程中，输入和输出的维度保持不变。

## 自注意力机制

Encoder 中的注意力机制通过对图像 Token 序列进行线性变换生成查询 (Q)、键 (K) 和值 (V)。通过计算每两个 Token 的相关性生成注意力权重，用于衡量输出特征需要从各 Token 中获取的信息量。最终，将注意力权重加权的值与线性变换后的 Token 结合，生成融合各 Token 信息的输出特征，从而实现更丰富的特征表达。

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(3)%20-%20%E6%8D%95%E6%8D%89%E5%85%A8%E5%B1%80%E5%85%B3%E7%B3%BB%EF%BC%9ATransformer%20Encoder%201b93135fac178057880cf15cf7dede36/image%202.png)

### **线性变换**

- 输入：图像Token序列，位置编码PE
- 输出：
    - 查询（Q）：维度[N, L, embed_dim]
    - 键（K）：维度[N, L, embed_dim]
    - 值（V）：维度[N, L, embed_dim]
- 操作：
    - 线性层 Q(embed_dim, embed_dim)
    - 线性层 K(embed_dim, embed_dim)
    - 线性层 V(embed_dim, embed_dim)
- 计算过程：
    - $Q = X_{pos} * W^Q$
    - $K = X_{pos} * W^K$
    - $V = X_{embed} * W^V$
    - 其中$X_{pos} = X_{embed} + PE$

### **分头（MultiHead）**

这一步是将Q，K，V分成多个Head：

- [N, L, embed_dim] → [N, num_heads, L, head_dim]
- 其中，head_dim = embed_dim // num_heads

### **计算注意力分数**

- 点积计算注意力分数并缩放：
$Attn = \dfrac{Q\cdot K^\top}{\sqrt{d_k}}$
- 使用 Softmax 归一化，并计算加权值：
$Attn = Softmax(Attn)$
    
    $Attn\ Output = Attn \cdot V$
    

### **头合并**

这一步是将Q，K，V分成多个Head再合并起来，主要是reshape和transpose操作：

- [N, num_heads, L, head_dim] → [N, L, embed_dim]

### **线性变换**

将合并头之后的输出再过一个线性层，用来融合多个头：

- $X_{out} = X * W^{out}$

## **FFN（前馈神经网络）**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(3)%20-%20%E6%8D%95%E6%8D%89%E5%85%A8%E5%B1%80%E5%85%B3%E7%B3%BB%EF%BC%9ATransformer%20Encoder%201b93135fac178057880cf15cf7dede36/image%203.png)

FFN 是 Transformer 中的重要组成部分，主要由线性层、激活层和残差连接构成。它对每个输入 Token 独立地进行非线性变换，进一步提升特征表达能力。

具体结构如下：

1. **线性层1 (Linear)**：
    - 首先通过一个线性变换将输入特征维度从embed_dim映射到更高的维度ffn_dim，通常ffn_dim >embed_dim。
2. **激活函数 (Activation)**：
    - 使用非线性激活函数（通常是 ReLU 或 GELU），引入非线性能力。
3. **线性层2(Linear)**：
    - 再次通过线性变换将特征维度从ffn_dim 投影回embed_dim。
4. **残差连接 (Residual Connection)**：
    - 输入直接跳跃连接到输出，缓解梯度消失问题，并加速模型训练。
5. **归一化层 (Layer Normalization)**：
    - 对最终输出进行层归一化，稳定训练。

公式表示为：

$\text{FFN}(x) = \text{LayerNorm}(x + W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2)$

其中，$W_1$，$W_2$， $b_1$，$b_2$ 为可学习参数，$\sigma$ 为激活函数。

FFN 逐 Token 独立处理，没有引入序列间的交互，专注于每个 Token 的局部特征优化。配合自注意力机制，提升全局特征表达的非线性建模能力。

## **残差连接和归一化**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(3)%20-%20%E6%8D%95%E6%8D%89%E5%85%A8%E5%B1%80%E5%85%B3%E7%B3%BB%EF%BC%9ATransformer%20Encoder%201b93135fac178057880cf15cf7dede36/image%204.png)

**1. 残差连接 (Residual Connection)**

- 类似于ResNet的方式，用于缓解深层网络中的梯度消失问题，同时加快收敛速度。
- **计算**：$y = x + f(x)$

**2. 归一化 (Layer Normalization)**

- 对每个输入的特征维度进行归一化，减去均值并除以标准差，使得每个 Token 的分布更加平稳。
- 给定输入 ，归一化公式为：
    
    $\text{LayerNorm}(y) = \dfrac{y - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$
    
    - $\mu$：特征均值；
    - $\sigma$：特征标准差；
    - $\epsilon$：防止除零的小常数；
    - $\gamma, \beta$：可学习参数，用于调整归一化后的特征

在 FFN 和自注意力机制中，残差连接将原始输入加入到变换后的输出，随后通过归一化层平滑特征分布，最终得到优化的输出。