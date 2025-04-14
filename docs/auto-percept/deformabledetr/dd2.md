# 从零开始学 Deformable DETR (2) - 从图像到特征：Image Backbone

DeformableDETR的Image Backbone采用了和DETR一样的CNN结构：ResNet。ResNet模型的输入是2D图像，经过多层卷积之后输出图像特征(feature map)。当使用ResNet作为Deformable DETR的backbone时，区别于图像分类的ResNet，有以下3点不同：

1. 去掉最后的AvgPool层和FC层（logits）层，只保留到Layer4的输出
2. 输出不仅是Layer4的最后一层特征，而是多尺度的特征图，例如，Layer1到Layer4每个block的输出都会被返回
3. 输出的图像特征还会进一步进行线性变换，得到的多尺度图像特征会作为输入给到Encoder当中。

前面2点和DETR保持一致，但是DETR仅使用最后一层特征，而Deformable DETR使用了多尺度特征进行计算。

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR%201b93135fac178006862ced462f6b4412/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201b93135fac17800386ebdc0b9a6b3663/image.png)

# ResNet的整体结构：

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR%201b93135fac178006862ced462f6b4412/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201b93135fac17800386ebdc0b9a6b3663/image%201.png)

ResNet的主要结构包括：

1. Stem Layers： 通常是由卷积层(Conv)，归一化层(BN)，激活层（ReLU），还有一个池化层(Maxpooling)组成。Stem Layers的主要作用，是对图像进行初步的特征提取，为接下来的层提供合适的输入。
2. Layer1 - Layer4： 通常是由多个Residual Blocks组成，每个Layer之间通常会有feature map的2x下采样（通过第一个conv2d的stride参数设置实现）
3. Residual Block：通常是由Conv层，BN层，ReLU层，还有一个bypass层（skip-connect层或者残差连接层）构成。

# ResNet50的结构：

### 1. 网络输入：

- 图像的维度：[N, C, H, W]
    
    例如，WxH = 1280x1080大小的彩色图像，Batch size为8时，输入的维度是：[8, 3, 1080, 1280]
    

### 2. Stem层：

- Conv2D： 7x7卷积，stride=2， padding=3，输出维度64；结果就是对输入图像下采样2倍，channel数为64
- BN + ReLU：BatchNorm归一化，然后使用激活函数
- Maxpool2D：3x3最大值池化，stride=2，padding=1；结果就是对特征图再下采样2倍，channel数不变

Stem层维度变化：[N, C, H, W] → [N, 64, H/4, W/4]

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR%201b93135fac178006862ced462f6b4412/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201b93135fac17800386ebdc0b9a6b3663/image%202.png)

### 3. 残差链接层Layer1-Layer4：

- Layer1到Layer4的Residual Block数量分别是：
    - 3，4，6，3

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR%201b93135fac178006862ced462f6b4412/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201b93135fac17800386ebdc0b9a6b3663/image%203.png)

**Layer1**：3个Residual Block，设置完全一致

1. Conv2D：3x3卷积，stride=1, padding=1，channel数不变；结果就是特征图大小和channel不变
2. BN + ReLU：BatchNorm归一化，然后使用激活函数
3. Conv2D：3x3卷积，stride=1, padding=1，channel数不变；结果就是特征图大小和channel不变
4. Bypass：Identity层（就是输入等于输出），因为特征图不变所以直接与卷积层输出相加

Layer1层维度变化：[N, 64, H/4, W/4] → [N, 64, H/4, W/4]

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR%201b93135fac178006862ced462f6b4412/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201b93135fac17800386ebdc0b9a6b3663/image%204.png)

**Layer2**：4个Residual Block，结构一样，但后三个block设置一致，第一个Block需要考虑channel数和feature map大小的变化

1. Block1：
    1. 结构：Conv2D - BN - ReLU - Conv2D + bypass
    2. 第一个Conv2D：stride=2， padding=1， channel = **2 x** in_channels；结果是feature map的尺寸下采样2倍，channel数增加2倍
    3. 第二个Conv2D：stride=1，padding=1，channel数不变；结果是feature map大小和channel都不变
    4. Bypass：因为conv层的输出变为了[N, 128, H/8, W/8]，所以Bypass需要增加一个1x1卷积，其stride=2, padding=1, channel数为2 x in_channels，这样才能与conv层的输出维度一致并相加
2. Block2-Block4：同Layer1的block，保持输入和输出feature map维度一致

Layer2层维度变化：[N, 64, H/4, W/4] → [N, 128, H/8, W/8]

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR%201b93135fac178006862ced462f6b4412/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201b93135fac17800386ebdc0b9a6b3663/image%205.png)

**Layer3**：6个Residual Block，结构与Layer2一样，都是第一个Block进行（1）feature map下采样2倍，（2）channel数增加2倍（3）bypass增加conv1x1保证输出维度一致；block2-block6 结构和参数一致，feature map 维度保持不变

Layer3层维度变化：[N, 128, H/8, W/8] → [N, 256, H/16, W/16]

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR%201b93135fac178006862ced462f6b4412/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201b93135fac17800386ebdc0b9a6b3663/image%206.png)

**Layer4**：3个Residual Block，结构与Layer2也一样，第一个block同上进行feature map变换，block2-block3结构参数一致，feature map维度保持不变

Layer4层维度变化：[N, 256, H/16, W/16] → [N, 512, H/32, W/32]

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR%201b93135fac178006862ced462f6b4412/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201b93135fac17800386ebdc0b9a6b3663/image%207.png)

- 通常讲的某层的feature map大小，指的是[N, C, H, W]维度中的后两维度，如果说N倍下采样，通常是相较于输入的图像大小。例如，ResNet50的Layer4的feature map大小是32倍下采样，指的是Layer4输出的feature map的后两维度分别是输入图像的1/32。

# 多尺度图像特征：

为了实现更好的检测效果，通常不会只使用最后一层的特征图，这是由于最后一层的特征图已经是1/32的原图大小，每个特征点“覆盖”的区域在原图中较大，容易影响小目标的检测效果。因此，使用多层特征融合的方式，能够较好的提升小目标的检测。在ResNet50中，Layer1-Layer4各层的输入特征常被用来作为多尺度图像特征，输入到检测头中。有时也会只使用Layer2-Layer4的图像特征，再Layer4额外使用卷积操作再获得一层特征，有时输出的特征也会经过一些卷积操作进一步融合（类似FPN网络，在后面的章节会详细介绍）。

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR%201b93135fac178006862ced462f6b4412/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201b93135fac17800386ebdc0b9a6b3663/image%208.png)

在ResNet50网络结构中，Layer1到Layer4的输出特征，可以在模型推理到具体某层时进行判断是否需要作为输出并保存到一个列表中，推理完成后返回列表即可，最终返回的特征图大小分别是（Layer1-Layer4）：

1. [N, 64, H/4, W/4]
2. [N, 128, H/8, W/8]
3. [N, 256, H/16, W/16]
4. [N, 512, H/32, W/32]

## Deformable DETR的多尺度图像特征：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20Deformable%20DETR%20(2)%20-%20%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%89%B9%E5%BE%81%EF%BC%9AImage%20Backbone%201ba3135fac17802aba32de33779f4b61/image.png)

在 Deformable DETR 中，图像特征通过一个名为 image projection 的层进行进一步处理，这个层包含卷积和 GroupNorm，主要目的是对特征进行适当的调整和归一化。

经过ResNet的特征可能具有不同的分辨率或通道数，因此需要通过卷积层进行统一处理，使得输入到Transformer的特征具有相同的维度和更高的表示能力

**GroupNorm** 在batch_size较小（甚至batch size=1）的情况下能提供良好的性能。它对特征分布进行归一化，减小数值不稳定对模型训练的影响

另外，不同的骨干网络生成的特征各异。通过这种标准化处理，Deformable DETR 可以更好地适应各种骨干网络输出的特征

具体来说，DeformableDETR的图像特征一共有4层，分别来自ResNet输出的：Layer2, Layer3, Layer4 和 Layer4层。

其中，前三层都是1x1卷积和GroupNorm归一化操作，第四层会从Layer4额外输出进一步下采样的图像特征，所以使用的是3x3卷积，并且stride=2，padding=1。最终得到的图像特征为：

1. **[N, embed_dim, H/8, W/8]**
2. **[N, embed_dim, H/16, W/16]**
3. **[N, embed_dim, H/32, W/32]**
4. **[N, embed_dim, H/64, W/64]**