# 从零开始学 BEVFormer (2) - 多摄像头特征提取

本节我们来学习BEVFormer模型的图像特征提取和预处理部分，这一部分主要是完成从环视图像到输入给Transformer的图像特征之间的操作，涉及到的主要是图像Backbone、FPN还有一些额外的卷积层等。

## 1. 图像输入

BEVFormer 采用了ResNet101 + FPN的结构来提取图片的特征。

在实现的时候，可以简单的使用一个循环，对每个camera都取一个batch的数据，按照维度1（维度0是batch维）保存到Tensor中，这个Tensor的维度是：`(batch_size, num_cams, C, H, W)`，在将数据输入Backbone之前，可以将前2维进行合并`（batch_size * num_cams, c, h, w)` ，因为backbone是2D的图像backbone，其输入是类似`[b, c, h, w]` 的4D Tensor。

例如：

- 单视角下的图像经过处理，得到`1200x1600`的分辨率，一共有6个camera
- 假设`batch_size=2`那输入的Tensor维度是：`(2, 6, 3, 1200, 1600)`
- 输入到backbone之前经过reshape的Tensor维度是：`(12, 3, 1200, 1600)`

## 2. 图像Backbone

图像Backbone部分是ResNet系列，BEVFormer base版本（还有small和tiny版本）使用的是ResNet101结构。模型的输出相较于标准ResNet去掉了logits层和一个pooling层，直接输出layer1到layer4各层的特征图。然后送入到Neck（FPN）中。

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(2)%20-%20%E5%A4%9A%E6%91%84%E5%83%8F%E5%A4%B4%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%201c13135fac1780e189dac8eee6ec9c92/image.png)

上图可以看到，和标准的ResNet不同的地方主要有两点：

1. Style：这部分主要是在BottleNeck中第一个卷积和第二个卷积的stride位置不同。为了和源代码对齐，所以使用了“caffe style”
2. DCN：主要是将最后两个layer，layer3和layer4中的所有卷积，替换为了可变形卷积。 

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(2)%20-%20%E5%A4%9A%E6%91%84%E5%83%8F%E5%A4%B4%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%201c13135fac1780e189dac8eee6ec9c92/image%201.png)

## 3. FPN

FPN主要的作用是将ResNet输出的特征进一步提取特征和融合，从而得到表达能力更强的多尺度图像特征。

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR3D%201bd3135fac17804e8a8fe06f94f339e8/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20DETR3D%20(1)%20-%20%E7%8E%AF%E8%A7%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E3%80%81%E5%9D%90%E6%A0%87%E7%B3%BB%E8%BD%AC%E6%8D%A2%E5%92%8C%E5%9B%BE%E5%83%8F%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%201bd3135fac17807e9638c9fbc27115a9/image%203.png)

从上图可以看到：

- ResNet101的输出特征，在FPN中只使用了最后三层： Layer2， Layer3，Layer4
- 在Layer4上又额外分出一个分支，单独进行了FPN_CONVS, 对特征图进一步下采样
- lateral_conv主要是对特征进行channel维度的降维，然后进一步对特征进行融合
- 融合部分，主要是从最高分辨率（feature map尺寸最小）的层，逐层往上进行特征图的上采样，然后与该尺寸的特征进行相加
- 得到的结果是FPN输出的多尺度特征，feature map尺寸从8倍一直到64倍下采样：
    
    `[0] (bs * cam, 256, h/8, w/8)`
    
    `[1] (bs * cam, 256, h/16, w/16)`
    
    `[2] (bs * cam, 256, h/32, w/32)`
    
    `[3] (bs * cam, 256, h/64, w/64)`
    

所以，整个特征提取的过程：

- 图像输入：
    - `[C, H, W] * num_cams`  → `(batch_size * num_cams, c, h, w)`
- ImageBackbone：
    - `(batch_size * num_cams, c, h, w)` →
    - `[0] (bs * cam, 256, h/4, w/4)`
        
        `[1] (bs * cam, 512, h/8, w/8)`
        
        `[2] (bs * cam, 1024, h/16, w/16)`
        
        `[3] (bs * cam, 2048, h/32, w/32)`
        
- FPN (Neck)：
    - `[1] (bs * cam, 512, h/8, w/8)`
        
        `[2] (bs * cam, 1024, h/16, w/16)`
        
        `[3] (bs * cam, 2048, h/32, w/32)`  →
        
    - `[0] (bs * cam, 256, h/8, w/8)`
        
        `[1] (bs * cam, 256, h/16, w/16)`
        
        `[2] (bs * cam, 256, h/32, w/32)`
        
        `[3] (bs * cam, 256, h/64, w/64)`
        
- 最终输出：
    - `[0] (bs, cam, 256, h/8, w/8)`
        
        `[1] (bs, cam, 256, h/16, w/16)`
        
        `[2] (bs, cam, 256, h/32, w/32)`
        
        `[3] (bs, cam, 256, h/64, w/64)`