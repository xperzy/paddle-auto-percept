# 从零开始学 BEVFormer (3) - Encoder的整体结构

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(3)%20-%20Encoder%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac17808a887adb4734664e80/image.png)

上图展示了BEVFormer Head部分（Backbone之后的部分）的结构图，可以看到其主要包括：

- Transformer：是一个Encoder-Decoder结构的多层Transformer结构
- Class Embeds： 分类头
- Bbox Embeds：框回归头

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(3)%20-%20Encoder%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac17808a887adb4734664e80/image%201.png)

Transformer部分，是BEVFormer的核心部分，包括：

- Encoder：多层的注意力结构，主要是结合图像特征和前序BEV特征，来优化更新当前的bev特征（query）
- Decoder：多层的注意力结构，主要是结合encoder的输出和object query

本节我们先来看Encoder的部分：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(3)%20-%20Encoder%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac17808a887adb4734664e80/image%202.png)

## Encoder的输入：

Encoder的输入可以分为下面3个部分：

- bev query：当前帧的bev特征查询
- prev bev embeds：前序帧的bev特征
- image features: 多尺度图像特征

### BEV Query：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(3)%20-%20Encoder%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac17808a887adb4734664e80/image%203.png)

**BEV Query** 是 BEVFormer 的 Transformer 架构中的输入查询（query），**用来生成 BEV 特征**。它通常是固定初始化的可学习参数，表示在鸟瞰图（Bird's Eye View, BEV）中离散化采样的各个位置。BEV Query 的数量对应 BEV 网格划分的数量，每个 Query 代表 BEV 空间中的一个点（一个格子）。需要注意的是，**BEV Query本身并不包含特征**，只是网络用来学习 BEV 空间中各点的特征表示的起点。

- BEV Query在注意力计算的时候，还需要加入位置信息，用来区别不同位置的bev query，这个位置信息叫做bev_pos_embeds，是通过positional encoding编码得到的。简单来说，就是使用一种映射，将每个位置的BEV点的序号，转换成与bev query维度相同的特征。在计算注意力的时候，query和key的部分会将bev query和bev pos embeds相加，从而达到融合位置信息的目的。注意：这里的位置是bev query的位置。
- BEVFormer中，还融合了canbus的信息，直接通过一个MLP将canbus中的值映射到bev query的维度也与之相加。canbus中主要存的是当前主车的一些姿态、速度、朝向等信息，通过这一步，可以将主车的一些真实位置信息也融合进来。

### Prev BEV Embeds：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(3)%20-%20Encoder%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac17808a887adb4734664e80/image%204.png)

就是前帧的BEV特征，经过encoder的输出被保存下来用于融合时序信息。简单来说，**BEV 特征** 是 BEV Query 经过 Encoder 处理后的输出结果，是**融合了多视角特征并映射到 BEV 空间后的特征表示。**

### Image Features：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(3)%20-%20Encoder%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac17808a887adb4734664e80/image%205.png)

图像特征部分，除了前文介绍的多尺度多视角图像特征外（经过了FPN的输出），BEVFormer还融合了相机和尺度维度的embedding，这两个embedding可以理解是两组可学习的权重，用来提升Transformer的表示能力。

**Camera Embedding：**

- 每个视图（camera view）都会有一个独立的嵌入向量，是一个可学习的参数，表示每个相机视图的特定信息。
- 主要的作用是：
    - 为每个试图提供一个唯一的表示（因为每个视图的视角等都不一样），让Transformer知道哪个特征来自哪个相机

**Level Embedding：**

- 每个尺度的特征都会有一个独立的嵌入向量，是一个可学习参数，表示每层特征的特定息。
- 主要的作用是：
    - 区分不同的特征层级：不同层级特征具有不同的分辨率和语义信息，这样显式地提供了特征的层级信息，使Transformer能够区分来自不同层级的特征。

### 输入数据的维度：

### BEV Query：

- query：`[batch_size, bev_h * bev_w, embed_dim]`
- pos_embeds: `[batch_size, bev_h * bev_w, embed_dim]`

### Prev BEV Embeds：

- `[batch_size, bev_h * bev_w, embed_dim]`

### Image Features：

- `[batch_size, num_cams, embed_dim, h1, w1]`
    
    `[batch_size, num_cams, embed_dim, h2, w2]`
    
    `[batch_size, num_cams, embed_dim, h3, w3]`
    
    `[batch_size, num_cams, embed_dim, h4, w4]`  
    
    concat到一起然后reshape：
    
    - `[num_cams, (h1*w1 + h2*w2 + h3*w3 + h4*w4), batch_size, embed_dim]`

注意力等部分的计算是在EncoderLayer中定义的，下一节我们介绍EncoderLayer的结构。