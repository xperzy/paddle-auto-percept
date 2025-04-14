# 从零开始学 BEVFormer (5) - Encoder - Temporal Self Attention：跨时间帧特征融合

BEVFormer的Encoder部分，最主要的目标是学习BEV特征，而BEV特征则是作为注意力计算中的Query，通过两种不同的注意力机制，分别获得时序信息，和图像信息，那就是Temporal Self Attention(TSA)和 Spatial Cross Attention(SCA)。本节我们先来介绍TSA。

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image.png)

TSA的目标是学习到前序帧的信息，也就是通过前序帧的BEV特征（bev_embeds）与当前帧结合，按照注意力机制的方式更新。具体来说：

### 输入：

- 当前帧的bev查询：bev_query； shape是`[bs, bev_h*bev_w, embed_dim]`
- 之前帧的bev特征：bev_embeds； shape是`[bs, bev_h*bev_w, embed_dim]`

### TSA分两种情况处理时序信息：

### **1. 有前一帧时：**

注意力计算的q：前一帧的bev特征与当前帧的bev query（加上位置编码）合并

- `x_q = [prev_bev, bev_query + pos_embed]`

注意力计算的value：前一帧的bev特征与当前帧的bev query合并

- `value = [prev_bev, bev_query]`

### **2. 没有前一帧时：**

注意力计算的q：当前帧的bev query（不加位置编码）与当前帧的bev query（加上位置编码）合并

- `x_q = [bev_query, bev_query + pos_embed]`

注意力计算的value：当前帧的bev query与当前帧的bev query合并

- `value = [bev_query, bev_query]`
- 可以看到，这里的时序合并比较“粗糙”，在BEVFormer论文之后的一些工作也确实对这一部分进行了更好的设计，取得了更好的效果。

上面的组合中，在计算x_q的时候只有后半部分加上了位置信息，这么做的原因是：

- `prev_bev` 是从前一帧直接传递过来的全局 BEV 表征。它已经是空间位置对齐的特征，包含了完整的空间上下文信息，**无需再额外添加位置编码**。
- 如果对 `prev_bev` 再加位置编码，反而可能破坏它原本的空间结构信息
- `bev_query` 是基于当前帧传感器信息生成的查询特征，位置编码的加入是为了提升其在时间序列中的空间定位能力。
- `pos_embed` 提供额外的几何信息，使得当前帧的查询能够更有效地与历史特征进行匹配

我们现在有了x_q 和 value，接下来要**计算可变形注意力。**为了计算可变形注意力，需要基于参考点和偏移量计算采样点，然后在bev特征的对应位置上进行特征采样。

### 参考点是如何计算的？

**2D参考点的说明：**

- 见上一节内容

**时序对齐：**

参考点是在bev 空间上含有具体位置信息的均匀采样点。对于前一帧的bev特征prev_bev和当前帧的bev特征bev_query，每个特征点，都是具有实际3D位置信息的，也就是说，每个特征，都被定义表示为空间中某一特定位置（3D空间中的平面位置）的特征表示。每一帧的BEV范围都是一样的，都是基于当前时刻，以主车的位置为中心（bev特征的中心点），前后左右bev_h, bev_w大小，单位通常是m。

但是，从**前一帧到当前帧，主车的位置和朝向发生了变化。**也就是说，对于主车来说，上一帧bev上的某个位置（例如正前方3m位置）在当前帧的BEV特征图上并不是在同一个坐标位置（可能是正前方2m位置，假设一帧主车未改变朝向向前移动了1m），也就是说，原本上一帧(x,y)位置的bev特征，代表的是真实空间里，(X, Y)位置，到了当前帧，由于主车位置发生了变化，真实空间中（X,Y）位置的特征表示就不再是特征图中的(x,y)了，而是当前特征图中的(x’, y’)位置了。 例如，如果主车从前一帧到当前帧，向右上移动了一个位置，那么上一帧bev的(1,1)位置，对于当前帧来说，其实是（2，0）位置。

这样会带来什么问题呢？主要是，BEVFormer会将前后帧的BEV特征拼起来进行注意力计算，在计算注意力的时候，会根据参考点位置进行特征采样。我们希望参考点的位置都是相对于当前帧的，所以，当我们取参考点的时候，**之前帧的参考点坐标是需要加上主车移动的偏移量**，才能将之前帧的bev特征上的每个位置，对应到当前帧的每个参考点位置上。

如何计算前一帧对于当前帧的位置偏移呢？

从img_meta中可以读取到`（delta_x, delta_y）`，这个向量可以计算出主车从上一位置到当前位置的角度和距离变化。可以理解成是整个bev特征图（按中心点）从某个位置移动到了另一个位置，然后又按照某个角度进行了旋转。

- `(delta_x, delta_y)`的计算是**在CustomNuscenesDataset代码中**，从`ego2global_translation`读取了`x，y`方向的偏移，然后在处理时序数据的时候(在`union2one`方法里)，做了前后帧相减，才将前后帧主车ego坐标系相对于global坐标系的translation，变为了前后帧的translation，也就是这里的`delta_x和delta_y`。

从img_meta中也可以读取到 `ego_angle`，表示当前帧的航向角（yaw角）。这个航向角，是从`ego2global_rotation`中提取的，表示的是**自车朝向相对于全局坐标系 X 轴的旋转角度。**

- **在CustomNuscenesDataset代码中**，首先从`ego2global_rotation`中，读取了`yaw`角度，然后同样是在`union2one`方法里，做了前后帧的角度相减，输出的变量里，**在最后一个位置存了前后帧角度的变化，倒数第二个位置保存的还是当前帧的航向角**。
- 在 nuScenes 数据集中，`ego2global_translation` 和 `ego2global_rotation` 用于描述**自车（ego vehicle）在全局坐标系（global coordinate system）中的位姿（pose）**。它们的含义如下：
    1. **ego2global_translation**: 自车在**全局坐标系**（global frame）下的位置，是一个 3D 向量 `[x, y, z]`，单位通常是**米（meters）**， `x,y` 代表水平位置，`z` 代表高度（在BEV任务中通常被忽略）
    2. **ego2global_rotation**: 自车在**全局坐标系**（global frame）下的旋转（方向），是一个4元数 `[qx, qy, qz, qw]`，这个4元数可以用来表示 **航向角（yaw）、俯仰角（pitch）和翻滚角（roll）**，其中在 BEV 任务中主要关注 **yaw（航向角）**

如下图，NuScenes数据集的世界坐标系定义为**地图的方向**：

**x轴：水平向右**

**y轴：水平向下**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%201.png)

假设某时刻t和时刻t+1，主车从a位置移动到了b位置：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%202.png)

首先我们可以得到向量`(delta_x, delta_y)`:

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%203.png)

主车在t+1时刻的ego_angle是该车在**世界坐标系**下的航向角：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%204.png)

我们需要的是“原本在t时刻，自车坐标系下的点，到t+1时刻自车坐标系下的位置”，也可以说是，我们需要找到t时刻的自车坐标系，如何转换到t+1时刻的自车坐标系：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%205.png)

也就是，求出shift_x, shift_y的值：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%206.png)

图上图，已知：

- `ego_angle:` 主车在当前时刻的航向角（主车方向与世界坐标系x轴的夹角）
- `translation_angle:`  是主车在时间t的偏移角度（世界坐标系下），可通过偏移向量求得：`arctan2(delta_y, delta_x)`
- `translation_len:`主车在时间t的偏移量（世界坐标系下），可通过偏移向量求得： `sqrt(delta_x^2, delta_y^2)`

可得：

`bev_angle = ego_angle - translation_angle`

`shift_x = translation_len * sin(bev_angle)`

`shift_x = translation_len * cos(bev_angle)`

**注意：**

- 这里没有考虑rotation，可能是觉得相邻帧之间的变化会很小，所以只考虑了translation
- 即便是translation，变化量应该也很小（论文作者在github issue里也提到了）

### TSA的注意力计算：

经过当前帧和历史帧的组合，TSA接下来就是进行可变形的注意力计算。

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%207.png)

**输入：**

- x_q：经过组合的bev_embeds,其中当前帧还加上了位置编码。
- value：经过组合的bev_embeds。
- ref_pts：经过组合并且添加了偏移的参考点

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%208.png)

**计算2D deformable attn：**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%209.png)

主要分为以下这么几个步骤：

1. value投影： x_v = value_proj(value)
2. 计算offset：offset = sampling_offset(x_q)，每个位置采样n个点
3. 计算sampling_locations：这里只有1层bev，在bev上计算实际的采样位置
4. 计算attn： attn = self.attn(x_q)
5. self.attn使用双线性插值sample_grid方法，从x_v中按照sampling_locations进行特征采样得到value
6. 计算输出： out = attn * value
7. value输出投影：out = out_proj(out)
8. 返回结果