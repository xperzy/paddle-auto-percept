# 从零开始实现 Deformable DETR (2) - Encoder和参考点计算

我们首先来看一下DeformableDETR的整体结构：

![image.png](ddc2/image.png)

在上一节中，我们实现了ResNet的Image Backbone，本节我们来实现Encoder部分。

### Encoder和EncoderLayer结构：

<img src="ddc2/image%201.png" style="width:50%;">

上图实际上展示了EncoderLayer的结构，在代码实现的时候，我们可以先实现`DeformableDetrEncoder`类，这个类实际上是包含了多个上图中的模块（多层EncoderLayer），并且还实现了注意力计算所需要的输入ReferencePoint的生成方法。

Encoder类的主要成员变量是一个LayerList，包含有多层的EncoderLayer：

```python
   class DeformableDetrEncoder(nn.Layer):
    """"Encoder for Deformable Detr"""
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 num_heads,
                 num_points,
                 num_levels,
                 num_layers,
                 dropout_rate=0.0):
        super().__init__()
        self.layers = nn.LayerList(
            [DeformableDetrEncoderLayer(embed_dim,
                                        ffn_dim,
                                        num_heads,
                                        num_points,
                                        num_levels,
                                        dropout_rate) for _ in range(num_layers)])
```

每个EncoderLayer如上图所示完成EncoderLayer各模块的计算，所以在forward方法中有：

```python
  def forward(self,
                input_embeds,
                attn_mask,
                pos_embeds,
                spatial_shapes,
                level_start_index,
                valid_ratios):
        x = input_embeds
        # 获得参考点位置：
        ref_pts = self.get_reference_points(spatial_shapes, valid_ratios)
        encoder_states = []
        all_attn_w = []
        # 计算 attn_mask:
        if attn_mask is not None:
            bs, seq_l = attn_mask.shape
            attn_mask = attn_mask.reshape([bs, 1, 1, seq_l])
            attn_mask = 1 - attn_mask  # now padded area is 1, image area is 0
            # set padded area with small value
            attn_mask = paddle.masked_fill(
                paddle.zeros(attn_mask.shape), attn_mask, paddle.finfo(paddle.float32).min)
        # 这一部分完成多层EncoderLayer的计算：
        for layer in self.layers:
            encoder_states.append(x)
            out = layer(x,
                        attn_mask,
                        pos_embeds,
                        ref_pts,
                        spatial_shapes,
                        level_start_index)
            x = out[0]
            attn_w = out[1]
            all_attn_w.append(attn_w)

        return x, encoder_states, all_attn_w
```

主要分为三个部分：

1. **计算参考点位置**
2. **计算attn_mask**
3. **对每一层EncoderLayer进行计算，并储存结果**

### 参考点：

```python
def get_reference_points(spatial_shapes, valid_ratios):
    reference_points_list = []
    for level, (h, w) in enumerate(spatial_shapes):
        ref_y, ref_x = paddle.meshgrid(
            paddle.linspace(0.5, h - 0.5, h.astype('int32')),
            paddle.linspace(0.5, w - 0.5, w.astype('int32')))
        ref_y = ref_y.reshape([-1])[None] / (valid_ratios[:, None, level, 1] * h)
        ref_x = ref_x.reshape([-1])[None] / (valid_ratios[:, None, level, 0] * w)
        # [batch, h * w, 2]
        ref = paddle.stack([ref_x, ref_y], -1)
        reference_points_list.append(ref)
    # [batch, num_levels * h * w, 2]
    reference_points = paddle.conat(reference_points_list, 1)
    # [batch, num_levels * h * w, 1, 2] * [batch, 1, num_levels, 2]
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points
```

- 从代码中可以看到，首先是进行分level计算：
    - 注意力计算的时候，**输入是多尺度图像特征**。实现的时候是一个包含多层（num_levels）特征图的`list`，其中每个元素（每个特征图）的尺寸并不相同。所以这里的`spatial_shapes`保存了各个特征图的尺寸`(h,w)`。
- `valid_ratios` 保存了每个feature map的有效区域的比例，用于对 reference points 进行裁剪，使得采样点的范围限制在特定的区域内（这部分的实现在上一级的`DeformableDetr`类中，作为参数传入Encoder）
    - `valid_ratios`包含多个： `(valid_h / h, valid_w / w)` ，其实就是非零区域和整张图的比值。这里的非零区域表示实际的图像区域，整张图表示经过padding后的输入到模型中的tensor大小（因为一个batch要求每张图像的尺寸相同，但实际加载进来的每张图大小可能不同，所以需要进行padding，使得经过padding后所有图像大小一致，但为了保持原图比例，所以每张图padding的区域不同，因此每张图会保存一个mask用来记录这些padding的区域）
    - 需要注意的是：由于mask是根据不同level进行下采样得到的，并不能保证所有level的实际padding的比例完全一致，可能出现微小的不同。所以，**valid_ratios是对于batch中的每个样本的每一层特征，都是不同的**。
- 在代码中循环处理每个level时：
    - `ref_y`和`ref_x` 首先通过linspace方法进行采样，这里的点是未做归一化的坐标点，范围是当前level的`(h,w)` 。例如，当前层的特征图的大小是 100 x 134，那么ref_y就是134列 [0.5，1.0， 1.5 … 99.5] 。
        
        ![image.png](ddc2/image%202.png)
        
    - 注意：这里的`(h,w)`是当前层（level）的特征图的尺寸。这个尺寸其实是pad之后的尺寸，其中有一部分是原图经过padding的非有效图像区域。我们在推理的时候，参考点的位置加上模型计算得到的偏移量，就是我们要的bbox的位置，这些都是根据原图实际大小来计算的，不考虑padding，所以要求这些参考点也是对于原图的。参考点的范围应该是相对于原图大小的相对坐标。
    - 也就是说，我们希望这些参考点是相对于`(valid_h，valid_w)`的相对坐标，而现在的坐标是在`(h,w)`上的绝对坐标，我们可以先将这些坐标除以h(或者w)，就得到了相对坐标位置。
    - 然后再除以`valid_ratio`，这样会将参考点（**相对于(h,w)的0到1的范围），转换为相对于(valid_h, valid_w)的范围，坐标值可能会大于1**。也就是说，原本采样的这些点，相对于有效区域，在哪些位置，如果在有效区域外，那这个坐标可能会大于1.0
    - 这样，我们就得到了多层特征图上的所有特征点的参考位置。
        
        ![image.png](ddc2/image%203.png)
        
    - 代码循环外：
        - reference_points 经过concat，shape=`[bs, (h1*w1+h2*w2+h3*w3+h4*w4)，2]`，
            - 其中h1,w1 到 h4,w4是各个level的featuremap大小
        - valid_ratios的shape： `[bs, num_levels, 2]`
        - 然后进行一个广播计算：
            - `[bs,(h1*w1+h2*w2+h3*w3+h4*w4),2]` → `[bs,(h1*w1+h2*w2+h3*w3+h4*w4),1,2]`
            - `[bs, num_levels, 2]`→`[bs, 1, num_levels, 2]`
            - 然后 ref_pts会复制num_level份
                - `[bs,(h1*w1+h2*w2+h3*w3+h4*w4),1,2]` → `[bs,(h1*w1+h2*w2+h3*w3+h4*w4),num_levels,2]`
                - 再与valid_ratios相乘：
                - `[bs,(h1*w1+h2*w2+h3*w3+h4*w4),num_levels,2]` * `[bs, 1, num_levels, 2]`
            - 得到的结果是：`[bs,(h1*w1+h2*w2+h3*w3+h4*w4),num_levels,2]`
        - 这一步计算的原理是：
            - 对于每一层的每一个特征点位置，都会有这个点在各个层（一共num_level个）的对应位置。
            - 这里乘以valid_ratios，可以理解为，每个点分别乘以各个level的valid_ratio。此时每个点都已经是相对其有效图像区域的相对坐标，当乘以某一层的valid_ratio时，就将这个相对位置，转换为了相对于当前层feature map尺寸的相对位置。
            - 也可以理解为，在这一步之前，每个特征点已经转化为了相对于真实图像大小（不是padded之后的featuremap，而是原图）的相对坐标，使用这样的方式是为了保证在每层采样的时候，都是基于原图中同一个位置。**然后我们要基于这一点，生成各个level上的参考点，并符合参考点的格式（归一化）要求。**
            - 这个相对位置，是符合后面做grid_sample时对grid的输入要求的（相对于feature map的相对坐标）。
                - 看grid_sample的api要求的是(-1,1)的范围，这里其实大部分情况是（0，1）的范围，这一步其实是在deformable_attention方法里转换的，所以在这里不需要操作
                - 这里有可能出现坐标值>1.0的情况发生，就会导致最终输入到grid_sample方法的坐标超过了(-1,1)范围。这里方法本身会忽略掉超出范围的点，所以不需要担心。
    - **所以，其实我们是先在每层的每个特征点，找到其相对于原图的位置，然后再根据这每层的每个特征点进一步得到各个层的参考点位置。**
        - 所以得到的参考点的shape是：`[bs,(h1*w1+h2*w2+h3*w3+h4*w4),num_levels,2]`
        - 其中`(h1*w1+h2*w2+h3*w3+h4*w4)` 已经是每层特征的位置数量相加了，后面还有一维num_levels（这里等于4），进一步说明了，对于每个多层特征的每个位置，都会有4个参考点位置会在特征采样时候参与计算。

### 完整代码：

```python
class DeformableDetrEncoder(nn.Layer):
    """"Encoder for Deformable Detr"""
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 num_heads,
                 num_points,
                 num_levels,
                 num_layers,
                 dropout_rate=0.0):
        super().__init__()
        self.layers = nn.LayerList(
            [DeformableDetrEncoderLayer(embed_dim,
                                        ffn_dim,
                                        num_heads,
                                        num_points,
                                        num_levels,
                                        dropout_rate) for _ in range(num_layers)])
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        ref_pts_list = []
        for level, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = paddle.meshgrid(
                paddle.linspace(0.5, h - 0.5, h.astype('int32')),
                paddle.linspace(0.5, w - 0.5, w.astype('int32')))
            # ([h, w] -> [1, h*w]) / ([bs, n_level, 2] -> [bs, 1] * h) = [bs, h*w]
            # valid_ratio is  valid_h / feature_h, valid_w / feature_w
            # h, w here are feature_h, feature_w,
            # * h means coords are now aligned in padded_feat, which is needed for attn
            # https://github.com/open-mmlab/mmdetection/issues/8656
            ref_y = ref_y.reshape([-1])[None] / (valid_ratios[:, None, level, 1] * h)
            ref_x = ref_x.reshape([-1])[None] / (valid_ratios[:, None, level, 0] * w)
            # [bs, h*w, 2]
            ref = paddle.stack((ref_x, ref_y), -1)
            ref_pts_list.append(ref)
        # [bs, seq_l, 2], seq_l = sum([h * w for h, w in spatial_shapes])
        ref_pts = paddle.concat(ref_pts_list, 1)
        # ref_pts: [bs, seq_l, 2] -> [bs, seq_l, 1, 2]
        # valid_ratios: [bs, n_level, 2] -> [bs, 1, n_level, 2]
        # ref_pts * valid_ratios: [bs, seq_l, n_level, 2]
        ref_pts = ref_pts[:, :, None] * valid_ratios[:, None]
        print('res: ', ref_pts.shape)
        return ref_pts

    def forward(self,
                input_embeds,
                attn_mask,
                pos_embeds,
                spatial_shapes,
                level_start_index,
                valid_ratios):
        x = input_embeds
        ref_pts = self.get_reference_points(spatial_shapes, valid_ratios)
        encoder_states = []
        all_attn_w = []
        if attn_mask is not None:
            bs, seq_l = attn_mask.shape
            attn_mask = attn_mask.reshape([bs, 1, 1, seq_l])
            attn_mask = 1 - attn_mask  # now padded area is 1, image area is 0
            # set padded area with small value
            attn_mask = paddle.masked_fill(
                paddle.zeros(attn_mask.shape), attn_mask, paddle.finfo(paddle.float32).min)

        for layer in self.layers:
            encoder_states.append(x)
            out = layer(x,
                        attn_mask,
                        pos_embeds,
                        ref_pts,
                        spatial_shapes,
                        level_start_index)
            x = out[0]
            attn_w = out[1]
            all_attn_w.append(attn_w)

        return x, encoder_states, all_attn_w
```

## Mask：

- 为什么会有Mask？
    - 因为希望一个Batch的输入可以支持不同尺寸的图像
    - 同时希望Batch中的图像各自都保持自己的长宽比不变
    - 所以Batch输入是同一个固定尺寸（例如800x1333），然后预处理的时候将图像都按比例缩放（例如最短边不小于800，同时最长边不超过1333）
    - 非图像部分即为pad的部分，需要使用mask记录下来。在计算Attention的时候，pad部分不参与计算

所以，采用和输入Batch大小相同的Tensor来保存mask信息，mask中的0和1的位置对于每张图来说是不同的，但是mask这个Tensor的尺寸，对于图像来说都是一样的，都是hxw。

![image.png](ddc2/image%204.png)

不仅输入的图像Batch有mask，经过Image Backbone的图像特征也有mask，这个mask是直接通过将输入mask按照feature map的大小进行下采样得到的。

## Valid ratio

![image.png](ddc2/image%205.png)

对于一个feature map，其中有效的部分的长宽与feature map的长宽的比值，就是valid_ratio。不难发现，对于一个batch的图像，单层feature map的valid ratio是不同的，因为每张图像的大小不同，pad的区域大小不同。对于一张图像的多层feature map（例如ResNet不同层得到的多层特征），valid_ratio也是不同的，这是因为feature map的mask是由输入图像的mask进行下采样得到的，因为每层的feature map大小不同，每次下采样不能保证pad区域都能够被整除，所以计算下来的valid_ratio也会不同。

因此，valid_ratio对于不同的level和不同的图像，都是不同的。
