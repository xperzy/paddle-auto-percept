# 从零开始实现 BEVFormer (4) - Encoder类

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(4)%20-%20EncoderLayer%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac1780c2b9fecb29ff06be86/image%201.png)

Encoder类主要是一个多层结构，包含了多个EncoderLayer，接受输入，并进行参考点等计算。具体的注意力计算则会放在EncoderLayer中，其中包含TSA、SCA等操作。

# 参考点计算：

参考点计算在BEVFormer中分为两部分，分别是给TSA用的2D参考点，以及给SCA用的3D参考点。

## 2D参考点：

主要使用:

- linspace方法：生成等间距的1维点
- meshgrid方法：生成2D网格，注意，返回的是2D的x矩阵和2D的y矩阵
- stack和reshape等操作，最终生成`[bs, h*w, 1, 2]` 的参考点，第二维度的1表示bev的层数，这里是1.

详细说明见代码注释

```python
    @staticmethod
    def get_reference_points_2d(h, w, bs=1)
        # used in temporal self-attention (TSA)
        ref_y, ref_x = paddle.meshgrid(
            paddle.linspace(0.5, h - 0.5, h),  # h and w are bev feature size
            paddle.linspace(0.5, w - 0.5, w))
        # ref_y: [h, w] -> [h*w] -> [1, h*w]
        # ref_x: [h, w] -> [h*w] -> [1, h*w]
        ref_y = ref_y.reshape([-1])[None] / h  # normalize to [0,1]
        ref_x = ref_x.reshape([-1])[None] / w  # normalize to [0,1]
        # ([1, h*w], [1, h*w]) ->  [1, h*w, 2]
        ref_2d = paddle.stack((ref_x, ref_y), -1)
        # [1, h*w, 2]-> [bs, h*w, 2]
        ref_2d = ref_2d.expand([bs, w * h, 2])
        # [bs, h*w, 2]-> [bs, h*w, 1, 2]
        ref_2d = ref_2d.unsqueeze(2)  # num_bev_level, here is 1

        return ref_2d
```

## 3D参考点：

和2D参考点不同的是，3D参考点多了一个Z轴，表示高度，并通过参数num_points_in_pillar控制在Z轴上采样点的个数。

详细说明见代码注释

```python
@staticmethod
    def get_reference_points_3d(h, w, z, num_points_in_pillar, bs=1):
        # 3d ref pts are used in spatial cross attention (SCA)
        xs = paddle.linspace(0.5, w-0.5, w)  # [w]
        ys = paddle.linspace(0.5, h-0.5, h)  # [h]
        zs = paddle.linspace(0.5, z-0.5, num_points_in_pillar)  # [n_p]

        xs = xs.reshape([1, 1, w]) # [1, 1, w]
        xs = xs.expand([num_points_in_pillar, h, w])  # [n_p, h, w]
        xs = xs / w  # normalize

        ys = ys.reshape([1, h, 1])
        ys = ys.expand([num_points_in_pillar, h, w])
        ys = ys / h

        zs = zs.reshape([num_points_in_pillar, 1, 1])
        zs = zs.expand([num_points_in_pillar, h, w])
        zs = zs / z

        ref_3d = paddle.stack((xs, ys, zs), -1)  # [num_points_in_pillar, h, w, 3]
        ref_3d = ref_3d.transpose([0, 3, 1, 2])  # [num_points_in_pillar, 3, h, w]
        ref_3d = ref_3d.flatten(2)  # [num_points_in_pillar, 3, h*w]
        ref_3d = ref_3d.transpose([0, 2, 1])  # [num_points_in_pillar, h*w, 3]
        # [bs, num_points_in_pillar, h*w, 3]
        ref_3d = ref_3d.expand([bs, num_points_in_pillar, h*w, 3])

        return ref_3d
```

**在推理的时候：**

```python
        # get 3d ref pts for SCA
        ref_3d = self.get_reference_points_3d(
            h=bev_h,
            w=bev_w,
            z=self.pc_range[5] - self.pc_range[2],
            num_points_in_pillar=self.num_points_in_pillar,
            bs=bs)
        # get 2d ref pts for TSA
        ref_2d = self.get_reference_points_2d(
            h=bev_h,
            w=bev_w,
            bs=bs)

        bs, bev_len, num_bev_level, _ = ref_2d.shape  # [bs, bev_h*bev_w, 1, 2]
        
        # sampling pts for each view
        ref_pts_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas)
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d = shift_ref_2d + shift[:, None, None, :]
```

可以看到，3D的参考点需要通过投影（point_sampling），计算得到各个view上的2D参考点位置，以便于下一步进行注意力计算的时候根据该位置进行特征采样（可变性注意力计算的原理）。

## 3D投影2D

因为在处理数据集的时候已经计算好lidar2img的投影矩阵（回忆一下，这里的参考坐标系已经被转换到lidar坐标系，所以是lidar2img，同时旋转平移都已经包含在这个齐次坐标表示的转换矩阵里了）

所以最主要的操作就是：`ref_pts_cam = paddle.matmul(lidar2img, ref_pts)`

还有mask，是对于ref_pts_cam进行过滤：（1）去掉深度为0的;（2）去掉投影到范围外的

**详细分析见代码注释**

```python
 def point_sampling(self, ref_pts, pc_range, img_metas):
        lidar2img = [img_meta['lidar2img'] for img_meta in img_metas]
        lidar2img = paddle.to_tensor(np.array(lidar2img), dtype='float32')  # [B, N, 4, 4]
        num_cams = lidar2img.shape[1]
        # [bs, num_p_in_pillar, h*w, 3]
        ref_pts = ref_pts.clone()

        # ref_pts to world coords
        # ':' is used to keep tensor dims
        ref_pts[..., 0:1] = ref_pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_pts[..., 1:2] = ref_pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_pts[..., 2:3] = ref_pts[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # to homogeneous coords, (x, y, z) -> (x, y, z, 1)
        # [bs, num_p_in_pillar, h*w, 4]
        ref_pts = paddle.concat([ref_pts, paddle.ones_like(ref_pts[..., :1])], -1)

        # [bs, n_pts_in_pillar, h*w, 4] -> [n_pts_in_pillar, bs, h*w, 4]
        ref_pts = ref_pts.transpose([1, 0, 2, 3])
        n_pts, bs, seq_l = ref_pts.shape[:3]
        # [n_pts_in_pillar, bs, h*w, 4] -> [n_pts_in_pillar, bs, 1, h*w, 4]
        ref_pts = ref_pts.unsqueeze(2)
        ref_pts = ref_pts.expand([n_pts, bs, num_cams, seq_l, 4])
        # [n_pts_in_pillar, bs, num_cams, h*w, 4] -> [n_pts_in_pillar, bs, num_cams, h*w, 4, 1]
        ref_pts = ref_pts.unsqueeze(-1)

        # [bs, num_cams, 4, 4] -> [1, bs, num_cams, 4, 4]
        lidar2img = lidar2img.reshape([1, bs, num_cams, 1, 4, 4])
        # [n_pts_in_pillar, bs, num_cams, h*w, 4, 4]
        lidar2img = lidar2img.expand([n_pts, bs, num_cams, seq_l, 4, 4])

        # lidar2img: [n_pts_in_pillar, bs, num_cams, h*w, 4, 4]
        # ref_pts:   [n_pts_in_pillar, bs, num_cams, h*w, 4, 1]
        ref_pts_cam = paddle.matmul(lidar2img, ref_pts)
        # [n_pts_in_pillar, bs, num_cams, h*w, 4]
        ref_pts_cam = ref_pts_cam.squeeze(-1)

        eps = 1e-5
        # keep the pts with positive depth
        bev_mask = (ref_pts_cam[..., 2:3] > eps)
        # /z to 2D pixel coords
        ref_pts_cam = (ref_pts_cam[..., 0:2] /
            paddle.maximum(ref_pts_cam[..., 2:3], paddle.ones_like(ref_pts_cam[...,2:3]) * eps))
        # normalize to (0, 1)
        ref_pts_cam[..., 0] = ref_pts_cam[..., 0] / img_metas[0]['img_shape'][0][1]
        ref_pts_cam[..., 1] = ref_pts_cam[..., 1] / img_metas[0]['img_shape'][0][0]
        # mask the out-of-view points
        bev_mask = bev_mask & (ref_pts_cam[..., 0:1] > 0.0)
        bev_mask = bev_mask & (ref_pts_cam[..., 0:1] < 1.0)
        bev_mask = bev_mask & (ref_pts_cam[..., 1:2] > 0.0)
        bev_mask = bev_mask & (ref_pts_cam[..., 1:2] < 1.0)

        # [num_cams, bs, h*w, n_pts_in_pillar, 4]
        ref_pts_cam = ref_pts_cam.transpose([2, 1, 3, 0, 4])
        # [num_cams, bs, h*w, n_pts_in_pillar, 1]
        bev_mask = bev_mask.transpose([2, 1, 3, 0, 4])
        # [num_cams, bs, h*w, n_pts_in_pillar]
        bev_mask = bev_mask.squeeze(-1)
        bev_mask = paddle.nan_to_num(bev_mask.astype('float32'))
        bev_mask = bev_mask.astype('bool')

        return ref_pts_cam, bev_mask
```

### 前后帧的参考点处理：

因为query在计算TSA的时候是会拼接前帧的bev_query，所以参考点（注意这里是2D参考点）也需要拼接。

- 如果没有前帧，当前帧的bev_query复制一个进行拼接，所以参考点也复制拼接即可：`paddle.stack([ref_2d, ref_2d], 1)`
- 如果有前帧，因为前一帧和当前帧的参考点计算方式是一致的，但是前一帧的位置相对于当前帧又有一帧的差别（主车运动引起的），所以在拼接参考点的时候，需要使用shifted_ref_2d，这个shifted_ref_2d是加上shift得到的，shift是之前计算出来的前后帧bev的偏移量

```python
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d = shift_ref_2d + shift[:, None, None, :]   
             
        # combine ref pts from prev bev
        if prev_bev is not None:
            prev_bev = paddle.stack([prev_bev, bev_query], 1)
            prev_bev = prev_bev.reshape([bs*2, bev_len, -1])

            hybrid_ref_2d = paddle.stack([shift_ref_2d, ref_2d], 1)  # [bs, 2, bev_h*bev_w, 1, 2]
            hybrid_ref_2d = hybrid_ref_2d.reshape([bs*2, bev_len, num_bev_level, 2])
        else:
            hybrid_ref_2d = paddle.stack([ref_2d, ref_2d], 1)  # [bs, 2, bev_h*bev_w, 1, 2]
            hybrid_ref_2d = hybrid_ref_2d.reshape([bs*2, bev_len, num_bev_level, 2])

```

# **完整代码：**

```python
class BEVFormerEncoder(nn.Layer):
    """BEVformer encoder"""
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_layers,
                 num_levels,
                 num_points,
                 num_points_in_pillar,
                 num_bev_queue,
                 num_cams,
                 pc_range,
                 ffn_dim,
                 self_attn_dropout,
                 cross_attn_dropout,
                 ffn_dropout):
        super().__init__()
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.layers = nn.LayerList([
            BEVFormerEncoderLayer(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                num_points=num_points,
                num_levels=num_levels,
                num_bev_queue=num_bev_queue,
                num_cams=num_cams,
                self_attn_dropout=self_attn_dropout,
                cross_attn_dropout=cross_attn_dropout,
                ffn_dropout=ffn_dropout) for idx in range(num_layers)])

    @staticmethod
    def get_reference_points_3d(h, w, z, num_points_in_pillar, bs=1):
        # 3d ref pts are used in spatial cross attention (SCA)
        xs = paddle.linspace(0.5, w-0.5, w)
        ys = paddle.linspace(0.5, h-0.5, h)
        zs = paddle.linspace(0.5, z-0.5, num_points_in_pillar)

        xs = xs.reshape([1, 1, w])
        xs = xs.expand([num_points_in_pillar, h, w])
        xs = xs / w

        ys = ys.reshape([1, h, 1])
        ys = ys.expand([num_points_in_pillar, h, w])
        ys = ys / h

        zs = zs.reshape([num_points_in_pillar, 1, 1])
        zs = zs.expand([num_points_in_pillar, h, w])
        zs = zs / z

        ref_3d = paddle.stack((xs, ys, zs), -1)  # [num_points_in_pillar, h, w, 3]
        ref_3d = ref_3d.transpose([0, 3, 1, 2])  # [num_points_in_pillar, 3, h, w]
        ref_3d = ref_3d.flatten(2)  # [num_points_in_pillar, 3, h*w]
        ref_3d = ref_3d.transpose([0, 2, 1])  # [num_points_in_pillar, h*w, 3]
        ref_3d = ref_3d.expand([bs, num_points_in_pillar, h*w, 3])

        return ref_3d

    @staticmethod
    def get_reference_points_2d(h, w, bs=1): 
        # used in temporal self-attention (TSA)
        ref_y, ref_x = paddle.meshgrid(
            paddle.linspace(0.5, h - 0.5, h),
            paddle.linspace(0.5, w - 0.5, w))
        ref_y = ref_y.reshape([-1])[None] / h
        ref_x = ref_x.reshape([-1])[None] / w
        ref_2d = paddle.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.expand([bs, w * h, 2])
        ref_2d = ref_2d.unsqueeze(2)  # num_bev_level

        return ref_2d

    def point_sampling(self, ref_pts, pc_range, img_metas):
        lidar2img = [img_meta['lidar2img'] for img_meta in img_metas]
        lidar2img = paddle.to_tensor(np.array(lidar2img), dtype='float32')  # [B, N, 4, 4]
        num_cams = lidar2img.shape[1]
        # [bs, num_p_in_pillar, h*w, 3]
        ref_pts = ref_pts.clone()

        # ref_pts to world coords
        # ':' is used to keep tensor dims
        ref_pts[..., 0:1] = ref_pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_pts[..., 1:2] = ref_pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_pts[..., 2:3] = ref_pts[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # to homogeneous coords, (x, y, z) -> (x, y, z, 1)
        # [bs, num_p_in_pillar, h*w, 4]
        ref_pts = paddle.concat([ref_pts, paddle.ones_like(ref_pts[..., :1])], -1)

        # [bs, n_pts_in_pillar, h*w, 4] -> [n_pts_in_pillar, bs, h*w, 4]
        ref_pts = ref_pts.transpose([1, 0, 2, 3])
        n_pts, bs, seq_l = ref_pts.shape[:3]
        # [n_pts_in_pillar, bs, h*w, 4] -> [n_pts_in_pillar, bs, 1, h*w, 4]
        ref_pts = ref_pts.unsqueeze(2)
        ref_pts = ref_pts.expand([n_pts, bs, num_cams, seq_l, 4])
        # [n_pts_in_pillar, bs, num_cams, h*w, 4] -> [n_pts_in_pillar, bs, num_cams, h*w, 4, 1]
        ref_pts = ref_pts.unsqueeze(-1)

        # [bs, num_cams, 4, 4] -> [1, bs, num_cams, 4, 4]
        lidar2img = lidar2img.reshape([1, bs, num_cams, 1, 4, 4])
        # [n_pts_in_pillar, bs, num_cams, h*w, 4, 4]
        lidar2img = lidar2img.expand([n_pts, bs, num_cams, seq_l, 4, 4])

        # lidar2img: [n_pts_in_pillar, bs, num_cams, h*w, 4, 4]
        # ref_pts:   [n_pts_in_pillar, bs, num_cams, h*w, 4, 1]
        ref_pts_cam = paddle.matmul(lidar2img, ref_pts)
        # [n_pts_in_pillar, bs, num_cams, h*w, 4]
        ref_pts_cam = ref_pts_cam.squeeze(-1)

        eps = 1e-5
        # keep the pts with positive depth
        bev_mask = (ref_pts_cam[..., 2:3] > eps)
        # /z to 2D pixel coords
        ref_pts_cam = (ref_pts_cam[..., 0:2] /
            paddle.maximum(ref_pts_cam[..., 2:3], paddle.ones_like(ref_pts_cam[...,2:3]) * eps))
        # normalize to (0, 1)
        ref_pts_cam[..., 0] = ref_pts_cam[..., 0] / img_metas[0]['img_shape'][0][1]
        ref_pts_cam[..., 1] = ref_pts_cam[..., 1] / img_metas[0]['img_shape'][0][0]
        bev_mask = bev_mask & (ref_pts_cam[..., 0:1] > 0.0)
        bev_mask = bev_mask & (ref_pts_cam[..., 0:1] < 1.0)
        bev_mask = bev_mask & (ref_pts_cam[..., 1:2] > 0.0)
        bev_mask = bev_mask & (ref_pts_cam[..., 1:2] < 1.0)

        # [num_cams, bs, h*w, n_pts_in_pillar, 4]
        ref_pts_cam = ref_pts_cam.transpose([2, 1, 3, 0, 4])
        # [num_cams, bs, h*w, n_pts_in_pillar, 1]
        bev_mask = bev_mask.transpose([2, 1, 3, 0, 4])
        # [num_cams, bs, h*w, n_pts_in_pillar]
        bev_mask = bev_mask.squeeze(-1)
        bev_mask = paddle.nan_to_num(bev_mask.astype('float32'))
        bev_mask = bev_mask.astype('bool')

        return ref_pts_cam, bev_mask

    def forward(self,
                bev_query,
                value,
                bev_h,
                bev_w,
                bev_pos,
                spatial_shapes,
                level_start_index,
                prev_bev,
                img_metas,
                shift=0):
        bs = bev_query.shape[0]  # bev query : [bs, bev_h*bev_w, embed_dim]
        # get 3d ref pts for SCA
        ref_3d = self.get_reference_points_3d(
            h=bev_h,
            w=bev_w,
            z=self.pc_range[5] - self.pc_range[2],
            num_points_in_pillar=self.num_points_in_pillar,
            bs=bs)
        # get 2d ref pts for TSA
        ref_2d = self.get_reference_points_2d(
            h=bev_h,
            w=bev_w,
            bs=bs)

        bs, bev_len, num_bev_level, _ = ref_2d.shape  # [bs, bev_h*bev_w, 1, 2]

        # sampling pts for each view
        ref_pts_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas)
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d = shift_ref_2d + shift[:, None, None, :]

        # combine ref pts from prev bev
        if prev_bev is not None:
            prev_bev = paddle.stack([prev_bev, bev_query], 1)
            prev_bev = prev_bev.reshape([bs*2, bev_len, -1])

            hybrid_ref_2d = paddle.stack([shift_ref_2d, ref_2d], 1)  # [bs, 2, bev_h*bev_w, 1, 2]
            hybrid_ref_2d = hybrid_ref_2d.reshape([bs*2, bev_len, num_bev_level, 2])
        else:
            hybrid_ref_2d = paddle.stack([ref_2d, ref_2d], 1)  # [bs, 2, bev_h*bev_w, 1, 2]
            hybrid_ref_2d = hybrid_ref_2d.reshape([bs*2, bev_len, num_bev_level, 2])

        intermediate = []
        for layer_idx, layer in enumerate(self.layers):
            output = layer(x=bev_query,
                           value=value,
                           ref_2d=hybrid_ref_2d,
                           ref_3d=ref_3d,
                           bev_h=bev_h,
                           bev_w=bev_w,
                           bev_pos=bev_pos,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index,
                           ref_pts_cam=ref_pts_cam,
                           bev_mask=bev_mask,
                           prev_bev=prev_bev)
            bev_query = output
            intermediate.append(output)
        return output, intermediat
```

# EncoderLayer

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(4)%20-%20EncoderLayer%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201c13135fac1780c2b9fecb29ff06be86/image%202.png)

EncoderLayer的结构相对比较清晰，完全按照上面的图进行搭建即可；

**其中TSA的输入参数：**

- x: 是当前帧的bev_query
- value：前一帧的bev和当前的bev_query拼接后的值，或者为空（没有前一帧的时候）
- pos_embed：是当前帧的bev_query对应的position embedding
- ref_pts：是bev_query对应的2D参考点，也就是根据bev尺寸均匀采样的网格，注意，这里是拼接前后帧完成后的**hybrid_ref_pts**
- spatial_shape：这里是2Dbev空间的注意力计算，bev只有一层所以就是bev的尺寸
- level_start_index：没有用到，而且只有一层bev特征所以起始点是0

```python
        x, self_attn_w = self.self_attn(x=x,  
                                        value=prev_bev,
                                        pos_embeds=bev_pos,
                                        ref_pts=ref_2d,
                                        attn_mask=None,
                                        spatial_shapes=paddle.to_tensor([[bev_h, bev_w]], dtype='int64'),
                                        level_start_index=paddle.zeros([1], dtype='int64'))
```

**SCA的参数：**

- x: 是当前帧的bev_query，其实是经过上面各个步骤计算之后的bev_query
- value：图像特征，是经过flatten的多层图像特征。
- pos_embed：空
- ref_pts：是BEV空间的3D参考点，是用于采样各个view上的点，所以是ref_3d
- ref_pts_cam: 3D参考点投影到view上并完成采样点生成后的采样点位置
- spatial_shape：这里是多层图像特征的shape，所以由图像部分作为参数传入，这里只传递下去即可
- level_start_index：没有用到(使用cuda kernel的时候需要，本文略)

```python
        x, cross_attn_w = self.cross_attn(x=x,
                                          value=value,
                                          pos_embeds=None,
                                          ref_pts=ref_3d,
                                          ref_pts_cam=ref_pts_cam,
                                          bev_mask=bev_mask,
                                          spatial_shapes=spatial_shapes,
                                          level_start_index=level_start_index)
```

```python
class BEVFormerEncoderLayer(nn.Layer):
    """Encoder layer for bevformer"""
    def  __init__(self, # W: Too many arguments (9/5)
                  embed_dim,
                  ffn_dim,
                  num_heads,
                  num_points,
                  num_levels,
                  num_cams,
                  num_bev_queue,
                  self_attn_dropout,
                  cross_attn_dropout,
                  ffn_dropout):
        super().__init__()
        # temporal self attn
        self.self_attn = TemporalSelfAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               num_levels=1,
                                               num_points=4,
                                               num_bev_queue=num_bev_queue)
        self.self_attn_dropout = nn.Dropout(self_attn_dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        # spatial cross attn
        self.cross_attn = SpatialCrossAttention(embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                num_cams=num_cams,
                                                num_points=num_points,  # 4
                                                num_levels=num_levels)  # 1
        self.cross_attn_dropout = nn.Dropout(cross_attn_dropout)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        # ffn
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.act = nn.ReLU()
        self.act_dropout = nn.Dropout(ffn_dropout)
        self.fc_dropout = nn.Dropout(ffn_dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
    def forward(self,
                x,
                value,
                ref_2d,
                ref_3d,
                bev_h,
                bev_w,
                bev_pos,
                ref_pts_cam,
                bev_mask,
                prev_bev,
                spatial_shapes,
                level_start_index):
        # self-attn: MultiHeadAttention
        h = x
        x, self_attn_w = self.self_attn(x=x,
                                        value=prev_bev,
                                        pos_embeds=bev_pos,
                                        ref_pts=ref_2d,
                                        attn_mask=None,
                                        spatial_shapes=paddle.to_tensor([[bev_h, bev_w]], dtype='int64'),
                                        level_start_index=paddle.zeros([1], dtype='int64'))
        x = self.self_attn_dropout(x)
        x = h + x
        x = self.self_attn_norm(x

        # cross-attn: BEVFormerDeformableAttention
        h = x
        x, cross_attn_w = self.cross_attn(x=x,
                                          value=value,
                                          pos_embeds=None,
                                          ref_pts=ref_3d,
                                          ref_pts_cam=ref_pts_cam,
                                          bev_mask=bev_mask,
                                          spatial_shapes=spatial_shapes,
                                          level_start_index=level_start_index)
        x = self.cross_attn_dropout(x)
        x = h + x
        x = self.cross_attn_norm(x)
        # ffn
        h = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.fc_dropout(x)
        x = h + x
        x = self.ffn_norm(x)

        return x
```