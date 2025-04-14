# 从零开始实现BEVFormer (5) - Encoder - Temporal Self Attention (TSA)

**TSA会在EncoderLayer中被调用：**

```python
        x, self_attn_w = self.self_attn(x=x,
                                        value=prev_bev,
                                        pos_embeds=bev_pos,
                                        ref_pts=ref_2d,
                                        attn_mask=None,
                                        spatial_shapes=paddle.to_tensor([[bev_h, bev_w]], dtype='int64'),
                                        level_start_index=paddle.zeros([1], dtype='int64'))
```

其中：

- x：是当前帧的bev_query
- value：是前一帧的bev_query，当没有前一帧的时候，这个值是空，会在TSA中进行拼接当前帧的操作，如果不为空的时候，在Encoder中已经完成前后帧的拼接：
    - `prev_bev = paddle.stack([prev_bev, bev_query], 1)
    prev_bev = prev_bev.reshape([bs*2, bev_len, -1])`
- ref_2d:这里已经是完成拼接的2D参考点，实际上传入的是Encoder中的`hybrid_ref_2d`

在实现TSA的时候，首先要搞清楚：

- x（query）：是拼接前后帧之后的bev queries。可传入的x是当前帧的bev_query，所以还需要加上前一帧的bev_query。前一帧的bev_query已经存在传入的value中了（如果value为空，则我们拼接两个当前帧的bev_query）
    - `if value is None:
                bs, bev_len, _ = x.shape
                # stack 2 current bev
                value = paddle.stack([x, x], 1)
                value = value.reshape([bs * 2, bev_len, -1])`
    - 然后完成拼接生成query：
        - `x_q = paddle.concat([value[:bs], x_q], -1)`。
- x_v（value）：如上，如果有前帧，已经在Encoder中完成拼接: `[prev_bev, bev_query]`,如果没有前帧，则是当前帧bev_query的拼接

**TSA的计算过程：**

- 参考点→采样点：通过线性层，输入`x_q`，输出采样点`偏移offset`
    - `sampling_offsets = self.sampling_offsets(x_q)`
- 采样点归一化：通过每层特征的shape，对采样点进行归一化，此时会归一化到0到1。（在注意力计算之前还需要再归一化一下到-1,1，因为grid_sample方法输入要求是-1,1）
    - `offset_normalizer = paddle.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)`
    - `sampling_locations = (ref_pts[:, :, None, :, None, :] +
                sampling_offsets / offset_normalizer[None, None, None, :, None, :])`
- 生成attn权重：
    - `attn = self.attn(x_q)`
    - `attn = self.softmax(attn)`
- 可变形注意力计算：
    - `out = multiscale_deformable_attention(x_v, spatial_shapes, sampling_locations, attn)`
- 然后输出前做投影：
    - `out = self.out_proj(out)`
- 其他中间步骤还涉及到很多的reshape，transpose操作等，具体见代码实现。

### 可变形注意力计算：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20BEVFormer%20(5)%20-%20Encoder%20-%20Temporal%20Self%20Atte%201c13135fac178079aee9e036785d0a64/image%209.png)

```python
def multiscale_deformable_attention(x_v, spatial_shapes, sampling_locations, attn): # W: Too many local variables (21/15)
    """deformable attention in paddle"""
    # get shapes
    bs, _, num_heads, head_dim = x_v.shape
    _, tgt_l, _, num_levels, num_points, _ = sampling_locations.shape
    # split each level
    spatial_shapes_list = spatial_shapes.numpy().tolist()
    v_list = x_v.split([h * w for h, w in spatial_shapes_list], axis=1)
    # normalize to -1, 1: grid_sample requires -1,1 range
    sampling_grids = 2 * sampling_locations - 1  # [bs, tgt_l, num_heads, num_levels, num_points, 2]
    sampling_v_list = []

    for level_idx, (h, w) in enumerate(spatial_shapes_list):
        # [bs, h*w, num_heads, head_dim] -> [bs, h*w, num_heads*head_dim]
        v_l = v_list[level_idx].flatten(2)
        v_l = v_l.transpose([0, 2, 1])  # -> [bs, num_heads*head_dim, h*w]
        v_l = v_l.reshape([bs * num_heads, head_dim, h, w])

        # [bs, tgt_l, num_heads, num_points, 2]
        sampling_grid_l = sampling_grids[:, :, :, level_idx]
        # [bs, num_heads, tgt_l, num_points, 2]
        sampling_grid_l = sampling_grid_l.transpose([0, 2, 1, 3, 4])
        # [bs*num_heads, tgt_l, num_points, 2]
        sampling_grid_l = sampling_grid_l.flatten(0, 1)

        sampling_v_l = F.grid_sample(v_l,
                                       sampling_grid_l,
                                       mode='bilinear',
                                       padding_mode='zeros',
                                       align_corners=False)
        sampling_v_list.append(sampling_v_l)

    # attn: [bs, tgt_l, num_heads, num_levels, num_points]
    attn = attn.transpose([0, 2, 1, 3, 4])  # [bs, num_heads, tgt_l, num_levels, num_points]
    attn = attn.reshape([bs*num_heads, 1, tgt_l, num_levels*num_points])

    # [bs*num_heads, head_dim, tgt_l, num_points] * num_levels ->
    # [bs*num_heads, head_dim, tgt_l, num_levels, num_points]
    out = paddle.stack(sampling_v_list, axis=-2)
    # [bs*num_heads, head_dim, tgt_l, num_levels * num_points]
    out = out.flatten(-2)
    out = out * attn
    # [bs*num_heads, head_dim, tgt_l]
    out = out.sum(-1)
    out = out.reshape([bs, num_heads * head_dim, tgt_l])
    out = out.transpose([0, 2, 1])  # [bs, tgt_l, embed_dim]
    return out
```

### 完整代码：

```python
class TemporalSelfAttention(nn.Layer):
    """Temporal Self Attention in BEVFormer"""
    def  __init__(self,
                  embed_dim=256,
                  num_heads=8,
                  num_levels=1,
                  num_points=4,
                  num_bev_queue=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dim * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attn = nn.Linear(
            embed_dim * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(axis=-1)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                x,
                attn_mask,
                value,
                pos_embeds,
                ref_pts,
                spatial_shapes,
                level_start_index):
        # if no prev_bev, stack two current bev queries as V
        if value is None:
            bs, bev_len, _ = x.shape
            # stack 2 current bev
            value = paddle.stack([x, x], 1)
            value = value.reshape([bs * 2, bev_len, -1])

        bs, tgt_l, _ = x.shape
        _, src_l, _ = value.shape

        x_q = x + pos_embeds if pos_embeds is not None else x
        # the 1st bev_query is considered as the prev query,
        # it has already been added with pos_embed in prev iter
        x_q = paddle.concat([value[:bs], x_q], -1) # concat prev and current 

        x_v = self.value_proj(value)
        if attn_mask is not None:
            x_v = x_v.masked_fill(attn_mask[..., None], 0.0)
        x_v = x_v.reshape([bs * self.num_bev_queue, src_l, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(x_q)
        sampling_offsets = sampling_offsets.reshape(
            [bs, tgt_l, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2])

        attn = self.attn(x_q)
        attn = attn.reshape(
            [bs, tgt_l, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points])
        attn = self.softmax(attn)
        attn = attn.reshape(
            [bs, tgt_l, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points])
        attn = attn.transpose([0, 3, 1, 2, 4, 5])
        attn = attn.reshape(
            [bs * self.num_bev_queue, tgt_l, self.num_heads, self.num_levels, self.num_points])

        sampling_offsets = sampling_offsets.transpose([0, 3, 1, 2, 4, 5, 6])
        sampling_offsets = sampling_offsets.reshape(
            [bs * self.num_bev_queue, tgt_l, self.num_heads, self.num_levels, self.num_points, 2])
        offset_normalizer = paddle.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (ref_pts[:, :, None, :, None, :] +
            sampling_offsets / offset_normalizer[None, None, None, :, None, :])
            
        out = multiscale_deformable_attention(x_v, spatial_shapes, sampling_locations, attn)

        # [bs*num_bev_queue, num_queries, embed_dim] -> [num_queries, embed_dim, bs*num_bev_queue]
        out = out.transpose([1, 2, 0])
        out = out.reshape([tgt_l, self.embed_dim, bs, self.num_bev_queue])
        out = out.mean(-1)
        out = out.transpose([2, 0, 1])  # [bs, num_queries, embed_dim]
        out = self.out_proj(out)
        return out, att
```