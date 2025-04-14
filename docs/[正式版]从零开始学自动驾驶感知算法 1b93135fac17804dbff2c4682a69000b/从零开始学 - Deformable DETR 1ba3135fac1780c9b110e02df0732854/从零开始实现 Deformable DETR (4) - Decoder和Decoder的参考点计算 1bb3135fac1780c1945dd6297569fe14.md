# 从零开始实现 Deformable DETR (4) - Decoder和Decoder的参考点计算

**Decoder的完整结构图如下：**

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20Deformable%20DETR%20(4)%20-%20Decoder%E5%92%8CDecoder%E7%9A%84%E5%8F%82%E8%80%83%E7%82%B9%E8%AE%A1%E7%AE%97%201bb3135fac1780c1945dd6297569fe14/image.png)

**Decoder本身是由N个DecoderLayer组成，每个DecoderLayer具有相同的结构**

所以在实现的时候，我们先来实现**Decoder**类：

```python
class DeformableDetrDecoder(nn.Layer):
    """"Decoder for Deformable Detr"""
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
            [DeformableDetrDecoderLayer(embed_dim,
                                        ffn_dim,
                                        num_heads,
                                        num_points,
                                        num_levels,
                                        dropout_rate) for _ in range(num_layers)])

    def forward(self,
                input_embeds,
                attn_mask,
                encoder_x,
                encoder_attn_mask,
                pos_embeds,
                ref_pts,
                spatial_shapes,
                level_start_index,
                valid_ratios):
        x = input_embeds
        decoder_states = []
        all_self_attn_w = []
        all_cross_attn_w = []
        intermediate_ref_pts = []

        if encoder_attn_mask is not None:
            bs, seq_l = encoder_attn_mask.shape
            #_, tgt_l, _ = input_embeds.shape  # [bs, tgt_l, embed_dim]
            encoder_attn_mask = encoder_attn_mask.reshape([bs, 1, 1, seq_l])
            #encoder_attn_mask = encoder_attn_mask.expand([bs, 1, tgt_l, seq_l])
            encoder_attn_mask = 1 - encoder_attn_mask  # now padded area is 1, image area is 0
            # set padded area with small value
            encoder_attn_mask = paddle.masked_fill(paddle.zeros(encoder_attn_mask.shape),
                                                   encoder_attn_mask,
                                                   paddle.finfo(paddle.float32).min)
        for layer in self.layers:
            ref_pts_input = ref_pts[:, :, None] * valid_ratios[:, None]
            decoder_states.append(x)
            intermediate_ref_pts.append(ref_pts)
            out = layer(x,
                        attn_mask,
                        encoder_x,
                        encoder_attn_mask,
                        pos_embeds,
                        ref_pts_input,
                        spatial_shapes,
                        level_start_index)
            x = out[0]
            # no bbox refinement here, so ref_pts is unchanged
            all_self_attn_w.append(out[1])
            all_cross_attn_w.append(out[2])

        decoder_states.append(x)
        intermediate_ref_pts.append(ref_pts)

        outputs = (x,
                   intermediate_ref_pts,
                   decoder_states,
                   all_self_attn_w,
                   all_cross_attn_w)
        return outputs
```

其中需要注意的是，Decoder的参考点和Encoder并不一致，这里的参考点是为了计算object query和encoder output的交叉注意力时，使用可变形注意力需要的参考点位置。注意力计算的时候：

- 查询（query）是object query
- 值（value）是encoder output
- 也就是说，我们要对每个object query，在encoder output中选择一些位置来进行注意力计算
- 所以在Decoder中，参考点的数量，是和object query的数量一致
- 参考点的位置是由object query的position embedding部分，通过一个可学习的线性层计算得到。
- 然后根据这些参考点，进一步进行offset采样。经过offset采样得到的采样偏移，与这些参考点的位置相加得到最终的采样位置。这一步是在注意力计算时候进行，上面的步骤需要在这里实现的时候进行相关处理。

具体来说，在Decoder中首先需要对输入进行处理，这部分是在DeformableDetr的forward方法中，调用decoder的forward方法之前进行：

- DeformableDetr类包含整个推理过程，主要调用的是：（1）backbone（2）encoder（3）decoder（4）det head（classification 和 bbox regression）

Encoder到Decoder之间的处理部分包括：

```python
  # decoder
  # pos_embeds contains 2 parts: object_query and object_query_position_embedding
  query_embeds = self.pos_embeds.weight  # [num_queries, embed_dim * 2]
  # split to get object_query and its pos_embeds
  query_pos, query = paddle.split(query_embeds, 2, axis=1)
  # expand to fit batch, each sample have different pos embeds
  # [num_queries, embed_dim] -> [bs, num_queries, embed_dim]
  query_pos = query_pos.unsqueeze(0).expand([bs, -1, -1])
  # [num_queries, embed_dim] -> [bs, num_queries, embed_dim] 
  query = query.unsqueeze(0).expand([bs, -1, -1])
  # compute the reference points for object queries
  # sigmoid is to normalize to 0 to 1
  ref_pts = self.reference_points(query_pos).sigmoid()
  # store the 1st reference point locations, ref_pts is keep updating
  init_ref_pts = ref_pts

  dec_out = self.decoder(input_embeds=query,
                         encoder_x=encoder_x,
                         attn_mask=None,
                         encoder_attn_mask=mask_flatten,
                         pos_embeds=query_pos,
                         ref_pts=ref_pts,
                         spatial_shapes=spatial_shapes,
                         level_start_index=level_start_index,
                         valid_ratios=valid_ratios)
```

其中pos_embed和reference_points的定义是：

```python
self.reference_points = nn.Linear(embed_dim, 2)
self.pos_embeds = nn.Embedding(num_queries, embed_dim * 2)
```

在上面代码中，调用decoder的时候，输入的参数:

- input_embeds表示query部分
- encoder_x表示value部分
- attn_mask为None表示对于object queries，我们不需要mask掉任何一个
- encoder_attn_mask是图像特征部分相关的mask，和encoder的保持一致
- pos_embeds是object queries的pos embeding，就是上面拆分出来的query_pos
- ref_pts是object queries的参考点位置，也是上面经过线性层计算得到的
- spatial shapes和level start index在不使用cuda kernel实现注意力计算时，是不需要的，这里可以暂时忽略
- valid_ratio: 和encoder部分保持一致，具体含义在encoder部分有详细说明

有了上面的基本实现，我们完成了：

1. Decoder输入的准备和处理
2. Decoder类的定义
3. Decoder类的forward方法
    1. 这一步主要是调用多层DecoderLayer的forward方法。

下一节我们会具体实现DecoderLayer类