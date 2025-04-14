# 从零开始实现 DETR3D (2) - Decoder的整体结构

上一节我们通过ResNet+FPN的方式，获得了图像特征，这里的图像是从多个视角下得到的环视图，并对每一张图进行了特征提取，本节我们继续来看Decoder的部分。

如上图所示

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20DETR3D%20(2)%20-%20Decoder%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201bd3135fac178009a03be324d477cbba/image.png)

如上图所示，Decoder的输入为：

- query和pos_embeds：分别是object query, 和每个object query对应的位置编码pos_embed。可以理解为两个nn.Embedding，当输入为某个query的id（就是0到n_query-1中的某一个数字）时，可以得到两个embed_dim维度的向量，一个是query本身，一个是这个query的位置信息。
- ref_pts：表示3D空间中的参考点，这些参考点位置是可学习的，通过将pos_embed输入到一个embedding中得到。
- img_features：经过FPN输出的多层图像特征，每一层图像特征又包含了6个不同View的图像特征。

### Decoder的结构：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20DETR3D%20(2)%20-%20Decoder%E7%9A%84%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84%201bd3135fac178009a03be324d477cbba/image%201.png)

DETR3D的decoder部分包含多个DecoderLayer，每个DecoderLayer主要包括3部分：

- Self-Attention：标准的自注意力计算，主要学习object query之间的相互关系。
- Cross-Attention：交叉注意力，输入是经过self-atten之后的object query，以及图像特征。这里的交叉注意力使用可变形注意力机制，通过将3D采样点位置（采样点是基于参考点位置在其周围进行采样得到的）投影到不同View上，得到对应的2D位置，以此位置的图像特征作为参与当前参考点交叉注意力计算的图像特征。
- FFN、Norm和残差连接：和标准的Transformer结构相同。

### 具体实现：

```python
class Detr3DDecoder(nn.Layer):
    def __init__(self,
                 num_queries,
                 pc_range,
                 num_feature_levels, 
                 num_cams,
                 num_layers,
                 embed_dim,
                 num_heads,
                 self_attn_dropout,
                 cross_attn_dropout,
                 ffn_dim,
                 ffn_dropout,
                 num_points=5,
                 return_intermediate=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        # reference points layer
        self.reference_points = paddle.nn.Linear(self.embed_dim, 3)
        # decoder transformer layer
        self.layers = nn.LayerList([
            Detr3DDecoderLayer(embed_dim=embed_dim,
                               ffn_dim=ffn_dim,
                               num_heads=num_heads,
                               num_cams=num_cams,
                               pc_range=pc_range,
                               num_points=num_points,
                               n_levels=num_feature_levels,
                               self_attn_dropout=self_attn_dropout,
                               cross_attn_dropout=cross_attn_dropout,
                               ffn_dropout=ffn_dropout) for idx in range(num_layers)])
        # dropout
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self,
                multi_level_feats,
                query_embeds,
                img_metas,
                bbox_embed=None):
        """
        Args:
            multi_level_feats: tuple/list of 5D tensor of shape (B, N, C, H, W)
            query_embeds: (num_queries, embed_dim * 2) 
        """
        bs = multi_level_feats[0].shape[0]
        query_embed, target = paddle.split(query_embeds, 2, axis=1)  # [num_queries, embed_dim * 2] -> [num_queries, embed_dim]
        query_embed = query_embed.unsqueeze(0).expand([bs, -1, -1])  # [bs, num_queries, embed_dim]
        target = target.unsqueeze(0).expand([bs, -1, -1])  # [bs, num_queries, embed_dim]
        reference_points = self.reference_points(query_embed).sigmoid()  # [bs, num_queries, 3]
        init_reference_points = reference_points

        # Note: detr3d pytorch code needs these transpose, since they call torch.nn.MultiheadAttention with batch_first=False
        # We here do not need it!
        #target = target.transpose([1, 0, 2])
        #query_embed = query_embed.transpose([1, 0, 2])

        all_hidden_states = ()
        all_self_attentions = ()
        all_cross_attentions = ()
        intermediate = ()
        intermediate_reference_points = ()

        hidden_states = target  # used to iterate through decoder layers

        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points
            # store all hidden states
            all_hidden_states = all_hidden_states + (hidden_states, )
            # inference 
            layer_out = decoder_layer(x=hidden_states,
                                      value=multi_level_feats,
                                      pos_embed=query_embed,
                                      ref_pts=reference_points_input,
                                      img_metas=img_metas)

            out = layer_out[0]

            # refine reference points
            if bbox_embed is not None:
                tmp = bbox_embed[idx](layer_out[0])
                new_reference_points = paddle.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            # save results
            hidden_states = out
            intermediate += (hidden_states, )
            intermediate_reference_points += (reference_points, )
            all_self_attentions += (layer_out[1], )
            all_cross_attentions += (layer_out[2], )

        intermediate = paddle.stack(intermediate, 1)
        intermediate_reference_points = paddle.stack(intermediate_reference_points, 1)

        all_hidden_states += (hidden_states, )
        return hidden_states, init_reference_points, intermediate, intermediate_reference_points, all_hidden_states, all_self_attentions, all_cross_attentions

```

**DecoderLayer部分的实现：**

- 这里需要注意的是，cross-attn部分的输入，区别于DETR的标准Cross-attn，DETR的cross-attn输入key和value都是来自于source，query是来自于target，DETR3D这里由于采用的是DeformableAttn的计算方式，attn weight是通过线性层直接学习得到的，所以不需要经过q*k’的计算，因此key在这里直接给None就可以了。

```python
class Detr3DDecoderLayer(paddle.nn.Layer):
    """DETR3D Decoder Layer"""
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 num_heads,
                 num_cams,
                 pc_range,
                 num_points,
                 n_levels,
                 self_attn_dropout,
                 cross_attn_dropout,
                 ffn_dropout):
        super().__init__()
        self.embed_dim = embed_dim
        # Self Attn
        self.self_attn = MultiheadAttention(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout_rate=self_attn_dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        # Cross Attn
        self.cross_attn = Detr3DCrossAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               num_points=num_points,
                                               num_cams=num_cams,
                                               pc_range=pc_range,
                                               dropout_rate=cross_attn_dropout,
                                               n_levels=n_levels)
        self.cross_attn_norm = nn.LayerNorm(self.embed_dim)
        # FFN
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.ReLU()
        self.act_dropout = nn.Dropout(ffn_dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.fc_dropout = nn.Dropout(ffn_dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, value, pos_embed, ref_pts, img_metas):
        # self-attn: MultiHeadAttention
        x, self_attn_w = self.self_attn(x=x, pos_embed=pos_embed)
        x = self.self_attn_norm(x)
        # cross-attn: Detr3DCrossAttention
        x, cross_attn_w = self.cross_attn(query=x,
                                          key=None,
                                          value=value,
                                          pos_embed=pos_embed,
                                          ref_pts=ref_pts,
                                          img_metas=img_metas,
                                          attn_mask=None)
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

        outputs = (x, self_attn_w, cross_attn_w)
        return outputs
```