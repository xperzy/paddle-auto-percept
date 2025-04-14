# 从零开始实现 BEVFormer (2) - BEVFormerHead类

BEVFormerHead，主要是定义了：

1. Transformer
2. 类别预测头
3. 框回归预测头
4. position encoding：是给bev_query加的位置信息
5. object query：是在Decoder中用到的query
    1. 这里定义的Embedding层的维度是 2 * embed_dim，包含了query本身和可学习的位置编码
    2. 会在计算attention的时候拆开
6. bev query：在encoder中用到的query，这个维度是embed_dim，是因为位置编码是self.pe_layer计算得到的，不用在这里包含。
7. get_box和decode方法主要是用来解析结果用的，更多的是格式转换和计算。

```python
class BEVFormerHead(nn.Layer):
    """BEVFormer detection head"""
    def __init__(self,
                 num_classes=10,
                 num_queries=900,
                 bev_h=200,
                 bev_w=200,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 num_feature_levels=4, # num of img feats from fpn
                 num_cams=6,  # num of cameras
                 num_encoder_layers=6,  # num of encoder layers
                 num_points_in_pillar=4,
                 num_decoder_layers=6,  # num of decoder layers
                 num_levels=4, # decoder img feat levels
                 embed_dim=256,
                 num_heads=8,  # same for self-attn and cross-attn
                 self_attn_dropout=0.1,
                 cross_attn_dropout=0.0,
                 ffn_dim=512,
                 ffn_dropout=0.1,
                 num_points=8):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]

        self.code_size = 10
        # code weights is used for training
        self.code_weights = paddle.create_parameter(shape=[self.code_size], dtype='float32')
        self.code_weights.set_value(
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))

        # position encoding
        self.pe_layer = pe.build_position_encoding(
            embed_dim, mode='learned', row_embed_dim=bev_h, col_embed_dim=bev_w)
        # query contains query_pos and target
        self.query_embeddings = nn.Embedding(num_queries, embed_dim*2)
        # bev embedding
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dim)
        # transformer
        self.transformer = BEVFormerTransformer(
            pc_range=pc_range,
            num_feature_levels=num_feature_levels,
            num_cams=num_cams,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            self_attn_dropout=self_attn_dropout,
            cross_attn_dropout=cross_attn_dropout,
            ffn_dim=ffn_dim,
            ffn_dropout=ffn_dropout,
            num_points=num_points,
            num_points_in_pillar=num_points_in_pillar)
        # classification head for category
        class_embed_list = []
        for idx in range(num_decoder_layers):
            class_embed_list.append(paddle.nn.Sequential(
                paddle.nn.Linear(embed_dim, embed_dim),
                paddle.nn.LayerNorm(embed_dim),
                paddle.nn.ReLU(),
                paddle.nn.Linear(embed_dim, embed_dim),
                paddle.nn.LayerNorm(embed_dim),
                paddle.nn.ReLU(),
                paddle.nn.Linear(embed_dim, num_classes)))
        self.class_embed = paddle.nn.LayerList(class_embed_list)
        # regression head for bbox
        bbox_embed_list = []
        for idx in range(num_decoder_layers):
            bbox_embed_list.append(paddle.nn.Sequential(
                paddle.nn.Linear(embed_dim, embed_dim),
                paddle.nn.ReLU(),
                paddle.nn.Linear(embed_dim, embed_dim),
                paddle.nn.ReLU(),
                paddle.nn.Linear(embed_dim, self.code_size)))
        self.bbox_embed = paddle.nn.LayerList(bbox_embed_list)

    def forward(self, multi_level_feats, img_metas, prev_bev=None):
        """
        Args:
            x: multi level features, a list/tuple of Tensor of shape [B, N, C, H, W] 
        """
        # multi_level_feats[0]: [bs, num_cams, c, h', w']
        # each level's h' and w' are different
        bs = multi_level_feats[0].shape[0]
        object_query_embeds = self.query_embeddings.weight
        bev_queries = self.bev_embedding.weight
        tensor_list = EasyDict()
        tensor_list['tensors'] = paddle.zeros([bs, self.bev_h, self.bev_w])
        bev_pos = self.pe_layer(tensor_list)

        out = self.transformer(multi_level_feats=multi_level_feats,
                               object_query_embeds=object_query_embeds,
                               bev_queries=bev_queries,
                               bev_h=self.bev_h,
                               bev_w=self.bev_w,
                               grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                               bev_pos=bev_pos,
                               bbox_embed=self.bbox_embed,
                               img_metas=img_metas,
                               prev_bev=prev_bev)
        bev_embed = out[0]
        all_hidden_states = out[1]  # [num_decoder_layers + 1, bs, num_queries, embed_dim]
        init_reference = out[2]
        inter_reference = out[3]
        # [bs, n_levels, num_queries, 3] -> [n_levels, bs, num_queries, 3]
        inter_reference = inter_reference.transpose([1, 0, 2, 3])

        output_classes = []
        output_coords = []
        for level_idx in range(len(all_hidden_states) - 1):
            if level_idx == 0:
                reference = init_reference  # 1st is the original ref_pts
            else:
                reference = inter_reference[level_idx - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level_idx](all_hidden_states[level_idx + 1])
            # refine bbox
            tmp = self.bbox_embed[level_idx](all_hidden_states[level_idx + 1])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] *
                (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] *
                (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] *
                (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
            output_coord = tmp

            output_classes.append(output_class)
            output_coords.append(output_coord)

        output_classes = paddle.stack(output_classes)
        output_coords = paddle.stack(output_coords)

        outs = {'bev_embed': bev_embed,
                'all_cls_scores': output_classes,
                'all_bbox_preds': output_coords}

        return outs
```