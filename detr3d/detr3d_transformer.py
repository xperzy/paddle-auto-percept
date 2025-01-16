# Copyright (c) 2024 PaddleAutoPercept Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DETR3D in Paddle

A Paddle Implementation of Deformable DETR3D as described in:
"DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries"
Paper Link: https://arxiv.org/abs/2110.06922
"""
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import position_embedding as pe


def inverse_sigmoid(x, eps=1e-5):
    """ inverse op of sigmoid"""
    # sigmoid: 1 / ( 1+ e**(-x))
    # inverse sigmoid: log(x / (1-x))
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


class Detr3DHead(nn.Layer):
    """"DETR 3D for object detection"""
    def __init__(self,
                 num_classes,
                 num_queries,
                 pc_range,
                 num_feature_levels=4,
                 num_cams=6,
                 num_layers=6,
                 embed_dim=256,
                 num_heads=8,
                 self_attn_dropout=0.1,
                 cross_attn_dropout=0.1,
                 ffn_dim=512,
                 ffn_dropout=0.1,
                 num_points=1):
        super().__init__()
        # code weights is used for training
        self.code_size = 10
        self.code_weights = paddle.create_parameter(shape=[self.code_size], dtype='float32')
        self.code_weights.set_value(
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))
        # TODO: Check 128 or 256; position encoding
        self.pe_layer = pe.build_position_encoding(128, 'sine')
        # query contains query and query_pos
        self.query_position_embeddings = paddle.nn.Embedding(num_queries, embed_dim * 2)
        self.pc_range = pc_range
        # Detr3D Decoder
        self.transformer = Detr3DDecoder(pc_range=pc_range,
                                         num_feature_levels=num_feature_levels,
                                         num_cams=num_cams,
                                         num_layers=num_layers,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         self_attn_dropout=self_attn_dropout,
                                         cross_attn_dropout=cross_attn_dropout,
                                         ffn_dim=ffn_dim,
                                         ffn_dropout=ffn_dropout,
                                         num_points=num_points)
        # classification head for category
        class_embed_list = []
        for idx in range(num_layers):
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
        for idx in range(num_layers):
            bbox_embed_list.append(paddle.nn.Sequential(
                paddle.nn.Linear(embed_dim, embed_dim),
                paddle.nn.ReLU(),
                paddle.nn.Linear(embed_dim, embed_dim),
                paddle.nn.ReLU(),
                paddle.nn.Linear(embed_dim, self.code_size)))
        self.bbox_embed = paddle.nn.LayerList(bbox_embed_list)

    def forward(self, multi_level_feats, img_metas):
        """
        Args:
            multi_level_feats: tuple/list of 5D tensor of shape (B, N, C, H, W)
        """
        query_pos_embeds = self.query_position_embeddings.weight
        out = self.transformer(multi_level_feats=multi_level_feats,
                               query_pos_embeds=query_pos_embeds,
                               bbox_embed=self.bbox_embed,
                               img_metas=img_metas)

        hidden_states = out[0]  # [bs, num_queries, embed_dim]
        init_reference = out[1]  # [bs, num_queries, 3]
        inter_reference = out[3]  # [bs, num_cams, num_queries, 3]
        inter_reference = inter_reference.transpose([1, 0, 2, 3])  # [num_cams, bs, num_queries, 3]
        all_hidden_states = out[4] # list of len 7, [bs, num_queries, embed_dim]

        output_classes = []
        output_coords = []
        for level_idx in range(len(all_hidden_states)-1): 
            if level_idx == 0:
                reference = init_reference  # 1st is the original ref_pts
            else:
                # 1st item in inter_reference is the output from 1st layer
                reference = inter_reference[level_idx - 1]
            reference = inverse_sigmoid(reference)
            # 1st element in all_hidden_states is the input (not needed here)
            output_class = self.class_embed[level_idx](all_hidden_states[level_idx + 1])
            # refine bbox
            tmp = self.bbox_embed[level_idx](all_hidden_states[level_idx + 1])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            tmp[..., 1:2] = tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            tmp[..., 4:5] = tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            output_coord = tmp

            output_classes.append(output_class)
            output_coords.append(output_coord)

        output_classes = paddle.stack(output_classes)
        output_coords = paddle.stack(output_coords)

        outs = {'all_cls_scores': output_classes, 'all_bbox_preds': output_coords}

        return outs

    def get_bboxes(self, pred_dicts, img_metas, rescale=False):
        """
        Arge:
            pred_dictsi['all_cls_scores']: [num_decoder_layers, bs, num_queries, class_out_channels]
            pred_dictsi['all_bbox_preds']: [num_decoder_layers, bs, num_queries, 9]
        """
        def denormalize_bboxes(norm_bboxes):
            # rotation
            rot_sine = norm_bboxes[..., 6:7] # ':' is used to keep the dim
            rot_cosine = norm_bboxes[..., 7:8]
            rot = paddle.atan2(rot_sine, rot_cosine)
            # center in bev
            cx = norm_bboxes[..., 0:1]
            cy = norm_bboxes[..., 1:2]
            cz = norm_bboxes[..., 4:5]
            # size
            w = norm_bboxes[..., 2:3]
            l = norm_bboxes[..., 3:4]
            h = norm_bboxes[..., 5:6]
            w = w.exp()
            l = l.exp()
            h = h.exp()
            # velocity
            vx = norm_bboxes[..., 8:9]
            vy = norm_bboxes[..., 9:10]
            # output
            denorm_bboxes = paddle.concat([cx, cy, cz, w, l, h, rot, vx, vy], axis=-1)
            return denorm_bboxes

        def decode(cls_scores,
                   bbox_preds,
                   pc_range,
                   max_num=300,
                   num_classes=10,
                   post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                   score_threshold=None):
            """
            Args:
                cls_scores: Tensor of shape [num_queries, cls_out_channels];  
                            cls_out_channels should include background
                bbox_preds: Tensor of shape [num_queries, 9];
                            (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)
            """
            cls_scores = cls_scores.sigmoid()
            scores, indices = cls_scores.reshape([-1]).topk(max_num)
            labels = indices % num_classes
            bbox_indices = indices // num_classes
            bbox_preds = bbox_preds[bbox_indices]

            final_bbox_preds = denormalize_bboxes(bbox_preds)  # [300, 10]
            final_scores = scores
            final_preds = labels

            if score_threshold is not None:
                thresh_mask = final_scores > score_threshold

            if post_center_range is not None:
                post_center_range = paddle.to_tensor(post_center_range)

                mask = (final_bbox_preds[..., :3] >= post_center_range[:3]).all(1)  # [300]
                mask = mask & (final_bbox_preds[..., :3] <= post_center_range[3:]).all(1)

                if score_threshold is not None:
                    mask = mask & thresh_mask

                boxes3d = final_bbox_preds[mask]
                scores = final_scores[mask]
                labels = final_preds[mask]

                pred_dict = {'bboxes': boxes3d,
                             'scores': scores,
                             'labels': labels}
                return pred_dict

        # use the last decoder layer output
        all_cls_scores = pred_dicts['all_cls_scores'][-1]
        all_bbox_preds = pred_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.shape[0]
        pred_dicts = []
        for i in range(batch_size):
            pred_dicts.append(decode(all_cls_scores[i], all_bbox_preds[i], self.pc_range))

        num_samples = len(pred_dicts)
        res_list = []
        for i in range(num_samples):
            preds = pred_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            scores = preds['scores']
            labels = preds['labels']
            res_list.append([bboxes, scores, labels])
        return res_list


class Detr3DDecoder(nn.Layer):
    """DETR3D Decoder"""
    def __init__(self,
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
                 num_points=1):
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
                query_pos_embeds,
                img_metas,
                bbox_embed=None):
        """
        Args:
            multi_level_feats: tuple/list of 5D tensor of shape (B, N, C, H, W)
            query_embeds: (num_queries, embed_dim * 2) 
        """
        bs = multi_level_feats[0].shape[0]
        query_pos, query = paddle.split(query_pos_embeds, 2, axis=1)  # 2 x [num_queries, embed_dim]
        query_pos = query_pos.unsqueeze(0).expand([bs, -1, -1])  # [bs, num_queries, embed_dim]
        query = query.unsqueeze(0).expand([bs, -1, -1])  # [bs, num_queries, embed_dim]
        reference_points = self.reference_points(query_pos).sigmoid()  # [bs, num_queries, 3]
        init_reference_points = reference_points

        # Note: detr3d pytorch code needs these transpose, 
        #       since they call torch.nn.MultiheadAttention with batch_first=False
        #       We do NOT need it!
        # target = target.transpose([1, 0, 2])
        # query_embed = query_embed.transpose([1, 0, 2])

        all_hidden_states = ()
        all_self_attentions = ()
        all_cross_attentions = ()
        intermediate = ()
        intermediate_reference_points = ()

        hidden_states = query  # used to iterate through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points
            # store all hidden states
            all_hidden_states = all_hidden_states + (hidden_states, )
            # inference
            layer_out = decoder_layer(x=hidden_states,
                                      value=multi_level_feats,
                                      pos_embed=query_pos,
                                      ref_pts=reference_points_input,
                                      img_metas=img_metas)
            out = layer_out[0]
            # refine reference points
            if bbox_embed is not None:
                tmp = bbox_embed[idx](layer_out[0])
                new_reference_points = paddle.zeros_like(reference_points)

                new_reference_points[..., :2] = (tmp[..., :2] +
                    inverse_sigmoid(reference_points[..., :2]))

                new_reference_points[..., 2:3] = (tmp[..., 4:5] +
                    inverse_sigmoid(reference_points[..., 2:3]))

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
        outputs = (hidden_states,
                   init_reference_points,
                   intermediate,
                   intermediate_reference_points,
                   all_hidden_states,
                   all_self_attentions,
                   all_cross_attentions)
        return outputs


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


class MultiheadAttention(nn.Layer):
    """ Multi head self attention"""
    def __init__(self, embed_dim, num_heads, dropout_rate, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.k = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)

        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(axis=-1)

    def reshape_to_multi_heads(self, x, seq_l, batch_size):
        """
        convert [batch_size, seq_l, embed_dim] -> [batch_size * num_heads, seq_len, head_dim]
        """
        x = x.reshape([batch_size, seq_l, self.num_heads, self.head_dim])
        x = x.transpose([0, 2, 1, 3])
        x = x.reshape([batch_size * self.num_heads, seq_l, self.head_dim])
        return x

    def forward(self, x, attn_mask=None, pos_embed=None):
        h = x
        bs, seq_l, _ = x.shape

        x_q = x + pos_embed if pos_embed is not None else x
        x_k = x_q
        x_v = x

        q = self.q(x_q) * self.scale
        q = self.reshape_to_multi_heads(q, seq_l, bs)  # [bs*num_heads, seq_l, head_dim]
        k = self.k(x_k)
        k = self.reshape_to_multi_heads(k, seq_l, bs)  # [bs*num_heads, seq_l, head_dim]
        v = self.v(x_v)
        v = self.reshape_to_multi_heads(v, seq_l, bs)  # [bs*num_heads, seq_l, head_dim]

        attn = paddle.matmul(q, k, transpose_y=True)  #[bs*num_heads, seq_l, seq_l]

        # mask
        if attn_mask is not None:
            attn_mask = paddle.masked_fill(paddle.zeros(attn_mask.shape) ,
                                          attn_mask,
                                          float('-inf'))

        attn = attn.reshape([bs, self.num_heads, seq_l, seq_l])
        attn = attn + attn_mask if attn_mask is not None else attn

        attn = attn.reshape([bs * self.num_heads, seq_l, seq_l])
        attn = self.softmax(attn)

        attn_reshaped = attn.reshape([bs, self.num_heads, seq_l, seq_l])
        attn = attn_reshaped.reshape([bs * self.num_heads, seq_l, seq_l])

        attn = self.dropout(attn)
        out = paddle.matmul(attn, v)

        # output
        out = out.reshape([bs, self.num_heads, seq_l, self.head_dim])
        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([bs, seq_l, self.num_heads * self.head_dim])
        out = self.out_proj(out)
        out = self.dropout(out)
        out = h + out

        return out, attn_reshaped


class Detr3DCrossAttention(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 n_levels,
                 num_points,
                 num_cams,
                 pc_range,
                 dropout_rate):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_points = num_points
        self.n_levels = n_levels
        self.num_cams = num_cams
        self.pc_range = pc_range
        self.dropout_rate = dropout_rate

        self.attn = nn.Linear(embed_dim, num_cams * n_levels * num_points)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU())

        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                query,
                key,
                value,
                attn_mask,
                pos_embed,
                ref_pts,
                img_metas):
        key = query if key is None else key
        value = key if value is None else value

        bs, seq_l, _ = key.shape

        h = query
        query = query + pos_embed if pos_embed is not None else query
        bs, n_queries, _ = query.shape

        attn = self.attn(query)  # [bs, num_queries, num_cams*num_points*n_levels]
        attn = attn.reshape([bs, 1, n_queries, self.num_cams, self.num_points, self.n_levels])

        reference_points_3d, output, mask = feature_sampling(value, ref_pts, self.pc_range, img_metas)

        # output: [bs, embed_dim, num_queries, num_cams, 1, n_levels]
        output = paddle.nan_to_num(output)
        # mask: [bs, 1, num_queries, num_cams, 1, 1]
        mask = paddle.nan_to_num(mask)

        attn = attn.sigmoid() * mask  # [bs, 1, num_queries, num_cams, num_points, n_levels]

        output = output * attn  # [bs, embed_dim, num_queries, num_cams, num_points, n_levels]
        output = output.sum(-1).sum(-1).sum(-1)  # [bs, embed_dim, num_queries]
        output = output.transpose([0, 2, 1])  # [bs, num_queries, embed_dim]

        output = self.out_proj(output)  # [num_queries, bs, embed_dim]
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d))

        output = self.dropout(output)
        output = h + output
        output = output + pos_feat

        return output, attn


def feature_sampling(multi_level_feats, reference_points, pc_range, img_metas):
    # get lidar2img projection matrix for each view
    bs = reference_points.shape[0]
    num_cams = len(img_metas[0]['lidar2img'])
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = paddle.to_tensor(np.asarray(lidar2img), dtype='float32')
    lidar2img = lidar2img.reshape([1, num_cams, 4, 4])
    lidar2img = lidar2img.expand([bs, num_cams, 4, 4])

    # refence points
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0] 
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1] 
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2] 
    # [bs, num_queries, 3] -> [bs, num_queries, 4]
    reference_points = paddle.concat([reference_points,
                                      paddle.ones_like(reference_points[..., :1])], axis=-1)
    num_queries = reference_points.shape[1]

    reference_points = reference_points.reshape([bs, 1, num_queries, 4])
    reference_points = reference_points.expand([bs, num_cams, num_queries, 4])
    reference_points = reference_points.unsqueeze(-1)  # [bs, num_cams, num_queries, 4, 1]

    lidar2img = lidar2img.unsqueeze(2)  # [bs, num_cams, 1, 4, 4]
    lidar2img = lidar2img.expand([bs, num_cams, num_queries, 4, 4])

    # [bs, n_cam, 1, 4, 4] * [bs, n_cam, n_query, 4, 1]
    reference_points_cam = paddle.matmul(lidar2img, reference_points)
    # [bs, n_cam, n_query, 4, 1] -> [bs, n_cam, n_query, 4]
    reference_points_cam = reference_points_cam.squeeze(-1)

    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)

    # [x, y, z, 1] -> [x/z, y/z]
    reference_points_cam = reference_points_cam[..., 0:2] / paddle.maximum(
        reference_points_cam[..., 2:3], paddle.ones_like(reference_points_cam[..., 2:3]) * eps)
    # normalize to (0, 1)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    # normalize to (-1, 1)
    reference_points_cam = (reference_points_cam - 0.5) * 2

    mask = mask & (reference_points_cam[..., 0:1] > -1.0)
    mask = mask & (reference_points_cam[..., 0:1] < 1.0)
    mask = mask & (reference_points_cam[..., 1:2] > -1.0)
    mask = mask & (reference_points_cam[..., 1:2] < 1.0)

    mask = mask.reshape([bs, num_cams, 1, num_queries, 1, 1])
    mask = mask.transpose([0, 2, 3, 1, 4, 5])  # [bs, 1, num_queries, num_cams, 1, 1]
    mask = mask * 1.0  # from boolean to float
    mask = paddle.nan_to_num(mask)

    sampled_feats = []
    for level, feat in enumerate(multi_level_feats):
        b, n, c, h, w = feat.shape  # N == num_cams
        feat = feat.reshape([b*n, c, h, w])
        reference_points_cam_level = reference_points_cam.reshape([b*n, num_queries, 1, 2])
        sampled_feat = F.grid_sample(feat, reference_points_cam_level, align_corners=False)
        sampled_feat = sampled_feat.reshape([b, n, c, num_queries, 1])
        sampled_feat = sampled_feat.transpose([0, 2, 3, 1, 4])  # [B, C, num_queries, N, 1]
        sampled_feats.append(sampled_feat)

    sampled_feats = paddle.stack(sampled_feats, -1)  # [B, C, num_queries, N, n_levels]
    sampled_feats = sampled_feats.reshape([b, c, num_queries, num_cams, 1, len(multi_level_feats)])

    return reference_points_3d, sampled_feats, mask
