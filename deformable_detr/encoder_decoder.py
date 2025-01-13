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
Deformable DETR Encoder in Paddle

A Paddle Implementation of Deformable DETR as described in:
"Deformable DETR: Deformable Transformers for End-to-End Object Detection"
Paper Link: https://arxiv.org/abs/2010.04159
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def multiscale_deformable_attention(x_v, spatial_shapes, sampling_locations, attn):
    """deformable attention in paddle"""
    # get shapes
    bs, _, num_heads, head_dim = x_v.shape
    _, tgt_l, _, num_levels, num_points, _ = sampling_locations.shape
    # split each level
    spatial_shapes_list = spatial_shapes.numpy().tolist()
    v_list = x_v.split([h * w for h, w in spatial_shapes_list], axis=1)
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


class MultiscaleDeformableAttention(nn.Layer):
    """Multi scale deformable attention for DeformableDetr"""
    def __init__(self, embed_dim, num_heads, num_points, num_levels):
        super().__init__()
        self. embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_points = num_points
        self.num_levels = num_levels

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attn = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.softmax = nn.Softmax(-1)

        self.v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, value, attn_mask, pos_embeds, ref_pts, spatial_shapes, level_start_index):
        bs, tgt_l, _ = x.shape
        value = x if value is None else value
        _, src_l, _ = value.shape

        x_v = self.v(value)
        x_q = x + pos_embeds if pos_embeds is not None else x
        if attn_mask is not None:
            x_v = x_v.masked_fill(attn_mask[..., None], 0.0)
        x_v = x_v.reshape([bs, src_l, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(x_q)
        sampling_offsets = sampling_offsets.reshape(
            [bs, tgt_l, self.num_heads, self.num_levels, self.num_points, 2])

        attn = self.attn(x_q)
        attn = attn.reshape([bs, tgt_l, self.num_heads, self.num_levels * self.num_points])
        attn = self.softmax(attn)
        attn = attn.reshape([bs, tgt_l, self.num_heads, self.num_levels, self.num_points])

        # spatial shapes: [num_levels, 2]
        offset_normalizer = paddle.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

        # ref_pts: [bs, tgt_l, num_levels, 2]
        # sampling_offsets: [bs, tgt_l, num_heads, num_levels, num_points, 2]
        # offset_normalizer: [num_levels, 2]
        # sampling_locations: [bs, tgt_l, num_heads, num_levels, num_points, 2]
        sampling_locations = (ref_pts[:, :, None, :, None, :] +
                              sampling_offsets / offset_normalizer[None, None, None, :, None, :])

        out = multiscale_deformable_attention(x_v, spatial_shapes, sampling_locations, attn)
        out = self.out_proj(out)
        return out, attn


class MultiheadAttention(nn.Layer):
    """Multi head attention for DeformableDetr"""
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(-1)

    def reshape_to_multi_heads(self, x, seq_l, bs):
        x = x.reshape([bs, seq_l, self.num_heads, self.head_dim])
        x = x.transpose([0, 2, 1, 3])
        x = x.reshape([bs * self.num_heads, seq_l, self.head_dim])
        return x

    def forward(self,
                x,
                attn_mask,
                pos_embeds,
                encoder_x=None,
                encoder_pos_embeds=None):
        x_q = x + pos_embeds if pos_embeds is not None else x

        if encoder_x is None:  # self-attn
            x_k = x_q
            x_v = x
        else:  # cross-attn
            x_k = encoder_x + encoder_pos_embeds if encoder_pos_embeds is not None else encoder_x
            x_v = encoder_x

        bs, tgt_l, _ = x_q.shape
        _, src_l, _ = x_v.shape

        q = self.q(x_q) * self.scale
        q = self.reshape_to_multi_heads(q, tgt_l, bs)  # [bs*num_heads, tgt_l, head_dim]
        k = self.k(x_k)
        k = self.reshape_to_multi_heads(k, src_l, bs)  # [bs*num_heads, src_l, head_dim]
        v = self.v(x_v)
        v = self.reshape_to_multi_heads(v, src_l, bs)  # [bs*num_heads, src_l, head_dim]

        attn = paddle.matmul(q, k, transpose_y=True)  # [bs*numheads, tgt_l, src_l]
        
        # attn mask: padded area is set to small number
        if attn_mask is not None:
            attn = attn.reshape([bs, self.num_heads, tgt_l, src_l])
            attn = attn + attn_mask  # [bs, num_heads, tgt_l, src_l] + [bs, 1, tgt_l, src_l]
            attn = attn.reshape([bs * self.num_heads, tgt_l, src_l])
        attn = self.softmax(attn)
        # return attn_reshaped, reshape back is to ensure attn keeps its gradient
        attn_reshaped = attn.reshape([bs, self.num_heads, tgt_l, src_l])
        attn = attn_reshaped.reshape([bs * self.num_heads, tgt_l, src_l])
        attn = self.dropout(attn)

        out = paddle.matmul(attn, v)
        out = out.reshape([bs, self.num_heads, tgt_l, self.head_dim])
        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([bs, tgt_l, self.num_heads * self.head_dim])
        out = self.out_proj(out)

        return out, attn_reshaped


class DeformableDetrEncoderLayer(nn.Layer):
    """"Encoder Layer for Deformable Detr"""
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 num_heads,
                 num_points,
                 num_levels,
                 dropout_rate=0.0):
        super().__init__()
        # Self-Attn
        self.self_attn = MultiscaleDeformableAttention(embed_dim, num_heads, num_points, num_levels)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        # FFN
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.ReLU()
        self.act_dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                x,
                attn_mask,
                pos_embeds,
                ref_pts,
                spatial_shapes,
                level_start_index):
        # Self-Attn
        h = x
        x, attn_w = self.self_attn(x=x,
                                   value=None,
                                   attn_mask=attn_mask,
                                   pos_embeds=pos_embeds,
                                   ref_pts=ref_pts,
                                   spatial_shapes=spatial_shapes,
                                   level_start_index=level_start_index)
        x = self.dropout(x)
        x = h + x
        x = self.self_attn_norm(x)
        # FFN
        h = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = h + x
        x = self.norm(x)

        outputs = (x, attn_w)
        return outputs


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
        print('ref_pts: ', ref_pts.shape)
        print('valid_ratios: ', valid_ratios.shape)
        print('after')
        print('ref_pts[:, :, None]: ', ref_pts[:, :, None].shape)
        print('valid_ratios[:, None]: ', valid_ratios[:, None].shape)

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


class DeformableDetrDecoderLayer(nn.Layer):
    """"Decoder Layer for Deformable Detr"""
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 num_heads,
                 num_points,
                 num_levels,
                 dropout_rate=0.0):
        super().__init__()
        # Self-Attn
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout_rate)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        # Cross-Attn
        self.cross_attn = MultiscaleDeformableAttention(embed_dim, num_heads, num_points, num_levels)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        # FFN
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.ReLU()
        self.act_dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                x,
                attn_mask,
                encoder_x,
                encoder_attn_mask,
                pos_embeds,
                ref_pts,
                spatial_shapes,
                level_start_index):
        # Self-Attn
        h = x
        x, self_attn_w = self.self_attn(x=x,
                                        attn_mask=attn_mask,
                                        pos_embeds=pos_embeds)
        x = self.dropout(x)
        x = h + x
        x = self.self_attn_norm(x)
        # Cross-Attn
        h = x
        x, cross_attn_w = self.cross_attn(x=x,
                                          value=encoder_x,
                                          attn_mask=encoder_attn_mask,
                                          pos_embeds=pos_embeds,
                                          ref_pts=ref_pts,
                                          spatial_shapes=spatial_shapes,
                                          level_start_index=level_start_index)
        x = self.dropout(x)
        x = h + x
        x = self.cross_attn_norm(x)
        # FFN
        h = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = h + x
        x = self.norm(x)


        outputs = (x, self_attn_w, cross_attn_w)
        return outputs



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
