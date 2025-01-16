# Copyright (c) 2024 PaddleAutoPercept Authors. All Rights Reserved. # W: Too many lines in module (1344/1000)
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
BEVFormer in Paddle
    
A Paddle Implementation of BEVFormer as described in:
"BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers" # W: Line too long (109/100)
Paper Link: https://arxiv.org/abs/2203.17270
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def inverse_sigmoid(x, eps=1e-5):
    """ inverse op of sigmoid"""
    # sigmoid: 1 / ( 1+ e**(-x))
    # inverse sigmoid: log(x / (1-x))
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


def multiscale_deformable_attention(x_v, spatial_shapes, sampling_locations, attn): # W: Too many local variables (21/15)
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


class BEVFormerDeformableAttention(nn.Layer): # W: Too many instance attributes (11/7)
    """ This class is used only in cross-attn of Decoder layer """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_levels,
                 num_points,
                 dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_points = num_points
        self.num_levels = num_levels

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attn = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.softmax = nn.Softmax(-1)

        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, # W: Number of parameters was 3 in 'Layer.forward' and is now 10 in overriding 'BEVFormerDeformableAttention.forward' method
                x,
                value,
                attn_mask,
                pos_embeds,
                ref_pts,
                spatial_shapes,
                level_start_index):
        bs, tgt_l, _ = x.shape
        value = x if value is None else value
        _, src_l, _ = value.shape

        h = x
        x_v = self.value_proj(value)
        x_q = x + pos_embeds if pos_embeds is not None else x

        if attn_mask is not None:  # [bs, seq_l]
            x_v = paddle.masked_fill(x_v, attn_mask[..., None], 0.0)
        x_v = x_v.reshape([bs, src_l, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(x_q)
        sampling_offsets = sampling_offsets.reshape(
            [bs, tgt_l, self.num_heads, self.num_levels, self.num_points, 2])

        attn = self.attn(x_q)
        attn = attn.reshape([bs, tgt_l, self.num_heads, self.num_levels * self.num_points])
        attn = self.softmax(attn)
        attn = attn.reshape([bs, tgt_l, self.num_heads, self.num_levels, self.num_points])

        # spatial_shapes: [num_levels, 2]
        offset_normalizer = paddle.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (ref_pts[:, :, None, :, None, :] +
                              sampling_offsets / offset_normalizer[None, None, None, :, None, :])

        out = multiscale_deformable_attention(x_v, spatial_shapes, sampling_locations, attn)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = h + out

        return out, attn


class MultiheadAttention(nn.Layer): # W: Too many instance attributes (10/7)
    """Multi head attention"""
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

    def reshape_to_multi_heads(self, x, seq_l, bs): # W: Missing function or method docstring
        x = x.reshape([bs, seq_l, self.num_heads, self.head_dim])
        x = x.transpose([0, 2, 1, 3])
        x = x.reshape([bs * self.num_heads, seq_l, self.head_dim])
        return x

    def forward(self, # W: Number of parameters was 3 in 'Layer.forward' and is now 6 in overriding 'MultiheadAttention.forward' method
                x,
                attn_mask,
                pos_embeds,
                encoder_x=None,
                encoder_pos_embeds=None):
        h = x
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
         # W: Trailing whitespace
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
        out = self.dropout(out)

        out = h + out
        return out, attn_reshaped


class BEVFormerDecoderLayer(nn.Layer):
    """decoder layer for bevformer"""
    def  __init__(self, # W: Too many arguments (9/5)
                  embed_dim,
                  ffn_dim,
                  num_heads,
                  num_points,
                  num_levels,
                  self_attn_dropout,
                  cross_attn_dropout,
                  ffn_dropout):
        super().__init__()
        # self attn
        self.self_attn = MultiheadAttention(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout_rate=self_attn_dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        # cross attn
        self.cross_attn = BEVFormerDeformableAttention(embed_dim=embed_dim,
                                                       num_heads=num_heads,
                                                       num_points=num_points,  # 4
                                                       num_levels=num_levels)  # 1
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
                attn_mask,
                value,
                pos_embeds,
                ref_pts,
                spatial_shapes,
                level_start_index):
        # self-attn: MultiHeadAttention
        x, self_attn_w = self.self_attn(x=x,
                                        pos_embeds=pos_embeds,
                                        attn_mask=None)
        x = self.self_attn_norm(x)
        # cross-attn: BEVFormerDeformableAttention
        x, cross_attn_w = self.cross_attn(x=x,
                                          value=value,
                                          attn_mask=attn_mask,
                                          pos_embeds=pos_embeds,
                                          ref_pts=ref_pts,
                                          spatial_shapes=spatial_shapes,
                                          level_start_index=level_start_index)
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


class BEVFormerDecoder(nn.Layer):
    """BEVformer decoder"""
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_layers,
                 num_levels,
                 num_points,
                 ffn_dim,
                 self_attn_dropout,
                 cross_attn_dropout,
                 ffn_dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = nn.LayerList([
            BEVFormerDecoderLayer(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                num_points=num_points,
                num_levels=num_levels,
                self_attn_dropout=self_attn_dropout,
                cross_attn_dropout=cross_attn_dropout,
                ffn_dropout=ffn_dropout) for idx in range(num_layers)])
    def forward(self,
                input_embeds,
                attn_mask,
                value,
                pos_embeds,
                ref_pts,
                spatial_shapes,
                level_start_index,
                img_metas,
                bbox_embed):
        init_ref_pts = ref_pts
        x = input_embeds

        decoder_states = []
        all_self_attn = []
        all_cross_attn = []
        intermediate_ref_pts = []

        for layer_idx, layer in enumerate(self.layers):
            # [bs, num_queries, 3] -> [bs, num_queries, 1, 2]
            ref_pts_input = ref_pts[..., :2].unsqueeze(2)
            decoder_states.append(x)
            out = layer(x=x,
                        attn_mask=attn_mask,
                        value=value,
                        pos_embeds=pos_embeds,
                        ref_pts=ref_pts_input,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index)
            x = out[0]
            #print('decoder out.shape: ', x.shape)
            # refine ref_pts using bbox_embed
            if bbox_embed is not None:
                tmp = bbox_embed[layer_idx](x)
                new_ref_pts = paddle.zeros_like(ref_pts)
                ref_pts = inverse_sigmoid(ref_pts)
                new_ref_pts[..., :2] = tmp[..., :2] + ref_pts[..., :2]
                new_ref_pts[..., 2:3] = tmp[..., 4:5] + ref_pts[..., 2:3]
                new_ref_pts = new_ref_pts.sigmoid()
                ref_pts = new_ref_pts.detach()

            # save results
            intermediate_ref_pts.append(ref_pts)
            all_self_attn.append(out[1])
            all_cross_attn.append(out[2])

        decoder_states.append(x)
        decoder_states = paddle.stack(decoder_states, 0)
        #print('decoder_states: ', decoder_states.shape)
        intermediate_ref_pts = paddle.stack(intermediate_ref_pts, 1)

        return decoder_states, init_ref_pts, intermediate_ref_pts, all_self_attn, all_cross_attn
