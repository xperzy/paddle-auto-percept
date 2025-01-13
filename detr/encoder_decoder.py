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
DETR in Paddle

A Paddle Implementation of DETR as described in:
"End-to-End Object Detection with Transformers"
Paper Link: https://arxiv.org/abs/2005.12872
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DetrMultiHeadAttention(nn.Layer):
    """Multi head attention for Detr self-attn and cross-attn"""
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
        out = self.dropout(out)

        return out, attn_reshaped


class DetrEncoderLayer(nn.Layer):
    """Detr Encoder Layer: self-attn and ffn"""
    def __init__(self, embed_dim, ffn_dim, num_heads, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        # Self-Attn
        self.self_attn = DetrMultiHeadAttention(embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout_rate=dropout_rate)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        # FFN
        self.act = nn.ReLU()
        self.act_dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                x,
                attn_mask,
                pos_embeds):
        # self-attn
        h = x
        x, attn_w = self.self_attn(x, attn_mask, pos_embeds)
        x = self.dropout(x)
        x = h + x
        x = self.self_attn_norm(x)

        # ffn
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


class DetrEncoder(nn.Layer):
    """Detr Encoder"""
    def __init__(self, embed_dim, ffn_dim, num_heads, num_encoder_layers=6, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.LayerList([
            DetrEncoderLayer(embed_dim=embed_dim,
                             ffn_dim=ffn_dim,
                             num_heads=num_heads,
                             dropout_rate=dropout_rate) for _ in range(num_encoder_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_embeds, attn_mask, pos_embeds):
        x = input_embeds
        x = self.dropout(x)
        encoder_states = []  # stores [input, layer1_out, layer2_out ... layerN_out]
        all_attn_w = []

        if attn_mask is not None:
            bs, seq_l = attn_mask.shape
            attn_mask = attn_mask.reshape([bs, 1, 1, seq_l])
            attn_mask = 1 - attn_mask  # now padded area is 1, image area is 0
            # set padded area with small value
            attn_mask = paddle.masked_fill(
                paddle.zeros(attn_mask.shape), attn_mask, paddle.finfo(paddle.float32).min)

        encoder_states.append(x)
        for encoder_layer in self.layers:
            # inference
            layer_out = encoder_layer(x, attn_mask, pos_embeds)
            x, attn_w = layer_out
            encoder_states.append(x)
            all_attn_w.append(attn_w)

        return x, encoder_states, all_attn_w


class DetrDecoderLayer(nn.Layer):
    """Detr Encoder Layer: self-attn, cross-attn and ffn"""
    def __init__(self, embed_dim, ffn_dim, num_heads, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        # Self-Attn
        self.self_attn = DetrMultiHeadAttention(embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout_rate=dropout_rate)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        # Cross-Attn
        self.cross_attn = DetrMultiHeadAttention(embed_dim=embed_dim,
                                                 num_heads=num_heads,
                                                 dropout_rate=dropout_rate)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        # FFN
        self.act = nn.ReLU()
        self.act_dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                x,
                attn_mask,
                pos_embeds,
                encoder_x,
                encoder_attn_mask,
                encoder_pos_embeds):
        # self-attn
        h = x
        x, attn_w = self.self_attn(x, attn_mask, pos_embeds)
        x = self.dropout(x)
        x = h + x
        x = self.self_attn_norm(x)
        # cross-attn
        h = x
        x, cross_attn_w = self.cross_attn(x=x,
                                          attn_mask=encoder_attn_mask,
                                          pos_embeds=pos_embeds,
                                          encoder_x=encoder_x,
                                          encoder_pos_embeds=encoder_pos_embeds)
        x = self.dropout(x)
        x = h + x
        x = self.cross_attn_norm(x)
        # ffn
        h = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = h + x
        x = self.norm(x)

        outputs = (x, attn_w, cross_attn_w)
        return outputs


class DetrDecoder(nn.Layer):
    """Detr Decoder"""
    def __init__(self, embed_dim, ffn_dim, num_heads, num_decoder_layers=6, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.LayerList([
            DetrDecoderLayer(embed_dim=embed_dim,
                             ffn_dim=ffn_dim,
                             num_heads=num_heads,
                             dropout_rate=dropout_rate) for _ in range(num_decoder_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self,
                input_embeds,
                attn_mask,
                pos_embeds,
                encoder_x,
                encoder_attn_mask,
                encoder_pos_embeds):
        x = input_embeds
        decoder_states = []  # stores [input, layer1_out, layer2_out ... layerN_out]
        all_self_attn_w = []
        all_cross_attn_w = []

        if encoder_attn_mask is not None:
            bs, seq_l = encoder_attn_mask.shape
            _, tgt_l, _ = input_embeds.shape  # [bs, tgt_l, embed_dim]
            encoder_attn_mask = encoder_attn_mask.reshape([bs, 1, 1, seq_l])
            encoder_attn_mask = encoder_attn_mask.expand([bs, 1, tgt_l, seq_l])
            encoder_attn_mask = 1 - encoder_attn_mask  # now padded area is 1, image area is 0
            # set padded area with small value
            encoder_attn_mask = paddle.masked_fill(paddle.zeros(encoder_attn_mask.shape),
                                                   encoder_attn_mask,
                                                   paddle.finfo(paddle.float32).min)

        for decoder_layer in self.layers:
            decoder_states.append(x)
            # inference
            layer_out = decoder_layer(x,
                                      attn_mask,
                                      pos_embeds,
                                      encoder_x,
                                      encoder_attn_mask,
                                      encoder_pos_embeds)
            x, self_attn_w, cross_attn_w = layer_out

            all_self_attn_w.append(self_attn_w)
            all_cross_attn_w.append(cross_attn_w)

        x = self.layernorm(x)
        decoder_states.append(x)


        return x, decoder_states, all_self_attn_w, all_cross_attn_w
