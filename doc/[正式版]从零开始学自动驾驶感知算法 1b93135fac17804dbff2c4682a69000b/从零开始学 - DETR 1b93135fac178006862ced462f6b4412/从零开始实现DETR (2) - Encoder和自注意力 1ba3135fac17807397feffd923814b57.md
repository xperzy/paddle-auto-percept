# 从零开始实现DETR (2) -  Encoder和自注意力

![Encoder Layer and Multi-head Self-Attention](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0DETR%20(2)%20-%20Encoder%E5%92%8C%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%201ba3135fac17807397feffd923814b57/image.png)

Encoder Layer and Multi-head Self-Attention

## MultiHead Self-Attention结构：

```python
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

        return out, attn_reshaped---
```

`MultiheadAttention`在实现时，需要注意：

1. `MultiheadAttention`方法需要同时支持Self-Attention和Cross Attention。也就是说，在self attention中，对于线性变换`self.q，self.k，self.v`的输入是相同的，都来自于同一个输入，而在Cross Attention中，`self.q`的输入来自上一层输出（第一层的输入是全0 tensor），`self.k` 和 `self.v`的输入与`self.q`的不同，他们都时来自于encoder的最后一层输出。
    1. `self.q`和`self.k`的输入需要加上各自的position embeddings，对于encoder中的self attn来说，是图像上的2D空间位置编码。对于decoder中的self attn来说，position embeddings是一个可学习的`query_pos_embedding`， 这个是`nn.embedding`层，用于把`num_queries`个query映射到`embed_dim`维度。对于decoder的cross attention，输入`x`是object query，需要加上的是`query_pos_embedding`, encoder的输出是`encoder_x`,需要加上的则是空间位置编码，是作为方法的参数传进来的。对于v是不需要加入位置信息。
2. Attn mask：输入是`[bs, 1, tgt_l, src_l]`的维度，通过broadcasting机制，与attn分数相加`[bs, num_heads, tgt_l, src_l]`。在attn mask中，图像特征区域被设置为0， padding部分设置为flaot的最小值，与attn分数相加之后，padding的部分再经过softmax就会变成0。对于attn分数来说，padding的部分表示不需要关注这部分。
    - softmax： $\mathrm{softmax}() = \dfrac{e^{z_i}}{\Sigma_i{e^{z_i}}}$，$z_i$越小，$e^{z_i}$就越接近0，所以加上float的最小值（是一个负数），经过softmax之后该项就会近似于0。

![Encoder Layer and Feed Forward Network (FFN)](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0DETR%20(2)%20-%20Encoder%E5%92%8C%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%201ba3135fac17807397feffd923814b57/image%201.png)

Encoder Layer and Feed Forward Network (FFN)

## Encoder Layer 和 FFN：

```python
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
```

这一部分实现了Encoder一层的计算，包括self attention和ffn，以及对应的norm和残差链接，这里无论是注意力计算还是ffn网络，都不改变输入tensor的维度。

## Encoder整体结构:

```python
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
```

这一部分需要注意的是mask的处理：

- attn_mask输入进来的维度是:`[bs, seq_l]`， `seq_l`对应的是feature map的大小h*w
- attn_mask输入进来的时候，有效区域设置为1，padding区域设置为0
- 这里把有效区域设置为0，padding区域设置为-inf，是为了计算attn的时候忽略padding部分区域不参与attn（softmax的时候-inf接近于0）。