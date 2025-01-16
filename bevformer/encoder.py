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
import numpy as np
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


class BEVFormerDeformableAttention3D(nn.Layer): # W: Too many instance attributes (11/7)
    """ This class is used only in SCA of Encoder layer """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_levels,
                 num_points):
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

    def forward(self, # W: Number of parameters was 3 in 'Layer.forward' and is now 10 in overriding 'BEVFormerDeformableAttention.forward' method
                x,
                value,
                attn_mask,
                pos_embeds,
                ref_pts,
                spatial_shapes,
                level_start_index):
        bs, tgt_l, _ = x.shape
        _, src_l, _ = value.shape

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

        # For each BEV query, in 3D space it has 'num_z_anchors' difference heights,
        # after projecting, there are 'num_z_anchors' ref pts in each 2D imge,
        # for each ref_pt, sample 'num_points' points,
        # therefore the overall num of sampling points is'num_points * num_z_anchors'
        bs, n_queries, n_z_anchors, xy = ref_pts.shape
        ref_pts = ref_pts.reshape([bs, n_queries, 1, 1, 1, n_z_anchors, xy])

        sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        bs, n_queries, n_heads, n_levels, n_all_points, xy = sampling_offsets.shape
        sampling_offsets = sampling_offsets.reshape(
            [bs, n_queries, n_heads, n_levels, n_all_points // n_z_anchors, n_z_anchors, xy]) # W: Line too long (133/10

        sampling_locations = ref_pts + sampling_offsets
        bs, n_queries, n_heads, n_levels, n_points, n_z_anchors, xy = sampling_locations.shape
        assert n_all_points == n_points * n_z_anchors
        sampling_locations = sampling_locations.reshape(
            [bs, n_queries, n_heads, n_levels, n_all_points, xy]) # W: Line too long (109/100)
        out = multiscale_deformable_attention(x_v, spatial_shapes, sampling_locations, attn)

        return out, attn


class SpatialCrossAttention(nn.Layer): # W: Missing class docstring
    """Spatial Cross Attention for bevformer"""
    def  __init__(self, # W: Too many arguments (6/5)
                  embed_dim=256,
                  num_cams=6,
                  num_levels=4,
                  num_heads=8,
                  num_points=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cams = num_cams
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # cross attn
        self.cross_attn = BEVFormerDeformableAttention3D(embed_dim=embed_dim,
                                                         num_heads=num_heads,
                                                         num_points=num_points,
                                                         num_levels=num_levels)

    def forward(self,
                x,
                value,
                pos_embeds,
                ref_pts,
                ref_pts_cam,
                spatial_shapes,
                level_start_index,
                bev_mask): # W: Line too long (110/100)
        bs, tgt_l, _  = x.shape # W: Unused variable 'n_queries'
        slots = paddle.zeros_like(x)
        slots.stop_gradient = True
        x_q = x + pos_embeds if pos_embeds is not None else x  # [num_queries, bs, embed_dim]

        indices = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indices.append(index_query_per_img)
        max_len = max([len(each) for each in indices]) # W: Consider using a generator instead 'max(len(each) for each in indices)'

        x_rebatch = paddle.zeros([bs, self.num_cams, max_len, self.embed_dim])
        ref_pts_rebatch = paddle.zeros([bs, self.num_cams, max_len, ref_pts_cam.shape[3], 2])

        for j in range(bs):
            for i, ref_pt_per_img in enumerate(ref_pts_cam):
                index_query_per_img = indices[i]
                x_rebatch[j, i, :len(index_query_per_img)] = x_q[j, index_query_per_img]
                ref_pts_rebatch[j, i, :len(index_query_per_img)] = ref_pt_per_img[j, index_query_per_img] # W: Line too long (105/100)

        value = value.transpose([2, 0, 1, 3])  #[bs, num_cam, H*W, embed_dim]
        value = value.reshape([bs * self.num_cams, -1, self.embed_dim])

        x_rebatch = x_rebatch.reshape([bs * self.num_cams, max_len, self.embed_dim])
        ref_pts_rebatch = ref_pts_rebatch.reshape(
            [bs * self.num_cams, max_len, ref_pts_cam.shape[3], 2]) # W: Line too long (105/100)

        output, cross_attn_w = self.cross_attn(x=x_rebatch,
                                               value=value,
                                               attn_mask=None,
                                               pos_embeds=pos_embeds,
                                               ref_pts=ref_pts_rebatch,
                                               spatial_shapes=spatial_shapes,
                                               level_start_index=level_start_index)
        output = output.reshape([bs, self.num_cams, max_len, self.embed_dim])

        for j in range(bs):
            for i, index_query_per_img in enumerate(indices):
                slots[j, index_query_per_img] += output[j, i, :len(index_query_per_img), :]

        count = bev_mask.sum(-1) > 0
        count = count.transpose([1, 2, 0]).sum(-1)
        count = paddle.clip(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.out_proj(slots)
        return slots, cross_attn_w


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
        if value is None:  # no prev_bev
            bs, bev_len, _ = x.shape
            # stack 2 current bev
            value = paddle.stack([x, x], 1)
            value = value.reshape([bs * 2, bev_len, -1])

        bs, tgt_l, _ = x.shape
        _, src_l, _ = value.shape

        #print('x =', x)
        x_q = x + pos_embeds if pos_embeds is not None else x
        print('x_q =', x_q)
        x_q = paddle.concat([value[:bs], x_q], -1)

        x_v = self.value_proj(value)
        if attn_mask is not None:
            x_v = x_v.masked_fill(attn_mask[..., None], 0.0)
        x_v = x_v.reshape([bs * self.num_bev_queue, src_l, self.num_heads, self.head_dim])

        #print('x_v: ', x_v)
        #print('value: ',value)

        sampling_offsets = self.sampling_offsets(x_q)
        print('-------------- Paddle -------------------')
        print('-------------- Paddle -------------------')
        print('-------------- Paddle -------------------')
        print('sampling_offsets: ', sampling_offsets)
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

        #print('-------------- Paddle -------------------')
        #print('-------------- Paddle -------------------')
        #print('-------------- Paddle -------------------')
        #print('value: ', x_v.shape, x_v)
        #print('spatial_shapes: ', spatial_shapes)
        #print('level_start_index: ', level_start_index)
        #print('sampling locations: ', sampling_locations.shape, sampling_locations)
        #print('attention_weights: ', attn.shape, attn)


        out = multiscale_deformable_attention(x_v, spatial_shapes, sampling_locations, attn)

        # [bs*num_bev_queue, num_queries, embed_dim] -> [num_queries, embed_dim, bs*num_bev_queue]
        out = out.transpose([1, 2, 0])
        out = out.reshape([tgt_l, self.embed_dim, bs, self.num_bev_queue])
        out = out.mean(-1)
        out = out.transpose([2, 0, 1])  # [bs, num_queries, embed_dim]
        out = self.out_proj(out)
        return out, attn


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
        x = self.self_attn_norm(x)
        print('Encoder TSA: ', x.shape, x)

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
        print('Encoder SCA: ', x.shape, x)
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
    def get_reference_points_2d(h, w, bs=1): # W: Missing function or method docstring
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
        return output, intermediate
