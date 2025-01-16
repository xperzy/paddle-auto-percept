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
BEVFormer in Paddle

A Paddle Implementation of BEVFormer as described in:
"BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"
Paper Link: https://arxiv.org/abs/2203.17270
"""
import numpy as np
from easydict import EasyDict
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.transforms import functional as VF
import position_embedding as pe
from encoder import BEVFormerEncoder, inverse_sigmoid
from decoder import BEVFormerDecoder


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

    def get_bboxes(self, pred_dicts, img_metas, rescale=False):
        """
        Arge:
            pred_dictsi['all_cls_scores']: [num_decoder_layers, bs, num_queries, class_out_channels]
            pred_dictsi['all_bbox_preds']: [num_decoder_layers, bs, num_queries, 9]
        """
        def denormalize_bboxes(norm_bboxes, pc_range):
            # rotation
            rot_sine = norm_bboxes[..., 6:7]  # NOTE: the : here is to keep the dim, [300, 10] -> [300, 1]
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
            #print('[denorm] cx, w, rot, vx: ', cx.shape, w.shape, rot.shape, vx.shape)

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
                cls_scores: Tensor of shape [num_queries, cls_out_channels];  cls_out_channels should include background
                bbox_preds: Tensor of shape [num_queries, 9]; (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)
            """
            cls_scores = cls_scores.sigmoid()
            scores, indices = cls_scores.reshape([-1]).topk(max_num)
            labels = indices % num_classes
            bbox_indices = indices // num_classes
            bbox_preds = bbox_preds[bbox_indices]
            #print('[inside decoder]: ', bbox_preds.shape)

            final_bbox_preds = denormalize_bboxes(bbox_preds, pc_range)  # [300, 10]
            final_scores = scores
            #print('[inside decoder]: final_bbox_preds: ', final_bbox_preds.shape)
            final_preds = labels

            if score_threshold is not None:
                thresh_mask = final_scores > score_threshold

            if post_center_range is not None:
                post_center_range = paddle.to_tensor(post_center_range)

                #mask = (final_bbox_preds[..., :3] >= post_center_range[:3])  # [300, 3]
                
                #print(final_bbox_preds.shape, final_bbox_preds)
                #print(mask.shape, mask)

                mask = (final_bbox_preds[..., :3] >= post_center_range[:3]).all(1)  # [300]
                mask = mask & (final_bbox_preds[..., :3] <= post_center_range[3:]).all(1)

                if score_threshold is not None:
                    mask = mask & thresh_mask

                #print('[insider decoder]: mask: ', mask)
                boxes3d = final_bbox_preds[mask]
                scores = final_scores[mask]
                labels = final_preds[mask]

                #print('[insider decoder]: boxes3d: ', boxes3d)
                #print('[insider decoder]: scores: ', scores)
                #print('[insider decoder]: labels: ', labels)

                pred_dict = {'bboxes': boxes3d,
                             'scores': scores,
                             'labels': labels}
                return pred_dict

        # use the last decoder layer output
        all_cls_scores = pred_dicts['all_cls_scores'][-1]
        all_bbox_preds = pred_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.shape[0]
        #print('batch_size: ', batch_size)
        #print('all_cls_scores[i]: ', all_cls_scores[0].shape, all_bbox_preds[0].shape)
        pred_dicts = []
        for i in range(batch_size):
            pred_dicts.append(decode(all_cls_scores[i], all_bbox_preds[i], self.pc_range))

        num_samples = len(pred_dicts)
        #### TODO: Check correctness
        res_list = []
        for i in range(num_samples):
            preds = pred_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            #print(f'\t------------- sample: {i} ----------------')
            #print('bboxes: ', bboxes)
            # bbox_type_3d = 'LiDAR'
            #bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
            scores = preds['scores']
            labels = preds['labels']
            #print('scores: ', scores)
            #print('labels: ', labels)
            res_list.append([bboxes, scores, labels])
        return res_list

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


class BEVFormerTransformer(nn.Layer):
    """BEVFormer transformer for object detection"""
    def __init__(self,
                 pc_range,
                 num_points_in_pillar,
                 num_feature_levels,
                 num_cams,
                 num_encoder_layers,
                 num_decoder_layers,
                 embed_dim,
                 num_heads,
                 self_attn_dropout,
                 cross_attn_dropout,
                 ffn_dim,
                 ffn_dropout,
                 num_points=8,
                 num_bev_queue=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.level_embeds = paddle.create_parameter(
            shape=[num_feature_levels, embed_dim], dtype='float32')
        self.cams_embeds = paddle.create_parameter(
            shape=[num_cams, embed_dim], dtype='float32')
        self.reference_points = paddle.nn.Linear(embed_dim, 3)
        self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.ReLU(),
                nn.LayerNorm(embed_dim))

        self.encoder = BEVFormerEncoder(embed_dim=embed_dim,
                                        ffn_dim=ffn_dim,
                                        num_heads=num_heads,
                                        num_layers=num_encoder_layers,
                                        num_levels=num_feature_levels,
                                        num_points=num_points,
                                        num_points_in_pillar=num_points_in_pillar,
                                        num_bev_queue=num_bev_queue,
                                        self_attn_dropout=self_attn_dropout,
                                        cross_attn_dropout=cross_attn_dropout,
                                        ffn_dropout=ffn_dropout,
                                        num_cams=num_cams,
                                        pc_range=pc_range)

        self.decoder = BEVFormerDecoder(embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        num_layers=num_decoder_layers,
                                        num_levels=1,  # BEV query only has 1 level
                                        num_points=4,  # cross attn on BEV query samples 4 points
                                        ffn_dim=ffn_dim,
                                        self_attn_dropout=self_attn_dropout,
                                        cross_attn_dropout=cross_attn_dropout,
                                        ffn_dropout=ffn_dropout)

    def forward(self,
                multi_level_feats,
                object_query_embeds,
                bev_queries,
                bev_h,
                bev_w,
                grid_length,
                bev_pos,
                bbox_embed,
                img_metas,
                prev_bev=None):
        """
        Args:
            multi_level_feats: list/tuple of Tensor, shape [bs, num_cams, embed_dim, h, w]
            query_embeds: object query embeds for decoder, [num_queries, embed_dim*2]
            bev_queries: [bev_h * bev_w, embed_dim]
            bev_pos: position embeddings for bev, with shape [bs, embed_dim, bev_h, bev_w]
        """
        bs = multi_level_feats[0].shape[0]
        bev_queries = bev_queries.unsqueeze(0)  # [bev_h*bev_w, c] -> [1, bev_h*bev_w, c]
        bev_queries = bev_queries.expand([bs, bev_h*bev_w, -1])  # [bs, bev_h*bev_w, c]
        bev_pos = bev_pos.flatten(2)  # [bs, c, bev_h, bev_w] -> [bs, c, bev_h*bev*w]
        bev_pos = bev_pos.transpose([0, 2, 1])  # [bs, bev_h*bev*w, c]

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([img_meta['can_bus'][0] for img_meta in img_metas])
        delta_y = np.array([img_meta['can_bus'][1] for img_meta in img_metas])
        ego_angle = np.array([img_meta['can_bus'][-2] / np.pi * 180 for img_meta in img_metas])
        grid_len_x, grid_len_y = grid_length[0], grid_length[1]
        translation_len = np.sqrt(delta_x ** 2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle

        shift_y = translation_len * np.cos(bev_angle / 180 * np.pi) / grid_len_y / bev_h
        shift_x = translation_len * np.sin(bev_angle / 180 * np.pi) / grid_len_x / bev_w
        shift = paddle.to_tensor([shift_x, shift_y])
        shift = shift.transpose([1, 0])

        if prev_bev is not None:
            for i in range(bs):
                rotation_angle = img_metas[i]['can_bus'][-1]
                # prev_bev: [bs, c, bev_h*bev_w]
                tmp_prev_bev = prev_bev[i]  # [c, bev_h*bev_w]
                tmp_prev_bev = tmp_prev_bev.reshape([-1, bev_h, bev_w])
                tmp_prev_bev = VF.rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                tmp_prev_bev = tmp_prev_bev.reshape([-1, bev_h*bev_w])
                tmp_prev_bev = tmp_prev_bev.unsqueeze(0)  # [c, bev_h*bev_h] -> [1, c, bev_h*bev_w]
                prev_bev[i] = tmp_prev_bev[0, :]

        # add can bus signals
        can_bus = paddle.to_tensor([img_meta['can_bus'] for img_meta in img_metas])
        can_bus = self.can_bus_mlp(can_bus)
        can_bus = can_bus[:, None, :]
        bev_queries = bev_queries + can_bus  # [bs, h*w, c]
        #print('bev_queries: ', bev_queries.shape)

        feat_flatten = []
        spatial_shapes = []
        for level, feat in enumerate(multi_level_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3)  # [bs, num_cam, c, h*w]
            feat = feat.transpose([1, 0, 3, 2])  # [num_cam, bs, h*w, c]
            feat = feat + self.cams_embeds[:, None, None, :]
            feat = feat + self.level_embeds[None, None, level:level+1, :]

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = paddle.concat(feat_flatten, 2)  # [num_cam, bs, sum(h*w), embed_dim]
        feat_flatten = feat_flatten.transpose([0, 2, 1, 3])  # [num_cam, sum(h*w), bs, embed_dim]
        spatial_shapes = paddle.to_tensor(spatial_shapes)
        level_start_index = paddle.concat([paddle.zeros(1, dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]])
        #print('feat_flatten: ', feat_flatten.shape)

        bev_embed, encoder_intermediate = self.encoder(bev_query=bev_queries,
                                                       value=feat_flatten,
                                                       bev_h=bev_h,
                                                       bev_w=bev_w,
                                                       bev_pos=bev_pos,
                                                       spatial_shapes=spatial_shapes,
                                                       level_start_index=level_start_index,
                                                       prev_bev=prev_bev,
                                                       img_metas=img_metas,
                                                       shift=shift)
        #print('bev_embed after encoder: ', bev_embed.shape)
        bs = multi_level_feats[0].shape[0]
        object_query_pos, object_query = paddle.split(object_query_embeds, 2, axis=1)
        object_query_pos = object_query_pos.unsqueeze(0).expand([bs, -1, -1])
        object_query = object_query.unsqueeze(0).expand([bs, -1, -1])
        reference_points = self.reference_points(object_query_pos).sigmoid()

        decoder_output = self.decoder(input_embeds=object_query,
                                      value=bev_embed,
                                      attn_mask=None,
                                      pos_embeds=object_query_pos,
                                      ref_pts=reference_points,
                                      img_metas=img_metas,
                                      bbox_embed=bbox_embed,
                                      spatial_shapes=paddle.to_tensor([[bev_h, bev_w]], dtype='int64'),
                                      level_start_index=paddle.zeros([1], dtype='int64'))
        output = (bev_embed, ) + decoder_output
        # output: bev_embed, decoder_states, init_ref_pts, intermediate_ref_pts, all_self_attn, all_cross_attn
        return output
