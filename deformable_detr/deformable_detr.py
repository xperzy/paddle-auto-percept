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
Deformable DETR in Paddle
    
A Paddle Implementation of Deformable DETR as described in:
"Deformable DETR: Deformable Transformers for End-to-End Object Detection"
Paper Link: https://arxiv.org/abs/2010.04159
"""
from collections import OrderedDict
import paddle
import paddle.nn as nn
from easydict import EasyDict
import resnet
import position_embedding as pe
from encoder_decoder import DeformableDetrEncoder, DeformableDetrDecoder

def get_valid_ratio(mask, dtype='float32'):
    """get valid ratios (valid_w/w, valid_h/h) of each feature map"""
    _, h, w = mask.shape
    v_h = paddle.sum(mask[:, :, 0], 1)
    v_w = paddle.sum(mask[:, 0, :], 1)
    v_r_h = paddle.cast(v_h, dtype) / h
    v_r_w = paddle.cast(v_w, dtype) / w
    v_r = paddle.stack([v_r_w, v_r_h], -1)
    return v_r


def inverse_sigmoid(x, eps=1e-5):
    """ inverse op of sigmoid"""
    # sigmoid: 1 / ( 1+ e**(-x))
    # inverse sigmoid: log(x / (1-x))
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


class Resnet50Feature(nn.LayerDict):
    """Extract multiple features from Resnet50"""
    def __init__(self, resnet_model, return_layers):
        orig_return_layers = return_layers
        return_layers = dict(return_layers.items())
        layers = OrderedDict()
        for name, module in resnet_model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x, mask):
        features = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                features[out_name] = x

        out = []
        for feature_name, feature_map in features.items():
            # downsample mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(mask[None].astype('float32'),
                                             size=feature_map.shape[-2:])[0]
            out.append((feature_name, feature_map, mask))
        return out


class DeformableDetr(nn.Layer):
    """Deformable Detr for object detection"""
    def __init__(self,
                 embed_dim=256,
                 ffn_dim=1024,
                 num_heads=8,
                 num_points=4,
                 num_feature_levels=4,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_queries=300,
                 num_classes=91,
                 bbox_refine=False):
        super().__init__()
        # output layer names from image backbone
        resnet_return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
        # num of output features from image backbone
        self.num_out_features = len(resnet_return_layers.items())
        # image backbone
        self.backbone = Resnet50Feature(resnet.resnet50(num_classes=0, with_pool=False),
                                        return_layers=resnet_return_layers)

        # input channels for input proj
        in_channels = [512, 1024, 2048, 2048]
        assert len(in_channels) == num_feature_levels
        # idx of img feature used in input proj
        # [layer2, layer3, layer4, layer4] -> [feat_1, feat_2, feat_3, feat_4]
        self.img_feat_idx = [0, 1, 2, 2]
        # input projections for multiscale features
        input_proj_list = []
        for level in range(num_feature_levels):
            in_channel = in_channels[self.img_feat_idx[level]]
            if level < self.num_out_features:
                input_proj = nn.Sequential(
                    nn.Conv2D(in_channel, embed_dim, kernel_size=1),
                    nn.GroupNorm(32, embed_dim))
            else:
                input_proj = nn.Sequential(
                    nn.Conv2D(in_channel, embed_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, embed_dim))
            input_proj_list.append(input_proj)
        self.input_proj = nn.LayerList(input_proj_list)

        # position encoding layer
        self.pe_layer = pe.build_position_encoding(embed_dim, 'sine')

        # level embed
        self.level_embed = paddle.create_parameter(
            shape=[num_feature_levels, embed_dim], dtype='float32')

        # encoder
        self.encoder = DeformableDetrEncoder(embed_dim,
                                             ffn_dim,
                                             num_heads,
                                             num_points,
                                             num_feature_levels,
                                             num_encoder_layers)
        # decoder
        self.decoder = DeformableDetrDecoder(embed_dim,
                                             ffn_dim,
                                             num_heads,
                                             num_points,
                                             num_feature_levels,
                                             num_decoder_layers)
        self.num_queries = num_queries
        self.reference_points = nn.Linear(embed_dim, 2)
        self.pos_embeds = nn.Embedding(num_queries, embed_dim * 2)

        # classification head
        self.bbox_refine = bbox_refine
        class_embed_list = []
        if self.bbox_refine is False:
            class_embed = nn.Linear(embed_dim, num_classes)
            class_embed_list = [class_embed for _ in range(num_decoder_layers)]
        else:
            class_embed_list = [nn.Linear(
                embed_dim, num_classes) for _ in range(num_decoder_layers)]
        self.class_embed = nn.LayerList(class_embed_list)

        # bbox regression head
        bbox_embed_list = []
        if self.bbox_refine is False:
            bbox_embed = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 4))
            bbox_embed_list = [bbox_embed for idx in range(num_decoder_layers)]
        else:
            for idx in range(num_decoder_layers):
                bbox_embed_list.append(nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, 4)))
        self.bbox_embed = nn.LayerList(bbox_embed_list)

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            bs, _, h, w = pixel_values.shape
            pixel_mask = paddle.ones([bs, h, w])
        # extract image features from backbone, returns [(feat_name, feat, mask)...]
        outputs = self.backbone(pixel_values, pixel_mask)
        img_feats = [out[1] for out in outputs]
        masks = [out[2] for out in outputs]

        # input proj
        img_feats_proj = []
        for level, input_proj in enumerate(self.input_proj):
            out = input_proj(img_feats[self.img_feat_idx[level]])
            img_feats_proj.append(out)
            # add mask for extra feature
            if level >= self.num_out_features:
                masks.append(nn.functional.interpolate(pixel_mask[None].astype('float32'),
                                                       size=out.shape[-2:])[0])

        # position encoding for each level
        pos_embeds = []
        for img_feat in img_feats_proj:
            tensor_list = EasyDict()
            tensor_list.tensors = None
            _, _, h, w = img_feat.shape
            tensor_list.mask = paddle.zeros([1, h, w])
            pos_embeds.append(self.pe_layer(tensor_list))

        # encoder
        spatial_shapes = []
        source_flatten = []
        mask_flatten = []
        level_pos_embed_flatten = []
        for level, (source, mask, pos_embed) in enumerate(zip(img_feats_proj, masks, pos_embeds)):
            bs, _, h, w = source.shape
            spatial_shapes.append((h, w))
            # [bs, c, h, w] -> [bs, h*w, c]
            source = source.flatten(2).transpose([0, 2, 1])
            source_flatten.append(source)
            # [bs, h, w] -> [bs, h*w]
            mask = mask.flatten(1)
            mask_flatten.append(mask)
            # [bs, c, h, w] -> [bs, h*w, c]
            pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])
            # [bs, h*w, c]  + [1, 1, c]
            level_pos_embed = pos_embed + self.level_embed[level].reshape([1, 1, -1])
            level_pos_embed_flatten.append(level_pos_embed)
        # convert list to tensor
        source_flatten = paddle.concat(source_flatten, 1)
        print('source_flatten.shape - ', source_flatten.shape)
        mask_flatten = paddle.concat(mask_flatten, 1)
        level_pos_embeds = paddle.concat(level_pos_embed_flatten, 1)

        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype='int64')
        level_start_index = paddle.concat([
            paddle.zeros(1, dtype='int64'),
            spatial_shapes.prod(1).cumsum(0)[:-1]])
        valid_ratios = paddle.stack([get_valid_ratio(m) for m in masks], 1)

        enc_out = self.encoder(input_embeds=source_flatten,
                               attn_mask=mask_flatten,
                               pos_embeds=level_pos_embeds,
                               spatial_shapes=spatial_shapes,
                               level_start_index=level_start_index,
                               valid_ratios=valid_ratios)
        encoder_x = enc_out[0]
        encoder_attn_w = enc_out[2]
        

        # decoder
        query_embeds = self.pos_embeds.weight
        query_pos, query = paddle.split(query_embeds, 2, axis=1)
        query_pos = query_pos.unsqueeze(0).expand([bs, -1, -1])
        query = query.unsqueeze(0).expand([bs, -1, -1])
        ref_pts = self.reference_points(query_pos).sigmoid()
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
        inter_ref_pts = dec_out[1]
        decoder_states = dec_out[2]
        decoder_self_attn_w = dec_out[3]
        decoder_cross_attn_w = dec_out[4]

        # class and box prediction
        output_coords = []
        output_classes = []
        for level in range(len(decoder_states) - 1):
            ref = inter_ref_pts[level]
            ref = inverse_sigmoid(ref)
            output_class = self.class_embed[level](decoder_states[level+1])
            print(output_class)
            delta_box = self.bbox_embed[level](decoder_states[level+1])
            delta_box[..., :2] += ref
            output_coord_logits = delta_box
            output_coord = output_coord_logits.sigmoid()

            output_classes.append(output_class)
            output_coords.append(output_coord)

        output_class = paddle.stack(output_classes)
        output_coords = paddle.stack(output_coords)

        logits = output_class[-1]
        pred_boxes = output_coord[-1]
        pred_boxes = pred_boxes.unsqueeze(0)

        outputs = {'logits': logits,
                   'pred_boxes': pred_boxes,
                   'encoder_attn_w': encoder_attn_w,
                   'decoder_self_attn_w': decoder_self_attn_w,
                   'decoder_cross_attn_w': decoder_cross_attn_w}
        outputs = EasyDict(outputs)
        return outputs
