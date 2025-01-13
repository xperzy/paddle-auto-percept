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

from collections import OrderedDict
from easydict import EasyDict
import paddle
import paddle.nn as nn
import resnet
import position_embedding as pe
from encoder_decoder import DetrEncoder, DetrDecoder


class Resnet50Feature(nn.LayerDict):
    """Extract Each layer's output from Resnet"""
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

    def forward(self, x, mask=None): # W: Number of parameters was 3 in 'Layer.forward' and is now 3 in overriding 'Resnet50Feature.forward' method
        features = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                features[out_name] = x
        out = []
        for feature_name, feature_map in features.items():
            # downsample mask to match shape of corresponding feature_map
            if mask is not None:
                mask = nn.functional.interpolate(mask[None].astype('float32'),
                                                 size=feature_map.shape[-2:])[0]
            out.append((feature_name, feature_map, mask))
        return out


class Detr(nn.Layer):
    """DETR for object detection"""
    def __init__(self,
                 embed_dim=256,
                 ffn_dim=2048,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_queries=100,
                 num_classes=91):
        super().__init__()
        # image feature layer names from resnet
        resnet_return_layers={'layer2': '0', 'layer3': '1', 'layer4': '2'}
        self.num_out_features = len(resnet_return_layers.items())
        # input channels (resnet out channels) for input_proj layers
        in_channels = [512, 1024, 2048]
        # Detr only uses last feature map in resnet
        self.img_feat_idx = [2]

        # backbone outputs multiple features
        self.backbone = Resnet50Feature(
                resnet.resnet50(num_classes=0, with_pool=False), return_layers=resnet_return_layers)
        # image feature proj
        self.input_proj = nn.Conv2D(in_channels[self.img_feat_idx[-1]], embed_dim, kernel_size=1)

        # position encoding layer
        self.pe_layer = pe.build_position_encoding(embed_dim, 'sine')

        # encoder
        self.encoder = DetrEncoder(embed_dim,
                                   ffn_dim,
                                   num_heads,
                                   num_encoder_layers,
                                   dropout_rate=0.1)

        # decoder
        self.pos_embeds = nn.Embedding(num_queries, embed_dim)
        self.decoder = DetrDecoder(embed_dim,
                                   ffn_dim,
                                   num_heads,
                                   num_decoder_layers,
                                   dropout_rate=0.1)

        # classification head
        self.class_embed = nn.Linear(embed_dim, num_classes)

        # bbox regression head
        self.bbox_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4))

    def forward(self, pixel_values, pixel_mask):
        # extract image features
        if pixel_mask is None:  # no padding
            bs, _, h, w = pixel_values.shape
            pixel_mask = paddle.ones([bs, h, w])
        outputs = self.backbone(pixel_values, pixel_mask)
        img_feats = [out[1] for out in outputs]
        masks = [out[2] for out in outputs]
        # DETR only uses last feature map
        img_feat = img_feats[-1]
        mask = masks[-1]

        # image feature proj
        img_feat_proj = self.input_proj(img_feat)

        # position encoding
        tensor_list = EasyDict()
        tensor_list.tensors = None
        bs, _, h, w = img_feat_proj.shape
        tensor_list.mask = paddle.zeros((1, h, w))
        pos_embeds = self.pe_layer(tensor_list)
        pos_embeds = pos_embeds.flatten(2).transpose([0, 2, 1])

        # encoder
        img_feat_proj_flatten = img_feat_proj.flatten(2).transpose([0, 2, 1])
        mask_flatten = mask.flatten(1)  # [bs, h*w]
        encoder_out = self.encoder(input_embeds=img_feat_proj_flatten,
                                   attn_mask=mask_flatten,
                                   pos_embeds=pos_embeds)
        encoder_x = encoder_out[0]
        #encoder_all_states = encoder_out[1]
        encoder_attn_w = encoder_out[2]

        # decoder
        query_pos_embeds = self.pos_embeds.weight
        # [num_queries, embed_dim] -> [bs, num_queries, embed_dim]
        query_pos_embeds =  query_pos_embeds.unsqueeze(0).expand([bs, -1, -1])
        query = paddle.zeros(query_pos_embeds.shape)
        decoder_out = self.decoder(input_embeds=query,
                                   attn_mask=None,
                                   pos_embeds=query_pos_embeds,
                                   encoder_x=encoder_x,
                                   encoder_pos_embeds=pos_embeds,
                                   encoder_attn_mask=mask_flatten)
        decoder_x = decoder_out[0]
        #decoder_all_states = decoder_out[1]
        decoder_self_attn_w = decoder_out[2]
        decoder_cross_attn_w = decoder_out[3]

        # classification head
        logits = self.class_embed(decoder_x)
        # bbox regression head
        pred_boxes = self.bbox_embed(decoder_x).sigmoid()

        outputs = {'logits': logits,
                   'pred_boxes': pred_boxes,
                   'encoder_attn_w': encoder_attn_w,
                   'decoder_self_attn_w': decoder_self_attn_w,
                   'decoder_cross_attn_w': decoder_cross_attn_w}
        outputs = EasyDict(outputs)
        return outputs
