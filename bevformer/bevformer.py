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
from collections import OrderedDict
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import resnet
from bevformer_transformer import BEVFormerHead


class DCNPack(nn.Layer):
    """ This is DCNv2 """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias_attr=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.bias_attr=bias_attr

        self.conv_offset = paddle.nn.Conv2D(
           self.in_channels,
           self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
           kernel_size=self.kernel_size,
           stride=self.stride,
           padding=self.padding,
           dilation=self.dilation,
           bias_attr=True)

        self.weight = paddle.create_parameter(shape=[out_channels, in_channels // self.groups,
                         *self.kernel_size], dtype='float32')

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = paddle.chunk(out, 3, axis=1)
        offset = paddle.concat((o1, o2), axis=1)
        mask = paddle.nn.functional.sigmoid(mask)
        out = paddle.vision.ops.deform_conv2d(x,
                                              offset,
                                              self.weight,
                                              mask=mask,
                                              bias=None,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation,
                                              deformable_groups=self.deform_groups,
                                              groups=self.groups)
        return out


class ResNetWithDCN(nn.Layer): # W: Missing class docstring
    
    """resnet with dcn layers"""
    def __init__(self, model, dcn_stages):
        super().__init__()
        self.model = model
        self._replace_layers(self.model, dcn_stages)

    def _replace_layers(self, layer, dcn_stages):
        for name, sublayer in layer.named_children():
            if name in dcn_stages:
                for sub_name, sub_sublayer in sublayer.named_children():
                    for ss_name, ss_layer in sub_sublayer.named_children():
                        if ss_name.endswith('conv2') and isinstance(ss_layer, paddle.nn.Conv2D):
                            in_channels = ss_layer._in_channels
                            out_channels = ss_layer._out_channels
                            kernel_size = ss_layer._kernel_size
                            stride = ss_layer._stride
                            padding = ss_layer._padding
                            dilation = ss_layer._dilation
                            groups = ss_layer._groups
                            bias_attr = ss_layer.bias is not None

                            deformable_conv = DCNPack(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)
                            #deformable_conv.weight.set_value(sub_sublayer.weight)
                            #if bias_attr:
                            #    deformable_conv.bias.set_value(sub_sublayer.bias)
                            setattr(sub_sublayer, ss_name, deformable_conv)

    def forward(self, x):
        return self.model(x)


class ResNet101Feature(nn.LayerDict):
    def __init__(self, resnet_model, return_layers):
        orig_return_layers = return_layers.copy()
        layers = OrderedDict()
        for name, module in resnet_model.named_children():
            layers[name] = module
            if name in return_layers:
                return_layers.remove(name)
            if not return_layers:
                break
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = []
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out.append(x)
        return out


class FPN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral

        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
        self.start_level = start_level
        self.end_level = end_level

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Sequential(
                    nn.Conv2D(in_channels[i], out_channels,1, 1, 0),
                    #nn.BatchNorm2D(out_channels, out_channels),
                    #nn.ReLU())
                    )
            fpn_conv = nn.Sequential(
                    nn.Conv2D(out_channels, out_channels, 3, 1, 1),
                    #nn.BatchNorm2D(out_channels, out_channels),
                    #nn.ReLU())
                    )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = self.num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs is True and extra_levels >=1:
            for i in range(extra_levels):
                extra_fpn_conv = nn.Sequential(
                    nn.Conv2D(out_channels, out_channels, 3, 2, 1),
                    #nn.BatchNorm2D(out_channels, out_channels),
                    #nn.ReLU())
                    )
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, x):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(x[i + self.start_level]))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels -1 , 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] += F.interpolate(laterals[i], size=prev_shape)
        outs = []
        for i in range(used_backbone_levels):
            outs.append(self.fpn_convs[i](laterals[i]))

        if self.num_outs > len(outs):
            # extra levels
            if not self.add_extra_convs:  #use max pool
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:  # add conv layers
                extra_source = outs[-1]
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BEVFormer(nn.Layer):
    """BEVFormer for object detection"""
    def __init__(self):
        super().__init__()
        # create backbone
        dcn_stages = ['layer3', 'layer4']
        # layer names of resnet outputs
        return_layers = ['layer2', 'layer3', 'layer4']
        # the style is set to caffe (same as official code)
        self.img_backbone = resnet.resnet101(num_classes=0, with_pool=False, style='caffe')
        self.img_backbone = ResNetWithDCN(self.img_backbone, dcn_stages)
        self.img_backbone = ResNet101Feature(self.img_backbone.model, return_layers)
        # create fpn
        self.img_neck = FPN(in_channels=[512, 1024, 2048],
                            out_channels=256,
                            start_level=0,
                            add_extra_convs=True,
                            num_outs=4,
                            relu_before_extra_convs=True)
        # create head
        self.pts_bbox_head = BEVFormerHead(num_classes=10,
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
                                           num_points=8)

    def extract_feat(self, img):
        """Extract faetures from resnet and fpn"""
        ## update real input shape of each single img
        #input_shape = img.shape[-2:] # img shape: [b, n, c, h, w]
        #img_metas[0]['img_shape'] = [input_shape for i in range(img.shape[1])]
        if img.dim() == 5:
            b, n, c, h, w = img.shape
            img = img.reshape([b * n, c, h, w])
        # inference backbone
        img_feats = self.img_backbone(img)
        # inference neck
        img_feats = self.img_neck(img_feats)
        # reshape feature for output
        img_feats_reshaped = []
        for out_feat in img_feats:
            bn, c, h, w = out_feat.shape
            img_feats_reshaped.append(out_feat.reshape([b, int(bn / b), c, h, w]))
        return img_feats_reshaped

    def forward(self, img, img_metas):
        img_feats = self.extract_feat(img)
        outputs = self.pts_bbox_head(img_feats, img_metas, prev_bev=None)
        return outputs
