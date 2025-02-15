#  Copyright (c) 2024 PaddleAutoPercept Authors. All Rights Reserved.
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

""" ResNet, add new features that allows changing dilation, and change norm layers,
Mostly refered:
https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/vision/models/resnet.py
"""

from functools import partial
import paddle
from paddle import  nn
from paddle.utils.download import get_weights_path_from_url

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

model_urls = {
    'resnet18': ('https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
                 'cf548f46534aa3560945be4b95cd11c4'),
    'resnet34': ('https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams',
                 '8d2275cf8706028345f78ac0e1d31969'),
    'resnet50': ('https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
                 'ca6f485ee1ab0492d38f323885b0ad80'),
    'resnet101': ('https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
                  '02f35f034ca3858e1e54d4036443c92d'),
    'resnet152': ('https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams',
                  '7ad16a2f1e7333859ff986138630fd7a'),
}


def init_weights(lr):
    weight_attr = paddle.ParamAttr(learning_rate=lr)
    bias_attr = paddle.ParamAttr(learning_rate=lr)
    return weight_attr, bias_attr


class BasicBlock(nn.Layer):
    expansion = 1
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 backbone_lr=1.0):
        super().__init__()
        if dilation > 1:
            raise ValueError('Basic block does not support dilation')
        if norm_layer is None:
            w_attr_1, b_attr_1 = init_weights(backbone_lr)
            norm_layer = partial(nn.BatchNorm2D, weight_attr=w_attr_1, bias_attr=b_attr_1)
        w_attr_2, b_attr_2 = init_weights(backbone_lr)
        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, weight_attr=w_attr_2, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        w_attr_3, b_attr_3 = init_weights(backbone_lr)
        self.conv2 = nn.Conv2D(
            planes, planes, 3, padding=1, weight_attr=w_attr_3, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Layer):
    expansion = 4
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 backbone_lr=1.0):
        super().__init__()
        if norm_layer is None:
            w_attr_1, b_attr_1 = init_weights(backbone_lr)
            norm_layer = partial(nn.BatchNorm2D, weight_attr=w_attr_1, bias_attr=b_attr_1)
        width = int(planes * (base_width / 64.)) * groups
        w_attr_2, b_attr_2 = init_weights(backbone_lr)
        self.conv1 = nn.Conv2D(inplanes, width, 1, weight_attr=w_attr_2, bias_attr=False)
        self.bn1 = norm_layer(width)

        w_attr_3, b_attr_3 = init_weights(backbone_lr)
        self.conv2 = nn.Conv2D(width,
                               width,
                               3,
                               padding=dilation,
                               stride=stride,
                               groups=groups,
                               dilation=dilation,
                               weight_attr=w_attr_3,
                               bias_attr=False)
        self.bn2 = norm_layer(width)
        w_attr_4, b_attr_4 = init_weights(backbone_lr)
        self.conv3 = nn.Conv2D(width,
                               planes * self.expansion,
                               1,
                               weight_attr=w_attr_4,
                               bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(self,
                 block,
                 depth,
                 num_classes=1000,
                 with_pool=True,
                 norm_layer=None,
                 replace_stride_with_dilation=None,
                 dilation=1,
                 backbone_lr=1.0):
        super().__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation shoule be None or 3-element tuple')

        layers = layer_cfg[depth]
        self.num_classes = num_classes
        self.with_pool = with_pool

        if norm_layer is None:
            w_attr_1, b_attr_1 = init_weights(backbone_lr)
            norm_layer = partial(nn.BatchNorm2D, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self._norm_layer = norm_layer


        self.inplanes = 64
        self.dilation = dilation

        w_attr_2, b_attr_2 = init_weights(backbone_lr)
        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            weight_attr=w_attr_2,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], backbone_lr=backbone_lr)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], backbone_lr=backbone_lr)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], backbone_lr=backbone_lr)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], backbone_lr=backbone_lr)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1,1))
        if num_classes > 0:
            w_attr_3, b_attr_3 = init_weights(backbone_lr)
            self.fc = nn.Linear(512 * block.expansion, num_classes, weight_attr=w_attr_3, bias_attr=b_attr_3)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, backbone_lr=1.0):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride =1
        if stride !=1 or self.inplanes != planes * block.expansion:
            w_attr_1, b_attr_1 = init_weights(backbone_lr)
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes*block.expansion,
                          1,
                          stride=stride,
                          weight_attr=w_attr_1,
                          bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 64,
                  previous_dilation, norm_layer, backbone_lr=backbone_lr))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, backbone_lr=backbone_lr))

        return nn.Sequential(*layers) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)
        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)

        return x

def _resnet(arch, Block, depth, pretrained, backbone_lr=1.0, **kwargs):
    model = ResNet(Block, depth, backbone_lr=1.0, **kwargs)
    if pretrained:
        assert arch in model_urls, f"{arch} model do not have a pretrained model now"
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])
        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def resnet18(pretrained=False, backbone_lr=1.0, **kwargs):
    return _resnet('resnet18', BasicBlock, 18, pretrained, backbone_lr, **kwargs)

def resnet34(pretrained=False, backbone_lr=1.0, **kwargs):
    return _resnet('resnet34', BasicBlock, 34, pretrained, backbone_lr, **kwargs)

def resnet50(pretrained=False, backbone_lr=1.0, **kwargs):
    return _resnet('resnet50', BottleneckBlock, 50, pretrained, backbone_lr, **kwargs)
        
def resnet101(pretrained=False, backbone_lr=1.0, **kwargs):
    return _resnet('resnet101', BottleneckBlock, 101, pretrained, backbone_lr, **kwargs)

def resnet152(pretrained=False, bakcbone_lr=1.0, **kwargs):
    return _resnet('resnet152', BottleneckBlock, 152, pretrained, backbone_lr, **kwargs)
