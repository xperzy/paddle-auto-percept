# 从零开始实现 BEVFormer (1) - BEVFormer类

```python
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

```

### 类的初始化方法：

1. dcn_stages定义了ResNet的层（layer1-layer4）中，哪些层的conv会被转换成DCN。
2. return_layers是resnet提特征的时候返回哪些层的特征结果，这里是返回最后3的layer的结果。
3. FPN中的参数：
    1. start_level=0：表示从resnet返回的特征的第1个特征开始
    2. num_outs=4：表示一共返回的FPN的层数，如果不够就从Resnet最后一层的特征再增加额外的特征层。

### 在推理过程中：

1. 分为特征提取和检测head的推理两部分
2. 提取特征的时候需要注意：
    1. 输入通常是5维的： [B,N,C,H,W]，分别表示：batch大小，相机（视角）数量，图像通道数（一般是3，RGB三通道），图像高度，图像宽度。
    2. 然后在进入image backbone推理的时候，把维度变换为[B*N,C,H,W]，最后再变换回来