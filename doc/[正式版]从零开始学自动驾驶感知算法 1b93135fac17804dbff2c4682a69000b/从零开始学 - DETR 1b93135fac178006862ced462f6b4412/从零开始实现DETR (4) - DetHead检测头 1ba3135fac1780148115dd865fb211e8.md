# 从零开始实现DETR (4) - DetHead检测头

### DETR的预测头分为两个部分：

- 分类预测分支：
    - 就是一个Linear层实现，从embed_dim到num_classes的变换。
- 框回归预测分支：
    - 一个类似MLP的多层网络
    - 输出归一化后的框信息：（x, y, w, h）；x,y表示中心点位置，h，w表示框的长宽。

```python
    # classification head
    self.class_embed = nn.Linear(embed_dim, num_classes)

    # bbox regression head
    self.bbox_embed = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, 4))
```

在模型forward方法中，decoder的输入将作为预测头的输入，分别计算得到每个类别和框信息，一共是num_queries个预测，所以会返回num_queries个“候选框”：

```python
    #classification head
    logits = self.class_embed(decoder_x)
    # bbox regression head
    pred_boxes = self.bbox_embed(decoder_x).sigmoid()
```

最后再经过后处理过滤，最终保留置信度较高的框作为图像预测的结果