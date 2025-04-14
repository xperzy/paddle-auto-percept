# 从零开始实现 Deformable DETR (6) - 预测头与目标输出

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20Deformable%20DETR%20(6)%20-%20%E9%A2%84%E6%B5%8B%E5%A4%B4%E4%B8%8E%E7%9B%AE%E6%A0%87%E8%BE%93%E5%87%BA%201bb3135fac178003a5ebdbffe6321d42/image.png)

### 检测头：

检测头主要是将Decoder的输出，变为预测框和对应的分类，在训练时可以和GT进行Loss计算并完成训练过程，在测试（推理）时，可以对输入图像完成物体框的回归和类别的识别。具体来说，检测头分为2个分支：

- 分类分支：由一个简单的Linear层组成，对于N个DecoderLayer，会对应N个分类层
- 框回归分支：由一组Linear和激活函数组成的MLP构成，同样N个DecoderLayer会有N个框回归MLP层

输出的形式：

- 分类：shape是[bs, num_queries, num_classes]的Tensor
- 框回归：shape是[bs, num_queries, 4]的Tensor，最后一维度表示 [delta_x, delta_y, w, h], 注意这里的delta_x, delta_y是相对于参考点的偏移量

正因为是相对于参考点坐标的偏移量，所以在计算最终box位置的时候，需要每个Decoder的参考点位置，上一节我们讲过，Decoder每一层进行计算的时候，DeformableDETR引入了动态更新参考点的方式，所以在代码实现的时候，需要提前保存每一层Decoder的Reference Points位置。（不过实际在推理的时候，DeformableDETR也只是使用了最后一层的输出结果）

**检测头部分的定义：**

```python
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

```

其中，如果self.box_refine是False，则表示bbox预测不更新每层的Reference Points位置，那分类头和框回归头，多层都使用同一个头。所以，会有代码中的 if else 部分：

- `[class_embed for _ in range(num_decoder_layer)]` : 因为class_embed是预先定义好的Linear层，这里是将其copy了多个，但指向的还是同一个Layer，并非是复制成新的实例。
- `[nn.Linear(embed_dim, num_classes) for _ in range(num_decoder_layer)]`，这里每次都会调用nn.Linear的初始化方法创建新的对象，所以这里的list中的每个layer都是不同的实例

**Forward方法:**

```python
		    # class and box prediction
        output_coords = []
        output_classes = []
        for level in range(len(decoder_states) - 1):
            ref = inter_ref_pts[level]
            ref = inverse_sigmoid(ref)
            output_class = self.class_embed[level](decoder_states[level+1])
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
```

- inverse_sigmoid：是将sigmoid之后的坐标位置变回sigmoid之前的绝对位置，因为bbox计算的时候是绝对偏移量，将绝对位置与其相加之后，再sigmoid，可以变为0,1的相对位置。
- decoder_state：中的第一个位置存的是第一层decoderlayer的输入，然后才是每层decoderlayer的输出，所以会取[level+1]的位置开始

### Inverse Sigmoid：

- **Sigmoid：**

$$
\mathrm{Sigmoid}(x) = \dfrac{1}{1+e^{-x}}
$$

- **Sigmoid的逆运算推导：**
    - Sigmoid的计算： $y = \dfrac{1}{1+e^{-x}}$
    - 展开：
        - ${1+e^{-x}} = \dfrac{1}{y}$
        - $e^{-x} = \dfrac{1-y}{y}$
        - ${e^x} = \dfrac{y}{1-y}$
    - 得到：${x} = \mathrm{log}(\dfrac{y}{1-y})$
- **所以Inverse Sigmoid：**

$$
\mathrm{InverseSigmoid}(x) = \mathrm{log}(\dfrac{x}{1-x})
$$

- 在实现的时候：
    - x需要在0到1之间，所以需要clip操作： `x  = x.clip(min=0, max=1)`
    - x不能为0，可以使用clip操作：`x1 = x.clip(min=1e-5)`
    - 1 - x 不能为0，仍然可以使用clip操作：`x2 = (1 - x).clip(min=1e-5)`