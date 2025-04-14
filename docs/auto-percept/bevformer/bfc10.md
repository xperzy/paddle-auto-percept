# 从零开始实现BEVFormer (10) - Classification Head 和 BBox Regression Head

BEVFormer的类别预测和框回归预测，是对Decoder的每一层都进行预测：

- 类别预测：
    - Linear - LN - ReLU - Linear - LN - ReLU - Linear
- 框回归：
    - Linear - ReLU - Linear - ReLU - Linear
    - 输出维度：10
        - 分别代表（cx, cy, w, l, cz, h, rot_sin, rot_cos, vx, vy）
    - 这里的cx, cy, 和cz都是基于当前层的参考点的偏移量

## 代码实现：

```python
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
```

## Forward方法：

```python
        for level_idx in range(len(all_hidden_states) - 1):
            if level_idx == 0:
                reference = init_reference  # 1st is the original ref_pts
            else:
                reference = inter_reference[level_idx - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level_idx](all_hidden_states[level_idx + 1])
            # compute bbox coord
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
```