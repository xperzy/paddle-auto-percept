# Copyright (c) 2024 PaddleAutoPercept Authors. All Rights Reserved.
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
import paddle
from nuscenes_dataset import NuscenesDataset, collate_fn
from bevformer import BEVFormer


def main():
    """ load image and run inference"""
    paddle.set_device("gpu")
    #paddle.set_device("cpu")

    # create dataset and dataloader
    data_root = 'data/nuscenes/'
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    dataset = NuscenesDataset(data_root=data_root,
                              anno_file=data_root + 'nuscenes_infos_temporal_val.pkl',
                              classes=class_names)
    dataloader = paddle.io.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # create model and load pretrained weights
    model = BEVFormer()
    model.eval()
    model.set_state_dict(paddle.load('./bevformer_r101.pdparams'))

    # inference
    for idx, data in enumerate(dataloader):
        with paddle.no_grad():
            img = data[0]
            img_metas = data[1]
            outputs = model(img=img, img_metas=img_metas)
            bbox_list = model.pts_bbox_head.get_bboxes(outputs, img_metas, rescale=False)
            print(bbox_list)
        if idx == 0:  # demo: only inference the 1st batch
            break


if __name__ == "__main__":
    main()
