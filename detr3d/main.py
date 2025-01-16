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
DETR3D in Paddle

A Paddle Implementation of Deformable DETR as described in:
"DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries"
Paper Link: https://arxiv.org/abs/2110.06922
"""
import paddle
from nuscenes_dataset import NuscenesDataset
from detr3d import Detr3D

def main():
    """load image from dataset and run inference"""
    paddle.set_device("gpu")  # ["gpu", "cpu"]

    # create dataset and dataloader
    data_root = 'data/nuscenes/'
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    dataset = NuscenesDataset(data_root=data_root,
                              anno_file=data_root + 'nuscenes_infos_val.pkl',
                              classes=class_names)
    dataloader = paddle.io.DataLoader(dataset, batch_size=1)

    # create model
    detr3d_model = Detr3D()
    detr3d_model.eval()
    # load pretrained weights
    weight_path = './detr3d.pdparams'
    state_dict = paddle.load(weight_path)
    detr3d_model.set_state_dict(state_dict)

    # inference
    for idx, data in enumerate(dataloader):
        with paddle.no_grad():
            img = data['img']
            img_metas = [data['img_metas']]
            outputs = detr3d_model(img=img, img_metas=img_metas)
            bbox_list = detr3d_model.pts_bbox_head.get_bboxes(outputs, img_metas, rescale=False)
            print(bbox_list)
        # only inference the 1st batch for demo
        if idx == 0:
            break


if __name__ == "__main__":
    main()
