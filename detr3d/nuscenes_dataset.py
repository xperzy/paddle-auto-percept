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

This code implements NuscenesDataset used in DETR3D

A Paddle Implementation of Deformable DETR3D as described in:
"DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries"
Paper Link: https://arxiv.org/abs/2110.06922
"""
import pickle
import numpy as np
import cv2
import paddle
from paddle.io import Dataset

class NuscenesDataset(Dataset):
    """Nuscenes dataset for testing"""
    def __init__(self, data_root, anno_file, classes):
        self.data_root = data_root
        self.anno_file = anno_file
        self.class_names = classes
        self.cat2id = {name: i for i, name in enumerate(self.class_names)}
        self.data_infos = self.load_annotations(self.anno_file)
        # now only support test mode
        self.test_mode = True
        assert self.test_mode is True

    def load_annotations(self, anno_file):
        with open(anno_file, 'rb') as infile:
            data = pickle.load(infile)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def load_multiview_image_from_files(self, input_dict):
        filename = input_dict['img_filename']
        img = np.stack([cv2.imread(name) for name in filename], axis=-1)
        img = img.astype(np.float32)
        input_dict['filename'] = filename
        # to list
        input_dict['img'] = [img[..., i] for i in range(img.shape[-1])]
        input_dict['img_shape'] = img.shape
        input_dict['ori_shape'] = img.shape
        input_dict['pad_shape'] = img.shape # init val
        input_dict['scale_factor'] = 1.0
        num_channels = img.shape[2]
        input_dict['img_norm_cfg'] = {"mean": np.zeros(num_channels, dtype=np.float32),
                                      "std": np.ones(num_channels, dtype=np.float32),
                                      "to_rgb": True}
        return input_dict

    def normalize_multiview_image(self, input_dict, mean, std, to_rgb=False):
        def im_normalize(img, mean, std, to_rgb):
            img = np.float32(img) if img.dtype != np.float32 else  img.copy()
            mean = np.float64(mean.reshape(1, -1))
            stdinv = 1 / np.float64(std.reshape(1, -1))
            if to_rgb:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            cv2.subtract(img, mean, img)
            cv2.multiply(img, stdinv, img)
            return img

        input_dict['img'] = [im_normalize(img, mean, std, to_rgb) for img in input_dict['img']]
        input_dict['img_norm_cfg'] = {'mean': mean, 'std': std, 'to_rgb': to_rgb}
        return input_dict

    def pad_multiview_image(self, input_dict, size=None, size_divisor=None, pad_val=0):
        def impad(img, size, pad_val):
            if len(size) < len(img.shape):
                size = size + (img.shape[-1], )
            pad = np.empty(size, dtype=img.dtype)
            pad[...] = pad_val
            pad[:img.shape[0], :img.shape[1], ...] = img
            return pad

        def impad_to_multiple(img, divisor, pad_val):
            # pad img to ensure each side to be multiple of some number
            pad_h = int(np.ceil(img.shape[0]/divisor)) * divisor
            pad_w = int(np.ceil(img.shape[1]/divisor)) * divisor
            return impad(img, (pad_h, pad_w), pad_val)

        padded_img = []
        if size is not None:
            padded_img = [impad(img, size, pad_val) for img in input_dict['img']]
        elif size_divisor is not None:
            padded_img = [impad_to_multiple(img,
                                            size_divisor,
                                            pad_val) for img in input_dict['img']]

        input_dict['img'] = padded_img
        input_dict['img_shape'] = [img.shape for img in padded_img]
        input_dict['pad_shape'] = [img.shape for img in padded_img]
        input_dict['pad_fixed_size'] = size
        input_dict['pad_size_divisor'] = size_divisor
        return input_dict

    def __getitem__(self, idx):
        if self.test_mode is True:
            info = self.data_infos[idx]
            input_dict = {"sample_idx": info['token'],
                          "pts_filename": info['lidar_path'],
                          "sweeps": info['sweeps'],
                          "timestamp": info['timestamp']/1e6,
                          "img_filename": [],
                          "lidar2img": [],
                          "cam_intrinsic": [],
                          "lidar2cam": []}
            # get lidar2cam params
            image_paths = []
            lidar2img_rts = []
            cam_intrinsics = []
            lidar2cam_rts = []
            for cam_type, cam_info in info['cams'].items():
                # get image file path
                image_paths.append(cam_info['data_path'])
                # get lidar to image transforms
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = np.matmul(cam_info['sensor2lidar_translation'], lidar2cam_r.T)
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = np.matmul(viewpad, lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict['img_filename']=image_paths
            input_dict['lidar2img']=lidar2img_rts
            input_dict['cam_intrinsic']=cam_intrinsics
            input_dict['lidar2cam']=lidar2cam_rts

            # load multi view images
            input_dict = self.load_multiview_image_from_files(input_dict)
            # normalize
            input_dict = self.normalize_multiview_image(input_dict,
                mean=np.array([103.530, 116.280, 123.675]),
                std=np.array([1.0, 1.0, 1.0]), to_rgb=True)
            # pad
            input_dict = self.pad_multiview_image(input_dict, size_divisor=32)
            input_dict['pcd_scale_factor'] = 1.0

            # img transpose and to tensor
            imgs = [img.transpose(2, 0, 1) for img in input_dict['img']]
            imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            input_dict['img'] = paddle.to_tensor(imgs)

            # img_meta: keys
            img_meta_keys = ['filename','ori_shape','img_shape','lidar2img','pad_shape',
                    'scale_factor', 'img_norm_cfg', 'sample_idx', 'pcd_scale_factor',
                    'pts_filename']
            img_metas = {}
            for key in img_meta_keys:
                if key in input_dict:
                    img_metas[key] = input_dict[key]
                else:
                    print(f'Key not found: {key}')

            data = {}
            data['img_metas'] = img_metas
            data['img'] = input_dict['img']
            return data
        else:
            raise ValueError("Now only support test mode!")

    def __len__(self):
        return len(self.data_infos)
