import os
import numpy as np
import pickle
import cv2
import paddle
from paddle.io import Dataset
from paddle.vision import image_load
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

def collate_fn(batch):
    img_data, img_metas = zip(*batch)
    img_data = paddle.stack(img_data, axis=0)

    return img_data, img_metas

class NuscenesDataset(Dataset):
    def __init__(self, data_root, anno_file, classes):
        self.data_root = data_root
        self.anno_file = anno_file
        self.CLASSES = classes

        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.anno_file)

        # now only support test mode
        self.test_mode = True
        assert self.test_mode == True

    def load_annotations(self, anno_file):
        with open('./data/nuscenes/nuscenes_infos_temporal_val.pkl', 'rb') as infile:
            data = pickle.load(infile)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def LoadMultiViewImageFromFiles(self, input_dict):
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
        input_dict['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=True)
        return input_dict

    def NormalizeMultiviewImage(self, input_dict, mean, std, to_rgb=False):
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
        input_dict['img_norm_cfg'] = dict(
            mean=mean,
            std=std,
            to_rgb=to_rgb)
        return input_dict

    def PadMultiviewImage(self, input_dict, size=None, size_divisor=None, pad_val=0):
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
            padded_img = [impad_to_multiple(img, size_divisor, pad_val) for img in input_dict['img']]

        input_dict['img'] = padded_img
        input_dict['img_shape'] = [img.shape for img in padded_img]
        input_dict['pad_shape'] = [img.shape for img in padded_img]
        input_dict['pad_fixed_size'] = size
        input_dict['pad_size_divisor'] = size_divisor

        return input_dict

    def __getitem__(self, idx):
        if self.test_mode is True:
            info = self.data_infos[idx]
            input_dict = dict(
                sample_idx = info['token'],
                pts_filename = info['lidar_path'],
                sweeps = info['sweeps'],
                ego2global_translation=info['ego2global_translation'],
                ego2global_rotation=info['ego2global_rotation'],
                timestamp = info['timestamp'] / 1e6,
                frame_idx=info['frame_idx'],
                prev_idx=info['prev'],
                next_idx=info['next'],
                scene_token=info['scene_token'],
                can_bus=info['can_bus'],
                img_filename=[],
                lidar2img=[],
                cam_intrinsic=[],
                lidar2cam=[])

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

            # can bus
            rotation = Quaternion(input_dict['ego2global_rotation'])
            translation = input_dict['ego2global_translation']
            can_bus = input_dict['can_bus']
            can_bus[:3] = translation
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle

            # load multi view images
            input_dict = self.LoadMultiViewImageFromFiles(input_dict)
            # normalize
            input_dict = self.NormalizeMultiviewImage(input_dict,
                mean=np.array([103.530, 116.280, 123.675]),
                std=np.array([1.0, 1.0, 1.0]), to_rgb=True)
            # pad
            input_dict = self.PadMultiviewImage(input_dict, size_divisor=32)
            input_dict['pcd_scale_factor'] = 1.0
            # img transpose and to tensor
            imgs = [img.transpose(2, 0, 1) for img in input_dict['img']]
            imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            input_dict['img'] = paddle.to_tensor(imgs)
            # img_meta
            img_meta_keys = ['filename','ori_shape','img_shape','lidar2img','pad_shape',
                    'scale_factor', 'img_norm_cfg', 'sample_idx', 'pcd_scale_factor',
                    'pts_filename', 'can_bus']
            img_metas = {}
            for key in img_meta_keys:
                if key in input_dict:
                    img_metas[key] = input_dict[key]
                else:
                    print(f'Key not found: {key}')

            return input_dict['img'], img_metas
        else:
            raise ValueError("Now only support test mode!")

    def __len__(self):
        return len(self.data_infos)
