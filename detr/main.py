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
Deformable DETR in Paddle

A Paddle Implementation of Deformable DETR as described in:
"Deformable DETR: Deformable Transformers for End-to-End Object Detection"
Paper Link: https://arxiv.org/abs/2010.04159
"""
from PIL import Image
import requests
import numpy as np
from matplotlib import pyplot as plt
import paddle
from detr import Detr

id2label = {0: 'N/A', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
            7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'N/A',
            13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog',
            19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
            25: 'giraffe', 26: 'N/A', 27: 'backpack', 28: 'umbrella', 29: 'N/A', 30: 'N/A',
            31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
            37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
            41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'N/A',
            46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
            52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
            58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
            64: 'potted plant', 65: 'bed', 66: 'N/A', 67: 'dining table', 68: 'N/A', 69: 'N/A',
            70: 'toilet', 71: 'N/A', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
            76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
            81: 'sink', 82: 'refrigerator', 83: 'N/A', 84: 'book', 85: 'clock', 86: 'vase',
            87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def plot_results(pil_img, scores, labels, boxes):
    """Plot bboxes and scores on image"""
    # colors for visualization
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    colors = colors * 100
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for score, label, (xmin, ymin, xmax, ymax), c  in zip(scores.tolist(),
                                                          labels.tolist(),
                                                          boxes.tolist(),
                                                          colors):
        ax.add_patch(plt.Rectangle((xmin, ymin),
                                   xmax - xmin,
                                   ymax - ymin,
                                   fill=False,
                                   color=c,
                                   linewidth=3))
        text = f'{id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox={'facecolor': 'yellow', 'alpha':0.5})
    plt.axis('off')
    plt.show()


def preprocess(image, pad=False):
    """
    Normalize Image and convert to Tensor
    Args:
        image: PIL.Image
    Returns:
        image: Paddle.Tensor, with shape [1, C, H, W]
    """
    max_size = (800, 1333)  # (shortest_edge, longest_edge)
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)

    # resize
    width, height = image.size
    w_scale = max_size[1] / width
    h_scale = max_size[0] / height
    min_scale = min(h_scale, w_scale)
    new_height = int(height * min_scale)
    new_width = int(width * min_scale)

    resized_image = image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    resized_image = np.array(resized_image)

    # normalize
    resized_image = resized_image.astype(np.float32)
    resized_image = resized_image / 255.0
    img_mean = np.array(img_mean, resized_image.dtype)
    img_std = np.array(img_std, resized_image.dtype)
    resized_image = (resized_image - img_mean) / img_std

    # reshape
    resized_image = resized_image[np.newaxis, :, :, :]
    resized_image = np.transpose(resized_image, [0, 3, 1, 2])

    # to tensor
    resized_image = paddle.to_tensor(resized_image, dtype='float32')

    # pad to same size (bs = 1), return image and mask (padded area set to 0)
    bs, c, original_h, original_w = resized_image.shape
    if pad is True:
        padded_size = [bs, c, max_size[0], max_size[1]]
        padded_image = paddle.zeros(padded_size, dtype='float32')
        padded_image[0, :, :original_h, :original_w] = resized_image
        padded_mask = paddle.zeros([bs, max_size[0], max_size[1]])
        padded_mask[:, :original_h, :original_w] = 1
    else:
        padded_image = resized_image
        padded_mask = paddle.ones([bs, original_h, original_w])

    return padded_image, padded_mask


def postprocess(logits, pred_boxes, target_sizes=None, threshold=0.9):
    """
    Args:
        logits:
        pred_boxes:
        target_sizes: Tensor, [batch_size, 2], original heights and widths for each image in batch
        threshold: float
        topk: int
    Returns:
        out: list of dict, each dict stores the res of one img in batch 
    """
    prob = paddle.nn.functional.softmax(logits, -1)
    scores = prob[..., :-1].max(-1)
    labels = prob[..., :-1].argmax(-1)
    # center to corner format
    center_x, center_y, width, height = pred_boxes.unbind(-1)
    boxes = paddle.stack([(center_x - 0.5 * width),
                          (center_y - 0.5 * height),
                          (center_x + 0.5 * width),
                          (center_y + 0.5 * height)], axis=-1)

    if target_sizes is not None:
        if isinstance(target_sizes, list):
            img_h = paddle.to_tensor([i[0] for i in target_sizes])
            img_w = paddle.to_tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)
        scale_factor = paddle.stack([img_w, img_h, img_w, img_h], 1)
        boxes = boxes * scale_factor

    results = []
    for s, l, b in zip(scores, labels, boxes):
        score = s[s > threshold]
        label = l[s > threshold]
        boxes = b[s > threshold]
        results.append({'scores': score, 'labels': label, 'boxes': boxes})

    return results


def main():
    """ load image and run inference"""
    #paddle.device.set_device("cpu")
    paddle.device.set_device("gpu")
    # load image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True, timeout=20).raw)
    # preprocess
    input_pixels, input_mask = preprocess(image, pad=False)
    # create model
    detr_model = Detr(embed_dim=256,
                      ffn_dim=2048,
                      num_heads=8,
                      num_encoder_layers=6,
                      num_decoder_layers=6,
                      num_queries=100,
                      num_classes=92)
    detr_model.eval()
    # load weights
    weight_path = './detr_r50.pdparams'
    state_dict = paddle.load(weight_path)
    detr_model.set_state_dict(state_dict)
    # inference
    outputs = detr_model(input_pixels, input_mask)
    # postprocess
    results = postprocess(logits=outputs['logits'],
                          pred_boxes=outputs['pred_boxes'],
                          target_sizes=[image.size[::-1]],
                          threshold=0.9)
    # only print the 1st results in batch
    results = results[0]
    # print and plot results
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    plot_results(image, results['scores'], results['labels'], results['boxes'])


if __name__ == "__main__":
    main()
