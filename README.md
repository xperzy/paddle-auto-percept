[English](./README.md) | [简体中文](./README_CN.md)

# Paddle Auto Percept #
_Learn Autonomous Driving Perception from Scratch – No Complex Dependencies Required_

[![GitHub](https://img.shields.io/github/license/xperzy/paddle-auto-percept?color=blue)](./LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/xperzy/paddle-auto-percept/badge)](https://www.codefactor.io/repository/github/xperzy/paddle-auto-percept)
[![CLA assistant](https://cla-assistant.io/readme/badge/xperzy/paddle-auto-percept)](https://cla-assistant.io/xperzy/paddle-auto-percept)
[![GitHub Repo stars](https://img.shields.io/github/stars/xperzy/paddle-auto-percept?style=social)](https://github.com/xperzy/paddle-auto-percept/stargazers)


<p align="center">    
    <img src="./pap.png" width="100%"/>
</p>

# Learn Autonomous Driving Perception Models from Scratch ##

:red_car: PaddleAutoPercept (also known as `paddle-auto-percept` or `PAP`) is an open-source, beginner-friendly project that implements core autonomous driving perception algorithms **from scratch** using the PaddlePaddle framework


:blue_car: This project covers a range of algorithms, from 2D object detection with DETR to 3D surround-view perception with BEVFormer, systematically showcasing their evolution and core concepts.

:bus: Each algorithm is implemented entirely in  [PaddlePaddle](https://www.paddlepaddle.org/), eliminating complex dependencies and deeply nested framework structures. The code is designed to be clear and intuitive, helping developers grasp the core logic and implementation details of each algorithm.

> At least one model’s inference results have been aligned with the official implementation's accuracy, verifying the reliability of the implementations.

## Challenges in Learning Autonomous Driving Perception ##
1. **High code complexity** – Official implementations often rely on complex framework designs and deeply nested class structures, making the code difficult to read and debug.
2. **Hard-to-understand implementation details** – Key operations like data processing and feature extraction are often poorly documented, making it challenging for beginners.
3. **Inconsistent concepts across models** – Similar concepts are implemented differently across different models, making it difficult for learners to form a systematic understanding.
4. **Gap between theory and practice** – There is often a significant gap between the descriptions in research papers and the actual code implementations, with a lack of beginner-friendly materials to bridge this gap.
5. **High hardware and environment requirements** – Many implementations require high-end hardware and complex dependency configurations, making it difficult for learners to experiment and run inference quickly.


## Project Highlights ##
- ✅ **Comprehensive Coverage**
  - This project implements core perception algorithms from scratch, ranging from DETR to BEVFormer, gradually demonstrating the evolution from 2D object detection to 3D surround-view perception. Each algorithm is independently implemented, free from the complex dependencies and framework nesting found in official repositories, helping learners deeply understand the core logic of each algorithm.

- 🔹 **Learn from Scratch**
  - The project adopts a simplified design, avoiding complex framework structures and interface hierarchies. The code is clean and well-commented, allowing learners to quickly get started and gradually master every aspect of algorithm implementation.

- ⚡ **Minimal Dependencies**
  - No complicated configurations or external dependencies. The inference pipeline runs on CPU, making it accessible even on low-resource devices like MacBooks. Significantly lowers the hardware barrier, allowing for easy learning and experimentation.
    
- 📐 **Unified Code Structure**
  - All models follow a consistent coding style and structure, making it easier for learners to compare and understand the similarities and differences between different models. Helps learners grasp the overall architecture and evolution of autonomous driving perception algorithms.
    
- 🛠 **Independent Implementations**
  - Data preprocessing is implemented independently, with a simple design, making it easier to understand data flow and input-output structures.

> The project is not just a code repository—it serves as a detailed learning guide to help every participant master the core logic of autonomous driving perception algorithms.


## Models ##
- [x] DETR (https://arxiv.org/abs/2005.12872)
- [x] DeformableDETR (https://arxiv.org/abs/2010.04159)
- [x] DETR3D (https://arxiv.org/abs/2110.06922)
- [x] BEVFormer (https://arxiv.org/abs/2203.17270)

## Installation ##
Ensure you have PaddlePaddle >= 2.6 installed:
- Paddle >= 2.6 (https://www.paddlepaddle.org.cn/)
  - CPU： `pip install paddlepaddle==2.6.2`
  - GPU： `pip install paddlepaddle-gpu==2.6.2`



## Quick Start ##
1. Clone the repository：
   ```shell
   git clone https://github.com/xperzy/paddle-auto-percept.git
   cd paddle-auto-percept
   cd detr
   ```
3. Download model weights：
   - DETR: https://huggingface.co/xperzy/detr-r50-paddle
   - DeformableDETR: https://huggingface.co/xperzy/deformable-detr-r50-paddle
   - DETR3D: https://huggingface.co/xperzy/detr3d-r50-paddle
   - BEVFormer: https://huggingface.co/xperzy/bevformer-r101-paddle

5. Run the inference：
   - Copy the downloaded model weights to the project directory (or modify the weight path in main.py). Then, execute:
       ```shell
       python main.py
       ```
   - (For DETR3D and BEVFormer) Preprocess the NuScenes dataset：
      - [ ] NuScenes dataset processing required before running DETR3D and BEVFormer. (Please refer to original paper github repos for now)
