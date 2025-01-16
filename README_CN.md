简体中文 | [English](./README.md)

# Paddle Auto Percept #
_不需要配环境、不需要依赖库，从0开始学习智驾感知技术_

[![GitHub](https://img.shields.io/github/license/xperzy/paddle-auto-percept?color=blue)](./LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/xperzy/paddle-auto-percept/badge)](https://www.codefactor.io/repository/github/xperzy/paddle-auto-percept)
[![CLA assistant](https://cla-assistant.io/readme/badge/xperzy/paddle-auto-percept)](https://cla-assistant.io/xperzy/paddle-auto-percept)
[![GitHub Repo stars](https://img.shields.io/github/stars/xperzy/paddle-auto-percept?style=social)](https://github.com/xperzy/paddle-auto-percept/stargazers)


<p align="center">    
    <img src="./PaddleViT.png" width="100%"/>
</p>



## 从零开始实现自动驾驶感知算法 ##

:red_car: PaddleAutoPercept (`paddle-auto-percept`或`PAP`) 是一个初学者友好的开源项目，在 PaddlePaddle 框架下**从零开始实现**自动驾驶领域的核心感知算法。

:blue_car: 项目涵盖的算法从 DETR 的 2D 目标检测到 BEVFormer 的 3D 环视感知，系统展示了这些算法的演进过程及其核心思想。

:bus: 每种算法的实现都完全基于 深度学习框架 [PaddlePaddle](https://www.paddlepaddle.org/)进行开发，去除了复杂的依赖关系和多层次的框架嵌套，代码设计简洁直观，帮助开发者深入理解算法的核心逻辑和实现细节。

> 至少一个模型的推理结果已与官方实现精度对齐，验证了实现的可靠性。

## 学习智驾感知算法的难点： ##

1. **代码复杂度高**： 官方实现通常依赖复杂的框架设计和高度嵌套的类结构，使得代码难以阅读和调试。

2. **细节实现难理解**： 关键操作（如数据处理、特征提取等）往往缺乏清晰说明，对新手尤其不友好。

3. **概念与实现不一致**： 不同模型对相似概念的实现方式不统一，导致学习者难以形成系统性的理解。

4. **理论与实践脱节**： 论文描述与实际代码实现之间存在显著差距，缺乏易于理解的桥梁材料。

5. **环境与资源要求高**： 高硬件需求和复杂的依赖配置使学习者难以快速上手实验和推理。

## 项目特色 ##
- **完整覆盖**
  - 项目从零开始实现从 DETR 到 BEVFormer 的核心感知算法，逐步展示从 2D 目标检测到 3D 环视感知的演进过程。每种算法都是独立实现，去除了官方代码中的复杂依赖关系和框架嵌套，帮助学习者深入理解每个算法的核心逻辑。

- **从零开始**
  - 采用简化的设计，避免了复杂的框架和接口结构，代码清晰且注释详尽，帮助学习者快速上手并逐步掌握算法实现的各个环节。

- **极简依赖**
  - 项目没有复杂的配置和外部依赖，推理部分可在 CPU 环境下运行，即使在 Mac 环境等资源有限的设备上也能顺利执行，大大降低了硬件要求，方便广泛的学习与实践。

- **统一结构**
  - 所有模型的实现遵循统一的代码风格和结构，便于学习者对比理解不同模型间的共性与差异，帮助学习者理清算法的整体架构和演进脉络。

- **独立实现**
  - 数据预处理部分独立实现，设计简洁，便于理解数据流动和输入输出结构。
 
> 这个项目不仅仅是代码的实现，更是一份详细的学习指南，让每一位参与者都能轻松掌握自动驾驶感知算法的核心逻辑。

