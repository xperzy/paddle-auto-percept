# 附录 - 如何跑通BEVFormer Pytorch源码

BEVFormer是基于**mmdetection3d**（以下简称mmdet3d）库为基础开发，以plugin的形式在mmdet3d的代码和pipeline上运行。

mmdet3d库依赖于mmdet库还有mmcv库等基础库，从环境配置到跑通代码都有一定的学习曲线。因此本文首先介绍如何配置环境，并跑通BEVFormer源代码，然后在其他章节再进一步对算法进行详细分析，最终从零实现BEVFormer。

## 1. 硬件环境

跑通源码阶段，建议选择**配有NVIDIA GPU**的硬件环境（台式机、笔记本电脑、服务器等）。

- 作者曾经尝试使用MacOS进行配置，安装mm系列库经常会报错，没有完全解决问题，最终还是选择Linux + NVIDIA GPU环境。
    
    如果不需要跑通源码，只学习本教程的从零开始实现BEVFormer，可以使用CPU环境和MacOS环境。
    

**本文运行的硬件环境：**

- **GPU**：NVIDIA RTX-4080 移动版，显存12G
- **CUDA Driver**：550.120
- **OS**：Ubuntu22.04
- **Memory**： 32G

## 2. 源码下载

直接从官方github下载源码：

[https://github.com/fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)

或者使用命令行：

```bash
git clone https://github.com/fundamentalvision/BEVFormer.git
```

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image.png)

## 3. 下载模型文件

从官方Github的文档中可以查找到**模型权重**和**config文件**的下载地址：

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%201.png)

下载完成后，将权重文件放在项目文件目录下，将config文件放在项目config文件夹下。例如：

- 权重文件放在`ckpts/`下：例如`BEVFormer/ckpts/bevformer_small_epoch_24.pth`
- 配置文件放在`projects/configs/bevformer`下

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%202.png)

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%203.png)

## 4. 下载数据

BEVFormer使用Nuscenes数据集，可以从数据集官方网站下载数据（需要注册一下），下载地址是：

https://www.nuscenes.org/download

代码学习和开发阶段，我们可以只下载mini版本：

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR3D%201bd3135fac17804e8a8fe06f94f339e8/%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ADETR3D%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac17804a9a4ef770eb0adff6/image%202.png)

下载完成后，同样将数据解压放在项目文件夹下，例如： `BEVFormer/data/` ，文件结构应该是类似下面截屏的内容：

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20DETR3D%201bd3135fac17804e8a8fe06f94f339e8/%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ADETR3D%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac17804a9a4ef770eb0adff6/image%203.png)

## 5. 环境配置

官方提供的模型权重Log中，提供了非常详细的运行环境，我们可以打开进行查阅和参考。实际开发过程中，由于源码的开发时间相对比较早，一些环境和版本对于新一些的硬件以及大家常用的版本已经有些出入，经过作者验证，有一些库是可以使用新的版本，但由于大家的软硬件环境各不相同，这里不推荐具体的软件版本，大家可以参考下面的配置流程。

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%204.png)

几个主要的版本（比较熟悉的可以直接对照和安装相关包）：

---

| **Package Name** | **Official BEVFormer Version** | **My Version** |
| --- | --- | --- |
| GPU | V100 x8 | RTX 4080 x1 |
| Linux | Linux | Ubuntu 22.04 |
| Python | 3.6.9 | 3.8.20 |
| NVCC(CUDA) | 10.1 | 11.1 |
| CuDNN | 7.6.5 | 8.0.5 |
| GCC | 5.4.0 | 7.5.0 |
| PyTorch | 1.8.1+cu90 | 1.9.1+cu111 |
| TorchVision | 0.8.0 |  |
| OpenCV | 4.1.1 |  |
| MMCV | 1.3.18 | 1.4.0 |
| MMDetection | 2.14.0 | 2.16.0 |
| MMSegmentation | 0.14.1 | 0.17.0 |
| MMDetection3D | 0.17.1 | 0.17.1 |
| numpy | N/A | 1.19.5 |
| numba | N/A | 0.48.0 |
| seaborn | N/A | 0.11.0 |

### **Step1. 安装Conda并创建Conda虚拟环境**

**安装并使用conda的主要好处是**：

（1）conda可以使用虚拟环境（也就是我们常用的conda env），对不同的python开发环境进行隔离

（2）conda里同样能够使用Pip安装(pip install)，pip安装不了的情况也可使用conda进行安装(conda install)

**安装Conda的方法比较简单**：

1. 参考官方安装教程：https://docs.anaconda.com/miniconda/install/#quick-command-line-install

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

1. 完成conda的初始化：

```bash
source ~/miniconda3/bin/activate
conda init --all
```

- anaconda 和 miniconda的区别？
    - TL;DR: 我们使用miniconda就行；anaconda有图形界面包含了很多库，miniconda是轻量化版本，需要自己安装库；
    - 具体可以参考官方说明：https://docs.anaconda.com/distro-or-miniconda/

**创建conda虚拟环境**：

```bash
conda create --name bevformer python=3.8
conda activate bevformer
```

### **Step2. 配置CUDA环境**

1. **下载并安装CUDA toolkit：**
    1. 连接（例如CUDA12.4）：https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local
    2. 选择对应的平台然后按照命令下载：
    
    ![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%205.png)
    
    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
    sudo sh cuda_12.4.0_550.54.14_linux.run
    
    ```
    
    - 安装CUDA：

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%206.png)

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%207.png)

![注意：这里不要勾选driver！](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%208.png)

注意：这里不要勾选driver！

安装完成后会显示如下内容：

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%209.png)

这一步之后，我们可以在路径`/usr/local/`下看到cuda文件夹。

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%2010.png)

这时我们可能会有多个CUDA版本，要怎么选择使用某一个CUDA版本呢？

1. **如何使用多个版本的CUDA：**
    - 22.04中，我安装了: cuda11.1, cuda11.6, cuda11.7,cuda12.4
    - CUDA的使用主要通过以下几个环境变量来控制：
        - `$PATH`
        - `$LD_LIBRARY_PATH`
        - `$CUDA_HOME`
    - 当我们需要使用某一版本的CUDA时，需要将其对应的路径添加到这些环境变量中，例如我们要使用CUDA11.1：
        
        ```bash
        export CUDA_HOME=/usr/local/cuda-11.1
        export PATH=/usr/local/cuda-11.1/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH 
        ```
        
    
    **如果我们想在不同的Conda虚拟环境中使用不同的CUDA版本，该怎么做？**
    
    - 可以通过设置conda激活时候的脚本来按照不同的conda环境使用不同版本的cuda：
        - 创建并在 ~/anaconda3/envs/YOUR_ENV/etc/conda/activate.d/activate.sh中添加：
            
            ```bash
            ORIGINAL_CUDA_HOME=$CUDA_HOME
            ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
            ORIGINAL_PATH=$PATH
            export CUDA_HOME=/usr/local/cuda-11.1
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
            ```
            
        - 创建并在 ~/anaconda3/envs/YOUR_ENV/etc/conda/deactivate.d/deactivate.sh中添加：
            
            ```bash
            # remove cuda bin from PATH
            REMOVE_PATH=$CUDA_HOME/bin
            export PATH=$(echo "$PATH" | sed -E "s;(^|:)$REMOVE_PATH(:|$);:;g; s;^:|:$;;g")
            export CUDA_HOME=$ORIGINAL_CUDA_HOME
            export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH
            unset ORIGINAL_CUDA_HOME
            unset ORIGINAL_LD_LIBRARY_PATH                                    
            ```
            

### Step3. conda 安装 gcc

mmdet3d在编译的时候，可能会出现报错，有可能是因为gcc版本过高引起的。我们可以在conda环境中安装gcc7.5，这样不会影响系统的gcc版本（而且ubuntu22.04安装gcc7.5不是特别方便）。

- 检查gcc 版本：
    
    ```bash
    gcc --version
    ```
    
- 使用conda安装gcc7.5:
    
    ```bash
    conda install -c conda-forge gxx_linux-64=7.5.0
    # 再将命令export到PATH中
    export PATH="home/$USER/anaconda3/envs/bev/libexec/gcc/x86_64-conda-linux-gnu/7.5.0:$PATH"
    ```
    

### Step4. 安装python相关依赖库

首先可以通过`conda env list` 命令查看当前所在的conda env，然后使用`conda activate your_env_name`激活对应的conda环境。

- **安装numpy:**
    
    ```bash
    pip install numpy==1.19.5
    ```
    
- **安装pytorch**
    
    官方安装连接：https://pytorch.org/get-started/previous-versions/#linux-and-windows-41
    
    ```bash
    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    
    验证版本和正确性：
    
    ![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%2011.png)
    
- **安装MMCV**
    
    ```bash
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html
    ```
    
    - 参考官方教程：https://mmcv.readthedocs.io/en/v1.4.1/get_started/installation.html
    - 安装过程中有时候会报numpy版本的错误，我安装成功的numpy版本是：1.19.5， opencv版本是4.10；安装过程中有时候会出现安装错误显示某些包未安装成功，可以手动pip install对应的包再重试安装mmcv
- **安装mmdet和mmsegmentation：**
    
    ```bash
    pip install mmdet==2.16.0
    pip install mmsegmentation==0.17.0
    ```
    
    - 参考官方教程：https://mmdetection.readthedocs.io/en/v2.16.0/get_started.html#install-mmdetection
    - 安装直接使用pip手动安装即可，不需要安装openmim
- **安装mmdet3d:**
    
    在DETR3D项目路径下：
    
    ```bash
    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    git checkout v0.17.1
    pip install -v -e .
    ```
    
    - 参考官方教程：https://mmdetection3d.readthedocs.io/en/v0.17.1/getting_started.html#installation
    - 如果出现报错是gcc版本的问题：按照上面说明在conda安装和配置gcc
    - 如果遇到报错： `Python.h:44:10: fatal error: crypt.h: No such file or directory`
        - 安装libxcrypt: `conda install --channel=conda-forge libxcrypt`
- 安装其他库：
    
    ```bash
    pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```
    

## 5. 运行源码

**处理数据：**

我们需要对下载好的Nuscenes数据进行处理，转换为pkl文件（python pickle），BEVFormer源代码中提供了相关的脚本，我们需要进行简单的修改。

- 官方说明：https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md

```bash
# 注意： v1.0 -> v1.0-mini
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data

```

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%2012.png)

运行成功后，会生成pkl文件：

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%2013.png)

**运行源码：**

项目源码中的脚本是在集群上运行的，我们在单机环境下，可以使用下面的脚本：   

```bash
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_r101_dcn_24ep.pth 1
```

当命令正确运行，我们可以看到如下的进度条显示：

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%2014.png)

此时的GPU被拉满，显存占用可以看到大约需要4.5G左右：

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%2015.png)

**查看结果：**

上面的命令运行完成（RTX4080移动版大约1min不到）后会有evaluation的结果显示：

![image.png](%E9%99%84%E5%BD%95%20-%20%E5%A6%82%E4%BD%95%E8%B7%91%E9%80%9ABEVFormer%20Pytorch%E6%BA%90%E7%A0%81%201bd3135fac178093a013d6934c59fd19/image%2016.png)

看到这些就表示运行成功了。