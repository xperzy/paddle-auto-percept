# 从零开始学 - 智驾基础：NuScenes 数据集

## 1. 下载数据

下载链接：[https://www.nuscenes.org/nuscenes#download](https://www.nuscenes.org/nuscenes#download)

需要用邮箱注册，登录后可以下载。

我们为了方便学习，这里先下载Mini版本，大约3.88G。

![Screenshot 2024-10-04 at 09.42.07.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20%E6%99%BA%E9%A9%BE%E5%9F%BA%E7%A1%80%EF%BC%9ANuScenes%20%E6%95%B0%E6%8D%AE%E9%9B%86%201bd3135fac1780e6a232c1a227160f03/Screenshot_2024-10-04_at_09.42.07.png)

## 传感器分布

官网上有一张图，展示了数据采集车的传感器分布，其中：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20%E6%99%BA%E9%A9%BE%E5%9F%BA%E7%A1%80%EF%BC%9ANuScenes%20%E6%95%B0%E6%8D%AE%E9%9B%86%201bd3135fac1780e6a232c1a227160f03/image.png)

- 1x Lidar：32线，20Hz， 360度
- 6x Camera：12Hz，1600x1200分辨率（crop到1600x900）
- 5x Radar
- 1x IMU & GPS
- 图中可以看到各个传感器的位置，以及**坐标系**

## 文件格式

文件下载并解压后可以看到如下的文件格式，这里我们

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20%E6%99%BA%E9%A9%BE%E5%9F%BA%E7%A1%80%EF%BC%9ANuScenes%20%E6%95%B0%E6%8D%AE%E9%9B%86%201bd3135fac1780e6a232c1a227160f03/image%201.png)

我们可以看到有个文件夹samples和sweeps的文件结构一样，这个两个文件夹存的是不同类型的传感器数据，具体来说：

- samples：关键帧的传感器数据；有严格标注，通常用于训练和评测。
- sweeps：非关键帧数据（也就是关键帧之间采集的数据），一般只包含LiDAR和雷达的点云数据 ；无标注，通常用于融合时序信息。

可以进一步看到，例如samples/CAM_FRONT文件夹，存的是jpg格式的图片；samples/LIDAR_TOP文件夹，存的是.pcd.bin的点云数据；v1.0-mini中的json文件，则是存放标注和meta data相关的信息。

**要使用这些数据，官方提供了一个python工具包，nuscenes-devkit，方便我们读取数据。**

## 数据读取

### 1. 安装库

```bash
pip install nuscenes-devkit
```

### 2. 导入数据集

假设数据集文件夹的路径是 ./data/v1.0-mini

```bash
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='./data/v1.0-mini/', verbose=True)
```

如果导入成功，那么会显示类似如下的信息：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20%E6%99%BA%E9%A9%BE%E5%9F%BA%E7%A1%80%EF%BC%9ANuScenes%20%E6%95%B0%E6%8D%AE%E9%9B%86%201bd3135fac1780e6a232c1a227160f03/image%202.png)

### 3. 数据组织结构

### Scene

数据集包含了多个scene（mini集合是10个，完整集合大概1000个），可以用`nusc.list_scenes()` 来list全部scene，也可以用`nusc.scene[idx]` 来索引多某一个scene

```python
nusc.list_scenes()  # 列出所有的scenes
my_scene = nusc.scene[0]
```

my_scene保存了多个metadata，例如：

```
{'token': 'cc8c0bf57f984915a77078b10eb33198',
 'log_token': '7e25a2c8ea1f41c5b0da1e69ecfa71a2',
 'nbr_samples': 39,
 'first_sample_token': 'ca9a282c9e77460f8360f564131a8af5',
 'last_sample_token': 'ed5fc18c31904f96a8f0dbb99ff069c0',
 'name': 'scene-0061',
 'description': 'Parked truck, construction, intersection, turn left, following a van'}
```

### Sample

在scene中，给定时间戳的一个标注关键帧用**sample**来表示，我们可以用下面的代码来可视化scene[0]中的第一个标注sample。

```bash
my_scene = nusc.scene[0]

first_sample_token = my_scene[’first_sample_token’]

nusc.render_sample(first_sample_token)
```

![Figure_1.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20%E6%99%BA%E9%A9%BE%E5%9F%BA%E7%A1%80%EF%BC%9ANuScenes%20%E6%95%B0%E6%8D%AE%E9%9B%86%201bd3135fac1780e6a232c1a227160f03/Figure_1.png)

可以使用.get(’sample’, token)方法获得metadata：

```bash
my_sample = nusc.get('sample', first_sample_token)
nusc.list_sample(my_sample['token'])
```

### sample_data

- 在sample中，使用‘data’ key来索引到sample对应的传感器数据： `my_sample[’data’]`
- 使用例如`nusc.get(’sample_data’, my_sample[’data’][’CAM_FRONT’])` 的方式，可以查看sample_data的metadata

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20%E6%99%BA%E9%A9%BE%E5%9F%BA%E7%A1%80%EF%BC%9ANuScenes%20%E6%95%B0%E6%8D%AE%E9%9B%86%201bd3135fac1780e6a232c1a227160f03/image%203.png)

- 使用例如`nusc.render_sample_data(cam_front_data['token'])` 的方式，可视化某一个sensor的sample_data

### sample_annotation

同理，对于某个sample中的某个annotation，我们也可以先获得其token，然后使用`nusc.get(’sample_annotation’, my_annotation_token)`的方式，获得annotation详细信息

- annotation中包含有instance

具体的信息可以参考官方教程：

[https://www.nuscenes.org/tutorials/nuscenes_tutorial.html](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html)

各个属性的查询：

[https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)