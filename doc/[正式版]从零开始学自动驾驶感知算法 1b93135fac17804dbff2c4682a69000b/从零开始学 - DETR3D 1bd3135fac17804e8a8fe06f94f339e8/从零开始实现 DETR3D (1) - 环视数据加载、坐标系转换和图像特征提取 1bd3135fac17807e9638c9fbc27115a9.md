# 从零开始实现 DETR3D (1) - 环视数据加载、坐标系转换和图像特征提取

本节我们从零开始实现DETR3D。在开始实现之前，我们可以将DETR3D的完整结构分为以下几个部分：

1. **输入**：主要是数据集（dataset & dataloader）的实现，包括数据加载，预处理等
2. **图像Backbone**：主要是图像特征提取的实现，包括ResNet，DCN结构，FPN Neck等
3. **Decoder**：主要是Transformer的实现，包括标准自注意力机制，可变形注意力机制以及参考点、采样点投影等
4. **Head**：主要是分类分支和框回归分支的实现

如下图所示：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20DETR3D%20(1)%20-%20%E7%8E%AF%E8%A7%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E3%80%81%E5%9D%90%E6%A0%87%E7%B3%BB%E8%BD%AC%E6%8D%A2%E5%92%8C%E5%9B%BE%E5%83%8F%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%201bd3135fac17807e9638c9fbc27115a9/image.png)

本节我们首先来看数据加载和特征提取。

## 数据集加载和实现：

DETR3D使用了Nuscenes数据集的环视图像部分，基于单帧进行3D目标检测。数据集的介绍可以参考之前的文章（[这里](https://www.notion.so/done-NuScenes-17e3135fac178004a546f81dc62cb974?pvs=21)），本节我们来看如何读取数据。

### 数据集转换：（这一步可以参考论文源代码，本文不实现这一部分）

为了方便对齐官方实现，我们也使用了官方的脚本，将Nuscenes数据转换成pkl文件，然后定义Dataset和Dataloader加载这些pkl文件。 

数据集转换的基本流程如下：

**输入**：Nuscenes数据集的路径（root_path），其文件结构基本如下（这里只使用v1.0-mini数据集）

![image.png](../%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%20-%20%E6%99%BA%E9%A9%BE%E5%9F%BA%E7%A1%80%EF%BC%9ANuScenes%20%E6%95%B0%E6%8D%AE%E9%9B%86%201bd3135fac1780e6a232c1a227160f03/image%201.png)

**操作过程**：

- 读取每一个有效的sample
- 对于每个sample：
    - 读取lidar的投影矩阵： lidar2ego,
    - 读取ego pose： ego2global
    - 对于6个camera分别读取：
        - 内参
        - 将camera的投影矩阵，转换为相对lidar坐标系
    - 读取frame和groundtruth

**输出**：pkl文件，其中包含我们需要的图像数据和对应的groundtruth，以及相关参数。例如：

- nuscenes_infos_train.pkl
- nuscenes_infos_val.pkl

这里需要注意的是：

1. 投影矩阵：最终得到的是sensor2lidar_rotation 和 sensor2lidar_translation，这个是外参矩阵，用于将各个camera投影到lidar坐标系
2. 各个camera也保存了内参，用于将图像坐标系投影到像素坐标系
3. 外参矩阵的计算过程（sensor2top）：
    1. sweep转为ego（相对于sweep）
    2. ego转为global
    3. global再转为ego（相对于lidar）
    4. ego再转为lidar

## Dataset实现：

通常在深度学习框架（pytorch或者paddle等）中实现Dataset时，主要是实现以下两个方法：

1. `init`方法：用于基本的数据加载和gt加载等
2. `__getitem__`方法：用来返回单个样本

对于DETR3D读取Nuscenes数据集来说：

- `init`方法，首先读入类别信息，标注信息和meta data（例如投影矩阵等）。
- `__getitem__`方法：首先计算外参矩阵，然后读取多视角图像，对图像进行归一化，然后pad到同一尺寸，最后转换为Tensor类型。

**计算外参矩阵：**

```python
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
```

- `cam_info[’sensor2lidar_rotation’]` : 是在dataset中已经经过转换的投影矩阵，这里是以3x3矩阵形式，表示从当前的cam坐标系到lidar坐标系的**旋转矩阵**。
- `cam_info[’sensor2lidar_translation’]` : 是在dataset中已经经过转换的投影矩阵，这里是以1x3矩阵形式，表示从当前的cam坐标系到lidar坐标系的**平移矩阵**。
- `cam_info[’cam_intrinsic’]` : 是当前cam的内参，用于将图像坐标系转换到像素坐标系。
- 我们的目标是：
    - 获得lidar2cam的**外参转换矩阵**，以齐次坐标的形式将rotation和translation一起表示为一个4x4的矩阵
    - 获得相机**内参矩阵**，用齐次坐标表示
    - 使用齐次坐标的目的是为了方便矩阵运算
- 计算过程：
    - 齐次坐标下，从cam坐标系转换到lidar坐标系下的计算方式为：
    
    $$
    \begin{bmatrix} P_{lidar} \\ 1 \end{bmatrix} = \begin{bmatrix} R && T \\  0 && 1 \end{bmatrix} \begin{bmatrix} P_{cam} \\ 1\end{bmatrix}
    $$
    
    - 我们的目标是：给3D坐标（lidar坐标系）点，将他们都投影到各个图像上（cam坐标系），这就是上面投影的逆投影。所以有：
    
    $$
    \begin{bmatrix} P_{cam} \\ 1 \end{bmatrix} = \begin{bmatrix} R && T \\  0 && 1 \end{bmatrix} ^{-1} \begin{bmatrix} P_{lidar} \\ 1\end{bmatrix}
    $$
    
    - 其中$\begin{bmatrix} R && T \\  0 && 1 \end{bmatrix} ^{-1}$ 可以理解为，先反向平移再反向旋转，将这两部分分开处理：
        - 反向平移： 平移是将原来的平移反向，并考虑到旋转的影响。$T_{\text{new}} = -R^T \cdot T$
        - 反向旋转： 旋转矩阵是正交阵，所以 $R^{-1} = R^T$
    - 所以：
        
        $$
        \begin{bmatrix} R && T \\  0 && 1 \end{bmatrix} ^{-1}  = \begin{bmatrix} R^T && -R^TT \\  0 && 1 \end{bmatrix}
        $$
        
    - 内参，通常用来将相机坐标系转换到像素坐标系，内参可以被表示为一个3x3的矩阵，其形式如下：
    
    $$
    K = \begin{bmatrix} f_x && 0 &&c_x \\ 0 && f_y && c_y \\ 0 && 0 && 1 \end{bmatrix}
    $$
    
    - 其中各个参数的含义为：
        - $f_x$,$f_y$：分别表示图像在 **水平（x 轴方向）和垂直（y 轴方向）** 上的 **等效焦距**，单位是像素
        - $c_x, c_y$：主点（Principal Point）的坐标，表示光轴与图像平面的交点，通常在图像中心附近。
    - 内参投影公式：
        - 假设一个 3D 点$[X_c, Y_c, Z_c]^T$在相机坐标系中，它在像素平面上的投影为：
            - $\mathbf{p} = K \cdot \begin{bmatrix} X_c / Z_c \\ Y_c / Z_c \\ 1 \end{bmatrix}$
            - 这里的$\begin{bmatrix} X_c / Z_c \\ Y_c / Z_c \\ 1 \end{bmatrix}$是一步归一化，将3D点投影到图像平面，所以
            - $\mathbf{p} = \begin{bmatrix} f_x && 0 &&c_x \\ 0 && f_y && c_y \\ 0 && 0 && 1 \end{bmatrix}\begin{bmatrix} X_c / Z_c \\ Y_c / Z_c \\ 1 \end{bmatrix} = \begin{bmatrix}
            f_x \cdot \frac{X_c}{Z_c} + c_x \\
            f_y \cdot \frac{Y_c}{Z_c} + c_y \\
            1
            \end{bmatrix}$
        
    - 外参和内参相乘得到投影矩阵：
        
        首先，将内参矩阵 K 扩展为 4x4 的齐次坐标形式：
        
        $K_{\text{homogeneous}} = \begin{bmatrix}
        f_x & 0 & c_x & 0 \\
        0 & f_y & c_y & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1
        \end{bmatrix}$
        
        和外参相乘，得到转换矩阵：
        
        - $K[R|T] = \begin{bmatrix}
        f_x & 0 & c_x & 0 \\
        0 & f_y & c_y & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1
        \end{bmatrix}\begin{bmatrix} r_{11} && r_{12} && r_{13} && t_1 \\ r_{21} && r_{22} && r_{23} && t_2 \\ r_{31} && r_{32} && r_{33} && t_3 \\ 0 && 0 && 0 && 1 \end{bmatrix}$
        
        最终投影到 2D 像素坐标时，需要归一化（除以深度）
        

**对应代码详细解析：**

dataset中的参数读取：(mmdet3d源代码：https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/nuscenes_converter.py#L146)

处理投影矩阵最主要的是_fill_trainval_infos方法，其中：

```python

        # 获得Lidar的sample_data
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        # 获得Lidar信息
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        # 获得 ego pose 信息
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmengine.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'num_features': 5,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            # lidar到ego的平移矩阵， [x, y, z]格式
            'lidar2ego_translation': cs_record['translation'],
            # lidar到ego的旋转矩阵，4元数格式
            'lidar2ego_rotation': cs_record['rotation'],
            # ego到global的平移矩阵，[x, y, z]格式
            'ego2global_translation': pose_record['translation'],
            # ego到global的旋转矩阵，4元数格式
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }
        
        # 例如： 
        # {'token': 'f4d2a6c281f34a7eb8bb033d82321f79',
			  # 'sensor_token': '47fcd48f71d75e0da5c8c1704a9bfe0a',
				# 'translation': [3.412, 0.0, 0.5],
			  # 'rotation': [0.9999984769132877, 0.0, 0.0, 0.0017453283658983088],
				# 'camera_intrinsic': []}
				#
	      # 将rotation转换为矩阵形式：
        # [[ 0.99999391 -0.00349065  0.        ]
        # [ 0.00349065  0.99999391  0.        ]
        # [ 0.          0.          1.        ]]

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        # 调用四元数库，计算得到3x3的旋转矩阵
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            # 获得相机内参： 3x3矩阵
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            ######################################
            #计算投影矩阵
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            ######################################
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                # 循环获得非关键帧
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
```

可以看到上面的部分代码中已经获得了：

- `l2e_r_mat`, `l2e_t`：Lidar到ego的旋转和平移矩阵
- `e2g_r_mat`, `e2g_t`：ego到global的旋转和平移矩阵
- `cam_intrinsic`：相机内参
- `cam_info`：主要是cam外参相关的信息，并且在这里**计算了cam到lidar的投影矩阵**

`cam_info`具体是通过`obtain_sensor2top`方法实现的(源码链接https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/nuscenes_converter.py#L283)，部分重要的代码如下：

```python
    # 获得当前Camera的sample_data
    sd_rec = nusc.get('sample_data', sensor_token)
    # 获得camera sensor的信息
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    # 获得当前的ego pose
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        # camera到ego的平移矩阵， [x, y, z]格式
        'sensor2ego_translation': cs_record['translation'],
        # camera到ego的旋转矩阵， 4元数格式
        'sensor2ego_rotation': cs_record['rotation'],
        # ego到global的平移矩阵， [x, y, z]格式
        'ego2global_translation': pose_record['translation'],
        # ego到global的平移矩阵， [x, y, z]格式
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

		#######################################################
		#################！！！重点部分！！！#####################
		#######################################################
    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    
    # 将4元数转换为3x3矩阵形式
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
	  # 获得从camear到lidar的旋转矩阵
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    # 获得从camear到lidar的平移矩阵    
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep
```

详细说明`sensor2lidar_rotation`和`sensor2lidar_tranlsation`的计算过程：

- 行向量和列向量表示点
    
    **一个是 R*P  + T 一个是 P*R +T   两者可以互相转换，因为R是正交阵，所以R*P+T = P*R‘+T**
    
    如果是列向量， 通常使用 R * P + T , 例如  3x3 * 3x1 + 3x1
    
    如果是行向量，可以使用P*R +T,例如 1x3 * 3x3 + 1x3
    
    如果是Tensor，一般是  (N,3)，那就是行向量，所以要注意转置计算
    
1. `sensor2lidar_rotation`：
    1. **坐标系转换**的过程是： 
        1. camera→ego：从相机坐标系（相机光心为原点，光心向前为z轴，向下为y轴，向右为x轴）转换到主车坐标系（主车后轴中心为原点，向上是z轴，向前是x轴，向左是y轴）
        2. ego→global: 从主车坐标系转换到世界坐标系（这里是地图坐标系，地图左上角为原点，向右为x轴，向下为y轴，向上为z轴）
        3. global→ego’：从世界坐标系转换到lidar的ego坐标（这里是lidar时间戳对齐的ego，可能与camear的ego不同，所以需要转换到global再转回ego）
        4. ego’ →lidar：从ego坐标系转换到lidar坐标系（lidar安装位置为原点，向右是x轴，向前是y轴，向上是z轴）
    2. 注意：我们的目标是**给定一个lidar坐标系下的3D点，通过投影，得到其在某个camera上的2D位置，所以计算投影矩阵是和上面的顺序反过来的**
    3. 也就是说，假设我们有一个点 $\mathbf{p}_{lidar}$，要计算这个投影过程：
        1. lidar→ego:   $\mathbf{p}_{ego} = \mathbf{R}_{l2e} \mathbf{p}_{lidar} +\mathbf{T}_{l2e}$
        2. ego→global:$\mathbf{p}_{global} = \mathbf{R}_{e2g} \mathbf{p}_{ego} +\mathbf{T}_{e2g}$
        3. global→ego:$\mathbf{p}_{ego'} = \mathbf{R}_{e2g'}^{-1} \mathbf{p}_{global} -\mathbf{R}_{e2g'}^{-1} \mathbf{T}_{e2g'}$
        4. ego’ → lidar:  $\mathbf{p}_{cam} = \mathbf{R}_{l2e'}^{-1} \mathbf{p}_{ego'} -\mathbf{R}_{l2e'}^{-1} \mathbf{T}_{l2e'}$
    4. 旋转矩阵计算：
        1. lidar→ego:   就是乘以 l2e_r_mat，代码中的`np.linalg.inv(l2e_r_mat).T` 其实不变，因为旋转矩阵是正交阵，所以$R^T = R^{-1}$，那么，$(R^{-1})^T = R$
        2. ego → global: 就是乘以e2g_r_mat，代码中是：`np.linalg.inv(e2g_r_mat).T` ，其实也是e2g_r_mat本身
        3. global →ego: 就是乘以 e2g_r_s_mat的逆，因为eg2_r_s_mat是ego到global，要反向投影需要取矩阵的逆，（因为是正交阵，所以逆等于转置）代码中是： `e2g_r_s_mat.T`
        4. ego→camera: 乘以 l2e_r_s_mat的逆，同理，代码中是：`l2e_r_s_mat.T`
        5. 计算顺序：   `(l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)` 
        6. 和公式对应：
            
            $$
             \mathbf{p}_{cam} = \mathbf{R}_{l2e'}^{-1} \cdot(\mathbf{R}_{e2g'}^{-1} \mathbf{p}_{global} -\mathbf{R}_{e2g'}^{-1} \mathbf{T}_{e2g'} ）-\mathbf{R}_{l2e'}^{-1} \mathbf{T}_{l2e'}
            $$
            
            $$
             \mathbf{p}_{cam} = \mathbf{R}_{l2e'}^{-1} \cdot(\mathbf{R}_{e2g'}^{-1} \cdot (\mathbf{R}_{e2g} \mathbf{p}_{ego} +\mathbf{T}_{e2g})-\mathbf{R}_{e2g'}^{-1} \mathbf{T}_{e2g'} ）-\mathbf{R}_{l2e'}^{-1} \mathbf{T}_{l2e'}
            $$
            
            $$
             \mathbf{p}_{cam} = \mathbf{R}_{l2e'}^{-1} \cdot(\mathbf{R}_{e2g'}^{-1} \cdot (\mathbf{R}_{e2g} \cdot(\mathbf{R}_{l2e} \mathbf{p}_{lidar} +\mathbf{T}_{l2e}) +\mathbf{T}_{e2g})-\mathbf{R}_{e2g'}^{-1} \mathbf{T}_{e2g'} ）-\mathbf{R}_{l2e'}^{-1} \mathbf{T}_{l2e'}
            $$
            
            $$
             \mathbf{p}_{cam} = \mathbf{R}_{l2e'}^{-1} \cdot(\mathbf{R}_{e2g'}^{-1}\cdot  (\mathbf{R}_{e2g} \mathbf{R}_{l2e}\cdot \mathbf{p}_{lidar} +\mathbf{R}_{e2g}\cdot\mathbf{T}_{l2e} +\mathbf{T}_{e2g})-\mathbf{R}_{e2g'}^{-1} \mathbf{T}_{e2g'} ）-\mathbf{R}_{l2e'}^{-1} \mathbf{T}_{l2e'}
            $$
            
            $$
             \mathbf{p}_{cam} = (\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1}) \cdot (\mathbf{R}_{e2g} \mathbf{R}_{l2e}\cdot \mathbf{p}_{lidar} +\mathbf{R}_{e2g}\cdot\mathbf{T}_{l2e} +\mathbf{T}_{e2g})-(\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1} )\cdot\mathbf{T}_{e2g'} -\mathbf{R}_{l2e'}^{-1} \mathbf{T}_{l2e'}
            $$
            
            最终可得：
            
            $$
             \mathbf{p}_{cam} = (\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1}) \cdot (\mathbf{R}_{e2g} \mathbf{R}_{l2e})\cdot \mathbf{p}_{lidar} +(\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1}) \cdot(\mathbf{R}_{e2g}\cdot\mathbf{T}_{l2e} +\mathbf{T}_{e2g})-((\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1}) \mathbf{T}_{e2g'} +\mathbf{R}_{l2e'}^{-1} \mathbf{T}_{l2e'})
            $$
            
            可以将上面的计算分为**旋转和平移：**
            
            - 旋转部分的整体投影矩阵：
                - 公式： $(\mathbf{R}_{l2e'}^{-1} \cdot\mathbf{R}_{e2g'}^{-1}) \cdot (\mathbf{R}_{e2g} \cdot\mathbf{R}_{l2e})$
                - 代码：  `R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)`
            - 平移部分：
                - 公式：$(\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1}) \cdot(\mathbf{R}_{e2g}\cdot\mathbf{T}_{l2e} +\mathbf{T}_{e2g})-((\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1}) \mathbf{T}_{e2g'} +\mathbf{R}_{l2e'}^{-1} \mathbf{T}_{l2e'})$
                - 因为T是行向量，所以公式可以写成：
                - $(\mathbf{T}_{l2e}\mathbf{R}_{e2g}^{-1}+\mathbf{T}_{e2g})\cdot(\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1})^{-1}-( \mathbf{T}_{e2g'}(\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1})^{-1} +\mathbf{T}_{l2e'} (\mathbf{R}_{l2e'}^{-1})^{-1} )$
                    - 这是因为：
                        - e2g_t_s是一个行向量，公式中是列向量，R本身是正交阵，所以，$(\mathbf{R}_{e2g}\cdot\mathbf{T}_{l2e} +\mathbf{T}_{e2g})=(\mathbf{T}_{l2e}^T\mathbf{R}_{e2g}^T\cdot\ +\mathbf{T}_{e2g}^T)$，其中T上面的转置表示从列向量转为行向量。同理可推出其他部分。
                        - 又由于 $(AB)^{-1} = B^{-1}A^{-1}$，所以$(\mathbf{R}_{l2e'}^{-1} \mathbf{R}_{e2g'}^{-1})^{-1} = (\mathbf{R}_{e2g'}^{-1})^{-1}(\mathbf{R}_{l2e'}^{-1})^{-1}$
                - 代码：
                    - `T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)`
                        - 对应公式：$(\mathbf{T}_{l2e}\mathbf{R}_{e2g}^{-1}+\mathbf{T}_{e2g})\cdot(\mathbf{R}_{e2g'}^{-1})^{-1}(\mathbf{R}_{l2e'}^{-1})^{-1}$
                    - `T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T`
                        - 对应公式：$-(\space\space \mathbf{T}_{e2g'}( \mathbf{R}_{e2g'}^{-1})^{-1}(\mathbf{R}_{l2e'}^{-1})^{-1} +\mathbf{T}_{l2e'} (\mathbf{R}_{l2e'}^{-1})^{-1}\space\space)$
    - 经过上面的计算，我们获得了：
        - **sensor2lidar的 rotation 和 translation矩阵：**表示从各个camera到lidar的投影矩阵
    - 接下来计算**lidar到image（像素坐标系）的投影**：
        - 坐标系转换的顺序是：  **lidar→camear → image(pixel)**
        - 投影公式（这里按照p作为行向量表示）：
        - $\mathbf{p}_{cam} = \mathbf{R}^{-1} \cdot\mathbf{p}_{lidar} +\mathbf{R}^{-1}\mathbf{T}$， 其中**$R$**是sensor2lidar，所以$R^{-1}$表示lidar2sensor
        - $\mathbf{p}_{img}^h = \mathbf{K}^h[\mathbf{(R}^{-1})^h| (\mathbf{R}^{-1}\mathbf{T})^h]\cdot\mathbf{p}_{lidar}^h$，其中$h$表示齐次坐标
        - **对应代码里的操作：**
            - **读取投影矩阵和平移矩阵：**
                - `lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])`
                    - lidar2cam_r = $\mathbf{R}^{-1}$
                - `lidar2cam_t = np.matmul(cam_info['sensor2lidar_translation'], lidar2cam_r.T)`
                    - $\mathbf{T}\cdot(\mathbf{R}^{-1})^{-1} = \mathbf{R}^{-1}\mathbf{T}^{-1}$，其中T的转置表示从列向量变为行向量，这一步代码中是按照T为行向量来计算的，对应公式中的列向量表示，计算是等价的。
            - **齐次坐标表示，合并旋转矩阵和平移矩阵：**
                - `idar2cam_rt = np.eye(4)`
                    - $[R|T]_{\text{homogeneous}} = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1
                    \end{bmatrix}$
                - `lidar2cam_rt[:3, :3] = lidar2cam_r.T`
                    - $[R|T]_{\text{homogeneous}} = \begin{bmatrix}
                    r_{11} & r_{12} & r_{13} & 0 \\
                    r_{21} & r_{22} & r_{23} & 0 \\
                    r_{31} & r_{32} & r_{33} & 0 \\
                    0 & 0 & 0 & 1
                    \end{bmatrix}$
                - `lidar2cam_rt[3, :3] = lidar2cam_t`
                    - $[R|T]_{\text{homogeneous}} = \begin{bmatrix}
                    r_{11} & r_{12} & r_{13} & t_1 \\
                    r_{21} & r_{22} & r_{23} & t_2 \\
                    r_{31} & r_{32} & r_{33} & t_3 \\
                    0 & 0 & 0 & 1
                    \end{bmatrix}$
            - **内参读取和投影矩阵合并：**
                - `intrinsic = cam_info['cam_intrinsic']`
                - `viewpad = np.eye(4)`
                    - $K_{\text{homogeneous}} = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1
                    \end{bmatrix}$
                - `viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic`
                    - $K_{\text{homogeneous}} = \begin{bmatrix}
                    f_x & 0 & c_x & 0 \\
                    0 & f_y & c_y & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1
                    \end{bmatrix}$
                - `lidar2img_rt = np.matmul(viewpad, lidar2cam_rt.T)`
                    - $R_{\text{homogeneous}}  = K_{\text{homogeneous}} \cdot[R|T]^{T}_{\text{homogeneous}}$
                    - 这里的转置是因为：
                        - 在读取数据的时候，保存的sensor2lidar_rotation取了一次转置，用来将 R*P转换为P*R的方式计算，代码是：`pointsweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T`
                        - 但是我们在后面使用的时候是使用R*P的方式，所以需要将其转换回来 ，根据 $(AB)^{-1} = B^{-1}A^{-1}$，其中A表示点的列向量形式，A逆表示点的行向量形式，B表示我们的转换矩阵R。具体使用这个方式代码如下(detr_transformer.py 中的 feature_sampling方法里)：
                            - `# [bs, n_cam, 1, 4, 4] * [bs, n_cam, n_query, 4, 1]`
                            - `reference_points_cam = paddle.matmul(lidar2img, reference_points)`
                            - 可以看到，点是行向量形式[…, 4, 1]，所以lidar2img需要在我们乘以内参之前进行转置。

**完整代码：**

```python
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
```

### Dataloader实现：

这里使用基础的Dataloader即可：

```python
dataloader = paddle.io.DataLoader(dataset, batch_size=1)
```

## 图像特征提取：

### ResNet：

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20DETR3D%20(1)%20-%20%E7%8E%AF%E8%A7%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E3%80%81%E5%9D%90%E6%A0%87%E7%B3%BB%E8%BD%AC%E6%8D%A2%E5%92%8C%E5%9B%BE%E5%83%8F%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%201bd3135fac17807e9638c9fbc27115a9/image%201.png)

**1. ResNet50的整体结构：ResNet类**

```python
class ResNet50(nn.Layer):
    def __init__(self):
        super().__init__()
        # stem
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = _make_layer(out_ch=64, blocks=3, stride=1)
        self.layer2 = _make_layer(out_ch=128, blocks=4, stride=2)
        self.layer3 = _make_layer(out_ch=256, blocks=6, stride=2)
        self.layer4 = _make_layer(out_ch=512, blocks=3, stride=2)
        
    def forward(self):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # res layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
```

- Stem层由conv-bn-relu-maxpool组成
- Residual Blocks包含在4个不同的block layers里，每个layer包含若干个block，这些block通过一个_make_layer方法来实现，在_make_layer方法中，会根据设置创建多个BasicBlock对象。
- Forward方法就是顺序执行这些layer，然后返回最后一层的feature map。

**2. ResNet50的Residual结构：BasicBlock类**

```python
class BasicBlock(nn.Layer):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2D(in_ch,
							                 out_ch,
							                 kernel_size=3,
							                 stride=stride,
							                 padding=1)
			  self.bn1 = nn.BatchNorm2D(out_ch)
			  self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_ch,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2D(out_ch)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        h = self.downsample(h) if self.downsample is not None else h
        x = x + h
        x = self.relu(x)
        return x
```

- BasicBlock主要实现的是Residual Block，主要的成员包括：
    - conv1：一个3x3卷积，stride可以是1或者2（根据网络设置取不同的值），stride=2时，feature map的大小会下采样2倍；out_ch控制feature map的channel数。
    - conv2：一个3x3卷积，stride=1，feature map大小不变，channels数也不变
    - downsample：一个1x1卷积，主要是在Residual bypass保持tensor维度一致（最后才能相加），stride和channel数都和conv1一致。这里为了方便支持多种ResNet结构，downsample放在了BasicBlock之外定义，然后以参数的形式传进下来。
    - Forward方法：分主线和bypass分别计算，bypass需要判断，如果没有downsample，那直接与主线结果相加并返回，如果有downsample，需要先计算downsample然后才能相加。

**3. 创建ResBlock：_make_layer方法**

```python
def _make_layer(self, out_ch, blocks, stride):
    downsample = None
    if stride != 1:
        downsample = nn.Sequential(
            nn.Conv2D(self.in_ch, out_ch, kernel_size=1, stride=stride),
            nn.BatchNorm2D(out_ch))
    layer_list = []
    layer_lsit.append(BasicBlock(self.in_ch, out_ch, stride, downsample))
    self.in_ch = out_ch
    for _ in range(1, blocks):
        layer_list.append(BasicBlock(self.in_ch, out_ch))
    layer = nn.Sequential(*layer_list)
    return layer
```

- 根据stride创建downsample，主要用于Residual bypass，作为参数传入BasicBlock。
- 第一个block单独处理，因为：
    - 第一个block可能会遇到以下的情况：
        - 输入的feature map的channel数与当前不一致，那么就需要downsample的Conv2D设置对应的channel数，这样在Residual相加的时候才能保证两个tensor维度一致。
        - stride == 2，block的第一个conv2D需要设置stride=2，同时需要downsample的conv2D也设置stride=2，这样才能保证feature map下采样2倍，Residual的feature map也下采样2倍
- 第二个block开始，循环加入到list中
- 所有的block由nn.Sequential串起来

**4. 多尺度图像特征提取：ResNet50Feature类**

```python
class Resnet50Feature(nn.LayerDict)：
    def __init__(self, resnet, return_layers):
        orig_return_layers = return_layers
        return_layers = dict(return_layers.items())
        layers = OrderedDict()
        for name, module in resnet.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__(layers)
        self.return_layers = orig_return_layers
    
    def forward(self, x, mask):
        features = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                features[out_name] = x
        out = []
        for feat_name, feat_map in features.items():
            mask = nn.functional.interpolate(mask[None].astype('float32'),
                                             size=feat_map.shape[-2:])[0]
            out.append((feat_name, feat_map, mask))
        return out
```

这个Resnet50Feature类的主要作用是从ResNet中，按照输入的layer names提取一层或多层feature maps，结果存到list中并返回。

- 主要实现通过nn.LayerDict来实现，LayerDict可以通过OrderedDict来创建，并且会按顺序注册每个子层
- 通常需要输出的feature map来自：
    - layer1 - layer4的输出
- 因此我们需要从ResNet中拿到的层有（例如return_layers = [”layer2”, “layer3”, “layer4”]）：
    - Stem层，保存到OrderedDict中，key是layer name
    - layer1 到 layerN （N是我们需要的最后的Layer名字，这里是4），保存到OrderedDict中，key是layer name
    - layerN 之后的所有层不需要，可以丢掉（这里是avepool层，fc层等）
    - 同时需要记录输出层的名字（ [”layer2”, “layer3”, “layer4”] 保存为成员变量）
- Forward方法：
    - 对于每一个named_children()，进行推理（调用forward方法）
    - 如果是要返回的层，那么记录结果
    - 返回结果
    - named_children() 和 named_sublayers()的区别：
        
        named_children: 返回下一层的子层
        
        named_sublayers: 递归返回所有子层，包含每一层子层的子层
        

### DCNPack

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20DETR3D%20(1)%20-%20%E7%8E%AF%E8%A7%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E3%80%81%E5%9D%90%E6%A0%87%E7%B3%BB%E8%BD%AC%E6%8D%A2%E5%92%8C%E5%9B%BE%E5%83%8F%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%201bd3135fac17807e9638c9fbc27115a9/image%202.png)

在DETR3D的backbone实现中，使用了DCNv2算子，这个算子使用可变形卷积操作，用来替换网络中的一部分Conv3x3操作，用来提升模型的表达能力（简单说就是使用这个操作效果会更好）。DCNv2的原理超出本文范围，一句话概括就是类似可变形注意力机制的概念，不使用固定的卷积window，而是动态的采样一些offset，并根据采样位置计算feature map。在实现的时候，需要注意源码中使用的DCNv2操作，Paddle 自带的paddle.vision.ops.deform_conv2d操作并不是对应的DCNv2，所以需要进一步自定义这个类，并增加offset的计算（通过一个卷积），具体实现如下。

- 有关DCN的技术细节可参考以下两篇论文：
    
    [1] Dai, Jifeng, et al. "Deformable convolutional networks." *Proceedings of the IEEE international conference on computer vision*. 2017.
    
    [2] Zhu, Xizhou, et al. "Deformable convnets v2: More deformable, better results." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2019.
    

Paddle版本的DCNPack实现：

```python
class DCNPack(nn.Layer):
    """ This is DCNv2 """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias_attr=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.bias_attr=bias_attr

        self.conv_offset = paddle.nn.Conv2D(
           self.in_channels,
           self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
           kernel_size=self.kernel_size,
           stride=self.stride,
           padding=self.padding,
           dilation=self.dilation,
           bias_attr=True)

        self.weight = paddle.create_parameter(shape=[out_channels, in_channels // self.groups,
                         *self.kernel_size], dtype='float32')

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = paddle.chunk(out, 3, axis=1)
        offset = paddle.concat((o1, o2), axis=1)
        mask = paddle.nn.functional.sigmoid(mask)
        out = paddle.vision.ops.deform_conv2d(x,
                                              offset,
                                              self.weight,
                                              mask=mask,
                                              bias=None,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation,
                                              deformable_groups=self.deform_groups,
                                              groups=self.groups)
```

### Feature & FPN：

DETR3D使用了FPN进一步提取并生成多层图像特征，具体来说FPN主要是通过上采样的方式，将各层的特征依次融合，最终输出的融合后的多层特征。模型结构如上图所示。

具体实现的时候：

1. Forward方法的输入：list，多层图像特征，如上图中的Layer2-Layer4的输出。
2. FPN支持自定义特征层，也就是支持选择从输入特征的哪一层开始，哪一层结束。
3. 特征层分为： 
    1. Backbone特征层： 从backbone的每层特征经过lateral_conv，然后与下一层经过lateral_conv的特征相加，再经过fpn_conv层得到输出特征
    2. Extra特征层：只从最后一个backbone特征层，经过（可选relu）extra_fpn_conv(stride=2的Conv2D)，直接得到输出特征

![image.png](%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20DETR3D%20(1)%20-%20%E7%8E%AF%E8%A7%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E3%80%81%E5%9D%90%E6%A0%87%E7%B3%BB%E8%BD%AC%E6%8D%A2%E5%92%8C%E5%9B%BE%E5%83%8F%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%201bd3135fac17807e9638c9fbc27115a9/image%203.png)

完整代码：

```python
class FPN(nn.Layer):
    """Feature Pyramid Network"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral

        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
        self.start_level = start_level
        self.end_level = end_level

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Sequential(
                    nn.Conv2D(in_channels[i], out_channels,1, 1, 0),
                    #nn.BatchNorm2D(out_channels, out_channels),
                    #nn.ReLU())
                    )
            fpn_conv = nn.Sequential(
                    nn.Conv2D(out_channels, out_channels, 3, 1, 1),
                    #nn.BatchNorm2D(out_channels, out_channels),
                    #nn.ReLU())
                    )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = self.num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs is True and extra_levels >=1:
            for i in range(extra_levels):
                extra_fpn_conv = nn.Sequential(
                    nn.Conv2D(out_channels, out_channels, 3, 2, 1),
                    #nn.BatchNorm2D(out_channels, out_channels),
                    #nn.ReLU())
                    )
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, x):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(x[i + self.start_level]))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels -1 , 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] += F.interpolate(laterals[i], size=prev_shape)

        outs = []
        for i in range(used_backbone_levels):
            outs.append(self.fpn_convs[i](laterals[i]))

        if self.num_outs > len(outs):
            # extra levels
            if not self.add_extra_convs:  # use max pool
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:  # add conv layers
                extra_source = outs[-1]
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
```