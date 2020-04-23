# 数据相关介绍

+ [数据相关介绍](#数据相关介绍)
  + [0. 前言](#0-前言)
  + [1. 特定数据集预处理](#1-特定数据集预处理)
    + [1.1. 数据集预处理目标](#11-数据集预处理目标)
    + [1.2. 特定数据集](#12-特定数据集)
  + [2. 源码中数据集的构建](#2-源码中数据集的构建)
  + [3. 源码中模型所需的数据预处理](#3-源码中模型所需的数据预处理)

## 0. 前言
+ 目标：介绍本项目中数据相关内容。
+ 主要内容包括
  + 数据集以及相关预处理介绍。
  + 源码中数据集如何构建。
  + 源码中模型训练、预测所需的数据如何预处理。

## 1. 特定数据集预处理

### 1.1. 数据集预处理目标
+ 不管数据集原始形式如何，数据集预处理后的结果都相同。
+ 处理后的结果分为三类：
  + 帧文件夹：
    + 该文件夹中有若干子文件夹，每个子文件夹代表一个动作，子文件夹中有若干帧图片文件。
    + 同一数据集的帧文件名形式相同，如 Jester 的帧图片形式为 `{:05d}.jpg`。
  + 数据集罗列文件：
    + 训练/验证集分别对应两个不同的文件。
    + 不存在文件头，每行代表一个样本。
    + 每个样本对应三个部分，分别是`对应文件夹绝对路径`(字符串)、`图片数量`(整数)、`所属类别`(整数)。
    + 样本举例：`/ssd4/zhangyiyang/data/jester-v1/20bn-jester-v1/56557 37 12`。
  + 分类类别文件：
    + 一行代表一个标签名称。
    + 一般就叫 `category.txt`。

### 1.2. 特定数据集
  + [Jester](Jester-v1.md)
  + [Something Something V2](something-something-v2.md)
  + Kinetics 400
    + 大概流程：先执行 `tools/vid2img_kinetics.py` 提取帧，再通过 `gen_label_kinetics.py` 形成数据集罗列文件。
    + 目前困难：原始数据集下载完成，但提取帧的结果所占硬盘空间太多，目前服务器不满足要求。

## 2. 源码中数据集的构建
+ 实现功能：针对不同数据集构建 `torch.utils.data.Dataset` 对象。
+ 相关代码：
  + `tsm/dataset/dataset_config.py`：对不同数据集构建的参数。
  + `tsm/dataset/video_dataset.py`：通过数据集参数构建数据集。
+ 数据集参数（`dataset_config.py`）
  + 所有数据集返回的数据集参数都相同，即：
    + `num_classes`：类别数量，整数。
    + `train_samples_file`：训练数据集罗列文件绝对路径，字符串，对应 `1.1. 特定数据集预处理` 中的数据集罗列文件。
    + `val_samples_file`：验证数据集罗列文件绝对路径，字符串，对应 `1.1. 特定数据集预处理` 中的数据集罗列文件。
    + `frames_path`：帧所在文件夹绝对路径，字符串，对应 `1.1. 特定数据集预处理` 中的帧文件夹。
    + `img_format`：图片名称样式，字符串，如 `{:05d}.jpg`。
  + 通过 `def return_dataset(dataset, modality)` 方法，根据数据集名称、数据样式（RGB/Flow）返回数据集参数。
+ 数据集构建：
  + 相关代码：
    + `tsm/dataset/transforms.py` 中的 `VideoRecord` 与 `TSNDataset` 两个类。
  + `TSNDataset` 继承了 `torch.data.Dataset`，主要提供了两个功能：
      + 功能一：解析数据集元数据。
      + 功能二：根据TSM的要求构建输入数据。
  + 功能一：解析数据集元数据。
      + 具体内容：通过数据集配置文件，将每个样本定义为一个`VideoRecord`对象。
      + 通过 `_parse_list` 方法实现，在 `__init__` 中调用。
      + 基本流程：
          + 读取训练/验证/测试数据集的信息，即 `1.1.` 中的 `train_samples_file/val_samples_file` 文件信息。
          + 解析每一行数据，然后构建 `VideoRecord` 对象，每个对象都包括了 `path`（样本所在文件夹的绝对路径）、`num_frames`（样本包含图片数量）、`label`（行为标签）。
  + 功能二：构建构建TSM所需数据。
      + 具体内容：主要任务就是
      + 通过 `__getitem__` 实现。
      + 基本流程：
          + 获取样本中第一张图片的路径，判断该图片是否存在。
          + 如果不存在就随机换一个样本。
          + 如果样本存在，就选择样本中的若干图片下标。（重点一）
          + 通过图片下标获取图片。（重点二，在`1.3. 数据预处理`中介绍）
      + 如何挑选样本中的图片（重点一）
          + 通过 `_sample_indices` 获取训练样本中的图片，首先将图片按顺序分为 `num_segments` 个部分，然后**随机**选择其中一张图片。
          + 通过 `_get_val_indices` 获取验证样本中的图片，首先将图片分为 `num_segments` 个部分，每个部分选择**正中间**的图片。
          + 通过 `_get_test_indices` 获取测试样本中的图片，方式与验证集相同。


## 3. 源码中模型所需的数据预处理
+ 相关代码：
  + `tsm/dataset/video_dataset.py` 中 `TSNDataset` 类。
  + `tsm/models/tsn.py` 中 `TSN` 类。
  + `tools/main.py`：构建 transforms。
+ 流程：
  + 通过 `get` 方法实现。
  + 基本流程：通过 `_load_image` （调用了 `Image.open`）读取每张图片，然后通过 `transform` 进行数据预处理。
  + 默认 `transform` 对象是在 `main.py` 中定义的。
  + 验证集的数据预处理流程（即`transforms`中）的功能有：
    + `GroupScale`：所有图片resize到 `scale_size*scale_size`。
    + `GroupCenterCrop`：所有图片进行center crop，得到 `crop_size*crop_size`。
    + `Stack`：将`list()`图片Stack起来，成为一个ndarray对象。
    + `ToTorchFormatTensor`：将图像类型从 uint8 转换为 `[0, 1]`，且格式从 H x W x C 转换为 C x H x W。
  + 训练集的数据预处理流程（即`transforms`中）的功能有：
    + `GroupMultiScaleCrop`：crop+resize。
    + `GroupRandomHorizontalFlip`：根据输入数据选择是否进行水平镜像。
    + `Stack`：将`list()`图片Stack起来，成为一个ndarray对象。
    + `ToTorchFormatTensor`：将图像类型从 uint8 转换为 `[0, 1]`，且格式从 H x W x C 转换为 C x H x W。