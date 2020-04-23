
# 模型相关介绍

+ [模型相关介绍](#模型相关介绍)
  + [0. 前言](#0-前言)
  + [1. TSM训练模型](#1-tsm训练模型)
    + [1.1. 模型总体结构](#11-模型总体结构)
    + [1.2. Shift 模块](#12-shift-模块)
    + [1.3. 其他参数](#13-其他参数)
  + [2. 在线模型](#2-在线模型)

## 0. 前言
+ 目标：介绍本项目中模型相关内容。
+ 主要内容：
  + TSM训练模型，分别介绍总体结构、shift模块以及其他相关模块。
  + TSM在线模型。

## 1. TSM训练模型

### 1.1. 模型总体结构
+ 主要功能就是搭建 TSM 模型。
+ 源码就是在 `tsm/models/tsn.py` 中，关键在于 `TSN` 类。
+ 主要流程（`TSN`本身继承了 `torch.nn.Module`，基本过程都是在 `__init__` 中实现的）：
    + 设置一系列成员变量。
    + 通过 `_prepare_base_model` 初始化 base model。
    + 通过 `_prepare_tsn` 初始化整个模型。
    + 设置 consensus/softmax/partial_bn。
+ base model
  + 从源码上看，支持的 base model 包括resnet/BNInception/mobilenetv2。
  + resnet
      + 通过 `getattr(torchvision.models, base_model)` 获取，其中 `base_model` 就是名称，如 `resnet50/resnet101/resnet152` 等。
      + 调用了 torchvision 的基本结构后，需要设置 temporal shift module、non local module，并设置 `base_model.avgpool`以及`base_model.last_layer_name`。
  + mobilenetv2
      + 通过 `archs/mobilenetv2.py` 构建基本对象。
      + 之后设置了 temporal shift module、`base_model.avgpool`以及`base_model.last_layer_name`。
  + BNInception
      + 通过 `archs/bn_inception.py` 构建基本对象。
      + 之后设置了 temporal shift module 以及`base_model.last_layer_name`。
  + **设置 temporal shift module** 相关代码位于 `ops/temporal_shift.py` 中，主要就是有两种方式：
      + mobilenetv2/bninception 的做法是通过 `TemporalShift` 来替代普通卷基层。
      + resnet 的做法是调用 `make_temporal_shift`。
+ prepare tsn
  + 主要就是构建一些模型其他参数。
  + 之前在 `base_model` 定义中都有 `last_layer_name` 这个参数，这里会设置 `last_layer_name` 为 Dropout 层。
  + 定义了 `new_fc` 层作为 Dropout 后的全连接层。
+ 正向传导（`forward`）
  + 第一步：执行 `base_model` 获得输入图像的特征，并执行dropout/softmax，得到 base_out。
  + 第二步：reshape输出变量，将结果shape改为`(-1, num_segments, ...)`。
  + 第三步：执行 `consensus` 操作。
      + 相关操作都定义在 `basic_ops.py` 中。
      + 主要实现方式就是调用了 `SegmentConsensus`。
      + 实现的功能就是对指定的dim进行 `consensus_type` 操作，包括avg或identity。
      + `avg` 操作的主要功能就是对 input_tensor 进行 mean 操作，即 `input_tensor.mean(dim=SegmentConsensus.dim, keepdim=True)`。
      + 所以一般这一步操作就是对 `(-1, num_segments, ...)` 中的 `num_segments` 进行avg后squeeze。
  + 最终结果shape：`(-1, ...)`，其中 `...` 就代表 `base_model` 输出的shape。
  + 其他：先不考虑输入数据形式为 Flow/RGBDiff 的情况，只考虑输入数据形式为 RGB 的情况。

### 1.2. Shift 模块
+ 代码都在 `tsm/models/temporal_shift.py` 中。
+ 相关参数：
  + `--shift`
  + `--shift_div`
  + `--shift_place`
+ 功能：实现TSM的中最关键的 temporal shift。
  + 最核心的代码就是 `TemporalShift` 类。
  + 使用方法就是在构建 base net 的时候调用相关代码：
    + 在 mobilenetv2 中直接调用 `TemporalShift`。
    + 在 resnet 中调用 `make_temporal_shift`，而该函数的本质还是调用了 `TemporalShift`。
+ `TemporalShift`
  + 本质就是在进行某些网络操作之前，先对输入数据进行shift操作，而不改变网络操作本身。

### 1.3. 其他参数
+ non local
+ temporal pool
+ consensus type
+ partial bn

## 2. 在线模型
+ 源码就是在 `tsm/models/mobilenet_v2_tsm_online.py` 中，关键在于 `MobileNetV2` 类。
+ 在线模型与训练模型的比较：
  + backbone-MobilenetV2本身结构相同，只是定义方式不一样，参数名称有点区别。
  + 两者输入数据不同：
    + 训练模型的输入是`num_segments`张图片。
    + 在线模型的输入是1张图片以及保留下来的buffer。
      + 所谓buffer就是中间进行shift操作层的`1/8*num_segments`的特征图。
      + 比如，如果进行shift操作的输入特征图尺寸为 `[1, 32, 28, 28]`，那么对应的buffer的尺寸就是 `[1, 4, 28, 28]`。
  + 两者的输出不同：
    + 训练模型的输出就是分类结果。
    + 在线模型的输出是分类结果以及buffers。
  + 两者shift操作的方法不同。
    + 训练模型中的shift操作输入数据的 shape 形如 `[batch_size, num_segments, num_channels, h, w]`，而输入进行shift操作的是 `num_channels`。
    + 在线模型中的 shift 操作其实是对 `[1, num_channels, h, w]` 中的 `num_channels` 进行操作。
      + 大概思路就是，在 t 时刻，在需要进行shift操作的层中：
        + 保留前 `1/8 * num_channels` 的特征，作为buffer传递，等到 t+1 时刻使用。
        + 使用 t-1 时刻保存的 `1/8 * num_channels` 的特征以及当前时刻剩下的 `7/8 * num_segments` 作为输入，进行普通卷积操作。 
+ 在线模型并不需要单独训练，只需要把训练模型的参数转换一下，就可以直接用于在线模型。
  + 不严格地说，训练模型的实质就是 `num_segments` 张图片分别计算了一次分类结果，然后取平均。在线模型就是拿一张图片算了分类结果。
