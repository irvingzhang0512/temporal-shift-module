# Jester v1

## 
+ Step 1: generate labels by `tools/gen_label_sthv1.py`. something v1 & jester v1 share the same label format.


```

python main.py jester RGB \
    --arch resnet50 --num_segments 8 --consensus_type=avg \
    --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
    --batch-size 64 -j 16 --dropout 0.5 --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres --npb \
    --gpus 0 1 2

python main.py jester RGB \
    --arch mobilenetv2 --num_segments 8 --consensus_type=avg \
    --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
    --batch-size 128 -j 16 --dropout 0.5 --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres --npb \
    --online \
    --gpu_devices 0,1,2,3 --gpus 0 1 2 3

```

## 1. Jester v1 Dataset

### 1.1. Raw dataset
+ `label` dir:
  + `jester-v1-labels.csv`：所有分类名称，每行一个名称，共27行。
  + `jester-v1-train.csv`：训练集列表，每行代表一个训练样本。每行通过`;`划分为两部分，前一部分是训练样本编号，后一部分是分类名称。分类名称对应上一个文件。
  + `jester-v1-validation.csv`：验证集列表，结构与训练集相同。
  + `jester-v1-test.csv`：测试集列表，只有样本编号，没有标签。
+ `20bn-jester-v1` dir:
  + 一共包含148093个文件夹。
  + 每个文件夹中有若干图片，代表一个样本。
  + 每个文件夹中的图片数量不一定，但都是从`00001.jpg`开始编号。

### 1.2. after preprocessing
+ 生成 `category.txt`，包含所有分类名称，且分类名称降序排列。
+ 生成 `train_videofolder.txt` 和 `val_videofolder.txt` 两个文件。
  + 每一行代表一个样本。
  + 每个样本分为三个部分，通过空格分割，三个部分分别是`对应文件夹绝对路径`(字符串)、`图片数量`(整数)、`所属类别`(整数)。


### 1.3. prepare for training
+ 对于每个样本：如果样本中图片数量大于 `num_segments`，则将样本中的图片按顺序分为 `num_segments` 个子集，随机选择每个子集中的一张图片作为训练样本。
+ 数据预处理包括
  + `GroupMultiScaleCrop`：随机切片并resize
  + `Stack(roll=False)`：RGB还是BGR，如果是False就是RGB
  + `ToTorchFormatTensor(div=True)`：图片数值转换为`[0, 1]`，shape转换为 `[C, H, W]`