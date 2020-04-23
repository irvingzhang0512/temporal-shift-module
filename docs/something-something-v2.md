# Something-Something-V2

## 0. Overview

```shell
python main.py something RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
     --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb
```

## 1. Something Something V2 Dataset Structure

### 1.1. Raw Dataset
+ Video Directory `20bn-something-something-v2`
  + Inlucding 220848 webm videos.
  + Video format: `%d.webm`
+ Label Directory `label`
  + `something-something-v2-labels.json`：原始标签文件，是个dict，key是category name，value是category id，一共174类。
  + `something-something-v2-train.json`：
    + 168913个样本，是个list，每个元素代表一个训练样本.
    + 每个训练样本有 `id`（样本编号）/`label`（详细标签，不存在something而是具体物体）/`template`（粗略标签，包括`[something]`的标签）/`placeholders`（即标签中`something`的具体内容）。
  + `something-something-v2-validation.json`：
    + 24777个样本
    + 形式与训练集相同。
  + `something-something-v2-test.json`：形式与训练集相同。
    + 27157个样本。
    + 只有 `id` 没有其他标签。

### 1.2. After preprocessing
+ Scripts:
  + Step 1: `tools/vid2img_sthv2.py`
    + Generate frames from videos by ffmpeg.
    + One video file converts to one frame directory.
    + 183G.
  + Step2: `tools/gen_label_sthv2.py`
    + Generate Category file `category.txt`. One Label converts to one line.
    + Generate train/val/test sample files.
      + 每一行代表一个样本。
      + 每个样本分为三个部分，通过空格分割，三个部分分别是`对应文件夹绝对路径`(字符串)、`图片数量`(整数)、`所属类别`(整数)。