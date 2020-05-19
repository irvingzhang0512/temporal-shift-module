# 记录一次训练/部署流程

## 0. 前言
+ 目标：记录一次在服务器训练模型、Jetbot部署模型的过程。
+ 总体步骤：
  + 第一步：通过 `tools/main.py` 训练模型，得到一个`离线模型`以及对应的 checkpoint 文件。
  + 第二步：通过 `tsm/utils/ckpt_transformation.py` 转换checkpoint文件结构。
  + 第三步：通过 `online_demo/jetbot/auto_tuning.py`，将pytorch模型转换为TVM形式，并优化。
  + 第四步：将前一步中得到的 graph/params/lib 文件保存到 Jetbot 上。
  + 第五步：在Jetbot本地运行 `online_demo/jetbot/jetbot_tvm_demo.py` 。


## 1. 训练脚本

### 1.1. Jester
```shell
# Jester resnet50
python tools/main.py jester RGB \
    --arch resnet50 --num_segments 8 --consensus_type=avg \
    --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
    --batch-size 64 -j 16 --dropout 0.5 --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres --npb \
    --gpu_devices 0,1,2,3 --gpus 0 1 2 3

# no shift
python tools/main.py jester RGB \
    --arch mobilenetv2 --num_segments 8 --consensus_type=avg \
    --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
    --batch-size 128 -j 16 --dropout 0.5 --eval-freq=1 \
    --npb \
    --gpu_devices 0,1,2,3 --gpus 0 1 2 3

# Jester mobilenet v2
python tools/main.py jester RGB \
    --arch mobilenetv2 --num_segments 8 --consensus_type=avg \
    --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
    --batch-size 128 -j 16 --dropout 0.5 --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres --npb \
    --logs_name shift1_4 \
    --gpu_devices 0,1,2,3 --gpus 0 1 2 3

```

### 1.2. AR

```shell
# AR mobilenet v2 training
python tools/main.py ar RGB \
    --arch mobilenetv2 --num_segments 8 \
    --gd 20 --wd 1e-4 --lr_steps 15 30 --epochs 50 \
    -j 16 --dropout 0.5 --eval-freq=1 --consensus_type=avg \
    --shift --shift_div=4 --shift_place=blockres --npb \
    --logs_name 5_9_dataset \
    --use_weighted_sampler --steps_per_epoch 200 --batch-size 32 --lr 0.005 \
    --gpu_devices 0,1

# AR resnet50 training
python tools/main.py ar RGB \
    --arch resnet50 --num_segments 16 \
    --gd 20 --wd 1e-4 --lr_steps 15 30 --epochs 50 \
    -j 16 --dropout 0.5 --eval-freq=1 --consensus_type=avg \
    --shift --shift_div=4 --shift_place=blockres --npb \
    --use_weighted_sampler --steps_per_epoch 200 --batch-size 32 --lr 0.005 \
    --gpu_devices 2,3 --logs_name 5_9_dataset
```