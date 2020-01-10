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
    --gpus 0 1 2 3

```