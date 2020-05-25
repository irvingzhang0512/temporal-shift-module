# AVA Dataset


## 1. 基本情况介绍
+ 简介：AVA全称Atomic Visual Actions,是由Google出品的视频理解数据集，共80类行为。
+ AVA与Kinetcis对比（样本结构对比）
  + 对于Kinetics/Jester/SomethingSomething数据集：
    + 一般称为**行为识别数据集**。
    + 本质就是视频领域的基本分类任务，一个样本对应一个视频与一个标签。
    + 视频可能是帧文件夹也可能是视频文件。
  + 对于AVA数据集：
    + 一般称为**时空行为识别（Spatio-Temporal Action Localization）数据集**。
    + 本质就是目标（一般是人）检测+分类（行为识别），一个样本对应一帧图像+一个人物bbox+一个标签。
    + 同一帧中的同一bbox可能有多个标签。
    + 同一帧图像中可能有多个bbox。
+ 其他信息：
  + 视频来源：若干（400+）电影，截取第15-30分钟。
  + 打标签的时间维度：对所有视频的每一秒分别进行标注。
+ 数据打标签过程
  + 选择需要标注的行为。
    + 选择类别的准则：通用（Generality）、原子（Atomicity，即不可分）、完备（Exhaustivity）。
  + 获取需要标注的视频。
    + 找不同国家的电视/电影来标注，不使用卡通、黑白、低分辨率、比赛视频。
    + 大于30分钟的、发布超过1年、观看次数超过1000的视频。
    + 每个视频都取15-30分钟作为输入，这样标注起来有相同的结构。
  + 对视频中的人画框。
    + 先用算法标注，再手工修正。
    + 算法实现时注意要高recall，从而减少手工标注的工作量。
    + 手工标注的工作量大概是5%。
  + 对连续帧中同一人的bbox建立联系。
    + 先通过计算相似度自动实现，再通过手工标注精细化处理。
    + 自动标注的方法是，获取相邻两张图片中所有bbox的embedding，然后通过匈牙利算法获取最佳匹配。
  + 对人的行为打标签。
    + 行为标签是通过众包方法获取的，没有自动生成的。
    + 标签分为三类：
      + pose action：可以理解为单个人的行为，这是必须标注的。
      + person-object interactions：人与物体交互，可选内容。
      + person-person interactions：人与人交互，可选内容。
    + 从实际来看，标注出错是不可能避免的，所以使用了以下流程：
      + action proposal：
      + verification：


## 2. 实际处理

### 2.1. 数据获取
+ 感谢[大哥们](https://github.com/cvdfoundation/ava-dataset)提供了aws的下载链接
  + 可以用**迅雷**下载。
  + 视频与标签都能下载。


### 2.2. 数据预处理
+ 基本上就参考SlowFast中的文档，但提供的代码都是单进程的，速度太慢了……
+ 处理流程：
  + 第一步：下载视频与标签。
    + 视频都下载到 `/path/to/ava/videos` 中。
    + 标签（`ava_v2.2.zip`中包含的）都下载到 `/path/to/ava/annotations` 中。
    + 帧列表文件下载到 `/path/to/ava/frame_lists` 中，这个是直接下载的。
  + 第二步：剪切视频，将所有电影的第15-30分钟剪切下来，保存到 `/path/to/ava/videos/videos_15min` 中。
    + SlowFast里给的脚本是用但进程做的，花了24小时才处理完毕。
    + 所以我自己写了一个脚本（`./tools/ava/cut_videos.py`），用多进程剪视频，但还没运行改过，不知道要多久处理完。
  + 第三步：对获取的若干15分钟视频提取帧，帧都保存到 `/path/to/ava/frames` 文件夹中。
    + 对所有15min的视频都分别提取帧。
    + 看原始脚本也是单进程处理的，所以也写了个多进程脚本 `./tools/ava/extract_frames.py` 来处理这一步操作。
    + 每个视频对应 `/path/to/ava/frames` 中的一个文件夹，文件夹名称就是视频的名称（无后缀）。
    + 帧文件的命名规则是 `{video_name}_{:06d}.jpg`。
+ 文件夹结构：
```
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
|_ annotations
   |_ [official AVA annotation files]
   |_ ava_train_predicted_boxes.csv
   |_ ava_val_predicted_boxes.csv
```

## 3. 将AVA转换为Kinetics形式的数据集
+ 目标形式：一段视频对应一个标签。
+ 脚本：`./toos/ava_to_kinetics_format.py`。
+ 数据格式转换：
  + 之前已经提取帧完成，所以这一步每个sample对应视频的形式是帧文件夹。
  + 先从train/val的csv文件中获取样本。
    + 每次处理一种标签、一个视频文件。
    + 目标获取一个长度为900（15分钟，900秒）的数组，表示每一秒是否包含该类别的行为。
    + 获取数组之后，再获取连续的片段作为视频。
+ 注意事项：
  + 目标形式不管视频中有几个人，有几种动作。
  + 一定要多线程处理，主要操作就是复制图片文件，慢死了。

## 3. 类别列表

+ answer phone：接电话
+ bend/bow (at the waist)：弯腰，比如爬山时候弯着腰
+ brush teeth：刷牙
+ carry/hold (an object)：拿着一个东西
+ catch (an object)：
+ chop
+ climb (e.g., a mountain)
+ clink glass
+ close (e.g., a door, a box)：关门/盒子等。
+ cook
+ crawl
+ crouch/kneel
+ cut
+ dance
+ dig
+ dress/put on clothing
+ drink
+ drive (e.g., a car, a truck)
+ eat
+ enter
+ exit
+ extract
+ fall down
+ fight/hit (a person)
+ fishing
+ get up
+ give/serve (an object) to (a person)
+ grab (a person)
+ hand clap
+ hand shake
+ hand wave
+ hit (an object)
+ hug (a person)
+ jump/leap
+ kick (a person)
+ kick (an object)
+ kiss (a person)
+ lie/sleep
+ lift (a person)
+ lift/pick up
+ listen (e.g., to music)
+ listen to (a person)
+ martial art
+ open (e.g., a window, a car door)
+ paint
+ play board game
+ play musical instrument
+ play with kids
+ play with pets
+ point to (an object)
+ press
+ pull (an object)
+ push (an object)
+ push (another person)
+ put down
+ read
+ ride (e.g., a bike, a car, a horse)
+ row boat
+ run/jog
+ sail boat
+ shoot
+ shovel
+ sing to (e.g., self, a person, a group)
+ sit
+ smoke
+ stand
+ stir
+ swim
+ take (an object) from (a person)
+ take a photo
+ talk to (e.g., self, a person, a group)
+ text on/look at a cellphone
+ throw
+ touch (an object)
+ turn (e.g., a screwdriver)
+ walk
+ watch (a person)
+ watch (e.g., TV)
+ work on a computer
+ write