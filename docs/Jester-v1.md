# Jester v1

## 1. Jester v1 Dataset

### 1.1. Raw dataset
+ Dataset Structured:
  + Root Directory: `jester-v1`
  + Labels Directory: `label`
    + `jester-v1-labels.csv`：所有分类名称，每行一个名称，共27行。
    + `jester-v1-train.csv`：训练集列表，每行代表一个训练样本。每行通过`;`划分为两部分，前一部分是训练样本编号，后一部分是分类名称。分类名称对应上一个文件。
    + `jester-v1-validation.csv`：验证集列表，结构与训练集相同。
    + `jester-v1-test.csv`：测试集列表，只有样本编号，没有标签。
  + Image Samples Directory: `20bn-jester-v1`
    + 一共包含148093个文件夹。
    + 每个文件夹中有若干图片，代表一个样本。
    + 每个文件夹中的图片数量不一定，但都是从`00001.jpg`开始编号。


### 1.2. after preprocessing
+ 在执行 `tools/gen_label_jester.py` 后会生成下面内容：
  + `category.txt`，包含所有分类名称，且分类名称降序排列。
  + `train_videofolder.txt` 和 `val_videofolder.txt` 两个文件。
    + 每一行代表一个样本。
    + 每个样本分为三个部分，通过空格分割，三个部分分别是`对应文件夹绝对路径`(字符串)、`图片数量`(整数)、`所属类别`(整数)。
