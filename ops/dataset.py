# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False,
                 dense_sample=False,
                 twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _parse_list(self):
        """初始化方法中调用"""
        # 获取 train_videofolder.txt 或 val_videofolder.txt 中的数据
        # 每一行一个样本，共有三个属性，通过空格分割
        # 三个属性分别是 绝对路径、图片数量、类别标签三个部分
        # check the frame number is large > 3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]

        # 构建 VideoRecord 列表
        # 该对象其实就是更方便的获取上述三个属性
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _load_image(self, directory, idx):
        """get函数中调用"""
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(
                    self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(
                    os.path.join(self.root_path, directory,
                                 self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1
                x_img = Image.open(
                    os.path.join(
                        self.root_path, '{:06d}'.format(int(directory)),
                        self.image_tmpl.format(int(directory),
                                               'x', idx))).convert('L')
                y_img = Image.open(
                    os.path.join(
                        self.root_path, '{:06d}'.format(int(directory)),
                        self.image_tmpl.format(int(directory),
                                               'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(
                        os.path.join(
                            self.root_path, directory,
                            self.image_tmpl.format(idx))).convert('RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory,
                                       self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(
                        self.root_path, directory,
                        self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with
                # (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _sample_indices(self, record):
        """
        训练集使用

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(
                0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) %
                       record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            # 则将图片分为 num_segments 个子集
            # 每个子集中随机选择一张图片
            average_duration = (record.num_frames -
                                self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(
                    list(range(self.num_segments)),
                    average_duration) + randint(average_duration,
                                                size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(
                    randint(record.num_frames - self.new_length + 1,
                            size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        """验证集中使用"""
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(
                0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) %
                       record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            # 验证集中，每次选择的样本都是固定的
            # 也是先将图片分为 num_segments 个部分，每次获取的都是中间的那个样本
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / \
                    float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x)
                                    for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        """测试集中使用"""
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [
                    (idx * t_stride + start_idx) %
                    record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / \
                float(self.num_segments)

            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                + [int(tick * x) for x in range(self.num_segments)]
            )

            return offsets + 1
        else:
            # 测试集中，获取方法与验证集相同
            # 每次获取的图片相同，获取每个子集中中间的哪张图片。
            tick = (record.num_frames - self.new_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
            return offsets + 1

    def get(self, record, indices):
        """
        根据下标获取样本中的若干图片，并进行数据预处理

        __getitem__ 中调用
        获取的最终结果是一个 [num_segments, ]
        """
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        # 获取第一张图片的绝对路径
        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(
                self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        # 如果当前样本不存在，则随机换一个样本
        while not os.path.exists(full_path):
            print('################## Not Found:', full_path)
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(
                    self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(
                    self.root_path,
                    '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(
                    self.root_path, record.path, file_name)

        # 选择每个样本中的若干张图片
        if not self.test_mode:
            segment_indices = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def __len__(self):
        return len(self.video_list)
