# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from __future__ import division, print_function

import os
import subprocess
import sys
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

n_thread = 100


def vid2jpg(file_name, class_path, dst_class_path):
    """每个视频执行一次该函数"""
    if '.mp4' not in file_name:
        return
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, name)

    video_file_path = os.path.join(class_path, file_name)
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'img_00001.jpg')):
                subprocess.call(
                    'rm -r \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                print('*** convert has been done: {}'.format(dst_directory_path))
                return
        else:
            os.mkdir(dst_directory_path)
    except:
        print(dst_directory_path)
        return
    # 视频提取帧，resize为 (331, -1)，提取所有帧
    # cmd = 'ffmpeg -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 1 \"{}/img_%05d.jpg\"'.format(
    #     video_file_path, dst_directory_path)
    cmd = 'ffmpeg -i \"{}\" -threads 1 -q:v 0 \"{}/img_%05d.jpg\"'.format(
        video_file_path, dst_directory_path)
    # print(cmd)
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def class_process(dir_path, dst_dir_path, class_name):
    # 判断输入路径中的某一类别的子目录是否存在
    print('*' * 20, class_name, '*'*20)
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        print('*** is not a dir {}'.format(class_path))
        return

    # 构建输出路径中对应的子目录
    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    # 获取输入子目录下所有视频
    vid_list = os.listdir(class_path)
    vid_list.sort()

    # 多进程执行任务
    p = Pool(n_thread)
    worker = partial(vid2jpg,
                     class_path=class_path,
                     dst_class_path=dst_class_path)
    for _ in tqdm(p.imap_unordered(worker, vid_list), total=len(vid_list)):
        pass
    # p.map(worker, vid_list)
    p.close()
    p.join()

    print('\n')


if __name__ == "__main__":
    # 输入两个路径，分别是原始视频所在的路径，一个是目标存储数据的路径
    # 第一个路径中应该有很多子目录，每个目录代表一个分类
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    # 依次遍历输入路径中的每个子目录
    # 即依次遍历每个类型
    class_list = os.listdir(dir_path)
    class_list.sort()
    for class_name in class_list:
        class_process(dir_path, dst_dir_path, class_name)

    class_name = 'test'
    class_process(dir_path, dst_dir_path, class_name)
