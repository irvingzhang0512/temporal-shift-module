import argparse
import pandas as pd
import numpy as np
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv-file", type=str,
                        default="/hdd01/zhangyiyang/data/ava/annotations/ava_train_v2.2.csv")
    parser.add_argument("--exclude-train", action="store_true")
    parser.add_argument("--val-csv-file", type=str,
                        default="/hdd01/zhangyiyang/data/ava/annotations/ava_val_v2.2.csv")
    parser.add_argument("--exclude-val", action="store_true")

    parser.add_argument("--action-list-file", type=str,
                        default="/hdd01/zhangyiyang/data/ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clip-radium-secs", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=50)
    parser.add_argument("--source-frames-dir",
                        type=str, default="/hdd01/zhangyiyang/data/ava/frames")
    parser.add_argument("--target-frames-dir",
                        type=str, default="/hdd01/zhangyiyang/data/ava/clips")
    parser.add_argument("--source-img-format", type=str,
                        default="%s_{:06d}.jpg")
    parser.add_argument("--target-img-format", type=str,
                        default="{:05d}.jpg")
    parser.add_argument("--clear", action="store_true")

    return parser.parse_args()


def _read_labelmap(labelmap_file):
    labelmap = []
    name = ""
    class_id = ""
    with open(labelmap_file, "r") as f:
        for line in f:
            if line.startswith("  name:"):
                name = line.split('"')[1]
            elif line.startswith("  id:") or line.startswith("  label_id:"):
                class_id = int(line.strip().split(" ")[-1])
                labelmap.append({"id": class_id, "name": name})
    return labelmap


def _get_start_end_pairs(label_timestamps, radius=2):
    results = np.zeros((901), np.int8)
    for t in label_timestamps:
        t -= 900
        s = max(0, t-radius)
        e = min(900, t+radius)
        results[s:e+1] = 1
    s_e_pairs = []
    start = -1
    for idx, t in enumerate(results):
        if t == 1 and start == -1:
            start = idx
        elif start != -1 and t == 0:
            s_e_pairs.append((start, idx))
            start = -1
    return s_e_pairs


def _copy_files(start_frame_idx,
                end_frame_idx,
                source_img_format,
                target_img_format,
                source_frames_dir,
                target_frames_dir
                ):
    for target_idx, source_idx in enumerate(
            range(start_frame_idx, end_frame_idx), 1):
        source_img_name = source_img_format.format(source_idx)
        target_img_name = target_img_format.format(target_idx)
        shutil.copy(
            os.path.join(source_frames_dir, source_img_name),
            os.path.join(target_frames_dir, target_img_name)
        )


def _handle_single_sample(video_name, cur_pair, fps,
                          source_img_format, source_frames_dir,
                          target_img_format, target_base_frames_dir,
                          ):
    start_frame_idx = cur_pair[0] * fps + 1
    end_frame_idx = cur_pair[1] * fps + 1

    # 构建当前clip所要复制到的文件路径
    cur_target_frames_dir = os.path.join(
        target_base_frames_dir,
        "_".join([video_name,
                  str(start_frame_idx),
                  str(end_frame_idx)])
    )
    if not os.path.exists(cur_target_frames_dir):
        os.makedirs(cur_target_frames_dir)
    else:
        return

    # 复制帧文件
    for target_idx, source_idx in enumerate(
            range(start_frame_idx, end_frame_idx), 1):
        source_img_name = source_img_format.format(source_idx)
        target_img_name = target_img_format.format(target_idx)
        shutil.copy(
            os.path.join(source_frames_dir, source_img_name),
            os.path.join(cur_target_frames_dir, target_img_name)
        )


def _handle_single_category(category_id, category_name,
                            train_video_list, train_df,
                            val_video_list, val_df,
                            args):
    t1 = time.time()
    print("start handling {}".format(category_name))

    sample_cnt = 0
    cur_base_target_frames_dir = os.path.join(
        args.target_frames_dir, category_name
    )
    pool = ThreadPoolExecutor(args.num_threads)

    # 获取一个字典
    # key为 video_name
    # value为一个列表，每个元素是一个元组，包含起止时间 `[start, end)`
    video_name_to_s_e_pairs = {}
    for cur_video in val_video_list:
        cur_df = val_df[(val_df[6] == category_id)
                        & (val_df[0] == cur_video)]
        cur_s_e_pairs = _get_start_end_pairs(
            list(cur_df[1].unique()), args.clip_radium_secs)
        video_name_to_s_e_pairs[cur_video] = cur_s_e_pairs
        sample_cnt += len(cur_s_e_pairs)
    for cur_video in train_video_list:
        cur_df = train_df[(train_df[6] == category_id) &
                          (train_df[0] == cur_video)]
        cur_s_e_pairs = _get_start_end_pairs(
            list(cur_df[1].unique()), args.clip_radium_secs)
        video_name_to_s_e_pairs[cur_video] = cur_s_e_pairs
        sample_cnt += len(cur_s_e_pairs)

    print(category_name, sample_cnt, 'samples')
    # 复制帧
    for video_name in video_name_to_s_e_pairs.keys():
        # 构建原始帧所在路径 `/path/to/ava/frames`
        cur_source_frames_dir = os.path.join(
            args.source_frames_dir, video_name
        )

        # 构建原始帧名称结构
        # 结果形如 `videoname_{:06d}.jpg`
        cur_source_img_format = args.source_img_format % video_name

        # 分别获取每个sample
        for cur_pair in video_name_to_s_e_pairs[video_name]:
            _handle_single_sample(
                video_name, cur_pair, args.fps,
                cur_source_img_format, cur_source_frames_dir,
                args.target_img_format, cur_base_target_frames_dir,
            )
    pool.shutdown(wait=True)
    print("finish handling {} with {:.2f} seconds".format(
        category_name, (time.time() - t1)))


def main(args):
    if not os.path.exists(args.target_frames_dir):
        os.makedirs(args.target_frames_dir)

    if args.clear:
        print('start clearing {}'.format(args.target_frames_dir))
        for category in os.listdir(args.target_frames_dir):
            cur_category_path = os.path.join(args.target_frames_dir, category)
            for sample in os.listdir(cur_category_path):
                frame_cnt = len(os.listdir(
                    os.path.join(cur_category_path, sample)))
                if frame_cnt == 0 or frame_cnt % args.fps != 0:
                    shutil.rmtree(os.path.join(cur_category_path, sample))
        print('finish clearning {}'.format(args.target_frames_dir))

    # 读取原始文件
    print('start getting df and video list')
    train_df = None
    val_df = None
    train_video_list = []
    val_video_list = []
    assert not (args.exclude_train and args.exclude_val), \
        'exclude_train/exclude_val cannot be BOTH set.'
    if not args.exclude_train:
        train_df = pd.read_csv(args.train_csv_file, header=None)
        train_video_list = train_df[0].unique()
    if not args.exclude_val:
        val_df = pd.read_csv(args.val_csv_file, header=None)
        val_video_list = val_df[0].unique()

    # 判断所需的数据是否存在
    exsisting_video_list = [f for f in os.listdir(args.source_frames_dir)]
    for v in list(train_video_list) + list(val_video_list):
        if v not in exsisting_video_list:
            raise ValueError("Couldn't find source file {}".format(v))
    print('finish getting df and video list')

    label_map_list = _read_labelmap(args.action_list_file)
    for label_map in label_map_list:
        category_name = label_map['name'].replace(" ", "_").replace("/", "_")
        category_id = label_map['id']
        _handle_single_category(category_id, category_name,
                                train_video_list, train_df,
                                val_video_list, val_df,
                                args)


if __name__ == '__main__':
    main(_parse_args())
