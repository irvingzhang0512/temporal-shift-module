"""
多进程剪切视频。
"""
from __future__ import division, print_function

import os
import subprocess
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

n_thread = 100
in_path = "/hdd01/zhangyiyang/data/ava/videos"
out_path = "/hdd01/zhangyiyang/data/ava/videos_15min"


def do_cut_video(in_video_path):
    out_video_path = os.path.join(out_path,
                                  os.path.basename(in_video_path))
    if os.path.exists(out_video_path):
        return
    cmd_format = ['ffmpeg',
                  '-ss 900 -t 901'
                  '-i', '\"{}\"',
                  '"{}"']
    cmd = (' '.join(cmd_format)).format(
        in_video_path, out_video_path)
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    vid_list = [os.path.join(in_path, n) for n in os.listdir(in_path)]
    p = Pool(n_thread)
    worker = partial(do_cut_video)
    for _ in tqdm(p.imap_unordered(worker, vid_list), total=len(vid_list)):
        pass
    p.close()
    p.join()
