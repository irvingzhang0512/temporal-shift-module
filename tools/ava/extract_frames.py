from __future__ import division, print_function

import os
import subprocess
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

n_thread = 100
img_name_format = "{}_%06d.jpg"
in_path = "/hdd01/zhangyiyang/data/ava/videos_15min"
out_path = "/hdd01/zhangyiyang/data/ava/frames"


def do_extract_frames(in_video_path, frames_dir):
    video_name = os.path.basename(in_video_path)
    video_name = video_name[:video_name.rfind(".")]

    out_video_dir = os.path.join(frames_dir, video_name)
    if os.path.exists(out_video_dir):
        return
    os.makedirs(out_video_dir)
    cur_img_format = img_name_format.format(video_name)
    cmd_format = ['ffmpeg',
                  '-i', '\"{}\"',
                  '-r 30 -q:v 1',
                  '"{}"']
    cmd = (' '.join(cmd_format)).format(
        in_video_path,
        os.path.join(out_video_dir, cur_img_format))
    print(cmd)
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    vid_list = [os.path.join(in_path, n) for n in os.listdir(in_path)]
    p = Pool(n_thread)
    worker = partial(do_extract_frames,
                     frames_dir=out_path)
    for _ in tqdm(p.imap_unordered(worker, vid_list), total=len(vid_list)):
        pass
    p.close()
    p.join()
