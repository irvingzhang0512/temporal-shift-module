# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V2

import os
import json


DATASET_NAME = 'something-something-v2'
DATASET_PATH = "/hdd02/zhangyiyang/data/something-something-v2"
LABEL_PATH = os.path.join(DATASET_PATH, 'label')
IMAGE_PATH = os.path.join(DATASET_PATH, '20bn-something-something-v2-frames')


if __name__ == '__main__':
    # 读取类别，并保存到文件中
    with open(os.path.join(LABEL_PATH, '%s-labels.json' % DATASET_NAME)) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)
    with open(os.path.join(DATASET_PATH, 'category.txt'), 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = ['%s-validation.json' % DATASET_NAME,
                   '%s-train.json' % DATASET_NAME,
                   '%s-test.json' % DATASET_NAME]
    files_output = ['val_videofolder.txt',
                    'train_videofolder.txt',
                    'test_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(os.path.join(LABEL_PATH, filename_input)) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(
                    dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curFolder = os.path.join(IMAGE_PATH, curFolder)
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(curFolder)
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(os.path.join(DATASET_PATH, filename_output), 'w') as f:
            f.write('\n'.join(output))
