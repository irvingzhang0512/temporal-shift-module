# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V1

import os

if __name__ == '__main__':
    dataset_name = 'something-something-v1'
    dataset_name = 'jester-v1'

    label_root_path = '/hdd02/zhangyiyang/data/%s/label' % dataset_name
    output_path = "/hdd02/zhangyiyang/data/%s" % dataset_name
    img_path = "/hdd02/zhangyiyang/data/%s/20bn-jester-v1" % dataset_name

    label_path = os.path.join(label_root_path, '%s-labels.csv' % dataset_name)
    category_path = os.path.join(output_path, 'category.txt')
    files_input = [
        os.path.join(label_root_path, '%s-validation.csv' % dataset_name),
        os.path.join(label_root_path, '%s-train.csv' % dataset_name)
    ]
    files_output = [
        os.path.join(output_path, 'val_videofolder.txt'),
        os.path.join(output_path, 'train_videofolder.txt'),
    ]

    with open(label_path) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    with open(category_path, 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            idx_categories.append(dict_categories[items[1]])
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_path = os.path.join(img_path, curFolder)
            dir_files = os.listdir(dir_path)
            output.append('%s %d %d' % (dir_path, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
