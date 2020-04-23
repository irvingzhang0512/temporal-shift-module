# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py

import os


dataset_path = '/ssd6/zhangyiyang/data/kinetics-400/images256/'
label_path = '/ssd5/zhangyiyang/data/kinetics-400/label'
label_map_path = '/ssd4/zhangyiyang/temporal-shift-module/tools/kinetics_label_map.txt'

if __name__ == '__main__':
    # 处理标签文件
    with open(label_map_path) as f:
        categories = f.readlines()
        categories = [c.strip().replace(' ', '_').replace('"', '').replace(
            '(', '').replace(')', '').replace("'", '') for c in categories]
    assert len(set(categories)) == 400
    dict_categories = {category: i for i, category in enumerate(categories)}
    print(dict_categories)

    files_input = ['val_256.csv', 'train_256.csv']
    files_output = ['val_videofolder.txt', 'train_videofolder.txt']
    files_folder = ['val_256', 'train_256']
    for (filename_input, filename_output, sub_folder) in \
            zip(files_input, files_output, files_folder):
        count_cat = {k: 0 for k in dict_categories.keys()}
        with open(os.path.join(label_path, filename_input)) as f:
            lines = f.readlines()[1:]
        folders = []
        idx_categories = []
        categories_list = []
        for line in lines:
            line = line.rstrip()
            items = line.split(',')
            cur_folder = items[1] + '_' + \
                ("%06d" % int(items[2])) + '_' + ("%06d" % int(items[3]))
            print(cur_folder)
            folders.append(cur_folder)
            this_catergory = items[0].replace(' ', '_').replace(
                '"', '').replace('(', '').replace(')', '').replace("'", '')
            categories_list.append(this_catergory)
            idx_categories.append(dict_categories[this_catergory])
            count_cat[this_catergory] += 1
        print(max(count_cat.values()))

        assert len(idx_categories) == len(folders)
        missing_folders = []
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            img_dir = os.path.join(dataset_path, sub_folder, categories_list[i], curFolder)
            if not os.path.exists(img_dir):
                missing_folders.append(img_dir)
                # print(missing_folders)
            else:
                dir_files = os.listdir(img_dir)
                output.append('%s %d %d' % (os.path.join(
                    categories_list[i], curFolder), len(dir_files), curIDX))
            print('%d/%d, missing %d' %
                  (i, len(folders), len(missing_folders)))
        with open(os.path.join(label_path, filename_output), 'w') as f:
            f.write('\n'.join(output))
        with open(os.path.join(label_path, 'missing_' + filename_output), 'w') as f:
            f.write('\n'.join(missing_folders))
