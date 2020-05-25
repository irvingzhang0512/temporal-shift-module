import os

osj = os.path.join

DATA_ROOT = "/ssd4/zhangyiyang/data"
DATASET_NAME = 'jester-v1'

if __name__ == '__main__':

    label_root_path = osj(DATA_ROOT, DATASET_NAME, 'label')
    output_path = osj(DATA_ROOT, DATASET_NAME)
    img_path = osj(DATA_ROOT, DATASET_NAME, '20bn-jester-v1')

    label_path = osj(label_root_path, '%s-labels.csv' % DATASET_NAME)
    category_path = osj(output_path, 'category.txt')
    files_input = [
        osj(label_root_path, '%s-validation.csv' % DATASET_NAME),
        osj(label_root_path, '%s-train.csv' % DATASET_NAME)
    ]
    files_output = [
        osj(output_path, 'val_videofolder.txt'),
        osj(output_path, 'train_videofolder.txt'),
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
            dir_path = osj(img_path, curFolder)
            dir_files = os.listdir(dir_path)
            output.append('%s %d %d' % (dir_path, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
