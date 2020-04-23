# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

opj = os.path.join
ROOT_DATASET = '/ssd4/zhangyiyang/data/'


def return_ucf101(modality):
    filename_categories = 'UCF101/labels/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/' \
            'ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/'\
            'ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, \
        filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, \
        filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/'\
            '20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, \
        filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    dataset_path = opj(ROOT_DATASET, 'something-something-v2')
    category_file = opj(dataset_path, 'category.txt')
    if modality == 'RGB':
        img_format = '{:06d}.jpg'
        frames_path = opj(dataset_path, '20bn-something-something-v2-frames')
        train_filename = opj(dataset_path, 'train_videofolder.txt')
        val_filename = opj(dataset_path, 'val_videofolder.txt')
    elif modality == 'Flow':
        img_format = '{:06d}.jpg'
        frames_path = opj(dataset_path, '20bn-something-something-v2-flow')
        train_filename = opj(dataset_path, 'train_videofolder_flow.txt')
        val_filename = opj(dataset_path, 'val_videofolder_flow.txt')
    else:
        raise NotImplementedError('no such modality:'+modality)
    return category_file, train_filename, val_filename, \
        frames_path, img_format


def return_jester(modality):
    dataset_path = opj(ROOT_DATASET, 'jester-v1')
    category_file = opj(dataset_path, 'category.txt')
    if modality == 'RGB':
        img_format = '{:05d}.jpg'
        frames_path = opj(dataset_path, '20bn-jester-v1')
        train_filename = opj(dataset_path, 'train_videofolder.txt')
        val_filename = opj(dataset_path, 'val_videofolder.txt')
    else:
        raise NotImplementedError('no such modality:'+modality)
    return category_file, train_filename, val_filename, \
        frames_path, img_format


def return_ar(modality):
    dataset_path = opj(ROOT_DATASET, 'AR')
    category_file = opj(dataset_path, 'category.txt')
    if modality == 'RGB':
        img_format = '{:06d}.jpg'
        frames_path = opj(dataset_path, 'frames')
        train_filename = opj(dataset_path, 'train_samples.txt')
        val_filename = opj(dataset_path, 'val_samples.txt')
    else:
        raise NotImplementedError('no such modality:'+modality)
    return category_file, train_filename, val_filename, \
        frames_path, img_format


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, \
        filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester,
                   'ar': return_ar,
                   'something': return_something,
                   'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101,
                   'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics}
    if dataset in dict_single:
        file_categories, \
            train_filename, val_filename, \
            frames_path, img_format = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    # train/val list file absolute filepath
    train_samples_file = os.path.join(ROOT_DATASET, train_filename)
    val_samples_file = os.path.join(ROOT_DATASET, val_filename)

    # get num_classes
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    num_classes = len(categories)
    print('{}: {} classes'.format(dataset, num_classes))

    return num_classes, \
        train_samples_file, val_samples_file, \
        frames_path, img_format
