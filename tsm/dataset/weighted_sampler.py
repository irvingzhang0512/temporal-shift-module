import torch
import numpy as np


def get_sampler_weights(num_classes, category_weights,
                        train_file_path):
    assert num_classes == len(category_weights)
    with open(train_file_path, "r") as f:
        samples = [l.strip() for l in f.readlines()]
    category_ids = [int(l.split(" ")[2]) for l in samples]
    cnts = np.bincount(category_ids)
    assert num_classes == len(cnts)
    weights = list(1.0 / cnts * np.array(category_weights))
    return np.array([weights[i] for i in category_ids])


def get_weighted_sampler(num_classes,
                         category_weights,
                         train_file_path,
                         num_samples,
                         replacement=True,
                         ):
    weights = get_sampler_weights(
        num_classes, category_weights, train_file_path)
    return torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples,
        replacement=replacement,
    )
