import os
import ntpath
import copy

from torchvision.datasets import ImageFolder

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def _get_all_csv_paths(dir):
    paths = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if path.split('.')[-1] == 'csv':
                paths.append(path)

    return paths


def _join_csv_into_dict_by_paths(paths):
    damage_dict = {}

    for csv_path in paths:
        csv_dir = os.path.dirname(csv_path)
        csv = pd.read_csv(csv_path)
        for _, row in csv.iterrows():
            image_path = os.path.join(csv_dir, row['Filename'])
            damage_dict[image_path] = row['Damage']
    return damage_dict


class GTSRB_Damaged(ImageFolder):

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        paths = _get_all_csv_paths(root)
        self.damage_dict = _join_csv_into_dict_by_paths(paths)

    def __getitem__(self, index):
        """
        Return:
            image_tensor, class_label, damage_label
        """
        image, label = super().__getitem__(index)
        image_path = self.samples[index][0]
        return image, label, self.damage_dict[image_path]


def split_ImageFolder(dataset, test_size=0.2, seed=42):
    """
    Return:
        train_dataset, test_dataset
    """
    np.random.seed(seed)
    samples = dataset.samples
    train, test = train_test_split(samples, test_size=test_size)
    train_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)

    train_dataset.samples = train
    train_dataset.imgs = train

    test_dataset.samples = test
    test_dataset.imgs = test
    return train_dataset, test_dataset
