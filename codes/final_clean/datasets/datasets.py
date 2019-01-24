import os
from glob import glob
import random

from PIL import Image
import pandas as pd
import numpy as np

from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split, KFold


def _join_csv_into_dict_by_paths(paths):
    damage_dict = {}

    for csv_path in paths:
        csv_dir = os.path.dirname(csv_path)
        csv = pd.read_csv(csv_path)
        for _, row in csv.iterrows():
            image_path = os.path.join(csv_dir, row['Filename'])
            damage_dict[image_path] = row['Damage']
    return damage_dict


class GTSRB(Dataset):

    def __init__(self, root, transform=None, train=True, size_filter=None, seed=42, test_size=0.2):
        super(GTSRB, self).__init__()
        root = os.path.expanduser(root)
        csv_paths = glob(os.path.join(root, '**/*.csv'))
        damage_dict = _join_csv_into_dict_by_paths(csv_paths)

        self.transform = transform
        self.train = train

        img_paths = sorted(glob(os.path.join(root, '**/*.ppm')))
        if size_filter:
            img_paths = filter(lambda x: size_filter(Image.open(x).size), img_paths)
            img_paths = list(img_paths)
        prefixes = list(set(p[:-10] for p in img_paths))

        np.random.seed(seed)
        train_prefixes, test_prefixes = train_test_split(prefixes, test_size=test_size)
        prefixes = train_prefixes if self.train else test_prefixes

        split_paths = []
        for p in prefixes:
            split_paths.extend(list(filter(lambda x: x.startswith(p), img_paths)))

        self.samples = []
        for p in split_paths:
            sign_class = int(p[-21:-16])
            damage_class = damage_dict[p]
            self.samples.append((p, sign_class, damage_class))

    def __getitem__(self, index):
        img, sign, damage = self.samples[index]

        with Image.open(img) as img:
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, sign, damage

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        str_ = self.__class__.__name__
        str_ += '\n\tSplit: {}'.format('train' if self.train else 'test')
        str_ += '\n\tImages: {}'.format(len(self))
        return str_


class BAM(Dataset):

    def __init__(self, bam_root, conversion_table_path, damage_types=['graffity'],
                 fillna_class=False, use_stacked=False,
                 use_unknown_types=True,
                 size_filter=None,
                 train=False, test_split=0.2, transform=None,
                 kfold_splits=None, kfold_flag=1):
        super(BAM, self).__init__()

        self.transform = transform
        self.train = train
        self.test_split = test_split
        self.class_names = pd.read_csv(conversion_table_path).set_index('NL')['DL'].dropna().astype(
            int).to_dict()

        self.bam_sequences = self._get_bam_sequences(bam_root, use_stacked, use_unknown_types,
                                                     damage_types, fillna_class)

        self.all_sequences = self.bam_sequences
    
        random.seed(42)
        random.shuffle(self.all_sequences)

        if kfold_splits:
            kf = KFold(n_splits=kfold_splits)

            split = list(kf.split(self.all_sequences))

            if self.train:
                self.used_sequences = [self.all_sequences[i] for i in split[kfold_flag][1]]

            else:
                self.used_sequences = [self.all_sequences[i] for i in split[kfold_flag][0]]

        else:

            if train:
                self.used_sequences = self.all_sequences[int(self.test_split * len(
                    self.all_sequences)):]

            else:
                self.used_sequences = self.all_sequences[:int(self.test_split * len(
                    self.all_sequences))]

        self.flattened_used_sequences = [image for sequence in self.used_sequences for image in
                                         sequence]

        if size_filter:
            filtered = filter(lambda x: size_filter(Image.open(x[0]).size),
                              self.flattened_used_sequences)
            self.flattened_used_sequences = list(filtered)

    def _get_bam_sequences(self, bam_root, use_stacked, use_unknown_types, damage_types,
                           fillna_class):

        annotations = pd.read_csv(f'{bam_root}/annotations.csv')

        if fillna_class:
            annotations['class'] = annotations['class'].fillna('undamaged')

        else:
            annotations = annotations.dropna(subset=['class'])

        annotations['signtype'] = annotations['signtype'].fillna('UNK')

        for v in annotations['signtype'].value_counts().index:
            if v not in self.class_names:
                self.class_names[v] = len(self.class_names)

        annotations['signtype'] = annotations['signtype'].replace(self.class_names).astype(int)

        if not use_unknown_types:
            annotations = annotations[annotations['signtype'] != 'UNK']

        if not use_stacked:
            annotations = annotations[annotations['stacked'] == 0]

        # annotations = annotations[annotations['area'] > min_area]

        #         sequences = annotations.groupby(['latitude', 'longitude', 'signtype'])['filename'].apply(list).tolist()
        #         classes = annotations.groupby(['latitude', 'longitude', 'signtype'])['signtype'].apply(list).tolist()
        annotations['damaged'] = 0

        annotations = annotations[annotations['class'].isin(damage_types + ['undamaged'])]

        annotations['damaged'][annotations['class'].isin(damage_types)] = 1

        assert annotations['damaged'].nunique() == 2

        sequences = annotations.groupby('coordinates_string')['filename'].apply(list)
        classes = annotations.groupby('coordinates_string')['signtype'].apply(list)
        damaged = annotations.groupby('coordinates_string')['damaged'].apply(list)

        pattern = os.path.join(bam_root, 'images', '{}')
        sequences = [[pattern.format(path) for path in seq] for seq in sequences]

        combined = [
            [(sequences[i][j], classes[i][j], damaged[i][j]) for j in range(len(sequences[i]))] for
            i in range(len(sequences))]

        return combined

    def __getitem__(self, index):
        """Return image, image label and damaged label."""
        image, sign, damage = self.flattened_used_sequences[index]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)

        return image, sign, damage

    def __len__(self):
        return len(self.flattened_used_sequences)
