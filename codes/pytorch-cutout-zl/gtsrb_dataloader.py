import os
from glob import glob
import random

from PIL import Image
import pandas as pd
import numpy as np

from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split


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


class GTSRB_Seq(Dataset):

    num_classes = 43
    images_per_sign = 30

    def __init__(self, root, transform=None, size_filter=None):
        super(GTSRB_Seq, self).__init__()
        root = os.path.expanduser(root)
        paths = glob(os.path.join(root, '**/*.csv'))
        damage_dict = _join_csv_into_dict_by_paths(paths)

        self.transform = transform
        self.sequences = []

        for class_idx in range(GTSRB_Seq.num_classes):
            dir_name = os.path.join(root, '{:05d}/*.ppm'.format(class_idx))
            all_filenames = glob(dir_name)
            filtered_prefixes = map(lambda x: x[:-9], all_filenames)
            num_seqs = len(set(filtered_prefixes))

            for seq_idx in range(num_seqs):
                seq = []
                for frame_idx in range(GTSRB_Seq.images_per_sign):
                    image_path = os.path.join(root, '{:05d}/{:05d}_{:05d}.ppm'.format(
                        class_idx, seq_idx, frame_idx
                    ))

                    if os.path.exists(image_path):
                        seq.append((image_path, class_idx, damage_dict[image_path]))

                assert len(set(s[1] for s in seq)) == 1
                assert len(set(s[2] for s in seq)) == 1
                images = [s[0] for s in seq]
                if size_filter:
                    images = filter(lambda x: size_filter(Image.open(x).size), images)
                    images = list(images)
                self.sequences.append((images, seq[0][1], seq[0][2]))

    def __getitem__(self, index):
        """
        Return:
            image_tensors, class_labels, damage_labels
        """
        seq = self.sequences[index]
        image_list, sign_class, damage_class = seq
        images = []
        for im in image_list:
            with Image.open(im) as img:
                images.append(img.convert('RGB'))

        if self.transform:
            images = [self.transform(img) for img in images]

        return images, [sign_class] * len(images), [damage_class] * len(images)

    def __len__(self):
        return len(self.sequences)

    def __repr__(self):
        str_ = name = self.__class__.__name__
        str_ += '\n\tSequences: {}'.format(len(self))
        str_ += '\n\tImages: {}'.format(sum([len(s[0]) for s in self.sequences]))
        str_ += '\n\tDamaged: {}'.format(sum([len(s[0]) * s[2] for s in self.sequences]))
        return str_


class FlattenSequences(Dataset):

    def __init__(self, sequence_dataset):
        super(FlattenSequences, self).__init__()
        self.sequence_dataset = sequence_dataset
        self.table = []
        for seq_idx, sequence in enumerate(self.sequence_dataset):
            length = len(sequence[0])
            self.table.extend([(seq_idx, i) for i in range(length)])

    def __getitem__(self, index):
        seq_idx, img_idx = self.table[index]
        sequence = self.sequence_dataset[seq_idx]
        return sequence[0][img_idx], sequence[1][img_idx], sequence[2][img_idx]

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        str_ = 'Flatten {}'.format(self.sequence_dataset.__class__.__name__)
        str_ += '\n\tImages: {}'.format(len(self))
        return str_


class BAM(Dataset):

    def __init__(self, bam_root, damage_types=['graffity'], fillna_class=False, use_stacked=False,
                 use_unknown_types=True, min_area=25 ** 2,
                 train=False, test_split=0.2, transform=None,
                 conversion_table_path='./data/BAM_data/convention_conversion.csv', seed=42):
        super(BAM, self).__init__()

        self.transform = transform
        self.class_names = pd.read_csv(conversion_table_path).set_index('NL')['DL'].dropna().astype(
            int).to_dict()

        self.bam_sequences = self._get_bam_sequences(bam_root, use_stacked, use_unknown_types,
                                                     min_area, damage_types, fillna_class)

        self.all_sequences = self.bam_sequences

        random.seed(seed)
        random.shuffle(self.all_sequences)

        if train:
            self.used_sequences = self.all_sequences[int(test_split * len(self.all_sequences)):]
            self.flattened_used_sequences = [image for sequence in self.used_sequences for image
                                             in sequence]
        else:
            self.used_sequences = self.all_sequences[:int(test_split * len(self.all_sequences))]
            self.flattened_used_sequences = [image for sequence in self.used_sequences for image
                                             in sequence]

    def _get_bam_sequences(self, bam_root, use_stacked, use_unknown_types, min_area, damage_types,
                           fillna_class):

        annotations = pd.read_csv('%s/annotations.csv'%bam_root)

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

        annotations = annotations[annotations['area'] > min_area]

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
        #print(self.used_sequences[0])
        image, sign, damage = self.flattened_used_sequences[index]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)

        return image, sign, damage

    def __len__(self):
        return len(self.used_sequences)
