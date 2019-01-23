import os
import ntpath
import copy
import glob

from PIL import Image
import pandas as pd

from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder


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
        super(GTSRB_Damaged, self).__init__(root=root,
                                            transform=transform,
                                            target_transform=target_transform)
        paths = _get_all_csv_paths(root)
        self.damage_dict = _join_csv_into_dict_by_paths(paths)

    def __getitem__(self, index):
        """
        Return:
            image_tensor, class_label, damage_label
        """
        image, label = super(GTSRB_Damaged, self).__getitem__(index)
        image_path = self.samples[index][0]
        return image, label, self.damage_dict[image_path]


class GTSRB_Seq(Dataset):

    num_classes = 43
    images_per_sign = 30

    def __init__(self, root, transform=None, size_filter=None):
        super(GTSRB_Seq, self).__init__()
        root = os.path.expanduser(root)
        paths = _get_all_csv_paths(root)
        damage_dict = _join_csv_into_dict_by_paths(paths)

        self.transform = transform
        self.sequences = []

        for class_idx in range(GTSRB_Seq.num_classes):
            dir_name = os.path.join(root, '{:05d}/*.ppm'.format(class_idx))
            all_filenames = glob.glob(dir_name)
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


import random


class GTSRB_BAM(torch.utils.data.Dataset):

    def __init__(self, gtsrb_root, bam_root,
                 conversion_table_path='./convention_conversion.csv',
                 train=False, test_split=0.2, transform=None):
        super(GTSRB_BAM, self).__init__()

        self.transform = transform
        self.class_names = pd.read_csv(conversion_table_path).set_index('NL')['DL'][:42].to_dict()

        self.bam_sequences = self._get_bam_sequences(bam_root)
        self.gtsrb_sequences = self._get_gtsrb_sequences(gtsrb_root)

        self.all_sequences = self.bam_sequences + self.gtsrb_sequences

        random.seed(42)
        random.shuffle(self.all_sequences)

        if train:
            self.used_sequences = self.all_sequences[int(test_split * len(self.all_sequences)):]

        else:
            self.used_sequences = self.all_sequences[:int(test_split * len(self.all_sequences))]

    def _get_bam_sequences(self, bam_root):

        annotations = pd.read_csv(f'{bam_root}/annotations.csv')

        for v in annotations['signtype'].value_counts().index:
            if v not in self.class_names:
                self.class_names[v] = len(self.class_names)

        annotations['signtype'].replace(self.class_names, inplace=True)

        #         sequences = annotations.groupby(['latitude', 'longitude', 'signtype'])['filename'].apply(list).tolist()
        #         classes = annotations.groupby(['latitude', 'longitude', 'signtype'])['signtype'].apply(list).tolist()

        annotations['class'] = annotations['class'].fillna('undamaged')

        damage_types = ['scratched', 'graffity', 'dirty']

        annotations['damaged'] = 0

        annotations['damaged'][annotations['class'].isin(damage_types)] = 1

        assert annotations['damaged'].nunique() == 2

        annotations['signtype'] = annotations['signtype'].fillna('UNK')

        sequences = annotations.groupby('coordinates_string')['filename'].apply(list)
        classes = annotations.groupby('coordinates_string')['signtype'].apply(list)
        damaged = annotations.groupby('coordinates_string')['damaged'].apply(list)

        sequences = [[f'{bam_root}/images/{path}' for path in seq] for seq in sequences]

        combined = [
            [(sequences[i][j], classes[i][j], damaged[i][j]) for j in range(len(sequences[i]))] for
            i in range(len(sequences))]

        return combined

    def _get_gtsrb_sequences(self, gtsrb_root, classes_to_use=range(43)):

        german_csvs = _get_all_csv_paths(gtsrb_root)

        german_csvs = [gcsv for gcsv in german_csvs if 'Test' not in gcsv]
        sequences = []
        classes = []
        damaged = []

        for c in classes_to_use:

            dir = (f"{gtsrb_root}/Final_Training/Images/{c:05d}")
            images = [f for f in os.listdir(dir) if f[-3:] == 'ppm']

            csv_path = f'{dir}/GT-{dir[-5:]}.csv'

            csv = pd.read_csv(csv_path)

            num_sequences = max(image.split('_')[0] for image in images)

            # Each sequence has 30 images.
            sequence = []
            class_id = []
            damage = []

            for i in range(int(num_sequences)):
                filename = f'{dir[-5:]}_{i:05d}.ppm'
                path = f'{dir}/{filename}'

                class_id.append(csv[csv['Filename'] == filename]['ClassId'])
                damage.append(csv[csv['Filename'] == filename]['Damage'])
                sequence.append(path)

            damaged.append(damage)
            classes.append(class_id)
            sequences.append(sequence)

        combined = [
            [(sequences[i][j], classes[i][j], damaged[i][j]) for j in range(len(sequences[i]))] for
            i in range(len(sequences))]

        return combined

    def __getitem__(self, index):

        """Return image, image label and damaged label."""

        seq = self.used_sequences[index]
        images = [Image.open(s[0]) for s in seq]
        if self.transform:
            images = [self.tranform(img) for img in images]
        return images, [s[1] for s in seq], [s[2] for s in seq]


    def __len__(self):
        return len(self.used_sequences)


class BAM(torch.utils.data.Dataset):

    def __init__(self, bam_root, conversion_table_path='./convention_conversion.csv',
                 train=False, test_split=0.2, transform=None):
        super(BAM, self).__init__()

        self.transform = transform
        self.class_names = pd.read_csv(conversion_table_path).set_index('NL')['DL'][:42].to_dict()

        self.bam_sequences = self._get_bam_sequences(bam_root)

        self.all_sequences = self.bam_sequences

        random.seed(42)
        random.shuffle(self.all_sequences)

        if train:
            self.used_sequences = self.all_sequences[int(test_split * len(self.all_sequences)):]

        else:
            self.used_sequences = self.all_sequences[:int(test_split * len(self.all_sequences))]

    def _get_bam_sequences(self, bam_root):

        annotations = pd.read_csv(f'{bam_root}/annotations.csv')

        for v in annotations['signtype'].value_counts().index:
            if v not in self.class_names:
                self.class_names[v] = len(self.class_names)

        annotations['signtype'].replace(self.class_names, inplace=True)

        #         sequences = annotations.groupby(['latitude', 'longitude', 'signtype'])['filename'].apply(list).tolist()
        #         classes = annotations.groupby(['latitude', 'longitude', 'signtype'])['signtype'].apply(list).tolist()

        annotations['class'] = annotations['class'].fillna('undamaged')

        damage_types = ['scratched', 'graffity', 'dirty']

        annotations['damaged'] = 0

        annotations['damaged'][annotations['class'].isin(damage_types)] = 1

        assert annotations['damaged'].nunique() == 2

        annotations['signtype'] = annotations['signtype'].fillna('UNK')

        sequences = annotations.groupby('coordinates_string')['filename'].apply(list)
        classes = annotations.groupby('coordinates_string')['signtype'].apply(list)
        damaged = annotations.groupby('coordinates_string')['damaged'].apply(list)

        sequences = [[f'{bam_root}/images/{path}' for path in seq] for seq in sequences]

        combined = [
            [(sequences[i][j], classes[i][j], damaged[i][j]) for j in range(len(sequences[i]))] for
            i in range(len(sequences))]

        return combined

    def __getitem__(self, index):

        """Return image, image label and damaged label."""

        seq = self.used_sequences[index]
        images = [Image.open(s[0]) for s in seq]
        if self.transform:
            images = [self.tranform(img) for img in images]

        return images, [s[1] for s in seq], [s[2] for s in seq]


    def __len__(self):
        return len(self.used_sequences)