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
