# ICT2019-Damaging-Traffic-Signs-Detection

In order to make the pipeline easier there is a `datasets.py` file.

How to

```python
from copy import copy
from torch.utils.data.dataset import random_split
from torchvision import transforms
from datasets import GTSRB_Seq, FlattenSequences

# filter any sizes you want
size_filter = lambda x: x > (32, 32)
dataset_path = './GTSRB/Final_Training/Images'

dataset = GTSRB_Seq(dataset_path, size_filter=size_filter)
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# in order to apply different transformations to dataset
train_dataset.dataset = copy(dataset)

train_dataset.dataset.transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomAffine(10, [0.1, 0.1], [0.9, 1.1], 0.05),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset.dataset.transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = FlattenSequences(train_dataset)
test_dataset = FlattenSequences(test_dataset)
```
