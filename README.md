# ICT2019-Damaging-Traffic-Signs-Detection

In order to make the pipeline easier there is a `datasets.py` file.

`GTSRB_Damaged` pytorch dataset class. It returns Image Tensor, Traffic sign label, Damage label

```python
from torchvision import transforms
from torch.utils.data.dataset import random_split
import datasets

data = datasets.GTSRB_Damaged('./GTSRB/Final_Training/images')

test_size = int(len(dataset) * 0.2)
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataset.dataset.transform = transforms.Compose([
    transforms.Resize(64),
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
```
