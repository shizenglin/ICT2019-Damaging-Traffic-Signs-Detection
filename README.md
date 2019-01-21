# ICT2019-Damaging-Traffic-Signs-Detection

In order to make the pipeline easier there is a `datasets.py` file.
Currently, it contains 2 main objects:
+ `GTSRB_Damaged` pytorch dataset class. It returns Image Tensor, Traffic sign label, Damage label
+ `split_ImageFolder` function for train / test split

```python
from torchvision import transforms
import datasets

data = datasets.GTSRB_Damaged('./GTSRB/Final_Training/images')
train_dataset, test_dataset = datasets.split_ImageFolder(data)

train_dataset.transform = transforms.Compose([
    transforms.Resize(64), 
    transforms.RandomAffine(10, [0.1, 0.1], [0.9, 1.1], 0.1),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor()
])

test_dataset.transform = transforms.Compose([
    transforms.Resize(64), 
    transforms.CenterCrop(64),
    transforms.ToTensor()
])
```
