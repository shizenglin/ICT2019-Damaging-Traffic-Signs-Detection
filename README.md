# ICT2019-Damaging-Traffic-Signs-Detection

In order to make the pipeline easier there is a `datasets.py` file.

How to

```python
gtsrb_path = './GTSRB/Final_Training/Images/'
bam_path = './BAM/'
convention_path = './convention_conversion.csv'

train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# create train/test for GTSRB
gtsrb_train = datasets.GTSRB(gtsrb_path, train_transform, train=True, size_filter=lambda x: x > (32, 32))
gtsrb_test = datasets.GTSRB(gtsrb_path, test_transform, train=False, size_filter=lambda x: x > (32, 32))

# create train/test for BAM
bam_train = datasets.BAM(bam_path, conversion_table_path=convention_path, train=True, transform=train_transform)
bam_test = datasets.BAM(bam_path, conversion_table_path=convention_path, train=False, transform=test_transform)

# combination
train_gtsrb_bam = torch.utils.data.dataset.ConcatDataset([gtsrb_train, bam_train])
test_gtsrb_bam = torch.utils.data.dataset.ConcatDataset([gtsrb_test, bam_test])
```
