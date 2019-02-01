from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from torchvision import datasets, transforms

from Cutout.model.resnet import ResNet18

test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# load trained model
cnn = ResNet18(num_classes=2)
cnn.load_state_dict(torch.load('checkpoints/bam-train-gtsrb-test-64_resnet18_1_190.pt'))
cnn = cnn.cuda()
cnn.eval()

image_ = Image.open('datasets/BAM_data/images/03053.jpg')
for i in tqdm(range(1000)):
    image = test_transform(image_)
    test_loader = data.DataLoader(dataset=[image],
                              batch_size=1)
    for image in test_loader:
        image = Variable(image, volatile=True).cuda()
        pred = cnn(image)
        pred = nn.functional.softmax(pred,dim=1)
        conscore = pred.data[0,1].item()
#        print(conscore)
