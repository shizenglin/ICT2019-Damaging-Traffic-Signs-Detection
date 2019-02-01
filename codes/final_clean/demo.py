from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from torchvision import datasets, transforms

from Cutout.model.resnet import ResNet18

vis_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32)
])
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

#image_ = Image.open('datasets/BAM_data/images/03053.jpg')
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Display the frame
    vis = np.array(vis_transform(frame))
    vis = cv2.cvtColor(vis,cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Do inference
    frame = test_transform(frame)
    test_loader = data.DataLoader(dataset=[frame], batch_size=1)
    for image in test_loader:
        image = Variable(image, volatile=True).cuda()
        pred = cnn(image)
        pred = nn.functional.softmax(pred,dim=1)
        conscore = pred.data[0,1].item()
        print(conscore)
