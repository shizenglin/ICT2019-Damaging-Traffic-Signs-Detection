# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20

import argparse
import numpy as np
from tqdm import tqdm
from copy import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout

from model.resnet import ResNet18
from model.wide_resnet import WideResNet
import torch.utils.data as data
from focalloss import *

from gtsrb_dataloader import GTSRB, BAM

from imblearn.metrics import classification_report_imbalanced
#from imblearn.metrics import specificity_score
from sklearn.metrics import average_precision_score

import utils.custom_transforms

from datasets.datasets import BAM_MEANS, BAM_STDS, GTSRB_MEANS, GTSRB_STDS


model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10','gtsrb']
padding_options = ['pad', 'fill']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='gtsrb',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=8,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--subset', '-s', type=int, default=None,
                    help='use subset of data')
parser.add_argument('--im_size', type=int, default=32, help='image resize size')
parser.add_argument('--pad', '-p', default='pad', choices=padding_options)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.dataset + '_' + args.model

print(args)

gtsrb_path = './datasets/GTSRB/Final_Training/Images/'
bam_path = './datasets/BAM_data/'
convention_path = './datasets/BAM_data/convention_conversion.csv'

if args.dataset == 'gtsrb':
    dat_trn_mean = GTSRB_MEANS
    dat_trn_std = GTSRB_STDS
else:
    dat_trn_mean = BAM_MEANS
    dat_trn_std = BAM_STDS

train_transform = transforms.Compose([
    utils.custom_transforms.Resize(args.im_size, args.pad),
    transforms.RandomCrop(args.im_size, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(dat_trn_mean, dat_trn_std)
])

if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


test_transform = transforms.Compose([
    utils.custom_transforms.Resize(args.im_size, args.pad),
    transforms.CenterCrop(args.im_size),
    transforms.ToTensor(),
    transforms.Normalize(GTSRB_MEANS, GTSRB_STDS)
])

# create train/test for GTSRB
train_dataset = GTSRB(gtsrb_path, train_transform, train=True, test_size=0.0)#, size_filter=lambda x: x > (24, 24
#gtsrb_test = GTSRB(gtsrb_path, train_transform, train=False, size_filter=lambda x: x > (24, 24))

# create train/test for BAM
test_dataset = BAM(bam_path, conversion_table_path=convention_path, train=True, test_split=0.0, transform=test_transform)
#bam_test = BAM(bam_path, conversion_table_path=convention_path, train=False, transform=test_transform)

# combination
#train_dataset = torch.utils.data.dataset.ConcatDataset([gtsrb_train, gtsrb_test])
#test_dataset = torch.utils.data.dataset.ConcatDataset([bam_train, bam_test])

############### Data Loader (Input Pipeline)####################
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               pin_memory=True,
                               num_workers=2)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=2)

num_classes = 2
if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

cnn = cnn.cuda()
criterion = FocalLoss()#nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.dataset == 'gtsrb':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

filename = 'logs/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    pred_list = []
    pred_soft_list = []
    label_list = []
    label_soft_list = []
    for images, _, labels in loader:
        #images = images[0]
        #labels = labels[0]
        images = Variable(images, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        pred = cnn(images)
        pred_soft_list.append(pred.data)
        pred = torch.max(pred.data, 1)[1]
        pred_list.append(pred)
        
        total += labels.size(0)
        label_t = labels.data
        label_list.append(label_t)
        correct += (pred == label_t).sum()

    #val_acc = float(correct) / float(total)
    pred_list = torch.cat(pred_list,0)
    label_list = torch.cat(label_list,0)
    pred_soft_list = torch.cat(pred_soft_list,0)
    label_soft_list = torch.eye(2)[label_list]#(len(label_list.cpu()),2).scatter_(1,label_list.cpu(),1)
    
    #val_sensitivity = sensitivity_score(label_list.cpu(), pred_list.cpu())
    #val_specificity = specificity_score(label_list.cpu(), pred_list.cpu())
    val_map = average_precision_score(label_soft_list.cpu(), pred_soft_list.cpu())
    target_names = ['undamaged', 'damaged']
    val_classification_report = classification_report_imbalanced(label_list.cpu(), pred_list.cpu(),target_names=target_names)
    
    cnn.train()
    return val_classification_report,val_map


for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, _, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        #images = images[0]
        #labels = labels[0]
        images = Variable(images).cuda(async=True)
        labels = Variable(labels).cuda(async=True)

        cnn.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.data#[0]

        # Calculate running average of accuracy
        _, pred = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (pred == labels.data).sum()
        accuracy = float(correct) / float(total)

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    if epoch%10==0: 
        test_report_imbal,test_map = test(test_loader)
        tqdm.write(test_report_imbal+'test_map: %.3f\n'%test_map)
        #tqdm.write('test_acc: %.3f\n test_recall: %.3f\n test_roc_auc:%.3f\n test_sensi:%.3f\n test_speci:%.3f\n test_geo:%.3f\n test_bal:%.3f\n test_report:%.3f\n' % (test_acc, test_recall,test_roc_auc,test_sensi,test_speci,test_geo,test_bal,test_report))

    scheduler.step(epoch)

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_report_imbal)}
    csv_logger.writerow(row)

torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()
