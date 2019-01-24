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

from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import brier_score_loss, cohen_kappa_score
from sklearn.metrics import jaccard_similarity_score, precision_score
from sklearn.metrics import recall_score, roc_auc_score

from imblearn.metrics import sensitivity_specificity_support
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import make_index_balanced_accuracy
from imblearn.metrics import classification_report_imbalanced

model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10','gtsrb']

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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.dataset + '_' + args.model

print(args)

gtsrb_path = './data/GTSRB/Final_Training/Images/'
bam_path = './data/BAM_data/'
convention_path = './data/BAM_data/convention_conversion.csv'

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
gtsrb_train = GTSRB(gtsrb_path, train_transform, train=True, size_filter=lambda x: x > (28, 28))
gtsrb_test = GTSRB(gtsrb_path, test_transform, train=False, size_filter=lambda x: x > (28, 28))

# create train/test for BAM
bam_train = BAM(bam_path, conversion_table_path=convention_path, train=True, transform=train_transform)
test_dataset = BAM(bam_path, conversion_table_path=convention_path, train=False, transform=test_transform)

# combination
train_dataset = torch.utils.data.dataset.ConcatDataset([gtsrb_train, gtsrb_test, bam_train])
#test_gtsrb_bam = torch.utils.data.dataset.ConcatDataset([gtsrb_test, bam_test])

##############Training on GTSRB Testing on BAM###############
#size_filter = lambda x: x > (28, 28)
#dataset_path = './data/GTSRB/Final_Training/Images'
#
#train_dataset = GTSRB_Seq(dataset_path, size_filter=size_filter)
#train_dataset.transform = transforms.Compose([
#    transforms.Resize(32),
#    transforms.RandomCrop(32, padding=4),
#    transforms.ToTensor(),
#    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#])
##if args.cutout:
##    train_dataset.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
##
#
#test_transform = transforms.Compose([
#    transforms.Resize(32),
#    transforms.CenterCrop(32),
#    transforms.ToTensor(),
#    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#])
#test_dataset = BAM('./data/BAM_data', train=True, test_split=0.0, transform=test_transform)


##############Training on BAM Testing on BAM###############
#train_transform = transforms.Compose([
#    transforms.Resize(32),
#    transforms.RandomCrop(32, padding=4),
#    transforms.ToTensor(),
#    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#])
#test_transform = transforms.Compose([
#    transforms.Resize(32),
#    transforms.CenterCrop(32),
#    transforms.ToTensor(),
#    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#])
#train_dataset = BAM('./data/BAM_data', train=True, test_split=0.0, transform=train_transform)
#test_dataset = BAM('./data/BAM_data', train=False, transform=test_transform)

##############Training on GTSRB+BAM Testing on BAM###############
#size_filter = lambda x: x > (28, 28)
#dataset_path = './data/GTSRB/Final_Training/Images'
#
#train_dataset_GTSRB = GTSRB_Seq(dataset_path, size_filter=size_filter)
#train_dataset_GTSRB.transform = transforms.Compose([
#    transforms.Resize(32),
#    transforms.RandomCrop(32, padding=4),
#    transforms.ToTensor(),
#    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#])

#train_transform = transforms.Compose([
#    transforms.Resize(32),
#    transforms.RandomCrop(32, padding=4),
#    transforms.ToTensor(),
#    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#])
#test_transform = transforms.Compose([
#    transforms.Resize(32),
#    transforms.CenterCrop(32),
#    transforms.ToTensor(),
#    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#])
#train_dataset = BAM('./data/BAM_data', train=True, test_split=0.0, transform=train_transform)
#test_dataset = BAM('./data/BAM_data', train=False, transform=test_transform)


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
    label_list = []
    for images, _, labels in loader:
        #images = images[0]
        #labels = labels[0]
        images = Variable(images, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        pred = cnn(images)
        pred = torch.max(pred.data, 1)[1]
        pred_list.append(pred)
        
        total += labels.size(0)
        label_t = labels.data
        label_list.append(label_t)
        correct += (pred == label_t).sum()

    #val_acc = float(correct) / float(total)
    pred_list = torch.cat(pred_list,0)
    label_list = torch.cat(label_list,0)
    
#    val_accuracy = accuracy_score(label_list.cpu(), pred_list.cpu())
#    val_recall = recall_score(label_list.cpu(),pred_list.cpu())
#    val_roc_auc = roc_auc_score(label_list.cpu(),pred_list.cpu())
#
#    val_sensitivity = sensitivity_score(label_list.cpu(), pred_list.cpu())
#    val_specificity = specificity_score(label_list.cpu(), pred_list.cpu())
#    val_geometric_mean = geometric_mean_score(label_list.cpu(), pred_list.cpu())
#    val_make_index_balanced = make_index_balanced_accuracy(label_list.cpu(), pred_list.cpu())
    target_names = ['undamaged', 'damaged']
    val_classification_report = classification_report_imbalanced(label_list.cpu(), pred_list.cpu(),target_names=target_names)
    
    cnn.train()
    return val_classification_report


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
        test_report_imbal = test(test_loader)
        tqdm.write(test_report_imbal)
        #tqdm.write('test_acc: %.3f\n test_recall: %.3f\n test_roc_auc:%.3f\n test_sensi:%.3f\n test_speci:%.3f\n test_geo:%.3f\n test_bal:%.3f\n test_report:%.3f\n' % (test_acc, test_recall,test_roc_auc,test_sensi,test_speci,test_geo,test_bal,test_report))

    scheduler.step(epoch)

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_report_imbal)}
    csv_logger.writerow(row)

torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()
