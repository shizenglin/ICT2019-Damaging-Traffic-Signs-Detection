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

#from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics.classification import precision_recall_fscore_support
from sklearn.metrics import average_precision_score


model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10','gtsrb']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--logname', '-d', default='bam-train-gtsrb-test-64',
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
parser.add_argument('--subfolder', type=int, default=1,
                    help='use subfolder of data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.logname + '_' + args.model+'_'+str(args.subfolder)

print(args)

gtsrb_path = './data/GTSRB/Final_Training/Images/'
bam_path = './data/BAM_data/'
convention_path = './data/BAM_data/convention_conversion.csv'

train_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


test_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# create train/test for GTSRB
train_dataset_gtsrb = GTSRB(gtsrb_path, train_transform, train=True, test_size=0.0)#, size_filter=lambda x: x > (24, 24
#gtsrb_test = GTSRB(gtsrb_path, train_transform, train=False, size_filter=lambda x: x > (24, 24))

# create train/test for BAM
train_dataset_bam = BAM(bam_path, conversion_table_path=convention_path, train=True, transform=train_transform, kfold_flag=args.subfolder)
test_dataset = BAM(bam_path, conversion_table_path=convention_path, train=False, transform=test_transform, kfold_flag=args.subfolder)

# combination
train_dataset = torch.utils.data.dataset.ConcatDataset([train_dataset_gtsrb, train_dataset_bam])
#test_dataset = torch.utils.data.dataset.ConcatDataset([bam_train, bam_test])

############### Data Loader (Input Pipeline)####################
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               pin_memory=True,
                               num_workers=4)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=4)

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

scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)

filename = 'logs/' + test_id + '.txt'
log_file = open(filename, "w")

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
        pred = nn.functional.softmax(pred,dim=1)
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
    
    test_precision, test_recall, _, _ = precision_recall_fscore_support(label_list.cpu(), pred_list.cpu())
    test_map = average_precision_score(label_soft_list.cpu(), pred_soft_list.cpu())
    #target_names = ['undamaged', 'damaged']
    #val_classification_report = classification_report_imbalanced(label_list.cpu(), pred_list.cpu(),target_names=target_names)
    
    cnn.train()
    return test_precision, test_recall, test_map



best_map = 0.0
best_epoch = 0
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
        test_precision, test_recall, test_map = test(test_loader)
        if test_map>best_map:
            best_map = test_map
            best_epoch = epoch
        torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '_' +str(epoch)+ '.pt')
        tqdm.write('precision: %.3f/%.3f, recall: %.3f/%.3f, map:%.3f, best_map:%.3f' % (test_precision[0],test_precision[1],test_recall[0],test_recall[1],test_map,best_map))
        log_file.write('precision: %.3f/%.3f, recall: %.3f/%.3f, map:%.3f, best_map:%.3f\n' % (test_precision[0],test_precision[1],test_recall[0],test_recall[1],test_map,best_map))

    scheduler.step(epoch)

log_file.close()
#torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
filename = 'logs/' + test_id + '_output.csv'
csv_logger = CSVLogger(args=args, fieldnames=['imgname', 'conscore', 'label'], filename=filename) 
best_mode_path = 'checkpoints/' + test_id + '_' +str(best_epoch)+ '.pt'

cnn = ResNet18(num_classes=num_classes)
cnn.load_state_dict(torch.load(best_mode_path))
cnn = cnn.cuda()
cnn.eval()

for images, image_path, labels in test_loader:
    images = Variable(images, volatile=True).cuda()
    labels = Variable(labels, volatile=True).cuda()

    pred = cnn(images)
    pred = nn.functional.softmax(pred,dim=1)
    row = {'imgname': image_path[0], 'conscore': str(pred.data[0,1].item()), 'label': str(labels.data[0].item())} 
    csv_logger.writerow(row)   
csv_logger.close()