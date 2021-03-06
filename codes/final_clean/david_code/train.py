from argparse import ArgumentParser
import os

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from torchvision.datasets import ImageFolder
from torchvision import transforms

from torchvision.models import resnet18

from misc import CSVLogger

import copy

from torch import nn

from tensorboardX import SummaryWriter

from B.datasets import GTSRB, BAM, BAM_MEANS, BAM_STDS, GSTRB_MEANS, GTSRB_STDS
from B import train_utils
from B import custom_transforms
from B import models

import timeit


def main():
    #########################################
    # arguments
    #########################################
    parser = ArgumentParser()
    parser.add_argument('--damage_weights', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--restore_model', type=str, default=None)
    parser.add_argument('--train_dataset', type=str, choices=['nl', 'de', 'de_nl'], required=True)
    parser.add_argument('--test_dataset', type=str, choices=['nl', 'nl_nl'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_sqrt', type=int, default=64)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--img_size', type=int, default=64)

    parser.add_argument('--cuda', type=eval, default=False, choices=[True, False])
    parser.add_argument('--frozen', type=eval, default=False, choices=[True, False])
    parser.add_argument('--model_dir', type=str, default='./saved_models')
    parser.add_argument('--tb_dir', type=str, default='./tb')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--bam_dir', type=str)
    parser.add_argument('--gtsrb_dir', type=str)
    parser.add_argument('--bam_conversion', type=str)

    args = parser.parse_args()

    print("Args:")
    for k, v in vars(args).items():
        print("  {}={}".format(k, v))
    print(flush=True)

    #########################################
    # Device
    #########################################
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device: {}'.format(device))
    print(flush=True)

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True


    #########################################
    # Data
    #########################################
    bam_train_transform = transforms.Compose([
        custom_transforms.Resize(args.img_size),
        transforms.RandomAffine(10, [0.1, 0.1], [0.9, 1.1], 0.1),
        transforms.RandomCrop(args.img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(BAM_MEANS, BAM_STDS)
    ])

    gtsrb_train_transform = transforms.Compose([
        custom_transforms.Resize(args.img_size),
        transforms.RandomAffine(10, [0.1, 0.1], [0.9, 1.1], 0.1),
        transforms.RandomCrop(args.img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(GSTRB_MEANS, GTSRB_STDS)
    ])

    bam_test_transform = transforms.Compose([
        custom_transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(BAM_MEANS, BAM_STDS)
    ])

    gtsrb_test_transform = transforms.Compose([
        custom_transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(GSTRB_MEANS, GTSRB_STDS)
    ])


    #########################################
    # Train Dataset
    #########################################
    def _size_filter(size):
        W, H = size
        return (W * H)**0.5 >= args.min_sqrt


    if args.train_dataset == 'nl':
        train_dataset = BAM(args.bam_dir, args.bam_conversion,
                            size_filter=_size_filter, transform=bam_train_transform, train=True)

    if args.train_dataset == 'de':
        train_1 = GTSRB(args.gtsrb_dir, transform=gtsrb_train_transform,
                        train=True, size_filter=_size_filter)
        train_2 = GTSRB(args.gtsrb_dir, transform=gtsrb_train_transform,
                        train=False, size_filter=_size_filter)
        train_dataset = ConcatDataset([train_1, train_2])


    if args.train_dataset == 'de_nl':
        train_1 = GTSRB(args.gtsrb_dir, transform=gtsrb_train_transform,
                        train=True, size_filter=_size_filter)
        train_2 = GTSRB(args.gtsrb_dir, transform=gtsrb_train_transform,
                        train=False, size_filter=_size_filter)
        train_3 = BAM(args.bam_dir, args.bam_conversion,
                      size_filter=_size_filter, transform=bam_train_transform, train=True)
        train_dataset = ConcatDataset([train_1, train_2, train_3])


    print('Train Dataset', train_dataset)
    print(train_dataset)
    print(len(train_dataset))


    #########################################
    # Test Dataset
    #########################################
    if args.test_dataset == 'nl':
        test_dataset = BAM(args.bam_dir, args.bam_conversion,
                           size_filter=_size_filter, transform=bam_test_transform, train=False)

    if args.test_dataset == 'nl_nl':
        test_1 = BAM(args.bam_dir, args.bam_conversion,
                     size_filter=_size_filter, transform=bam_test_transform, train=True)
        test_2 = BAM(args.bam_dir, args.bam_conversion,
                     size_filter=_size_filter, transform=bam_test_transform, train=False)
        test_dataset = ConcatDataset([test_1, test_2])

    print('Test Dataset', test_dataset)
    print(len(test_dataset))
    print(flush=True)

    train_loader = DataLoader(train_dataset, args.batch_size, True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, args.batch_size, False, pin_memory=True, num_workers=2)

    #########################################
    # Model
    #########################################
    model = models.resnet18(2, pretrained=False).to(device)

    # # Freeze first two layers.
    # num_freeze_layers = 2
    # for i, layer in enumerate(model.children()):
    #     for parameter in layer.parameters():
    #         parameter.requires_grad = False
    #
    #     if i == num_freeze_layers:
    #         break
    #


    if args.restore_model:
        state_dict = torch.load(args.restore_model)
        model.load_state_dict(state_dict)
        print('model is restored from {}'.format(args.restore_model))


    #########################################
    # optimizer
    #########################################
    if args.frozen:
        parameters = model.fc.parameters()
    else:
        parameters = model.parameters()
    parameters = filter(lambda x: x.requires_grad, parameters)
    optimizer = optim.Adam(parameters)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 60, 0.1)

    #########################################
    # Damage weights
    #########################################
    if args.damage_weights is not None:
        damage_weights = torch.Tensor([1., args.damage_weights]).to(device)
    else:
        damage_weights = None
    print('Damage_weights: {}'.format(damage_weights), flush=True)


    #########################################
    # training
    #########################################
    summary_path = os.path.join(args.tb_dir, args.experiment)
    summary = SummaryWriter(summary_path)
    print('logging to: {}'.format(summary_path))
    print('Training\n' + '-'*30)

    MAP_best = -1

    for epoch in range(args.epochs):
        lr_scheduler.step()

        start = timeit.default_timer()

        train_utils.train_nll(model, optimizer, train_loader, device, weights=damage_weights)
        damage_nll, damage_acc, true_p, false_p, MAP = train_utils.test_nll(
            model, test_loader, device, weights=None)
        summary.add_scalar('test/damage_acc', damage_acc, global_step=epoch)
        summary.add_scalar('test/damage_nll', damage_nll, global_step=epoch)
        summary.add_scalar('test/nll', damage_nll, global_step=epoch)
        summary.add_scalar('test/true_p_recall', true_p, global_step=epoch)
        summary.add_scalar('test/false_p', false_p, global_step=epoch)
        summary.add_scalar('test/precision', true_p / (true_p + false_p + 1e-5), global_step=epoch)
        summary.add_scalar('test/MAP', MAP, global_step=epoch)

        end = timeit.default_timer()
        print('Epoch took %d seconds' % (end-start))

        if (MAP > MAP_best):
            best_model = copy.deepcopy(model)
            MAP_best = MAP

        print('# {:3d}/{:3d}| MAP: {:3.1f}%, True_p: {:3.1f}%'.format(
            epoch + 1, args.epochs, 100 * MAP, 100 * true_p
        ), flush=True)

    print('-'*30)
    print('Training is finished')

    print('Best MAP was %4.4f' % MAP_best)
    model = best_model



    #########################################
    # save models
    #########################################
    model_dir = os.path.join(args.model_dir, args.experiment.replace('/', '-'))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'model_%s_%3.1f.pt' % (args.experiment, MAP_best))
    torch.save(model.state_dict(), model_path)
    print('Model saved: "{}"'.format(model_path))

    # Build CSV

    csv_path = os.path.join(model_dir, 'test_%s.csv' % args.experiment)
    csv_logger = CSVLogger(fieldnames=['imgname', 'conscore', 'label'], filename=csv_path)

    model.eval()
    device=torch.device('cuda')
    with torch.no_grad():
        for images, image_path, labels in test_loader:
            images = images.to(device)
            pred = model(images)
            for i in range(0, len(images)):
                row = {'imgname': image_path[i], 'conscore': str(pred.data[i,1].item()), 'label': str(labels.data[i].item())}
                csv_logger.writerow(row)
    csv_logger.close()

if __name__ == '__main__':
    main()

