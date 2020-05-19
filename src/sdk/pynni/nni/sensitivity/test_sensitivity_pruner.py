# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import torch
import json
import torch.nn as nn
import torchvision
import torchvision.models as models
#import sensitivity_analyze
from .sensitivity_analyze import SensitivityAnalysis
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
#import sensitivity_pruner
from .sensitivity_pruner import SensitivityPruner
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None,
                    help='resume from the sensitivity results')
parser.add_argument('--outdir', help='save the result in this directory')
parser.add_argument('--target_ratio', type=float, default=0.5,
                    help='Target ratio of the remained weights compared to the original model')
parser.add_argument('--maxiter', type=int, default=None,
                    help='max iteration of the sentivity pruning')
parser.add_argument('--ratio_step', type=float, default=0.1,
                    help='the amount of the pruned weight in each prune iteration')
parser.add_argument('--threshold', type=float, default=0.05,
                    help='The accuracy drop threshold during the sensitivity analysis')
parser.add_argument('--finetune_epoch', type=int, default=1, help='the number of epochs for finetune')
parser.add_argument('--lr', type=float, default=1e-3, help='Learing rate for finetune')
args = parser.parse_args()

train_dir = '/mnt/imagenet/raw_jpeg/2012/train/'
val_dir = '/mnt/imagenet/raw_jpeg/2012/val/'


criterion = nn.CrossEntropyLoss()
batch_size = 128
imagenet_tran_train = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
]

imagenet_tran_test = [
    transforms.Resize(int(224 / 0.875)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
]
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_dir, transforms.Compose(imagenet_tran_train)),
    batch_size=batch_size, shuffle=True,
    num_workers=12, pin_memory=True, sampler=None)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_dir, transforms.Compose(imagenet_tran_test)),
    batch_size=128, shuffle=False,
    num_workers=12, pin_memory=True)


def val(model):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batchid, (data, label) in enumerate(val_loader):
            # print(batchid)
            data, label = data.cuda(), label.cuda()
            out = model(data)
            loss = criterion(out, label)
            total_loss += loss.item()
            _, predicted = out.max(1)
            total += data.size(0)
            correct += predicted.eq(label).sum().item()
            # return correct / total
    print('Loss: ', total_loss/(batchid+1))
    print('Accuracy: ', correct / total)
    return correct / total


def train(model):
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=4e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    model.train()
    for epochid in range(args.finetune_epoch):    
        total_loss = 0
        total = 0
        correct = 0
        for batchid, (data, lable) in enumerate(train_loader):
            if batchid > 1000:
                break
            data, lable = data.cuda(), lable.cuda()
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, lable)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = out.max(1)
            correct += predicted.eq(lable).sum().item()
            total += lable.size(0)
            if batchid % 100 == 0:
                print('Epoch%d Batch %d Loss:%.3f Acc:%.3f LR:%f' %
                    (epochid, batchid, total_loss/(batchid+1), correct/total, lr_scheduler.get_lr()[0]))
        lr_scheduler.step()
        val(model)


if __name__ == '__main__':

    net = models.resnet34(pretrained=True)
    net.cuda()
    # pruner = SensitivityPruner(net, val, train , 'resnet18_sensitivity.json')
    if args.resume:
        pruner = SensitivityPruner(net, val, train, args.resume)
    else:
        pruner = SensitivityPruner(net, val, train)

    net = pruner.compress(args.target_ratio, threshold=args.threshold,
                          ratio_step=args.ratio_step, MAX_ITERATION=args.maxiter, checkpoint_dir=args.outdir)
    model_file = os.path.join(args.outdir, 'resnet34_sensitivity_prune.pth')
    pruner_cfg_file = os.path.join(args.outdir, 'resnet34_pruner.json')
    os.makedirs(args.outdir, exist_ok=True)
    pruner.export(model_file, pruner_cfg_file)
