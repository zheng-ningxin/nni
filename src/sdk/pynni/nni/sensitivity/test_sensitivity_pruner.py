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

train_dir = '/mnt/imagenet/raw_jpeg/2012/train/'
val_dir = '/mnt/imagenet/raw_jpeg/2012/val/'
lr = 0.001

criterion = nn.CrossEntropyLoss()
batch_size = 64
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
    num_workers=4, pin_memory=True, sampler=None)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_dir, transforms.Compose(imagenet_tran_test)),
    batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True)


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
    print('Accuracy: ', correct / total)
    return correct / total


def train(model):
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    model.train()
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
        print('Batch %d Loss:%.3f Acc:%.3f' %
              (batchid, total_loss/(batchid+1), correct/total))


if __name__ == '__main__':
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
    args = parser.parse_args()
    net = models.resnet18(pretrained=True)
    net.cuda()
    # pruner = SensitivityPruner(net, val, train , 'resnet18_sensitivity.json')
    if args.resume:
        pruner = SensitivityPruner(net, val, train, args.resume)
    else:
        pruner = SensitivityPruner(net, val, train)
    net = pruner.compress(args.target_ratio, threshold=args.threshold,
                          ratio_step=args.ratio_step, MAX_ITERATION=args.maxiter)
    model_file = os.path.join(args.outdir, 'resnet18_sensitivity_prune.pth')
    pruner_cfg_file = os.path.join(args.outdir, 'resnet18_pruner.json')
    pruner.export(model_file, pruner_cfg_file)
