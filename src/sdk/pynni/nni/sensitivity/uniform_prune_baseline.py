# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import torch
import json
import torch.nn as nn
import torchvision
import torchvision.models as models
#import sensitivity_analyze
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
#import sensitivity_pruner
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--outdir', help='save the result in this directory')
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

from nni.compression.torch import L1FilterPruner
import copy
if __name__ == '__main__':

    net = models.resnet34(pretrained=True)
    net.cuda()
    # pruner = SensitivityPruner(net, val, train , 'resnet18_sensitivity.json')
    ori_stat_dict = copy.deepcopy(net.state_dict())
    for ratio in np.arange(0.1, 0.6, 0.1):
        ratio = np.round(ratio, 2)
        cfg = [{'sparsity' : ratio, 'op_types':['Conv2d']}]
        pruner = L1FilterPruner(net, cfg)
        pruner.compress()
        print('Uniformly prune baseline Prune ratio', ratio)
        val(net)
        train(net)
        val(net)
        pruner._unwrap_model()
        del pruner
        net.load_state_dict(ori_stat_dict)
        
