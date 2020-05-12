# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import json
import torch.nn as nn
import torchvision 
import torchvision.models as models
import sensitivity_analyze 
from sensitivity_analyze import SensitivityAnalysis
import torchvision.transforms as transforms
import torchvision.datasets as datasets

val_dir = '/mnt/imagenet/raw_jpeg/2012/val/'
criterion = nn.CrossEntropyLoss()
imagenet_tran_test = [
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
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
        for batchid, (data,label) in enumerate(val_loader):
            #print(batchid)
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
    

if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    net.cuda()
    s_analyzer = SensitivityAnalysis(net, val, 0.1)
    sensitivity = s_analyzer.analysis()
    print(sensitivity)
    with open('resnet18_sensitivity.json', 'w') as jf:
        json.dump(sensitivity, jf)