# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import unittest
from unittest import TestCase, main
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

from nni.compression.torch import L1FilterPruner
from nni.analysis_utils.topology.torch.graph_from_trace import VisualGraph
from nni.analysis_utils.topology.torch.shape_dependency import ChannelDependency
from nni.analysis_utils.topology.torch.mask_conflict import MaskConflict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prefix = 'analysis_test'
model_names = ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg19',
               'resnet18', 'resnet34', 'squeezenet1_1',
               'shufflenet_v2_x1_0', 'mobilenet_v2', 'wide_resnet50_2']

channel_dependency_ground_truth = {
    'resnet18': [{'layer1.0.conv2', 'layer1.1.conv2', 'conv1'},
                 {'layer2.1.conv2', 'layer2.0.conv2', 'layer2.0.downsample.0'},
                 {'layer3.0.downsample.0', 'layer3.1.conv2', 'layer3.0.conv2'},
                 {'layer4.0.downsample.0', 'layer4.1.conv2', 'layer4.0.conv2'}],
    'resnet34': [{'conv1', 'layer1.2.conv2', 'layer1.1.conv2', 'layer1.0.conv2'},
                 {'layer2.3.conv2', 'layer2.0.conv2', 'layer2.0.downsample.0',
                  'layer2.1.conv2', 'layer2.2.conv2'},
                 {'layer3.3.conv2', 'layer3.0.conv2', 'layer3.4.conv2', 'layer3.0.downsample.0',
                  'layer3.5.conv2', 'layer3.1.conv2', 'layer3.2.conv2'},
                 {'layer4.0.downsample.0', 'layer4.1.conv2', 'layer4.2.conv2', 'layer4.0.conv2'}],
    'mobilenet_v2': [{'features.3.conv.2', 'features.2.conv.2'},
                     {'features.6.conv.2', 'features.4.conv.2', 'features.5.conv.2'},
                     {'features.8.conv.2', 'features.7.conv.2',
                      'features.10.conv.2', 'features.9.conv.2'},
                     {'features.11.conv.2', 'features.13.conv.2',
                      'features.12.conv.2'},
                     {'features.14.conv.2', 'features.16.conv.2', 'features.15.conv.2'}],
    'wide_resnet50_2': [{'layer1.2.conv3', 'layer1.1.conv3', 'layer1.0.conv3', 'layer1.0.downsample.0'},
                        {'layer2.1.conv3', 'layer2.0.conv3', 'layer2.0.downsample.0',
                         'layer2.2.conv3', 'layer2.3.conv3'},
                        {'layer3.3.conv3', 'layer3.0.conv3', 'layer3.2.conv3', 'layer3.0.downsample.0',
                         'layer3.1.conv3', 'layer3.4.conv3', 'layer3.5.conv3'},
                        {'layer4.1.conv3', 'layer4.2.conv3', 'layer4.0.downsample.0', 'layer4.0.conv3'}],
    'alexnet': [],
    'vgg11': [],
    'vgg11_bn': [],
    'vgg13': [],
    'vgg19': [],
    'squeezenet1_1': [],
    'googlenet': [],
    'shufflenet_v2_x1_0': []
}

unittest.TestLoader.sortTestMethodsUsing = None


class AnalysisUtilsTest(TestCase):
    @unittest.skipIf(torch.__version__ < "1.3.0", "not supported")
    def test_channel_dependency(self):
        outdir = os.path.join(prefix, 'dependency')
        os.makedirs(outdir, exist_ok=True)
        for name in model_names:
            print('Analyze channel dependency for %s' % name)
            model = getattr(models, name)
            net = model().to(device)
            dummy_input = torch.ones(1, 3, 224, 224).to(device)
            channel_depen = ChannelDependency(net, dummy_input)
            depen_sets = channel_depen.dependency_sets
            d_set_count = 0
            for d_set in depen_sets:
                if len(d_set) > 1:
                    d_set_count += 1
                    assert d_set in channel_dependency_ground_truth[name]
            assert d_set_count == len(channel_dependency_ground_truth[name])
            fpath = os.path.join(outdir, name)
            channel_depen.export(fpath)
    # comments the visulization test temporarily
    # because, this test needs the graphviz package
    # in ths os.(apt install graphviz)
    # def test_visulization(self):
    #     outdir = os.path.join(prefix, 'visual')
    #     os.makedirs(outdir, exist_ok=True)
    #     for name in model_names:
    #         print('Visualization for %s' % name)
    #         model = getattr(models, name)
    #         net = model().to(device)
    #         dummy_input = torch.ones(1, 3, 224, 224).to(device)
    #         vg = VisualGraph(net, dummy_input)
    #         picpath = os.path.join(outdir, name)
    #         depen_file = os.path.join('analysis_test/dependency', name)
    #         vg.visualization(picpath, dependency_file=depen_file)

    def get_pruned_index(self, mask):
        pruned_indexes = []
        shape = mask.size()
        for i in range(shape[0]):
            if torch.sum(mask[i]).item() == 0:
                pruned_indexes.append(i)

        return pruned_indexes

    @unittest.skipIf(torch.__version__ < "1.3.0", "not supported")
    def test_mask_conflict(self):
        outdir = os.path.join(prefix, 'masks')
        os.makedirs(outdir, exist_ok=True)
        for name in model_names:
            print('Test mask conflict for %s' % name)
            model = getattr(models, name)
            net = model().to(device)
            dummy_input = torch.ones(1, 3, 224, 224).to(device)
            # random generate the prune sparsity for each layer
            cfglist = []
            for layername, layer in net.named_modules():
                if isinstance(layer, nn.Conv2d):
                    # pruner cannot allow the sparsity to be 0 or 1
                    sparsity = np.random.uniform(0.01, 0.99)
                    cfg = {'op_types': ['Conv2d'], 'op_names': [
                        layername], 'sparsity': sparsity}
                    cfglist.append(cfg)
            pruner = L1FilterPruner(net, cfglist)
            pruner.compress()
            ck_file = os.path.join(outdir, '%s.pth' % name)
            mask_file = os.path.join(outdir, '%s_mask' % name)
            pruner.export_model(ck_file, mask_file)
            pruner._unwrap_model()
            # Fix the mask conflict
            mf = MaskConflict(mask_file, net, dummy_input)
            fixed_mask = mf.fix_mask_conflict()
            mf.export(os.path.join(outdir, '%s_fixed_mask' % name))
            # use the channel dependency groud truth to check if
            # fix the mask conflict successfully
            for dset in channel_dependency_ground_truth[name]:
                lset = list(dset)
                for i, _ in enumerate(lset):
                    assert fixed_mask[lset[0]]['weight'].size(
                        0) == fixed_mask[lset[i]]['weight'].size(0)
                    w_index1 = self.get_pruned_index(
                        fixed_mask[lset[0]]['weight'])
                    w_index2 = self.get_pruned_index(
                        fixed_mask[lset[i]]['weight'])
                    assert w_index1 == w_index2
                    if hasattr(fixed_mask[lset[0]], 'bias'):
                        b_index1 = self.get_pruned_index(
                            fixed_mask[lset[0]]['bias'])
                        b_index2 = self.get_pruned_index(
                            fixed_mask[lset[i]]['bias'])
                        assert b_index1 == b_index2


if __name__ == '__main__':
    main()
