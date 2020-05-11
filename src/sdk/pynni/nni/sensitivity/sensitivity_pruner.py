# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import json
import torch
import torch.nn as nn
from sensitivity_analyze import SensitivityAnalysis


class sensitivity_pruner:
    def __init__(self, model, val_func, finetune_func, resume_frome=None):
        self.model = model
        self.val_func = val_func
        self.finetune_func = finetune_func
        if not resume_frome:
        # Sensitivity Analysis
        # TODO: support loading the sensitivities from the json file
            self.analyzer = SensitivityAnalysis(self.model, self.val_func)
            self.sensitivities = self.analyzer.analysis()
        else:
            self.sensitivities = self.load_sensitivitis(resume_frome)
        # Get the original accuracy of the pretrained model
        self.ori_acc = self.val_func(self.model)
        # Copy the original weights before pruning
        self.ori_state_dict = copy.deepcopy(self.model.state_dict())
        # Save the weight count for each layer
        self.weight_count = {}
        # Map the layer name to the layer module
        self.named_module = {}
        for name, submodule in self.model.named_modules():
            self.named_module[name] = submodule
            if hasattr(submodule, 'weight'):
                self.weight_count[name] = submodule.weight.data.numel()
                self.weight_sum += self.weight_count[name]

    
    def load_sensitivitis(self, filepath):
        """
        Load the sensitivity analysis result from file
        """
        assert os.path.exists(filepath)
        with open(filepath, 'r') as jf:
            sensitivities = json.load(jf)
            # convert string type to float
            for name in sensitivities:
                sensitivities[name] = { float(k): float(v) for k, v in sensitivities[name]}
            return sensitivities


    def _max_prune_ratio(self, ori_acc, threshold, sensitivities):
        """
        Find the maximum prune ratio for a single layer whose accuracy 
        drop is lower than the threshold.
        
        Parameters
        ----------
            ori_acc:
                Original accuracy 
            threshold:
                Accuracy drop threshold 
            sensitivities:
                The dict object that stores the sensitivity results for each layer.
                For example: {'conv1' : {0.1: 0.9, 0.2 : 0.8}}
        Returns
        -------
            max_ratios:
                return the maximum prune ratio for each layer. For example:
                {'conv1':0.1, 'conv2':0.2}
        """
        max_ratio = {}
        for layer in sensitivities:
            prune_ratios = sorted(sensitivities[layer].keys())
            last_ratio = 0
            for ratio in prune_ratios:
                cur_acc = sensitivities[layer][ratio]
                if cur_acc + threshold < ori_acc:
                    break
                last_ratio = ratio
            max_ratio[layer] = last_ratio
        return max_ratio


    def compress(self, target_ratio, ratio_step=0.1, threshold=0.05):
        """
        We iteratively prune the model according to the results of 
        the sensitivity analysis.
        
        """
        ori_acc = self.ori_acc
        