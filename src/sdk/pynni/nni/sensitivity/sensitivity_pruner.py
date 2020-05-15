# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import json
import torch
import torch.nn as nn
from .sensitivity_analyze import SensitivityAnalysis
from .sensitivity_analyze import SUPPORTED_OP_TYPE
from .sensitivity_analyze import SUPPORTED_OP_NAME
from nni.compression.torch import L1FilterPruner
from nni.compression.torch import L2FilterPruner

MAX_PRUNE_RATIO_PER_ITER = 0.95


class SensitivityPruner:
    def __init__(self, model, val_func, finetune_func=None, resume_frome=None):
        self.model = model
        self.val_func = val_func
        self.finetune_func = finetune_func
        self.analyzer = SensitivityAnalysis(self.model, self.val_func)
        self.resume_from = resume_frome
        # Get the original accuracy of the pretrained model
        self.ori_acc = self.val_func(self.model)
        # Copy the original weights before pruning
        self.ori_state_dict = copy.deepcopy(self.model.state_dict())
        self.sensitivities = {}
        # Save the weight count for each layer
        self.weight_count = {}
        self.weight_sum = 0
        # Map the layer name to the layer module
        self.named_module = {}
        for name, submodule in self.model.named_modules():
            self.named_module[name] = submodule
            if name in self.analyzer.target_layer:
                # Currnetly, only count the weights in the conv layers
                # else the fully connected layer (which contains
                # the most weights) may make the pruner prune the
                # model too hard
                # if hasattr(submodule, 'weight'): # Count all the weights of the model
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
                sensitivities[name] = {float(k): float(v)
                                       for k, v in sensitivities[name].items()}
            return sensitivities

    def _max_prune_ratio(self, ori_acc, threshold, sensitivities):
        # TODO as default, provide customerize lambda function  quantified (sensi)
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

    def normalize(self, ratios, target_pruned):
        """
        Normalize the prune ratio of each layer according to the
        total already pruned ratio and the finnal target total prune 
        ratio

        Parameters
        ----------
            ratios:
                Dict object that save the prune ratio for each layer
            target_pruned:
                The amount of the weights expected to be pruned in this
                iteration

        Returns
        -------
            new_ratios:
                return the normalized prune ratios for each layer.

        """
        # TODO: filter out the very sensitive layers. Only prune the
        # layers that are not very sensitive
        w_sum = 0
        _Max = 0
        for layername, ratio in ratios.items():
            wcount = self.weight_count[layername]
            w_sum += ratio * wcount * \
                (1-self.analyzer.already_pruned[layername])
        target_count = self.weight_sum * target_pruned
        for layername in ratios:
            ratios[layername] = ratios[layername] * target_count / w_sum
            _Max = max(_Max, ratios[layername])
        # Cannot Prune too much in a single iteration
        # If a layer's prune ratio is larger than the
        # MAX_PRUNE_RATIO_PER_ITER we rescal all prune
        # ratios under this threshold
        if _Max > MAX_PRUNE_RATIO_PER_ITER:
            for layername in ratios:
                ratios[layername] = ratios[layername] * \
                    MAX_PRUNE_RATIO_PER_ITER / _Max
        return ratios

    def create_cfg(self, ratios):
        """
        Generate the cfg_list for the pruner according to the prune ratios.

        Parameters
        ---------
            ratios:
                For example: {'conv1' : 0.2}

        Returns
        -------
            cfg_list:
                For example: [{'sparsity':0.2, 'op_names':['conv1'], 'op_types':['Conv2d']}]
        """
        cfg_list = []
        for layername in ratios:
            prune_ratio = ratios[layername]
            remain = 1 - self.analyzer.already_pruned[layername]
            sparsity = remain * prune_ratio + \
                self.analyzer.already_pruned[layername]
            if sparsity > 0:
                # Pruner does not allow the prune ratio to be zero
                cfg = {'sparsity': sparsity, 'op_names': [layername], 'op_types': ['Conv2d']}
                cfg_list.append(cfg)
        return cfg_list

    def current_sparsity(self):
        """
        The sparisity of the weight.
        """
        pruned_weight = 0
        for layer_name in self.analyzer.already_pruned:
            w_count = self.weight_count[layer_name]
            prune_ratio = self.analyzer.already_pruned[layer_name]
            pruned_weight += w_count * prune_ratio
        return pruned_weight / self.weight_sum

    def compress(self, target_ratio, ratio_step=0.1, threshold=0.05, MAX_ITERATION=None):
        """
        We iteratively prune the model according to the results of 
        the sensitivity analysis.

        """
        if not self.resume_from:
            # TODO: support loading the sensitivities from the json file
            self.sensitivities = self.analyzer.analysis()
        else:
            self.sensitivities = self.load_sensitivitis(self.resume_from)

        cur_ratio = 1.0
        ori_acc = self.ori_acc
        iteration_count = 0

        while cur_ratio > target_ratio:
            iteration_count += 1
            if MAX_ITERATION is not None and iteration_count > MAX_ITERATION:
                break
            # Each round have three steps:
            # 1) Get the current sensitivity for each layer
            # 2) Prune each layer according the sensitivies
            # 3) finetune the model
            print('Current base accuracy', ori_acc)
            print('Current remained ratio', cur_ratio)
            # Use the max prune ratio whose accuracy drop is smaller
            # than the threshold(0.05) as the quantified sensitivity
            # for each layer
            # ps: the smaller the max_prune_ratio, more
            # sensitive the layer is
            quantified = self._max_prune_ratio(
                ori_acc, threshold, self.sensitivities)
            new_pruneratio = self.normalize(quantified, ratio_step)
            cfg_list = self.create_cfg(new_pruneratio)
            print(cfg_list)
            pruner = L1FilterPruner(self.model, cfg_list)
            pruner.compress()
            pruned_acc = self.val_func(self.model)
            print('Accuracy after pruning:', pruned_acc)
            # TODO: support the multiple parameters for the fine_tune function
            self.finetune_func(self.model)
            finetune_acc = self.val_func(self.model)
            print('Accuracy after finetune:', finetune_acc)
            ori_acc = finetune_acc
            # unwrap the pruner
            pruner._unwrap_model()
            # update the already prune ratio of each layer befor the new
            # sensitivity analysis
            for layer_cfg in cfg_list:
                name = layer_cfg['op_names'][0]
                sparsity = layer_cfg['sparsity']
                self.analyzer.already_pruned[name] = sparsity
            cur_ratio = 1 - self.current_sparsity()
            del pruner
            print('Currently Remained ratio:', cur_ratio)
            if MAX_ITERATION is not None and iteration_count < MAX_ITERATION:
                # If this is the last prune iteration, skip the time-consuming
                # sensitivity analysis
                self.analyzer.load_state_dict(self.model.state_dict())
                self.sensitivities = self.analyzer.analysis()
            # update the cur_ratio

        print('After Pruning: %.2f weights remains' % cur_ratio)
        return self.model

    def export(self, model_path, pruner_path=None):
        """
        Export the pruned results of the target model.

        Parameters
        ----------
            model_path:
                Path of the checkpoint of the pruned model.
            pruner_path:
                If not none, save the config of the pruner to this file.
        """
        torch.save(self.model.state_dict(), model_path)
        if pruner_path is not None:
            sparsity_ratios = {}
            for layername in self.analyzer.already_pruned:
                sparsity_ratios[layername] = self.analyzer.already_pruned[layername]
                cfg_list = self.create_cfg(sparsity_ratios)
            with open(pruner_path, 'w') as pf:
                json.dump(cfg_list, pf)
