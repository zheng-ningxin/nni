# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class AutoMaskInference:
    """
    Given some masks (output mask, input mask, weight mask, for example) of the module,
    infer the rest masks for the target module automatically.
    """

    def __init__(self, module, dummy_input, in_masks=None, weight_mask=None, output_mask=None):
        """
        Infer the masks automatically for the unknow function op_func.
        Parameters
        ----------
        module: nn.Module
            The target module to infer the mask for.
        dummy_input: list
            The input list of the module. The forward inference should be called as module(*dummy_input).
            In case some module may have multiple input tensors the dummy_input is designed as a list.
        in_masks: list
            The mask of the corresponding inputs tensors.
        weight_mask: dict
        output_mask: list
        """
        assert isinstance(module, torch.nn.Module)
        self.module = module
        if isinstance(dummy_input, list):
            assert isinstance(in_masks, list)
            self.dummy_input = dummy_input
            self.in_masks = in_masks
        else:
            self.dummy_input = [dummy_input]
            self.in_masks = [in_masks]
        self.weight_mask = weight_mask
        self.output_mask = output_mask
        self.weights = {}
        # get all the parameter tensors of the target module
        for name, para in module.named_parameters():
            self.weights[name] = para.data

    def _random_init(self, start=1, end=10):
        """
        Random initialize the weights of the module. The value of
        the tensor will not affect the mask auto inference.
        """
        for tensor in self.dummy_input:
            if isinstance(tensor, torch.Tensor):
                nn.init.uniform_(tensor.data, start, end)
        for _, para in self.module.named_parameters():
            nn.init.uniform_(para.data, start, end)

    def _zero_grad(self):
        # set the weight's gradient to zero
        self.module.zero_grad()
        # also zero the gradient of the input tensors
        for tensor in self.dummy_input:
            if isinstance(tensor, torch.Tensor):
                tensor.data.zero_()

    def _apply_mask(self):
        for tid, in_tensor in enumerate(self.dummy_input):
            if isinstance(in_tensor, torch.Tensor) and self.in_masks[tid] is not None:
                in_tensor.data *= self.in_masks[tid]
        for name, para in self.module.named_parameters():
            if name in self.weight_mask:
                para.data *= self.weight_mask[name].data

    def _inputs_internal(self, target_tensor, target_mask):
        """
        Analyze the mask relationship between the multiple inputs of the
        module. Here we can also take the weights as an input of the new operation.

        A classic example for this function is pruning the input channel of the conv
        layers. If we prune the input channel of the conv layers (NxCxKxK, prune the C
        dimension), then the corresponding channels of the input tensor can also be pruned.
        However, pruning the input channel of weights `W` will not affect the mask of the output
        tensor, which means we cannot get the output mask first and than get the mask of input
        tenosr from the output tensor mask by _backwards_nan. That's why we need this function
        to handle the mask relationship between the multiple input tensors.

        Parameters
        ----------
        target_tensor: tensor
            We try to infer the mask for the target_tensor.
        target_mask: tensor
            The mask of the target_tensor
        """
        # we need to trace the mask relationship by the gradient
        self.module.train()
        self._zero_grad()
        # enable the grad for the target tensor
        target_tensor.requires_grad_()
        assert isinstance(target_tensor, torch.Tensor)
        tmp_out = self.module(*self.dummy_input)
        loss = torch.sum(tmp_out)
        # modify the value of the other input tensors and weights tensor
        # before call the backward, following operations should not change
        # the gradient dependent chain
        with torch.no_grad():
            for _id, tensor in enumerate(self.dummy_input):
                if self.in_masks[_id] is not None:
                    tensor.data.copy_(self.in_masks[_id])
                else:
                    tensor.data = 1
            for para_name in self.weights:
                if para_name in self.weight_mask:
                    self.weights[para_name].copy_(
                        self.weight_mask[para_name].data)
        loss.backward()
        # now all the positions that the gradient equals to zeros
        # can be masked in the target tensor
        grad = target_tensor.grad.data
        target_mask.data[grad == 0] = 0
        return target_mask

    def _backwards_nan(self, out_tensor, out_mask, in_tensor, in_mask=None):
        """
        Find out which parts of the input are only used to calculate the part of the
        output that is masked out.
        Parameters
        ----------
        out_tensor: torch.Tensor
            The output tensor of the target module.
        out_mask: torch.Tensor
            The mask tensor of the output tensor. out_mask only contains 0s and 1s, and
            0 indicate the corresponding position of the output tensor is pruned.
        in_tensor: torch.Tensor
            The target tensor to infer the mask for.
        in_mask: torch.Tensor
            The mask of the target input tensor.
        Returns
        -------
        """
        # create a new mask that has 1s and nans, replace the 0s
        # with nan
        nan_mask = out_mask.clone().detach()
        nan_mask[nan_mask == 0] = float('nan')
        n_dim = len(in_tensor.size())
        if in_mask is None:
            in_mask = torch.ones_like(in_tensor)
        _ori_in = in_tensor.clone().detach()
        flag = False
        with torch.no_grad():
            for _dim in range(n_dim):
                if flag:
                    # We search the dim from 0, if the high-dimension
                    # already meets the requirements, then we can just
                    # skip the following dimensions
                    break
                # try to mask as much positions as possible
                # only if the out_put tensor is equal to nan_mask * out_tensor
                for _index in range(in_tensor.size(_dim)):
                    # build the slice object for the tensor for
                    # corresponding dimension and position.
                    _slice = [slice(None, None, None) for i in range(n_dim)]
                    _slice[_dim] = _index
                    _slice = tuple(_slice)
                    in_tensor.data.copy_(_ori_in)
                    # set part of the input tensor to be nan, if the new output
                    # tensor is same with out_mask * out_tensor, then these
                    # masked values in the input tesor are only used to calculate
                    # the values masked in the output tensor, which means it's safe
                    # to mask these values in the input tensor
                    in_tensor.data[_slice] = float('nan')
                    tmp_out_tensor = self.module(*self.dummy_input)
                    _logger.debug('Try the dimension: %d, Index: %d', _dim, _index)
                    _logger.debug('The output tensor after nan colored: \n%s', str(tmp_out_tensor))
                    _logger.debug('The expected output tensor: \n%s', str(nan_mask*out_tensor))
                    _nan_pos1 = nan_mask.cpu().numpy()
                    _nan_pos2 = tmp_out_tensor.cpu().numpy()
                    if np.array_equal(np.isnan(_nan_pos1), np.isnan(_nan_pos2)):
                        # if the position of the nans is exactly the same
                        # we can mask these values for the input tensor
                        _logger.info('Masking the %s', str(_slice))
                        in_mask[_slice] = 0
                        flag = True
        return in_mask

    def _forwards_outmask(self, out_mask):
        """
        Infer the output mask for the output tensor.
        """
        # random initialize the input_tensors and the weights tensors
        self._random_init()
        # get the origin output tensor
        _ori_out = self.module(*self.dummy_input)
        # apply the mask for the input tensor and the weight tensor
        self._apply_mask()
        _new_out = self.module(*self.dummy_input)
        _new_zeors = _new_out == 0
        _ori_zeros = _ori_out == 0
        # only mask the newly added zeros
        _new_zeors[_ori_zeros] = False
        out_mask[_new_zeors] = 0
        return out_mask


class MaskInferFunction:
    """
    This class analyze the calculation logic of the module, and
    get the relationship between the input tensor, module's weight
    and the output tensor.
    """

    def __init__(self, module, dummy_inputs):
        """
        Parameters
        ----------
        module: nn.module
            The module to analyze the relationship betweent the input
            tensor, weight, and the output tensor.
        dummy_inputs: tensor/list
            The dummy input for the target module, it can be an tensor or
            a list. If it is a tensor, then the module only take a tensor
            as input.
        """
        self.module = module
        self.dummy_inputs = dummy_inputs
        self._analyze()

    def _analyze(self):

        pass
