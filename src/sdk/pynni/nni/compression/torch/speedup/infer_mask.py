# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class AutoMaskInference:
    def __init__(self, module, dummy_input, in_masks=None, weight_mask=None, output_mask=None):
        errmsg = '%s is not callable, should pass the nn.Module/function' % str(
            module)
        assert callable(module), errmsg
        self.module = module

        # Initialize the dummy_input
        if isinstance(dummy_input, list):
            # if there are multiple input variables
            self.dummy_input = dummy_input
        else:
            # if there is only one input variable
            self.dummy_input = [dummy_input]

        # Initialize the masks for input tensors
        self.in_masks = in_masks if in_masks is not None else [
            None] * len(self.dummy_input)
        for in_id, _ in enumerate(self.in_masks):
            if self.in_masks[in_id] is None and \
                    isinstance(self.dummy_input[in_id], torch.Tensor):
                # if the input mask is None then create a all-ones mask for corresponding input tensor
                self.in_masks[in_id] = torch.ones_like(self.dummy_input[in_id])
                # ones_like will put the created mask on the same device with the dummy_input

        # Initialize the mask for output tensors
        self.output = self.module(*dummy_input)
        if output_mask is not None:
            # assume the given output mask is right
            self.output_mask = output_mask
        else:
            errmsg = 'Only support the module/function that returns tensor/tuple of tensors/list of tensors'
            if isinstance(self.output, torch.Tensor):
                self.output_mask = torch.ones_like(self.output)
            elif isinstance(self.output, list) or isinstance(self.output, tuple):
                self.output_mask = []
                for o_tensor in self.output:
                    assert isinstance(o_tensor, torch.Tensor), errmsg
                    self.output_mask.append(torch.ones_like(o_tensor))
            else:
                raise ValueError(errmsg)

        # Initialize the parameter mask for the parameters
        self.weights = {}
        self.weight_mask = {}
        if weight_mask:
            self.weight_mask.update(weight_mask)
        if isinstance(self.module, nn.Module):
            # the function should not has parameters
            # get all the parameter tensors of the target module
            for name, para in module.named_parameters():
                self.weights[name] = para
                if name not in self.weight_mask:
                    self.weight_mask[name] = torch.ones_like(para.data)

    def update_input_mask(self):
        raise NotImplementedError

    def update_output_mask(self):
        raise NotImplementedError

    def update_weight_mask(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def apply_mask(self):
        raise NotImplementedError

    def update_direct_sparsity(self):
        raise NotImplementedError

    def update_indirect_sparsity(self):
        raise NotImplementedError

    def random_init(self, start=1, end=10):
        """
        Random initialize the weights of the module. The value of
        the tensor will not affect the mask auto inference.
        """
        with torch.no_grad():
            for tensor in self.dummy_input:
                if isinstance(tensor, torch.Tensor):
                    nn.init.uniform_(tensor.data, start, end)
            for para in self.weights:
                nn.init.uniform_(self.weights[para].data, start, end)

    def zero_grad(self):
        """
        Set the gradient of the weight, input tensor to be zeros.
        """
        with torch.no_grad():
            # set the weight's gradient to zero
            if isinstance(self.module, nn.Module):
                self.module.zero_grad()
            # also zero the gradient of the input tensors
            for tensor in self.dummy_input:
                if isinstance(tensor, torch.Tensor):
                    if tensor.grad is not None:
                        tensor.grad.data.zero_()

    def requires_grad_(self, flag=True):
        """
        Set the requires_grad of input tensor and parameters to flag.
        """
        for t_in in self.dummy_input:
            if isinstance(t_in, torch.Tensor):
                # enable the auto gradient
                t_in.requires_grad_(flag)
        for para_name in self.weights:
            self.weights[para_name].requires_grad_(flag)

    def apply_mask_zero(self):
        """
        Set the masked values to zero.
        """
        with torch.no_grad():
            # apply the input mask
            for tid, in_tensor in enumerate(self.dummy_input):
                if isinstance(in_tensor, torch.Tensor) and self.in_masks[tid] is not None:
                    in_tensor.data *= self.in_masks[tid]
            # apply the weight mask
            for para in self.weights:
                if para in self.weight_mask:
                    self.weights[para].data *= self.weight_mask[para].data

    def apply_mask_nan(self):
        """
        Set the masked values to Nan.
        """
        with torch.no_grad():
            # apply the input mask
            for tid, in_tensor in enumerate(self.dummy_input):
                if isinstance(in_tensor, torch.Tensor) and self.in_masks[tid] is not None:
                    nan_mask = self.in_masks[tid].clone().detach()
                    nan_mask[nan_mask == 0] = float('nan')
                    in_tensor.data *= nan_mask
            # apply the weight mask
            for para in self.weights:
                if para in self.weight_mask:
                    nan_mask = self.weight_mask[para].clone().detach()
                    nan_mask[nan_mask == 0] = float('nan')
                    self.weights[para].data *= nan_mask


class AutoMaskInferenceZero(AutoMaskInference):
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
        super(AutoMaskInferenceZero, self).__init__(
            module, dummy_input, in_masks, weight_mask, output_mask)

    def clac_out_sparsity(self):
        """
        Calculate the output sparsity.
        """
        # we don't need the gradient in the forward inference
        out_mask = None
        with torch.no_grad():
            self.random_init()
            # Note: due to the in-place operator, such as relu_,
            # ori_out may be the same tensor with dummy_input,
            # so we use clone and detach to create a new tensor with
            # the same values.
            ori_out = self.module(*self.dummy_input).clone().detach()
            # print('Ori output')
            # print(ori_out)
            # Note: we need randomly init the input one more time here!
            # Because some operation have the in-place operation, such as relu_,
            # the in-place operation may modify or write 0s into the dummy_input
            self.random_init()
            # apply the mask for the input tensor and the weight tensor
            self.apply_mask_zero()
            # print('New output')
            # Same to ori_out, in order to avoid the interference between
            # ori_out and new_out, we just also detach the new_out from the
            # graph
            new_out = self.module(*self.dummy_input).clone().detach()
            # print(new_out)
            if isinstance(ori_out, torch.Tensor):
                new_zeors = new_out == 0
                ori_zeros = ori_out == 0
                # only mask the newly added zeros
                # print('New zeros')
                # print(new_zeors)
                # print('ori_zeros')
                # print(ori_zeros)
                new_zeors[ori_zeros] = False
                # print('Final new zeros')
                # print(new_zeors)
                out_mask = torch.ones_like(ori_out)
                out_mask[new_zeors] = 0
            elif isinstance(ori_out, tuple) or isinstance(ori_out, list):
                out_mask = []
                for ori_tensor, new_tensor in zip(ori_out, new_out):
                    new_zeros = new_tensor == 0
                    ori_zeros = ori_tensor == 0
                    new_zeros[ori_zeros] = False
                    _sparsity = torch.ones_like(ori_tensor)
                    _sparsity[new_zeros] = 0
                    out_mask.append(_sparsity)
            else:
                _logger.warn(
                    'Only support the OP whose output is tensor/tuple of tensor/list of tensor')
        # print('out_mask')
        # print(out_mask)
        return out_mask

    def update_direct_sparsity(self):
        with torch.no_grad():
            out_sparsity = self.clac_out_sparsity()
            if isinstance(out_sparsity, torch.Tensor):
                assert isinstance(self.output_mask, torch.Tensor)
                self.output_mask *= out_sparsity
            elif isinstance(out_sparsity, list):
                for i, _ in enumerate(out_sparsity):
                    self.output_mask[i] *= out_sparsity[i]
            else:
                _logger.warn('Update the output sparsity Failed!')

    def update_indirect_sparsity(self):
        """
        Find those hidden sparsity through gradient.
        """
        # update the output mask
        if isinstance(self.output, torch.Tensor) and self.output.grad:
            # if output have gradient which means this node has successor
            # nodes and the successor nodes have already update their indirect
            # sparsity
            _grad_zero = self.output.grad.data == 0
            self.output_mask[_grad_zero] = 0
        elif isinstance(self.output, tuple) or isinstance(self.output, list):
            # TODO
            pass
        self.requires_grad_(True)
        # Forward inference with auto gradient enabled
        # Note: tensors that need gradient cannot be used in the in-place operator
        self.random_init()
        self.apply_mask_zero()
        tmp_dummy_input = [x.clone() if isinstance(
            x, torch.Tensor) else x for x in self.dummy_input]
        output = self.module(*tmp_dummy_input)
        # Note: output maybe tensor or list/tuple of tensors
        if isinstance(output, torch.Tensor):
            output.backward(self.output_mask)
        elif isinstance(output, list) or isinstance(output, tuple):
            for tid, t_out in enumerate(output):
                t_out.backward(self.output_mask[tid])
        print('\n\nself.output')
        print(self.output)
        print('\n\nself.output_mask\n\n')
        print(self.output_mask)
        print('\n\nself.dummy_input\n\n')
        print(self.dummy_input)
        print('\n\nself.in_mask\n\n')
        print(self.in_masks)
        print('\n\noutput\n\n')
        print(output)

        # print(self.weight)
        # update the sparsity of the paramters
        for para_name in self.weights:
            print("!!!!!!!!!!!!")
            print(para_name)
            print(self.weights[para_name].grad)
            grad_zero = self.weights[para_name].grad.data == 0
            self.weight_mask[para_name][grad_zero] = 0

        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    def unmask(self, t_unmask):
        """
        Unmask some values to resolve the conflict/interference between the masks.
        Note: the t_unmask indicates the values that should be unmasked in the output
        tensors. We work backwards to resolve the mask conflicts in the model. We can only
        infer the values need to be unmasked in the input tensor/parameters from the unmasked
        values in the output tensor.
        Parameters
        ---------
        t_unmask: torch.Tensor
            This tensor indicates the values that should be unmasked in the output tensor.
        Returns
        -------
        input_unmask: list
            The values in the input tensors that should be unmasked
        """
        # Enable the gradient
        self.requires_grad_()
        self.zero_grad()
        self.random_init()
        # in case there is in_place operation in this node
        tmp_dummy_input = [x.clone() if isinstance(
            x, torch.Tensor) else x for x in self.dummy_input]
        output = self.module(*tmp_dummy_input)
        # backwards to get the gradient
        if isinstance(t_unmask, torch.Tensor):
            gradient_nan = torch.ones_like(output)
            # find all the positions that need to be unmasked
            unmask_pos = t_unmask > 0
            gradient_nan[unmask_pos] = float('nan')
            output.backward(gradient_nan)
            # update the output mask
            self.output_mask[unmask_pos] = 1
        elif isinstance(t_unmask, list) or isinstance(t_unmask, tuple):
            assert isinstance(output, list) or isinstance(output, tuple)
            # the length of unmask tensor list should be exactly same with t_unmask
            assert len(output) == len(t_unmask)
            for i, _ in enumerate(t_unmask):
                _unmask = t_unmask[i]
                _output = output[i]
                gradient_nan = torch.ones_like(_output)
                unmask_pos = _output > 0
                gradient_nan[unmask_pos] = float('nan')
                _output.backward(gradient_nan)
                self.output_mask[i][unmask_pos] = 1
        # all the values whose gradient is Nan should be unmasked
        # unmask the values in the parameters
        for para_name in self.weights:
            gradient = self.weights[para_name].grad.data
            unmask_pos = torch.isnan(gradient)
            self.weight_mask[para_name][unmask_pos] = 1
        # check if there are values in the input tensors that should be unmasked
        input_debug = []
        input_unmask = []
        for i, _ in enumerate(self.dummy_input):
            if not isinstance(self.dummy_input[i], torch.Tensor):
                continue
            gradient = self.dummy_input[i].grad.data
            unmask_pos = torch.isnan(gradient)

            if torch.sum(unmask_pos.to(torch.float32) - self.in_masks[i]>0 ) > 0:
                # if there is a masked value need to be unmasked, 1 in the unmask_pos
                # and 0 in self.in_masks[i]
                self.in_masks[i][unmask_pos] = 1
                input_debug.append(self.input_debugname[i])
                input_unmask.append(unmask_pos.to(torch.float32))
        return input_debug, input_unmask

    def update_sparsity(self):
        self.update_direct_sparsity()
        self.update_indirect_sparsity

    def _inputs_internal(self, target_tensor):
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
        """
        # we need to trace the mask relationship by the gradient
        if isinstance(self.module, nn.Module):
            self.module.train()
        self.zero_grad()
        # enable the grad for the target tensor
        target_tensor.requires_grad_()
        assert isinstance(target_tensor, torch.Tensor)
        tmp_out = self.module(*self.dummy_input)
        loss = torch.sum(tmp_out)
        # modify the value of the other input tensors and weights tensor
        # before call the backward, following operations should not change
        # the gradient dependent chain
        with torch.no_grad():
            self.random_init()
            # set the masked values to zero and the unmasked valued to positive
            # values
            for _id, tensor in enumerate(self.dummy_input):
                if self.in_masks[_id] is not None:
                    tensor.data *= self.in_masks[_id]

            for para_name in self.weights:
                if para_name in self.weight_mask:
                    self.weights[para_name] *= self.weight_mask[para_name].data

        loss.backward()
        # now all the positions that the gradient equals to zeros
        # can be masked in the target tensor
        grad = target_tensor.grad.data
        # shut down the gradient
        # target_tensor.requires_grad_(False)
        return grad == 0

    def _backwards_nan(self, in_tensor):
        """
        Find out which parts of the input are only used to calculate the part of the
        output that is masked out.
        Parameters
        ----------
        in_tensor: torch.Tensor
            The target tensor to infer the mask for.

        Returns
        -------
        """
        out_tensor = self.module(*self.dummy_input)
        # create a new mask that has 1s and nans, replace the 0s
        # with nan
        nan_mask = self.output_mask.clone().detach()
        nan_mask[nan_mask == 0] = float('nan')
        n_dim = len(in_tensor.size())

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
                    _logger.debug(
                        'Try the dimension: %d, Index: %d', _dim, _index)
                    _logger.debug(
                        'The output tensor after nan colored: \n%s', str(tmp_out_tensor))
                    _logger.debug('The expected output tensor: \n%s',
                                  str(nan_mask*out_tensor))
                    _nan_pos1 = nan_mask.cpu().numpy()
                    _nan_pos2 = tmp_out_tensor.cpu().numpy()
                    if np.array_equal(np.isnan(_nan_pos1), np.isnan(_nan_pos2)):
                        # if the position of the nans is exactly the same
                        # we can mask these values for the input tensor
                        _logger.info('Masking the %s', str(_slice))
                        in_mask[_slice] = 0
                        flag = True
        return in_mask

    def _forwards_outmask(self):
        """
        Infer the output mask for the output tensor.
        """
        # random initialize the input_tensors and the weights tensors
        self.random_init()
        # get the origin output tensor
        _ori_out = self.module(*self.dummy_input)
        # apply the mask for the input tensor and the weight tensor
        self.apply_mask_zero()
        _new_out = self.module(*self.dummy_input)
        _new_zeors = _new_out == 0
        _ori_zeros = _ori_out == 0
        # only mask the newly added zeros
        _new_zeors[_ori_zeros] = False
        out_mask = torch.ones_like(_ori_out)
        out_mask[_new_zeors] = 0
        return out_mask

    def update_output_mask(self):
        """
        Infer and update the output mask.
        """

        _out_m = self._forwards_outmask()
        # print(_out_m.size())
        # print(self.output_mask.size())

        self.output_mask *= _out_m

    def update_weight_mask(self):
        """
        Update the masks of parameters
        """
        if not isinstance(self.module, nn.Module):
            # function don't have parameters
            return
        for name, para in self.module.named_parameters():
            # update the masks of all parameters
            self.weight_mask[name] *= self._inputs_internal(para)
            self.weight_mask[name] *= self._backwards_nan(para)

    def update_input_mask(self):
        """
        Update the masks of the input tensors.
        """
        for in_id, _input in enumerate(self.dummy_input):
            if isinstance(_input, torch.Tensor):
                self.in_masks[in_id] *= self._inputs_internal(_input)
                self.in_masks[in_id] *= self._backwards_nan(_input)

    def update(self):
        # with torch.autograd.set_detect_anomaly(True):
        self.update_output_mask()
        self.update_input_mask()
        self.update_weight_mask()


class AutoMaskInferenceRemove(AutoMaskInferenceZero):
    def __init__(self, module, dummy_input, in_masks=None, weight_mask=None, output_mask=None):
        super(AutoMaskInferenceRemove, self).__init__(
            module, dummy_input, in_masks, weight_mask, output_mask)

    def _forwards_outmask(self):
        """
        Infer the output mask of the output tensor.
        """
        self.random_init()
        self.apply_mask_nan()
        out_tensor = self.module(*self.dummy_input)
        _tmp_mask = torch.ones_like(out_tensor)
        # the nan value in the out tensor should be masked
        masked_pos = torch.isnan(out_tensor)
        _tmp_mask[masked_pos] = 0
        return _tmp_mask

    def update_output_mask(self):
        out_m = self._forwards_outmask()
        self.output_mask *= out_m

    def update_weight_mask(self):
        if not isinstance(self.module, nn.Module):
            # there is no parameters
            return
        for name, para in self.module.named_parameters():
            self.weight_mask[name] *= self._backwards_nan(para)

    def update_input_mask(self):
        for in_id, _input in enumerate(self.dummy_input):
            if isinstance(_input, torch.Tensor):
                self.in_masks[in_id] *= self._backwards_nan(_input)

    def update(self):
        self.update_output_mask()
        self.update_input_mask()
        self.update_weight_mask()

    def update_direct_sparsity(self):
        with torch.no_grad():
            out_sparsity = self.clac_out_sparsity()
            if isinstance(out_sparsity, torch.Tensor):
                assert isinstance(self.output_mask, torch.Tensor)
                self.output_mask *= out_sparsity
            elif isinstance(out_sparsity, list):
                for i, _ in enumerate(out_sparsity):
                    self.output_mask[i] *= out_sparsity[i]
            else:
                _logger.warn('Update direct sparsity failed!')

    def clac_out_sparsity(self):
        # in case the dummy_input which created by the predecessors
        # is not clean (For example still have nan in it)
        self.random_init()
        self.apply_mask_nan()
        out_mask = None
        with torch.no_grad():
            new_out = self.module(*self.dummy_input)
            if isinstance(new_out, torch.Tensor):
                nan_pos = torch.isnan(new_out)
                out_mask = torch.ones_like(new_out)
                out_mask[nan_pos] = 0
            elif isinstance(new_out, list) or isinstance(new_out, tuple):
                out_mask = []
                for t_out in new_out:
                    assert isinstance(t_out, torch.Tensor)
                    nan_pos = torch.isnan(t_out)
                    tmp_mask = torch.ones_like(t_out)
                    tmp_mask[nan_pos] = 0
                    out_mask.append(tmp_mask)
            else:
                _logger.warn(
                    'Only support the OP whose output is tensor/list of tensors')

        return out_mask
