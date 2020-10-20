# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
import torch.nn as nn
from .infer_shape import ModuleMasks

_logger = logging.getLogger(__name__)

replace_module = {
    'BatchNorm2d': lambda module, auto_infer: replace_batchnorm2d(module, auto_infer),
    'Conv2d': lambda module, auto_infer: replace_conv2d(module, auto_infer),
    'Linear': lambda module, auto_infer: replace_linear(module, auto_infer),
    'MaxPool2d': lambda module, auto_infer: no_replace(module, auto_infer),
    'AvgPool2d': lambda module, auto_infer: no_replace(module, auto_infer),
    'AdaptiveAvgPool2d': lambda module, auto_infer: no_replace(module, auto_infer),
    'ReLU': lambda module, auto_infer: no_replace(module, auto_infer),
    'ReLU6': lambda module, auto_infer: no_replace(module, auto_infer),
    'Dropout': lambda module, auto_infer: no_replace(module, auto_infer),
    'Dropout2d': lambda module, auto_infer: no_replace(module, auto_infer),
    'Dropout3d': lambda module, auto_infer: no_replace(module, auto_infer)
}


def convert_to_coarse_mask(t_mask, dim):
    """
    Convert the mask tensor to the coarse-grained mask tensor.
    Parameters
    ---------
    t_mask: torch.Tensor
        The tensor only have 1s and 0s, 0 indicates this value is masked
        and 1 indicates the corresponding value is not masked.
    dim: int
        Try to reduce the mask tensor on this dimension.

    Returns
    -------
    indexes: torch.Tensor
        The indexes of the sparsity that can be structurally removed.
    remained_indexes: torch.Tensor
        The indexes of values that need to be remained.
    """
    assert isinstance(t_mask, torch.Tensor)
    shape = list(t_mask.size())
    n_dims = len(shape)
    dim_list = list(range(n_dims))
    # try to reduce the mask from the 0-th dimension
    dim_list.remove(dim)

    t_merged = torch.sum(t_mask, dim_list)
    assert t_merged.size(0) == shape[dim]
    all_pruned = t_merged == 0
    need_remain = t_merged != 0
    # return the indexes of the sparsity that can be removed
    indexes = torch.nonzero(all_pruned, as_tuple=True)[0]
    remained_indexes = torch.nonzero(need_remain, as_tuple=True)[0]
    return indexes, remained_indexes


def no_replace(module, auto_infer):
    """
    No need to replace
    """
    _logger.debug("no need to replace")
    return module


def replace_linear(linear, auto_infer):
    """
    Parameters
    ----------
    linear : torch.nn.Linear
        The linear module to be replace
    auto_infer : AutoMaskInference
        The auto mask inference object that contains the input,
        parameter and output masks.

    Returns
    -------
    torch.nn.Linear
        The new linear module
    """
    assert isinstance(linear, nn.Linear)
    assert len(auto_infer.in_masks) == 0
    assert isinstance(auto_infer.output_mask, torch.tensor)
    # in_mask = auto_infer.in_masks[0]
    # output_mask = auto_infer.output_mask
    weight_mask = auto_infer.weight_mask['weight']

    pruned_in, remained_in = convert_to_coarse_mask(weight_mask, 1)
    pruned_out, remained_out = convert_to_coarse_mask(weight_mask, 0)
    n_remained_in = weight_mask.size(1) - pruned_in.size()
    n_remained_out = weight_mask.size(0) - pruned_out.size()
    remained_in, remained_out = remained_in.to(
        linear.weight.device), remained_out.to(linear.weight.device)
    _logger.info("replace linear with new in_features: %d, out_features: %d",
                 n_remained_in, n_remained_out)
    new_linear = torch.nn.Linear(in_features=n_remained_in,
                                 out_features=n_remained_out,
                                 bias=linear.bias is not None)
    new_linear.to(linear.weight.device)
    # Copy the remained weight from the original module
    with torch.no_grad():
        tmp_weight_data = torch.index_select(
            linear.weight.data, 0, remained_out)
        new_linear.weight.data = torch.index_select(
            tmp_weight_data, 1, remained_in)

        if linear.bias is not None:
            new_linear.bias.data = torch.index_select(
                linear.bias.data, 0, remained_out)
    return new_linear


def replace_batchnorm2d(norm, auto_infer):
    """
    Parameters
    ----------
    norm : torch.nn.BatchNorm2d
        The batchnorm module to be replace
    auto_infer : AutoMaskInference
        The auto mask inference object that contains the input,
        parameter and output masks.

    Returns
    -------
    torch.nn.BatchNorm2d
        The new batchnorm module
    """
    assert isinstance(norm, nn.BatchNorm2d)
    in_mask = auto_infer.in_masks[0]
    output_mask = auto_infer.output_mask
    # N, C, H, W
    _, remained_in = convert_to_coarse_mask(in_mask, 1)
    _, remained_out = convert_to_coarse_mask(output_mask, 1)
    assert remained_in.size(0) == remained_out.size(0)

    num_features = remained_in.size(0)
    _logger.info("replace batchnorm2d with num_features: %d", num_features)
    new_norm = torch.nn.BatchNorm2d(num_features=num_features,
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)
    # assign weights
    new_norm.weight.data = torch.index_select(norm.weight.data, 0, remained_in)
    new_norm.bias.data = torch.index_select(norm.bias.data, 0, remained_in)

    new_norm.running_mean.data = torch.index_select(
        norm.running_mean.data, 0, remained_in)
    new_norm.running_var.data = torch.index_select(
        norm.running_var.data, 0, remained_in)
    return new_norm


def replace_conv2d(conv, auto_infer):
    """
    Parameters
    ----------
    conv : torch.nn.Conv2d
        The conv2d module to be replaced
    auto_infer : AutoMaskInference
        The auto mask inference object that contains the mask of the input
        tensor, output tensor and parameters

    Returns
    -------
    torch.nn.Conv2d
        The new conv2d module
    """
    assert isinstance(conv, nn.Conv2d)
    # the conv layer should only have one input tensor
    assert len(auto_infer.in_masks) == 1
    assert isinstance
    in_mask = auto_infer.in_masks[0]
    output_mask = auto_infer.output_mask
    weight_mask = auto_infer.weight_mask['weight']
    pruned_in, remained_in = convert_to_coarse_mask(in_mask, 1)
    pruned_out, remained_out = convert_to_coarse_mask(output_mask, 1)
    n_remained_in = weight_mask.size(1) - pruned_in.size()
    n_remained_out = weight_mask.size(0) - pruned_out.size()
    assert n_remained_in == remained_in.size()
    assert n_remained_out == remained_out.size()
    k_size1, k_size2 = conv.kernel_size
    tmp_weight = torch.ones(n_remained_out, n_remained_in, k_size1, k_size2)
    tmp_weight = tmp_weight.to(conv.weight.device)
    # Note: We should resolve the group dependency of the conv layers before
    # run into here.
    # check if the mask tensor meets the group dependency and calculate the
    # new number of the groups after pruning
    # the original step size of the input channel for each group
    ori_inchannel_step = int(conv.in_channels/conv.groups)
    # the original step size of the output channel for each group
    ori_outchannel_step = int(conv.out_channels/conv.groups)
    new_inchannel_step = new_outchannel_step = None
    new_groups = 0
    for groupid in range(conv.groups):
        in_start = groupid * ori_inchannel_step
        in_end = in_start + ori_inchannel_step
        out_start = groupid * ori_outchannel_step
        out_end = out_start + ori_outchannel_step
        current_input_index = list(
            filter(lambda x: in_start <= x and x < in_end, remained_in.tolist()))
        current_output_index = list(
            filter(lambda x: out_start <= x and x < out_end, remained_out.tolist()))
        # remap the global index to the group index
        current_input_index = [x-in_start for x in current_input_index]
        if len(current_input_index) == 0:
            # if the whole group are pruned
            assert len(current_output_index) == 0
            continue
        # check if the number of remained channel of each group are the same
        if new_inchannel_step:
            assert len(current_input_index) == new_inchannel_step
            assert len(current_output_index) == new_outchannel_step
        else:
            # update the number of remained channels after pruning
            new_inchannel_step = len(current_input_index)
            new_outchannel_step = len(current_output_index)
        # copy the weight into tmp_weight
        new_out_start = new_outchannel_step * new_groups
        new_out_end = new_out_start + new_outchannel_step
        tmp_weight[new_out_start:new_out_end] = torch.index_select(
            conv.weight[current_output_index], 1, current_input_index)
        new_groups += 1

    _logger.debug("replace conv2d with in_channels: %d, out_channels: %d",
                  n_remained_in, n_remained_out)
    new_conv = torch.nn.Conv2d(in_channels=n_remained_in,
                               out_channels=n_remained_out,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=new_groups,
                               bias=conv.bias is not None,
                               padding_mode=conv.padding_mode)

    new_conv.to(conv.weight.device)
    new_conv.weight.copy_(tmp_weight)
    # copy the bias data
    if conv.bias is not None:
        new_conv.bias.data.copy_(torch.index_select(
            conv.bias.data, 0, remained_out))

    return new_conv
