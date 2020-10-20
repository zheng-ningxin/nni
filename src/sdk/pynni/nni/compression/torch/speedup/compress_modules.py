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
    tmp_dim_list = dim_list.remove(dim)
    t_merged = torch.sum(t_mask, tmp_dim_list)
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
            new_linear.bias.data = torch.index_select(linear.bias.data, 0, remained_out)
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


def replace_conv2d(conv, mask):
    """
    Parameters
    ----------
    conv : torch.nn.Conv2d
        The conv2d module to be replaced
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.Conv2d
        The new conv2d module
    """
    assert isinstance(mask, ModuleMasks)
    if mask.input_mask is None:
        in_channels = conv.in_channels
    else:
        in_channels_index = mask.input_mask.mask_index[1]
        in_channels = in_channels_index.size()[0]
    if mask.output_mask is None:
        out_channels = conv.out_channels
    else:
        out_channels_index = mask.output_mask.mask_index[1]
        out_channels = out_channels_index.size()[0]

    _logger.debug("replace conv2d with in_channels: %d, out_channels: %d",
                  in_channels, out_channels)
    new_conv = torch.nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=conv.bias is not None,
                               padding_mode=conv.padding_mode)

    new_conv.to(conv.weight.device)
    tmp_weight_data = tmp_bias_data = None

    if mask.output_mask is not None:
        tmp_weight_data = torch.index_select(
            conv.weight.data, 0, out_channels_index)
        if conv.bias is not None:
            tmp_bias_data = torch.index_select(
                conv.bias.data, 0, out_channels_index)
    else:
        tmp_weight_data = conv.weight.data
    # For the convolutional layers that have more than one group
    # we need to copy the weight group by group, because the input
    # channal is also divided into serveral groups and each group
    # filter may have different input channel indexes.
    input_step = int(conv.in_channels / conv.groups)
    in_channels_group = int(in_channels / conv.groups)
    filter_step = int(out_channels / conv.groups)
    if mask.input_mask is not None:
        for groupid in range(conv.groups):
            start = groupid * input_step
            end = (groupid + 1) * input_step
            current_input_index = list(
                filter(lambda x: start <= x and x < end, in_channels_index.tolist()))
            # shift the global index into the group index
            current_input_index = [x-start for x in current_input_index]
            # if the groups is larger than 1, the input channels of each
            # group should be pruned evenly.
            assert len(current_input_index) == in_channels_group, \
                'Input channels of each group are not pruned evenly'
            current_input_index = torch.tensor(current_input_index).to(
                tmp_weight_data.device)  # pylint: disable=not-callable
            f_start = groupid * filter_step
            f_end = (groupid + 1) * filter_step
            new_conv.weight.data[f_start:f_end] = torch.index_select(
                tmp_weight_data[f_start:f_end], 1, current_input_index)
    else:
        new_conv.weight.data.copy_(tmp_weight_data)

    if conv.bias is not None:
        new_conv.bias.data.copy_(
            conv.bias.data if tmp_bias_data is None else tmp_bias_data)

    return new_conv
