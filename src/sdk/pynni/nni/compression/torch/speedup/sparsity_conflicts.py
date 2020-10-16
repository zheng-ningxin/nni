# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch


def cat_conflict(node, input_masks, output_masks):
    # TODO does cat need to resolve the mask conflict?
    pass

def add_conflict(node, input_masks, output_mask):
    """
    Find the positions of the input masks that need
    to be unmasked to resolve the mask conflicts.

    Parameters
    ----------
    node: NodePyGroup
        The add operation node that to resolve conflict.
    input_masks: list
        List of tensors, each element corresponds to a mask tensor of inputs.
    output_mask: torch.Tensor
        The mask of the output tensor.
    Returns
    ------
    need_unmask: list
        This list has the same length with input masks. The element of the list
        will be None or torch.Tensor, if it is None, then the corresponding input
        mask doesn't need to unmask any value, else we should unmask the values in the
        tensor.
    """
    # in the add operation, we should align the input mask
    # with the output mask.
    assert isinstance(input_masks, list)
    assert isinstance(output_mask, torch.Tensor)
    need_unmask = []
    for t_in in input_masks:
        # find the position that was masked(0) in the input tensor
        # but not masked in the output tensor(1)
        need_unmask.append(output_mask - t_in)
    for i, t_unmask in enumerate(need_unmask):
        if torch.sum(t_unmask) == 0:
            # no need to unmask any value
            need_unmask[i] = None
    return need_unmask

ConflictUnmask = {
    'aten::cat': cat_conflict,
    'aten::add': add_conflict,
    'aten::add_': add_conflict,
    'aten::mul': add_conflict,
    'aten::mul_': add_conflict

}

def calc_unmask(node, input_masks, output_mask):
    cacl_func = ConflictUnmask[node.op_type]
    return cacl_func(node, input_masks, output_mask)
