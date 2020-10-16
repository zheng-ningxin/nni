# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch


def cat_conflict(node, input_masks, output_masks):
    pass

def add_conflict(node, input_masks, output_mask):
    """
    Find the positions of the input masks that need
    to be unmasked to resolve the mask conflicts.
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
