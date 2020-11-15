# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import logging
from functools import partial
import torch
import torchvision

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def translate_list(list_node, speedup=None):
    """
    Get the list of values from the list construct node.
    Parameters
    ---------
    list_node: Torch.C.Value
        The cpp node of the target list.
    speedup: ModuleSpeed
        The Module speedup module.
    Returns
    -------
    values: list
        The list of values in the target cpp list node.
    """
    # the node that create the list
    create_node = list_node.node()
    assert create_node.kind() == 'prim::ListConstruct'
    inputs = list(create_node.inputs())
    values = []
    for _i in inputs:
        debugName = _i.debugName()
        if speedup is not None and debugName in speedup.internal_result:
            # this value is the result of the other nodes, such as
            # ate::size
            values.append(speedup.internal_result[debugName])
        else:
            # if the corresponding value is a constant
            values.append(_i.toIValue())
    return values


def dropout_python(node, speedup):
    return torch.dropout


def flatten_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    start_dim = inputs[1].toIValue()
    end_dim = inputs[2].toIValue()
    new_flatten = partial(torch.flatten, start_dim=start_dim, end_dim=end_dim)
    return new_flatten


def relu_inplace_python(node, speedup):
    return torch.relu_


def relu_python(node, speedup):
    return torch.relu


def mean_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_list = translate_list(inputs[1], speedup)
    keep_dim = inputs[2].toIValue()
    print(dim_list)
    print(keep_dim)
    new_mean = partial(torch.mean, dim=tuple(dim_list), keepdim=keep_dim)
    return new_mean


def add_python(node, speedup):
    return torch.add


def slice_python(node, speedup):
    class SliceMoudle(torch.nn.Module):
        def __init__(self, sliceobj):
            super(SliceMoudle, self).__init__()
            self.sliceobj = sliceobj

        def forward(self, x):
            return x[self.sliceobj]

    c_node = node.key_node
    inputs = list(c_node.inputs())
    print(inputs)
    slice_dim = inputs[1].toIValue()
    slice_start = inputs[2].toIValue()
    slice_end = inputs[3].toIValue()
    slice_step = inputs[4].toIValue()
    slice_obj = slice(slice_start, slice_end, slice_step)
    slice_list = []
    for i in range(slice_dim-1):
        slice_list.append(slice(None, None))
    slice_list.append(slice_obj)
    return SliceMoudle(tuple(slice_list))


def size_python(node, speedup):
    # return None
    class SizeMoudle(torch.nn.Module):
        def __init__(self, sizedim):
            super(SizeMoudle, self).__init__()
            self.sizedim = sizedim

        def forward(self, x):
            return torch.tensor(x.size(self.sizedim))
            # return x.size(self.sizedim)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    size_dim = inputs[1].toIValue()
    return SizeMoudle(size_dim)


def transpose_python(node, speedup):
    return torch.t


def transpose2_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_1 = inputs[1].toIValue()
    dim_2 = inputs[2].toIValue()
    new_transpose = partial(torch.transpose, dim0=dim_1, dim1=dim_2)
    return new_transpose


def toint_python(node, speedup):
    class ToIntModule(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.int)
    return ToIntModule()


def view_python(node, speedup):
    class ViewModule(torch.nn.Module):
        def __init__(self, shape):
            super(ViewModule, self).__init__()
            self.shape = shape
            print(self.shape)
        def forward(self, x):
            print(x)
            # exit()
            # print(args[0])
            return x.view(self.shape)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    shape = translate_list(inputs[1], speedup)
    return ViewModule(shape)

def permute_python(node, speedup):
    class PermuteModule(torch.nn.Module):
        def __init__(self, dimlist):
            super(PermuteModule, self).__init__()
            self.dimlist = dimlist
        def forward(self, x):
            return x.permute(self.dimlist)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_list = translate_list(inputs[1], speedup)
    return PermuteModule(dim_list)

def constructlist_python(node, speedup):
    class ListModule(torch.nn.Module):
        def forward(self, *args):
            return args

trans_from_jit_to_python = {
    # 'aten::cat': cat_python,
    'aten::add': add_python,
    'aten::add_': add_python,
    'aten::relu': relu_python,
    # 'aten::tanh': tanh_python,
    # 'aten::tanh_': tanh_python,
    'aten::flatten': flatten_python,
    'aten::mean': mean_python,
    'aten::dropout': dropout_python,
    'aten::relu_': relu_inplace_python,
    'aten::slice': slice_python,
    'aten::size': size_python,
    'aten::t': transpose_python,
    'aten::transpose': transpose2_python,
    'aten::Int': toint_python,
    'aten::view': view_python
}


def jit_to_python_function(node, speedup):
    logger.debug(
        'Translate C function %s into its python version', node.op_type)
    return trans_from_jit_to_python[node.op_type](node, speedup)
