# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import logging
from functools import partial
import torch
import torchvision

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def dropout_python(node):
    return torch.dropout

def flatten_python(node):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    start_dim = inputs[1].toIValue()
    end_dim = inputs[2].toIValue()
    new_flatten = partial(torch.flatten, start_dim=start_dim, end_dim=end_dim)
    return new_flatten

def relu_inplace_python(node):
    return torch.relu_

def relu_python(node):
    return torch.relu

def mean_python(node):
    return torch.mean

def add_python(node):
    return torch.add

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
    'aten::relu_': relu_inplace_python
}

def jit_to_python_function(node):
    logger.debug('Translate C function %s into its python version', node.op_type)
    return trans_from_jit_to_python[node.op_type](node)