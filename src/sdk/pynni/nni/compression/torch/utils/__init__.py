# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch

torch_float_dtype = [torch.float, torch.float16, torch.float32, torch.float64, torch.half, torch.double]
torch_integer_dtype = [torch.uint8, torch.int16, torch.short, torch.int16, torch.long, torch.bool]

def rand_like_with_shape(shape, ori_t):
    assert isinstance(ori_t, torch.Tensor)
    device = ori_t.device
    dtype = ori_t.dtype
    require_grad = ori_t.requires_grad
    lower_bound = torch.min(ori_t)
    higher_bound = torch.max(ori_t)
    if dtype in [torch.uint8, torch.int16, torch.short, torch.int16, torch.long, torch.bool]:
        return torch.randint(lower_bound, higher_bound+1, shape, dtype=dtype, device=device)
    else:
        return torch.rand(shape, dtype=dtype, device=device, requires_grad=require_grad)