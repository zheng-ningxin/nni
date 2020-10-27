# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .infer_mask import AutoMaskInferenceRemove
from .infer_mask import AutoMaskInferenceZero

AutoMaskInferenceType={
    'aten::add': AutoMaskInferenceZero,
    'aten::add_': AutoMaskInferenceZero,

    'aten::relu_': AutoMaskInferenceRemove,
    'aten::relu': AutoMaskInferenceRemove,
    'ReLU': AutoMaskInferenceRemove,
    'ReLU6': AutoMaskInferenceRemove,


    'Conv2d': AutoMaskInferenceZero,
    'Linear': AutoMaskInferenceZero,

    'aten::flatten': AutoMaskInferenceZero,
    'aten::mean': AutoMaskInferenceZero,
    'MaxPool2d': AutoMaskInferenceZero,
    'AdaptiveAvgPool2d': AutoMaskInferenceZero,
    'Dropout': AutoMaskInferenceRemove,
    # 'aten::dropout': AutoMaskInferenceRemove,

    'BatchNorm2d': AutoMaskInferenceRemove
}