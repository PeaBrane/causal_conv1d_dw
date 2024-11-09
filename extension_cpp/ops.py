import torch
from torch import Tensor


def causal_dw_conv1d(input, kernel):
    return torch.ops.extension_cpp.causal_dw_conv1d.default(input, kernel)
