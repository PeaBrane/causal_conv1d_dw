import torch
from torch.nn import functional as F
from extension_cpp.ops import causal_dw_conv1d



def causal_dw_con1d_ref(input, kernel):
    return F.conv1d(F.pad(input.moveaxis(-1, -2), (3, 0)), kernel.T[:, None, :], groups=channels).moveaxis(-1, -2).contiguous()


length = 2048
channels = 512

input = torch.rand(4, length, channels, device='cuda')
kernel = torch.rand(4, channels, device='cuda')

output = causal_dw_conv1d(input, kernel)
output_ref = causal_dw_con1d_ref(input, kernel)

print((output - output_ref).abs().max())