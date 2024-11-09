import torch
from torch.nn import functional as F
from extension_cpp.ops import causal_dw_conv1d


input = torch.rand(4, 2048, 512, device='cuda')
kernel = torch.rand(4, 512, device='cuda')

output = causal_dw_conv1d(input, kernel)
output_ref = F.conv1d(F.pad(input.moveaxis(-1, -2), (3, 0)), kernel.T[:, None, :], groups=512).moveaxis(-1, -2)

print((output - output_ref).abs().max())