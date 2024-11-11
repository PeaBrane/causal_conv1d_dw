import torch
from triton import cdiv


BLOCK = 32
KERNEL_SIZE = 4


class CausalDwConv1d(torch.autograd.Function):
    @torch.amp.custom_fwd(device_type='cuda')
    @staticmethod
    def forward(ctx, input: torch.Tensor, kernel: torch.Tensor):
        ctx.save_for_backward(input, kernel)
        return torch.ops.extension_cpp.causal_dw_conv1d_fwd.default(input, kernel)
    
    @torch.amp.custom_bwd(device_type='cuda')
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, kernel = ctx.saved_tensors
        assert grad_output.is_contiguous()
        batch, length, channels = grad_output.shape
        grad_input = torch.empty_like(grad_output)
        grad_kernel = torch.empty((KERNEL_SIZE, batch, cdiv(length, BLOCK - 2 * KERNEL_SIZE), channels), 
                                  dtype=kernel.dtype, device=kernel.device)
        
        torch.ops.extension_cpp.causal_dw_conv1d_bwd.default(input, kernel, grad_output, grad_input, grad_kernel)
        grad_kernel = grad_kernel.sum((1, 2))  # (k, chs)
        return grad_input, grad_kernel



causal_dw_conv1d = CausalDwConv1d.apply
