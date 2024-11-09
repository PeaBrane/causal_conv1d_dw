import torch


class CausalDwConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel):
        ctx.save_for_backward(kernel)
        return torch.ops.extension_cpp.causal_dw_conv1d_fwd.default(input, kernel)
        
    @staticmethod
    def backward(ctx, grad_output):
        kernel, = ctx.saved_tensors
        grad_input = torch.ops.extension_cpp.causal_dw_conv1d_bwd.default(grad_output, kernel)
        return grad_input, torch.zeros_like(kernel)



causal_dw_conv1d = CausalDwConv1d.apply
