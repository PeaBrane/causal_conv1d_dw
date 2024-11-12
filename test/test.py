import sys
import time

import torch
import triton
from torch.nn import functional as F

from extension_cpp.ops import causal_dw_conv1d


def causal_dw_conv1d_ref(input, kernel):
    input_t = input.moveaxis(-1, -2)
    output = F.conv1d(F.pad(input_t, (3, 0)), kernel.T[:, None, :], groups=channels)
    output_t = output.moveaxis(-1, -2)
    return F.silu(output_t).contiguous()


def zero_grad(*args):
    for arg in args:
        arg.grad = None


causal_dw_conv1d_compiled = torch.compile(causal_dw_conv1d_ref)

batch, length, channels = 4, 2048, 512

input = torch.rand(batch, length, channels, device='cuda', requires_grad=True, dtype=torch.half)
kernel = torch.rand(batch, channels, device='cuda', requires_grad=True)
gradient = torch.rand_like(input.detach())

with torch.autocast('cuda'):
    output = causal_dw_conv1d(input, kernel)
output_1 = output.detach().clone()
output.backward(gradient)
input_grad = input.grad.clone()
k_grad = kernel.grad.clone()
input.grad.zero_()
kernel.grad.zero_()

with torch.autocast('cuda'):
    output = causal_dw_conv1d_ref(input, kernel)
output_ref = output.detach().clone()
output.backward(gradient)
input_grad_ref = input.grad.clone()
k_grad_ref = kernel.grad.clone()

assert torch.allclose(output_1, output_ref, rtol=1e-3, atol=1e-2)
assert torch.allclose(k_grad, k_grad_ref, rtol=1e-3, atol=1e-2)
assert torch.allclose(input_grad, input_grad_ref, rtol=1e-3, atol=1e-2)


def do_backward(layer, args, gradient, forward_only=False):
    with torch.autocast('cuda'):
        if forward_only:
            with torch.no_grad():
                return layer(*args)
        output = layer(*args)
    
    output.backward(gradient)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(10, 20, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'compiled', 'cuda', 'clone'],  # Possible values for `line_arg`.
        line_names=['torch', 'compiled', 'cuda', 'clone'],  # Label name for the lines.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))


def benchmark(size, provider, forward_only=False):
    input = torch.rand((batch, size, channels), device='cuda', requires_grad=True, dtype=torch.half)
    kernel = torch.rand((4, channels), device='cuda', requires_grad=True)
    gradient = torch.rand_like(input.detach())
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: do_backward(causal_dw_conv1d_ref, (input, kernel), gradient, forward_only=forward_only), quantiles=quantiles
        )
    if provider == 'compiled':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: do_backward(causal_dw_conv1d_compiled, (input, kernel), gradient, forward_only=forward_only), quantiles=quantiles
        )
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: do_backward(causal_dw_conv1d, (input, kernel), gradient, forward_only=forward_only), quantiles=quantiles, 
        )
    if provider == 'clone':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: input.clone(), quantiles=quantiles)
        
    scale = 2 if (provider == 'clone' or forward_only) else 8
    gbps = lambda ms: scale * input.numel() * input.element_size() * 1e-9 / (ms * 1e-3)
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, save_path='.')