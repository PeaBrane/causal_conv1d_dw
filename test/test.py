import torch
from torch.nn import functional as F
from extension_cpp.ops import causal_dw_conv1d
import triton
import sys

torch.set_grad_enabled(False)


def causal_dw_conv1d_ref(input, kernel):
    input = input.float()
    input_t = input.moveaxis(-1, -2)
    output = F.conv1d(F.pad(input_t, (3, 0)), kernel.T[:, None, :], groups=channels)
    output_t = output.moveaxis(-1, -2)
    return F.silu(output_t).contiguous().half()


causal_dw_conv1d_compiled = torch.compile(causal_dw_conv1d_ref)


batch = 4
length = 2048
channels = 512

input = torch.rand(batch, length, channels, device='cuda', dtype=torch.half)
kernel = torch.rand(batch, channels, device='cuda')

output = causal_dw_conv1d(input, kernel)
output_ref = causal_dw_conv1d_ref(input, kernel)

print((output - output_ref).abs().max())

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(10, 16, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'compiled', 'dummy', 'cuda'],  # Possible values for `line_arg`.
        line_names=['torch', 'compiled', 'dummy', 'cuda'],  # Label name for the lines.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))


def benchmark(size, provider):
    input = torch.rand((batch, size, channels), device='cuda', dtype=torch.half)
    kernel = torch.rand((4, channels), device='cuda')
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: causal_dw_conv1d_ref(input, kernel), quantiles=quantiles)
    if provider == 'compiled':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: causal_dw_conv1d_compiled(input, kernel), quantiles=quantiles)
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: causal_dw_conv1d(input, kernel), quantiles=quantiles)
    if provider == 'dummy':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: input.clone(), quantiles=quantiles)
    gbps = lambda ms: 2 * input.numel() * input.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, save_path='.')