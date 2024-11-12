#include <torch/extension.h>

#include <vector>

namespace extension_cpp {

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
  m.def("causal_dw_conv1d_fwd(Tensor input, Tensor kernel) -> Tensor");
  m.def("causal_dw_conv1d_bwd(Tensor input, Tensor kernel, Tensor grad_output, Tensor(a!) grad_input, Tensor(b!) grad_kernel) -> ()");
}

}
