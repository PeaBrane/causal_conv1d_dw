#include <torch/extension.h>

#include <vector>

namespace extension_cpp {

// at::Tensor causal_dw_conv1d_fwd(const at::Tensor& input, const at::Tensor& kernel) {
//   return input;
// }

// at::Tensor causal_dw_conv1d_bwd(const at::Tensor& grad_output, const at::Tensor& kernel) {
//   return grad_output;
// }

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
  m.def("causal_dw_conv1d_fwd(Tensor input, Tensor kernel) -> Tensor");
  m.def("causal_dw_conv1d_bwd(Tensor grad_output, Tensor kernel) -> Tensor");
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
// TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
//   m.impl("causal_dw_conv1d_fwd", &causal_dw_conv1d_fwd);
//   m.impl("causal_dw_conv1d_bwd", &causal_dw_conv1d_bwd);
// }

}
