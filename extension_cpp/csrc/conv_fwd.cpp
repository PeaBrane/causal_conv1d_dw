#include <torch/extension.h>

#include <vector>

namespace extension_cpp {

at::Tensor causal_dw_conv1d(const at::Tensor& input, const at::Tensor& kernel) {
  return input;
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
  m.def("causal_dw_conv1d(Tensor input, Tensor kernel) -> Tensor");
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("causal_dw_conv1d", &causal_dw_conv1d);
}

}
