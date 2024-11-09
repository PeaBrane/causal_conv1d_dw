#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp {


#define KERNEL_SIZE 4
#define BLOCK 32


int cdiv(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void causal_dw_conv1d_kernel(
  const float* input, const float* kernel, float* output, int length, int chs
) {
  __shared__ float s_input[BLOCK][BLOCK+1];
  __shared__ float s_kernel[KERNEL_SIZE][BLOCK];

  const int b_id = blockIdx.z;
  const int bl_stride = BLOCK - KERNEL_SIZE + 1;
  const int start_pos_id = blockIdx.y * bl_stride - KERNEL_SIZE + 1;
  const int start_ch_id = blockIdx.x * blockDim.x;
  int ch_id = start_ch_id + threadIdx.x;

  // load input block into SRAM
  for (int l = 0; l < BLOCK; ++l) {
    int pos_id = start_pos_id + l;
    if (pos_id >= 0 && pos_id < length) {
      s_input[l][threadIdx.x] = input[b_id * length * chs + pos_id * chs + ch_id];
    } else {
      s_input[l][threadIdx.x] = 0.0f;
    }
  }

  // load kernel block into SRAM
  for (int k = 0; k < KERNEL_SIZE; ++k) {
    s_kernel[k][threadIdx.x] = kernel[k * chs + ch_id];
  }

  __syncthreads();

  int pos_id = start_pos_id + threadIdx.x;
  int store_pos_id = pos_id + KERNEL_SIZE - 1;

  // circular convolution with masked write to output
  for (int c = 0; c < BLOCK; ++c) {
    ch_id = start_ch_id + c;
    float sum = 0.0f;
    for (int k = 0; k < KERNEL_SIZE; ++k) {
      int l_index = (threadIdx.x + k) % BLOCK;
      sum += s_kernel[k][c] * s_input[l_index][c];
    }
    if (threadIdx.x <= BLOCK - KERNEL_SIZE && store_pos_id < length) {
      output[b_id * length * chs + store_pos_id * chs + ch_id] = sum;
    }
  }

}


at::Tensor causal_dw_conv1d_cuda(const at::Tensor& input, const at::Tensor& kernel) {
  at::Tensor input_contig = input.contiguous();
  at::Tensor kernel_contig = kernel.contiguous();
  at::Tensor output = torch::empty(input.sizes(), input.options());
  const float* input_ptr = input_contig.data_ptr<float>();
  const float* kernel_ptr = kernel_contig.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();

  int batch = input_contig.size(0);
  int length = input_contig.size(1);
  int chs = input_contig.size(2);

  dim3 gridDim(cdiv(chs, BLOCK), cdiv(length, BLOCK - KERNEL_SIZE + 1), batch);
  dim3 blockDim(BLOCK, 1, 1);

  causal_dw_conv1d_kernel<<<gridDim, blockDim>>>(input_ptr, kernel_ptr, output_ptr, length, chs);
  return output;
}


__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  int numel = a_contig.numel();
  muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);
  return result;
}

__global__ void mul_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx];
}

at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = a_contig.numel();
  mul_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
  return result;
}

__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  int numel = a_contig.numel();
  add_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
}


// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
  m.impl("mymul", &mymul_cuda);
  m.impl("myadd_out", &myadd_out_cuda);
  m.impl("causal_dw_conv1d", &causal_dw_conv1d_cuda);
}

}
