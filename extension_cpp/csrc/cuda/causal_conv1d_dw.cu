#include <cmath>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace extension_cpp {


#define KERNEL_SIZE 4
#define BLOCK 32


int cdiv(int a, int b) {
  return (a + b - 1) / b;
}


__device__ __inline__ float sigmoid(float x) {
  return 1 / (1.0f + expf(-x));
}


__device__ __inline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}


__device__ __inline__ float silu_jacob(float x) {
  float x_sig = sigmoid(x);
  return x_sig * (1 + x * (1 - x_sig));
}


__global__ void causal_dw_conv1d_fwd_kernel(
  const float* input, const float* kernel, float* output, int length, int chs
) {
  __shared__ float s_input[BLOCK][BLOCK];
  __shared__ float s_kernel[KERNEL_SIZE][BLOCK];

  constexpr int bl_stride = BLOCK - KERNEL_SIZE;
  const int b_id = blockIdx.z;
  const int start_pos_id = blockIdx.y * bl_stride - KERNEL_SIZE;
  const int start_ch_id = blockIdx.x * blockDim.x;
  const int ch_id = start_ch_id + threadIdx.x;

  // load input block into SRAM
  for (int l = 0; l < BLOCK; ++l) {
    int pos_id = start_pos_id + l;
    if (pos_id >= 0 && pos_id < length && ch_id < chs) {
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

  // compute output
  for (int l = 1; l <= BLOCK - KERNEL_SIZE; ++l) {
    int store_pos_id = start_pos_id + l + KERNEL_SIZE - 1;
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) {
      sum += s_kernel[k][threadIdx.x] * (s_input[l + k][threadIdx.x]);
    }
    if (store_pos_id < length && ch_id < chs) {
      output[b_id * length * chs + store_pos_id * chs + ch_id] = silu(sum);
    }
  }

}


at::Tensor causal_dw_conv1d_fwd_cuda(const at::Tensor& input, const at::Tensor& kernel) {
  at::Tensor output = torch::empty(input.sizes(), input.options());
  const float* input_ptr = input.data_ptr<float>();
  const float* kernel_ptr = kernel.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();
  // const __half* input_ptr = reinterpret_cast<const __half*>(input_contig.data_ptr<at::Half>());
  // const float* kernel_ptr = kernel_contig.data_ptr<float>();
  // __half* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

  int batch = input.size(0);
  int length = input.size(1);
  int chs = input.size(2);

  dim3 gridDim(cdiv(chs, BLOCK), cdiv(length, BLOCK - KERNEL_SIZE), batch);
  dim3 blockDim(BLOCK, 1, 1);

  causal_dw_conv1d_fwd_kernel<<<gridDim, blockDim>>>(input_ptr, kernel_ptr, output_ptr, length, chs);
  return output;
}


__global__ void causal_dw_conv1d_bwd_kernel(
  const float* input, const float* kernel, const float* grad_output, 
  float* grad_input, float* grad_kernel, 
  int length, int chs
) {
  __shared__ float s_input[BLOCK][BLOCK];
  __shared__ float s_output[BLOCK-KERNEL_SIZE][BLOCK];
  __shared__ float s_kernel[KERNEL_SIZE][BLOCK];
  __shared__ float s_grad_output[BLOCK-KERNEL_SIZE][BLOCK];

  constexpr int bl_stride = BLOCK - 2 * KERNEL_SIZE;  // halos on both sides
  const int b_id = blockIdx.z;
  const int start_pos_id = blockIdx.y * bl_stride;
  const int start_ch_id = blockIdx.x * blockDim.x;
  const int ch_id = start_ch_id + threadIdx.x;

  // load input block into SRAM
  for (int l = 0; l < BLOCK; ++l) {
    int pos_id = start_pos_id + l - KERNEL_SIZE;
    if (pos_id >= 0 && pos_id < length && ch_id < chs) {
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

  // recompute output
  for (int l = 1; l < BLOCK - KERNEL_SIZE; ++l) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) {
      sum += s_kernel[k][threadIdx.x] * (s_input[l + k][threadIdx.x]);
    }
    s_output[l-1][threadIdx.x] = silu_jacob(sum);
  }

  __syncthreads();

  // load and modify grad_output block into SRAM
  for (int l = 0; l < BLOCK - KERNEL_SIZE - 1; ++l) {
    int pos_id = start_pos_id + l;
    if (pos_id < length && ch_id < chs) {
      int load_id = b_id * length * chs + pos_id * chs + ch_id;
      s_grad_output[l][threadIdx.x] = grad_output[load_id] * s_output[l][threadIdx.x];
    } else {
      s_grad_output[l][threadIdx.x] = 0.0f;
    }
  }

  __syncthreads();
  
  // compute grad_input
  for (int l = 0; l < BLOCK - 2 * KERNEL_SIZE; ++l) {
    int store_pos_id = start_pos_id + l;
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) {
      sum += s_kernel[KERNEL_SIZE - 1 - k][threadIdx.x] * s_grad_output[l + k][threadIdx.x];
    }
    if (store_pos_id < length && ch_id < chs) {
      grad_input[b_id * length * chs + store_pos_id * chs + ch_id] = sum;
    }
  }

  // compute grad_kernel
  for (int k = 0; k < KERNEL_SIZE; ++k) {
    int store_id = (KERNEL_SIZE - 1 - k) * gridDim.z * gridDim.y * chs + b_id * gridDim.y * chs + blockIdx.y * chs + ch_id;
    float sum = 0.0f;
    for (int l = 0; l < BLOCK - 2 * KERNEL_SIZE; ++l) {
      sum += s_input[l + KERNEL_SIZE][threadIdx.x] * s_grad_output[l + k][threadIdx.x];
    }
    grad_kernel[store_id] = sum;
  }
}


void causal_dw_conv1d_bwd_cuda(
  const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& grad_output, 
  at::Tensor& grad_input, at::Tensor& grad_kernel
) {
  const float* input_ptr = input.data_ptr<float>();
  const float* grad_output_ptr = grad_output.data_ptr<float>();
  const float* kernel_ptr = kernel.data_ptr<float>();
  float* grad_input_ptr = grad_input.data_ptr<float>();
  float* grad_kernel_ptr = grad_kernel.data_ptr<float>();

  int batch = grad_output.size(0);
  int length = grad_output.size(1);
  int chs = grad_output.size(2);

  dim3 gridDim(cdiv(chs, BLOCK), cdiv(length, BLOCK - 2 * KERNEL_SIZE), batch);
  dim3 blockDim(BLOCK, 1, 1);

  causal_dw_conv1d_bwd_kernel<<<gridDim, blockDim>>>(
    input_ptr, kernel_ptr, grad_output_ptr, grad_input_ptr, grad_kernel_ptr, length, chs
  );
}


// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("causal_dw_conv1d_fwd", &causal_dw_conv1d_fwd_cuda);
  m.impl("causal_dw_conv1d_bwd", &causal_dw_conv1d_bwd_cuda);
}

}
