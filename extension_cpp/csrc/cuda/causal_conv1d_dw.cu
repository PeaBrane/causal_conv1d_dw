#include <cmath>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace extension_cpp {

#define KERNEL_SIZE 4
#define BLOCK 64
#define BL 32

int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __inline__ float sigmoid(float x) { return 1 / (1.0f + expf(-x)); }

__device__ __inline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __inline__ float silu_jacob(float x) {
  float x_sig = sigmoid(x);
  return x_sig * (1 + x * (1 - x_sig));
}

__global__ void causal_dw_conv1d_fwd_kernel(
  __half* input, const float* kernel, __half* output, int length, int chs
) {
  __shared__ __half2 s_input[BL][BLOCK / 2];
  __shared__ float s_kernel[KERNEL_SIZE][BLOCK];

  constexpr int hb = BLOCK / 2;  // this should be 32 (the warp size)
  const int tid = threadIdx.x;
  const int b_id = blockIdx.z;
  const int ch_id = blockIdx.x * BLOCK + tid * 2;

  constexpr int bl_stride = BL - KERNEL_SIZE;
  const int start_pos_id = blockIdx.y * bl_stride - KERNEL_SIZE;

  // load input block into SRAM
  for (int l = 0; l < BL; ++l) {
    int pos_id = start_pos_id + l;
    int offset = b_id * length * chs + pos_id * chs + ch_id;
    s_input[l][tid] = (pos_id >= 0 && pos_id < length && ch_id < chs) ?
                      reinterpret_cast<__half2*>(input + offset)[0] :
                      __float2half2_rn(0.0f);
  }

  // load kernel block into SRAM (transposed), no bank conflicts
  #pragma unroll
  for (int k = 0; k < KERNEL_SIZE; ++k) {
    s_kernel[k][tid] = kernel[k * chs + ch_id];
    s_kernel[k][tid + hb] = kernel[k * chs + ch_id + 1];
  }

  // compute output
  for (int l = 1; l <= BL - KERNEL_SIZE; ++l) {
    int pos_id = start_pos_id + l + KERNEL_SIZE - 1;
    int offset = b_id * length * chs + pos_id * chs + ch_id;
    float tmp1 = 0.0f, tmp2 = 0.0f;
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) { 
      tmp1 += s_kernel[k][tid] * __low2float(s_input[l+k][tid]);
      tmp2 += s_kernel[k][tid + hb] * __high2float(s_input[l+k][tid]);
    }
    __half2 pair = __floats2half2_rn(silu(tmp1), silu(tmp2));
    if (pos_id < length && ch_id < chs) {
      reinterpret_cast<__half2*>(output + offset)[0] = pair;
    }
  }
}

at::Tensor causal_dw_conv1d_fwd_cuda(const at::Tensor& input, const at::Tensor& kernel) {
  __half* input_ptr = reinterpret_cast<__half*>(input.data_ptr<at::Half>());
  const float* kernel_ptr = kernel.data_ptr<float>();
  at::Tensor output = torch::empty(input.sizes(), input.options());
  __half* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

  int batch = input.size(0);
  int length = input.size(1);
  int chs = input.size(2);

  dim3 gridDim(cdiv(chs, BLOCK), cdiv(length, BL - KERNEL_SIZE), batch);
  dim3 blockDim(BLOCK / 2, 1, 1);

  causal_dw_conv1d_fwd_kernel<<<gridDim, blockDim>>>(input_ptr, kernel_ptr, output_ptr, length, chs);
  return output;
}

__global__ void causal_dw_conv1d_bwd_kernel(
  __half* input, const float* kernel, __half* grad_output, 
  __half* grad_input, float* grad_kernel, 
  int length, int chs
) {
  __shared__ __half2 s_input[BL][BLOCK / 2];
  __shared__ float s_output[BL - KERNEL_SIZE][BLOCK];
  __shared__ float s_kernel[KERNEL_SIZE][BLOCK];

  constexpr int hb = BLOCK / 2;
  const int tid = threadIdx.x;
  const int b_id = blockIdx.z;
  const int ch_id = blockIdx.x * BLOCK + tid * 2;

  constexpr int bl_stride = BL - 2 * KERNEL_SIZE;  // halos on both sides
  const int start_pos_id = blockIdx.y * bl_stride;

  // load input block into SRAM
  for (int l = 0; l < BL; ++l) {
    int pos_id = start_pos_id + l - KERNEL_SIZE;
    int offset = b_id * length * chs + pos_id * chs + ch_id;
    s_input[l][tid] = (pos_id >= 0 && pos_id < length && ch_id < chs) ?
                      reinterpret_cast<__half2*>(input + offset)[0] :
                      __float2half2_rn(0.0f);
  }

  // load kernel block into SRAM (striped)
  #pragma unroll
  for (int k = 0; k < KERNEL_SIZE; ++k) {
    s_kernel[k][tid] = kernel[k * chs + ch_id];
    s_kernel[k][tid + hb] = kernel[k * chs + ch_id + 1];
  }

  // load grad_output block into SRAM
  for (int l = 0; l < BL - KERNEL_SIZE - 1; ++l) {
    int pos_id = start_pos_id + l;
    int offset = b_id * length * chs + pos_id * chs + ch_id;
    __half2 tmp = reinterpret_cast<__half2*>(grad_output + offset)[0];
    s_output[l][tid] = (pos_id < length && ch_id < chs) ? __low2float(tmp) : 0.0f;
    s_output[l][tid + hb] = (pos_id < length && ch_id < chs) ? __high2float(tmp) : 0.0f;
  }

  // recompute output
  for (int l = 1; l < BL - KERNEL_SIZE; ++l) {
    float tmp1 = 0.0f, tmp2 = 0.0f;
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) { 
      tmp1 += s_kernel[k][tid] * __low2float(s_input[l + k][tid]);
      tmp2 += s_kernel[k][tid+hb] * __high2float(s_input[l + k][tid]);
    }
    s_output[l-1][tid] *= silu_jacob(tmp1); s_output[l-1][tid + hb] *= silu_jacob(tmp2);
  }
  
  // compute grad_input
  for (int l = 0; l < BL - 2 * KERNEL_SIZE; ++l) {
    int store_pos_id = start_pos_id + l;
    int offset = b_id * length * chs + store_pos_id * chs + ch_id;
    float tmp1 = 0.0f, tmp2 = 0.0f;
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) { 
      tmp1 += s_kernel[KERNEL_SIZE - 1 - k][tid] * s_output[l + k][tid];
      tmp2 += s_kernel[KERNEL_SIZE - 1 - k][tid + hb] * s_output[l + k][tid + hb];
    }
    __half2 pair = __floats2half2_rn(tmp1, tmp2);
    if (store_pos_id < length && ch_id < chs) {
      reinterpret_cast<__half2*>(grad_input + offset)[0] = pair;
    }
  }

  // compute grad_kernel
  for (int k = 0; k < KERNEL_SIZE; ++k) {
    int offset = (KERNEL_SIZE - 1 - k) * gridDim.z * gridDim.y * chs + b_id * gridDim.y * chs + blockIdx.y * chs + ch_id;
    float tmp1 = 0.0f, tmp2 = 0.0f;
    for (int l = 0; l < BL - 2 * KERNEL_SIZE; ++l) {
      tmp1 += __low2float(s_input[l + KERNEL_SIZE][tid]) * s_output[l + k][tid];
      tmp2 += __high2float(s_input[l + KERNEL_SIZE][tid]) * s_output[l + k][tid + hb];
    }
    if (ch_id < chs) {
      grad_kernel[offset] = tmp1; grad_kernel[offset+1] = tmp2;
    }
  }
}

void causal_dw_conv1d_bwd_cuda(
  const at::Tensor& input, const at::Tensor& kernel, const at::Tensor& grad_output, 
  at::Tensor& grad_input, at::Tensor& grad_kernel
) {
  // const float* input_ptr = input.data_ptr<float>();
  __half* input_ptr = reinterpret_cast<__half*>(input.data_ptr<at::Half>());
  __half* grad_output_ptr = reinterpret_cast<__half*>(grad_output.data_ptr<at::Half>());
  const float* kernel_ptr = kernel.data_ptr<float>();
  __half* grad_input_ptr = reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>());
  float* grad_kernel_ptr = grad_kernel.data_ptr<float>();

  int batch = grad_output.size(0);
  int length = grad_output.size(1);
  int chs = grad_output.size(2);

  dim3 gridDim(cdiv(chs, BLOCK), cdiv(length, BL - 2 * KERNEL_SIZE), batch);
  dim3 blockDim(BLOCK / 2, 1, 1);

  causal_dw_conv1d_bwd_kernel<<<gridDim, blockDim>>>(
    input_ptr, kernel_ptr, grad_output_ptr, grad_input_ptr, grad_kernel_ptr, length, chs
  );
}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("causal_dw_conv1d_fwd", &causal_dw_conv1d_fwd_cuda);
  m.impl("causal_dw_conv1d_bwd", &causal_dw_conv1d_bwd_cuda);
}

}
