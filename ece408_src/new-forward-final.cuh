
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_SIZE 16
#define CU_MAX_THREAD 1024

/* kernel buffer elelment size */
#define FAKE_K (5)
#define FAKE_M (10)
#define FAKE_C (20)
#define TOTAL_KERNEL_SIZE (FAKE_M * FAKE_K * FAKE_K * FAKE_C)

namespace mxnet
{
namespace op
{
__constant__ float cons_mem[TOTAL_KERNEL_SIZE];

/* shared memory */
__global__ void forward_kernel_shmem(float *y, const float *x, const int H, const int W, const int M, const int C, const int K, const int B, const int W_grid)
{
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
	#define k4d_constant(i3, i2, i1, i0) cons_mem[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  int n, m, h0, w0, h_base, w_base, h, w;
  int X_tile_width = TILE_SIZE + K - 1;

  extern __shared__ float shmem[];
  float * X_shared = &shmem[0];
  int k_start = X_tile_width * X_tile_width;
  float * K_shared = &shmem[k_start];

  n = blockIdx.x;
  m = blockIdx.y;
  h0 = threadIdx.x;
  w0 = threadIdx.y;
  h_base = blockIdx.z / W_grid * TILE_SIZE;
  w_base = blockIdx.z % W_grid * TILE_SIZE;
  h = h_base + h0;
  w = w_base + w0;

  float acc = 0.0f;

  int c, i, ii, j, p, pp, q;
  for (c = 0; c < C; c++) {
    if (h0 < K && w0 < K)
        K_shared[h0*K+w0] = k4d(m, c, h0, w0);
		__syncthreads();

		// thread block size should be TILF_WIDTH * TILE_SIZE, and the data may be reload here,
    for (i = h, ii = h0; i < h_base + X_tile_width; i += TILE_SIZE, ii += TILE_SIZE) {
      for (j = w; j < w_base + X_tile_width; j += TILE_SIZE) {
        X_shared[ii*X_tile_width+j-w_base] = x4d(n, c, i, j);
      }
    }
		__syncthreads();

		// TODO: change x4d to X_shared index.    original: acc += x4d(b, c, h + p, w + q) * k4d_constant(m, c, p, q);
	  for (p = 0; p < K; p++) {
        for (q = 0; q < K; q++) {
          // acc += x2d_shmem(w0+p, h0+q) * k2d_shmem(p, q);
          acc += X_shared[(h0 + p)*X_tile_width + w0 + q] * K_shared[p*K + q];
        }
      }
  }
  if (h < H_out && w < W_out)
		y4d(n, m, h, w) = acc;

  #undef y4d
  #undef x4d
  #undef k4d
	#undef k4d_constant
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    const int B = x.shape_[0]; // batch_size
    const int C = x.shape_[1]; // in_channels
    const int H = x.shape_[2]; // height
    const int W = x.shape_[3]; // weight
    const int M = y.shape_[1]; // out_channels
    const int K = w.shape_[3]; // kernel_size
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = C*K*K;
    const int H_unroll = H_out*W_out;

  const int H_grid = ceil((float)H_out / TILE_SIZE);
  const int W_grid = ceil((float) W_out / TILE_SIZE);
  const int Z =  H_grid * W_grid;
 
  dim3 gridDim(B, M, Z);
  dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);

    /*  shared memory optimization */
/*   
 	forward_kernel_shmem<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, H, W, M, C, K, W_grid);
 */   
/* constant memory optimization */
/*
    forward_kernel_consmem<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B, M, C, H, W, K, W_grid);
*/
		size_t shmem_size = sizeof(float)  * ((TILE_SIZE + K - 1) * (TILE_SIZE + K -1 ) + K * K);
		int TRUE_KERNEL_SIZE = K * K * M * C;
		cudaMemcpyToSymbol(cons_mem, w.dptr_, sizeof(float) * TRUE_KERNEL_SIZE);
		forward_kernel_final<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, H, W, M, C, K, B, W_grid);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#undef TILE_SIZE

#endif
