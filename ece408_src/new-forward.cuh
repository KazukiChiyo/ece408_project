
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_SIZE 16
#define CU_MAX_THREAD 1024

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int W_grid)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    int b, m, h, w, c, p, q;
    b = blockIdx.x;
    m = blockIdx.y;
    h = blockIdx.z/W_grid*TILE_SIZE + threadIdx.y;
    w = blockIdx.z%W_grid*TILE_SIZE + threadIdx.x;
    float acc = 0.0f;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    if (h < H_out && w < W_out) {
        for (c = 0; c < C; ++c) {
            for (p = 0; p < K; ++p) {
                for (q = 0; q < K; ++q) {
                    acc += x4d(b, c, h + p, w + q)*k4d(m, c, p, q);
                }
            }
            y4d(b, m, h, w) = acc;
        }
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void __gemm(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    __shared__ float subTileM[TILE_SIZE][TILE_SIZE];
    __shared__ float subTileN[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by*TILE_SIZE + ty;
    int Col = bx*TILE_SIZE + tx;
    float Pvalue = 0.0f;

    for (int m = 0; m < ceil((float) numAColumns/TILE_SIZE); m++) {
        if (Row < numCRows && m*TILE_SIZE + tx < numAColumns)
            subTileM[ty][tx] = A[Row*numAColumns+m*TILE_SIZE+tx];
        else
            subTileM[ty][tx] = 0.0f;
        if (Col < numCColumns && m*TILE_SIZE + ty < numBRows)
            subTileN[ty][tx] = B[(m*TILE_SIZE + ty)*numBColumns+Col];
        else
            subTileN[ty][tx] = 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++)
            Pvalue += subTileM[ty][k]*subTileN[k][tx];
        __syncthreads();
    }

    if (Row < numCRows && Col < numCColumns)
        C[Row*numCColumns+Col] = Pvalue;
}


__global__ void __unroll(int C, int H, int W, int K, float* X, float* X_unroll) {
    int t = blockIdx.x*CU_MAX_THREAD + threadIdx.x;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out*W_out;
    if (t < C*W_unroll) {
        int c = t/W_unroll;
        int s = t%W_unroll;
        int h_out = s/W_out;
        int w_out = s%W_out;
        int w_unroll = h_out*W_out + w_out;
        int w_base = c*K*K;

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = w_base + p*K + q;
                X_unroll[h_unroll*W_unroll + w_unroll] = X[c*H*W + (h_out + p)*W + w_out + q];
            }
        }
    }
}


void gemm(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    dim3 dimGrid(ceil((float) numCColumns/TILE_SIZE), ceil((float) numCRows/TILE_SIZE), 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    __gemm<<<dimGrid, dimBlock>>>(A, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
}


void unroll(int C, int H, int W, int K, float* X, float* X_unroll) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int num_threads = C*H_out*W_out;
    const int num_blocks = ceil(num_threads/CU_MAX_THREAD);
    __unroll<<<num_blocks, CU_MAX_THREAD>>>(C, H, W, K, X, X_unroll);
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
    float* X_unrolled;
    cudaMalloc(&X_unrolled, W_unroll*H_unroll*sizeof(float));

    #pragma unroll
    for (int b = 0; b < B; b++) {
        float* x_dptr = &x.dptr_[b*C*H*W];
        unroll(C, H, W, K, x_dptr, X_unrolled);
        float* y_dptr = &y.dptr_[b*M*H_unroll];
        gemm(w.dptr_, X_unrolled, y_dptr, M, W_unroll, W_unroll, H_unroll, M, H_unroll);
    }
    cudaFree(X_unrolled);

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
