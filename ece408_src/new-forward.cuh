
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_SIZE_1 12
#define TILE_SIZE_2 24
#define CU_MAX_THREAD 1024

/* kernel buffer elelment size */
#define TOTAL_KERNEL_SIZE 5000

namespace mxnet
{
    namespace op
    {
        __constant__ float cons_mem[TOTAL_KERNEL_SIZE];

        /* shared memory */
        __global__ void __shmem_1(float *y, const float *x, const int H, const int W, const int M, const int C, const int K, const int B, const int W_grid) {
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
            #define k4d_constant(i3, i2, i1, i0) cons_mem[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            int n, m, h0, w0, h_base, w_base, h, w;
            int X_tile_width = TILE_SIZE_1 + K - 1;
            extern __shared__ float shmem[];
            float * X_shared = &shmem[0];

            n = blockIdx.x;
            m = blockIdx.y;
            h0 = threadIdx.x;
            w0 = threadIdx.y;
            h_base = blockIdx.z / W_grid * TILE_SIZE_1;
            w_base = blockIdx.z % W_grid * TILE_SIZE_1;
            h = h_base + h0;
            w = w_base + w0;

            float acc = 0.0f;

            int c, i, ii, j, p, q;
            #pragma unroll 5
            for (c = 0; c < C; c++) {

                // thread block size should be TILF_WIDTH * TILE_SIZE, and the data may be reload here
                for (i = h, ii = h0; i < h_base + X_tile_width; i += TILE_SIZE_1, ii += TILE_SIZE_1) {
                    for (j = w; j < w_base + X_tile_width; j += TILE_SIZE_1) {
                        X_shared[ii*X_tile_width+j-w_base] = x4d(n, c, i, j);
                    }
                }
                __syncthreads();

                #pragma unroll 5
                for (p = 0; p < K; p++) {
                    #pragma unroll 5
                    for (q = 0; q < K; q++) {
                        acc += X_shared[(h0 + p)*X_tile_width + w0 + q]*k4d_constant(m, c, p, q);
                    }
                }
                __syncthreads();
            }

            if (h < H_out && w < W_out)
                y4d(n, m, h, w) = acc;

            #undef y4d
            #undef x4d
            #undef k4d
            #undef k4d_constant
        }

        /* shared memory */
        __global__ void __shmem_2(float *y, const float *x, const int H, const int W, const int M, const int C, const int K, const int B, const int W_grid) {
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
            #define k4d_constant(i3, i2, i1, i0) cons_mem[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            int n, m, h0, w0, h_base, w_base, h, w;
            int X_tile_width = TILE_SIZE_2 + K - 1;
            extern __shared__ float shmem[];
            float * X_shared = &shmem[0];

            n = blockIdx.x;
            m = blockIdx.y;
            h0 = threadIdx.x;
            w0 = threadIdx.y;
            h_base = blockIdx.z / W_grid * TILE_SIZE_2;
            w_base = blockIdx.z % W_grid * TILE_SIZE_2;
            h = h_base + h0;
            w = w_base + w0;

            float acc = 0.0f;

            int c, i, ii, j, p, q;
            #pragma unroll 5
            for (c = 0; c < C; c++) {

                // thread block size should be TILF_WIDTH * TILE_SIZE, and the data may be reload here
                for (i = h, ii = h0; i < h_base + X_tile_width; i += TILE_SIZE_2, ii += TILE_SIZE_2) {
                    for (j = w; j < w_base + X_tile_width; j += TILE_SIZE_2) {
                        X_shared[ii*X_tile_width+j-w_base] = x4d(n, c, i, j);
                    }
                }
                __syncthreads();

                #pragma unroll 5
                for (p = 0; p < K; p++) {
                    #pragma unroll 5
                    for (q = 0; q < K; q++) {
                        acc += X_shared[(h0 + p)*X_tile_width + w0 + q]*k4d_constant(m, c, p, q);
                    }
                }
                __syncthreads();
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

            if (M == 12) {
                const int H_grid = ceil((float)H_out / TILE_SIZE_1);
                const int W_grid = ceil((float) W_out / TILE_SIZE_1);
                const int Z =  H_grid * W_grid;

                dim3 gridDim(B, M, Z);
                dim3 blockDim(TILE_SIZE_1, TILE_SIZE_1, 1);

                size_t shmem_size = sizeof(float)  * ((TILE_SIZE_1 + K - 1) * (TILE_SIZE_1 + K -1) + K * K);
                int kernel_size = C*M*K*K;
                cudaMemcpyToSymbol(cons_mem, w.dptr_, sizeof(float)*kernel_size);
                __shmem_1<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, H, W, M, C, K, B, W_grid);
            }
            else {
                const int H_grid = ceil((float)H_out / TILE_SIZE_2);
                const int W_grid = ceil((float) W_out / TILE_SIZE_2);
                const int Z =  H_grid * W_grid;

                dim3 gridDim(B, M, Z);
                dim3 blockDim(TILE_SIZE_2, TILE_SIZE_2, 1);

                size_t shmem_size = sizeof(float)  * ((TILE_SIZE_2 + K - 1) * (TILE_SIZE_2 + K -1) + K * K);
                int kernel_size = C*M*K*K;
                cudaMemcpyToSymbol(cons_mem, w.dptr_, sizeof(float)*kernel_size);
                __shmem_2<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, H, W, M, C, K, B, W_grid);
            }


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

#undef TILE_SIZE_1
#undef TILE_SIZE_2
#undef CU_MAX_THREAD
#undef TOTAL_KERNEL_SIZE

#endif
