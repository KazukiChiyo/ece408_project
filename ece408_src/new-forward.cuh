
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_SIZE_1 16
#define TILE_SIZE_2 22
#define CU_MAX_THREAD 1024

/* kernel buffer elelment size */
#define FIRST_INPUT 1
#define FIRST_OUTPUT 6
#define SECOND_INPUT FIRST_OUTPUT
#define SECOND_OUTPUT 16
#define K 5
#define TOTAL_KERNEL_SIZE_LARGE 2400
#define TOTAL_KERNEL_SIZE_SMALL 150
#define FIRST_H 48
#define FIRST_W FIRST_H
#define SECOND_H 22
 // 10 12 14 15 - 25, 42-43 tested
#define SECOND_W SECOND_H

namespace mxnet
{
    namespace op
    {
        // __constant__ float cons_mem[TOTAL_KERNEL_SIZE];
		__constant__ float cons_mem_small[TOTAL_KERNEL_SIZE_SMALL];
		__constant__ float cons_mem_large[TOTAL_KERNEL_SIZE_LARGE];

        /* shared memory */
        __global__ void __shmem_1(float *y, const float *x, const int B, const int W_grid) {
            #define H FIRST_H
            #define W FIRST_W
            #define C FIRST_INPUT
            #define M FIRST_OUTPUT
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) k[(i3) * (C * 25) + (i2) * 25 + (i1) * (K) + i0]
            #define k4d_constant(i3, i2, i1, i0) cons_mem_small[(i3) * (C * 25) + (i2) * 25 + (i1) * (K) + i0]

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
                        acc += X_shared[(h0 + p)*X_tile_width + w0 + q] * k4d_constant(m, c, p, q);
                        // acc += X_shared[(h0 + p)*X_tile_width + w0 + q] * cons_mem_small[m * 25 + c * 25 + p * 5 + q];
                    }
                }
                __syncthreads();
            }

            if (h < H_out && w < W_out)
                y4d(n, m, h, w) = acc;
                // cons_mem_small[25 * n + 25 * m + h * 5 + w] = acc;

            #undef H
            #undef W
            #undef C
            #undef M
            #undef y4d
            #undef x4d
            #undef k4d
            #undef k4d_constant
        }

        /* shared memory */
        __global__ void __shmem_2(float *y, const float *x, const int H, const int W, const int B, const int W_grid) {
            #define H SECOND_H
            #define W SECOND_W
            #define C SECOND_INPUT
            #define M SECOND_OUTPUT
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) k[(i3) * (C * 25) + (i2) * 25 + (i1) * (K) + i0]
            #define k4d_constant(i3, i2, i1, i0) cons_mem_large[(i3) * (C * 25) + (i2) * 25 + (i1) * (K) + i0]

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

            #undef H
            #undef W
            #undef C
            #undef M
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
            const int H = x.shape_[2]; // height
            const int W = x.shape_[3]; // weight
            const int M = y.shape_[1]; // out_channels
            const int H_out = H - K + 1;
            const int W_out = W - K + 1;

            if (M == FIRST_OUTPUT) {
                const int H_grid = ceil((float)H_out / TILE_SIZE_1);
                const int W_grid = ceil((float) W_out / TILE_SIZE_1);
                const int Z =  H_grid * W_grid;

                dim3 gridDim(B, M, Z);
                dim3 blockDim(TILE_SIZE_1, TILE_SIZE_1, 1);

                size_t shmem_size = sizeof(float) * ((TILE_SIZE_1 + K - 1) * (TILE_SIZE_1 + K -1) + K * K);
                int kernel_size = TOTAL_KERNEL_SIZE_SMALL;
                cudaMemcpyToSymbol(cons_mem_small, w.dptr_, sizeof(float)*kernel_size);
                __shmem_1<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, B, W_grid);
            }
			if (M == SECOND_OUTPUT) {
                const int H_grid = ceil((float)H_out / TILE_SIZE_2);
                const int W_grid = ceil((float) W_out / TILE_SIZE_2);
                const int Z =  H_grid * W_grid;

                dim3 gridDim(B, M, Z);
                dim3 blockDim(TILE_SIZE_2, TILE_SIZE_2, 1);

                size_t shmem_size = sizeof(float)  * ((TILE_SIZE_2 + K - 1) * (TILE_SIZE_2 + K -1) + K * K);
                int kernel_size =TOTAL_KERNEL_SIZE_LARGE;
                cudaMemcpyToSymbol(cons_mem_large, w.dptr_, sizeof(float)*kernel_size);
                __shmem_2<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, H, W, B, W_grid);
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
#undef FIRST_INPUT
#undef FIRST_OUTPUT
#undef SECOND_INPUT
#undef SECOND_OUTPUT
#undef K
#undef TOTAL_KERNEL_SIZE_LARGE
#undef TOTAL_KERNEL_SIZE_SMALL
#undef FIRST_H
#undef FIRST_W
#undef SECOND_H
#undef SECOND_W

#endif
