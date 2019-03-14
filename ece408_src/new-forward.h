
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    const int B = x.shape_[0]; // batch_size
    const int C = x.shape_[1]; // in_channels
    const int H = x.shape_[2]; // height
    const int W = x.shape_[3]; // weight
    const int M = y.shape_[1]; // out_channels
    const int K = k.shape_[3]; // kernel_size

    for (int b = 0; b < B; ++b) {
        /* ... a bunch of nested loops later...
            y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
        */
       int H_out = H - K + 1;
       int W_out = W - K + 1;
       for (int m = 0; m < M; ++m) {
           for (int h = 0; h < H_out; ++h) {
               for (int w = 0; w < W_out; ++w) {
                   y[b][m][h][w] = 0.0f;
                   for (int c = 0; c < C; ++c) {
                       for (int p = 0; p < K; ++p) {
                           for (int q = 0; q < K; ++q) {
                               y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
                           }
                       }
                   }
               }
           }
       }
    }

}
}
}

#endif
