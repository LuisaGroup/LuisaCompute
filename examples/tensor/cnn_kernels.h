// =============================================================================
// cnn_kernels.h — New Tile DSL kernels for the TinyCNN inference
// =============================================================================
// Port of backup_old_tile/examples/tensor/cnn_kernels.h to the
// execution-structure-first C++ Tile DSL (<luisa/tile/dsl.h>). Mirrors the
// PyTorch network in cnn_train.py:
//   input [B,1,8,8]
//     -> conv1 (1->4, 3x3) + ReLU        -> [B,4,6,6]
//     -> conv2 (4->8, 3x3) + ReLU        -> [B,8,4,4]
//     -> flatten                         -> [B,128]
//     -> fc1 (128->32) + ReLU            -> [B,32]
//     -> fc2 (32->4)                     -> [B,4]  logits
//     -> softmax(dim=1)                  -> [B,4]  probabilities
//
// The Tile DSL has no conv2d op, so every convolution is a GEMM over an
// im2col matrix built on the host (see cnn_inference.cpp), exactly like the
// old example: the bias is folded into the weight matrix as an extra all-ones
// row of the im2col matrix, so each layer is a single tile::mma plus an
// elementwise ReLU / softmax.
//
// Each factory captures exactly one root parallel(shape(1)) nest, loads the
// whole layer tiles, and stores the result; all layer matrices are tiny
// (largest tile is [10, 144]), matching the old single-block tile programs.
// =============================================================================

#pragma once

#include <cstdint>
#include <luisa/tile/dsl.h>

namespace tensor_example::cnn {

namespace tile = luisa::compute::tile;

// ---- problem dimensions (must match cnn_train.py / the exported .bin) -------
inline constexpr int64_t CB = 4;   // batch
inline constexpr int64_t CNC = 4;  // number of classes
inline constexpr int64_t CIMG = 8; // input spatial size
inline constexpr int64_t CC1 = 4;  // conv1 out channels
inline constexpr int64_t CC2 = 8;  // conv2 out channels
inline constexpr int64_t CF1 = 32; // fc1 out features

// conv1: Y[C1, B*36] = relu( W1'[C1, 10]  @ col1'[10, B*36] )
[[nodiscard]] tile::Kernel make_conv1_relu_kernel();

// conv2: Y[C2, B*16] = relu( W2'[C2, 37]  @ col2'[37, B*16] )
[[nodiscard]] tile::Kernel make_conv2_relu_kernel();

// fc1:   Y[B, F1]    = relu( col_fc1[B, 129] @ Wfc1T[129, F1] )
[[nodiscard]] tile::Kernel make_fc1_relu_kernel();

// fc2:   Y[B, NC]    =       col_fc2[B, 33]  @ Wfc2T[33, NC]
[[nodiscard]] tile::Kernel make_fc2_kernel();

// row-wise softmax over logits[B, NC]
[[nodiscard]] tile::Kernel make_cnn_softmax_kernel();

}// namespace tensor_example::cnn
