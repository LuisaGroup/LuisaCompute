// =============================================================================
// cnn_kernels.h — Luisa tile-language kernels for the TinyCNN inference
// =============================================================================
// Mirrors the PyTorch network in examples/tensor/cnn_train.py:
//   input [B,1,8,8]
//     -> conv1 (1->4, 3x3) + ReLU        -> [B,4,6,6]
//     -> conv2 (4->8, 3x3) + ReLU        -> [B,8,4,4]
//     -> flatten                         -> [B,128]
//     -> fc1 (128->32) + ReLU            -> [B,32]
//     -> fc2 (32->4)                     -> [B,4]  logits
//     -> softmax(dim=1)                  -> [B,4]  probabilities
//
// The tile language has no conv2d op, so every convolution is expressed as a
// GEMM over an im2col matrix built on the host (the standard TileLang/CUTLASS
// trick): with the bias folded into the weight matrix as an extra +1 row of
// the im2col matrix, each layer is a single pure T.gemm plus an elementwise
// ReLU / softmax.
//
// Each kernel is a separate tile program (exactly one T.Kernel) traced with
// tile::jit(...).compile() and lowered to a real Luisa kernel by
// tile_to_kernel.  All kernels use a single block (gx = 1) and full-tile
// copies, which the lowering indexes row-major, so the host-built im2col and
// weight matrices map 1:1 to the device buffers.
// =============================================================================

#pragma once

#include <luisa/dsl/tensor.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace cnn {

// constexpr handle for the tile DSL (T.* spelling, like TileLang's `T`)
constexpr auto T = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

// ---- problem dimensions (must match cnn_train.py / the exported .bin) -------
constexpr tile_i32 B = 4;    // batch
constexpr tile_i32 NC = 4;   // number of classes
constexpr tile_i32 IMG = 8;  // input spatial size
constexpr tile_i32 C1 = 4;   // conv1 out channels
constexpr tile_i32 C2 = 8;   // conv2 out channels
constexpr tile_i32 F1 = 32;  // fc1 out features

// conv1: Y[C1, B*36] = relu( W1'[C1, 10]      @ col1'[10, B*36] )
Tensor<tile_f32, 2> conv1_relu(Tensor<tile_f32, 2> W1, Tensor<tile_f32, 2> col1);
// conv2: Y[C2, B*16] = relu( W2'[C2, 37]      @ col2'[37, B*16] )
Tensor<tile_f32, 2> conv2_relu(Tensor<tile_f32, 2> W2, Tensor<tile_f32, 2> col2);
// fc1:   Y[B, F1]   = relu( col_fc1[B, 129]   @ Wfc1T[129, F1] )
Tensor<tile_f32, 2> fc1_relu(Tensor<tile_f32, 2> col_fc1, Tensor<tile_f32, 2> Wfc1T);
// fc2:   Y[B, NC]   =       col_fc2[B, 33]    @ Wfc2T[33, NC]
Tensor<tile_f32, 2> fc2(Tensor<tile_f32, 2> col_fc2, Tensor<tile_f32, 2> Wfc2T);
// softmax over each row of logits[B, NC]
Tensor<tile_f32, 2> softmax(Tensor<tile_f32, 2> logits);

}// namespace cnn
