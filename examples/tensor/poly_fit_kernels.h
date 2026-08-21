// =============================================================================
// poly_fit_kernels.h — Luisa tile-language kernels for the polynomial fit
// =============================================================================
// Mirrors the PyTorch training loop in examples/tensor/poly_fit_train.py:
// fit y = sin(x) on [-pi, pi] with a degree-3 polynomial, i.e. a single
// nn.Linear(3, 1) on the Vandermonde features [x, x^2, x^3], trained by
// manually applying the gradients (param -= lr * param.grad).
//
// The bias is folded into the feature matrix as an extra all-ones column
// (the same trick cnn_kernels uses with the im2col bias row), so the model is
// a pure GEMM:
//   X[N,4] = [x, x^2, x^3, 1]        W[4,1] = [w1, w2, w3, b]
//   forward : Y[N,1] = X @ W
//   error   : err    = Y - y                        (MSE residual)
//   gradient: G[4,1] = 2 * XT @ err                 (d sum(err^2)/dW)
//   update  : W     -= lr * G                       (manual SGD step)
//
// Each step of the training loop is expressed with the well-tested tile ops:
//   T.copy      global -> shared staging (row-major full-tile copies)
//   T.clear / T.gemm    software GEMM: C += A @ B
//   fragment arithmetic (scalar broadcast) for the residual / SGD update
//
// Each kernel is a separate tile program (exactly one T.Kernel) traced with
// tile::jit(...).compile() and lowered to a real Luisa kernel by
// tile_to_kernel.  All kernels use a single block (gx = 1) and full-tile
// copies, which the lowering indexes row-major, so the host-built matrices
// map 1:1 to the device buffers.
// =============================================================================

#pragma once

#include <luisa/dsl/tensor.h>
#include <luisa/dsl/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace polyfit {

// constexpr handle for the tile DSL (T.* spelling, like TileLang's `T`)
constexpr auto T = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

// ---- problem dimensions -------------------------------------------------------
constexpr tile_i32 N_TRAIN = 512;// training samples (matches poly_fit_train.py --n-train)
constexpr tile_i32 N_TEST = 128; // held-out inference samples (--n-test)
constexpr tile_i32 F = 4;        // features [x, x^2, x^3, 1] (bias folded in)

// ---- hyperparameters (poly_fit_train.py --lr/--steps, scaled for N=512) -------
constexpr float LR = 4e-6f;      // manual learning rate (no optimizer)
constexpr int STEPS = 2000;      // gradient-descent steps

// forward : Y[N,1] = X[N,4] @ W[4,1]  (templated on N for train/test grids)
template <tile_i32 N>
Tensor<tile_f32, 2> poly_forward(Tensor<tile_f32, 2> X, Tensor<tile_f32, 2> W);
// error   : err[N,1] = Y[N,1] - y[N,1]
Tensor<tile_f32, 2> poly_error(Tensor<tile_f32, 2> Y, Tensor<tile_f32, 2> y);
// gradient: G[4,1] = XT[4,N] @ err[N,1]  (the factor 2 is folded into the update)
Tensor<tile_f32, 2> poly_grad(Tensor<tile_f32, 2> XT, Tensor<tile_f32, 2> err);
// update  : W'[4,1] = W[4,1] - (2*lr) * G[4,1]   (manual SGD step)
Tensor<tile_f32, 2> poly_update(Tensor<tile_f32, 2> W, Tensor<tile_f32, 2> G);

}// namespace polyfit
