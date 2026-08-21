// =============================================================================
// linear_regression_kernels.h — Luisa tile-language kernels for linear &
// logistic regression training
// =============================================================================
// The C++ twin of examples/tensor/linear_regression_train.py.  Both models
// are single-layer: logits = Xb @ W where Xb is the feature matrix with an
// extra all-ones column (bias folded into the weight vector), exactly like
// poly_fit_kernels.h.  Training uses full-batch gradient descent with the
// mean-gradient normalisation (gradient = XT @ residual / N):
//
//   linear  : forward Y[N,1] = Xb[N,D] @ W[D,1]
//             error  err     = Y - y                         (MSE residual)
//             grad   G[D,1]  = XT[D,N] @ err[N,1] / N        (dMSE/dW)
//             update W      -= lr * G
//   logistic: forward Z[N,1] = Xb[N,D] @ W[D,1]
//             residual res   = sigmoid(Z) - y                (BCE gradient)
//             grad   G[D,1]  = XT[D,N] @ res[N,1] / N        (dBCE/dW)
//             update W      -= lr * G
//
// The bias column of Xb makes the gradient of the bias term the last row of
// G automatically, so no separate bias parameter is needed.
//
// Every kernel is a single-block tile program (one T.Kernel) traced with
// tile::jit(...).compile() and lowered to a real Luisa kernel by
// tile_to_kernel, following the exact pattern of poly_fit_kernels.cpp.
// =============================================================================

#pragma once

#include <luisa/dsl/tensor.h>
#include <luisa/dsl/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace lreg {

// constexpr handle for the tile DSL (T.* spelling, like TileLang's `T`)
constexpr auto T = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

constexpr tile_i32 THREADS = 64;

// ---- problem dimensions (linear_regression_train.py defaults) ---------------
constexpr tile_i32 LIN_N = 512;   // linear-regression training samples
constexpr tile_i32 LIN_D = 4;     // linear-regression feature dim
constexpr tile_i32 LIN_K = LIN_D + 1;// 5 = features + bias column
constexpr tile_i32 LIN_NT = 128;  // linear-regression held-out samples
constexpr tile_i32 LOG_N = 600;   // logistic-regression training samples
constexpr tile_i32 LOG_D = 2;     // logistic-regression feature dim
constexpr tile_i32 LOG_K = LOG_D + 1;// 3 = features + bias column
constexpr tile_i32 LOG_NT = 300;  // logistic-regression held-out samples
constexpr tile_i32 NT = 300;      // shared inference cap (>= all held-out)

// ---- hyperparameters ---------------------------------------------------------
constexpr float LIN_LR = 0.1f;    // linear SGD learning rate
constexpr float LOG_LR = 0.1f;    // logistic SGD learning rate
constexpr int LIN_STEPS = 200;    // linear gradient steps
constexpr int LOG_STEPS = 200;    // logistic gradient steps

// forward : Y[N,1] = Xb[N,D] @ W[D,1]
template <tile_i32 N, tile_i32 D>
Tensor<tile_f32, 2> forward(Tensor<tile_f32, 2> Xb, Tensor<tile_f32, 2> W);
// error   : err[N,1] = Y[N,1] - y[N,1]   (MSE residual)
template <tile_i32 N>
Tensor<tile_f32, 2> linear_error(Tensor<tile_f32, 2> Y, Tensor<tile_f32, 2> y);
// residual: res[N,1] = sigmoid(Z[N,1]) - y[N,1]   (BCE gradient)
template <tile_i32 N>
Tensor<tile_f32, 2> logistic_residual(Tensor<tile_f32, 2> Z, Tensor<tile_f32, 2> y);
// grad    : G[D,1] = XT[D,N] @ R[N,1] / N
template <tile_i32 N, tile_i32 D>
Tensor<tile_f32, 2> grad(Tensor<tile_f32, 2> XT, Tensor<tile_f32, 2> R);
// update  : W'[D,1] = W[D,1] - lr_eff * G[D,1]  (global slice store, no fragment)
Tensor<tile_f32, 2> update_lin(Tensor<tile_f32, 2> W, Tensor<tile_f32, 2> G);
Tensor<tile_f32, 2> update_log(Tensor<tile_f32, 2> W, Tensor<tile_f32, 2> G);

}// namespace lreg
