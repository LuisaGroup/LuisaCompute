// =============================================================================
// poly_fit_kernels.h — New Tile DSL kernels for the polynomial fit
// =============================================================================
// Mirrors the PyTorch training loop in examples/tensor/poly_fit_train.py:
// fit y = sin(x) on [-pi, pi] with a degree-3 polynomial, i.e. a single
// nn.Linear(3, 1) on the Vandermonde features [x, x^2, x^3], trained by
// manually applying the gradients (param -= lr * param.grad).
//
// The bias is folded into the feature matrix as an extra all-ones column, so
// the model is a pure GEMM:
//   X[N,4] = [x, x^2, x^3, 1]        W[4,1] = [w1, w2, w3, b]
//   forward : Y[N,1] = X @ W          (tile::mma)
//   error   : err    = Y - y          (MSE residual)
//   gradient: G[4,1] = XT @ err       (d sum(err^2)/dW without the factor 2)
//   update  : W     -= (2*lr) * G     (manual SGD step)
//
// Port of backup_old_tile/examples/tensor/poly_fit_kernels.{h,cpp}. Every
// kernel is a single root parallel nest over shape(1) with whole-matrix tiles,
// exactly like the old single-block (gx = 1) tile programs; the host-built
// row-major matrices map 1:1 to the device buffers. The old scalar-division
// idiom for the update (`W + G / (-1/(2*lr))`, because the removed dialect had
// no binary minus) is an ordinary `W - G * (2*lr)` here.
// =============================================================================

#pragma once

#include <cstdint>

#include <luisa/tile/dsl.h>

namespace tensor_example {

namespace tile = luisa::compute::tile;

namespace polyfit {

// ---- problem dimensions ---------------------------------------------------
constexpr int64_t N_TRAIN = 512;// training samples (matches poly_fit_train.py --n-train)
constexpr int64_t N_TEST = 128; // held-out inference samples (--n-test)
constexpr int64_t F = 4;        // features [x, x^2, x^3, 1] (bias folded in)

// ---- hyperparameters (poly_fit_train.py --lr/--steps, scaled for N=512) ----
constexpr float LR = 4e-6f;     // manual learning rate (no optimizer)
constexpr int STEPS = 2000;     // gradient-descent steps

// forward : Y[N,1] = X[N,4] @ W[4,1]  (templated on N for train/test grids)
template<int64_t N>
[[nodiscard]] inline tile::Kernel make_poly_forward_kernel() {
    auto definition = tile::tile_kernel("poly_forward", [=](
                                            tile::TensorView<const float, 2> X,
                                            tile::TensorView<const float, 2> W,
                                            tile::TensorView<float, 2> Y) {
        auto n = tile::axis("n", N), f = tile::axis("f", F), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto x = X.tile(tile::coord(0, 0), tile::shape(n, f)).load();
            auto w = W.tile(tile::coord(0, 0), tile::shape(f, one)).load();
            auto acc = tile::zeros<float>(tile::shape(n, one));
            Y.tile(tile::coord(0, 0), tile::shape(n, one)).store(tile::mma(x, w, acc));
        }
    });
    return definition.capture(tile::tensor_shape(N, F), tile::tensor_shape(F, 1), tile::tensor_shape(N, 1));
}

// error : err[N,1] = Y[N,1] - y[N,1]   (MSE residual)
[[nodiscard]] tile::Kernel make_poly_error_kernel();

// gradient : G[4,1] = XT[4,N] @ err[N,1]  (the factor 2 is folded into the update)
[[nodiscard]] tile::Kernel make_poly_grad_kernel();

// update : W'[4,1] = W[4,1] - (2*lr) * G[4,1]   (manual SGD step)
[[nodiscard]] tile::Kernel make_poly_update_kernel();

}// namespace polyfit
}// namespace tensor_example
