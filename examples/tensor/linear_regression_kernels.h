// =============================================================================
// linear_regression_kernels.h — New Tile DSL kernels for linear & logistic
// regression training
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
// Port of backup_old_tile/examples/tensor/linear_regression_kernels.{h,cpp}.
// Every kernel is a single root parallel nest over shape(1) with whole-matrix
// tiles, matching the old single-block tile programs.  The old sigmoid via
// `rsqrt(exp(-z)+1)^2` (the removed dialect had no tile subtraction) is the
// direct `1/(1+exp(-z))` composition here; the old update's scalar-division
// idiom is an ordinary `W - G * lr_eff`.
// =============================================================================

#pragma once

#include <cstdint>

#include <luisa/tile/dsl.h>

namespace tensor_example {

namespace tile = luisa::compute::tile;

namespace lreg {

// ---- problem dimensions (linear_regression_train.py defaults) ---------------
constexpr int64_t LIN_N = 512;       // linear-regression training samples
constexpr int64_t LIN_D = 4;         // linear-regression feature dim
constexpr int64_t LIN_K = LIN_D + 1; // 5 = features + bias column
constexpr int64_t LIN_NT = 128;      // linear-regression held-out samples
constexpr int64_t LOG_N = 600;       // logistic-regression training samples
constexpr int64_t LOG_D = 2;         // logistic-regression feature dim
constexpr int64_t LOG_K = LOG_D + 1; // 3 = features + bias column
constexpr int64_t LOG_NT = 300;      // logistic-regression held-out samples

// ---- hyperparameters ---------------------------------------------------------
constexpr float LIN_LR = 0.1f;       // linear SGD learning rate
constexpr float LOG_LR = 0.1f;       // logistic SGD learning rate
constexpr int LIN_STEPS = 200;       // linear gradient steps
constexpr int LOG_STEPS = 200;       // logistic gradient steps

// forward : Y[N,1] = Xb[N,D] @ W[D,1]
template<int64_t N, int64_t D>
[[nodiscard]] inline tile::Kernel make_forward_kernel() {
    auto definition = tile::tile_kernel("lreg_forward", [=](
                                            tile::TensorView<const float, 2> Xb,
                                            tile::TensorView<const float, 2> W,
                                            tile::TensorView<float, 2> Y) {
        auto n = tile::axis("n", N), d = tile::axis("d", D), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto xb = Xb.tile(tile::coord(0, 0), tile::shape(n, d)).load();
            auto w = W.tile(tile::coord(0, 0), tile::shape(d, one)).load();
            auto acc = tile::zeros<float>(tile::shape(n, one));
            Y.tile(tile::coord(0, 0), tile::shape(n, one)).store(tile::mma(xb, w, acc));
        }
    });
    return definition.capture(tile::tensor_shape(N, D), tile::tensor_shape(D, 1), tile::tensor_shape(N, 1));
}

// error : err[N,1] = Y[N,1] - y[N,1]   (MSE residual)
[[nodiscard]] tile::Kernel make_linear_error_kernel();

// residual : res[N,1] = sigmoid(Z[N,1]) - y[N,1]   (BCE gradient)
[[nodiscard]] tile::Kernel make_logistic_residual_kernel();

// grad : G[D,1] = XT[D,N] @ R[N,1]   (sum-gradient; the /N mean normalisation
// is folded into the update step via lr_eff = lr / N)
template<int64_t N, int64_t D>
[[nodiscard]] inline tile::Kernel make_grad_kernel() {
    auto definition = tile::tile_kernel("lreg_grad", [=](
                                            tile::TensorView<const float, 2> XT,
                                            tile::TensorView<const float, 2> R,
                                            tile::TensorView<float, 2> G) {
        auto d = tile::axis("d", D), n = tile::axis("n", N), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto xt = XT.tile(tile::coord(0, 0), tile::shape(d, n)).load();
            auto r = R.tile(tile::coord(0, 0), tile::shape(n, one)).load();
            auto acc = tile::zeros<float>(tile::shape(d, one));
            G.tile(tile::coord(0, 0), tile::shape(d, one)).store(tile::mma(xt, r, acc));
        }
    });
    return definition.capture(tile::tensor_shape(D, N), tile::tensor_shape(N, 1), tile::tensor_shape(D, 1));
}

// update : W'[D,1] = W[D,1] - lr_eff * G[D,1]  (lr_eff = lr / N folds the
// mean-gradient normalisation from the `grad` kernel into the update step)
template<int64_t D, float LR_EFF>
[[nodiscard]] inline tile::Kernel make_update_kernel() {
    auto definition = tile::tile_kernel("lreg_update", [=](
                                            tile::TensorView<const float, 2> W,
                                            tile::TensorView<const float, 2> G,
                                            tile::TensorView<float, 2> W_new) {
        auto d = tile::axis("d", D), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto w = W.tile(tile::coord(0, 0), tile::shape(d, one)).load();
            auto g = G.tile(tile::coord(0, 0), tile::shape(d, one)).load();
            W_new.tile(tile::coord(0, 0), tile::shape(d, one)).store(w - g * LR_EFF);
        }
    });
    return definition.capture(tile::tensor_shape(D, 1), tile::tensor_shape(D, 1), tile::tensor_shape(D, 1));
}

// update (linear) : W' = W - (LIN_LR / LIN_N) * G
[[nodiscard]] inline tile::Kernel make_update_lin_kernel() {
    return make_update_kernel<LIN_K, LIN_LR / static_cast<float>(LIN_N)>();
}

// update (logistic) : W' = W - (LOG_LR / LOG_N) * G
[[nodiscard]] inline tile::Kernel make_update_log_kernel() {
    return make_update_kernel<LOG_K, LOG_LR / static_cast<float>(LOG_N)>();
}

}// namespace lreg
}// namespace tensor_example
