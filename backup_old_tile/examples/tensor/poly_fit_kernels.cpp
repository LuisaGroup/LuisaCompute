// =============================================================================
// poly_fit_kernels.cpp — Luisa tile-language kernels for the polynomial fit
// =============================================================================
// Implementation of the four tile kernels declared in poly_fit_kernels.h.
// Each kernel is traced with tile::jit(...).compile() in poly_fit.cpp and
// lowered to a real Luisa kernel via tile_to_kernel (see tile::jit in
// <luisa/dsl/tensor.h>).
// =============================================================================

#include "poly_fit_kernels.h"

namespace polyfit {

namespace {

constexpr tile_i32 THREADS = 64;

}// namespace

// ---------------------------------------------------------------------------
// forward : Y[N,1] = X[N,F] @ W[F,1]
// ---------------------------------------------------------------------------
template <tile_i32 N>
Tensor<tile_f32, 2> poly_forward(Tensor<tile_f32, 2> X, Tensor<tile_f32, 2> W) {
    Tensor<tile_f32, 2> Y = T.empty(T.shape(N, 1), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto X_sh = T.alloc_shared(T.shape(N, F), tile_f32{});
        auto W_sh = T.alloc_shared(T.shape(F, 1), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(N, 1), tile_f32{});
        T.copy(X(0, 0), X_sh(N, F));
        T.copy(W(0, 0), W_sh(F, 1));
        T.clear(acc);
        T.gemm(X_sh(N, F), W_sh(F, 1), acc(N, 1));
        T.copy(acc(N, 1), Y(0, 0));
    }
    return Y;
}

// explicit instantiations for the train / test grids
template Tensor<tile_f32, 2> poly_forward<N_TRAIN>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);
template Tensor<tile_f32, 2> poly_forward<N_TEST>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);

// ---------------------------------------------------------------------------
// error : err[N,1] = Y[N,1] - y[N,1]   (MSE residual)
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> poly_error(Tensor<tile_f32, 2> Y, Tensor<tile_f32, 2> y) {
    constexpr tile_i32 N = N_TRAIN;
    Tensor<tile_f32, 2> err = T.empty(T.shape(N, 1), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto Y_loc = T.alloc_fragment(T.shape(N, 1), tile_f32{});
        auto y_loc = T.alloc_fragment(T.shape(N, 1), tile_f32{});
        T.copy(Y(0, 0), Y_loc(N, 1));
        T.copy(y(0, 0), y_loc(N, 1));
        // err = Y - y; the tile dialect has no binary minus, so spell it as
        // Y + (-y) (division by -1.0f is an exact sign flip).
        Y_loc(N, 1) = Y_loc(N, 1) + y_loc(N, 1) / (-1.0f);
        T.copy(Y_loc(N, 1), err(0, 0));
    }
    return err;
}

// ---------------------------------------------------------------------------
// gradient : G[F,1] = XT[F,N] @ err[N,1]
// (d sum(err^2)/dW without the factor 2 — folded into the update step)
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> poly_grad(Tensor<tile_f32, 2> XT, Tensor<tile_f32, 2> err) {
    constexpr tile_i32 N = N_TRAIN;
    Tensor<tile_f32, 2> G = T.empty(T.shape(F, 1), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto XT_sh = T.alloc_shared(T.shape(F, N), tile_f32{});
        auto err_sh = T.alloc_shared(T.shape(N, 1), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(F, 1), tile_f32{});
        T.copy(XT(0, 0), XT_sh(F, N));
        T.copy(err(0, 0), err_sh(N, 1));
        T.clear(acc);
        T.gemm(XT_sh(F, N), err_sh(N, 1), acc(F, 1));
        T.copy(acc(F, 1), G(0, 0));
    }
    return G;
}

// ---------------------------------------------------------------------------
// update : W' = W - (2*lr) * G   (the manual `param -= lr * param.grad` step,
// with the MSE factor 2 folded into the effective learning rate)
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> poly_update(Tensor<tile_f32, 2> W, Tensor<tile_f32, 2> G) {
    Tensor<tile_f32, 2> W_new = T.empty(T.shape(F, 1), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto W_loc = T.alloc_fragment(T.shape(F, 1), tile_f32{});
        auto G_loc = T.alloc_fragment(T.shape(F, 1), tile_f32{});
        T.copy(W(0, 0), W_loc(F, 1));
        T.copy(G(0, 0), G_loc(F, 1));
        // W' = W - (2*lr) * G; the tile dialect has no binary minus and no
        // tile*scalar, so spell it as W + G / (-1/(2*lr)).
        W_loc(F, 1) = W_loc(F, 1) + G_loc(F, 1) / (-1.0f / (2.0f * LR));
        T.copy(W_loc(F, 1), W_new(0, 0));
    }
    return W_new;
}

}// namespace polyfit
