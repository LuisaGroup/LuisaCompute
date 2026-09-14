// =============================================================================
// linear_regression_kernels.cpp — Luisa tile-language kernels for linear &
// logistic regression training
// =============================================================================
// Implementation of the tile kernels declared in linear_regression_kernels.h.
// Each kernel is traced with tile::jit(...).compile() in linear_regression.cpp
// and lowered to a real Luisa kernel via tile_to_kernel.
//
// All layers are expressed with the well-tested tile ops:
//   T.copy      global -> shared staging (row-major full-tile copies)
//   T.clear / T.gemm    software GEMM: C += A @ B
//   T.rsqrt / T.exp     sigmoid via rsqrt(exp(-x)+1)^2
//   global slice store  elementwise SGD update without a large fragment
// =============================================================================

#include "linear_regression_kernels.h"

namespace lreg {

namespace {

// update : W' = W - lr_eff * G   (lr_eff = lr / N folds the mean-gradient
// normalisation from the `grad` kernel into the update step).
template <tile_i32 D, float LR_EFF>
Tensor<tile_f32, 2> update_impl(Tensor<tile_f32, 2> W, Tensor<tile_f32, 2> G) {
    Tensor<tile_f32, 2> W_new = T.empty(T.shape(D, 1), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        // W' = W + G / (-1/lr_eff)  ==  W - lr_eff*G  (scalar-division idiom).
        W_new(T.range(0, D), T.range(0, 1)) =
            W(T.range(0, D), T.range(0, 1)) +
            G(T.range(0, D), T.range(0, 1)) / (-1.0f / LR_EFF);
    }
    return W_new;
}

}// namespace

// ---------------------------------------------------------------------------
// forward : Y[N,1] = Xb[N,D] @ W[D,1]
// ---------------------------------------------------------------------------
template <tile_i32 N, tile_i32 D>
Tensor<tile_f32, 2> forward(Tensor<tile_f32, 2> Xb, Tensor<tile_f32, 2> W) {
    Tensor<tile_f32, 2> Y = T.empty(T.shape(N, 1), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto Xb_sh = T.alloc_shared(T.shape(N, D), tile_f32{});
        auto W_sh = T.alloc_shared(T.shape(D, 1), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(N, 1), tile_f32{});
        T.copy(Xb(0, 0), Xb_sh(N, D));
        T.copy(W(0, 0), W_sh(D, 1));
        T.clear(acc);
        T.gemm(Xb_sh(N, D), W_sh(D, 1), acc(N, 1));
        T.copy(acc(N, 1), Y(0, 0));
    }
    return Y;
}

// ---------------------------------------------------------------------------
// error : err[N,1] = Y[N,1] - y[N,1]   (MSE residual)
// ---------------------------------------------------------------------------
template <tile_i32 N>
Tensor<tile_f32, 2> linear_error(Tensor<tile_f32, 2> Y, Tensor<tile_f32, 2> y) {
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
// residual : res[N,1] = sigmoid(Z[N,1]) - y[N,1]   (BCE gradient)
// ---------------------------------------------------------------------------
template <tile_i32 N>
Tensor<tile_f32, 2> logistic_residual(Tensor<tile_f32, 2> Z, Tensor<tile_f32, 2> y) {
    Tensor<tile_f32, 2> res = T.empty(T.shape(N, 1), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto z = Z(T.range(0, N), T.range(0, 1));
        auto yy = y(T.range(0, N), T.range(0, 1));
        auto rr = res(T.range(0, N), T.range(0, 1));
        // sigmoid(z) = 1/(1+exp(-z)) = rsqrt(exp(-z)+1)^2; res = sigmoid - y.
        // Global slice stores (element-local reads/writes) keep this kernel
        // valid for large N without large shared-backed fragments.
        rr = T.exp(z / (-1.0f)) + 1.0f;                       // denom
        rr = T.rsqrt(rr) * T.rsqrt(rr);                       // sigmoid
        rr = rr + yy / (-1.0f);                               // sigmoid - y
    }
    return res;
}

// ---------------------------------------------------------------------------
// grad : G[D,1] = XT[D,N] @ R[N,1]   (sum-gradient; the /N mean normalisation
// is folded into the update step via lr_eff = lr / N)
// ---------------------------------------------------------------------------
template <tile_i32 N, tile_i32 D>
Tensor<tile_f32, 2> grad(Tensor<tile_f32, 2> XT, Tensor<tile_f32, 2> R) {
    Tensor<tile_f32, 2> G = T.empty(T.shape(D, 1), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto XT_sh = T.alloc_shared(T.shape(D, N), tile_f32{});
        auto R_sh = T.alloc_shared(T.shape(N, 1), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(D, 1), tile_f32{});
        T.copy(XT(0, 0), XT_sh(D, N));
        T.copy(R(0, 0), R_sh(N, 1));
        T.clear(acc);
        T.gemm(XT_sh(D, N), R_sh(N, 1), acc(D, 1));
        T.copy(acc(D, 1), G(0, 0));
    }
    return G;
}

// ---------------------------------------------------------------------------
// update (linear) : W' = W - (LIN_LR / LIN_N) * G
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> update_lin(Tensor<tile_f32, 2> W, Tensor<tile_f32, 2> G) {
    return update_impl<LIN_K, LIN_LR / static_cast<float>(LIN_N)>(W, G);
}

// ---------------------------------------------------------------------------
// update (logistic) : W' = W - (LOG_LR / LOG_N) * G
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> update_log(Tensor<tile_f32, 2> W, Tensor<tile_f32, 2> G) {
    return update_impl<LOG_K, LOG_LR / static_cast<float>(LOG_N)>(W, G);
}

// explicit instantiations
template Tensor<tile_f32, 2> forward<LIN_N, LIN_K>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);
template Tensor<tile_f32, 2> forward<LIN_NT, LIN_K>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);
template Tensor<tile_f32, 2> forward<LOG_N, LOG_K>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);
template Tensor<tile_f32, 2> forward<LOG_NT, LOG_K>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);
template Tensor<tile_f32, 2> linear_error<LIN_N>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);
template Tensor<tile_f32, 2> logistic_residual<LOG_N>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);
template Tensor<tile_f32, 2> grad<LIN_N, LIN_K>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);
template Tensor<tile_f32, 2> grad<LOG_N, LOG_K>(Tensor<tile_f32, 2>, Tensor<tile_f32, 2>);

}// namespace lreg
