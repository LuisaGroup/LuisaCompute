// =============================================================================
// tensor_basics_kernels.h — Luisa tile-language kernels for the basics tour
// =============================================================================
// Header-only tile kernels for the `--basics` driver (C++ twin of
// tensor_basics.py): elementwise arithmetic, a tiny autograd-style quadratic
// and a 1 -> 1 neural network trained with SGD (the relu is identity on the
// positive training inputs, so the tiny net reduces to linear regression).
// =============================================================================

#pragma once

#include <luisa/dsl/tensor.h>
#include <luisa/dsl/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace basics {

constexpr auto T = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

// ---------------------------------------------------------------------------
// basic_addmul : C = A + B;  D = A * B   (rank-1, N elements)
// ---------------------------------------------------------------------------
template <tile_i32 N>
Tensor<tile_f32, 1> basic_addmul(Tensor<tile_f32, 1> A, Tensor<tile_f32, 1> B,
                                 Tensor<tile_f32, 1> D) {
    Tensor<tile_f32, 1> C = T.empty(T.shape(N), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto a = T.alloc_fragment(T.shape(N), tile_f32{});
        auto b = T.alloc_fragment(T.shape(N), tile_f32{});
        auto c = T.alloc_fragment(T.shape(N), tile_f32{});
        auto d = T.alloc_fragment(T.shape(N), tile_f32{});
        T.copy(A(0), a(N));
        T.copy(B(0), b(N));
        c(N) = a(N) + b(N);
        d(N) = a(N) * b(N);
        T.copy(c(N), C(0));
        T.copy(d(N), D(0));
    }
    return C;
}

// ---------------------------------------------------------------------------
// basic_square_grad : Y = x^2 + 2x + 1;  DY = 2x + 2   (autograd demo)
// ---------------------------------------------------------------------------
template <tile_i32 N>
Tensor<tile_f32, 1> basic_square_grad(Tensor<tile_f32, 1> X, Tensor<tile_f32, 1> DY) {
    Tensor<tile_f32, 1> Y = T.empty(T.shape(N), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto x = T.alloc_fragment(T.shape(N), tile_f32{});
        auto y = T.alloc_fragment(T.shape(N), tile_f32{});
        auto dy = T.alloc_fragment(T.shape(N), tile_f32{});
        T.copy(X(0), x(N));
        // y = x*x + 2*x + 1  (2*x = x / 0.5 in the scalar-division idiom)
        y(N) = x(N) * x(N) + x(N) / 0.5f + 1.0f;
        dy(N) = x(N) / 0.5f + 2.0f;
        T.copy(y(N), Y(0));
        T.copy(dy(N), DY(0));
    }
    return Y;
}

// ---------------------------------------------------------------------------
// nn_forward : Y[B,1] = Xb[B,2] @ W[2,1]   (tiny 1 -> 1 net, bias folded in)
// ---------------------------------------------------------------------------
template <tile_i32 B>
Tensor<tile_f32, 2> nn_forward(Tensor<tile_f32, 2> Xb, Tensor<tile_f32, 2> W) {
    Tensor<tile_f32, 2> Y = T.empty(T.shape(B, 1), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto Xb_sh = T.alloc_shared(T.shape(B, 2), tile_f32{});
        auto W_sh = T.alloc_shared(T.shape(2, 1), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(B, 1), tile_f32{});
        T.copy(Xb(0, 0), Xb_sh(B, 2));
        T.copy(W(0, 0), W_sh(2, 1));
        T.clear(acc);
        T.gemm(Xb_sh(B, 2), W_sh(2, 1), acc(B, 1));
        T.copy(acc(B, 1), Y(0, 0));
    }
    return Y;
}

// ---------------------------------------------------------------------------
// nn_error : err[B,1] = Y[B,1] - targets[B,1]   (MSE residual)
// ---------------------------------------------------------------------------
template <tile_i32 B>
Tensor<tile_f32, 2> nn_error(Tensor<tile_f32, 2> Y, Tensor<tile_f32, 2> targets) {
    Tensor<tile_f32, 2> err = T.empty(T.shape(B, 1), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto y = T.alloc_fragment(T.shape(B, 1), tile_f32{});
        auto t = T.alloc_fragment(T.shape(B, 1), tile_f32{});
        T.copy(Y(0, 0), y(B, 1));
        T.copy(targets(0, 0), t(B, 1));
        y(B, 1) = y(B, 1) + t(B, 1) / (-1.0f);
        T.copy(y(B, 1), err(0, 0));
    }
    return err;
}

// ---------------------------------------------------------------------------
// nn_grad : G[2,1] = XT[2,B] @ err[B,1]
// ---------------------------------------------------------------------------
template <tile_i32 B>
Tensor<tile_f32, 2> nn_grad(Tensor<tile_f32, 2> XT, Tensor<tile_f32, 2> err) {
    Tensor<tile_f32, 2> G = T.empty(T.shape(2, 1), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto XT_sh = T.alloc_shared(T.shape(2, B), tile_f32{});
        auto err_sh = T.alloc_shared(T.shape(B, 1), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(2, 1), tile_f32{});
        T.copy(XT(0, 0), XT_sh(2, B));
        T.copy(err(0, 0), err_sh(B, 1));
        T.clear(acc);
        T.gemm(XT_sh(2, B), err_sh(B, 1), acc(2, 1));
        T.copy(acc(2, 1), G(0, 0));
    }
    return G;
}

// ---------------------------------------------------------------------------
// nn_update : W'[2,1] = W[2,1] - 0.05/4 * G[2,1]   (global slice store)
// ---------------------------------------------------------------------------
template <tile_i32 D>
Tensor<tile_f32, 2> nn_update(Tensor<tile_f32, 2> W, Tensor<tile_f32, 2> G) {
    constexpr float LR_EFF = 0.05f / 4.0f;
    Tensor<tile_f32, 2> W_new = T.empty(T.shape(D, 1), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        W_new(T.range(0, D), T.range(0, 1)) =
            W(T.range(0, D), T.range(0, 1)) +
            G(T.range(0, D), T.range(0, 1)) / (-1.0f / LR_EFF);
    }
    return W_new;
}

}// namespace basics
