// =============================================================================
// cnn_kernels.cpp — Luisa tile-language kernels for TinyCNN inference
// =============================================================================
// Implementation of the five tile kernels declared in cnn_kernels.h.  Each
// layer is traced with tile::jit(...).compile() in cnn_inference.cpp and
// lowered to a real Luisa kernel via tile_to_kernel (see tile::jit in
// <luisa/dsl/tensor.h>).
//
// All layers are expressed with the well-tested tile ops:
//   T.copy      global -> shared staging (row-major full-tile copies)
//   T.clear / T.gemm    software GEMM: C += A @ B
//   T.max       ReLU
//   T.exp / T.reduce_sum / T.rsqrt   softmax
// The bias is folded into the weight matrix via an extra all-ones row of the
// im2col matrix, so no separate bias add is needed.
// =============================================================================

#include "cnn_kernels.h"

namespace cnn {

namespace {

constexpr tile_i32 THREADS = 64;

// conv1 output spatial size: (IMG - 3 + 1) = 6 -> 36 pixels
constexpr tile_i32 P1 = 36;
// conv2 output spatial size: (6 - 3 + 1) = 4 -> 16 pixels
constexpr tile_i32 P2 = 16;

}// namespace

// ---------------------------------------------------------------------------
// conv1 + ReLU : Y[C1, B*36] = relu( W1'[C1,10] @ col1'[10, B*36] )
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> conv1_relu(Tensor<tile_f32, 2> W1, Tensor<tile_f32, 2> col1) {
    constexpr tile_i32 Co = C1;
    constexpr tile_i32 KK = 10;// C1*1*3*3 + bias row = 9 + 1
    constexpr tile_i32 P = B * P1;
    Tensor<tile_f32, 2> Y = T.empty(T.shape(Co, P), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto W_sh = T.alloc_shared(T.shape(Co, KK), tile_f32{});
        auto col_sh = T.alloc_shared(T.shape(KK, P), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(Co, P), tile_f32{});
        T.copy(W1(0, 0), W_sh(Co, KK));
        T.copy(col1(0, 0), col_sh(KK, P));
        T.clear(acc);
        T.gemm(W_sh(Co, KK), col_sh(KK, P), acc(Co, P));
        acc(Co, P) = T.max(acc(Co, P), 0.0f);// ReLU
        T.copy(acc(Co, P), Y(0, 0));
    }
    return Y;
}

// ---------------------------------------------------------------------------
// conv2 + ReLU : Y[C2, B*16] = relu( W2'[C2,37] @ col2'[37, B*16] )
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> conv2_relu(Tensor<tile_f32, 2> W2, Tensor<tile_f32, 2> col2) {
    constexpr tile_i32 Co = C2;
    constexpr tile_i32 KK = C1 * 9 + 1;// 4*9 + bias row = 37
    constexpr tile_i32 P = B * P2;
    Tensor<tile_f32, 2> Y = T.empty(T.shape(Co, P), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto W_sh = T.alloc_shared(T.shape(Co, KK), tile_f32{});
        auto col_sh = T.alloc_shared(T.shape(KK, P), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(Co, P), tile_f32{});
        T.copy(W2(0, 0), W_sh(Co, KK));
        T.copy(col2(0, 0), col_sh(KK, P));
        T.clear(acc);
        T.gemm(W_sh(Co, KK), col_sh(KK, P), acc(Co, P));
        acc(Co, P) = T.max(acc(Co, P), 0.0f);// ReLU
        T.copy(acc(Co, P), Y(0, 0));
    }
    return Y;
}

// ---------------------------------------------------------------------------
// fc1 + ReLU : Y[B, F1] = relu( col_fc1[B, 129] @ Wfc1T[129, F1] )
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> fc1_relu(Tensor<tile_f32, 2> col_fc1, Tensor<tile_f32, 2> Wfc1T) {
    constexpr tile_i32 M = B;
    constexpr tile_i32 K = C2 * P2 + 1;// 8*16 + bias row = 129
    constexpr tile_i32 N = F1;
    Tensor<tile_f32, 2> Y = T.empty(T.shape(M, N), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto col_sh = T.alloc_shared(T.shape(M, K), tile_f32{});
        auto W_sh = T.alloc_shared(T.shape(K, N), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(M, N), tile_f32{});
        T.copy(col_fc1(0, 0), col_sh(M, K));
        T.copy(Wfc1T(0, 0), W_sh(K, N));
        T.clear(acc);
        T.gemm(col_sh(M, K), W_sh(K, N), acc(M, N));
        acc(M, N) = T.max(acc(M, N), 0.0f);// ReLU
        T.copy(acc(M, N), Y(0, 0));
    }
    return Y;
}

// ---------------------------------------------------------------------------
// fc2 (logits) : Y[B, NC] = col_fc2[B, 33] @ Wfc2T[33, NC]
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> fc2(Tensor<tile_f32, 2> col_fc2, Tensor<tile_f32, 2> Wfc2T) {
    constexpr tile_i32 M = B;
    constexpr tile_i32 K = F1 + 1;// 32 + bias row = 33
    constexpr tile_i32 N = NC;
    Tensor<tile_f32, 2> Y = T.empty(T.shape(M, N), tile_f32{});
    for (auto bx : T.Kernel(1, THREADS)) {
        auto col_sh = T.alloc_shared(T.shape(M, K), tile_f32{});
        auto W_sh = T.alloc_shared(T.shape(K, N), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(M, N), tile_f32{});
        T.copy(col_fc2(0, 0), col_sh(M, K));
        T.copy(Wfc2T(0, 0), W_sh(K, N));
        T.clear(acc);
        T.gemm(col_sh(M, K), W_sh(K, N), acc(M, N));
        T.copy(acc(M, N), Y(0, 0));
    }
    return Y;
}

// ---------------------------------------------------------------------------
// row-wise softmax : P[B, NC] = softmax(logits[B, NC])
// ---------------------------------------------------------------------------
Tensor<tile_f32, 2> softmax(Tensor<tile_f32, 2> logits) {
    constexpr tile_i32 M = B;
    constexpr tile_i32 N = NC;
    Tensor<tile_f32, 2> P = T.empty(T.shape(M, N), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto loc = T.alloc_fragment(T.shape(M, N), tile_f32{});
        auto row_sum = T.alloc_fragment(T.shape(M), tile_f32{});
        T.copy(logits(T.range(0, M), T.all()), loc(M, N));
        loc(M, N) = T.exp(loc(M, N));
        T.reduce_sum(loc(M, N), row_sum(M), /*dim=*/1);
        loc(M, N) *= T.rsqrt(row_sum(M));
        loc(M, N) *= T.rsqrt(row_sum(M));// 1/sum via rsqrt(x)*rsqrt(x)
        T.copy(loc(M, N), P(T.range(0, M), T.all()));
    }
    return P;
}

}// namespace cnn
