// =============================================================================
// mlp_kernels.h — Luisa tile-language kernels for MLP / MNIST training
// =============================================================================
// Header-only tile kernels shared by the `--mlp` and `--mnist` drivers:
// a fully-connected ReLU MLP trained with minibatch SGD and cross-entropy.
// The network layout matches the PyTorch scripts:
//   mlp_train.py  : 50 -> 30 -> 15 -> 4   (XOR-style synthetic task)
//   mnist_train.py: 784 -> 128 -> 10      (here: 8x8 = 64-input TinyMNIST
//                                          stand-in so the whole tile stays in
//                                          on-chip shared memory, matching the
//                                          repo's TinyCNN scale)
//
// Layout / bias handling:
//   Every layer keeps its weights W[K,O] and bias Bias[1,O] as separate
//   full-width matrices; the bias is applied by a second GEMM
//       Z = A @ W + Ones[B,1] @ Bias[1,O]
//   where Ones[B,1] is a constant all-ones buffer.  This avoids the
//   bias-folded augmented matrix [A | 1] entirely, because the current
//   tile_to_kernel lowering indexes sub-row-width global/fragment slices with
//   the slice extent as the row stride (i.e. it does not support writing only
//   the first O columns of a [B, O+1] matrix).
//
// Backward (manual backprop, no autograd):
//   dZ = dA .* relu'(Z),  relu'(z) = min(relu(z)/1e-8, 1)
//   dW = AT @ dZ          (AT is the host-precomputed input transpose for
//        layer 1 and a device-transposed activation for hidden layers)
//   db = OnesT[1,B] @ dZ  (sum over the minibatch)
//   dA_prev = dZ @ WT     (WT is the device-transposed weight; no bias row)
//   W -= lr * dW;  Bias -= lr * db   (cross-entropy gradient is 1/B-normalised)
//
// Kernels are single-block tile programs (one T.Kernel) traced with
// tile::jit(...).compile() and lowered by tile_to_kernel.  GEMM operands are
// staged through shared memory (the lowering's row-major indexing relies on
// it); pointwise ops use global slice stores so large weight updates never
// allocate a giant fragment.
// =============================================================================

#pragma once

#include <luisa/dsl/tensor.h>
#include <luisa/dsl/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace mlp {

// constexpr handle for the tile DSL (T.* spelling, like TileLang's `T`)
constexpr auto T = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

// effective learning rate for the SGD update kernel.  The cross-entropy
// gradient kernel already normalises by the minibatch size (G = (P-Y)/B), so
// the update is the plain learning rate lr=0.1 (verified in Python to converge).
constexpr float MLP_LR_EFF = 0.1f;

namespace detail {

constexpr tile_i32 THREADS = 64;

}// namespace detail

// ---------------------------------------------------------------------------
// fc_relu : Z[B,O] = A[B,K] @ W[K,O] + Ones[B,1] @ Bias[1,O];  A_out = relu(Z)
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 K, tile_i32 O>
Tensor<tile_f32, 2> fc_relu(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> W,
                            Tensor<tile_f32, 2> Bias, Tensor<tile_f32, 2> Ones,
                            Tensor<tile_f32, 2> A_out) {
    Tensor<tile_f32, 2> Z = T.empty(T.shape(B, O), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        auto A_sh = T.alloc_shared(T.shape(B, K), tile_f32{});
        auto W_sh = T.alloc_shared(T.shape(K, O), tile_f32{});
        auto Bias_sh = T.alloc_shared(T.shape(1, O), tile_f32{});
        auto Ones_sh = T.alloc_shared(T.shape(B, 1), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(B, O), tile_f32{});
        auto accb = T.alloc_fragment(T.shape(B, O), tile_f32{});
        T.copy(A(0, 0), A_sh(B, K));
        T.copy(W(0, 0), W_sh(K, O));
        T.copy(Bias(0, 0), Bias_sh(1, O));
        T.copy(Ones(0, 0), Ones_sh(B, 1));
        T.clear(acc);
        T.gemm(A_sh(B, K), W_sh(K, O), acc(B, O));
        T.clear(accb);
        T.gemm(Ones_sh(B, 1), Bias_sh(1, O), accb(B, O));
        acc(B, O) = acc(B, O) + accb(B, O);
        T.copy(acc(B, O), Z(0, 0));// pre-activation (needed by relu backward)
        acc(B, O) = T.max(acc(B, O), 0.0f);// ReLU
        T.copy(acc(B, O), A_out(0, 0));
    }
    return Z;
}

// ---------------------------------------------------------------------------
// fc : Z[B,O] = A[B,K] @ W[K,O] + Ones[B,1] @ Bias[1,O]   (last layer)
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 K, tile_i32 O>
Tensor<tile_f32, 2> fc(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> W,
                       Tensor<tile_f32, 2> Bias, Tensor<tile_f32, 2> Ones) {
    Tensor<tile_f32, 2> Z = T.empty(T.shape(B, O), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        auto A_sh = T.alloc_shared(T.shape(B, K), tile_f32{});
        auto W_sh = T.alloc_shared(T.shape(K, O), tile_f32{});
        auto Bias_sh = T.alloc_shared(T.shape(1, O), tile_f32{});
        auto Ones_sh = T.alloc_shared(T.shape(B, 1), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(B, O), tile_f32{});
        auto accb = T.alloc_fragment(T.shape(B, O), tile_f32{});
        T.copy(A(0, 0), A_sh(B, K));
        T.copy(W(0, 0), W_sh(K, O));
        T.copy(Bias(0, 0), Bias_sh(1, O));
        T.copy(Ones(0, 0), Ones_sh(B, 1));
        T.clear(acc);
        T.gemm(A_sh(B, K), W_sh(K, O), acc(B, O));
        T.clear(accb);
        T.gemm(Ones_sh(B, 1), Bias_sh(1, O), accb(B, O));
        acc(B, O) = acc(B, O) + accb(B, O);
        T.copy(acc(B, O), Z(0, 0));
    }
    return Z;
}

// ---------------------------------------------------------------------------
// softmax : P[B,C] = softmax(logits[B,C]) along dim 1
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 C>
Tensor<tile_f32, 2> softmax(Tensor<tile_f32, 2> logits) {
    Tensor<tile_f32, 2> P = T.empty(T.shape(B, C), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto loc = T.alloc_fragment(T.shape(B, C), tile_f32{});
        auto row_sum = T.alloc_fragment(T.shape(B), tile_f32{});
        T.copy(logits(T.range(0, B), T.all()), loc(B, C));
        loc(B, C) = T.exp(loc(B, C));
        T.reduce_sum(loc(B, C), row_sum(B), /*dim=*/1);
        loc(B, C) *= T.rsqrt(row_sum(B));
        loc(B, C) *= T.rsqrt(row_sum(B));// 1/sum via rsqrt(x)*rsqrt(x)
        T.copy(loc(B, C), P(T.range(0, B), T.all()));
    }
    return P;
}

// ---------------------------------------------------------------------------
// ce_grad : G[B,C] = (P[B,C] - Y[B,C]) / B   (cross-entropy gradient)
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 C>
Tensor<tile_f32, 2> ce_grad(Tensor<tile_f32, 2> P, Tensor<tile_f32, 2> Y) {
    Tensor<tile_f32, 2> G = T.empty(T.shape(B, C), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        G(T.range(0, B), T.range(0, C)) =
            (P(T.range(0, B), T.range(0, C)) +
             Y(T.range(0, B), T.range(0, C)) / (-1.0f)) /
            static_cast<float>(B);
    }
    return G;
}

// ---------------------------------------------------------------------------
// relu_backward : dZ[B,O] = dA[B,O] * min(relu(Z)/1e-8, 1)
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 O>
Tensor<tile_f32, 2> relu_backward(Tensor<tile_f32, 2> Z, Tensor<tile_f32, 2> dA) {
    Tensor<tile_f32, 2> dZ = T.empty(T.shape(B, O), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        dZ(T.range(0, B), T.range(0, O)) =
            dA(T.range(0, B), T.range(0, O)) *
            T.min(T.max(Z(T.range(0, B), T.range(0, O)), 0.0f) / 1e-8f, 1.0f);
    }
    return dZ;
}

// ---------------------------------------------------------------------------
// grad : dW[K,O] = AT[K,B] @ dZ[B,O]
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 K, tile_i32 O>
Tensor<tile_f32, 2> grad(Tensor<tile_f32, 2> AT, Tensor<tile_f32, 2> dZ) {
    Tensor<tile_f32, 2> dW = T.empty(T.shape(K, O), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        auto AT_sh = T.alloc_shared(T.shape(K, B), tile_f32{});
        auto dZ_sh = T.alloc_shared(T.shape(B, O), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(K, O), tile_f32{});
        T.copy(AT(0, 0), AT_sh(K, B));
        T.copy(dZ(0, 0), dZ_sh(B, O));
        T.clear(acc);
        T.gemm(AT_sh(K, B), dZ_sh(B, O), acc(K, O));
        T.copy(acc(K, O), dW(0, 0));
    }
    return dW;
}

// ---------------------------------------------------------------------------
// grad_bias : db[1,O] = OnesT[1,B] @ dZ[B,O]   (sum over the minibatch)
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 O>
Tensor<tile_f32, 2> grad_bias(Tensor<tile_f32, 2> OnesT, Tensor<tile_f32, 2> dZ) {
    Tensor<tile_f32, 2> db = T.empty(T.shape(1, O), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto OnesT_sh = T.alloc_shared(T.shape(1, B), tile_f32{});
        auto dZ_sh = T.alloc_shared(T.shape(B, O), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(1, O), tile_f32{});
        T.copy(OnesT(0, 0), OnesT_sh(1, B));
        T.copy(dZ(0, 0), dZ_sh(B, O));
        T.clear(acc);
        T.gemm(OnesT_sh(1, B), dZ_sh(B, O), acc(1, O));
        T.copy(acc(1, O), db(0, 0));
    }
    return db;
}

// ---------------------------------------------------------------------------
// fc_backward : dA[B,K] = dZ[B,O] @ WT[O,K]   (WT = W^T; no bias row to drop)
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 K, tile_i32 O>
Tensor<tile_f32, 2> fc_backward(Tensor<tile_f32, 2> dZ, Tensor<tile_f32, 2> WT) {
    Tensor<tile_f32, 2> dA = T.empty(T.shape(B, K), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        auto dZ_sh = T.alloc_shared(T.shape(B, O), tile_f32{});
        auto WT_sh = T.alloc_shared(T.shape(O, K), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(B, K), tile_f32{});
        T.copy(dZ(0, 0), dZ_sh(B, O));
        T.copy(WT(0, 0), WT_sh(O, K));
        T.clear(acc);
        T.gemm(dZ_sh(B, O), WT_sh(O, K), acc(B, K));
        T.copy(acc(B, K), dA(0, 0));
    }
    return dA;
}

// ---------------------------------------------------------------------------
// transpose : dst[N,M] = src[M,N]^T
// ---------------------------------------------------------------------------
template <tile_i32 M, tile_i32 N>
Tensor<tile_f32, 2> transpose(Tensor<tile_f32, 2> src) {
    Tensor<tile_f32, 2> dst = T.empty(T.shape(N, M), tile_f32{});
    for (auto bx : T.Kernel(1, 1, 32)) {
        auto src_sh = T.alloc_shared(T.shape(M, N), tile_f32{});
        auto dst_sh = T.alloc_shared(T.shape(N, M), tile_f32{});
        T.copy(src(0, 0), src_sh(M, N));
        T.transpose(src_sh(M, N), dst_sh(N, M));
        T.sync_threads();
        T.copy(dst_sh(N, M), dst(0, 0));
    }
    return dst;
}

// ---------------------------------------------------------------------------
// update : W_new[K,O] = W[K,O] - MLP_LR_EFF * dW[K,O]   (global slice store)
// ---------------------------------------------------------------------------
template <tile_i32 K, tile_i32 O>
Tensor<tile_f32, 2> update(Tensor<tile_f32, 2> W, Tensor<tile_f32, 2> dW) {
    Tensor<tile_f32, 2> W_new = T.empty(T.shape(K, O), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        W_new(T.range(0, K), T.range(0, O)) =
            W(T.range(0, K), T.range(0, O)) +
            dW(T.range(0, K), T.range(0, O)) / (-1.0f / MLP_LR_EFF);
    }
    return W_new;
}

// ---------------------------------------------------------------------------
// update_bias : Bias_new[1,O] = Bias[1,O] - MLP_LR_EFF * db[1,O]
// ---------------------------------------------------------------------------
template <tile_i32 O>
Tensor<tile_f32, 2> update_bias(Tensor<tile_f32, 2> Bias, Tensor<tile_f32, 2> db) {
    Tensor<tile_f32, 2> Bias_new = T.empty(T.shape(1, O), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        Bias_new(T.range(0, 1), T.range(0, O)) =
            Bias(T.range(0, 1), T.range(0, O)) +
            db(T.range(0, 1), T.range(0, O)) / (-1.0f / MLP_LR_EFF);
    }
    return Bias_new;
}



}// namespace mlp
