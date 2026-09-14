// =============================================================================
// rnn_kernels.h — Luisa tile-language kernels for RNN sequence classification
// =============================================================================
// Header-only tile kernels for the `--rnn` driver (C++ twin of rnn_train.py):
// a tanh RNN that counts ones in binary sequences of length T.
//
// Model (batch B, hidden H):
//   X_t[B,1]        = input bit at timestep t
//   H_state[B,H]    = hidden state (no bias column; see mlp_kernels.h why)
//   Wih[1,H], Whh[H,H], Bias_ih[1,H], Bias_hh[1,H]
//   step: z = X_t @ Wih + H_state @ Whh + Bias_ih + Bias_hh;  h = tanh(z)
//   logits = H_final @ Wfc + Bias_fc,  Wfc[H,C], Bias_fc[1,C]
//
// Backward through time (BPTT):
//   dZ = dH .* (1 - h^2)                             (tanh derivative)
//   dWih += Xt_t @ dZ;  dWhh += H_prevT @ dZ
//   db_ih += OnesT @ dZ;  db_hh += OnesT @ dZ
//   dH_prev = dZ @ Whh^T
//
// The fc / softmax / cross-entropy / transpose / update / grad kernels are
// shared with the MLP module (mlp_kernels.h); this header only adds the
// RNN-specific step, tanh-backward and gradient-accumulation kernels.
// =============================================================================

#pragma once

#include "mlp_kernels.h"

namespace rnn {

using luisa::compute::tile::language::Tensor;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;
using mlp::T;

// ---------------------------------------------------------------------------
// rnn_step : H_next[B,H] = tanh(X @ Wih + H @ Whh + Bias_ih + Bias_hh)
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 H>
Tensor<tile_f32, 2> rnn_step(Tensor<tile_f32, 2> X, Tensor<tile_f32, 2> H_state,
                             Tensor<tile_f32, 2> Wih, Tensor<tile_f32, 2> Whh,
                             Tensor<tile_f32, 2> Bias_ih, Tensor<tile_f32, 2> Bias_hh,
                             Tensor<tile_f32, 2> Ones) {
    Tensor<tile_f32, 2> H_new = T.empty(T.shape(B, H), tile_f32{});
    for (auto bx : T.Kernel(1, 64)) {
        auto X_sh = T.alloc_shared(T.shape(B, 1), tile_f32{});
        auto H_sh = T.alloc_shared(T.shape(B, H), tile_f32{});
        auto Wih_sh = T.alloc_shared(T.shape(1, H), tile_f32{});
        auto Whh_sh = T.alloc_shared(T.shape(H, H), tile_f32{});
        auto Bi_sh = T.alloc_shared(T.shape(1, H), tile_f32{});
        auto Bh_sh = T.alloc_shared(T.shape(1, H), tile_f32{});
        auto Ones_sh = T.alloc_shared(T.shape(B, 1), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(B, H), tile_f32{});
        auto h = T.alloc_fragment(T.shape(B, H), tile_f32{});
        T.copy(X(0, 0), X_sh(B, 1));
        T.copy(H_state(0, 0), H_sh(B, H));
        T.copy(Wih(0, 0), Wih_sh(1, H));
        T.copy(Whh(0, 0), Whh_sh(H, H));
        T.copy(Bias_ih(0, 0), Bi_sh(1, H));
        T.copy(Bias_hh(0, 0), Bh_sh(1, H));
        T.copy(Ones(0, 0), Ones_sh(B, 1));
        T.clear(acc);
        T.gemm(X_sh(B, 1), Wih_sh(1, H), acc(B, H));
        T.gemm(H_sh(B, H), Whh_sh(H, H), acc(B, H));
        T.gemm(Ones_sh(B, 1), Bi_sh(1, H), acc(B, H));
        T.gemm(Ones_sh(B, 1), Bh_sh(1, H), acc(B, H));
        h(B, H) = T.tanh(acc(B, H));
        T.copy(h(B, H), H_new(0, 0));
    }
    return H_new;
}

// ---------------------------------------------------------------------------
// tanh_backward : dZ[B,H] = dH[B,H] .* (1 - h^2)
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 H>
Tensor<tile_f32, 2> tanh_backward(Tensor<tile_f32, 2> H_state, Tensor<tile_f32, 2> dH) {
    Tensor<tile_f32, 2> dZ = T.empty(T.shape(B, H), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto h = H_state(T.range(0, B), T.range(0, H));
        dZ(T.range(0, B), T.range(0, H)) =
            dH(T.range(0, B), T.range(0, H)) * ((h * h) / (-1.0f) + 1.0f);
    }
    return dZ;
}

// ---------------------------------------------------------------------------
// grad_accum : dW[K,O] = dW[K,O] + AT[K,B] @ dZ[B,O]
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 K, tile_i32 O>
Tensor<tile_f32, 2> grad_accum(Tensor<tile_f32, 2> AT, Tensor<tile_f32, 2> dZ,
                               Tensor<tile_f32, 2> dW) {
    Tensor<tile_f32, 2> dW_out = T.empty(T.shape(K, O), tile_f32{});
    for (auto bx : T.Kernel(1, 64)) {
        auto AT_sh = T.alloc_shared(T.shape(K, B), tile_f32{});
        auto dZ_sh = T.alloc_shared(T.shape(B, O), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(K, O), tile_f32{});
        T.copy(dW(0, 0), acc(K, O));// start from the existing accumulator
        T.copy(AT(0, 0), AT_sh(K, B));
        T.copy(dZ(0, 0), dZ_sh(B, O));
        T.gemm(AT_sh(K, B), dZ_sh(B, O), acc(K, O));// acc += AT @ dZ
        T.copy(acc(K, O), dW_out(0, 0));
    }
    return dW_out;
}

// ---------------------------------------------------------------------------
// grad_accum_bias : db[1,O] = db[1,O] + OnesT[1,B] @ dZ[B,O]
// ---------------------------------------------------------------------------
template <tile_i32 B, tile_i32 O>
Tensor<tile_f32, 2> grad_accum_bias(Tensor<tile_f32, 2> OnesT, Tensor<tile_f32, 2> dZ,
                                    Tensor<tile_f32, 2> db) {
    Tensor<tile_f32, 2> db_out = T.empty(T.shape(1, O), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        auto OnesT_sh = T.alloc_shared(T.shape(1, B), tile_f32{});
        auto dZ_sh = T.alloc_shared(T.shape(B, O), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(1, O), tile_f32{});
        T.copy(db(0, 0), acc(1, O));
        T.copy(OnesT(0, 0), OnesT_sh(1, B));
        T.copy(dZ(0, 0), dZ_sh(B, O));
        T.gemm(OnesT_sh(1, B), dZ_sh(B, O), acc(1, O));// acc += OnesT @ dZ
        T.copy(acc(1, O), db_out(0, 0));
    }
    return db_out;
}

// ---------------------------------------------------------------------------
// clear2d : dst[K,O] = 0   (used to zero the BPTT gradient accumulators;
// the same buffer may be passed for src and dst because the fill never reads
// the source)
// ---------------------------------------------------------------------------
template <tile_i32 K, tile_i32 O>
Tensor<tile_f32, 2> clear2d(Tensor<tile_f32, 2> buf) {
    Tensor<tile_f32, 2> out = T.empty(T.shape(K, O), tile_f32{});
    for (auto bx : T.Kernel(1, 32)) {
        T.fill(out(T.range(0, K), T.range(0, O)), 0.0f);
    }
    return out;
}

}// namespace rnn
