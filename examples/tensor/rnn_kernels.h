// =============================================================================
// rnn_kernels.h — New Tile DSL kernels for RNN sequence classification
// =============================================================================
// Header-only port of backup_old_tile/examples/tensor/rnn_kernels.h (plus the
// mlp_kernels.h inventory the old --rnn driver compiled) to the
// execution-structure-first C++ Tile DSL (<luisa/tile/dsl.h>). Model
// (batch B, hidden H), the C++ twin of rnn_train.py:
//   X_t[B,1]        = input bit at timestep t
//   H_state[B,H]    = hidden state
//   Wih[1,H], Whh[H,H], Bias_ih[1,H], Bias_hh[1,H]
//   step: z = X_t @ Wih + H_state @ Whh + Bias_ih + Bias_hh;  h = tanh(z)
//   logits = H_final @ Wfc + Bias_fc,  Wfc[H,C], Bias_fc[1,C]
//
// Backward through time (BPTT), mirroring the old kernel inventory:
//   dZ = dH .* (1 - h^2)                             (tanh derivative)
//   dWih += Xt_t @ dZ;  dWhh += H_prevT @ dZ
//   db_ih += OnesT @ dZ;  db_hh += OnesT @ dZ
//   dH_prev = dZ @ Whh^T
//
// Layout / bias handling is exactly the old MLP/RNN convention: weights and
// biases are separate full-width matrices; the bias is applied by a GEMM
// against an all-ones buffer (Z = A @ W + Ones[B,1] @ Bias[1,O]).
//
// Every kernel is one root parallel(shape(1)) nest over whole-tile loads,
// matching the old single-block tile programs. GEMMs compose tile::mma;
// transposes compose reindex; clears store zeros.
// =============================================================================

#pragma once

#include <cstdint>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/dsl.h>

namespace tensor_example::rnntrain {

namespace tile = luisa::compute::tile;

// Effective learning rate for the SGD update kernels. The cross-entropy
// gradient kernel already normalises by the minibatch size (G = (P-Y)/B), so
// the update is the plain learning rate lr=0.1 (verified in Python to
// converge) — same constant as the old mlp_kernels.h.
inline constexpr float RNN_LR_EFF = 0.1f;

// ---------------------------------------------------------------------------
// rnn_step : H_next[B,H] = tanh(X @ Wih + H @ Whh + Bias_ih + Bias_hh)
// (the four T.gemm accumulations of the old kernel are one mma accumulator
// chain)
// ---------------------------------------------------------------------------
template<int64_t B, int64_t H>
[[nodiscard]] inline tile::Kernel make_rnn_step_kernel() {
    auto definition = tile::tile_kernel("rnn_step", [=](
                                            tile::TensorView<const float, 2> X,
                                            tile::TensorView<const float, 2> H_state,
                                            tile::TensorView<const float, 2> Wih,
                                            tile::TensorView<const float, 2> Whh,
                                            tile::TensorView<const float, 2> Bias_ih,
                                            tile::TensorView<const float, 2> Bias_hh,
                                            tile::TensorView<const float, 2> Ones,
                                            tile::TensorView<float, 2> H_new) {
        using namespace tile;
        auto b = axis("b", B), one = axis("one", 1), h = axis("h", H);
        // The contraction axis of H @ Whh must be a dimension distinct from
        // the output axis: a shape cannot contain the same axis twice, and a
        // named-dimension MMA contracts exactly the axes shared by the
        // operands and absent from the accumulator.
        auto hk = axis("hk", H);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto x = X.tile(coord(0, 0), shape(b, one)).load();
            auto hs = H_state.tile(coord(0, 0), shape(b, hk)).load();
            auto wih = Wih.tile(coord(0, 0), shape(one, h)).load();
            auto whh = Whh.tile(coord(0, 0), shape(hk, h)).load();
            auto bi = Bias_ih.tile(coord(0, 0), shape(one, h)).load();
            auto bh = Bias_hh.tile(coord(0, 0), shape(one, h)).load();
            auto ones = Ones.tile(coord(0, 0), shape(b, one)).load();
            auto acc = mma(x, wih, zeros<float>(shape(b, h)));
            acc = mma(hs, whh, acc);
            acc = mma(ones, bi, acc);
            acc = mma(ones, bh, acc);
            H_new.tile(coord(0, 0), shape(b, h)).store(tanh(acc));
        }
    });
    return definition.capture(tile::tensor_shape("X", B, 1),
                              tile::tensor_shape("H_state", B, H),
                              tile::tensor_shape("Wih", 1, H),
                              tile::tensor_shape("Whh", H, H),
                              tile::tensor_shape("Bias_ih", 1, H),
                              tile::tensor_shape("Bias_hh", 1, H),
                              tile::tensor_shape("Ones", B, 1),
                              tile::tensor_shape("H_new", B, H));
}

// ---------------------------------------------------------------------------
// fc : Z[B,O] = A[B,K] @ W[K,O] + Ones[B,1] @ Bias[1,O]   (last layer)
// ---------------------------------------------------------------------------
template<int64_t B, int64_t K, int64_t O>
[[nodiscard]] inline tile::Kernel make_rnn_fc_kernel() {
    auto definition = tile::tile_kernel("rnn_fc", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<const float, 2> W,
                                            tile::TensorView<const float, 2> Bias,
                                            tile::TensorView<const float, 2> Ones,
                                            tile::TensorView<float, 2> Z) {
        using namespace tile;
        auto b = axis("b", B), k = axis("k", K), o = axis("o", O), one = axis("one", 1);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto a = A.tile(coord(0, 0), shape(b, k)).load();
            auto w = W.tile(coord(0, 0), shape(k, o)).load();
            auto bias = Bias.tile(coord(0, 0), shape(one, o)).load();
            auto ones = Ones.tile(coord(0, 0), shape(b, one)).load();
            auto acc = mma(a, w, zeros<float>(shape(b, o)));
            acc = mma(ones, bias, acc);
            Z.tile(coord(0, 0), shape(b, o)).store(acc);
        }
    });
    return definition.capture(tile::tensor_shape("A", B, K),
                              tile::tensor_shape("W", K, O),
                              tile::tensor_shape("Bias", 1, O),
                              tile::tensor_shape("Ones", B, 1),
                              tile::tensor_shape("Z", B, O));
}

// ---------------------------------------------------------------------------
// softmax : P[B,C] = softmax(logits[B,C]) along dim 1. The old kernel used
// exp(x)/sum via rsqrt(row_sum)^2; the new TileIR has no rsqrt opcode, so
// this divides by the broadcast row sum (equal to ~1 ulp, far inside
// tolerance).
// ---------------------------------------------------------------------------
template<int64_t B, int64_t C>
[[nodiscard]] inline tile::Kernel make_rnn_softmax_kernel() {
    auto definition = tile::tile_kernel("rnn_softmax", [=](
                                            tile::TensorView<const float, 2> logits,
                                            tile::TensorView<float, 2> probs) {
        using namespace tile;
        auto b = axis("b", B), c = axis("c", C);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto l = logits.tile(coord(0, 0), shape(b, c)).load();
            auto e = exp(l);
            auto denom = reduce(e, c, add);
            probs.tile(coord(0, 0), shape(b, c)).store(e / denom);
        }
    });
    return definition.capture(tile::tensor_shape("logits", B, C),
                              tile::tensor_shape("probs", B, C));
}

// ---------------------------------------------------------------------------
// ce_grad : G[B,C] = (P[B,C] - Y[B,C]) / B   (cross-entropy gradient)
// ---------------------------------------------------------------------------
template<int64_t B, int64_t C>
[[nodiscard]] inline tile::Kernel make_rnn_ce_grad_kernel() {
    auto definition = tile::tile_kernel("rnn_ce_grad", [=](
                                            tile::TensorView<const float, 2> P,
                                            tile::TensorView<const float, 2> Y,
                                            tile::TensorView<float, 2> G) {
        using namespace tile;
        auto b = axis("b", B), c = axis("c", C);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto p = P.tile(coord(0, 0), shape(b, c)).load();
            auto y = Y.tile(coord(0, 0), shape(b, c)).load();
            G.tile(coord(0, 0), shape(b, c)).store((p - y) / static_cast<float>(B));
        }
    });
    return definition.capture(tile::tensor_shape("P", B, C),
                              tile::tensor_shape("Y", B, C),
                              tile::tensor_shape("G", B, C));
}

// ---------------------------------------------------------------------------
// tanh_backward : dZ[B,H] = dH[B,H] .* (1 - h^2)
// ---------------------------------------------------------------------------
template<int64_t B, int64_t H>
[[nodiscard]] inline tile::Kernel make_rnn_tanh_backward_kernel() {
    auto definition = tile::tile_kernel("rnn_tanh_backward", [=](
                                            tile::TensorView<const float, 2> H_state,
                                            tile::TensorView<const float, 2> dH,
                                            tile::TensorView<float, 2> dZ) {
        using namespace tile;
        auto b = axis("b", B), h = axis("h", H);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto hs = H_state.tile(coord(0, 0), shape(b, h)).load();
            auto dh = dH.tile(coord(0, 0), shape(b, h)).load();
            dZ.tile(coord(0, 0), shape(b, h)).store(dh * (1.0f - hs * hs));
        }
    });
    return definition.capture(tile::tensor_shape("H_state", B, H),
                              tile::tensor_shape("dH", B, H),
                              tile::tensor_shape("dZ", B, H));
}

// ---------------------------------------------------------------------------
// fc_backward : dA[B,K] = dZ[B,O] @ WT[O,K]   (WT = W^T; no bias row to drop)
// ---------------------------------------------------------------------------
template<int64_t B, int64_t K, int64_t O>
[[nodiscard]] inline tile::Kernel make_rnn_fc_backward_kernel() {
    auto definition = tile::tile_kernel("rnn_fc_backward", [=](
                                            tile::TensorView<const float, 2> dZ,
                                            tile::TensorView<const float, 2> WT,
                                            tile::TensorView<float, 2> dA) {
        using namespace tile;
        auto b = axis("b", B), o = axis("o", O), k = axis("k", K);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto dz = dZ.tile(coord(0, 0), shape(b, o)).load();
            auto wt = WT.tile(coord(0, 0), shape(o, k)).load();
            auto acc = mma(dz, wt, zeros<float>(shape(b, k)));
            dA.tile(coord(0, 0), shape(b, k)).store(acc);
        }
    });
    return definition.capture(tile::tensor_shape("dZ", B, O),
                              tile::tensor_shape("WT", O, K),
                              tile::tensor_shape("dA", B, K));
}

// ---------------------------------------------------------------------------
// grad : dW[K,O] = AT[K,B] @ dZ[B,O]
// ---------------------------------------------------------------------------
template<int64_t B, int64_t K, int64_t O>
[[nodiscard]] inline tile::Kernel make_rnn_grad_kernel() {
    auto definition = tile::tile_kernel("rnn_grad", [=](
                                            tile::TensorView<const float, 2> AT,
                                            tile::TensorView<const float, 2> dZ,
                                            tile::TensorView<float, 2> dW) {
        using namespace tile;
        auto k = axis("k", K), b = axis("b", B), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto at = AT.tile(coord(0, 0), shape(k, b)).load();
            auto dz = dZ.tile(coord(0, 0), shape(b, o)).load();
            auto acc = mma(at, dz, zeros<float>(shape(k, o)));
            dW.tile(coord(0, 0), shape(k, o)).store(acc);
        }
    });
    return definition.capture(tile::tensor_shape("AT", K, B),
                              tile::tensor_shape("dZ", B, O),
                              tile::tensor_shape("dW", K, O));
}

// ---------------------------------------------------------------------------
// grad_bias : db[1,O] = OnesT[1,B] @ dZ[B,O]   (sum over the minibatch)
// ---------------------------------------------------------------------------
template<int64_t B, int64_t O>
[[nodiscard]] inline tile::Kernel make_rnn_grad_bias_kernel() {
    auto definition = tile::tile_kernel("rnn_grad_bias", [=](
                                            tile::TensorView<const float, 2> OnesT,
                                            tile::TensorView<const float, 2> dZ,
                                            tile::TensorView<float, 2> db) {
        using namespace tile;
        auto one = axis("one", 1), b = axis("b", B), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto ones_t = OnesT.tile(coord(0, 0), shape(one, b)).load();
            auto dz = dZ.tile(coord(0, 0), shape(b, o)).load();
            auto acc = mma(ones_t, dz, zeros<float>(shape(one, o)));
            db.tile(coord(0, 0), shape(one, o)).store(acc);
        }
    });
    return definition.capture(tile::tensor_shape("OnesT", 1, B),
                              tile::tensor_shape("dZ", B, O),
                              tile::tensor_shape("db", 1, O));
}

// ---------------------------------------------------------------------------
// grad_accum : dW_out[K,O] = dW[K,O] + AT[K,B] @ dZ[B,O]   (BPTT accumulator)
// ---------------------------------------------------------------------------
template<int64_t B, int64_t K, int64_t O>
[[nodiscard]] inline tile::Kernel make_rnn_grad_accum_kernel() {
    auto definition = tile::tile_kernel("rnn_grad_accum", [=](
                                            tile::TensorView<const float, 2> AT,
                                            tile::TensorView<const float, 2> dZ,
                                            tile::TensorView<const float, 2> dW,
                                            tile::TensorView<float, 2> dW_out) {
        using namespace tile;
        auto k = axis("k", K), b = axis("b", B), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto at = AT.tile(coord(0, 0), shape(k, b)).load();
            auto dz = dZ.tile(coord(0, 0), shape(b, o)).load();
            auto dw = dW.tile(coord(0, 0), shape(k, o)).load();
            auto acc = mma(at, dz, zeros<float>(shape(k, o)));
            dW_out.tile(coord(0, 0), shape(k, o)).store(dw + acc);
        }
    });
    return definition.capture(tile::tensor_shape("AT", K, B),
                              tile::tensor_shape("dZ", B, O),
                              tile::tensor_shape("dW", K, O),
                              tile::tensor_shape("dW_out", K, O));
}

// ---------------------------------------------------------------------------
// grad_accum_bias : db_out[1,O] = db[1,O] + OnesT[1,B] @ dZ[B,O]
// ---------------------------------------------------------------------------
template<int64_t B, int64_t O>
[[nodiscard]] inline tile::Kernel make_rnn_grad_accum_bias_kernel() {
    auto definition = tile::tile_kernel("rnn_grad_accum_bias", [=](
                                            tile::TensorView<const float, 2> OnesT,
                                            tile::TensorView<const float, 2> dZ,
                                            tile::TensorView<const float, 2> db,
                                            tile::TensorView<float, 2> db_out) {
        using namespace tile;
        auto one = axis("one", 1), b = axis("b", B), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto ones_t = OnesT.tile(coord(0, 0), shape(one, b)).load();
            auto dz = dZ.tile(coord(0, 0), shape(b, o)).load();
            auto dbias = db.tile(coord(0, 0), shape(one, o)).load();
            auto acc = mma(ones_t, dz, zeros<float>(shape(one, o)));
            db_out.tile(coord(0, 0), shape(one, o)).store(dbias + acc);
        }
    });
    return definition.capture(tile::tensor_shape("OnesT", 1, B),
                              tile::tensor_shape("dZ", B, O),
                              tile::tensor_shape("db", 1, O),
                              tile::tensor_shape("db_out", 1, O));
}

// ---------------------------------------------------------------------------
// clear2d : dst[K,O] = 0   (zeroes the BPTT gradient accumulators)
// ---------------------------------------------------------------------------
template<int64_t K, int64_t O>
[[nodiscard]] inline tile::Kernel make_rnn_clear2d_kernel() {
    auto definition = tile::tile_kernel("rnn_clear2d", [=](
                                            tile::TensorView<float, 2> dst) {
        using namespace tile;
        auto k = axis("k", K), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            dst.tile(coord(0, 0), shape(k, o)).store(zeros<float>(shape(k, o)));
        }
    });
    return definition.capture(tile::tensor_shape("dst", K, O));
}

// ---------------------------------------------------------------------------
// transpose : dst[N,M] = src[M,N]^T   (reindex, like kernels.h::transpose)
// ---------------------------------------------------------------------------
template<int64_t M, int64_t N>
[[nodiscard]] inline tile::Kernel make_rnn_transpose_kernel() {
    auto definition = tile::tile_kernel("rnn_transpose", [=](
                                            tile::TensorView<const float, 2> src,
                                            tile::TensorView<float, 2> dst) {
        using namespace tile;
        auto m = axis("m", M), n = axis("n", N);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto s = src.tile(coord(0, 0), shape(m, n)).load();
            auto t = reindex(s, shape(n, m), [&](const Nest &element) {
                return coord(element.index(m), element.index(n));
            });
            dst.tile(coord(0, 0), shape(n, m)).store(t);
        }
    });
    return definition.capture(tile::tensor_shape("src", M, N),
                              tile::tensor_shape("dst", N, M));
}

// ---------------------------------------------------------------------------
// update : W_new[K,O] = W[K,O] - RNN_LR_EFF * dW[K,O]
// ---------------------------------------------------------------------------
template<int64_t K, int64_t O>
[[nodiscard]] inline tile::Kernel make_rnn_update_kernel() {
    auto definition = tile::tile_kernel("rnn_update", [=](
                                            tile::TensorView<const float, 2> W,
                                            tile::TensorView<const float, 2> dW,
                                            tile::TensorView<float, 2> W_new) {
        using namespace tile;
        auto k = axis("k", K), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto w = W.tile(coord(0, 0), shape(k, o)).load();
            auto dw = dW.tile(coord(0, 0), shape(k, o)).load();
            W_new.tile(coord(0, 0), shape(k, o)).store(w - dw * RNN_LR_EFF);
        }
    });
    return definition.capture(tile::tensor_shape("W", K, O),
                              tile::tensor_shape("dW", K, O),
                              tile::tensor_shape("W_new", K, O));
}

// ---------------------------------------------------------------------------
// update_bias : Bias_new[1,O] = Bias[1,O] - RNN_LR_EFF * db[1,O]
// ---------------------------------------------------------------------------
template<int64_t O>
[[nodiscard]] inline tile::Kernel make_rnn_update_bias_kernel() {
    auto definition = tile::tile_kernel("rnn_update_bias", [=](
                                            tile::TensorView<const float, 2> Bias,
                                            tile::TensorView<const float, 2> db,
                                            tile::TensorView<float, 2> Bias_new) {
        using namespace tile;
        auto one = axis("one", 1), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto bias = Bias.tile(coord(0, 0), shape(one, o)).load();
            auto dbias = db.tile(coord(0, 0), shape(one, o)).load();
            Bias_new.tile(coord(0, 0), shape(one, o)).store(bias - dbias * RNN_LR_EFF);
        }
    });
    return definition.capture(tile::tensor_shape("Bias", 1, O),
                              tile::tensor_shape("db", 1, O),
                              tile::tensor_shape("Bias_new", 1, O));
}

}// namespace tensor_example::rnntrain
