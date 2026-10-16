// =============================================================================
// mlp_kernels.h — new Tile DSL kernels for MLP / MNIST training
// =============================================================================
// Header-only execution-structure-first Tile kernels shared by the `--mlp`
// and `--mnist` drivers: a fully-connected ReLU MLP trained with minibatch SGD
// and cross-entropy.  Port of `backup_old_tile/examples/tensor/mlp_kernels.h`
// (LuisaTensor / tile::jit / T.gemm) to the C++ Tile DSL of
// <luisa/tile/dsl.h>.
//
// Network layout (unchanged from the old example / the PyTorch scripts):
//   mlp_train.py  : 50 -> 30 -> 15 -> 4   (XOR-style synthetic task)
//   mnist_train.py: 64 -> 32 -> 10        (8x8 TinyMNIST stand-in)
//
// Layout / bias handling (preserved):
//   Every layer keeps its weights W[K,O] and bias Bias[1,O] as separate
//   full-width matrices; the bias is applied by a second GEMM
//       Z = A @ W + Ones[B,1] @ Bias[1,O]
//   where Ones[B,1] is a constant all-ones buffer, exactly like the old
//   kernels (the old comment about the augmented-matrix slice-stride
//   lowering bug is historical; the second GEMM is kept to mirror semantics).
//
// Backward (manual backprop, no autograd — same math as the old kernels):
//   dZ = dA .* relu'(Z),  relu'(z) = min(relu(z)/1e-8, 1)
//   dW = AT @ dZ          (AT is the host-precomputed input transpose for
//        layer 1 and a device-transposed activation for hidden layers)
//   db = OnesT[1,B] @ dZ  (sum over the minibatch)
//   dA_prev = dZ @ WT     (WT is the device-transposed weight; no bias row)
//   W -= lr * dW;  Bias -= lr * db   (cross-entropy gradient is 1/B-normalised)
//
// New-DSL staging (changed from the old syntax, not the algorithm):
//   Each kernel is a single-root `parallel(shape(1))` tile program that loads
//   whole matrices as tiles and uses `tile::mma` for every GEMM.  GEMM
//   operands use named axes, so the contraction is an explicit named-dimension
//   contraction (the all-ones bias GEMM contracts the singleton axis "s").
//   Transposes compose `tile::reindex` (see examples/compute/tile/kernels.h).
//   The old global-slice-store updates become whole-tile stores; `update` /
//   `update_bias` read and rewrite their parameter tile in place (sizes are
//   tiny), which the invocation ABI allows because each kernel has a single
//   writable argument view.
//
//   The old softmax divided by the row sum via rsqrt(row_sum)^2 because the
//   old TileLang surface had no divide-by-tile; the new kernel divides the
//   exponential tile by the broadcast row sum directly (same math, and the
//   unstabilised exp/sum form of the old device kernel is kept — logits are
//   small in both demos).
// =============================================================================

#pragma once

#include <cstdint>

#include <luisa/tile/algorithms.h>
#include <luisa/tile/dsl.h>

namespace mlp {

namespace tile = luisa::compute::tile;

// Effective learning rate for the SGD update kernels.  The cross-entropy
// gradient kernel already normalises by the minibatch size (G = (P-Y)/B), so
// the update is the plain learning rate lr=0.1 (verified in Python to
// converge).  Mirrors the old MLP_LR_EFF.
constexpr float MLP_LR_EFF = 0.1f;

// ---------------------------------------------------------------------------
// fc_relu : Z[B,O] = A[B,K] @ W[K,O] + Ones[B,1] @ Bias[1,O];  A_out = relu(Z)
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_fc_relu(int64_t B, int64_t K, int64_t O) {
    auto definition = tile::tile_kernel("mlp_fc_relu", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<const float, 2> W,
                                            tile::TensorView<const float, 2> Bias,
                                            tile::TensorView<const float, 2> Ones,
                                            tile::TensorView<float, 2> Z,
                                            tile::TensorView<float, 2> A_out) {
        using namespace tile;
        auto b = axis("b", B), k = axis("k", K), o = axis("o", O), s = axis("s", 1);
        for (auto &nest : parallel(shape(1))) {
            auto a = A.tile(coord(0, 0), shape(b, k)).load();
            auto w = W.tile(coord(0, 0), shape(k, o)).load();
            auto bias = Bias.tile(coord(0, 0), shape(s, o)).load();
            auto ones = Ones.tile(coord(0, 0), shape(b, s)).load();
            auto z = mma(a, w, zeros<float>(shape(b, o)));
            z = mma(ones, bias, z);// bias via the all-ones GEMM, as in the old kernel
            Z.tile(coord(0, 0), shape(b, o)).store(z);// pre-activation (relu backward input)
            A_out.tile(coord(0, 0), shape(b, o)).store(max(z, 0.0f));// ReLU
        }
    });
    return definition.capture(tile::tensor_shape("A", B, K), tile::tensor_shape("W", K, O),
                              tile::tensor_shape("Bias", 1, O), tile::tensor_shape("Ones", B, 1),
                              tile::tensor_shape("Z", B, O), tile::tensor_shape("A_out", B, O));
}

// ---------------------------------------------------------------------------
// fc : Z[B,O] = A[B,K] @ W[K,O] + Ones[B,1] @ Bias[1,O]   (last layer)
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_fc(int64_t B, int64_t K, int64_t O) {
    auto definition = tile::tile_kernel("mlp_fc", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<const float, 2> W,
                                            tile::TensorView<const float, 2> Bias,
                                            tile::TensorView<const float, 2> Ones,
                                            tile::TensorView<float, 2> Z) {
        using namespace tile;
        auto b = axis("b", B), k = axis("k", K), o = axis("o", O), s = axis("s", 1);
        for (auto &nest : parallel(shape(1))) {
            auto a = A.tile(coord(0, 0), shape(b, k)).load();
            auto w = W.tile(coord(0, 0), shape(k, o)).load();
            auto bias = Bias.tile(coord(0, 0), shape(s, o)).load();
            auto ones = Ones.tile(coord(0, 0), shape(b, s)).load();
            auto z = mma(a, w, zeros<float>(shape(b, o)));
            z = mma(ones, bias, z);
            Z.tile(coord(0, 0), shape(b, o)).store(z);
        }
    });
    return definition.capture(tile::tensor_shape("A", B, K), tile::tensor_shape("W", K, O),
                              tile::tensor_shape("Bias", 1, O), tile::tensor_shape("Ones", B, 1),
                              tile::tensor_shape("Z", B, O));
}

// ---------------------------------------------------------------------------
// softmax : P[B,C] = softmax(logits[B,C]) along dim 1 (unstabilised exp/sum,
// matching the old device kernel; the host reference is stabilised and both
// agree at these logit magnitudes)
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_softmax(int64_t B, int64_t C) {
    auto definition = tile::tile_kernel("mlp_softmax", [=](
                                            tile::TensorView<const float, 2> logits,
                                            tile::TensorView<float, 2> P) {
        using namespace tile;
        auto b = axis("b", B), c = axis("c", C);
        for (auto &nest : parallel(shape(1))) {
            auto x = logits.tile(coord(0, 0), shape(b, c)).load();
            auto e = exp(x);
            auto denom = reduce(e, c, add);
            P.tile(coord(0, 0), shape(b, c)).store(e / denom);
        }
    });
    return definition.capture(tile::tensor_shape("logits", B, C), tile::tensor_shape("P", B, C));
}

// ---------------------------------------------------------------------------
// ce_grad : G[B,C] = (P[B,C] - Y[B,C]) / B   (cross-entropy gradient)
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_ce_grad(int64_t B, int64_t C) {
    auto definition = tile::tile_kernel("mlp_ce_grad", [=](
                                            tile::TensorView<const float, 2> P,
                                            tile::TensorView<const float, 2> Y,
                                            tile::TensorView<float, 2> G) {
        using namespace tile;
        auto b = axis("b", B), c = axis("c", C);
        for (auto &nest : parallel(shape(1))) {
            auto p = P.tile(coord(0, 0), shape(b, c)).load();
            auto y = Y.tile(coord(0, 0), shape(b, c)).load();
            G.tile(coord(0, 0), shape(b, c)).store((p - y) / static_cast<float>(B));
        }
    });
    return definition.capture(tile::tensor_shape("P", B, C), tile::tensor_shape("Y", B, C),
                              tile::tensor_shape("G", B, C));
}

// ---------------------------------------------------------------------------
// relu_backward : dZ[B,O] = dA[B,O] * min(relu(Z)/1e-8, 1)
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_relu_backward(int64_t B, int64_t O) {
    auto definition = tile::tile_kernel("mlp_relu_backward", [=](
                                            tile::TensorView<const float, 2> Z,
                                            tile::TensorView<const float, 2> dA,
                                            tile::TensorView<float, 2> dZ) {
        using namespace tile;
        auto b = axis("b", B), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            auto z = Z.tile(coord(0, 0), shape(b, o)).load();
            auto da = dA.tile(coord(0, 0), shape(b, o)).load();
            dZ.tile(coord(0, 0), shape(b, o)).store(da * min(max(z, 0.0f) / 1e-8f, 1.0f));
        }
    });
    return definition.capture(tile::tensor_shape("Z", B, O), tile::tensor_shape("dA", B, O),
                              tile::tensor_shape("dZ", B, O));
}

// ---------------------------------------------------------------------------
// grad : dW[K,O] = AT[K,B] @ dZ[B,O]
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_grad(int64_t B, int64_t K, int64_t O) {
    auto definition = tile::tile_kernel("mlp_grad", [=](
                                            tile::TensorView<const float, 2> AT,
                                            tile::TensorView<const float, 2> dZ,
                                            tile::TensorView<float, 2> dW) {
        using namespace tile;
        auto k = axis("k", K), b = axis("b", B), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            auto at = AT.tile(coord(0, 0), shape(k, b)).load();
            auto dz = dZ.tile(coord(0, 0), shape(b, o)).load();
            auto acc = mma(at, dz, zeros<float>(shape(k, o)));
            dW.tile(coord(0, 0), shape(k, o)).store(acc);
        }
    });
    return definition.capture(tile::tensor_shape("AT", K, B), tile::tensor_shape("dZ", B, O),
                              tile::tensor_shape("dW", K, O));
}

// ---------------------------------------------------------------------------
// grad_bias : db[1,O] = OnesT[1,B] @ dZ[B,O]   (sum over the minibatch)
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_grad_bias(int64_t B, int64_t O) {
    auto definition = tile::tile_kernel("mlp_grad_bias", [=](
                                            tile::TensorView<const float, 2> OnesT,
                                            tile::TensorView<const float, 2> dZ,
                                            tile::TensorView<float, 2> db) {
        using namespace tile;
        auto s = axis("s", 1), b = axis("b", B), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            auto ones_t = OnesT.tile(coord(0, 0), shape(s, b)).load();
            auto dz = dZ.tile(coord(0, 0), shape(b, o)).load();
            auto acc = mma(ones_t, dz, zeros<float>(shape(s, o)));
            db.tile(coord(0, 0), shape(s, o)).store(acc);
        }
    });
    return definition.capture(tile::tensor_shape("OnesT", 1, B), tile::tensor_shape("dZ", B, O),
                              tile::tensor_shape("db", 1, O));
}

// ---------------------------------------------------------------------------
// fc_backward : dA[B,K] = dZ[B,O] @ WT[O,K]   (WT = W^T; no bias row to drop)
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_fc_backward(int64_t B, int64_t K, int64_t O) {
    auto definition = tile::tile_kernel("mlp_fc_backward", [=](
                                            tile::TensorView<const float, 2> dZ,
                                            tile::TensorView<const float, 2> WT,
                                            tile::TensorView<float, 2> dA) {
        using namespace tile;
        auto b = axis("b", B), o = axis("o", O), k = axis("k", K);
        for (auto &nest : parallel(shape(1))) {
            auto dz = dZ.tile(coord(0, 0), shape(b, o)).load();
            auto wt = WT.tile(coord(0, 0), shape(o, k)).load();
            auto acc = mma(dz, wt, zeros<float>(shape(b, k)));
            dA.tile(coord(0, 0), shape(b, k)).store(acc);
        }
    });
    return definition.capture(tile::tensor_shape("dZ", B, O), tile::tensor_shape("WT", O, K),
                              tile::tensor_shape("dA", B, K));
}

// ---------------------------------------------------------------------------
// transpose : dst[N,M] = src[M,N]^T   (composes tile::reindex)
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_transpose(int64_t M, int64_t N) {
    auto definition = tile::tile_kernel("mlp_transpose", [=](
                                            tile::TensorView<const float, 2> src,
                                            tile::TensorView<float, 2> dst) {
        using namespace tile;
        auto m = axis("m", M), n = axis("n", N);
        for (auto &nest : parallel(shape(1))) {
            auto value = src.tile(coord(0, 0), shape(m, n)).load();
            auto transposed = reindex(value, shape(n, m), [&](const Nest &element) {
                return coord(element.index(m), element.index(n));
            });
            dst.tile(coord(0, 0), shape(n, m)).store(transposed);
        }
    });
    return definition.capture(tile::tensor_shape("src", M, N), tile::tensor_shape("dst", N, M));
}

// ---------------------------------------------------------------------------
// update : W[K,O] -= MLP_LR_EFF * dW[K,O]   (whole-tile in-place store)
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_update(int64_t K, int64_t O) {
    auto definition = tile::tile_kernel("mlp_update", [=](
                                            tile::TensorView<const float, 2> dW,
                                            tile::TensorView<float, 2> W) {
        using namespace tile;
        auto k = axis("k", K), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            auto w = W.tile(coord(0, 0), shape(k, o)).load();
            auto dw = dW.tile(coord(0, 0), shape(k, o)).load();
            W.tile(coord(0, 0), shape(k, o)).store(w - dw * MLP_LR_EFF);
        }
    });
    return definition.capture(tile::tensor_shape("dW", K, O), tile::tensor_shape("W", K, O));
}

// ---------------------------------------------------------------------------
// update_bias : Bias[1,O] -= MLP_LR_EFF * db[1,O]
// ---------------------------------------------------------------------------
[[nodiscard]] inline tile::Kernel make_update_bias(int64_t O) {
    auto definition = tile::tile_kernel("mlp_update_bias", [=](
                                            tile::TensorView<const float, 2> db,
                                            tile::TensorView<float, 2> Bias) {
        using namespace tile;
        auto s = axis("s", 1), o = axis("o", O);
        for (auto &nest : parallel(shape(1))) {
            auto bias = Bias.tile(coord(0, 0), shape(s, o)).load();
            auto grad = db.tile(coord(0, 0), shape(s, o)).load();
            Bias.tile(coord(0, 0), shape(s, o)).store(bias - grad * MLP_LR_EFF);
        }
    });
    return definition.capture(tile::tensor_shape("db", 1, O), tile::tensor_shape("Bias", 1, O));
}

}// namespace mlp
