// =============================================================================
// cnn_kernels.cpp — New Tile DSL kernels for TinyCNN inference
// =============================================================================
// Implementation of the five kernels declared in cnn_kernels.h. Each layer is
// a single tile::mma over host-built im2col/weight matrices (bias folded in as
// an extra im2col row), plus an elementwise ReLU or a row-wise softmax.
//
// Semantics vs. the old tile-language kernels (backup_old_tile
// cnn_kernels.cpp):
//   * T.gemm(C += A @ B) on a cleared accumulator == tile::mma(A, B, zeros).
//   * The old softmax computed exp(x)/sum via rsqrt(row_sum)^2; the new
//     TileIR has no rsqrt opcode, so this port divides by the broadcast row
//     sum directly (1/sum via rsqrt^2 and a true division agree to ~1 ulp,
//     far inside the driver's 1e-3 tolerance).
// =============================================================================

#include "cnn_kernels.h"

namespace tensor_example::cnn {

// ---------------------------------------------------------------------------
// conv1 + ReLU : Y[C1, B*36] = relu( W1'[C1,10] @ col1'[10, B*36] )
// ---------------------------------------------------------------------------
tile::Kernel make_conv1_relu_kernel() {
    constexpr int64_t Co = CC1;
    constexpr int64_t KK = 10;// 1*3*3 + bias row = 9 + 1
    constexpr int64_t P = CB * 36;
    auto definition = tile::tile_kernel("cnn_conv1_relu", [=](
                                            tile::TensorView<const float, 2> W1,
                                            tile::TensorView<const float, 2> col1,
                                            tile::TensorView<float, 2> Y) {
        using namespace tile;
        auto co = axis("co", Co), kk = axis("kk", KK), p = axis("p", P);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto w = W1.tile(coord(0, 0), shape(co, kk)).load();
            auto x = col1.tile(coord(0, 0), shape(kk, p)).load();
            auto acc = mma(w, x, zeros<float>(shape(co, p)));
            Y.tile(coord(0, 0), shape(co, p)).store(max(acc, 0.0f));// ReLU
        }
    });
    return definition.capture(tile::tensor_shape("W1", Co, KK),
                              tile::tensor_shape("col1", KK, P),
                              tile::tensor_shape("Y", Co, P));
}

// ---------------------------------------------------------------------------
// conv2 + ReLU : Y[C2, B*16] = relu( W2'[C2,37] @ col2'[37, B*16] )
// ---------------------------------------------------------------------------
tile::Kernel make_conv2_relu_kernel() {
    constexpr int64_t Co = CC2;
    constexpr int64_t KK = CC1 * 9 + 1;// 4*9 + bias row = 37
    constexpr int64_t P = CB * 16;
    auto definition = tile::tile_kernel("cnn_conv2_relu", [=](
                                            tile::TensorView<const float, 2> W2,
                                            tile::TensorView<const float, 2> col2,
                                            tile::TensorView<float, 2> Y) {
        using namespace tile;
        auto co = axis("co", Co), kk = axis("kk", KK), p = axis("p", P);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto w = W2.tile(coord(0, 0), shape(co, kk)).load();
            auto x = col2.tile(coord(0, 0), shape(kk, p)).load();
            auto acc = mma(w, x, zeros<float>(shape(co, p)));
            Y.tile(coord(0, 0), shape(co, p)).store(max(acc, 0.0f));// ReLU
        }
    });
    return definition.capture(tile::tensor_shape("W2", Co, KK),
                              tile::tensor_shape("col2", KK, P),
                              tile::tensor_shape("Y", Co, P));
}

// ---------------------------------------------------------------------------
// fc1 + ReLU : Y[B, F1] = relu( col_fc1[B, 129] @ Wfc1T[129, F1] )
// ---------------------------------------------------------------------------
tile::Kernel make_fc1_relu_kernel() {
    constexpr int64_t M = CB;
    constexpr int64_t K = CC2 * 16 + 1;// 8*16 + bias row = 129
    constexpr int64_t N = CF1;
    auto definition = tile::tile_kernel("cnn_fc1_relu", [=](
                                            tile::TensorView<const float, 2> col_fc1,
                                            tile::TensorView<const float, 2> Wfc1T,
                                            tile::TensorView<float, 2> Y) {
        using namespace tile;
        auto m = axis("m", M), k = axis("k", K), n = axis("n", N);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto a = col_fc1.tile(coord(0, 0), shape(m, k)).load();
            auto w = Wfc1T.tile(coord(0, 0), shape(k, n)).load();
            auto acc = mma(a, w, zeros<float>(shape(m, n)));
            Y.tile(coord(0, 0), shape(m, n)).store(max(acc, 0.0f));// ReLU
        }
    });
    return definition.capture(tile::tensor_shape("col_fc1", M, K),
                              tile::tensor_shape("Wfc1T", K, N),
                              tile::tensor_shape("Y", M, N));
}

// ---------------------------------------------------------------------------
// fc2 (logits) : Y[B, NC] = col_fc2[B, 33] @ Wfc2T[33, NC]
// ---------------------------------------------------------------------------
tile::Kernel make_fc2_kernel() {
    constexpr int64_t M = CB;
    constexpr int64_t K = CF1 + 1;// 32 + bias row = 33
    constexpr int64_t N = CNC;
    auto definition = tile::tile_kernel("cnn_fc2", [=](
                                            tile::TensorView<const float, 2> col_fc2,
                                            tile::TensorView<const float, 2> Wfc2T,
                                            tile::TensorView<float, 2> Y) {
        using namespace tile;
        auto m = axis("m", M), k = axis("k", K), n = axis("n", N);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto a = col_fc2.tile(coord(0, 0), shape(m, k)).load();
            auto w = Wfc2T.tile(coord(0, 0), shape(k, n)).load();
            auto acc = mma(a, w, zeros<float>(shape(m, n)));
            Y.tile(coord(0, 0), shape(m, n)).store(acc);
        }
    });
    return definition.capture(tile::tensor_shape("col_fc2", M, K),
                              tile::tensor_shape("Wfc2T", K, N),
                              tile::tensor_shape("Y", M, N));
}

// ---------------------------------------------------------------------------
// row-wise softmax : P[B, NC] = softmax(logits[B, NC])
// ---------------------------------------------------------------------------
tile::Kernel make_cnn_softmax_kernel() {
    constexpr int64_t M = CB;
    constexpr int64_t N = CNC;
    auto definition = tile::tile_kernel("cnn_softmax", [=](
                                            tile::TensorView<const float, 2> logits,
                                            tile::TensorView<float, 2> probs) {
        using namespace tile;
        auto m = axis("m", M), n = axis("n", N);
        for (auto &nest : parallel(shape(1))) {
            static_cast<void>(nest);
            auto l = logits.tile(coord(0, 0), shape(m, n)).load();
            auto e = exp(l);
            auto denom = reduce(e, n, add);
            probs.tile(coord(0, 0), shape(m, n)).store(e / denom);
        }
    });
    return definition.capture(tile::tensor_shape("logits", M, N),
                              tile::tensor_shape("probs", M, N));
}

}// namespace tensor_example::cnn
