// =============================================================================
// tensor_basics_kernels.h — New Tile DSL kernels for the basics tour
// =============================================================================
// Header-only kernels for the `--basics` driver (C++ twin of
// tensor_basics.py): elementwise arithmetic, an autograd-style quadratic and a
// 1 -> 1 neural network trained with SGD (the relu is identity on the positive
// training inputs, so the tiny net reduces to linear regression).
//
// Ports backup_old_tile/examples/tensor/tensor_basics_kernels.h to the
// execution-structure-first C++ Tile DSL (<luisa/tile/dsl.h>). Every kernel is
// a single root parallel nest over shape(1) with whole-problem tiles, matching
// the old single-block (gx = 1) tile programs. The old scalar-division idioms
// (no binary minus / no tile*scalar in the removed dialect) collapse into the
// ordinary `-` and `*` operators the new TileIR supports.
// =============================================================================

#pragma once

#include <cstdint>

#include <luisa/tile/dsl.h>

namespace tensor_example {

namespace tile = luisa::compute::tile;

namespace basics {

// ---------------------------------------------------------------------------
// basic_addmul : C = A + B;  D = A * B   (rank-1, N elements)
// ---------------------------------------------------------------------------
template<int64_t N>
[[nodiscard]] inline tile::Kernel make_basic_addmul_kernel() {
    auto definition = tile::tile_kernel("basic_addmul", [=](
                                            tile::TensorView<const float, 1> A,
                                            tile::TensorView<const float, 1> B,
                                            tile::TensorView<float, 1> C,
                                            tile::TensorView<float, 1> D) {
        auto n = tile::axis("n", N);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto a = A.tile(tile::coord(0), tile::shape(n)).load();
            auto b = B.tile(tile::coord(0), tile::shape(n)).load();
            C.tile(tile::coord(0), tile::shape(n)).store(a + b);
            D.tile(tile::coord(0), tile::shape(n)).store(a * b);
        }
    });
    return definition.capture(tile::tensor_shape(N), tile::tensor_shape(N),
                              tile::tensor_shape(N), tile::tensor_shape(N));
}

// ---------------------------------------------------------------------------
// basic_square_grad : Y = x^2 + 2x + 1;  DY = 2x + 2   (autograd demo)
// ---------------------------------------------------------------------------
template<int64_t N>
[[nodiscard]] inline tile::Kernel make_basic_square_grad_kernel() {
    auto definition = tile::tile_kernel("basic_square_grad", [=](
                                            tile::TensorView<const float, 1> X,
                                            tile::TensorView<float, 1> Y,
                                            tile::TensorView<float, 1> DY) {
        auto n = tile::axis("n", N);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto x = X.tile(tile::coord(0), tile::shape(n)).load();
            Y.tile(tile::coord(0), tile::shape(n)).store(x * x + x * 2.0f + 1.0f);
            DY.tile(tile::coord(0), tile::shape(n)).store(x * 2.0f + 2.0f);
        }
    });
    return definition.capture(tile::tensor_shape(N), tile::tensor_shape(N), tile::tensor_shape(N));
}

// ---------------------------------------------------------------------------
// nn_forward : Y[B,1] = Xb[B,2] @ W[2,1]   (tiny 1 -> 1 net, bias folded in)
// ---------------------------------------------------------------------------
template<int64_t B>
[[nodiscard]] inline tile::Kernel make_nn_forward_kernel() {
    auto definition = tile::tile_kernel("nn_forward", [=](
                                            tile::TensorView<const float, 2> Xb,
                                            tile::TensorView<const float, 2> W,
                                            tile::TensorView<float, 2> Y) {
        auto b = tile::axis("b", B), two = tile::axis("two", 2), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto xb = Xb.tile(tile::coord(0, 0), tile::shape(b, two)).load();
            auto w = W.tile(tile::coord(0, 0), tile::shape(two, one)).load();
            auto acc = tile::zeros<float>(tile::shape(b, one));
            Y.tile(tile::coord(0, 0), tile::shape(b, one)).store(tile::mma(xb, w, acc));
        }
    });
    return definition.capture(tile::tensor_shape(B, 2), tile::tensor_shape(2, 1), tile::tensor_shape(B, 1));
}

// ---------------------------------------------------------------------------
// nn_error : err[B,1] = Y[B,1] - targets[B,1]   (MSE residual)
// ---------------------------------------------------------------------------
template<int64_t B>
[[nodiscard]] inline tile::Kernel make_nn_error_kernel() {
    auto definition = tile::tile_kernel("nn_error", [=](
                                            tile::TensorView<const float, 2> Y,
                                            tile::TensorView<const float, 2> targets,
                                            tile::TensorView<float, 2> err) {
        auto b = tile::axis("b", B), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto y = Y.tile(tile::coord(0, 0), tile::shape(b, one)).load();
            auto t = targets.tile(tile::coord(0, 0), tile::shape(b, one)).load();
            err.tile(tile::coord(0, 0), tile::shape(b, one)).store(y - t);
        }
    });
    return definition.capture(tile::tensor_shape(B, 1), tile::tensor_shape(B, 1), tile::tensor_shape(B, 1));
}

// ---------------------------------------------------------------------------
// nn_grad : G[2,1] = XT[2,B] @ err[B,1]
// ---------------------------------------------------------------------------
template<int64_t B>
[[nodiscard]] inline tile::Kernel make_nn_grad_kernel() {
    auto definition = tile::tile_kernel("nn_grad", [=](
                                            tile::TensorView<const float, 2> XT,
                                            tile::TensorView<const float, 2> err,
                                            tile::TensorView<float, 2> G) {
        auto two = tile::axis("two", 2), b = tile::axis("b", B), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto xt = XT.tile(tile::coord(0, 0), tile::shape(two, b)).load();
            auto e = err.tile(tile::coord(0, 0), tile::shape(b, one)).load();
            auto acc = tile::zeros<float>(tile::shape(two, one));
            G.tile(tile::coord(0, 0), tile::shape(two, one)).store(tile::mma(xt, e, acc));
        }
    });
    return definition.capture(tile::tensor_shape(2, B), tile::tensor_shape(B, 1), tile::tensor_shape(2, 1));
}

// ---------------------------------------------------------------------------
// nn_update : W'[D,1] = W[D,1] - (0.05/4) * G[D,1]
// ---------------------------------------------------------------------------
template<int64_t D>
[[nodiscard]] inline tile::Kernel make_nn_update_kernel() {
    constexpr float LR_EFF = 0.05f / 4.0f;
    auto definition = tile::tile_kernel("nn_update", [=](
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

}// namespace basics
}// namespace tensor_example
