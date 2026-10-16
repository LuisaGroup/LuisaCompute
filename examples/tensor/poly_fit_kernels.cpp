// =============================================================================
// poly_fit_kernels.cpp — New Tile DSL kernels for the polynomial fit
// =============================================================================
// Implementation of the non-template kernels declared in poly_fit_kernels.h.
// Each kernel is captured with tile_kernel(...).capture(...) and compiled by
// the driver through tensor_example::compile_tile.
// =============================================================================

#include "poly_fit_kernels.h"

namespace tensor_example::polyfit {

// ---------------------------------------------------------------------------
// error : err[N,1] = Y[N,1] - y[N,1]   (MSE residual)
// ---------------------------------------------------------------------------
[[nodiscard]] tile::Kernel make_poly_error_kernel() {
    auto definition = tile::tile_kernel("poly_error", [=](
                                            tile::TensorView<const float, 2> Y,
                                            tile::TensorView<const float, 2> y,
                                            tile::TensorView<float, 2> err) {
        auto n = tile::axis("n", N_TRAIN), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto yy = Y.tile(tile::coord(0, 0), tile::shape(n, one)).load();
            auto yt = y.tile(tile::coord(0, 0), tile::shape(n, one)).load();
            err.tile(tile::coord(0, 0), tile::shape(n, one)).store(yy - yt);
        }
    });
    return definition.capture(tile::tensor_shape(N_TRAIN, 1), tile::tensor_shape(N_TRAIN, 1),
                              tile::tensor_shape(N_TRAIN, 1));
}

// ---------------------------------------------------------------------------
// gradient : G[F,1] = XT[F,N] @ err[N,1]
// (d sum(err^2)/dW without the factor 2 — folded into the update step)
// ---------------------------------------------------------------------------
[[nodiscard]] tile::Kernel make_poly_grad_kernel() {
    auto definition = tile::tile_kernel("poly_grad", [=](
                                            tile::TensorView<const float, 2> XT,
                                            tile::TensorView<const float, 2> err,
                                            tile::TensorView<float, 2> G) {
        auto f = tile::axis("f", F), n = tile::axis("n", N_TRAIN), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto xt = XT.tile(tile::coord(0, 0), tile::shape(f, n)).load();
            auto e = err.tile(tile::coord(0, 0), tile::shape(n, one)).load();
            auto acc = tile::zeros<float>(tile::shape(f, one));
            G.tile(tile::coord(0, 0), tile::shape(f, one)).store(tile::mma(xt, e, acc));
        }
    });
    return definition.capture(tile::tensor_shape(F, N_TRAIN), tile::tensor_shape(N_TRAIN, 1),
                              tile::tensor_shape(F, 1));
}

// ---------------------------------------------------------------------------
// update : W' = W - (2*lr) * G   (the manual `param -= lr * param.grad` step,
// with the MSE factor 2 folded into the effective learning rate)
// ---------------------------------------------------------------------------
[[nodiscard]] tile::Kernel make_poly_update_kernel() {
    auto definition = tile::tile_kernel("poly_update", [=](
                                            tile::TensorView<const float, 2> W,
                                            tile::TensorView<const float, 2> G,
                                            tile::TensorView<float, 2> W_new) {
        auto f = tile::axis("f", F), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto w = W.tile(tile::coord(0, 0), tile::shape(f, one)).load();
            auto g = G.tile(tile::coord(0, 0), tile::shape(f, one)).load();
            W_new.tile(tile::coord(0, 0), tile::shape(f, one)).store(w - g * (2.0f * LR));
        }
    });
    return definition.capture(tile::tensor_shape(F, 1), tile::tensor_shape(F, 1), tile::tensor_shape(F, 1));
}

}// namespace tensor_example::polyfit
