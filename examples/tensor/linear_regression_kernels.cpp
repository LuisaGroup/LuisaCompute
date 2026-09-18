// =============================================================================
// linear_regression_kernels.cpp — New Tile DSL kernels for linear & logistic
// regression training
// =============================================================================
// Implementation of the non-template kernels declared in
// linear_regression_kernels.h.  Each kernel is captured with
// tile_kernel(...).capture(...) and compiled by the driver through
// tensor_example::compile_tile.
// =============================================================================

#include "linear_regression_kernels.h"

namespace tensor_example::lreg {

// ---------------------------------------------------------------------------
// error : err[N,1] = Y[N,1] - y[N,1]   (MSE residual)
// ---------------------------------------------------------------------------
[[nodiscard]] tile::Kernel make_linear_error_kernel() {
    auto definition = tile::tile_kernel("lreg_linear_error", [=](
                                            tile::TensorView<const float, 2> Y,
                                            tile::TensorView<const float, 2> y,
                                            tile::TensorView<float, 2> err) {
        auto n = tile::axis("n", LIN_N), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto yy = Y.tile(tile::coord(0, 0), tile::shape(n, one)).load();
            auto yt = y.tile(tile::coord(0, 0), tile::shape(n, one)).load();
            err.tile(tile::coord(0, 0), tile::shape(n, one)).store(yy - yt);
        }
    });
    return definition.capture(tile::tensor_shape(LIN_N, 1), tile::tensor_shape(LIN_N, 1),
                              tile::tensor_shape(LIN_N, 1));
}

// ---------------------------------------------------------------------------
// residual : res[N,1] = sigmoid(Z[N,1]) - y[N,1]   (BCE gradient)
// sigmoid(z) = 1/(1+exp(-z)); the old kernel spelled it rsqrt(exp(-z)+1)^2
// because the removed dialect had no tile subtraction — same function.
// ---------------------------------------------------------------------------
[[nodiscard]] tile::Kernel make_logistic_residual_kernel() {
    auto definition = tile::tile_kernel("lreg_logistic_residual", [=](
                                            tile::TensorView<const float, 2> Z,
                                            tile::TensorView<const float, 2> y,
                                            tile::TensorView<float, 2> res) {
        auto n = tile::axis("n", LOG_N), one = tile::axis("one", 1);
        for ([[maybe_unused]] auto &nest : tile::parallel(tile::shape(1))) {
            auto z = Z.tile(tile::coord(0, 0), tile::shape(n, one)).load();
            auto yy = y.tile(tile::coord(0, 0), tile::shape(n, one)).load();
            auto sigmoid = 1.0f / (1.0f + tile::exp(-z));
            res.tile(tile::coord(0, 0), tile::shape(n, one)).store(sigmoid - yy);
        }
    });
    return definition.capture(tile::tensor_shape(LOG_N, 1), tile::tensor_shape(LOG_N, 1),
                              tile::tensor_shape(LOG_N, 1));
}

}// namespace tensor_example::lreg
