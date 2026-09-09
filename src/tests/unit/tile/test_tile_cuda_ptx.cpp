// CUDA native-Tile end-to-end tests: LuisaCompute tile kernels lowered through
// the shared TIRx bridge to a CUDA device artifact, compiled to PTX by the
// standalone NVRTC pipeline, and launched with a direct static cuLaunchKernel.
//
// When built without the TIRx bridge (LUISA_TEST_TILE_CUDA_TIRX undefined),
// the same executable verifies the backend fails closed with a clear error
// instead of silently falling back.

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/tile/runtime.h>
#include <luisa/runtime/stream.h>

#ifdef LUISA_TEST_TILE_CUDA_TIRX
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

[[nodiscard]] bool close_tolerance(float actual, double expected) noexcept {
    return std::isfinite(actual) &&
           std::abs(static_cast<double>(actual) - expected) <=
               2e-4 + 2e-4 * std::abs(expected);
}

void test_elementwise(Device &device, tile::CompileOptions options) {
    constexpr auto n = 1003;
    auto definition = tile::tile_kernel("cuda_tile_axpy", [](tile::TensorView<const float, 1> x,
                                                             tile::TensorView<float, 1> result) {
        auto element = tile::axis("element", n);
        for (auto &item : tile::parallel(tile::shape(element))) {
            auto index = item.index();
            result(index).store(1.25f * x(index).load() + 0.5f);
        }
    });
    auto kernel = definition.capture(tile::tensor_shape("x", n), tile::tensor_shape("result", n));
    luisa::vector<float> x_values(n);
    for (auto i = 0u; i < n; i++) { x_values[i] = static_cast<float>(i % 37u) * 0.125f - 2.0f; }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto x = device.create_buffer<float>(n);
    auto r = device.create_buffer<float>(n);
    luisa::vector<float> actual(n);
    stream << x.copy_from(x_values.data())
           << shader(x, r).dispatch() << r.copy_to(actual.data()) << synchronize();
    for (auto i = 0u; i < n; i++) {
        expect(close_tolerance(actual[i], 1.25 * x_values[i] + 0.5));
    }
}

void test_row_sum(Device &device, tile::CompileOptions options) {
    constexpr auto rows = 37;
    constexpr auto columns = 19;
    auto definition = tile::tile_kernel("cuda_tile_row_sum", [](tile::TensorView<const float, 2> x,
                                                                tile::TensorView<float, 1> result) {
        auto row = tile::axis("row", rows);
        auto column = tile::axis("column", columns);
        for (auto &row_nest : tile::parallel(tile::shape(row))) {
            auto sum = tile::Scalar<float>{0.0f};
            for (auto &item : row_nest.reduce(tile::shape(column))) {
                sum += x(row_nest.index(row), item.index(column)).load();
            }
            result(row_nest.index(row)).store(sum);
        }
    });
    auto kernel = definition.capture(
        tile::tensor_shape("x", rows, columns), tile::tensor_shape("result", rows));
    luisa::vector<float> values(rows * columns);
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int>(i % 23u) - 11) * 0.0625f;
    }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    expect(shader.metadata().realization.find("TIRx -> ") != luisa::string::npos);
    expect(!shader.metadata().source.empty());
    expect(shader.metadata().dispatch_size.x * shader.metadata().dispatch_size.y *
               shader.metadata().dispatch_size.z >
           0u);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto x = device.create_buffer<float>(values.size());
    auto r = device.create_buffer<float>(rows);
    luisa::vector<float> actual(rows);
    stream << x.copy_from(values.data())
           << shader(x, r).dispatch() << r.copy_to(actual.data()) << synchronize();
    for (auto row = 0; row < rows; row++) {
        auto expected = 0.0;
        for (auto column = 0; column < columns; column++) {
            expected += values[static_cast<size_t>(row * columns + column)];
        }
        expect(close_tolerance(actual[row], expected));
    }
}

void test_row_extrema(Device &device, tile::CompileOptions options) {
    constexpr auto rows = 17;
    constexpr auto columns = 31;
    auto definition = tile::tile_kernel("cuda_tile_extrema", [](tile::TensorView<const float, 2> input,
                                                                tile::TensorView<float, 1> minima,
                                                                tile::TensorView<float, 1> maxima) {
        auto row = tile::axis("row", rows);
        auto one = tile::axis("one", 1);
        auto column = tile::axis("column", columns);
        for (auto &nest : tile::parallel(tile::shape(row))) {
            auto value = input[coord(nest.index(), 0), tile::shape(one, column)];
            minima(coord(nest.index()), tile::shape(one))
                .store(tile::reduce(value, column, tile::minimum));
            maxima(coord(nest.index()), tile::shape(one))
                .store(tile::reduce(value, column, tile::maximum));
        }
    });
    auto kernel = definition.capture(tile::tensor_shape(rows, columns),
                                     tile::tensor_shape(rows),
                                     tile::tensor_shape(rows));
    luisa::vector<float> values(static_cast<size_t>(rows * columns));
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int>(i % 47u) - 23) * 0.125f;
    }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto x = device.create_buffer<float>(values.size());
    auto lo = device.create_buffer<float>(rows);
    auto hi = device.create_buffer<float>(rows);
    luisa::vector<float> actual_lo(rows), actual_hi(rows);
    stream << x.copy_from(values.data())
           << shader(x, lo, hi).dispatch()
           << lo.copy_to(actual_lo.data()) << hi.copy_to(actual_hi.data()) << synchronize();
    for (auto row = 0; row < rows; row++) {
        auto expected_lo = std::numeric_limits<double>::infinity();
        auto expected_hi = -std::numeric_limits<double>::infinity();
        for (auto column = 0; column < columns; column++) {
            auto v = static_cast<double>(values[static_cast<size_t>(row * columns + column)]);
            expected_lo = std::min(expected_lo, v);
            expected_hi = std::max(expected_hi, v);
        }
        expect(close_tolerance(actual_lo[row], expected_lo));
        expect(close_tolerance(actual_hi[row], expected_hi));
    }
}

void test_row_softmax(Device &device, tile::CompileOptions options) {
    constexpr auto rows = 9;
    constexpr auto columns = 13;
    auto definition = tile::tile_kernel("cuda_tile_row_softmax", [](tile::TensorView<const float, 2> input,
                                                                    tile::TensorView<float, 2> output) {
        auto row = tile::axis("row", rows);
        auto one = tile::axis("one", 1);
        auto column = tile::axis("column", columns);
        for (auto &nest : tile::parallel(tile::shape(row))) {
            auto origin = coord(nest.index(), 0);
            auto x = input.tile(origin, tile::shape(one, column)).load();
            auto e = tile::exp(x - tile::reduce(x, column, tile::maximum));
            output(origin, tile::shape(one, column)).store(e / tile::reduce(e, column, tile::add));
        }
    });
    auto kernel = definition.capture(tile::tensor_shape(rows, columns),
                                     tile::tensor_shape(rows, columns));
    luisa::vector<float> values(static_cast<size_t>(rows * columns));
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int>(i % 31u) - 15) * 0.5f;
    }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto x = device.create_buffer<float>(values.size());
    auto out = device.create_buffer<float>(values.size());
    luisa::vector<float> actual(values.size());
    stream << x.copy_from(values.data())
           << shader(x, out).dispatch() << out.copy_to(actual.data()) << synchronize();
    for (auto row = 0; row < rows; row++) {
        double sum = 0.0;
        for (auto column = 0; column < columns; column++) {
            sum += std::exp(static_cast<double>(values[static_cast<size_t>(row * columns + column)]));
        }
        for (auto column = 0; column < columns; column++) {
            auto expected = std::exp(static_cast<double>(values[static_cast<size_t>(row * columns + column)])) / sum;
            expect(close_tolerance(actual[static_cast<size_t>(row * columns + column)], expected));
        }
    }
}

struct GemmConfig {
    int64_t m{32}, n{32}, k{16}, tile{16};
    bool transpose_a{false}, transpose_b{false};
    bool allow_reassociation{true};
};

[[nodiscard]] tile::Kernel gemm_kernel(GemmConfig cfg) {
    using namespace tile;
    auto definition = tile_kernel("cuda_tile_gemm", [=](TensorView<const float, 2> A,
                                                        TensorView<const float, 2> B,
                                                        TensorView<float, 2> C) {
        auto gm = axis("groups_m", ceil_div(cfg.m, cfg.tile));
        auto gn = axis("groups_n", ceil_div(cfg.n, cfg.tile));
        auto row = axis("row", cfg.tile);
        auto col = axis("column", cfg.tile);
        auto k_axis = axis("k", cfg.k);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto m0 = nest.index(gm) * cfg.tile;
            auto n0 = nest.index(gn) * cfg.tile;
            auto a = cfg.transpose_a ? A.tile(coord(0, m0), shape(k_axis, row)).load() :
                                       A.tile(coord(m0, 0), shape(row, k_axis)).load();
            auto b = cfg.transpose_b ? B.tile(coord(n0, 0), shape(col, k_axis)).load() :
                                       B.tile(coord(0, n0), shape(k_axis, col)).load();
            auto acc = zeros<float>(shape(row, col));
            acc = mma(a, b, acc, MmaPolicy{cfg.allow_reassociation});
            C(coord(m0, n0), shape(row, col)).store(acc);
        }
    });
    return definition.capture(
        tensor_shape(cfg.transpose_a ? cfg.k : cfg.m, cfg.transpose_a ? cfg.m : cfg.k),
        tensor_shape(cfg.transpose_b ? cfg.n : cfg.k, cfg.transpose_b ? cfg.k : cfg.n),
        tensor_shape(cfg.m, cfg.n));
}

void test_gemm(Device &device, tile::CompileOptions options, GemmConfig cfg) {
    auto kernel = gemm_kernel(cfg);
    auto a_elements = cfg.m * cfg.k;
    auto b_elements = cfg.k * cfg.n;
    auto c_elements = cfg.m * cfg.n;
    luisa::vector<float> a(static_cast<size_t>(a_elements));
    luisa::vector<float> b(static_cast<size_t>(b_elements));
    luisa::vector<float> actual(static_cast<size_t>(c_elements));
    for (auto i = 0u; i < a.size(); i++) { a[i] = std::sin(static_cast<float>(i) * .371f + .13f) * 1.375f; }
    for (auto i = 0u; i < b.size(); i++) { b[i] = std::cos(static_cast<float>(i) * .213f + .47f) * .875f; }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto ab = device.create_buffer<float>(a.size());
    auto bb = device.create_buffer<float>(b.size());
    auto cb = device.create_buffer<float>(static_cast<size_t>(c_elements));
    stream << ab.copy_from(a.data()) << bb.copy_from(b.data())
           << shader(ab, bb, cb).dispatch() << cb.copy_to(actual.data()) << synchronize();
    for (auto m = 0; m < cfg.m; m++) {
        for (auto n = 0; n < cfg.n; n++) {
            double expected = 0.0;
            for (auto k = 0; k < cfg.k; k++) {
                auto ai = cfg.transpose_a ? k * cfg.m + m : m * cfg.k + k;
                auto bi = cfg.transpose_b ? n * cfg.k + k : k * cfg.n + n;
                expected += static_cast<double>(a[ai]) * b[bi];
            }
            expect(close_tolerance(actual[m * cfg.n + n], expected));
        }
    }
}

void run_fail_closed_without_tirx(Device &device) {
    auto definition = tile::tile_kernel("cuda_tile_no_bridge", [](tile::TensorView<float, 1> output) {
        auto element = tile::axis("element", 32);
        for (auto &item : tile::parallel(tile::shape(element))) {
            output(item.index()).store(1.0f);
        }
    });
    auto kernel = definition.capture(tile::tensor_shape(32));
    tile::CompileOptions options;
    options.lowering = tile::Lowering::TIRX;
    auto shader = tile::compile(device, kernel, options);
    expect(!shader);
    expect(shader.metadata().error.find("TIRx") != luisa::string::npos) << shader.metadata().error;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
#ifdef LUISA_TEST_TILE_CUDA_TIRX
    tile::CompileOptions options;
    options.lowering = tile::Lowering::TIRX;
    "tile_cuda_ptx_elementwise"_test = [&] { test_elementwise(device, options); };
    "tile_cuda_ptx_reduce"_test = [&] { test_row_sum(device, options); };
    "tile_cuda_ptx_extrema"_test = [&] { test_row_extrema(device, options); };
    "tile_cuda_ptx_softmax"_test = [&] { test_row_softmax(device, options); };
    "tile_cuda_ptx_gemm"_test = [&] {
        for (auto ta : {false, true}) {
            for (auto tb : {false, true}) {
                for (auto reassociation : {true, false}) {
                    test_gemm(device, options, {32, 32, 16, 16, ta, tb, reassociation});
                }
            }
        }
        test_gemm(device, options, {48, 32, 24, 16, false, true, true});
        test_gemm(device, options, {32, 64, 32, 16, true, false, false});
    };
#else
    "tile_cuda_ptx_fail_closed_without_bridge"_test = [&] { run_fail_closed_without_tirx(device); };
#endif
}
