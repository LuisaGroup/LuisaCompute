#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_xir_test_utils.h"
#include "tile_reduction_policy_test_utils.h"
#include <bit>
#include <luisa/runtime/stream.h>
#include <luisa/tile/runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef LUISA_TEST_TILE_XIR_TIRX
#include "tile_tirx_test_utils.h"
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

void reduction_fold_policies(Device &device) {
    namespace cases = test::tile_reduction;
    constexpr auto rows = int64_t{3};
    for (auto dimensions : {std::pair{0, 3}, std::pair{2, 0}, std::pair{1, 1},
                            std::pair{1, 3}, std::pair{2, 3}, std::pair{3, 5}}) {
        auto [outer, inner] = dimensions;
        auto width = outer * inner;
        auto stride = std::max(width, 1);
        for (auto seed : {0.0f, -0.0f, 3.0f}) {
            auto kernel = cases::folds(rows, outer, inner, seed);
            auto shader = tile::compile(device, kernel, {}, {.enable_fast_math = true});
            expect(static_cast<bool>(shader)) << shader.metadata().error;
            if (!shader) { continue; }
            expect(shader.metadata().realization.find("fast_math=false; ordered_reduction=true") != string::npos);
            vector<float> values(rows * stride), actual(rows * cases::outputs);
            for (auto r = int64_t{0}; r < rows; r++) {
                for (auto i = 0; i < width; i++) {
                    constexpr float cancellation[]{16777216.0f, 1.0f, -16777216.0f, 3.0f, -2.0f};
                    values[r * stride + i] = r == 0 ? cancellation[i % 5] : static_cast<float>((i + 1) * (r + 1));
                }
            }
            auto input = device.create_buffer<float>(values.size());
            auto output = device.create_buffer<float>(actual.size());
            auto stream = device.create_stream(StreamTag::COMPUTE);
            stream << input.copy_from(values.data()) << shader(input, output).dispatch()
                   << output.copy_to(actual.data()) << synchronize();
            for (auto r = int64_t{0}; r < rows; r++) {
                auto expected = cases::reference(span<const float>{values}.subspan(r * stride, width), seed);
                for (auto mode = int64_t{0}; mode < cases::outputs; mode++) {
                    expect(eq(std::bit_cast<uint32_t>(actual[r * cases::outputs + mode]), std::bit_cast<uint32_t>(expected[mode])))
                        << "shape=" << outer << "," << inner << " row=" << r << " mode=" << mode << " seed=" << seed;
                }
            }
        }
    }
}

[[nodiscard]] bool close(span<const float> actual, span<const double> expected) {
    if (actual.size() != expected.size()) { return false; }
    for (size_t i = 0u; i < actual.size(); i++) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-5 + 2e-5 * std::abs(expected[i])) { return false; }
    }
    return true;
}

void gemm(Device &device, test::tile_xir::Gemm cfg, bool compare_tirx) {
    auto kernel = test::tile_xir::gemm(cfg);
    expect(kernel.valid());
    auto shader = tile::compile(device, kernel);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    expect(shader.metadata().realization.find("XIR SSA") != string::npos);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    constexpr auto pad = size_t{19u};
    constexpr auto guard = -731.25f;
    vector<float> a(cfg.m * cfg.k), b(cfg.k * cfg.n), c(cfg.m * cfg.n + 2u * pad, guard);
    vector<double> expected(cfg.m * cfg.n);
    auto ab = device.create_buffer<float>(a.size() + pad);
    auto bb = device.create_buffer<float>(b.size() + pad + 1u);
    auto cb = device.create_buffer<float>(c.size());
    auto av = ab.view(pad, a.size()), bv = bb.view(pad + 1u, b.size()), cv = cb.view(pad, expected.size());
    for (auto repeat = 0; repeat < 2; repeat++) {
        for (size_t i = 0u; i < a.size(); i++) { a[i] = std::sin(static_cast<float>(i) * .371f + .13f + repeat); }
        for (size_t i = 0u; i < b.size(); i++) { b[i] = std::cos(static_cast<float>(i) * .213f + .47f - repeat); }
        for (int64_t m = 0; m < cfg.m; m++) {
            for (int64_t n = 0; n < cfg.n; n++) {
                auto sum = static_cast<double>(cfg.initial);
                for (int64_t k = 0; k < cfg.k; k++) {
                    sum += static_cast<double>(a[cfg.transpose_a ? k * cfg.m + m : m * cfg.k + k]) * b[cfg.transpose_b ? n * cfg.k + k : k * cfg.n + n];
                }
                expected[m * cfg.n + n] = sum;
            }
        }
        std::fill(c.begin() + pad, c.end() - pad, std::numeric_limits<float>::quiet_NaN());
        stream << av.copy_from(a.data()) << bv.copy_from(b.data()) << cb.copy_from(c.data())
               << shader(av, bv, cv).dispatch() << cb.copy_to(c.data()) << synchronize();
        expect(close(span{c}.subspan(pad, expected.size()), expected));
        expect(std::all_of(c.begin(), c.begin() + pad, [](float x) { return x == guard; }));
        expect(std::all_of(c.end() - pad, c.end(), [](float x) { return x == guard; }));
    }
    auto moved = std::move(shader);
    expect(!shader && static_cast<bool>(moved));
    shader = std::move(moved);
    stream << shader(av, bv, cv).dispatch() << cb.copy_to(c.data()) << synchronize();
    expect(close(span{c}.subspan(pad, expected.size()), expected));
#ifdef LUISA_TEST_TILE_XIR_TIRX
    if (compare_tirx) {
        test::tile_tirx::Runtime runtime{"cpu"};
        auto executable = runtime.build(kernel);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { return; }
        auto ta = runtime.upload<float>({cfg.transpose_a ? cfg.k : cfg.m, cfg.transpose_a ? cfg.m : cfg.k}, a);
        auto tb = runtime.upload<float>({cfg.transpose_b ? cfg.n : cfg.k, cfg.transpose_b ? cfg.k : cfg.n}, b);
        auto tc = runtime.allocate<float>({cfg.m, cfg.n});
        (*executable.entry)(ta, tb, tc);
        expect(close(runtime.download<float>(tc, expected.size()), expected));
    }
#else
    static_cast<void>(compare_tirx);
#endif
}

void rows(Device &device, int64_t width, bool softmax) {
    using namespace tile;
    constexpr int64_t count = 17;
    auto definition = tile_kernel("row_ops", [=](TensorView<const float, 2> A, TensorView<float, 2> B) {
        auto m = axis("m", 1), n = axis("n", width);
        for (auto &nest : parallel(shape(count))) {
            auto x = A.tile(coord(nest.index(), 0), shape(m, n)).load();
            auto y = x * 1.25f - 0.75f;
            if (softmax) {
                auto e = exp(y - reduce(y, n, maximum));
                y = e / reduce(e, n, add);
            } else {
                y = ite(y > 0.0f, y, -y) + reduce(x, n, add);
            }
            B(coord(nest.index(), 0), shape(m, n)).store(y);
        }
    });
    auto kernel = definition.capture(tensor_shape(count, width), tensor_shape(count, width));
    expect(kernel.valid());
    auto shader = compile(device, kernel);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    vector<float> a(count * width), b(a.size());
    vector<double> expected(a.size());
    for (size_t i = 0u; i < a.size(); i++) { a[i] = std::sin(static_cast<float>(i) * .173f); }
    for (int64_t row = 0; row < count; row++) {
        auto sum = 0.0, denom = 0.0;
        for (int64_t col = 0; col < width; col++) {
            sum += a[row * width + col];
            denom += std::exp(a[row * width + col] * 1.25 - .75);
        }
        for (int64_t col = 0; col < width; col++) {
            auto y = a[row * width + col] * 1.25 - .75;
            expected[row * width + col] = softmax ? std::exp(y) / denom : std::abs(y) + sum;
        }
    }
    auto ab = device.create_buffer<float>(a.size()), bb = device.create_buffer<float>(b.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << ab.copy_from(a.data()) << shader(ab, bb).dispatch() << bb.copy_to(b.data()) << synchronize();
    expect(close(b, expected));
}

void recurrence(Device &device, int64_t iterations, bool pipelined) {
    using namespace tile;
    constexpr int64_t count = 19;
    auto definition = tile_kernel("recurrence", [=](TensorView<float, 1> input, TensorView<float, 1> output) {
        for (auto &nest : parallel(shape(count))) {
            auto a = input.tile(coord(nest.index()), shape(1)).load();
            auto b = a, snapshot = a;
            input(coord(nest.index()), shape(1)).store(full<float>(shape(1), 17.0f));
            auto range = pipelined ? nest.pipeline(shape(iterations)) : nest.serial(shape(iterations));
            for (auto &step : range) {
                if (pipelined) { step.stage("compute"); }
                auto old_a = a;
                a += b + snapshot;
                b = old_a;
            }
            output(coord(nest.index() * 3), shape(1)).store(a);
            output(coord(nest.index() * 3 + 1), shape(1)).store(b);
            output(coord(nest.index() * 3 + 2), shape(1)).store(snapshot);
        }
    });
    auto kernel = definition.capture(tensor_shape(count), tensor_shape(count * 3));
    expect(kernel.valid());
    auto shader = compile(device, kernel);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    vector<float> input(count), output(count * 3), overwritten(count);
    vector<double> expected(count * 3);
    for (int64_t i = 0; i < count; i++) {
        input[i] = static_cast<float>(i + 1) * .03125f;
        auto a = static_cast<double>(input[i]), b = a;
        for (int64_t k = 0; k < iterations; k++) {
            auto old_a = a;
            a += b + input[i];
            b = old_a;
        }
        expected[i * 3] = a;
        expected[i * 3 + 1] = b;
        expected[i * 3 + 2] = input[i];
    }
    auto ab = device.create_buffer<float>(input.size()), bb = device.create_buffer<float>(output.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << ab.copy_from(input.data()) << shader(ab, bb).dispatch()
           << bb.copy_to(output.data()) << ab.copy_to(overwritten.data()) << synchronize();
    expect(close(output, expected)) << "iterations=" << iterations << " pipelined=" << pipelined
                                    << " actual=" << output[0] << "," << output[1] << "," << output[2]
                                    << " expected=" << expected[0] << "," << expected[1] << "," << expected[2];
    expect(std::all_of(overwritten.begin(), overwritten.end(), [](float x) { return x == 17.0f; }));
}

void clipped_origin(Device &device, bool overflow) {
    using namespace tile;
    auto definition = tile_kernel("clipped_origin", [=](TensorView<const float, 1> A, TensorView<float, 1> B) {
        for (auto &nest : parallel(shape(3))) {
            auto origin = overflow ? nest.index() * INT64_MAX : nest.index() - 1;
            auto x = A.tile(coord(origin), shape(1), bounds::zero).load();
            B(coord(nest.index()), shape(1)).store(x);
        }
    });
    auto kernel = definition.capture(tensor_shape(2), tensor_shape(3));
    auto shader = compile(device, kernel);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    vector<float> a{2.5f, -3.0f}, b(3, std::numeric_limits<float>::quiet_NaN());
    auto ab = device.create_buffer<float>(2), bb = device.create_buffer<float>(3);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << ab.copy_from(a.data()) << bb.copy_from(b.data()) << shader(ab, bb).dispatch() << bb.copy_to(b.data()) << synchronize();
    expect(close(b, overflow ? vector<double>{2.5, 0.0, 0.0} : vector<double>{0.0, 2.5, -3.0}));
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
    "tile_xir_runtime_reduction_fold_policies"_test = [&] { reduction_fold_policies(device); };
    "tile_xir_runtime_gemm"_test = [&] {
        gemm(device, {16, 24, 16, 1, 1, 8}, true);
        gemm(device, {17, 19, 13, 2, 3, 4, false, false, .25f}, true);
        for (auto ta : {false, true}) {
            for (auto tb : {false, true}) { gemm(device, {7, 11, 9, 2, 3, 4, ta, tb, .5f, 1u}, false); }
        }
    };
    "tile_xir_runtime_elementwise_reductions_softmax"_test = [&] {
        for (auto width : {1, 7, 17}) {
            rows(device, width, false);
            rows(device, width, true);
        }
    };
    "tile_xir_runtime_aligned_and_ragged_packet_gemm"_test = [&] {
        for (auto columns : {32, 33, 128}) {
            for (auto rows_per_tile : {1, 4}) {
                gemm(device, {19, columns, 65, rows_per_tile, 1, 8}, false);
            }
        }
    };
    "tile_xir_runtime_loop_carries_and_load_snapshot"_test = [&] {
        for (auto iterations : {0, 1, 5}) {
            recurrence(device, iterations, false);
            recurrence(device, iterations, true);
        }
    };
    "tile_xir_bounds_proof_rejects_negative_and_overflowing_origins"_test = [&] {
        clipped_origin(device, false);
        clipped_origin(device, true);
    };
}
