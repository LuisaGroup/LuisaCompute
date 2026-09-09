#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_xir_test_utils.h"
#include "tile_reduction_policy_test_utils.h"
#include <bit>
#include <luisa/runtime/stream.h>
#include <luisa/tile/runtime.h>
#include <luisa/tile/bridge/xir/planner.h>
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

[[nodiscard]] bool close(span<const float> actual, span<const double> expected);

void shared_pointwise(Device &device, int64_t width, uint32_t lanes, uint32_t variant, int32_t alias_shift, bool shared_buffer = false) {
    using namespace tile;
    // One program for shifted aliases: this exercises intra-program snapshot
    // semantics without introducing a cross-parallel-iteration data race.
    shared_buffer |= alias_shift != 0;
    auto rows = shared_buffer ? int64_t{1} : int64_t{17};
    auto stride = width * 2;
    auto kernel = tile_kernel("shared_pointwise", [=](TensorView<const float, 2> input,
                                                      TensorView<float, 2> output, TensorView<float, 2> other) {
                      auto m = axis("m", 1), n = axis("n", width);
                      for (auto &nest : parallel(shape(rows))) {
                          auto x = input[coord(nest.index(), 0), shape(m, n)];
                          auto y = input[coord(nest.index(), width), shape(m, n)];
                          auto shared = x * y + x;
                          output(coord(nest.index(), 0), shape(m, n)).store(shared + y);
                          if (variant == 0u) {
                              output(coord(nest.index(), width), shape(m, n)).store(shared - y);
                          } else if (variant == 1u) {
                              other(coord(nest.index(), 0), shape(m, n)).store(shared - y);
                          } else {
                              // Overlapping writes must keep whole-store order.
                              output(coord(nest.index(), 1), shape(m, n)).store(shared - y);
                          }
                      }
                  }).capture(tensor_shape(rows, stride), tensor_shape(rows, stride), tensor_shape(rows, stride));
    constexpr auto pad = size_t{19u};
    constexpr auto guard = -731.25f;
    auto count = static_cast<size_t>(rows * stride);
    auto extra = static_cast<size_t>(std::abs(alias_shift));
    vector<float> seed(count + extra + pad * 2u, guard);
    for (size_t i = 0u; i < count + extra; i++) { seed[pad + i] = static_cast<float>(static_cast<int32_t>(i % 31u) - 15) * .125f; }
    vector<float> baseline_a, baseline_b, baseline_c;
    for (auto enabled : {false, true}) {
        auto options = bridge::xir::PlannerOptions{.block_size = 32u, .local_lanes = lanes, .enable_pointwise_fusion = enabled};
        auto shader = compile(device, kernel, {.xir = &options}, {.enable_fast_math = false});
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (!shader) { return; }
        auto admitted = enabled && variant != 2u;
        expect(shader.metadata().realization.find(format("fused_pointwise_loads={};", admitted ? 2u : 0u)) != string::npos);
        auto a = device.create_buffer<float>(seed.size());
        auto b = device.create_buffer<float>(seed.size());
        auto c = device.create_buffer<float>(seed.size());
        auto x_offset = pad + static_cast<size_t>(std::max(-alias_shift, 0));
        auto y_offset = pad + static_cast<size_t>(std::max(alias_shift, 0));
        auto x_view = a.view(x_offset, count);
        auto y_view = shared_buffer ? a.view(y_offset, count) : b.view(pad, count);
        // In the separate-output case, also test output/output aliases.
        auto z_view = variant == 1u && shared_buffer ? a.view(x_offset, count) : c.view(pad, count);
        vector<float> expected_a = seed, expected_b = seed, expected_c = seed;
        auto &dest = shared_buffer ? expected_a : expected_b;
        auto dest_offset = shared_buffer ? y_offset : pad;
        auto &second = variant == 1u && shared_buffer ? expected_a : expected_c;
        auto second_offset = variant == 1u && shared_buffer ? x_offset : pad;
        for (int64_t r = 0; r < rows; r++) {
            vector<float> first(width), last(width);
            for (int64_t col = 0; col < width; col++) {
                auto x = seed[x_offset + r * stride + col], y = seed[x_offset + r * stride + width + col];
                auto shared = x * y + x;
                first[col] = shared + y;
                last[col] = shared - y;
            }
            for (int64_t col = 0; col < width; col++) { dest[dest_offset + r * stride + col] = first[col]; }
            for (int64_t col = 0; col < width; col++) {
                if (variant == 1u) {
                    second[second_offset + r * stride + col] = last[col];
                } else {
                    dest[dest_offset + r * stride + (variant == 0u ? width : 1) + col] = last[col];
                }
            }
        }
        auto stream = device.create_stream(StreamTag::COMPUTE);
        auto actual_a = seed, actual_b = seed, actual_c = seed;
        stream << a.copy_from(span{seed}) << b.copy_from(span{seed}) << c.copy_from(span{seed})
               << shader(x_view, y_view, z_view).dispatch()
               << a.copy_to(span{actual_a}) << b.copy_to(span{actual_b}) << c.copy_to(span{actual_c}) << synchronize();
        // Dyadic finite inputs make this operation sequence exactly representable.
        // Compare all three allocations, including unchanged regions and guards.
        expect(actual_a == expected_a && actual_b == expected_b && actual_c == expected_c) << "width=" << width << " lanes=" << lanes << " alias shift=" << alias_shift;
        if (!enabled) {
            baseline_a = actual_a;
            baseline_b = actual_b;
            baseline_c = actual_c;
        } else {
            expect(actual_a == baseline_a && actual_b == baseline_b && actual_c == baseline_c);
        }
    }
}

void task_grain(Device &device, int64_t rows, uint32_t lanes) {
    using namespace tile;
    constexpr auto width = int64_t{65};
    auto kernel = tile_kernel("task_grain", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
                      auto m = axis("m", 1), n = axis("n", width);
                      for (auto &nest : parallel(shape(rows))) {
                          auto x = input[coord(nest.index(), 0), shape(m, n)];
                          output(coord(nest.index(), 0), shape(m, n)).store(x + reduce(x, n, add));
                      }
                  }).capture(tensor_shape(rows, width), tensor_shape(rows, width));
    constexpr auto pad = size_t{17u};
    constexpr auto guard = -731.25f;
    vector<float> input(rows * width), initial(rows * width + 2u * pad, guard), baseline;
    vector<double> expected(rows * width);
    for (int64_t row = 0; row < rows; row++) {
        double sum = 0.0;
        for (int64_t col = 0; col < width; col++) {
            auto value = static_cast<float>((row * 3 + col * 7) % 31 - 15) * .125f;
            input[row * width + col] = value;
            sum += value;
        }
        for (int64_t col = 0; col < width; col++) { expected[row * width + col] = input[row * width + col] + sum; }
    }
    auto a = device.create_buffer<float>(input.size()), b = device.create_buffer<float>(initial.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.copy_from(span{input}) << synchronize();
    string baseline_llvm;
    for (auto grain : {0u, 1u, 3u, 16u, UINT32_MAX}) {
        auto options = bridge::xir::PlannerOptions{.block_size = 32u, .local_lanes = lanes, .blocks_per_task = grain};
        auto shader = compile(device, kernel, {.xir = &options}, {.enable_fast_math = false});
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (!shader) { continue; }
        expect(shader.metadata().realization.find(format("blocks_per_task={};", grain)) != string::npos);
        auto output = initial;
        stream << b.copy_from(span{initial}) << shader(a, b.view(pad, input.size())).dispatch()
               << b.copy_to(span{output}) << synchronize();
        expect(close(span{output}.subspan(pad, input.size()), expected));
        expect(std::all_of(output.begin(), output.begin() + pad, [](float x) { return x == guard; }));
        expect(std::all_of(output.end() - pad, output.end(), [](float x) { return x == guard; }));
        if (grain == 0u) {
            baseline = output;
            baseline_llvm = shader.metadata().source;
        } else {
            expect(output == baseline);
            expect(shader.metadata().source == baseline_llvm) << "CPU task grain must not alter native kernel code";
        }
    }
}

void fused_load_reductions(Device &device, int64_t width, uint32_t lanes, uint32_t variant) {
    using namespace tile;
    constexpr auto rows = int64_t{17};
    auto definition = tile_kernel("fused_load_reductions", [=](TensorView<const float, 2> input,
                                                               TensorView<float, 2> alias, TensorView<float, 2> output) {
        auto m = axis("m", 1), n = axis("n", width);
        for (auto &nest : parallel(shape(rows))) {
            auto x = input[coord(nest.index(), 0), shape(m, n)];
            auto y = alias[coord(nest.index(), 0), shape(m, n)];
            if (variant == 2u) { alias(coord(nest.index(), 0), shape(m, n)).store(full<float>(shape(m, n), 9.0f)); }
            auto sum = reduce(x * y, n, add);
            if (variant == 1u) { alias(coord(nest.index(), 0), shape(m, n)).store(full<float>(shape(m, n), 9.0f)); }
            output(coord(nest.index(), 0), shape(m, n)).store(variant == 1u ? x + sum : full<float>(shape(m, n), sum.at(coord(0))));
        }
    });
    auto kernel = definition.capture(tensor_shape(rows, width), tensor_shape(rows, width), tensor_shape(rows, width));
    constexpr auto pad = size_t{17};
    constexpr auto guard = -731.25f;
    vector<float> original(rows * width + 2 * pad, guard);
    vector<double> expected(rows * width);
    for (int64_t row = 0; row < rows; row++) {
        double sum = 0.0;
        for (int64_t i = 0; i < width; i++) {
            auto x = static_cast<float>((i * 7 + row * 3) % 17 - 8) * .125f;
            original[pad + row * width + i] = x;
            sum += x * x;
        }
        for (int64_t i = 0; i < width; i++) { expected[row * width + i] = sum + (variant == 1u ? original[pad + row * width + i] : 0.0); }
    }
    vector<float> baseline;
    for (auto fusion : {false, true}) {
        auto options = bridge::xir::PlannerOptions{.block_size = 32u, .local_lanes = lanes, .enable_load_reduction_fusion = fusion};
        auto shader = compile(device, kernel, {.xir = &options}, {.enable_fast_math = false});
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (!shader) { return; }
        auto fused = fusion && variant != 2u ? 2u : 0u;
        expect(shader.metadata().realization.find(format("fused_reduction_loads={};", fused)) != string::npos);
        vector<float> data = original, actual(original.size(), guard);
        auto a = device.create_buffer<float>(data.size()), b = device.create_buffer<float>(actual.size());
        auto av = a.view(pad, expected.size()), bv = b.view(pad, expected.size());
        auto stream = device.create_stream(StreamTag::COMPUTE);
        stream << a.copy_from(span{data}) << b.copy_from(span{actual}) << shader(av, av, bv).dispatch()
               << a.copy_to(span{data}) << b.copy_to(span{actual}) << synchronize();
        expect(close(span{actual}.subspan(pad, expected.size()), expected));
        for (size_t i = 0u; i < expected.size(); i++) { expect(eq(data[pad + i], variant == 0u ? original[pad + i] : 9.0f)); }
        for (auto values : {span{actual}, span{data}}) {
            expect(std::all_of(values.begin(), values.begin() + pad, [](float x) { return x == guard; }));
            expect(std::all_of(values.end() - pad, values.end(), [](float x) { return x == guard; }));
        }
        if (!fusion) {
            baseline = actual;
        } else {
            expect(actual == baseline);
        }
    }
}

void fused_load_scopes(Device &device, int64_t iterations, bool staged, bool retained) {
    using namespace tile;
    constexpr auto width = int64_t{69};
    auto definition = tile_kernel("fused_load_scopes", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
        auto m = axis("m", 3), n = axis("n", 23);
        for (auto &nest : parallel(shape(1))) {
            auto range = staged ? nest.pipeline(shape(iterations)) : nest.serial(shape(iterations));
            for (auto &step : range) {
                // Different captured origins, an out-of-bounds first row,
                // and a multidimensional/permuted reduction domain.
                auto x = input.tile(coord(step.index() * 3 - 3, 0), shape(m, n), bounds::zero).load();
                if (staged) { step.stage("consumer"); }
                auto sum = Scalar<float>{2.5f};
                for (auto &element : step.reduce(shape(n, m))) { sum += x.at(coord(element.index(m), element.index(n))); }
                auto y = retained ? x + sum : full<float>(shape(m, n), sum);
                output(coord(step.index() * 3, 0), shape(m, n)).store(y);
            }
        }
    });
    auto kernel = definition.capture(tensor_shape(9, 23), tensor_shape(9, 23));
    vector<float> input(3 * width), expected(3 * width, -731.25f), baseline;
    for (size_t i = 0u; i < input.size(); i++) { input[i] = static_cast<float>(i % 13u) * .125f; }
    for (int64_t step = 0; step < iterations; step++) {
        auto sum = 2.5f;
        if (step != 0) {
            for (int64_t i = 0; i < width; i++) { sum += input[(step - 1) * width + i]; }
        }
        for (int64_t i = 0; i < width; i++) { expected[step * width + i] = sum + (retained && step != 0 ? input[(step - 1) * width + i] : 0.0f); }
    }
    for (auto fusion : {false, true}) {
        auto options = bridge::xir::PlannerOptions{.enable_load_reduction_fusion = fusion};
        auto shader = compile(device, kernel, {.xir = &options}, {.enable_fast_math = false});
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (!shader) { return; }
        if (staged) { expect(shader.metadata().realization.find("fused_reduction_loads=0;") != string::npos); }
        constexpr auto pad = size_t{17};
        constexpr auto guard = -731.25f;
        vector<float> actual(input.size() + 2u * pad, guard);
        auto a = device.create_buffer<float>(input.size()), b = device.create_buffer<float>(actual.size());
        auto stream = device.create_stream(StreamTag::COMPUTE);
        stream << a.copy_from(span{input}) << b.copy_from(span{actual}) << shader(a, b.view(pad, input.size())).dispatch()
               << b.copy_to(span{actual}) << synchronize();
        expect(std::equal(expected.begin(), expected.end(), actual.begin() + pad));
        expect(std::all_of(actual.begin(), actual.begin() + pad, [](float x) { return x == guard; }));
        expect(std::all_of(actual.end() - pad, actual.end(), [](float x) { return x == guard; }));
        if (!fusion) {
            baseline = actual;
        } else {
            expect(actual == baseline);
        }
    }
}

void reduction_fold_policies(Device &device) {
    namespace cases = test::tile_reduction;
    constexpr auto rows = int64_t{3};
    for (auto dimensions : {std::pair{0, 3}, std::pair{2, 0}, std::pair{1, 1},
                            std::pair{1, 3}, std::pair{2, 3}, std::pair{3, 5}, std::pair{5, 13}}) {
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
    if (!close(b, expected)) {
        for (size_t i = 0u; i < b.size(); i++) {
            if (!std::isfinite(b[i]) || std::abs(b[i] - expected[i]) > 2e-5 + 2e-5 * std::abs(expected[i])) {
                auto row = i / width;
                auto sequential = 0.0f;
                for (auto col = int64_t{0}; col < width; col++) { sequential += a[row * width + col]; }
                auto y = a[i] * 1.25f - .75f;
                LUISA_WARNING("Row mismatch width={} softmax={} index={} actual={} fp64={} fp32_sequential={}",
                              width, softmax, i, b[i], expected[i], std::abs(y) + sequential);
                break;
            }
        }
    }
    expect(close(b, expected)) << "width=" << width << " softmax=" << softmax;
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

void clipped_origin(Device &device, bool overflow, bool fused = false) {
    using namespace tile;
    constexpr auto width = int64_t{65};
    auto definition = tile_kernel("clipped_origin", [=](TensorView<const float, 1> A, TensorView<float, 1> B) {
        auto element = axis("element", width);
        for (auto &nest : parallel(shape(3))) {
            auto origin = overflow ? nest.index() * INT64_MAX : nest.index() - 1;
            auto x = A.tile(coord(origin), shape(element), bounds::zero).load();
            B(coord(nest.index() * width), shape(element)).store(x);
        }
    });
    auto kernel = definition.capture(tensor_shape(2), tensor_shape(3 * width));
    auto options = bridge::xir::PlannerOptions{.enable_pointwise_fusion = fused};
    auto shader = compile(device, kernel, {.xir = &options});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    expect(shader.metadata().realization.find(format("fused_pointwise_loads={};", fused ? 1u : 0u)) != string::npos);
    vector<float> a{2.5f, -3.0f}, b(3 * width, std::numeric_limits<float>::quiet_NaN());
    vector<double> expected(3 * width, 0.0);
    for (int64_t r = 0; r < 3; r++) {
        // Tile Index arithmetic wraps; spell out the wrapped origin so the
        // host oracle itself never evaluates an overflowing signed multiply.
        auto origin = overflow ? (r == 0 ? int64_t{0} : r == 1 ? INT64_MAX :
                                                                 int64_t{-2}) :
                                 r - 1;
        if (origin > 1) { continue; }
        for (int64_t c = 0; c < width; c++) {
            auto index = origin + c;
            if (index >= 0 && index < 2) { expected[r * width + c] = a[index]; }
        }
    }
    auto ab = device.create_buffer<float>(2), bb = device.create_buffer<float>(3 * width);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << ab.copy_from(a.data()) << bb.copy_from(b.data()) << shader(ab, bb).dispatch() << bb.copy_to(b.data()) << synchronize();
    expect(close(b, expected));
}

void indexed_snapshots(Device &device, int64_t width, int64_t iterations, bool pipelined) {
    using namespace tile;
    constexpr auto count = int64_t{19};
    auto definition = tile_kernel("indexed_snapshots", [=](TensorView<const float, 2> input,
                                                           TensorView<float, 2> alias, TensorView<float, 2> output) {
        for (auto &nest : parallel(shape(count))) {
            auto snapshot = input[coord(nest.index(), 0), shape(1, width)];
            auto a = snapshot, b = snapshot + 100.0f;
            // The const View aliases this writable parameter at runtime.
            alias(coord(nest.index(), 0), shape(1, width)).store(full<float>(shape(1, width), 17.0f));
            auto trace = Scalar<float>{0.0f};
            auto range = pipelined ? nest.pipeline(shape(iterations)) : nest.serial(shape(iterations));
            for (auto &step : range) {
                if (pipelined) { step.stage("compute"); }
                auto index = step.index() % width;
                trace += a.at(coord(0, index)) + 2.0f * b.at(coord(0, index)) + snapshot.at(coord(0, index));
                auto old_a = a;
                a = b + 3.0f;
                b = old_a - 2.0f;
            }
            auto index = nest.index() % width;
            for (auto column = 0; column < 4; column++) {
                auto value = column == 0 ? trace : column == 1 ? a.at(coord(0, index)) :
                                               column == 2     ? b.at(coord(0, index)) :
                                                                 snapshot.at(coord(0, index));
                output(coord(nest.index(), column), shape(1, 1)).store(full<float>(shape(1, 1), value));
            }
        }
    });
    auto kernel = definition.capture(tensor_shape(count, width), tensor_shape(count, width), tensor_shape(count, 4));
    expect(kernel.valid());
    auto shader = compile(device, kernel);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    constexpr auto pad = size_t{17};
    constexpr auto guard = -731.25f;
    vector<float> input(count * width), actual(count * 4 + 2 * pad, guard), overwritten(input.size());
    vector<double> expected(count * 4);
    for (auto r = int64_t{0}; r < count; r++) {
        vector<double> a(width), b(width);
        for (auto i = int64_t{0}; i < width; i++) {
            input[r * width + i] = static_cast<float>(r * width + i) * .125f;
            a[i] = input[r * width + i];
            b[i] = a[i] + 100.0;
        }
        auto trace = 0.0;
        for (auto k = int64_t{0}; k < iterations; k++) {
            auto i = k % width;
            trace += a[i] + 2.0 * b[i] + input[r * width + i];
            auto old_a = a;
            for (auto j = int64_t{0}; j < width; j++) {
                a[j] = b[j] + 3.0;
                b[j] = old_a[j] - 2.0;
            }
        }
        expected[r * 4] = trace;
        expected[r * 4 + 1] = a[r % width];
        expected[r * 4 + 2] = b[r % width];
        expected[r * 4 + 3] = input[r * width + r % width];
    }
    auto ab = device.create_buffer<float>(input.size() + pad), cb = device.create_buffer<float>(actual.size());
    auto av = ab.view(pad, input.size()), cv = cb.view(pad, expected.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << av.copy_from(input.data()) << cb.copy_from(actual.data()) << shader(av, av, cv).dispatch()
           << cb.copy_to(actual.data()) << av.copy_to(overwritten.data()) << synchronize();
    expect(close(span{actual}.subspan(pad, expected.size()), expected)) << "width=" << width << " iterations=" << iterations << " pipeline=" << pipelined;
    expect(std::all_of(actual.begin(), actual.begin() + pad, [](float x) { return x == guard; }));
    expect(std::all_of(actual.end() - pad, actual.end(), [](float x) { return x == guard; }));
    expect(std::all_of(overwritten.begin(), overwritten.end(), [](float x) { return x == 17.0f; }));
}

void indexed_bounds(Device &device, int64_t width) {
    using namespace tile;
    auto definition = tile_kernel("indexed_bounds", [=](TensorView<const float, 1> input, TensorView<float, 1> output) {
        for (auto &nest : parallel(shape(width + 2))) {
            auto x = input[coord(0), shape(width)];
            auto value = x.at(coord(nest.index() - 1));
            output(coord(nest.index()), shape(1)).store(full<float>(shape(1), value));
        }
    });
    auto input_count = std::max(width, int64_t{1});
    auto kernel = definition.capture(tensor_shape(input_count), tensor_shape(width + 2));
    auto shader = compile(device, kernel);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    vector<float> input(input_count), actual(width + 2);
    vector<double> expected(width + 2, 0.0);
    for (auto i = int64_t{0}; i < width; i++) { expected[i + 1] = input[i] = static_cast<float>(i + 1) * .25f; }
    auto ab = device.create_buffer<float>(input.size()), cb = device.create_buffer<float>(actual.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << ab.copy_from(input.data()) << shader(ab, cb).dispatch() << cb.copy_to(actual.data()) << synchronize();
    expect(close(actual, expected)) << "width=" << width;
}

void bounded_transpose_alias(Device &device, int64_t rows = 7, int64_t columns = 11) {
    using namespace tile;
    constexpr auto count = int64_t{67};
    auto definition = tile_kernel("bounded_transpose_alias", [=](TensorView<const float, 3> input, TensorView<float, 3> output) {
        auto b = axis("b", 1), m = axis("m", rows), n = axis("n", columns);
        for (auto &nest : parallel(shape(count))) {
            auto snapshot = input[coord(nest.index(), 0, 0), shape(b, m, n)];
            auto transposed = map<float>(shape(b, n, m), [&](const Nest &element) {
                return snapshot.at(coord(0, element.index(m), element.index(n))) * 1.25f + cast<float>(nest.index());
            });
            output(coord(nest.index(), 0, 0), shape(b, n, m)).store(transposed);
        }
    });
    auto kernel = definition.capture(tensor_shape(count, rows, columns), tensor_shape(count, columns, rows));
    auto shader = compile(device, kernel, {.threads_per_group = 32u});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    if (rows * columns * 8 > 65536) {
        expect(shader.metadata().realization.find("private_workspace_bytes=0;") == string::npos);
    }
    constexpr auto pad = size_t{17};
    constexpr auto guard = -731.25f;
    vector<float> data(count * rows * columns + 2 * pad, guard);
    vector<double> expected(count * rows * columns);
    for (auto b = int64_t{0}; b < count; b++) {
        for (auto m = int64_t{0}; m < rows; m++) {
            for (auto n = int64_t{0}; n < columns; n++) {
                auto value = static_cast<float>(b * rows * columns + m * columns + n) * .125f;
                data[pad + b * rows * columns + m * columns + n] = value;
                expected[b * rows * columns + n * rows + m] = value * 1.25 + b;
            }
        }
    }
    auto buffer = device.create_buffer<float>(data.size());
    auto view = buffer.view(pad, expected.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << buffer.copy_from(span{data}) << shader(view, view).dispatch() << buffer.copy_to(span{data}) << synchronize();
    expect(close(span{data}.subspan(pad, expected.size()), expected));
    expect(std::all_of(data.begin(), data.begin() + pad, [](float x) { return x == guard; }));
    expect(std::all_of(data.end() - pad, data.end(), [](float x) { return x == guard; }));
}

void partitioned_reductions(Device &device, int64_t width, uint32_t partitions) {
    using namespace tile;
    constexpr auto rows = int64_t{4};
    auto stride = std::max(width, int64_t{1});
    auto definition = tile_kernel("partitioned_reductions", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
        for (auto &nest : parallel(shape(rows))) {
            auto x = input[coord(nest.index(), 0), shape(1, width)];
            auto sum = ite(nest.index() == 0, Scalar<float>{-0.0f}, Scalar<float>{2.5f});
            auto product = Scalar<float>{2.0f};
            auto dependent = Scalar<float>{1.0f};
            for (auto &step : nest.reduce(shape(width))) { sum += x.at(coord(0, step.index())); }
            for (auto &step : nest.reduce(shape(width))) { product *= x.at(coord(0, step.index())); }
            // This is not a closed associative combine and must keep its
            // original recurrence despite unordered contribution permission.
            for (auto &step : nest.reduce(shape(width))) { dependent = dependent * .5f + x.at(coord(0, step.index())); }
            output(coord(nest.index(), 0), shape(1, 1)).store(full<float>(shape(1, 1), sum));
            output(coord(nest.index(), 1), shape(1, 1)).store(full<float>(shape(1, 1), product));
            output(coord(nest.index(), 2), shape(1, 1)).store(full<float>(shape(1, 1), dependent));
        }
    });
    auto kernel = definition.capture(tensor_shape(rows, stride), tensor_shape(rows, 3));
    auto options = bridge::xir::PlannerOptions{.reduction_partitions = partitions};
    auto shader = compile(device, kernel, {.xir = &options});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    vector<float> input(rows * stride), actual(rows * 3), expected(rows * 3);
    for (int64_t row = 0; row < rows; row++) {
        for (int64_t i = 0; i < width; i++) { input[row * stride + i] = row == 0 ? -0.0f : row == 1 ? .25f :
                                                                                                      (i % 3 == 0 ? 1.0f : .5f); }
        auto values = span<const float>{input}.subspan(row * stride, width);
        auto fold = [&](float seed, auto combine) {
            auto count = width > 64 && partitions > 1 ? partitions : 1u;
            if (count == 1u || width == 0) {
                for (auto value : values) { seed = combine(seed, value); }
            } else {
                for (auto p = 0u; p < count; p++) {
                    auto partial = values[p];
                    for (auto i = static_cast<size_t>(p + count); i < values.size(); i += count) { partial = combine(partial, values[i]); }
                    seed = combine(seed, partial);
                }
            }
            return seed;
        };
        expected[row * 3] = fold(row == 0 ? -0.0f : 2.5f, [](float a, float b) { return a + b; });
        expected[row * 3 + 1] = fold(2.0f, [](float a, float b) { return a * b; });
        auto dependent = 1.0f;
        for (auto value : values) { dependent = dependent * .5f + value; }
        expected[row * 3 + 2] = dependent;
    }
    auto a = device.create_buffer<float>(input.size()), b = device.create_buffer<float>(actual.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.copy_from(span{input}) << shader(a, b).dispatch() << b.copy_to(span{actual}) << synchronize();
    for (size_t i = 0u; i < actual.size(); i++) {
        expect(eq(std::bit_cast<uint32_t>(actual[i]), std::bit_cast<uint32_t>(expected[i]))) << "width=" << width << " partitions=" << partitions << " index=" << i;
    }
}

void packet_local_reductions(Device &device, int64_t count, int64_t width, uint32_t partitions) {
    using namespace tile;
    auto lanes = device.compute_warp_size();
    auto definition = tile_kernel("packet_local_reductions", [=](TensorView<float, 2> data, TensorView<float, 2> output) {
        auto m = axis("m", 1), n = axis("n", width);
        for (auto &nest : parallel(shape(count))) {
            auto x = data[coord(nest.index(), 0), shape(m, n)];
            // The load is a snapshot even when the resource is overwritten
            // before its distributed reduction/pointwise consumers execute.
            data(coord(nest.index(), 0), shape(m, n)).store(full<float>(shape(m, n), 9.0f));
            auto after = data[coord(nest.index(), 0), shape(m, n)];
            auto sum = ite(nest.index() == 0, Scalar<float>{-0.0f}, Scalar<float>{2.5f});
            auto product = Scalar<float>{2.0f};
            auto low = Scalar<float>{8.0f}, high = Scalar<float>{-8.0f};
            for (auto &step : nest.reduce(shape(n))) { sum += x.at(coord(0, step.index())); }
            for (auto &step : nest.reduce(shape(n))) { product *= x.at(coord(0, step.index())); }
            for (auto &step : nest.reduce(shape(n))) { low = min(low, x.at(coord(0, step.index()))); }
            for (auto &step : nest.reduce(shape(n))) { high = max(high, x.at(coord(0, step.index()))); }
            output(coord(nest.index(), 0), shape(1, 1)).store(full<float>(shape(1, 1), sum));
            output(coord(nest.index(), 1), shape(1, 1)).store(full<float>(shape(1, 1), product));
            output(coord(nest.index(), 2), shape(1, 1)).store(full<float>(shape(1, 1), low));
            output(coord(nest.index(), 3), shape(1, 1)).store(full<float>(shape(1, 1), high));
            data(coord(nest.index(), 0), shape(m, n)).store(x + after);
        }
    });
    auto kernel = definition.capture(tensor_shape(count, width), tensor_shape(count, 4));
    expect(kernel.valid());
    auto options = bridge::xir::PlannerOptions{.block_size = 32u, .reduction_partitions = partitions, .local_lanes = lanes};
    auto shader = compile(device, kernel, {.xir = &options});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    expect(shader.metadata().realization.find(format("local_lanes={};", lanes)) != string::npos);
    constexpr auto pad = size_t{17};
    constexpr auto guard = -731.25f;
    vector<float> data(count * width + 2 * pad, guard), actual(count * 4 + 2 * pad, guard);
    vector<double> expected_data(count * width), expected_output(count * 4);
    for (int64_t row = 0; row < count; row++) {
        auto sum = row == 0 ? -0.0 : 2.5, product = 2.0, low = 8.0, high = -8.0;
        for (int64_t i = 0; i < width; i++) {
            auto x = row == 0 ? -0.0f : (i % 3 == 0 ? 1.0f : .5f);
            data[pad + row * width + i] = x;
            expected_data[row * width + i] = x + 9.0;
            sum += x;
            product *= x;
            low = std::min(low, static_cast<double>(x));
            high = std::max(high, static_cast<double>(x));
        }
        expected_output[row * 4] = sum;
        expected_output[row * 4 + 1] = product;
        expected_output[row * 4 + 2] = low;
        expected_output[row * 4 + 3] = high;
    }
    auto a = device.create_buffer<float>(data.size()), b = device.create_buffer<float>(actual.size());
    auto av = a.view(pad, expected_data.size()), bv = b.view(pad, expected_output.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.copy_from(span{data}) << b.copy_from(span{actual}) << shader(av, bv).dispatch()
           << a.copy_to(span{data}) << b.copy_to(span{actual}) << synchronize();
    expect(close(span{data}.subspan(pad, expected_data.size()), expected_data));
    expect(close(span{actual}.subspan(pad, expected_output.size()), expected_output)) << "rows=" << count << " width=" << width << " partitions=" << partitions;
    // No invented additive identity and no repeated initial accumulator.
    expect(eq(std::bit_cast<uint32_t>(actual[pad]), std::bit_cast<uint32_t>(-0.0f)));
    for (auto values : {span{data}, span{actual}}) {
        expect(std::all_of(values.begin(), values.begin() + pad, [](float x) { return x == guard; }));
        expect(std::all_of(values.end() - pad, values.end(), [](float x) { return x == guard; }));
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
    "tile_xir_runtime_shared_pointwise_alias_paths"_test = [&] {
        for (auto lanes : {1u, device.compute_warp_size()}) {
            for (auto width : {65, 128, 257}) {
                for (auto variant : {0u, 1u, 2u}) {
                    for (auto shift : {-1, 0, 1}) { shared_pointwise(device, width, lanes, variant, shift); }
                    shared_pointwise(device, width, lanes, variant, 0, true);
                }
                // Equality at the interval boundary is disjoint, even when
                // the two resource views share an underlying allocation.
                for (auto shift : {-2 * width, 2 * width}) { shared_pointwise(device, width, lanes, 0u, shift); }
            }
        }
    };
    "tile_xir_runtime_task_grain_preserves_kernel_and_guards"_test = [&] {
        for (auto rows : {17, 129}) {
            for (auto lanes : {1u, device.compute_warp_size()}) { task_grain(device, rows, lanes); }
        }
    };
    "tile_xir_runtime_fused_loads_preserve_alias_snapshots"_test = [&] {
        for (auto lanes : {1u, device.compute_warp_size()}) {
            for (auto width : {65, 256, 4096}) {
                for (auto variant : {0u, 1u, 2u}) { fused_load_reductions(device, width, lanes, variant); }
            }
        }
        for (auto iterations : {0, 1, 3}) {
            for (auto staged : {false, true}) {
                for (auto retained : {false, true}) { fused_load_scopes(device, iterations, staged, retained); }
            }
        }
    };
    "tile_xir_runtime_packet_local_reductions_and_snapshot"_test = [&] {
        auto width = static_cast<int64_t>(device.compute_warp_size());
        for (auto partitions : {1u, 3u, 4u, 16u}) {
            for (auto n : {width, width + 1, int64_t{65}, int64_t{127}, int64_t{256}}) {
                packet_local_reductions(device, 17, n, partitions);
            }
        }
        packet_local_reductions(device, 1, 4096, 4u);
        packet_local_reductions(device, 67, 16384, 4u);
    };
    "tile_xir_runtime_reduction_fold_policies"_test = [&] { reduction_fold_policies(device); };
    "tile_xir_runtime_bounded_transpose_preserves_alias_snapshot"_test = [&] {
        bounded_transpose_alias(device);
        bounded_transpose_alias(device, 129, 65);
    };
    "tile_xir_runtime_partitioned_closed_reductions"_test = [&] {
        for (auto partitions : {1u, 3u, 4u, 16u}) {
            for (auto width : {0, 1, 65, 66, 67, 128}) { partitioned_reductions(device, width, partitions); }
        }
    };
    "tile_xir_runtime_gemm"_test = [&] {
        gemm(device, {16, 24, 16, 1, 1, 8}, true);
        gemm(device, {17, 19, 13, 2, 3, 4, false, false, .25f}, true);
        for (auto ta : {false, true}) {
            for (auto tb : {false, true}) { gemm(device, {7, 11, 9, 2, 3, 4, ta, tb, .5f, 1u}, false); }
        }
        gemm(device, {7, 11, 129, 3, 5, 65, true, true, .25f}, false);
        gemm(device, {11, 13, 17, 9, 9, 5, false, false, .25f}, false);
    };
    "tile_xir_runtime_elementwise_reductions_softmax"_test = [&] {
        for (auto width : {1, 7, 17, 65, 129, 4096}) {
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
        for (auto fused : {false, true}) {
            clipped_origin(device, false, fused);
            clipped_origin(device, true, fused);
        }
    };
    "tile_xir_runtime_indexable_snapshots_and_simultaneous_carries"_test = [&] {
        for (auto width : {1, 7, 65, 127}) {
            for (auto iterations : {0, 1, 5}) {
                indexed_snapshots(device, width, iterations, false);
                indexed_snapshots(device, width, iterations, true);
            }
        }
        for (auto width : {0, 1, 7, 65, 129}) { indexed_bounds(device, width); }
    };
}
