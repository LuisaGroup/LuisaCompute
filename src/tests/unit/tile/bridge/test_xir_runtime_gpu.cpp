// Portable GPU subset of the Tile XIR runtime tests, exercising the
// TileIR -> XIR bridge -> xir2ast -> create_shader fallback implemented by the
// DX and VK backends (src/backends/common/tile_xir_kernel.h). Modeled on the
// SIMD test_xir_runtime.cpp, without its SIMD-only environment constraints and
// without packet-local (local_lanes > 1) coverage, which the GPU fallback does
// not support in v1.
#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_reduction_policy_test_utils.h"

#include <bit>
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/runtime.h>
#include <algorithm>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

[[nodiscard]] bool close(span<const float> actual, span<const double> expected) {
    if (actual.size() != expected.size()) { return false; }
    for (size_t i = 0u; i < actual.size(); i++) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-5 + 2e-5 * std::abs(expected[i])) { return false; }
    }
    return true;
}

[[nodiscard]] bool bitwise_equal(span<const float> lhs, span<const float> rhs) {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), [](float a, float b) {
        return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
    });
}

// Storage-precision comparison for F16/BF16 accumulators: the contraction
// folds in FP32 (wide accumulation) and rounds once at write-back, so only
// the final element type's storage precision is observable.
[[nodiscard]] bool close_storage(span<const float> actual, span<const double> expected) {
    if (actual.size() != expected.size()) { return false; }
    for (size_t i = 0u; i < actual.size(); i++) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-2 + 2e-2 * std::abs(expected[i])) { return false; }
    }
    return true;
}

void check_metadata_basics(const tile::KernelMetadata &metadata, size_t argument_count) {
    expect(metadata.error.empty());
    expect(eq(metadata.arguments.size(), argument_count));
    expect(!metadata.disjoint_writes);
    expect(metadata.realization.find("TileIR -> XIR SSA -> AST -> ") != string::npos) << metadata.realization;
    expect(metadata.realization.find("local_lanes=1") != string::npos) << metadata.realization;
}

// Pointwise map + store through TensorViews with disjoint buffers.
void pointwise_map_store(Device &device) {
    using namespace tile;
    constexpr int64_t rows = 17, width = 65;
    auto kernel = tile_kernel("gpu_pointwise", [](TensorView<const float, 2> input, TensorView<float, 2> output) {
        auto m = axis("m", 1), n = axis("n", width);
        for (auto &nest : parallel(shape(rows))) {
            auto x = input[coord(nest.index(), 0), shape(m, n)];
            auto scaled = x * 2.0f + 1.0f;
            auto shifted = map<float>(shape(m, n), [&](const Nest &element) {
                return scaled.at(coord(0, element.index(n))) + cast<float>(nest.index());
            });
            output(coord(nest.index(), 0), shape(m, n)).store(shifted);
        }
    }).capture(tensor_shape(rows, width), tensor_shape(rows, width));
    constexpr size_t pad = 17u;
    constexpr float guard = -731.25f;
    auto count = static_cast<size_t>(rows * width);
    vector<float> input(count + 2u * pad, guard), initial(count + 2u * pad, guard);
    vector<double> expected(count);
    for (size_t i = 0u; i < count; i++) {
        auto value = static_cast<float>(static_cast<int32_t>(i % 31u) - 15) * .125f;
        input[pad + i] = value;
        expected[i] = value * 2.0 + 1.0 + static_cast<double>(i / width);
    }
    auto shader = tile::compile(device, kernel, {}, {.enable_fast_math = false});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    check_metadata_basics(shader.metadata(), 2u);
    expect(shader.metadata().arguments[0].usage == Usage::READ);
    expect(shader.metadata().arguments[1].usage == Usage::WRITE);
    expect(eq(shader.metadata().dispatch_size.x, static_cast<uint32_t>(rows)));
    expect(eq(shader.metadata().dispatch_size.y, 1u));
    expect(eq(shader.metadata().dispatch_size.z, 1u));
    auto block = shader.block_size().x;
    expect(block != 0u && (block & (block - 1u)) == 0u);
    auto a = device.create_buffer<float>(input.size());
    auto b = device.create_buffer<float>(initial.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto actual = initial, after = input;
    stream << a.copy_from(span{input}) << b.copy_from(span{initial})
           << shader(a.view(pad, count), b.view(pad, count)).dispatch()
           << a.copy_to(span{after}) << b.copy_to(span{actual}) << synchronize();
    expect(after == input);// read-only input, including both guards
    expect(close(span{actual}.subspan(pad, count), expected));
    expect(std::all_of(actual.begin(), actual.begin() + pad, [](float x) { return x == guard; }));
    expect(std::all_of(actual.end() - pad, actual.end(), [](float x) { return x == guard; }));
}

// Static-shape copy and scatter transpose.
void copy_transpose(Device &device) {
    using namespace tile;
    constexpr int64_t rows = 12, cols = 20;
    auto kernel = tile_kernel("gpu_copy_transpose", [](TensorView<const float, 2> input,
                                                       TensorView<float, 2> transposed,
                                                       TensorView<float, 2> copied) {
        auto r = axis("r", rows), c = axis("c", cols);
        for (auto &nest : parallel(shape(r, c))) {
            auto value = input[coord(nest.index(r), nest.index(c)), shape(1, 1)];
            copied(coord(nest.index(r), nest.index(c)), shape(1, 1)).store(value);
            transposed(coord(nest.index(c), nest.index(r)), shape(1, 1)).store(value * 2.0f);
        }
    }).capture(tensor_shape(rows, cols), tensor_shape(cols, rows), tensor_shape(rows, cols));
    constexpr size_t pad = 13u;
    constexpr float guard = -719.5f;
    vector<float> input(rows * cols + 2u * pad, guard);
    for (size_t i = 0u; i < static_cast<size_t>(rows * cols); i++) {
        input[pad + i] = static_cast<float>(static_cast<int32_t>(i % 37u) - 18) * .25f;
    }
    auto shader = tile::compile(device, kernel, {}, {.enable_fast_math = false});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    check_metadata_basics(shader.metadata(), 3u);
    expect(shader.metadata().arguments[0].usage == Usage::READ);
    expect(shader.metadata().arguments[1].usage == Usage::WRITE);
    expect(shader.metadata().arguments[2].usage == Usage::WRITE);
    auto a = device.create_buffer<float>(input.size());
    auto b = device.create_buffer<float>(cols * rows + 2u * pad);
    auto c = device.create_buffer<float>(input.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto transposed = vector<float>(cols * rows + 2u * pad, guard), copied = input, after = input;
    stream << a.copy_from(span{input})
           << b.copy_from(span{transposed})
           << c.copy_from(span{copied})
           << shader(a.view(pad, rows * cols),
                     b.view(pad, cols * rows),
                     c.view(pad, rows * cols))
                  .dispatch()
           << a.copy_to(span{after})
           << b.copy_to(span{transposed})
           << c.copy_to(span{copied})
           << synchronize();
    expect(after == input);
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t cc = 0; cc < cols; cc++) {
            auto value = input[pad + r * cols + cc];
            expect(eq(copied[pad + r * cols + cc], value));
            expect(eq(transposed[pad + cc * rows + r], value * 2.0f));
        }
    }
    expect(std::all_of(transposed.begin(), transposed.begin() + pad, [](float x) { return x == guard; }));
    expect(std::all_of(transposed.end() - pad, transposed.end(), [](float x) { return x == guard; }));
}

// Unordered tree reduction plus explicit fold_left/fold_right policies with
// dyadic-exact inputs, so ordering differences cannot hide (bitwise compare).
void reduction_fold_policies(Device &device) {
    namespace cases = test::tile_reduction;
    constexpr auto rows = int64_t{3};
    for (auto dimensions : {std::pair{0, 3}, std::pair{2, 0}, std::pair{1, 3},
                            std::pair{3, 5}, std::pair{5, 13}}) {
        auto [outer, inner] = dimensions;
        auto width = outer * inner;
        auto stride = std::max(width, 1);
        for (auto seed : {0.0f, -0.0f, 3.0f}) {
            auto kernel = cases::folds(rows, outer, inner, seed);
            auto shader = tile::compile(device, kernel, {}, {.enable_fast_math = true});
            expect(static_cast<bool>(shader)) << shader.metadata().error;
            if (!shader) { continue; }
            // Every fold in this kernel is order-sensitive: the backend must
            // disable fast math even when the caller enables it.
            expect(shader.metadata().realization.find("fast_math=false; ordered_reduction=true") != string::npos)
                << shader.metadata().realization;
            vector<float> values(rows * stride), actual(rows * cases::outputs);
            for (auto r = int64_t{0}; r < rows; r++) {
                for (auto i = 0; i < width; i++) {
                    // Small dyadic values: every partial sum is exact in fp32,
                    // so the unordered tree must match the host fold bitwise.
                    values[r * stride + i] = static_cast<float>(r == 0 ? i % 7 : (i + 1) * (r + 1)) * .25f;
                }
            }
            auto input = device.create_buffer<float>(values.size());
            auto output = device.create_buffer<float>(actual.size());
            auto stream = device.create_stream(StreamTag::COMPUTE);
            stream << input.copy_from(span{values}) << shader(input, output).dispatch()
                   << output.copy_to(span{actual}) << synchronize();
            for (auto r = int64_t{0}; r < rows; r++) {
                auto reference = cases::reference(span<const float>{values}.subspan(r * stride, width), seed);
                for (auto mode = int64_t{0}; mode < cases::outputs; mode++) {
                    expect(eq(std::bit_cast<uint32_t>(actual[r * cases::outputs + mode]),
                              std::bit_cast<uint32_t>(reference[mode])))
                        << "shape=" << outer << "," << inner << " row=" << r << " mode=" << mode << " seed=" << seed;
                }
            }
        }
    }
}

// Serial/pipeline loop nests and explicit fold directions inside one root
// parallel; two planner orders must agree bitwise (all updates are exact).
void loop_nests(Device &device) {
    using namespace tile;
    constexpr int64_t rows = 4 * 3 * 10, steps = 5, modes = 4;
    auto kernel = tile_kernel("gpu_loop_nests", [](TensorView<const float, 2> input, TensorView<float, 2> output) {
        auto a = axis("a", 4), b = axis("b", 3), c = axis("c", 10);
        for (auto &nest : parallel(shape(a, b, c))) {
            auto row = (nest.index(a) * 3 + nest.index(b)) * 10 + nest.index(c);
            auto values = input[coord(row, 0), shape(1, steps)];
            for (auto mode = int64_t{0}; mode < modes; mode++) {
                auto state = cast<float>(row) * .125f;
                if (mode == 0) {
                    for (auto &step : nest.serial(shape(steps))) { state = state * 2.0f + values.at(coord(0, step.index())); }
                } else if (mode == 1) {
                    for (auto &step : nest.pipeline(shape(steps), {.stages = 2u, .initiation_interval = 1u})) {
                        step.stage("load");
                        auto value = input[coord(row, step.index()), shape(1, 1)];
                        step.stage("compute");
                        state = state * 2.0f + value.at(coord(0, 0));
                    }
                } else {
                    auto policy = mode == 2 ? reduction::fold_left : reduction::fold_right;
                    for (auto &step : nest.reduce(shape(steps), policy)) { state = state * 2.0f + values.at(coord(0, step.index())); }
                }
                output(coord(row, mode), shape(1, 1)).store(full<float>(shape(1, 1), state));
            }
        }
    }).capture(tensor_shape(rows, steps), tensor_shape(rows, modes));
    constexpr size_t pad = 17u;
    constexpr float guard = -731.25f;
    vector<float> input(rows * steps + 2u * pad, guard), initial(rows * modes + 2u * pad, guard);
    vector<double> expected(rows * modes);
    for (int64_t row = 0; row < rows; row++) {
        for (int64_t i = 0; i < steps; i++) { input[pad + row * steps + i] = static_cast<float>((row * 3 + i * 7) % 31 - 15) * .125f; }
        for (int64_t mode = 0; mode < modes; mode++) {
            auto state = static_cast<double>(row) * .125;
            for (int64_t i = 0; i < steps; i++) { state = state * 2.0 + input[pad + row * steps + (mode == 3 ? steps - 1 - i : i)]; }
            expected[row * modes + mode] = state;
        }
    }
    // Dyadic inputs keep every update exact. Different fold directions really
    // differ, so bitwise agreement cannot conceal an accidentally reordered fold.
    expect(expected[0] != expected[3]);
    auto a = device.create_buffer<float>(input.size()), b = device.create_buffer<float>(initial.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    vector<float> baseline;
    for (auto order : vector<vector<uint32_t>>{{0u, 1u, 2u}, {2u, 0u, 1u}}) {
        auto options = bridge::xir::PlannerOptions{.root_axis_order = order};
        auto shader = tile::compile(device, kernel, {.xir = &options}, {.enable_fast_math = false});
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (!shader) { continue; }
        check_metadata_basics(shader.metadata(), 2u);
        auto actual = initial, after = input;
        stream << a.copy_from(span{input}) << b.copy_from(span{initial})
               << shader(a.view(pad, rows * steps), b.view(pad, rows * modes)).dispatch()
               << a.copy_to(span{after}) << b.copy_to(span{actual}) << synchronize();
        expect(after == input);// includes readonly input and both input guards
        expect(close(span{actual}.subspan(pad, rows * modes), expected));
        expect(std::all_of(actual.begin(), actual.begin() + pad, [](float x) { return x == guard; }));
        expect(std::all_of(actual.end() - pad, actual.end(), [](float x) { return x == guard; }));
        if (baseline.empty()) { baseline = actual; }
        expect(bitwise_equal(span{actual}.subspan(pad, rows * modes), span{baseline}.subspan(pad, rows * modes)));
    }
}

// Exact block-width constraints are honored; conflicts and non-warp multiples
// are rejected fail-closed.
void block_width_constraints(Device &device) {
    using namespace tile;
    constexpr int64_t rows = 33, width = 65;
    auto kernel = tile_kernel("gpu_block_width", [](TensorView<const float, 2> input, TensorView<float, 2> output) {
        auto m = axis("m", 1), n = axis("n", width);
        for (auto &nest : parallel(shape(rows))) {
            auto x = input[coord(nest.index(), 0), shape(m, n)];
            output(coord(nest.index(), 0), shape(m, n)).store(x + 1.0f);
        }
    }).capture(tensor_shape(rows, width), tensor_shape(rows, width));
    auto count = static_cast<size_t>(rows * width);
    vector<float> input(count);
    for (size_t i = 0u; i < count; i++) { input[i] = static_cast<float>(static_cast<int32_t>(i % 29u) - 14) * .5f; }
    {
        auto shader = tile::compile(device, kernel, {.threads_per_group = 128u}, {.enable_fast_math = false});
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (!shader) { return; }
        expect(eq(shader.block_size().x, 128u));
        expect(shader.metadata().realization.find("128 threads/group") != string::npos)
            << shader.metadata().realization;
        auto a = device.create_buffer<float>(count), b = device.create_buffer<float>(count);
        auto stream = device.create_stream(StreamTag::COMPUTE);
        auto actual = vector<float>(count, 0.0f);
        stream << a.copy_from(span{input}) << shader(a, b).dispatch() << b.copy_to(span{actual}) << synchronize();
        for (size_t i = 0u; i < count; i++) { expect(eq(actual[i], input[i] + 1.0f)); }
    }
    {
        bridge::xir::PlannerOptions options{.block_size = 64u};
        auto shader = tile::compile(device, kernel, {.xir = &options}, {.enable_fast_math = false});
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (shader) { expect(eq(shader.block_size().x, 64u)); }
    }
    {
        bridge::xir::PlannerOptions options{.block_size = 64u};
        auto shader = tile::compile(device, kernel, {.threads_per_group = 128u, .xir = &options});
        expect(!shader);
        expect(shader.metadata().error == "Conflicting XIR and Runtime block width constraints")
            << shader.metadata().error;
        expect(shader.metadata().arguments.empty());
        expect(shader.metadata().source.empty());
    }
    {
        // 48 is not a multiple of any GPU warp/subgroup width.
        auto shader = tile::compile(device, kernel, {.threads_per_group = 48u});
        expect(!shader);
        expect(shader.metadata().error == "invalid XIR block width constraint") << shader.metadata().error;
    }
}

// Fail-closed option matrix: TIRx lowering, compile-only archives, CPU
// task-grain constraints, and warp-distributed local lanes are all rejected
// with a clear metadata.error instead of a partial realization.
void fail_closed_options(Device &device) {
    using namespace tile;
    constexpr int64_t rows = 5, width = 9;
    auto kernel = tile_kernel("gpu_fail_closed", [](TensorView<const float, 2> input, TensorView<float, 2> output) {
        auto m = axis("m", 1), n = axis("n", width);
        for (auto &nest : parallel(shape(rows))) {
            auto x = input[coord(nest.index(), 0), shape(m, n)];
            output(coord(nest.index(), 0), shape(m, n)).store(x);
        }
    }).capture(tensor_shape(rows, width), tensor_shape(rows, width));
    {
        auto shader = tile::compile(device, kernel, {.lowering = tile::Lowering::TIRX});
        expect(!shader);
        expect(shader.metadata().error.find("TIRx") != string::npos) << shader.metadata().error;
        expect(shader.metadata().arguments.empty());
        expect(shader.metadata().source.empty());
        expect(shader.metadata().realization.empty());
    }
    {
        auto shader = tile::compile(device, kernel, {}, {.compile_only = true});
        expect(!shader);
        expect(!shader.metadata().error.empty());
    }
    {
        bridge::xir::PlannerOptions options{.blocks_per_task = 3u};
        auto shader = tile::compile(device, kernel, {.xir = &options});
        expect(!shader);
        expect(shader.metadata().error == "XIR target does not support CPU task-grain constraints")
            << shader.metadata().error;
    }
    {
        // Local-axis distribution is deliberately unsupported in v1.
        bridge::xir::PlannerOptions options{.local_lanes = device.compute_warp_size()};
        auto shader = tile::compile(device, kernel, {.xir = &options});
        expect(!shader);
        expect(shader.metadata().error.find("local-axis distribution") != string::npos) << shader.metadata().error;
    }
}

// Overlapping writable views of one allocation are allowed at invocation
// because the XIR bridge realization sets disjoint_writes == false and keeps
// whole-store program order.
void overlapping_writable_views(Device &device) {
    using namespace tile;
    constexpr int64_t rows = 4, width = 64;
    auto kernel = tile_kernel("gpu_overlapping_views", [](TensorView<float, 2> first, TensorView<float, 2> second) {
        auto m = axis("m", 1), n = axis("n", width);
        for (auto &nest : parallel(shape(rows))) {
            first(coord(nest.index(), 0), shape(m, n)).store(full<float>(shape(m, n), 1.0f + cast<float>(nest.index())));
            second(coord(nest.index(), 0), shape(m, n)).store(full<float>(shape(m, n), 7.0f + cast<float>(nest.index())));
        }
    }).capture(tensor_shape(rows, width), tensor_shape(rows, width));
    auto shader = tile::compile(device, kernel, {}, {.enable_fast_math = false});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    check_metadata_basics(shader.metadata(), 2u);
      expect(shader.metadata().arguments[0].usage == Usage::WRITE);
      expect(shader.metadata().arguments[1].usage == Usage::WRITE);
    // One allocation, two overlapping writable views: the second store of each
    // program must overwrite the overlapping middle in program order.
    constexpr auto shift = width / 2;
    constexpr auto total = rows * width + shift;
    auto buffer = device.create_buffer<float>(total);
    auto initial = vector<float>(total, -5.0f);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto actual = initial;
    stream << buffer.copy_from(span{initial})
           << shader(buffer.view(0, rows * width), buffer.view(shift, rows * width)).dispatch()
           << buffer.copy_to(span{actual})
           << synchronize();
    for (int64_t i = 0; i < total; i++) {
        auto value = i < shift ? 1.0f + static_cast<float>(i / width) : 7.0f + static_cast<float>((i - shift) / width);
        expect(eq(actual[static_cast<size_t>(i)], value)) << "index " << i;
    }
}

// MMA contraction on real hardware: FP32 with a depth beyond the unroll bound
// (130 = 16 full chunks + a 2-step tail), plus F16/BF16 accumulators, which
// the XIR bridge realizes as FP32 accumulation with a single rounding at
// write-back (wide accumulation). Reference values come from a double oracle.
void mma_accumulation(Device &device) {
    using namespace tile;
    constexpr int64_t m = 4, n = 5;
    auto values_a = [](int64_t i, int64_t l) { return static_cast<float>((i * 7 + l * 3) % 23 - 11) / 16.0f; };
    auto values_b = [](int64_t l, int64_t j) { return static_cast<float>((l * 5 + j * 11) % 19 - 9) / 8.0f; };
    auto run = [&](int64_t k, int64_t variant, bool storage_precision) {
        auto kernel = tile_kernel("gpu_mma", [=](TensorView<const float, 2> A, TensorView<const float, 2> B, TensorView<float, 2> C) {
            auto i = axis("i", m), j = axis("j", n), l = axis("l", k);
            for (auto &nest : parallel(shape(1))) {
                auto a = A.tile(coord(0, 0), shape(i, l)).load();
                auto b = B.tile(coord(0, 0), shape(l, j)).load();
                if (variant == 0) {
                    C(coord(0, 0), shape(i, j)).store(mma(a, b, zeros<float>(shape(i, j))));
                } else if (variant == 1) {
                    C(coord(0, 0), shape(i, j)).store(cast<float>(mma(cast<half>(a), cast<half>(b), zeros<half>(shape(i, j)))));
                } else {
                    C(coord(0, 0), shape(i, j)).store(cast<float>(mma(cast<bfloat16>(a), cast<bfloat16>(b), zeros<bfloat16>(shape(i, j)))));
                }
            }
        }).capture(tensor_shape(m, k), tensor_shape(k, n), tensor_shape(m, n));
        auto shader = tile::compile(device, kernel, {}, {.enable_fast_math = false});
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (!shader) { return; }
        check_metadata_basics(shader.metadata(), 3u);
        vector<float> va(static_cast<size_t>(m) * k), vb(static_cast<size_t>(k) * n);
        vector<double> expected(static_cast<size_t>(m) * n, 0.0);
        for (int64_t i = 0; i < m; i++) {
            for (int64_t l = 0; l < k; l++) {
                va[static_cast<size_t>(i) * k + l] = values_a(i, l);
                for (int64_t j = 0; j < n; j++) {
                    if (i == 0) { vb[static_cast<size_t>(l) * n + j] = values_b(l, j); }
                    expected[static_cast<size_t>(i) * n + j] += static_cast<double>(values_a(i, l)) * values_b(l, j);
                }
            }
        }
        auto a = device.create_buffer<float>(va.size());
        auto b = device.create_buffer<float>(vb.size());
        auto c = device.create_buffer<float>(expected.size());
        auto stream = device.create_stream(StreamTag::COMPUTE);
        auto actual = vector<float>(expected.size(), 0.0f);
        stream << a.copy_from(span{va}) << b.copy_from(span{vb})
               << shader(a, b, c).dispatch() << c.copy_to(span{actual}) << synchronize();
        if (storage_precision) {
            expect(close_storage(span{actual}, span{expected})) << "variant " << variant;
        } else {
            expect(close(span{actual}, span{expected})) << "variant " << variant;
        }
    };
    run(130, 0, false);
    run(32, 1, true);
    run(32, 2, true);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
    "tile_xir_runtime_gpu_mma_accumulation"_test = [&] { mma_accumulation(device); };
    "tile_xir_runtime_gpu_pointwise_map_store"_test = [&] { pointwise_map_store(device); };
    "tile_xir_runtime_gpu_copy_and_transpose"_test = [&] { copy_transpose(device); };
    "tile_xir_runtime_gpu_reduction_fold_policies"_test = [&] { reduction_fold_policies(device); };
    "tile_xir_runtime_gpu_loop_nests"_test = [&] { loop_nests(device); };
    "tile_xir_runtime_gpu_block_width_constraints"_test = [&] { block_width_constraints(device); };
    "tile_xir_runtime_gpu_fail_closed_options"_test = [&] { fail_closed_options(device); };
    "tile_xir_runtime_gpu_overlapping_writable_views"_test = [&] { overlapping_writable_views(device); };
}
