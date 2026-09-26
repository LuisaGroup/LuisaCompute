// Shared packet-local reduction execution checks for SIMD and Metal4.
// Covers short/ragged contributions, nonidentity seeds, valid bounds fills,
// signed zero, eager snapshots and guarded Runtime buffer views.
#pragma once

#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/stream.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/dsl.h>
#include <luisa/tile/runtime.h>
#include <luisa/core/stl/memory.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace luisa::test::tile_xir {

inline void packet_local_reductions(compute::Device &device, int64_t count, int64_t width,
                                    uint32_t partitions, bool zero_fill = false) {
    using namespace compute;
    using namespace compute::tile;
    using namespace boost::ut;
    LUISA_ASSERT(count > 0 && width > 0 && (!zero_fill || width == 7), "Invalid packet reduction fixture");
    auto lanes = device.compute_warp_size();
    auto physical_width = zero_fill ? int64_t{1} : width;
    auto definition = tile_kernel("packet_local_reductions", [=](TensorView<float, 2> data, TensorView<float, 2> output) {
        auto m = axis("m", 1), n = axis("n", width);
        for (auto &nest : parallel(shape(count))) {
            auto x = data.tile(coord(nest.index(), 0), shape(m, n), bounds::zero).load();
            // The fill-only fixture leaves its input unchanged. The other
            // fixture requires all later consumers to use the old snapshot.
            if (!zero_fill) { data(coord(nest.index(), 0), shape(m, n)).store(full<float>(shape(m, n), 9.0f)); }
            auto after = data.tile(coord(nest.index(), 0), shape(m, n), bounds::zero).load();
            auto sum = ite(nest.index() == 0, Scalar<float>{-0.0f}, Scalar<float>{2.5f});
            auto product = Scalar<float>{2.0f};
            auto low = Scalar<float>{8.0f}, high = Scalar<float>{-8.0f};
            auto shifted_sum = Scalar<float>{2.5f};
            for (auto &step : nest.reduce(shape(n))) { sum += x.at(coord(0, step.index())); }
            for (auto &step : nest.reduce(shape(n))) { product *= x.at(coord(0, step.index())); }
            for (auto &step : nest.reduce(shape(n))) { low = min(low, x.at(coord(0, step.index()))); }
            for (auto &step : nest.reduce(shape(n))) { high = max(high, x.at(coord(0, step.index()))); }
            // A zero-filled point is a real contribution, including after
            // elementwise transformation; it is not an absent reduction lane.
            for (auto &step : nest.reduce(shape(n))) { shifted_sum += x.at(coord(0, step.index())) + 3.0f; }
            output(coord(nest.index(), 0), shape(1, 1)).store(full<float>(shape(1, 1), sum));
            output(coord(nest.index(), 1), shape(1, 1)).store(full<float>(shape(1, 1), product));
            output(coord(nest.index(), 2), shape(1, 1)).store(full<float>(shape(1, 1), low));
            output(coord(nest.index(), 3), shape(1, 1)).store(full<float>(shape(1, 1), high));
            output(coord(nest.index(), 4), shape(1, 1)).store(full<float>(shape(1, 1), shifted_sum));
            if (!zero_fill) { data(coord(nest.index(), 0), shape(m, n)).store(x + after); }
        }
    });
    auto kernel = definition.capture(tensor_shape(count, physical_width), tensor_shape(count, 5));
    expect(kernel.valid());
    auto options = bridge::xir::PlannerOptions{.block_size = std::max(32u, lanes), .reduction_partitions = partitions, .local_lanes = lanes};
    auto shader = compile(device, kernel, {.xir = &options}, {.enable_fast_math = false});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    expect(shader.metadata().realization.find(format("local_lanes={};", lanes)) != string::npos);
    expect(shader.metadata().realization.find("fast_math=false;") != string::npos);
    expect(eq(shader.metadata().dispatch_size.x, static_cast<uint32_t>(count) * lanes));
    constexpr auto pad = size_t{17};
    constexpr auto guard = -731.25f;
    vector<float> data(count * physical_width + 2 * pad, guard), actual(count * 5 + 2 * pad, guard);
    std::fill(actual.begin() + pad, actual.end() - pad, std::numeric_limits<float>::quiet_NaN());
    vector<double> expected_data(count * physical_width), expected_output(count * 5);
    for (int64_t row = 0; row < count; row++) {
        auto sum = row == 0 ? -0.0 : 2.5, product = 2.0, low = 8.0, high = -8.0, shifted_sum = 2.5;
        for (int64_t i = 0; i < width; i++) {
            auto x = zero_fill ? (i == 0 ? 1.0f : 0.0f) : row == 0 ? -0.0f :
                                                                     (i % 3 == 0 ? 1.0f : .5f);
            if (i < physical_width) {
                data[pad + row * physical_width + i] = x;
                expected_data[row * physical_width + i] = zero_fill ? x : x + 9.0;
            }
            sum += x;
            product *= x;
            low = std::min(low, static_cast<double>(x));
            high = std::max(high, static_cast<double>(x));
            shifted_sum += static_cast<double>(x) + 3.0;
        }
        expected_output[row * 5] = sum;
        expected_output[row * 5 + 1] = product;
        expected_output[row * 5 + 2] = low;
        expected_output[row * 5 + 3] = high;
        expected_output[row * 5 + 4] = shifted_sum;
    }
    auto a = device.create_buffer<float>(data.size()), b = device.create_buffer<float>(actual.size());
    auto av = a.view(pad, expected_data.size()), bv = b.view(pad, expected_output.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.copy_from(span{data}) << b.copy_from(span{actual}) << shader(av, bv).dispatch()
           << a.copy_to(span{data}) << b.copy_to(span{actual}) << synchronize();
    auto check_close = [&](span<const float> values, span<const double> expected) {
        auto correct = values.size() == expected.size();
        for (size_t i = 0u; correct && i < values.size(); i++) {
            correct = std::isfinite(values[i]) && std::abs(values[i] - expected[i]) <= 2e-5 + 2e-5 * std::abs(expected[i]);
        }
        expect(correct) << "rows=" << count << " width=" << width << " partitions=" << partitions << " zero_fill=" << zero_fill;
    };
    check_close(span{data}.subspan(pad, expected_data.size()), expected_data);
    check_close(span{actual}.subspan(pad, expected_output.size()), expected_output);
    if (zero_fill) {
        for (int64_t row = 0; row < count; row++) {
            expect(eq(actual[pad + row * 5 + 1], 0.0f));
            expect(eq(actual[pad + row * 5 + 2], 0.0f));
            expect(eq(actual[pad + row * 5 + 4], 24.5f));
            expect(eq(data[pad + row], 1.0f));
        }
    } else {
        // No invented additive identity and no repeated initial accumulator.
        expect(eq(luisa::bit_cast<uint32_t>(actual[pad]), luisa::bit_cast<uint32_t>(-0.0f)));
    }
    for (auto values : {span{data}, span{actual}}) {
        expect(std::all_of(values.begin(), values.begin() + pad, [](float x) { return x == guard; }));
        expect(std::all_of(values.end() - pad, values.end(), [](float x) { return x == guard; }));
    }
}

}// namespace luisa::test::tile_xir
