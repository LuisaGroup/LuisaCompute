#include "ut/ut.hpp"
#include "test_device.h"

#include <cmath>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint kWarpSize = 32u;
constexpr uint kGroupLanes = 8u;
constexpr uint kGroupsPerWarp = kWarpSize / kGroupLanes;
constexpr uint kVecSize = 2u;
constexpr uint kHeadDim = kGroupLanes * kVecSize;

void test_grouped_float2_prefix_extract(Device &device) {
    static constexpr uint output_count = 64u;
    static constexpr uint input_count = output_count * kHeadDim;
    luisa::vector<float> input(input_count);
    for (auto i = 0u; i < input_count; i++) {
        input[i] = std::sin(static_cast<float>(i) * 0.17f + 0.25f) * 0.5f + 0.75f;
    }

    auto input_buffer = device.create_buffer<float>(input_count);
    auto output_buffer = device.create_buffer<float>(output_count);
    auto stream = device.create_stream();

    Kernel1D kernel = [](BufferFloat in, BufferFloat out) noexcept {
        set_block_size(256u, 1u, 1u);
        set_warp_size(kWarpSize);

        auto idx = dispatch_id().x;
        auto lane = warp_lane_id();
        auto group_id = lane / kGroupLanes;
        auto group_lane = lane % kGroupLanes;
        auto output_idx = (idx / kWarpSize) * kGroupsPerWarp + group_id;

        auto d0 = group_lane * kVecSize;
        auto d1 = d0 + 1u;
        auto base = output_idx * kHeadDim;
        auto value = make_float2(in.read(base + d0),
                                 in.read(base + d1));

        auto prefix = warp_prefix_sum(value);
        auto inclusive = prefix + value;
        auto last_lane = group_id * kGroupLanes + (kGroupLanes - 1u);
        auto incl_last = warp_read_lane(inclusive, last_lane);
        auto prev_last = ite(group_id == 0u, 0u, last_lane - kGroupLanes);
        auto prev_incl = warp_read_lane(inclusive, prev_last);
        auto total = incl_last - ite(group_id == 0u, make_float2(0.0f), prev_incl);

        $if (group_lane == 0u) {
            out.write(output_idx, total.x + total.y);
        };
    };

    auto shader = device.compile(kernel);
    stream << input_buffer.copy_from(input.data())
           << shader(input_buffer, output_buffer).dispatch(output_count * kGroupLanes)
           << synchronize();

    luisa::vector<float> output(output_count);
    stream << output_buffer.copy_to(output.data()) << synchronize();

    auto ok = true;
    for (auto i = 0u; i < output_count; i++) {
        auto expected = 0.0f;
        for (auto d = 0u; d < kHeadDim; d++) {
            expected += input[i * kHeadDim + d];
        }
        auto diff = std::abs(output[i] - expected);
        if (diff > 1e-4f) {
            LUISA_WARNING("MHA grouped float2 reduction mismatch at {}: got {}, expected {}, diff {}",
                          i, output[i], expected, diff);
            ok = false;
            break;
        }
    }
    expect(ok) << "grouped float2 warp_prefix_sum/warp_read_lane should match CPU sums";
}

void test_grouped_softmax(Device &device) {
    static constexpr uint row_count = 32u;
    static constexpr uint row_size = kGroupLanes;
    static constexpr uint value_count = row_count * row_size;
    luisa::vector<float> input(value_count);
    for (auto i = 0u; i < value_count; i++) {
        input[i] = std::cos(static_cast<float>(i) * 0.11f + 1.5f);
    }

    auto input_buffer = device.create_buffer<float>(value_count);
    auto output_buffer = device.create_buffer<float>(value_count);
    auto stream = device.create_stream();

    Kernel1D kernel = [](BufferFloat in, BufferFloat out) noexcept {
        set_block_size(256u, 1u, 1u);
        set_warp_size(kWarpSize);

        auto idx = dispatch_id().x;
        auto lane = warp_lane_id();
        auto group_id = lane / kGroupLanes;
        auto group_lane = lane % kGroupLanes;
        auto row = (idx / kWarpSize) * kGroupsPerWarp + group_id;
        auto base = row * kGroupLanes;

        auto m = in.read(base + group_lane);
        m = max(m, warp_read_lane(m, lane ^ 4u));
        m = max(m, warp_read_lane(m, lane ^ 2u));
        m = max(m, warp_read_lane(m, lane ^ 1u));

        auto e = exp(in.read(base + group_lane) - m);
        auto prefix = warp_prefix_sum(e);
        auto inclusive = prefix + e;
        auto last_lane = group_id * kGroupLanes + (kGroupLanes - 1u);
        auto incl_last = warp_read_lane(inclusive, last_lane);
        auto prev_last = ite(group_id == 0u, 0u, last_lane - kGroupLanes);
        auto prev_incl = warp_read_lane(inclusive, prev_last);
        auto sum = incl_last - ite(group_id == 0u, 0.0f, prev_incl);

        out.write(base + group_lane, e / sum);
    };

    auto shader = device.compile(kernel);
    stream << input_buffer.copy_from(input.data())
           << shader(input_buffer, output_buffer).dispatch(row_count * row_size)
           << synchronize();

    luisa::vector<float> output(value_count);
    stream << output_buffer.copy_to(output.data()) << synchronize();

    auto ok = true;
    for (auto row = 0u; row < row_count; row++) {
        auto base = row * row_size;
        auto max_value = input[base];
        for (auto i = 1u; i < row_size; i++) {
            max_value = std::max(max_value, input[base + i]);
        }
        auto sum = 0.0f;
        for (auto i = 0u; i < row_size; i++) {
            sum += std::exp(input[base + i] - max_value);
        }
        for (auto i = 0u; i < row_size; i++) {
            auto expected = std::exp(input[base + i] - max_value) / sum;
            auto diff = std::abs(output[base + i] - expected);
            if (diff > 1e-4f) {
                LUISA_WARNING("MHA grouped softmax mismatch at row {}, lane {}: got {}, expected {}, diff {}",
                              row, i, output[base + i], expected, diff);
                ok = false;
                break;
            }
        }
        if (!ok) { break; }
    }
    expect(ok) << "grouped scalar warp_read_lane softmax should match CPU reference";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "mha_grouped_float2_prefix_extract"_test = [&] {
        test_grouped_float2_prefix_extract(dc->device);
    };
    "mha_grouped_softmax"_test = [&] {
        test_grouped_softmax(dc->device);
    };
    return 0;
}
