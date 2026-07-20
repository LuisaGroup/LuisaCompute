// Test for warp collectives under a sparse active-lane mask.
// This test covers:
// - active reductions over physical lanes {0, 1, 6}
// - exclusive prefix sum/product over the same irregular mask
// - component-wise vector all-equal voting
// - 16-bit three-component and matrix lane shuffles

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto sparse_lane_count = 3u;
constexpr std::array sparse_lanes{0u, 1u, 6u};
constexpr auto collective_field_count = 12u;

void test_sparse_collectives(Device &device) {
    auto warp_size = device.compute_warp_size();
    expect(warp_size >= 7u) << "sparse warp test requires at least seven lanes";
    if (warp_size < 7u) { return; }

    constexpr auto sentinel = 0xdeadbeefu;
    luisa::vector<uint> output(warp_size * collective_field_count, sentinel);
    auto output_buffer = device.create_buffer<uint>(output.size());

    Kernel1D kernel = [warp_size](BufferUInt output) noexcept {
        set_block_size(warp_size, 1u, 1u);
        set_warp_size(warp_size);
        auto lane = warp_lane_id();
        auto is_sparse_lane = (lane == 0u) | (lane == 1u) | (lane == 6u);
        $if (is_sparse_lane) {
            UInt value = 5u;
            $if (lane == 0u) { value = 2u; };
            $if (lane == 1u) { value = 3u; };
            auto base = lane * collective_field_count;
            output.write(base + 0u, warp_active_sum(value));
            output.write(base + 1u, warp_active_product(value));
            output.write(base + 2u, warp_active_min(value));
            output.write(base + 3u, warp_active_max(value));
            output.write(base + 4u, warp_active_bit_and(value));
            output.write(base + 5u, warp_active_bit_or(value));
            output.write(base + 6u, warp_active_bit_xor(value));
            output.write(base + 7u, warp_prefix_sum(value));
            output.write(base + 8u, warp_prefix_product(value));
            auto all_equal = warp_active_all_equal(
                make_uint3(42u, lane & 1u, 7u));
            output.write(base + 9u, ite(all_equal.x, 1u, 0u));
            output.write(base + 10u, ite(all_equal.y, 1u, 0u));
            output.write(base + 11u, ite(all_equal.z, 1u, 0u));
        };
    };
    auto shader = device.compile(kernel);
    auto stream = device.create_stream();
    stream << output_buffer.copy_from(luisa::span{output})
           << shader(output_buffer).dispatch(warp_size)
           << output_buffer.copy_to(luisa::span{output})
           << synchronize();

    constexpr std::array expected_prefix_sum{0u, 2u, 5u};
    constexpr std::array expected_prefix_product{1u, 2u, 6u};
    for (auto i = 0u; i < sparse_lane_count; i++) {
        auto lane = sparse_lanes[i];
        auto base = lane * collective_field_count;
        expect(output[base + 0u] == 10u) << "sparse warp sum mismatch at lane " << lane;
        expect(output[base + 1u] == 30u) << "sparse warp product mismatch at lane " << lane;
        expect(output[base + 2u] == 2u) << "sparse warp min mismatch at lane " << lane;
        expect(output[base + 3u] == 5u) << "sparse warp max mismatch at lane " << lane;
        expect(output[base + 4u] == 0u) << "sparse warp bit-and mismatch at lane " << lane;
        expect(output[base + 5u] == 7u) << "sparse warp bit-or mismatch at lane " << lane;
        expect(output[base + 6u] == 4u) << "sparse warp bit-xor mismatch at lane " << lane;
        expect(output[base + 7u] == expected_prefix_sum[i]) << "sparse warp prefix-sum mismatch at lane " << lane;
        expect(output[base + 8u] == expected_prefix_product[i]) << "sparse warp prefix-product mismatch at lane " << lane;
        expect(output[base + 9u] == 1u) << "sparse warp all-equal x mismatch at lane " << lane;
        expect(output[base + 10u] == 0u) << "sparse warp all-equal y mismatch at lane " << lane;
        expect(output[base + 11u] == 1u) << "sparse warp all-equal z mismatch at lane " << lane;
    }
    for (auto lane = 0u; lane < warp_size; lane++) {
        if (lane == sparse_lanes[0] || lane == sparse_lanes[1] || lane == sparse_lanes[2]) { continue; }
        for (auto field = 0u; field < collective_field_count; field++) {
            expect(output[lane * collective_field_count + field] == sentinel)
                << "inactive lane unexpectedly wrote a collective result";
        }
    }
}

void test_lane_shuffle_packing(Device &device) {
    auto warp_size = device.compute_warp_size();
    if (warp_size < 7u) { return; }

    auto short_buffer = device.create_buffer<short3>(1u);
    auto ushort_buffer = device.create_buffer<ushort3>(1u);
    auto half_buffer = device.create_buffer<half3>(1u);
    auto matrix_buffer = device.create_buffer<float3x3>(1u);

    Kernel1D kernel = [warp_size](BufferShort3 short_output,
                                  BufferUShort3 ushort_output,
                                  BufferHalf3 half_output,
                                  BufferFloat3x3 matrix_output) noexcept {
        set_block_size(warp_size, 1u, 1u);
        set_warp_size(warp_size);
        auto lane = warp_lane_id();
        auto is_sparse_lane = (lane == 0u) | (lane == 1u) | (lane == 6u);
        $if (is_sparse_lane) {
            auto float_lane = cast<float>(lane);
            Short3 short_value = make_short3(
                cast<short>(lane + 11u),
                cast<short>(lane + 12u),
                cast<short>(lane + 13u));
            UShort3 ushort_value = make_ushort3(
                cast<ushort>(lane + 21u),
                cast<ushort>(lane + 22u),
                cast<ushort>(lane + 23u));
            Half3 half_value = make_half3(
                cast<half>(float_lane + 1.5f),
                cast<half>(float_lane + 2.5f),
                cast<half>(float_lane + 3.5f));
            Float3x3 matrix_value = make_float3x3(
                make_float3(float_lane + 1.0f, float_lane + 2.0f, float_lane + 3.0f),
                make_float3(float_lane + 4.0f, float_lane + 5.0f, float_lane + 6.0f),
                make_float3(float_lane + 7.0f, float_lane + 8.0f, float_lane + 9.0f));
            auto short_first = warp_read_first_active_lane(short_value);
            auto ushort_first = warp_read_first_active_lane(ushort_value);
            auto half_first = warp_read_first_active_lane(half_value);
            auto matrix_first = warp_read_first_active_lane(matrix_value);
            $if (lane == 6u) {
                short_output.write(0u, short_first);
                ushort_output.write(0u, ushort_first);
                half_output.write(0u, half_first);
                matrix_output.write(0u, matrix_first);
            };
        };
    };
    auto shader = device.compile(kernel);

    std::array<short3, 1u> short_output{};
    std::array<ushort3, 1u> ushort_output{};
    std::array<half3, 1u> half_output{};
    std::array<float3x3, 1u> matrix_output{};
    auto stream = device.create_stream();
    stream << shader(short_buffer, ushort_buffer, half_buffer, matrix_buffer).dispatch(warp_size)
           << short_buffer.copy_to(luisa::span{short_output})
           << ushort_buffer.copy_to(luisa::span{ushort_output})
           << half_buffer.copy_to(luisa::span{half_output})
           << matrix_buffer.copy_to(luisa::span{matrix_output})
           << synchronize();

    expect(short_output[0].x == 11 && short_output[0].y == 12 && short_output[0].z == 13)
        << "short3 lane shuffle must preserve all 48 bits";
    expect(ushort_output[0].x == 21u && ushort_output[0].y == 22u && ushort_output[0].z == 23u)
        << "ushort3 lane shuffle must preserve all 48 bits";
    expect(std::abs(static_cast<float>(half_output[0].x) - 1.5f) < 1e-3f &&
           std::abs(static_cast<float>(half_output[0].y) - 2.5f) < 1e-3f &&
           std::abs(static_cast<float>(half_output[0].z) - 3.5f) < 1e-3f)
        << "half3 lane shuffle must preserve all 48 bits";
    for (auto column = 0u; column < 3u; column++) {
        for (auto row = 0u; row < 3u; row++) {
            auto expected = static_cast<float>(column * 3u + row + 1u);
            expect(std::abs(matrix_output[0][column][row] - expected) < 1e-6f)
                << "matrix lane shuffle mismatch at column " << column << ", row " << row;
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    test_sparse_collectives(dc->device);
    test_lane_shuffle_packing(dc->device);
}
