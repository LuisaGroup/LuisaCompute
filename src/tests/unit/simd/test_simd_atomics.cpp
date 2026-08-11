#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <cstring>

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    Context context{argc > 0 ? argv[0] : ""};
    DeviceConfig config{};
    config.extension =
        luisa::make_unique<SIMDDeviceConfigExt>(8u, 4u);
    auto device = context.create_device("simd", &config);

    // Exercise conflicting atomics across many concurrently scheduled blocks
    // and a non-divisible dispatch tail.
    constexpr auto thread_count = 32u * 17u + 7u;
    auto counter = device.create_buffer<uint>(1u);
    auto old_values = device.create_buffer<uint>(thread_count);
    auto filtered_counter = device.create_buffer<uint>(1u);
    auto filtered_old_values = device.create_buffer<uint>(thread_count);
    auto winner = device.create_buffer<uint>(1u);
    auto compare_old_values = device.create_buffer<uint>(thread_count);
    auto vector_value = device.create_buffer<float3>(1u);
    auto matrix_value = device.create_buffer<float2x2>(1u);
    using NestedArray = std::array<std::array<float4, 3u>, 5u>;
    auto nested_value = device.create_buffer<NestedArray>(1u);

    Kernel1D kernel = [](BufferUInt counter_buffer,
                         BufferUInt old_buffer,
                         BufferUInt filtered_counter_buffer,
                         BufferUInt filtered_old_buffer,
                         BufferUInt winner_buffer,
                         BufferUInt compare_old_buffer,
                         BufferFloat3 vector_buffer,
                         BufferFloat2x2 matrix_buffer,
                         BufferVar<NestedArray> nested_buffer) noexcept {
        set_block_size(32u, 1u, 1u);
        auto tid = dispatch_x();
        auto old = counter_buffer.atomic(0u).fetch_add(1u);
        old_buffer.write(tid, old);

        filtered_old_buffer.write(tid, ~0u);
        $if ((tid % 3u) != 1u) {
            auto filtered_old =
                filtered_counter_buffer.atomic(0u).fetch_add(1u);
            filtered_old_buffer.write(tid, filtered_old);
        };

        auto compare_old = winner_buffer.atomic(0u).compare_exchange(
            0u, tid + 1u);
        compare_old_buffer.write(tid, compare_old);

        vector_buffer.atomic(0u).y.fetch_add(1.0f);
        matrix_buffer.atomic(0u)[1u].x.fetch_add(1.0f);
        nested_buffer.atomic(0u)[1u][2u][3u].fetch_add(1.0f);
    };

    uint zero = 0u;
    float3 vector_zero = make_float3(0.0f);
    auto matrix_zero = float2x2::fill(0.0f);
    NestedArray nested_zero{};
    std::memset(&nested_zero, 0, sizeof(nested_zero));
    auto shader = device.compile(kernel);
    auto stream = device.create_stream();
    stream << counter.copy_from(luisa::span{&zero, 1u})
           << filtered_counter.copy_from(luisa::span{&zero, 1u})
           << winner.copy_from(luisa::span{&zero, 1u})
           << vector_value.copy_from(luisa::span{&vector_zero, 1u})
           << matrix_value.copy_from(luisa::span{&matrix_zero, 1u})
           << nested_value.copy_from(luisa::span{&nested_zero, 1u})
           << shader(counter, old_values,
                     filtered_counter, filtered_old_values,
                     winner, compare_old_values,
                     vector_value, matrix_value, nested_value)
                  .dispatch(thread_count)
           << synchronize();

    uint final_counter = 0u;
    uint final_filtered_counter = 0u;
    uint final_winner = 0u;
    std::array<uint, thread_count> old{};
    std::array<uint, thread_count> filtered_old{};
    std::array<uint, thread_count> compare_old{};
    float3 final_vector{};
    float2x2 final_matrix{};
    NestedArray final_nested{};
    stream << counter.copy_to(luisa::span{&final_counter, 1u})
           << filtered_counter.copy_to(
                  luisa::span{&final_filtered_counter, 1u})
           << winner.copy_to(luisa::span{&final_winner, 1u})
           << old_values.copy_to(luisa::span{old})
           << filtered_old_values.copy_to(luisa::span{filtered_old})
           << compare_old_values.copy_to(luisa::span{compare_old})
           << vector_value.copy_to(luisa::span{&final_vector, 1u})
           << matrix_value.copy_to(luisa::span{&final_matrix, 1u})
           << nested_value.copy_to(luisa::span{&final_nested, 1u})
           << synchronize();

    expect(final_counter == thread_count)
        << "conflicting atomic increments must not be lost";
    std::sort(old.begin(), old.end());
    for (auto i = 0u; i < thread_count; i++) {
        expect(old[i] == i)
            << "fetch_add must return every old value exactly once";
    }

    constexpr auto filtered_count = thread_count * 2u / 3u;
    expect(final_filtered_counter == filtered_count)
        << "inactive lanes must not execute atomics";
    luisa::vector<uint> active_filtered_old;
    for (auto tid = 0u; tid < thread_count; tid++) {
        if (tid % 3u == 1u) {
            expect(filtered_old[tid] == ~0u)
                << "inactive lanes must retain their sentinel";
        } else {
            active_filtered_old.emplace_back(filtered_old[tid]);
        }
    }
    std::sort(active_filtered_old.begin(), active_filtered_old.end());
    for (auto i = 0u; i < active_filtered_old.size(); i++) {
        expect(active_filtered_old[i] == i)
            << "predicated atomics must return a dense old-value sequence";
    }

    auto winning_lane_count = 0u;
    for (auto value : compare_old) {
        if (value == 0u) {
            winning_lane_count++;
        } else {
            expect(value == final_winner)
                << "compare-exchange losers must observe the winner";
        }
    }
    expect(winning_lane_count == 1u)
        << "exactly one compare-exchange lane must win";
    expect(final_winner >= 1u && final_winner <= thread_count)
        << "compare-exchange must publish a valid winner";

    expect(final_vector.x == 0.0f &&
           final_vector.y == static_cast<float>(thread_count) &&
           final_vector.z == 0.0f)
        << "vector-component atomic offset must select only .y";
    expect(final_matrix.cols[0].x == 0.0f &&
           final_matrix.cols[0].y == 0.0f &&
           final_matrix.cols[1].x == static_cast<float>(thread_count) &&
           final_matrix.cols[1].y == 0.0f)
        << "matrix-component atomic offset must select [1].x";
    expect(final_nested[1u][2u].w == static_cast<float>(thread_count))
        << "nested array atomic offset must select [1][2][3]";
    expect(final_nested[0u][0u].x == 0.0f &&
           final_nested[1u][2u].x == 0.0f)
        << "nested aggregate atomics must preserve neighboring leaves";
}
