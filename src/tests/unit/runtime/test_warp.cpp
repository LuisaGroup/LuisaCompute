#include "ut/ut.hpp"
#include "test_device.h"
// Test for warp-level matrix multiplication.
//
// This test demonstrates optimized matrix multiplication using warp-level
// primitives for efficient GPU utilization. The kernel uses:
// - Warp-level collective operations (warp_active_sum)
// - Coalesced memory access patterns
// - Tiled computation for cache efficiency
//
// The implementation targets GPUs supporting Shader Model 6.6 or CUDA.

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/device.h>
#include <luisa/dsl/sugar.h>
#include <algorithm>
#include <random>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_warp_lane_id_multidimensional(Device &device) {
    auto warp_size = device.compute_warp_size();
    constexpr uint block_size_x = 8u;
    expect(warp_size != 0u && warp_size % block_size_x == 0u) << "warp size must be a non-zero multiple of the test block width";
    if (warp_size == 0u || warp_size % block_size_x != 0u) { return; }

    auto dispatch_size_y = 2u * warp_size / block_size_x;
    auto block_thread_count = std::max(32u, 2u * warp_size);
    auto block_size_y = block_thread_count / block_size_x;
    auto thread_count = block_size_x * dispatch_size_y;
    auto lane_ids = device.create_buffer<uint>(thread_count);
    auto shader = device.compile<2>([=](BufferUInt output) noexcept {
        set_block_size(block_size_x, block_size_y, 1u);
        set_warp_size(warp_size);
        auto index = dispatch_id().y * block_size_x + dispatch_id().x;
        output.write(index, warp_lane_id());
    });

    luisa::vector<uint> host_lane_ids(thread_count);
    auto stream = device.create_stream();
    stream << shader(lane_ids).dispatch(block_size_x, dispatch_size_y)
           << lane_ids.copy_to(luisa::span{host_lane_ids})
           << synchronize();

    luisa::vector<uint> lane_counts(warp_size);
    auto all_in_range = true;
    for (auto lane_id : host_lane_ids) {
        if (lane_id < warp_size) {
            lane_counts[lane_id]++;
        } else {
            all_in_range = false;
        }
    }
    auto every_lane_seen_twice = std::all_of(
        lane_counts.cbegin(), lane_counts.cend(),
        [](auto count) noexcept { return count == 2u; });
    expect(all_in_range) << "warp lane IDs must be smaller than the native warp size";
    expect(every_lane_seen_twice) << "two complete multidimensional warps must each contain every lane ID exactly once";
}

void test_warp(Device &device) {
    auto stream = device.create_stream();

    // Helper lambdas for matrix element access in row-major layout
    // buffer[y * size.x + x] stores element at (x, y)
    auto get_matrix = [&](BufferVar<float> const &buffer, UInt2 const &size, UInt2 const &idx) {
        return buffer.read(size.x * idx.y + idx.x);
    };
    auto set_matrix = [&](BufferVar<float> const &buffer, UInt2 const &size, UInt2 const &idx, Float const &value) {
        return buffer.write(size.x * idx.y + idx.x, value);
    };

    // Use the backend's native warp/subgroup width. Vulkan on AMD commonly
    // exposes 64-lane subgroups, while CUDA/DX paths usually use 32.
    auto warp_size = device.compute_warp_size();

    // Warp-level matrix multiplication kernel
    // Computes: result = lhs * rhs where lhs is [M x K] and rhs is [K x N]
    // Each warp computes a tile of the output matrix
    auto mat_mul_kernel = [&](BufferVar<float> lhs_buffer,
                              BufferVar<float> rhs_buffer,
                              BufferVar<float> result_buffer,
                              UInt lhs_row_size) {
        set_block_size(128, 1, 1);
        // Note: Requires Shader Model 6.6 (DirectX) or CUDA
        // The warp size is implementation-defined for modern GPUs
        set_warp_size(warp_size);

        // Calculate matrix dimensions from dispatch size
        UInt2 lhs_matrix_size = make_uint2(lhs_row_size, dispatch_size().y);
        UInt2 rhs_matrix_size = make_uint2(dispatch_size().x / warp_size, lhs_row_size);

        // Each warp processes one output tile
        UInt lhs_y = dispatch_id().x / warp_size;// Row in output
        UInt rhs_x = dispatch_id().y;              // Column in output
        UInt warp_local_id = warp_lane_id();       // Thread index within warp (0-31)

        // Calculate number of tiles along K dimension
        UInt lhs_row_batch_count = (lhs_matrix_size.x + warp_size - 1) / warp_size;
        Float curr_lane_value = 0.f;

        Float local_v;
        // Process K dimension in tiles of warp_size
        for (auto lhs_row_batch : dynamic_range(lhs_row_batch_count)) {
            // Index within current tile
            UInt lhs_x = lhs_row_batch * warp_size + warp_local_id;

            // Load and multiply if within bounds
            $if (lhs_x < lhs_matrix_size.x) {
                local_v = get_matrix(lhs_buffer, lhs_matrix_size, make_uint2(lhs_x, lhs_y));
                local_v *= get_matrix(rhs_buffer, rhs_matrix_size, make_uint2(rhs_x, lhs_x));
            }
            $else {
                local_v = 0.f;
            };

            // Warp-level sum reduction: all 32 threads contribute
            // This is more efficient than shared memory reduction
            curr_lane_value += warp_active_sum(local_v);
        }

        // Only thread 0 in each warp writes the result
        $if (warp_local_id == 0) {
            set_matrix(result_buffer, make_uint2(rhs_matrix_size.x, lhs_matrix_size.y), make_uint2(rhs_x, lhs_y), curr_lane_value);
        };
    };

    // Compile the kernel
    auto mat_mul_shader = device.compile<2>(std::move(mat_mul_kernel));

    // Initialize random data for testing
    std::mt19937 gen{42u};
    std::uniform_real_distribution<float> dist(0.5, 1.5);

    // Matrix dimensions
    constexpr uint k_matrix_size = 256;
    luisa::vector<float> lhs_matrix;
    luisa::vector<float> rhs_matrix;
    luisa::vector<float> result_matrix;
    lhs_matrix.resize(k_matrix_size * k_matrix_size);
    rhs_matrix.resize(k_matrix_size * k_matrix_size);
    result_matrix.resize(k_matrix_size * k_matrix_size);

    // Helper for row-major index calculation
    auto idx = [](auto x, auto y) {
        return y * k_matrix_size + x;
    };

    // Fill matrices with random values
    for (int x = 0; x < k_matrix_size; ++x)
        for (int y = 0; y < k_matrix_size; ++y) {
            lhs_matrix[idx(x, y)] = dist(gen);
            rhs_matrix[idx(x, y)] = dist(gen);
        }

    // Create GPU buffers
    auto lhs_buffer = device.create_buffer<float>(lhs_matrix.size());
    auto rhs_buffer = device.create_buffer<float>(rhs_matrix.size());
    auto result_buffer = device.create_buffer<float>(result_matrix.size());

    // Execute kernel
    stream
        << lhs_buffer.copy_from(luisa::span{lhs_matrix})
        << rhs_buffer.copy_from(luisa::span{rhs_matrix})
        // Dispatch: x dimension accounts for warp grouping, y is matrix rows
        << mat_mul_shader(lhs_buffer, rhs_buffer, result_buffer, k_matrix_size).dispatch(k_matrix_size * warp_size, k_matrix_size)
        << result_buffer.copy_to(luisa::span{result_matrix})
        << synchronize();

    // Host-side validation
    auto mismatch_count = 0u;
    for (int x = 0; x < k_matrix_size; ++x) {
        for (int y = 0; y < k_matrix_size; ++y) {
            float result = 0.f;
            // Standard matrix multiplication: C[y][x] = sum(A[y][k] * B[k][x])
            for (int row = 0; row < k_matrix_size; ++row) {
                result += lhs_matrix[idx(row, y)] * rhs_matrix[idx(x, row)];
            }
            // Validate with tolerance for floating point errors
            if (abs(result - result_matrix[idx(x, y)]) > 1e-2f) {
                if (mismatch_count < 16u) {
                    LUISA_WARNING("Warp matmul mismatch at ({},{}): expected {} got {}",
                                  x, y, result, result_matrix[idx(x, y)]);
                }
                mismatch_count++;
            }
        }
    }
    if (mismatch_count > 16u) {
        LUISA_WARNING("Warp matmul had {} mismatches (only the first 16 are shown).", mismatch_count);
    }
    expect(mismatch_count == 0u) << "warp_matmul_correctness";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_warp_lane_id_multidimensional(device);
    test_warp(device);
}
