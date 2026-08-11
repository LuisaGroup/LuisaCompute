// Test for async copy (CUDA cp.async / LDGSTS, CC 8.0+).
// Features tested:
// - basic global -> shared -> global copy with pipeline commit/wait
// - multi-stage (two-stage) prefetch pipeline
// - 16-byte (float4) copy
//
// Only the CUDA backend implements the async-copy pipeline ops
// (lc_pipeline_memcpy_async / lc_pipeline_commit / lc_pipeline_wait_prior,
// backed by cp.async PTX instructions). On other backends the tests are
// skipped.

#include "ut/ut.hpp"
#include "test_device.h"

#include <cmath>
#include <vector>

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_async_copy_basic(Device &device) {
    constexpr auto N = 1024u;
    constexpr auto block_size = 256u;

    luisa::vector<float> host_src(N);
    for (auto i = 0u; i < N; ++i) {
        host_src[i] = static_cast<float>(i);
    }

    Buffer<float> src = device.create_buffer<float>(N);
    Buffer<float> dst = device.create_buffer<float>(N);
    Stream stream = device.create_stream();
    stream << src.copy_from(luisa::span{host_src}) << synchronize();

    Kernel1D kernel = [&](BufferFloat src_buf, BufferFloat dst_buf) noexcept {
        set_block_size(block_size, 1u, 1u);

        $shared<float> tile{block_size};
        $uint tid = thread_x();
        $uint base = block_x() * block_size + tid;

        // Async copy one float (4 bytes) from global to shared. dst is the
        // shared-memory lvalue; src is the global source address.
        async_copy(1u, tile[tid],
                   src_buf.device_address() + cast<ulong>(base * 4u),
                   4u, 1u, 4u, 0u);
        pipeline_commit();
        pipeline_wait_prior(0u);

        sync_block();  // make the copied data visible to the whole block

        $float val = tile[tid];
        $if (base < N) {
            dst_buf.write(base, val);
        };
    };

    auto shader = device.compile(kernel);
    // dispatch(N) launches N total threads (ceil(N / block_size) blocks).
    stream << shader(src, dst).dispatch(N) << synchronize();

    luisa::vector<float> result(N);
    stream << dst.copy_to(luisa::span{result}) << synchronize();

    bool all_correct = true;
    for (auto i = 0u; i < N; ++i) {
        if (std::abs(result[i] - static_cast<float>(i)) > 1e-4f) {
            LUISA_WARNING("async_copy_basic mismatch at [{}]: got {}, expected {}",
                          i, result[i], i);
            all_correct = false;
            break;
        }
    }
    expect(all_correct) << "basic async copy should copy src to shared and back";
}

void test_async_copy_two_stage(Device &device) {
    constexpr auto N = 2048u;
    constexpr auto block_size = 256u;
    constexpr auto elements_per_block = 2u * block_size;
    constexpr auto num_blocks = N / elements_per_block;
    luisa::vector<float> host_src(N);
    for (auto i = 0u; i < N; ++i) {
        host_src[i] = static_cast<float>(i);
    }

    Buffer<float> src = device.create_buffer<float>(N);
    Buffer<float> dst = device.create_buffer<float>(N);
    Stream stream = device.create_stream();
    stream << src.copy_from(luisa::span{host_src}) << synchronize();

    Kernel1D kernel = [&](BufferFloat src_buf, BufferFloat dst_buf) noexcept {
        set_block_size(block_size, 1u, 1u);

        $shared<float> tile_a{block_size};
        $shared<float> tile_b{block_size};
        $uint tid = thread_x();

        auto src_base = src_buf.device_address();

        // Stage 1: prefetch tile A.
        $uint base_a = block_x() * elements_per_block + tid;
        async_copy(1u, tile_a[tid],
                   src_base + cast<ulong>(base_a * 4u), 4u, 1u, 4u, 0u);
        pipeline_commit();

        // Stage 2: prefetch tile B.
        $uint base_b = base_a + block_size;
        async_copy(1u, tile_b[tid],
                   src_base + cast<ulong>(base_b * 4u), 4u, 1u, 4u, 0u);
        pipeline_commit();

        // Wait for tile A (one stage still in flight).
        pipeline_wait_prior(1u);
        sync_block();
        $float val_a = tile_a[tid];
        $if (base_a < N) {
            dst_buf.write(base_a, val_a);
        };

        // Wait for tile B.
        pipeline_wait_prior(0u);
        sync_block();
        $float val_b = tile_b[tid];
        $if (base_b < N) {
            dst_buf.write(base_b, val_b);
        };
    };

    auto shader = device.compile(kernel);
    // Each thread handles two elements (one per tile), so dispatch
    // num_blocks * block_size total threads.
    stream << shader(src, dst).dispatch(num_blocks * block_size) << synchronize();

    luisa::vector<float> result(N);
    stream << dst.copy_to(luisa::span{result}) << synchronize();

    bool all_correct = true;
    for (auto i = 0u; i < N; ++i) {
        if (std::abs(result[i] - static_cast<float>(i)) > 1e-4f) {
            LUISA_WARNING("async_copy_two_stage mismatch at [{}]: got {}, expected {}",
                          i, result[i], i);
            all_correct = false;
            break;
        }
    }
    expect(all_correct) << "two-stage prefetch should copy both tiles correctly";
}

void test_async_copy_float4(Device &device) {
    constexpr auto N4 = 256u;          // float4 count
    constexpr auto block_size = 64u;

    luisa::vector<float4> host_src(N4);
    for (auto i = 0u; i < N4; ++i) {
        host_src[i] = make_float4(
            static_cast<float>(4 * i),
            static_cast<float>(4 * i + 1),
            static_cast<float>(4 * i + 2),
            static_cast<float>(4 * i + 3));
    }

    Buffer<float4> src = device.create_buffer<float4>(N4);
    Buffer<float4> dst = device.create_buffer<float4>(N4);
    Stream stream = device.create_stream();
    stream << src.copy_from(luisa::span{host_src}) << synchronize();

    Kernel1D kernel = [&](BufferVar<float4> src_buf, BufferVar<float4> dst_buf) noexcept {
        set_block_size(block_size, 1u, 1u);

        $shared<float4> tile{block_size};
        $uint tid = thread_x();
        $uint base = block_x() * block_size + tid;

        // Async copy one float4 (16 bytes) from global to shared.
        async_copy(1u, tile[tid],
                   src_buf.device_address() + cast<ulong>(base * 16u),
                   16u, 1u, 16u, 0u);
        pipeline_commit();
        pipeline_wait_prior(0u);

        sync_block();

        $float4 val = tile[tid];
        $if (base < N4) {
            dst_buf.write(base, val);
        };
    };

    auto shader = device.compile(kernel);
    // dispatch(N4) launches N4 total threads (ceil(N4 / block_size) blocks).
    stream << shader(src, dst).dispatch(N4) << synchronize();

    luisa::vector<float4> result(N4);
    stream << dst.copy_to(luisa::span{result}) << synchronize();

    bool all_correct = true;
    for (auto i = 0u; i < N4; ++i) {
        auto expected = make_float4(
            static_cast<float>(4 * i),
            static_cast<float>(4 * i + 1),
            static_cast<float>(4 * i + 2),
            static_cast<float>(4 * i + 3));
        if (std::abs(result[i].x - expected.x) > 1e-4f ||
            std::abs(result[i].y - expected.y) > 1e-4f ||
            std::abs(result[i].z - expected.z) > 1e-4f ||
            std::abs(result[i].w - expected.w) > 1e-4f) {
            LUISA_WARNING("async_copy_float4 mismatch at [{}]: got {}, expected {}",
                          i, result[i], expected);
            all_correct = false;
            break;
        }
    }
    expect(all_correct) << "16-byte async copy should copy float4 values correctly";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    // argv[1] selects the runtime backend; it is not a Boost.UT name filter.
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    if (device.backend_name() != "cuda") {
        LUISA_INFO("Async copy is only supported on the CUDA backend (got '{}'); skipping.",
                   device.backend_name());
        return 0;
    }
    log_level_verbose();
    test_async_copy_basic(device);
    test_async_copy_two_stage(device);
    test_async_copy_float4(device);
}
