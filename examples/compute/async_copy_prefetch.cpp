// Async Copy (LDGSTS) prefetch example.
// Demonstrates multi-stage data prefetching using CUDA cp.async + pipeline
// primitives. Requires CUDA CC 8.0+.
//
// Usage:
//   xmake run example_async_copy_prefetch cuda
//   LUISA_DUMP_SOURCE=1 xmake run example_async_copy_prefetch cuda

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    luisa::log_level_verbose();

    Context ctx{argv[0]};
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream();

    constexpr auto N = 2048u;
    constexpr auto block_size = 256u;

    // Fill source data
    luisa::vector<float> host_src(N);
    for (auto i = 0u; i < N; ++i) {
        host_src[i] = static_cast<float>(i);
    }

    Buffer<float> src = device.create_buffer<float>(N);
    Buffer<float> dst = device.create_buffer<float>(N);
    stream << src.copy_from(luisa::span{host_src}) << synchronize();

    // Two-stage prefetch kernel
    Kernel1D kernel = [&](BufferFloat src_buf, BufferFloat dst_buf) noexcept {
        set_block_size(block_size, 1u, 1u);

        $shared<float> tile_a{block_size};
        $shared<float> tile_b{block_size};
        $uint tid = thread_x();

        auto src_base = src_buf.device_address();

        // Stage 1: Prefetch first tile into tile_a
        $uint base_a = block_x() * 2u * block_size + tid;
        async_copy(1u, tile_a[tid],
                   src_base + cast<ulong>(base_a * 4u), 4u, 1u, 4u, 0u);
        pipeline_commit();

        // Stage 2: Prefetch second tile into tile_b
        $uint base_b = base_a + block_size;
        async_copy(1u, tile_b[tid],
                   src_base + cast<ulong>(base_b * 4u), 4u, 1u, 4u, 0u);
        pipeline_commit();

        // Wait for stage 1 to complete (1 prior stage still in flight)
        pipeline_wait_prior(1u);
        sync_block();

        // Process tile_a
        $float val = tile_a[tid];
        $uint gid_a = block_x() * 2u * block_size + tid;
        $if (gid_a < N) {
            dst_buf.write(gid_a, val * 2.0f);
        };

        // Wait for stage 2 to complete
        pipeline_wait_prior(0u);
        sync_block();

        // Process tile_b
        val = tile_b[tid];
        $uint gid_b = gid_a + block_size;
        $if (gid_b < N) {
            dst_buf.write(gid_b, val * 2.0f);
        };
    };

    auto shader = device.compile(kernel);
    auto num_blocks = (N + 2u * block_size - 1u) / (2u * block_size);

    auto time = Clock{};
    // Each thread handles two elements (one per tile), so dispatch
    // num_blocks * block_size total threads.
    stream << shader(src, dst).dispatch(num_blocks * block_size) << synchronize();
    auto elapsed = time.toc();

    // Verify results
    luisa::vector<float> result(N);
    stream << dst.copy_to(luisa::span{result}) << synchronize();

    bool ok = true;
    for (auto i = 0u; i < N; ++i) {
        auto expected = static_cast<float>(i) * 2.0f;
        if (std::abs(result[i] - expected) > 1e-4f) {
            LUISA_ERROR("Mismatch at [{}]: got {}, expected {}", i, result[i], expected);
            ok = false;
            break;
        }
    }

    if (ok) {
        LUISA_INFO("Async copy two-stage prefetch: PASSED ({} ms)", elapsed);
    }

    return ok ? 0 : 1;
}
