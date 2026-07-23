// Cluster Launch Control example.
// Demonstrates work-stealing using NVIDIA Cluster Launch Control (CUDA Blackwell, SM 10.0+).
//
// Usage:
//   xmake run example_cluster_launch_control cuda
//   LUISA_DUMP_SOURCE=1 xmake run example_cluster_launch_control cuda

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/cluster_launch_control.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    luisa::log_level_verbose();

    Context ctx{argv[0]};
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream();

    constexpr auto N = 1024 * 1024u;
    constexpr auto block_size = 256u;
    constexpr auto num_blocks = 64u; // fewer than N/block_size to trigger work-stealing

    // Fill input data
    luisa::vector<float> host(N);
    for (auto i = 0u; i < N; ++i) {
        host[i] = static_cast<float>(i);
    }

    Buffer<float> buf = device.create_buffer<float>(N);
    stream << buf.copy_from(luisa::span{host}) << synchronize();

    // Work-stealing kernel using cluster launch control
    Kernel1D kernel = [&](BufferFloat b, UInt n) noexcept {
        set_block_size(block_size, 1u, 1u);

        // Use the high-level work-stealing helper
        cluster_launch_control_work_stealing_1d([&](Int bx) noexcept {
            auto i = bx * block_size + thread_x();
            $if (i < n) {
                b.write(i, b.read(i) * 2.0f);
            };
        });
    };

    auto shader = device.compile(kernel);
    auto time = Clock{};
    stream << shader(buf, N).dispatch(num_blocks) << synchronize();
    auto elapsed = time.toc();

    // Verify results
    luisa::vector<float> result(N);
    stream << buf.copy_to(luisa::span{result}) << synchronize();

    bool ok = true;
    for (auto i = 0u; i < N; ++i) {
        if (std::abs(result[i] - static_cast<float>(i) * 2.0f) > 1e-4f) {
            LUISA_ERROR("Mismatch at [{}]: got {}, expected {}", i, result[i], static_cast<float>(i) * 2.0f);
            ok = false;
            break;
        }
    }

    if (ok) {
        LUISA_INFO("Cluster launch control work-stealing: PASSED ({} ms)", elapsed);
    }

    return ok ? 0 : 1;
}
