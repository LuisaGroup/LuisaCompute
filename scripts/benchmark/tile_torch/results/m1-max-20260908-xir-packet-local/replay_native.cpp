// Diagnostic replay of verified generated entries, not another implementation
// of RMSNorm. The caller owns all arguments, guards, and numerical validation.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

#include "backends/simd/llvm/llvm_schedule_codegen.h"

using namespace luisa::compute::simd;

extern "C" int replay_native(
    void *address, uint32_t abi, uint32_t rows, uint32_t columns,
    uint32_t block_size, uint32_t local_lanes, const float *x, const float *gamma,
    float *partial, float *output, size_t workspace_bytes,
    uint32_t sample_count, double *samples, uint64_t *repetitions) {
    if (!address || !rows || !columns || !block_size || !sample_count ||
        !local_lanes || local_lanes > 16u || rows > UINT32_MAX / local_lanes ||
        abi > 1u || workspace_bytes > simd_max_private_workspace_bytes) { return 1; }
    using TileEntry = void (*)(const void *, void *, SIMDPacketLaunchConfig *, uint32_t);
    using TorchEntry = void (*)(const float *, const float *, float *, float *);
    auto tile_entry = reinterpret_cast<TileEntry>(address);
    auto torch_entry = reinterpret_cast<TorchEntry>(address);
    auto count = static_cast<size_t>(rows) * columns;
    std::array<SIMDHostBufferView, 4> arguments{{{const_cast<float *>(x), count * sizeof(float)},
                                                 {const_cast<float *>(gamma), columns * sizeof(float)},
                                                 {const_cast<float *>(gamma), columns * sizeof(float)},
                                                 {output, count * sizeof(float)}}};
    struct alignas(64) Chunk {
        std::byte bytes[64];
    };
    std::vector<Chunk> workspace((workspace_bytes + 63u) / 64u);
    SIMDPacketLaunchConfig launch{};
    launch.dispatch_size[0] = rows * local_lanes;
    launch.dispatch_size[1] = launch.dispatch_size[2] = 1u;
    launch.block_size[0] = block_size;
    launch.grid_size[0] = (launch.dispatch_size[0] - 1u) / block_size + 1u;
    launch.private_workspace = workspace.data();
    auto invoke = [&] {
        if (abi == 0u) {
            // This tiny record reset is inside the measured native-entry
            // boundary. No Runtime, thread pool, Python, or allocation runs.
            launch.block_id[0] = launch.block_id[1] = launch.block_id[2] = 0u;
            launch.thread_index = 0u;
            tile_entry(arguments.data(), nullptr, &launch, launch.grid_size[0]);
        } else {
            torch_entry(x, gamma, partial, output);
        }
    };
    using Clock = std::chrono::steady_clock;
    auto measure = [&](uint64_t repeat) {
        auto start = Clock::now();
        for (uint64_t i = 0; i < repeat; i++) { invoke(); }
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    };
    auto start = Clock::now();
    do { invoke(); } while (Clock::now() - start < std::chrono::milliseconds{100});
    uint64_t repeat = 1u;
    while (measure(repeat) < 30000.0 && repeat < 1048576u) { repeat *= 2u; }
    for (uint32_t i = 0; i < sample_count; i++) { samples[i] = measure(repeat) / repeat; }
    *repetitions = repeat;
    return 0;
}
