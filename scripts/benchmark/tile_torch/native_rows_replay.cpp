// Replay actual generated entries. This contains no operator implementation.
// Python validates the emitted ABI, buffer views, full outputs and guards.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/simd/llvm/llvm_schedule_codegen.h"

using namespace luisa::compute::simd;

namespace {

struct NativeContext {
    void *address;
    void *const *arguments;
};

using Invoke = void (*)(void *);

template<uint32_t Mask, size_t Index>
using NativeArgument = std::conditional_t<(Mask & (1u << Index)) != 0u, const float *, float *>;

template<uint32_t Mask, size_t... I>
void invoke_native_impl(NativeContext &context, std::index_sequence<I...>) {
    using Entry = void (*)(NativeArgument<Mask, I>...);
    auto entry = reinterpret_cast<Entry>(context.address);
    entry(static_cast<NativeArgument<Mask, I>>(context.arguments[I])...);
}

template<size_t Count, uint32_t Mask>
void invoke_native(void *opaque) {
    invoke_native_impl<Mask>(*static_cast<NativeContext *>(opaque), std::make_index_sequence<Count>{});
}

template<size_t Count, uint32_t Mask = 0u>
Invoke select_native(uint32_t mask) {
    if (mask == Mask) { return invoke_native<Count, Mask>; }
    if constexpr (Mask + 1u < (1u << Count)) { return select_native<Count, Mask + 1u>(mask); }
    return nullptr;
}

using TileEntry = void (*)(const void *, void *, SIMDPacketLaunchConfig *, uint32_t);

struct TileContext {
    TileEntry entry;
    std::array<SIMDHostBufferView, 4> arguments;
    SIMDPacketLaunchConfig launch;
    uint32_t packet_width;
};

void invoke_blocks(void *opaque) {
    auto &context = *static_cast<TileContext *>(opaque);
    auto &launch = context.launch;
    launch.block_id[0] = launch.block_id[1] = launch.block_id[2] = 0u;
    launch.thread_index = 0u;
    context.entry(context.arguments.data(), nullptr, &launch, launch.grid_size[0]);
}

void invoke_packets(void *opaque) {
    auto &context = *static_cast<TileContext *>(opaque);
    auto &launch = context.launch;
    for (auto block = 0u; block < launch.grid_size[0]; block++) {
        launch.block_id[0] = block;
        launch.block_id[1] = launch.block_id[2] = 0u;
        launch.thread_index = 0u;
        context.entry(context.arguments.data(), nullptr, &launch, launch.block_size[0] / context.packet_width);
    }
}

}// namespace

extern "C" int replay_native_rows(
    void *address, uint32_t abi, uint32_t count, uint32_t const_mask,
    void *const *arguments, const size_t *argument_bytes,
    uint32_t rows, uint32_t columns, uint32_t packet_width,
    uint32_t block_size, uint32_t local_lanes, size_t workspace_bytes,
    uint32_t sample_count, uint32_t warmup_ms, uint32_t target_ms,
    double *samples, uint64_t *repetitions) {
    if (!address || !arguments || !argument_bytes || !samples || !repetitions ||
        !rows || !columns || !sample_count || sample_count > 100u ||
        !target_ms || target_ms > 10000u || warmup_ms > 10000u ||
        abi > 2u || count < 3u || count > 6u || const_mask >= (1u << count) ||
        !packet_width || !local_lanes || local_lanes > packet_width ||
        rows > UINT32_MAX / local_lanes || !block_size || block_size % packet_width ||
        workspace_bytes > simd_max_private_workspace_bytes) { return 1; }
    for (auto i = 0u; i < count; i++) {
        if (!arguments[i] || !argument_bytes[i]) { return 2; }
    }
    NativeContext native{address, arguments};
    TileContext tile{};
    void *context = &native;
    Invoke invoke = nullptr;
    struct alignas(64) Chunk {
        std::byte data[64];
    };
    // Runtime-owned private scratch stays outside the timer. Include guards
    // around the precise byte extent, including unused alignment padding.
    std::vector<Chunk> workspace((workspace_bytes + 63u) / 64u + 2u);
    auto *workspace_begin = reinterpret_cast<std::byte *>(workspace.data());
    auto *workspace_end = workspace_begin + workspace.size() * sizeof(Chunk);
    std::fill(workspace_begin, workspace_end, std::byte{0xa5});
    std::fill(workspace_begin + 64u, workspace_begin + 64u + workspace_bytes, std::byte{0});
    if (abi == 1u) {
        switch (count) {
            case 3u: invoke = select_native<3u>(const_mask); break;
            case 4u: invoke = select_native<4u>(const_mask); break;
            case 5u: invoke = select_native<5u>(const_mask); break;
            case 6u: invoke = select_native<6u>(const_mask); break;
            default: return 3;
        }
    } else {
        if (count != 4u) { return 3; }
        tile.entry = reinterpret_cast<TileEntry>(address);
        for (auto i = 0u; i < 4u; i++) { tile.arguments[i] = {arguments[i], argument_bytes[i]}; }
        auto &launch = tile.launch;
        launch.dispatch_size[0] = rows * local_lanes;
        launch.dispatch_size[1] = launch.dispatch_size[2] = 1u;
        launch.block_size[0] = block_size;
        launch.grid_size[0] = (launch.dispatch_size[0] - 1u) / block_size + 1u;
        launch.private_workspace = workspace_begin + 64u;
        tile.packet_width = packet_width;
        context = &tile;
        invoke = abi == 0u ? invoke_blocks : invoke_packets;
    }
    if (!invoke) { return 3; }
    using Clock = std::chrono::steady_clock;
    auto measure = [&](uint64_t repeat) {
        auto start = Clock::now();
        for (uint64_t i = 0u; i < repeat; i++) { invoke(context); }
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    };
    auto start = Clock::now();
    do { invoke(context); } while (Clock::now() - start < std::chrono::milliseconds{warmup_ms});
    uint64_t repeat = 1u;
    while (measure(repeat) < static_cast<double>(target_ms) * 1000.0 && repeat < 1048576u) { repeat *= 2u; }
    for (auto i = 0u; i < sample_count; i++) { samples[i] = measure(repeat) / static_cast<double>(repeat); }
    *repetitions = repeat;
    auto intact = [](std::byte value) { return value == std::byte{0xa5}; };
    return std::all_of(workspace_begin, workspace_begin + 64u, intact) &&
                   std::all_of(workspace_begin + 64u + workspace_bytes, workspace_end, intact) ?
               0 :
               4;
}
