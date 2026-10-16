// Replay captured SIMD Tile packet entries; this contains no operator code.
// The external driver must inspect the emitted ABI and imports, link the actual
// ORC object, and retain its source/object/library hashes. Cooperative entries,
// aliased buffers, resources other than buffers, and scalar arguments are not
// supported. Immutable input payloads are raw bytes; writable payloads are FP32
// and require an independent FP64 oracle for every element.
//
// ABI 0: packet_batch.blocks, fourth argument is the flattened block count.
// ABI 2: packet_batch, fourth argument is the number of packets in one block.
// All launch dimensions are the actual compiler metadata, not tensor extents.
//
// Each call makes private 64-byte-aligned guarded copies before timing. Caller
// allocation, Runtime dispatch, Python, JIT, validation and copies are excluded;
// launch-record resets, block traversal and compiler-emitted libc/allocation
// calls remain inside the common C++ timer loop. This is single-thread native
// entry wall time, not CPU cycles or multithread Runtime throughput.
//
// Return codes: 0 success; 1 invalid argument; 2 unsupported extent/alias;
// 3 invalid oracle; 4 buffer guard corruption; 5 workspace guard corruption;
// 6 immutable input changed; 7 full-output mismatch; 8 launch metadata changed;
// 9 allocation/other C++ exception. Writable payloads are copied back even when
// validation fails, and sample/repetition outputs are invalid on any failure.
// A zero sample count requests exactly one validation-only invocation, with no
// warmup, timer, calibration or sample loop. Useful for admission and ABI tests.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "backends/simd/llvm/llvm_schedule_codegen.h"

using namespace luisa::compute::simd;

namespace {

using TileEntry = void (*)(const void *, void *, SIMDPacketLaunchConfig *, uint32_t);

struct alignas(64) Chunk {
    std::byte bytes[64];
};

struct GuardedStorage {
    std::vector<Chunk> storage;
    size_t payload_size;

    explicit GuardedStorage(size_t size)
        : storage((size + 63u) / 64u + 2u), payload_size{size} {
        std::fill(begin(), end(), std::byte{0xa5});
    }

    [[nodiscard]] std::byte *begin() noexcept { return reinterpret_cast<std::byte *>(storage.data()); }
    [[nodiscard]] std::byte *data() noexcept { return begin() + 64u; }
    [[nodiscard]] std::byte *end() noexcept { return begin() + storage.size() * sizeof(Chunk); }
    [[nodiscard]] bool intact() noexcept {
        auto guard = [](std::byte value) noexcept { return value == std::byte{0xa5}; };
        return std::all_of(begin(), data(), guard) &&
               std::all_of(data() + payload_size, end(), guard);
    }
};

struct TileContext {
    TileEntry entry;
    SIMDHostBufferView *arguments;
    SIMDPacketLaunchConfig launch;
    uint32_t block_count;
    uint32_t packets_per_block;
    uint32_t abi;
};

void invoke(TileContext &context) {
    auto &launch = context.launch;
    if (context.abi == 0u) {
        launch.block_id[0] = launch.block_id[1] = launch.block_id[2] = 0u;
        launch.thread_index = 0u;
        context.entry(context.arguments, nullptr, &launch, context.block_count);
    } else {
        for (auto block = 0u; block < context.block_count; block++) {
            launch.block_id[0] = block % launch.grid_size[0];
            auto remaining = block / launch.grid_size[0];
            launch.block_id[1] = remaining % launch.grid_size[1];
            launch.block_id[2] = remaining / launch.grid_size[1];
            launch.thread_index = 0u;
            context.entry(context.arguments, nullptr, &launch, context.packets_per_block);
        }
    }
}

int replay_impl(
    void *address, uint32_t abi, uint32_t count,
    void *const *arguments, const size_t *argument_bytes,
    const uint32_t *writable, const double *const *expected,
    const uint32_t *dispatch_size, const uint32_t *block_size,
    uint32_t packet_width, size_t workspace_bytes,
    uint32_t sample_count, uint32_t warmup_ms, uint32_t target_ms,
    double atol, double rtol, double *samples,
    uint64_t *repetitions, double *max_abs_error) {
    if (!address || !arguments || !argument_bytes || !writable || !expected ||
        !dispatch_size || !block_size || !repetitions || !max_abs_error ||
        count == 0u || count > 32u || (abi != 0u && abi != 2u) ||
        !packet_width || packet_width > 64u || (packet_width & (packet_width - 1u)) != 0u ||
        sample_count > 100u || (sample_count != 0u && (!samples || !target_ms)) ||
        target_ms > 10000u || warmup_ms > 10000u ||
        !std::isfinite(atol) || !std::isfinite(rtol) || atol < 0.0 || rtol < 0.0 ||
        workspace_bytes > simd_max_private_workspace_bytes) { return 1; }
    *repetitions = 0u;
    *max_abs_error = 0.0;
    if (samples != nullptr) { std::fill_n(samples, sample_count, std::numeric_limits<double>::quiet_NaN()); }
    auto thread_count = uint64_t{1u};
    auto block_count = uint64_t{1u};
    std::array<uint32_t, 3u> grid{};
    for (auto axis = 0u; axis < 3u; axis++) {
        if (!dispatch_size[axis] || !block_size[axis]) { return 2; }
        grid[axis] = (dispatch_size[axis] - 1u) / block_size[axis] + 1u;
        thread_count *= block_size[axis];
        block_count *= grid[axis];
        if (thread_count > UINT32_MAX || block_count > UINT32_MAX) { return 2; }
    }
    if (thread_count % packet_width != 0u) { return 2; }
    auto total_bytes = size_t{0u};
    auto output_count = 0u;
    for (auto i = 0u; i < count; i++) {
        auto size = argument_bytes[i];
        if (!arguments[i] || !size || size > (size_t{1u} << 30u) || writable[i] > 1u) { return 2; }
        total_bytes += size;
        if (total_bytes > (size_t{2u} << 30u)) { return 2; }
        if (writable[i] != 0u) {
            output_count++;
            if (!expected[i] || size % sizeof(float) != 0u) { return 3; }
            for (auto j = size_t{0u}; j < size / sizeof(float); j++) {
                // Finite-only admission is deliberate: NaN payloads, infinities
                // and signed-zero contracts need a separate explicit policy.
                if (!std::isfinite(expected[i][j])) { return 3; }
            }
        } else if (expected[i] != nullptr) {
            return 3;
        }
        auto first = reinterpret_cast<uintptr_t>(arguments[i]);
        if (first > UINTPTR_MAX - size) { return 2; }
        for (auto j = 0u; j < i; j++) {
            auto other = reinterpret_cast<uintptr_t>(arguments[j]);
            if (first < other + argument_bytes[j] && other < first + size) { return 2; }
        }
    }
    if (output_count == 0u) { return 3; }
    std::vector<GuardedStorage> buffers;
    std::vector<SIMDHostBufferView> views;
    buffers.reserve(count);
    views.reserve(count);
    for (auto i = 0u; i < count; i++) {
        auto &buffer = buffers.emplace_back(argument_bytes[i]);
        std::memcpy(buffer.data(), arguments[i], argument_bytes[i]);
        views.emplace_back(SIMDHostBufferView{buffer.data(), argument_bytes[i]});
    }
    GuardedStorage workspace{workspace_bytes};
    std::fill_n(workspace.data(), workspace_bytes, std::byte{0});
    TileContext context{};
    context.entry = reinterpret_cast<TileEntry>(address);
    context.arguments = views.data();
    context.block_count = static_cast<uint32_t>(block_count);
    context.packets_per_block = static_cast<uint32_t>(thread_count / packet_width);
    context.abi = abi;
    for (auto axis = 0u; axis < 3u; axis++) {
        context.launch.dispatch_size[axis] = dispatch_size[axis];
        context.launch.block_size[axis] = block_size[axis];
        context.launch.grid_size[axis] = grid[axis];
    }
    context.launch.private_workspace = workspace.data();
    auto validate = [&] {
        for (auto i = 0u; i < count; i++) {
            if (!buffers[i].intact()) { return 4; }
            if (writable[i] == 0u) {
                if (std::memcmp(buffers[i].data(), arguments[i], argument_bytes[i]) != 0) { return 6; }
            } else {
                auto data = reinterpret_cast<const float *>(buffers[i].data());
                for (auto j = size_t{0u}; j < argument_bytes[i] / sizeof(float); j++) {
                    auto actual = static_cast<double>(data[j]);
                    auto reference = expected[i][j];
                    auto error = std::abs(actual - reference);
                    if (!std::isfinite(actual)) { return 7; }
                    *max_abs_error = std::max(*max_abs_error, error);
                    if (error > atol + rtol * std::abs(reference)) { return 7; }
                }
            }
        }
        if (!workspace.intact()) { return 5; }
        for (auto axis = 0u; axis < 3u; axis++) {
            if (context.launch.dispatch_size[axis] != dispatch_size[axis] ||
                context.launch.block_size[axis] != block_size[axis] ||
                context.launch.grid_size[axis] != grid[axis]) { return 8; }
        }
        if (context.launch.private_workspace != workspace.data()) { return 8; }
        return 0;
    };
    auto finish = [&](int result) {
        for (auto i = 0u; i < count; i++) {
            if (writable[i] != 0u) { std::memcpy(arguments[i], buffers[i].data(), argument_bytes[i]); }
        }
        return result;
    };
    // Validate the exact entry before timing; do not report fast failures as
    // performance. Validation-only mode is also used by external ABI tests.
    invoke(context);
    if (auto result = validate(); result != 0) { return finish(result); }
    if (sample_count == 0u) { return finish(0); }
    using Clock = std::chrono::steady_clock;
    auto measure = [&](uint64_t repeat) {
        auto start = Clock::now();
        for (auto i = uint64_t{0u}; i < repeat; i++) { invoke(context); }
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    };
    auto start = Clock::now();
    do { invoke(context); } while (Clock::now() - start < std::chrono::milliseconds{warmup_ms});
    if (auto result = validate(); result != 0) { return finish(result); }
    auto repeat = uint64_t{1u};
    while (measure(repeat) < static_cast<double>(target_ms) * 1000.0 && repeat < 1048576u) { repeat *= 2u; }
    if (auto result = validate(); result != 0) { return finish(result); }
    for (auto i = 0u; i < sample_count; i++) {
        samples[i] = measure(repeat) / static_cast<double>(repeat);
        if (auto result = validate(); result != 0) { return finish(result); }
    }
    *repetitions = repeat;
    return finish(0);
}

}// namespace

extern "C" int replay_native_tile(
    void *address, uint32_t abi, uint32_t count,
    void *const *arguments, const size_t *argument_bytes,
    const uint32_t *writable, const double *const *expected,
    const uint32_t *dispatch_size, const uint32_t *block_size,
    uint32_t packet_width, size_t workspace_bytes,
    uint32_t sample_count, uint32_t warmup_ms, uint32_t target_ms,
    double atol, double rtol, double *samples,
    uint64_t *repetitions, double *max_abs_error) noexcept {
    try {
        return replay_impl(address, abi, count, arguments, argument_bytes,
                           writable, expected, dispatch_size, block_size,
                           packet_width, workspace_bytes, sample_count,
                           warmup_ms, target_ms, atol, rtol, samples,
                           repetitions, max_abs_error);
    } catch (...) {
        return 9;
    }
}
