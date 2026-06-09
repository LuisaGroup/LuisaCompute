#pragma once

#include <luisa/core/stl/vector.h>

namespace luisa::compute::coro {

/// Result of a frame buffer compaction operation.
struct CompactionResult {
    size_t alive_count_before{0u};
    size_t alive_count_after{0u};
    size_t capacity{0u};
    double load_factor_before{0.0};
    double load_factor_after{0.0};
    bool compacted{false};
};

/// Compact alive frame instances to the front of a buffer.
///
/// Each frame instance occupies `frame_stride` consecutive uint32_t
/// elements in the buffer. The `alive` vector indicates which instances
/// are still active. When the load_factor (alive_count / capacity) falls
/// below `threshold`, alive instances are compacted to positions
/// [0 .. alive_count-1] at the buffer front. The data of each alive
/// instance is preserved exactly.
///
/// @param buffer        Frame data buffer (uint32_t elements).
/// @param alive         Alive flags for each instance.
/// @param frame_stride  Number of uint32_t elements per frame instance.
/// @param result        Output: compaction statistics.
/// @param threshold     Load factor below which compaction triggers (default 0.5).
inline void compact_frame_buffer(
    luisa::vector<uint32_t> &buffer,
    const luisa::vector<bool> &alive,
    size_t frame_stride,
    CompactionResult &result,
    double threshold = 0.5) noexcept {

    const size_t capacity = alive.size();
    result.capacity = capacity;

    // Count alive instances
    size_t alive_count = 0u;
    for (size_t i = 0u; i < capacity; ++i) {
        if (alive[i]) { ++alive_count; }
    }

    result.alive_count_before = alive_count;
    result.alive_count_after = alive_count;
    result.load_factor_before = capacity > 0u
                                    ? static_cast<double>(alive_count) / static_cast<double>(capacity)
                                    : 0.0;
    result.load_factor_after = result.load_factor_before;
    result.compacted = false;

    // Skip if no compaction needed
    if (capacity == 0u || frame_stride == 0u ||
        result.load_factor_before >= threshold ||
        alive_count == capacity) {
        return;
    }

    // Compact: move alive instances to buffer front
    size_t write_pos = 0u;
    for (size_t i = 0u; i < capacity; ++i) {
        if (alive[i]) {
            if (i != write_pos) {
                // Copy frame_stride elements from source to destination
                const size_t src_base = i * frame_stride;
                const size_t dst_base = write_pos * frame_stride;
                for (size_t j = 0u; j < frame_stride; ++j) {
                    buffer[dst_base + j] = buffer[src_base + j];
                }
            }
            ++write_pos;
        }
    }

    // After compaction, alive instances are at [0 .. alive_count-1]
    result.compacted = true;
    result.load_factor_after = static_cast<double>(alive_count) / static_cast<double>(capacity);
}

} // namespace luisa::compute::coro
