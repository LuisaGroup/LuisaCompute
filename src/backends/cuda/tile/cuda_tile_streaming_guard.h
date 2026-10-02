#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::cuda::native_tile {
struct StreamingScanGuard {
    uint32_t input_slot{0u};
    uint32_t output_slot{0u};
    uint64_t input_bytes{0u};
    uint64_t output_bytes{0u};
};

// The static declared view intervals use final device ABI pointers, after
// argument reordering and BufferView offsets. No stronger alignment is assumed.
template<typename Address>
[[nodiscard]] bool streaming_scan_disjoint(StreamingScanGuard guard,
                                           luisa::span<const Address> pointers) noexcept {
    static_assert(std::is_integral_v<Address> && sizeof(Address) <= sizeof(uint64_t));
    if (guard.input_slot >= pointers.size() || guard.output_slot >= pointers.size() ||
        guard.input_slot == guard.output_slot || guard.input_bytes == 0u || guard.output_bytes == 0u) { return false; }
    auto input = static_cast<uint64_t>(pointers[guard.input_slot]);
    auto output = static_cast<uint64_t>(pointers[guard.output_slot]);
    if (input == 0u || output == 0u ||
        guard.input_bytes > std::numeric_limits<uint64_t>::max() - input ||
        guard.output_bytes > std::numeric_limits<uint64_t>::max() - output) { return false; }
    return input + guard.input_bytes <= output || output + guard.output_bytes <= input;
}
}// namespace luisa::compute::cuda::native_tile
