#pragma once

// Size-carrying copy helpers for the NTC example.
//
// `LUISA_ENABLE_SAFE_MODE` (enabled by scripts/options.lua) compiles the raw
// pointer `copy_from(void *)` / `copy_to(void *)` overloads of Buffer and
// BufferView out, because a bare pointer cannot state how many bytes to
// transfer.  The span overloads are always available, so these helpers derive
// the byte count from the buffer itself -- which is exactly what the removed
// pointer overloads did -- and keep the call sites short.
//
// Both helpers take the buffer by reference: every caller already owns an
// lvalue, and a buffer is a cheap handle rather than storage.  This mirrors
// examples/embedding_mm/embed/buffer_copy.h.

#include <luisa/core/stl/memory.h>
#include <luisa/runtime/buffer.h>

namespace luisa::ntc {

// Uploads `source` into `destination`; the transfer covers exactly `destination`.
template<typename Destination, typename T>
[[nodiscard]] auto copy_to_device(Destination &destination, const T *source) noexcept {
    return destination.copy_from(luisa::span<const T>{source, destination.size()});
}

// Downloads `source` into `destination`; the transfer covers exactly `source`.
template<typename Source, typename T>
[[nodiscard]] auto copy_from_device(Source &source, T *destination) noexcept {
    return source.copy_to(luisa::span<T>{destination, source.size()});
}

}// namespace luisa::ntc
