#pragma once

// Size-carrying copy helpers for the embedding_mm example.
//
// `LUISA_ENABLE_SAFE_MODE` compiles the raw-pointer `copy_from(void *)` /
// `copy_to(void *)` overloads of Buffer and BufferView out, because a bare
// pointer cannot state how many bytes to transfer. The span overloads are always
// available, so these helpers derive the byte count from the buffer itself and
// keep the call sites short. A debug build asserts that the span and the
// destination agree on the size.
//
// Both helpers take their first argument by reference: every caller already owns
// an lvalue, and a buffer (view) is a cheap handle rather than storage.

#include <luisa/core/stl/memory.h>
#include <luisa/runtime/buffer.h>

namespace luisa::embed_mm {

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

}// namespace luisa::embed_mm
