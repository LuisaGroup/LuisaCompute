#include "simd_buffer.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>

namespace luisa::compute::simd {

SIMDBuffer::SIMDBuffer(size_t size_bytes) noexcept
    : _size{size_bytes} {
    _data = luisa::allocate_with_allocator<std::byte>(_size);
}

SIMDBuffer::SIMDBuffer(std::byte *data, size_t size_bytes) noexcept
    : _data{data}, _size{size_bytes}, _external{true} {}

SIMDBuffer::~SIMDBuffer() noexcept {
    if (!_external) { luisa::deallocate_with_allocator(_data); }
}

SIMDHostBufferView SIMDBuffer::view(
    size_t offset, size_t size) noexcept {
    LUISA_DEBUG_ASSERT(
        offset <= _size && size <= _size - offset,
        "SIMD buffer view out of range.");
    return {.data = _data + offset, .size_bytes = size};
}

SIMDHostBufferView SIMDBuffer::view_with_offset(
    size_t offset) noexcept {
    LUISA_DEBUG_ASSERT(offset <= _size, "SIMD buffer view out of range.");
    return view(offset, _size - offset);
}

}// namespace luisa::compute::simd
