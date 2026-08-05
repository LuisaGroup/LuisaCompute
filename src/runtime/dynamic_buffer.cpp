#include <luisa/runtime/dynamic_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

DynamicBuffer Device::create_dynamic_buffer(size_t byte_size) noexcept {
    if (byte_size == 0u) [[unlikely]] {
        detail::error_buffer_size_is_zero();
    }
    byte_size = luisa::align(byte_size, sizeof(uint));
    if (byte_size >= dynamic_buffer_invalid_offset) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Dynamic buffer capacity must fit in a 32-bit byte offset (requested: {}).",
            byte_size);
    }
    return DynamicBuffer{create_byte_buffer(byte_size),
                         create_buffer<uint>(1u),
                         create_buffer<uint>(1u), byte_size};
}

}// namespace luisa::compute
