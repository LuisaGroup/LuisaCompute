#pragma once
#include "./../types/vec.hpp"

namespace luisa::shader {
struct [[builtin("bindless_array")]] BindlessBuffer {
    template <typename T>
    [[callop("TYPED_BINDLESS_BUFFER_READ")]] T buffer_read(uint32 buffer_index, uint32 elem_index);
    template <typename T>
    [[callop("TYPED_BINDLESS_BYTE_BUFFER_READ")]] T byte_buffer_read(uint32 buffer_index, uint32 byte_offset);
    [[callop("TYPED_BINDLESS_BUFFER_SIZE")]] uint32 _buffer_size(uint32 buffer_index, uint32 stride);
    template <typename T>
    [[callop("TYPED_UNIFORM_BINDLESS_BUFFER_READ")]] T uniform_idx_buffer_read(uint32 buffer_index, uint32 elem_index);
    template <typename T>
    [[callop("TYPED_UNIFORM_BINDLESS_BYTE_BUFFER_READ")]] T uniform_idx_byte_buffer_read(uint32 buffer_index, uint32 byte_offset);
    [[callop("TYPED_UNIFORM_BINDLESS_BUFFER_SIZE")]] uint32 _uniform_idx_buffer_size(uint32 buffer_index, uint32 stride);
    template <typename T>
    [[noignore]] uint32 buffer_size(uint32 buffer_index) {
        return _buffer_size(buffer_index, sizeof(T));
    }
    template <typename T>
    [[noignore]] uint32 uniform_idxbuffer_size(uint32 buffer_index) {
        return _uniform_idx_buffer_size(buffer_index, sizeof(T));
    }
};
}// namespace luisa::shader