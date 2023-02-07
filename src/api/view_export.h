#pragma once

#include <runtime/buffer.h>
#include <runtime/image.h>

namespace luisa::compute {

class ViewExporter {
public:
    template<typename T>
    static BufferView<T> create_buffer_view(uint64_t handle, size_t offset_bytes, size_t size, size_t total_size) noexcept {
        return {handle, offset_bytes, size, total_size};
    }
    template<typename T>
    static ImageView<T> create_image_view(uint64_t handle, PixelStorage storage, uint level, uint2 size) noexcept {
        return {handle, storage, level, size};
    }
};

}// namespace luisa::compute
