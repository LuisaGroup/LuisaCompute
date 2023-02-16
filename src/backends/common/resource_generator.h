#pragma once
#include <runtime/image.h>
#include <runtime/buffer.h>
namespace luisa::compute {
class ResourceGenerator {
public:
    template<typename T>
    static Image<T> create_native_image(const ResourceCreationInfo &create_info, DeviceInterface *device, PixelStorage storage, uint2 size, uint mip_levels) noexcept {
        return {create_info, device, storage, size, mip_levels};
    }
    template<typename T>
    static Buffer<T> create_native_buffer(const BufferCreationInfo &create_info, DeviceInterface *device, size_t size) noexcept {
        return {device, size, create_info};
    }
};
}// namespace luisa::compute