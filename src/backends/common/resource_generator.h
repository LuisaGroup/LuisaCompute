#pragma once

#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/raster/depth_buffer.h>
#include <runtime/buffer.h>

namespace luisa::compute {

class ResourceGenerator {

public:
    template<typename T>
    [[nodiscard]] static Image<T> create_native_image(const ResourceCreationInfo &create_info, DeviceInterface *device, PixelStorage storage, uint2 size, uint mip_levels) noexcept {
        return {device, create_info, storage, size, mip_levels};
    }
    template<typename T>
    [[nodiscard]] static Volume<T> create_native_volume(const ResourceCreationInfo &create_info, DeviceInterface *device, PixelStorage storage, uint3 size, uint mip_levels) noexcept {
        return {device, create_info, storage, size, mip_levels};
    }
    template<typename T>
    [[nodiscard]] static Buffer<T> create_native_buffer(const BufferCreationInfo &create_info, DeviceInterface *device) noexcept {
        return {device, create_info};
    }
    [[nodiscard]] static DepthBuffer create_native_depth_buffer(const ResourceCreationInfo &create_info, DeviceInterface *device, DepthFormat format, uint2 size) noexcept {
        return {create_info, device, format, size};
    }
    template<typename T>
    [[nodiscard]] static BufferView<T> create_buffer_view(uint64_t handle, size_t offset_bytes, size_t size, size_t total_size) noexcept {
        return {handle, offset_bytes, size, total_size};
    }
    template<typename T>
    [[nodiscard]] static ImageView<T> create_image_view(uint64_t handle, PixelStorage storage, uint level, uint2 size) noexcept {
        return {handle, storage, level, size};
    }
    template<typename T>
    [[nodiscard]] static VolumeView<T> create_volume_view(uint64_t handle, PixelStorage storage, uint level, uint3 size) noexcept {
        return {handle, storage, level, size};
    }
};

}// namespace luisa::compute
