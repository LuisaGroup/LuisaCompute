#pragma once

#include <luisa/ast/type_registry.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/volume.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/runtime/buffer.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/backends/ext/native_resource_ext_interface.h>

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
        return {create_info, static_cast<RasterExt *>(device->extension(RasterExt::name)), device, format, size};
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

template<typename T>
Buffer<T> NativeResourceExt::create_native_buffer(void *native_ptr, size_t elem_count, void *custom_data) noexcept {
    auto type = Type::of<T>();
    return ResourceGenerator::create_native_buffer<T>(
        register_external_buffer(native_ptr, type, elem_count, custom_data),
        _device);
}
inline DepthBuffer NativeResourceExt::create_native_depth_buffer(
    void *native_ptr,
    DepthFormat format,
    uint width,
    uint height,
    void *custom_data) noexcept {
    return ResourceGenerator::create_native_depth_buffer(
        register_external_depth_buffer(native_ptr, format, width, height, custom_data),
        _device, format, {width, height});
}
template<typename T>
Image<T> NativeResourceExt::create_native_image(
    void *external_ptr,
    uint width,
    uint height,
    PixelStorage storage,
    uint mip,
    void *custom_data) noexcept {
    auto fmt = pixel_storage_to_format<T>(storage);
    return ResourceGenerator::create_native_image<T>(
        register_external_image(external_ptr, fmt, 2, width, height, 1, mip, custom_data),
        _device,
        storage,
        uint2{width, height},
        mip);
}
template<typename T>
Volume<T> NativeResourceExt::create_native_volume(
    void *external_ptr,
    uint width,
    uint height,
    uint volume,
    PixelStorage storage,
    uint mip,
    void *custom_data) noexcept {
    auto fmt = pixel_storage_to_format<T>(storage);
    return ResourceGenerator::create_native_volume<T>(
        register_external_image(external_ptr, fmt, 3, width, height, volume, mip, custom_data),
        _device, storage, uint3{width, height, volume}, mip);
}
template<typename T>
uint64_t NativeResourceExt::get_device_address(const Buffer<T> &buffer) noexcept {
    return get_native_resource_device_address(buffer.native_handle());
}
template<typename T>
uint64_t NativeResourceExt::get_device_address(const Image<T> &image) noexcept {
    return get_native_resource_device_address(image.native_handle());
}
template<typename T>
uint64_t NativeResourceExt::get_device_address(const Volume<T> &volume) noexcept {
    return get_native_resource_device_address(volume.native_handle());
}

}// namespace luisa::compute
