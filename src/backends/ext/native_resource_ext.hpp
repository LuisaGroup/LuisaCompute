#pragma once

#include <ast/type_registry.h>
#include <backends/ext/native_resource_ext_interface.h>
#include <backends/common/resource_generator.h>

namespace luisa::compute {
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
    void *custom_data) {
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

}// namespace luisa::compute
