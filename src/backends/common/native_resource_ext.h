#pragma once
#include <ast/type_registry.h>
#include <runtime/device_interface.h>
#include <backends/common/resource_generator.h>
namespace luisa::compute {
class NativeResourceExt : public DeviceExtension {
protected:
    DeviceInterface *_device;
    NativeResourceExt(DeviceInterface *device) noexcept : _device{device} {}
    ~NativeResourceExt() noexcept = default;

public:
    virtual BufferCreationInfo register_external_buffer(
        void *external_ptr,
        const Type *element,
        size_t elem_count,
        // custom data see backends' header
        void *custom_data) noexcept = 0;
    virtual ResourceCreationInfo register_external_image(
        void *external_ptr,
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels,
        // custom data see backends' header
        void *custom_data) noexcept = 0;
    template<typename T>
    [[nodiscard]] Buffer<T> create_native_buffer(void *native_ptr, size_t elem_count, void *custom_data) noexcept {
        auto type = Type::of<T>();
        return ResourceGenerator::create_native_buffer<T>(
            register_external_buffer(native_ptr, type, elem_count, custom_data),
            _device);
    }
    template<typename T>
    [[nodiscard]] Image<T> create_native_image(
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
    [[nodiscard]] Volume<T> create_native_volume(
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
};
}// namespace luisa::compute