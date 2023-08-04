#pragma once
#include <luisa/runtime/rhi/device_interface.h>
namespace luisa::compute {
class DepthBuffer;
template<typename T>
class Buffer;
template<typename T>
class Image;
template<typename T>
class Volume;
class NativeResourceExt : public DeviceExtension {
    DeviceInterface *_device;

protected:
    ~NativeResourceExt() noexcept = default;

public:
    explicit NativeResourceExt(DeviceInterface *device) noexcept
        : _device{device} {}

    static constexpr luisa::string_view name = "NativeResourceExt";
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
    virtual ResourceCreationInfo register_external_depth_buffer(
        void *external_ptr,
        DepthFormat format,
        uint width,
        uint height,
        // custom data see backends' header
        void *custom_data) noexcept = 0;
    virtual uint64_t get_native_resource_device_address(
        void *native_handle) noexcept = 0;
    template<typename T>
    [[nodiscard]] uint64_t get_device_address(const Buffer<T> &buffer) noexcept;
    template<typename T>
    [[nodiscard]] uint64_t get_device_address(const Image<T> &image) noexcept;
    template<typename T>
    [[nodiscard]] uint64_t get_device_address(const Volume<T> &volume) noexcept;
    template<typename T>
    [[nodiscard]] Buffer<T> create_native_buffer(void *native_ptr, size_t elem_count, void *custom_data) noexcept;
    [[nodiscard]] DepthBuffer create_native_depth_buffer(
        void *native_ptr,
        DepthFormat format,
        uint width,
        uint height,
        void *custom_data) noexcept;
    template<typename T>
    [[nodiscard]] Image<T> create_native_image(
        void *external_ptr,
        uint width,
        uint height,
        PixelStorage storage,
        uint mip,
        void *custom_data) noexcept;
    template<typename T>
    [[nodiscard]] Volume<T> create_native_volume(
        void *external_ptr,
        uint width,
        uint height,
        uint volume,
        PixelStorage storage,
        uint mip,
        void *custom_data) noexcept;
};
}// namespace luisa::compute
