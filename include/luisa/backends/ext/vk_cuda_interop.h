#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/volume.h>
#include <luisa/vstl/meta_lib.h>
#include <luisa/backends/ext/native_resource_ext.hpp>
#include <luisa/backends/ext/cuda_config_ext.h>
namespace luisa::compute {
class VkCudaInterop;
namespace vk_cuda_interop {
struct Signal {
    VkCudaInterop *ext;
    uint64_t handle;
    uint64_t fence;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};
struct Wait {
    VkCudaInterop *ext;
    uint64_t handle;
    uint64_t fence;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};
}// namespace vk_cuda_interop

class VkCudaInterop : public DeviceExtension {
    friend class VkCudaTimelineEvent;
public:
    static constexpr luisa::string_view name = "VkCudaInterop";

public:
    [[nodiscard]] virtual BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count) noexcept = 0;
    [[nodiscard]] virtual ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept = 0;
    virtual void _vk_signal(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept = 0;
    virtual void _vk_wait(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept = 0;

public:
    [[nodiscard]] virtual CudaDeviceConfigExt::ExternalVkDevice get_external_vk_device() const noexcept = 0;
    virtual void cuda_buffer(uint64_t vk_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept = 0;
    [[nodiscard]] virtual /*CUexternalMemory* */ uint64_t cuda_texture(uint64_t vk_texture_handle) noexcept = 0;
    virtual void unmap(void *cuda_ptr, void *cuda_handle) = 0;
    virtual DeviceInterface *device() = 0;

    vk_cuda_interop::Signal vk_signal(TimelineEvent const &cuda_event, uint64_t fence_index) noexcept {
        return vk_cuda_interop::Signal{
            this,
            cuda_event.handle(),
            fence_index};
    }
    vk_cuda_interop::Wait vk_wait(TimelineEvent const &cuda_event, uint64_t fence_index) noexcept {
        return vk_cuda_interop::Wait{
            this,
            cuda_event.handle(),
            fence_index};
    }
    vk_cuda_interop::Signal vk_signal(Event const &cuda_event) noexcept {
        auto signal = cuda_event.signal();
        return vk_cuda_interop::Signal{
            this,
            signal.handle,
            signal.fence};
    }
    vk_cuda_interop::Wait vk_wait(Event const &cuda_event, uint64_t fence = std::numeric_limits<uint64_t>::max()) noexcept {
        auto wait = cuda_event.wait(fence);
        return vk_cuda_interop::Wait{
            this,
            wait.handle,
            wait.fence};
    }
    template<typename T>
    Buffer<T> create_buffer(size_t elem_count) noexcept {
        return Buffer<T>{device(), create_interop_buffer(Type::of<T>(), elem_count)};
    }
    template<typename T>
    Image<T> create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Image<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 2, width, height, 1, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            uint2(width, height),
            mip_levels};
    }
    template<typename T>
    Image<T> create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Image<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 2, size.x, size.y, 1, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            size,
            mip_levels};
    }
    template<typename T>
    Volume<T> create_volume(PixelStorage pixel, uint width, uint height, uint volume, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Volume<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 3, width, height, volume, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            uint3(width, height, volume),
            mip_levels};
    }
    template<typename T>
    Volume<T> create_image(PixelStorage pixel, uint3 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Volume<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 3, size.x, size.y, size.z, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            size,
            mip_levels};
    }

    virtual ~VkCudaInterop() noexcept = default;
};
LUISA_MARK_STREAM_EVENT_TYPE(vk_cuda_interop::Signal)
LUISA_MARK_STREAM_EVENT_TYPE(vk_cuda_interop::Wait)
namespace vk_cuda_interop {
inline void Signal::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    ext->_vk_signal(handle, stream_handle, fence);
}
inline void Wait::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    ext->_vk_wait(handle, stream_handle, fence);
}
}// namespace vk_cuda_interop
}// namespace luisa::compute