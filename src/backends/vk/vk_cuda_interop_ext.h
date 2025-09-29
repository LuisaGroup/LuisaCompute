#pragma once
#include <luisa/backends/ext/vk_cuda_interop.h>
#include <luisa/runtime/device.h>
#include <cuda.h>
namespace lc::vk {
class Device;
using namespace luisa;
using namespace luisa::compute;
class VkCudaInteropImpl : public VkCudaInterop {
    CUcontext _cu_context{};
    CUdevice _cu_device{};
    Device *_device;
public:
    VkCudaInteropImpl(Device *device);
    VkCudaInteropImpl(VkCudaInteropImpl const &) = delete;
    VkCudaInteropImpl(VkCudaInteropImpl &&) = delete;
    ~VkCudaInteropImpl();
    BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count) noexcept override;
    [[nodiscard]] CudaDeviceConfigExt::ExternalVkDevice get_external_vk_device() const noexcept override;
    ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept override;
    void _vk_signal(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept override;
    void _vk_wait(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept override;
    void cuda_buffer(uint64_t vk_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept override;
    /*CUexternalMemory* */ uint64_t cuda_texture(uint64_t vk_texture_handle) noexcept override;
    void unmap(void *cuda_ptr, void *cuda_handle) override;
    DeviceInterface *device() override;
};
}// namespace lc::vk