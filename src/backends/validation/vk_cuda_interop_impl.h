#pragma once
#include <luisa/backends/ext/vk_cuda_interop.h>
namespace lc::validation {
using namespace luisa::compute;
using namespace luisa;
class VkCudaInteropImpl : public VkCudaInterop {
public:
    VkCudaInterop *impl;

    BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count) noexcept override;
    ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept override;
    void vk_signal(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept override {
        impl->vk_signal(cuda_event_handle, vk_stream, fence_index);
    }
    void vk_wait(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept override {
        impl->vk_wait(cuda_event_handle, vk_stream, fence_index);
    }
    CUDADeviceConfigExt::ExternalVkDevice get_external_vk_device() const noexcept override {
        return impl->get_external_vk_device();
    }
    void cuda_buffer(uint64_t vk_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept override {
        impl->cuda_buffer(vk_buffer_handle, cuda_ptr, cuda_handle);
    }
    uint64_t cuda_texture(uint64_t vk_texture_handle) noexcept override {
        return impl->cuda_texture(vk_texture_handle);
    }
    void unmap(void *cuda_ptr, void *cuda_handle) noexcept override {
        impl->unmap(cuda_ptr, cuda_handle);
    }
    int cuda_device_index() const noexcept override {
        return impl->cuda_device_index();
    }
    DeviceInterface *device() noexcept override {
        return impl->device();
    }
    bool cuda_kernel_launch_supported() const noexcept override {
        return impl->cuda_kernel_launch_supported();
    }
    uint64_t create_cuda_kernel_shader(
        const vk_cuda_interop::CudaKernelShaderOption &option) noexcept override {
        return impl->create_cuda_kernel_shader(option);
    }
    void destroy_cuda_kernel_shader(uint64_t handle) noexcept override {
        impl->destroy_cuda_kernel_shader(handle);
    }
};
}// namespace lc::validation