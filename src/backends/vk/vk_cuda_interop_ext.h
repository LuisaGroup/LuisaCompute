#pragma once
#include <luisa/backends/ext/vk_cuda_interop.h>
#include <luisa/runtime/device.h>
#include <volk.h>
#include <cuda.h>
namespace lc::vk {
class Device;
using namespace luisa;
using namespace luisa::compute;

// Manually-loaded VK_NV_cuda_kernel_launch entry points. volk.c is compiled
// without VK_ENABLE_BETA_EXTENSIONS, so its global table never contains these.
struct CudaKernelLaunchFuncs {
    PFN_vkCreateCudaModuleNV create_cuda_module{};
    PFN_vkCreateCudaFunctionNV create_cuda_function{};
    PFN_vkDestroyCudaModuleNV destroy_cuda_module{};
    PFN_vkDestroyCudaFunctionNV destroy_cuda_function{};
    PFN_vkCmdCudaLaunchKernelNV cmd_cuda_launch_kernel{};
    [[nodiscard]] bool valid() const noexcept {
        return create_cuda_module != nullptr &&
               create_cuda_function != nullptr &&
               destroy_cuda_module != nullptr &&
               destroy_cuda_function != nullptr &&
               cmd_cuda_launch_kernel != nullptr;
    }
};

// Heap-allocated CUDA kernel shader (VkCudaModuleNV + VkCudaFunctionNV pair);
// the object address is exposed publicly as an opaque uint64_t handle.
struct CudaKernelShader {
    VkCudaModuleNV module{};
    VkCudaFunctionNV function{};
};

class VkCudaInteropImpl : public VkCudaInterop {
    CUcontext _cu_context{};
    CUdevice _cu_device{};
    int _cuda_device{-1};
    Device *_device{};
    CudaKernelLaunchFuncs _cuda_launch_funcs{};
    uint32_t _cuda_compute_capability{0u};// major * 10 + minor, 0 when unknown
public:
    VkCudaInteropImpl(Device *device) noexcept;
    VkCudaInteropImpl(VkCudaInteropImpl const &) = delete;
    VkCudaInteropImpl(VkCudaInteropImpl &&) = delete;
    ~VkCudaInteropImpl() noexcept override;
    [[nodiscard]] BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count) noexcept override;
    [[nodiscard]] CUDADeviceConfigExt::ExternalVkDevice get_external_vk_device() const noexcept override;
    [[nodiscard]] ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept override;
    void vk_signal(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept override;
    void vk_wait(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept override;
    void cuda_buffer(uint64_t vk_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept override;
    [[nodiscard]] /*CUexternalMemory* */ uint64_t cuda_texture(uint64_t vk_texture_handle) noexcept override;
    void unmap(void *cuda_ptr, void *cuda_handle) noexcept override;
    [[nodiscard]] DeviceInterface *device() noexcept override;
    [[nodiscard]] int cuda_device_index() const noexcept override {
        return _cuda_device;
    }

public:
    [[nodiscard]] bool cuda_kernel_launch_supported() const noexcept override;
    [[nodiscard]] uint64_t create_cuda_kernel_shader(
        const vk_cuda_interop::CudaKernelShaderOption &option) noexcept override;
    void destroy_cuda_kernel_shader(uint64_t handle) noexcept override;
    [[nodiscard]] const CudaKernelLaunchFuncs &cuda_kernel_launch_funcs() const noexcept {
        return _cuda_launch_funcs;
    }
};

// Records a vkCmdCudaLaunchKernelNV for the given command into cmdbuffer.
// Buffer arguments are packed as 64-bit device addresses; uniform arguments
// point into the command's embedded uniform blob.
void cuda_launch_kernel(Device *device, VkCommandBuffer cmdbuffer,
                        const vk_cuda_interop::CudaKernelLaunchCommand *cmd) noexcept;
}// namespace lc::vk
