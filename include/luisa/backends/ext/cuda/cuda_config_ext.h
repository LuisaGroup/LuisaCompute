#pragma once

#include <luisa/runtime/context.h>
#include <luisa/runtime/rhi/device_interface.h>

extern "C" {
typedef struct VkDevice_T *VkDevice;
typedef struct VkPhysicalDevice_T *VkPhysicalDevice;
}

namespace luisa::compute {

class CUDADeviceConfigExt : public DeviceConfigExt {
public:
    /////// External vulkan
    struct ExternalVkDevice {
        VkPhysicalDevice physical_device{nullptr};
        VkDevice device{nullptr};
    };
    [[nodiscard]] virtual ExternalVkDevice get_external_vk_device() const noexcept {
        return {};
    }
    // Force the software BVH fallback and never initialise OptiX.
    //
    // When this is `true` the CUDA backend answers every acceleration-structure
    // request (`create_mesh` / `create_accel` / their build commands) with
    // `lc::fallback_rtx::FallbackRtxDevice` and never touches OptiX, even on a
    // device that could trace in hardware.  When it is `false` the backend uses
    // the hardware path whenever the OptiX runtime can be loaded, and falls back
    // to the software one only when it cannot (`optix::available()`), so the
    // default behaviour is unchanged.
    [[nodiscard]] virtual bool use_fallback_rtx() const noexcept { return false; }

    // Enable cudadevrt linking for indirect dispatch and device-side graphs.
    // Disable it for plain compute kernels imported through
    // VK_NV_cuda_kernel_launch: those drivers expect self-contained plain PTX,
    // and the imported kernels cannot use either cudadevrt feature anyway.
    [[nodiscard]] virtual bool use_cudadevrt() const noexcept { return true; }

    // Compile the automatic PDL wait-at-entry prologue into every DSL compute
    // kernel of this device, so a launch can be relaxed without the user writing
    // `grid_dependency_synchronize()`. Off by default: it changes the generated
    // code (and the shader cache key) of every kernel. Inert below compute
    // capability 9.0, where the wait compiles to nothing.
    //
    // This is only the *codegen* half. The launch policy that uses it is on by
    // default, because relaxing a grid that already waits is what this backend did
    // before the prologue existed; use
    // `CUDAPdlStreamExt::set_pdl_stream_enabled()` (runtime) or
    // `LUISA_DISABLE_PDL_STREAM=1` (whole process, both halves) to switch it off.
    [[nodiscard]] virtual bool enable_programmatic_stream_serialization() const noexcept { return false; }
};

}// namespace luisa::compute
