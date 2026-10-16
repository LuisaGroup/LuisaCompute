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
};

}// namespace luisa::compute
