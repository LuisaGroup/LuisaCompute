#pragma once

#include <luisa/runtime/context.h>
#include <luisa/runtime/rhi/device_interface.h>
extern "C" {
typedef struct VkDevice_T *VkDevice;
typedef struct VkPhysicalDevice_T *VkPhysicalDevice;
}
namespace luisa::compute {
class CudaDeviceConfigExt : public DeviceConfigExt {
public:
    /////// External vulkan
    struct ExternalVkDevice {
        VkPhysicalDevice physical_device{nullptr};
        VkDevice device{nullptr};
    };
    [[nodiscard]] virtual ExternalVkDevice get_external_vk_device() const noexcept {
        return {};
    }
};
}// namespace luisa::compute