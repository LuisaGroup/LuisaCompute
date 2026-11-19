#pragma once

#include <utility>

#include <luisa/runtime/event.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream_event.h>

namespace luisa::compute {
class Swapchain;
struct LUISA_RUNTIME_API SwapchainPresent {
    const Swapchain *chain{nullptr};
    ImageView<float> frame;
    void operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept;
};
class LUISA_RUNTIME_API Swapchain final : public Resource {

public:
    using Present = SwapchainPresent;
private:
    friend class Device;
    friend class ResourceGenerator;
    friend class DXHDRExt;
    PixelStorage _storage{};
    Swapchain(DeviceInterface *device, const SwapchainCreationInfo &create_info) noexcept;
    Swapchain(DeviceInterface *device, const SwapchainOption &option, uint64_t stream_handle) noexcept;

public:
    Swapchain() noexcept = default;
    ~Swapchain() noexcept override { _release(); }
    using Resource::operator bool;
    Swapchain(Swapchain &&) noexcept = default;
    Swapchain(Swapchain const &) noexcept = default;
    Swapchain &operator=(Swapchain &&rhs) noexcept = default;
    Swapchain &operator=(Swapchain const &rhs) noexcept = default;
    [[nodiscard]] PixelStorage backend_storage() const {
        _check_is_valid();
        return _storage;
    }
    [[nodiscard]] Present present(ImageView<float> frame) const noexcept;
    SwapchainCreationInfo release() noexcept {
        return SwapchainCreationInfo{
            Resource::release(),
            _storage};
    }

protected:
    void destroy() noexcept override;
};

LUISA_MARK_STREAM_EVENT_TYPE(SwapchainPresent)

}// namespace luisa::compute
