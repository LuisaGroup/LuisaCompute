#pragma once

#include <utility>

#include <core/spin_mutex.h>
#include <runtime/rhi/resource.h>
#include <runtime/event.h>
#include <runtime/command_list.h>
#include <runtime/image.h>
#include <runtime/stream_event.h>

namespace luisa::compute {

class LC_RUNTIME_API Swapchain final : public Resource {

public:
    struct LC_RUNTIME_API Present {
        const Swapchain *chain{nullptr};
        ImageView<float> frame;
        void operator()(DeviceInterface *device, uint64_t stream_handle) const noexcept;
    };

private:
    friend class Device;
    PixelStorage _storage{};
    Swapchain(DeviceInterface *device, const SwapchainCreationInfo &create_info) noexcept;
    Swapchain(DeviceInterface *device, uint64_t window_handle,
              uint64_t stream_handle, uint width, uint height,
              bool allow_hdr, bool vsync, uint back_buffer_size) noexcept;

public:
    Swapchain() noexcept = default;
    ~Swapchain() noexcept override;
    using Resource::operator bool;
    Swapchain(Swapchain &&) noexcept = default;
    Swapchain(Swapchain const &) noexcept = delete;
    Swapchain &operator=(Swapchain &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Swapchain &operator=(Swapchain const &) noexcept = delete;
    [[nodiscard]] PixelStorage backend_storage() const { return _storage; }
    [[nodiscard]] Present present(ImageView<float> frame) const noexcept;
};

LUISA_MARK_STREAM_EVENT_TYPE(Swapchain::Present)

}// namespace luisa::compute
