#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

Swapchain::Swapchain(DeviceInterface *device, const SwapchainCreationInfo &create_info) noexcept
    : Resource{device, Tag::SWAP_CHAIN, create_info},
      _storage{create_info.storage} {}

Swapchain Device::create_swapchain(uint64_t window_handle,
                                   Stream const &stream,
                                   uint2 resolution,
                                   bool allow_hdr,
                                   bool vsync,
                                   uint back_buffer_size) noexcept {
    if (stream.stream_tag() != StreamTag::GRAPHICS) [[unlikely]] {
        LUISA_ERROR("Only graphics queue can create swap chain!");
    }
    return {impl(), window_handle, stream.handle(),
            resolution.x, resolution.y, allow_hdr, vsync, back_buffer_size};
}

Swapchain::Swapchain(DeviceInterface *device, uint64_t window_handle, uint64_t stream_handle,
                     uint width, uint height, bool allow_hdr, bool vsync, uint back_buffer_size) noexcept
    : Swapchain{device,
                device->create_swapchain(
                    window_handle, stream_handle, width, height,
                    allow_hdr, vsync, back_buffer_size)} {}

Swapchain::Present Swapchain::present(ImageView<float> frame) const noexcept {
    _check_is_valid();
    LUISA_ASSERT(frame.level() == 0u,
                 "Only the base-level image is presentable in a swapchain.");
    return {this, frame};
}

Swapchain::~Swapchain() noexcept {
    if (*this) { device()->destroy_swap_chain(handle()); }
}

void Swapchain::Present::operator()(
    DeviceInterface *device,
    uint64_t stream_handle) && noexcept {
    device->present_display_in_stream(stream_handle, chain->handle(), frame.handle());
}

}// namespace luisa::compute

