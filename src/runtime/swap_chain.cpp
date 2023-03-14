#include <runtime/swap_chain.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <core/logging.h>

namespace luisa::compute {

SwapChain Device::create_swapchain(uint64_t window_handle,
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

SwapChain::SwapChain(DeviceInterface *device, uint64_t window_handle, uint64_t stream_handle,
                     uint width, uint height, bool allow_hdr, bool vsync, uint back_buffer_size) noexcept
    : Resource{device, Tag::SWAP_CHAIN,
               [&] {
                   auto create_info = device->create_swap_chain(
                       window_handle, stream_handle, width, height,
                       allow_hdr, vsync, back_buffer_size);
                   _storage = create_info.storage;
                   return create_info;
               }()} {
}

SwapChain::Present SwapChain::present(ImageView<float> frame) const noexcept {
    return {this, frame};
}

}// namespace luisa::compute
