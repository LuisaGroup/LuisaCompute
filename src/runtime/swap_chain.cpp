#include <runtime/swap_chain.h>
#include <runtime/device.h>
#include <runtime/stream.h>
namespace luisa::compute {

SwapChain Device::create_swapchain(
    uint64_t window_handle,
    Stream const &stream,
    uint width,
    uint height,
    bool allow_hdr,
    uint back_buffer_size) noexcept {
    return SwapChain(impl(), window_handle, stream.handle(), width, height, allow_hdr, back_buffer_size);
}

SwapChain::SwapChain(Device::Interface *device, uint64_t window_handle, uint64_t stream_handle,
                     uint width, uint height, bool allow_hdr, uint back_buffer_size) noexcept
    : Resource{device, Tag::SWAP_CHAIN,
               device->create_swap_chain(
                   window_handle, stream_handle, width, height,
                   allow_hdr, back_buffer_size)} {
}

PixelStorage SwapChain::backend_storage() const {
    return device()->swap_chain_pixel_storage(handle());
}

SwapChain::Present SwapChain::present(ImageView<float> frame) const noexcept {
    return {this, frame};
}

}// namespace luisa::compute