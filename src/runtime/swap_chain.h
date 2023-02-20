#pragma once

#include <utility>

#include <core/spin_mutex.h>
#include <runtime/resource.h>
#include <runtime/event.h>
#include <runtime/command_list.h>
#include <runtime/image.h>

namespace luisa::compute {

class LC_RUNTIME_API SwapChain final : public Resource {

public:
    struct Present {
        const SwapChain *chain;
        ImageView<float> frame;
    };

private:
    friend class Device;
    PixelStorage _storage;
    SwapChain(DeviceInterface *device, uint64_t window_handle,
              uint64_t stream_handle, uint width, uint height,
              bool allow_hdr, bool vsync, uint back_buffer_size) noexcept;

public:
    SwapChain() noexcept = default;
    using Resource::operator bool;
    SwapChain(SwapChain &&) noexcept = default;
    SwapChain(SwapChain const &) noexcept = delete;
    SwapChain &operator=(SwapChain &&) noexcept = default;
    SwapChain &operator=(SwapChain const &) noexcept = delete;
    [[nodiscard]] PixelStorage backend_storage() const { return _storage; }
    [[nodiscard]] Present present(ImageView<float> frame) const noexcept;
};

}// namespace luisa::compute
