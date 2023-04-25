#pragma once

#include <utility>

#include <core/spin_mutex.h>
#include <runtime/rhi/resource.h>
#include <runtime/event.h>
#include <runtime/command_list.h>
#include <runtime/image.h>

namespace luisa::compute {

class LC_RUNTIME_API SwapChain final : public Resource {

public:
    struct Present {
        const SwapChain *chain{nullptr};
        ImageView<float> frame;
    };

private:
    friend class Device;
    PixelStorage _storage{};
    SwapChain(DeviceInterface *device, const SwapChainCreationInfo &create_info) noexcept;
    SwapChain(DeviceInterface *device, uint64_t window_handle,
              uint64_t stream_handle, uint width, uint height,
              bool allow_hdr, bool vsync, uint back_buffer_size) noexcept;

public:
    SwapChain() noexcept = default;
    ~SwapChain() noexcept override;
    using Resource::operator bool;
    SwapChain(SwapChain &&) noexcept = default;
    SwapChain(SwapChain const &) noexcept = delete;
    SwapChain &operator=(SwapChain &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    SwapChain &operator=(SwapChain const &) noexcept = delete;
    [[nodiscard]] PixelStorage backend_storage() const { return _storage; }
    [[nodiscard]] Present present(ImageView<float> frame) const noexcept;
};

}// namespace luisa::compute
