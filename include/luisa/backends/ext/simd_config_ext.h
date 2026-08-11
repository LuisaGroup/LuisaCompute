#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

// Backend-specific device configuration for the SIMD CPU runtime.
// A width of zero keeps the backend default. Nonzero widths select the fixed
// LLVM vector specialization used by every shader created on the device.
class SIMDDeviceConfigExt final : public DeviceConfigExt {

private:
    uint _warp_width{0u};

public:
    explicit SIMDDeviceConfigExt(uint warp_width = 0u) noexcept
        : _warp_width{warp_width} {}

    [[nodiscard]] uint warp_width() const noexcept { return _warp_width; }
    void set_warp_width(uint warp_width) noexcept {
        _warp_width = warp_width;
    }
};

}// namespace luisa::compute
