#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

// Backend-specific device configuration for the SIMD CPU runtime.
// A width of zero keeps the backend default. Nonzero widths select the fixed
// LLVM vector specialization used by every shader created on the device. A
// worker count of zero uses the host hardware concurrency; one forces serial
// block execution for diagnostics and benchmarking.
class SIMDDeviceConfigExt final : public DeviceConfigExt {

private:
    uint _warp_width{0u};
    uint _worker_count{0u};

public:
    explicit SIMDDeviceConfigExt(
        uint warp_width = 0u, uint worker_count = 0u) noexcept
        : _warp_width{warp_width},
          _worker_count{worker_count} {}

    [[nodiscard]] uint warp_width() const noexcept { return _warp_width; }
    void set_warp_width(uint warp_width) noexcept {
        _warp_width = warp_width;
    }
    [[nodiscard]] uint worker_count() const noexcept {
        return _worker_count;
    }
    void set_worker_count(uint worker_count) noexcept {
        _worker_count = worker_count;
    }
};

}// namespace luisa::compute
