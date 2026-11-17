#pragma once

#include <luisa/backends/ext/cuda/cuda_pdl_ext.h>

namespace luisa::compute::cuda {

class CUDADevice;

class CUDAPdlStreamExtImpl final : public CUDAPdlStreamExt {

private:
    CUDADevice *_device;

public:
    explicit CUDAPdlStreamExtImpl(CUDADevice *device) noexcept
        : _device{device} {}
    ~CUDAPdlStreamExtImpl() noexcept override = default;

    [[nodiscard]] bool pdl_supported() const noexcept override;
    [[nodiscard]] bool auto_prologue_enabled() const noexcept override;
    [[nodiscard]] bool pdl_stream_enabled() const noexcept override;
    void set_pdl_stream_enabled(bool enabled) noexcept override;
    [[nodiscard]] LaunchStats launch_stats() const noexcept override;
    void reset_launch_stats() noexcept override;
};

}// namespace luisa::compute::cuda
