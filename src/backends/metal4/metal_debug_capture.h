#pragma once

#include <luisa/backends/ext/debug_capture_ext.hpp>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalDevice;

class MetalDebugCaptureExt final : public DebugCaptureExt {

private:
    [[nodiscard]] uint64_t create_device_capture_scope(luisa::string_view label, const Option &option) const noexcept override;
    [[nodiscard]] uint64_t create_stream_capture_scope(uint64_t stream_handle, luisa::string_view label, const Option &option) const noexcept override;
    void destroy_capture_scope(uint64_t handle) const noexcept override;
    void start_debug_capture(uint64_t handle) const noexcept override;
    void stop_debug_capture() const noexcept override;
    void mark_scope_begin(uint64_t handle) const noexcept override;
    void mark_scope_end(uint64_t handle) const noexcept override;


private:
    MTL::Device *_device;

public:
    explicit MetalDebugCaptureExt(MetalDevice *device) noexcept;
    ~MetalDebugCaptureExt() noexcept;
};

}// namespace luisa::compute::metal
