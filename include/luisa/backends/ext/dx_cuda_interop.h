#pragma once
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {
class DxCudaInterop : public DeviceExtension {
protected:
    ~DxCudaInterop() noexcept = default;

public:
    virtual uint64_t cuda_buffer(uint64_t dx_buffer_handle) noexcept = 0;
    virtual uint64_t cuda_texture(uint64_t dx_texture_handle) noexcept = 0;
    virtual uint64_t cuda_event(uint64_t dx_event_handle) noexcept = 0;
};
}// namespace luisa::compute
