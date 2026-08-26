#pragma once

#if LUISA_BACKEND_ENABLE_OIDN

#include "../common/oidn_denoiser.h"
#include "metal_device.h"

namespace luisa::compute::metal {

class MetalDenoiserExt final : public DenoiserExt {

private:
    MetalDevice *_device;

public:
    explicit MetalDenoiserExt(MetalDevice *device) noexcept : _device{device} {}
    luisa::shared_ptr<Denoiser> create(uint64_t stream) noexcept override;
    luisa::shared_ptr<Denoiser> create(Stream &stream) noexcept override;
};

}// namespace luisa::compute::metal

#endif
