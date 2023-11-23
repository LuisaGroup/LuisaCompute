#pragma once
#include "../common/oidn_denoiser.h"
#include <luisa/backends/ext/dx_cuda_interop.h>
#include <DXApi/LCDevice.h>
namespace lc::dx {
class DXOidnDenoiser : public OidnDenoiser {
    DxCudaInterop *_interop{};
    struct InteropImage {
        DenoiserExt::Image img;
        BufferCreationInfo shared_buffer;
        bool read;
    };
    luisa::vector<InteropImage> _interop_images;
    oidn::BufferRef get_buffer(const DenoiserExt::Image &img, bool read) noexcept override;
    void reset() noexcept override;

    void prepare() noexcept;
    void post_sync() noexcept;
    void execute(bool async) noexcept override;
public:
    DXOidnDenoiser(LCDevice *_device, oidn::DeviceRef &&oidn_device, uint64_t stream);
};
class DXOidnDenoiserExt : public DenoiserExt {
    LCDevice *_device;
public:
    explicit DXOidnDenoiserExt(LCDevice *device) noexcept;
    luisa::shared_ptr<Denoiser> create(uint64_t stream) noexcept override;
    luisa::shared_ptr<Denoiser> create(Stream &stream) noexcept override;
};
}// namespace lc::dx