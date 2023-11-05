#pragma once
#ifndef LUISA_COMPUTE_OIDN_UNSUPPORTED
#include <luisa/backends/ext/denoiser_ext.h>
#include <OpenImageDenoise/oidn.hpp>
#include <mutex>
namespace luisa::compute {
class OidnDenoiser : public DenoiserExt::Denoiser {
protected:
    DeviceInterface *_device;
    oidn::DeviceRef _oidn_device;
    std::mutex _mutex;
    luisa::vector<oidn::FilterRef> _filters;
    oidn::FilterRef _albedo_prefilter;
    oidn::FilterRef _normal_prefilter;
    bool _is_cpu = false;
    void reset() noexcept;
public:
    explicit OidnDenoiser(DeviceInterface *device, oidn::DeviceRef &&oidn_device, bool is_cpu = false) noexcept;
    void init(const DenoiserExt::DenoiserInput &input) noexcept override;
    void execute(uint64_t stream_handle, bool async) noexcept override;
    ~OidnDenoiser() noexcept override = default;
};
}// namespace luisa::compute
#endif
