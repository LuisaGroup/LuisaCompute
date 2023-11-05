#pragma once

#include <luisa/core/platform.h>
#include <luisa/backends/ext/denoiser_ext.h>
#include <OpenImageDenoise/oidn.hpp>
#include <luisa/core/dll_export.h>
#include <shared_mutex>

namespace luisa::compute {

class LC_BACKEND_API OidnDenoiser : public DenoiserExt::Denoiser {
protected:
    DeviceInterface *_device;
    oidn::DeviceRef _oidn_device;
    std::shared_mutex _mutex;
    luisa::vector<oidn::FilterRef> _filters;
    oidn::FilterRef _albedo_prefilter;
    oidn::FilterRef _normal_prefilter;
    uint64_t _stream;
    bool _is_cpu = false;
    void reset() noexcept;
public:
    explicit OidnDenoiser(DeviceInterface *device, oidn::DeviceRef &&oidn_device, uint64_t stream, bool is_cpu = false) noexcept;
    void init(const DenoiserExt::DenoiserInput &input) noexcept override;
    void execute(bool async) noexcept override;
    ~OidnDenoiser() noexcept override = default;
};

}// namespace luisa::compute
