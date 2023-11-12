#pragma once

#include <luisa/core/platform.h>
#include <luisa/backends/ext/denoiser_ext.h>
#include <OpenImageDenoise/oidn.hpp>
#include <luisa/core/dll_export.h>
#include <shared_mutex>

namespace luisa::compute {

class OidnDenoiser : public DenoiserExt::Denoiser {
protected:
    DeviceInterface *_device;
    oidn::DeviceRef _oidn_device;
    // shared access on ::execute
    // exclusive access on ::init
    std::shared_mutex _mutex;
    luisa::vector<oidn::FilterRef> _filters;
    luisa::vector<oidn::BufferRef> _input_buffers, _output_buffers;
    oidn::BufferRef _albedo_buffer;
    oidn::BufferRef _normal_buffer;
    oidn::FilterRef _albedo_prefilter;
    oidn::FilterRef _normal_prefilter;
    uint64_t _stream;
    virtual void reset() noexcept;
    virtual oidn::BufferRef get_buffer(const DenoiserExt::Image &img, bool read) noexcept;
    void exec_filters() noexcept;
public:
    explicit OidnDenoiser(DeviceInterface *device, oidn::DeviceRef &&oidn_device, uint64_t stream) noexcept;
    void init(const DenoiserExt::DenoiserInput &input) noexcept override;
    ~OidnDenoiser() noexcept override = default;
};

}// namespace luisa::compute
