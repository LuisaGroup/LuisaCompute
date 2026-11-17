#include "cuda_pdl_stream.h"
#include "../cuda_device.h"

namespace luisa::compute::cuda {

bool CUDAPdlStreamExtImpl::pdl_supported() const noexcept {
    return _device->pdl_supported();
}

bool CUDAPdlStreamExtImpl::auto_prologue_enabled() const noexcept {
    return _device->pdl_auto_prologue();
}

bool CUDAPdlStreamExtImpl::pdl_stream_enabled() const noexcept {
    return _device->pdl_stream_enabled();
}

void CUDAPdlStreamExtImpl::set_pdl_stream_enabled(bool enabled) noexcept {
    _device->set_pdl_stream_enabled(enabled);
}

CUDAPdlStreamExt::LaunchStats CUDAPdlStreamExtImpl::launch_stats() const noexcept {
    auto stats = _device->pdl_launch_stats();
    return LaunchStats{
        .relaxed_launches = stats.relaxed_launches,
        .strict_launches = stats.strict_launches,
        .driver_fallbacks = stats.driver_fallbacks};
}

void CUDAPdlStreamExtImpl::reset_launch_stats() noexcept {
    _device->reset_pdl_launch_stats();
}

}// namespace luisa::compute::cuda
