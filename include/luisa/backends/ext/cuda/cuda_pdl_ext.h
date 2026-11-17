#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

// Runtime control and observability for the CUDA backend's stream-form
// programmatic dependent launch (PDL) path
// (obtained with `Device::extension<CUDAPdlStreamExt>()`).
//
// PDL lets a kernel that waits for its programmatic dependency
// (`grid_dependency_synchronize()`, or the automatic wait-at-entry prologue of
// `CUDADeviceConfigExt::enable_programmatic_stream_serialization()`) be launched
// with CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, so the driver may
// start it before the previous kernel in the stream retired.
//
// `pdl_stream_enabled()` is the launch policy only: it can be flipped at runtime
// and applies to the next submission. `auto_prologue_enabled()` is a property of
// the *device*, fixed at creation time, because the wait is compiled into the
// shader. `LUISA_DISABLE_PDL_STREAM=1` forces the policy off process-wide (and
// stops the prologue from being compiled in).
//
// The policy is on by default: a kernel that waits for its programmatic
// dependency has always been relaxed by this backend when its predecessor is a
// kernel launch. `CUDADeviceConfigExt::enable_programmatic_stream_serialization()`
// is what makes *every* kernel wait, and that is the part that is off by default.
class CUDAPdlStreamExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "CUDAPdlStreamExt";

    struct LaunchStats {
        // Grids launched through cuLaunchKernelEx with the attribute.
        uint64_t relaxed_launches{0u};
        // Grids launched through the strictly-ordered cuLaunchKernel path.
        uint64_t strict_launches{0u};
        // Launches whose attribute was refused by the driver (latched off).
        uint64_t driver_fallbacks{0u};
    };

    // Whether the device and driver can support stream-form PDL at all
    // (compute capability >= 9.0 and driver >= 12.0).
    [[nodiscard]] virtual bool pdl_supported() const noexcept = 0;
    // Whether every DSL compute kernel of this device carries the automatic
    // wait-at-entry prologue (from the device config ext).
    [[nodiscard]] virtual bool auto_prologue_enabled() const noexcept = 0;
    // Whether the backend may relax a launch right now.
    [[nodiscard]] virtual bool pdl_stream_enabled() const noexcept = 0;
    virtual void set_pdl_stream_enabled(bool enabled) noexcept = 0;
    [[nodiscard]] virtual LaunchStats launch_stats() const noexcept = 0;
    virtual void reset_launch_stats() noexcept = 0;

protected:
    // `DeviceExtension` carries no virtual destructor (device_interface.h), so
    // the interface has to declare one itself, exactly like CUDAExternalExt.
    virtual ~CUDAPdlStreamExt() noexcept = default;
};

}// namespace luisa::compute
