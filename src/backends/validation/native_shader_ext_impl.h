#pragma once

#include <luisa/backends/ext/native_shader_ext.h>

namespace lc::validation {

using namespace luisa::compute;

// Validation-layer forwarder for the native shader extension: every call is
// forwarded verbatim to the backend implementation, and the shader instances
// it hands out - compute and raster alike - are registered with the validation
// resource tracker so that `destroy_shader`/`destroy_raster_shader` bookkeeping
// (and stream misuse checks) stay consistent with a direct backend run (R8).
class NativeShaderExtImpl : public NativeShaderExt {

public:
    NativeShaderExt *_native{nullptr};

    NativeShaderExtImpl(DeviceInterface *device, NativeShaderExt *native) noexcept
        : NativeShaderExt{device}, _native{native} {}

    [[nodiscard]] NativeShaderCompileResult compile(
        const NativeShaderCompileInfo &info) noexcept override {
        return _native->compile(info);
    }
    [[nodiscard]] NativeShaderMetadata load(
        const NativeShaderCompileResult &result,
        luisa::span<const Usage> usage_override = {}) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    // Raster half: forwarded verbatim, but a handle produced by `load_raster()`
    // is registered with the tracker exactly like a compute one, so that a
    // `RASTER_DRAW_SCENE` referring to it (and its destruction) is checked
    // against a real resource instead of failing as an unknown handle.
    [[nodiscard]] NativeRasterShaderCompileResult compile_raster(
        const NativeRasterShaderCompileInfo &info) noexcept override {
        return _native->compile_raster(info);
    }
    [[nodiscard]] NativeShaderMetadata load_raster(
        const NativeRasterShaderCompileResult &result,
        luisa::span<const Usage> usage_override = {}) noexcept override;
    void destroy_raster_shader(uint64_t handle) noexcept override;
};

}// namespace lc::validation
