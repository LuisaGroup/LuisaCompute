#pragma once

#include <mutex>

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDACommandEncoder;
class CUDANativeShader;

// Native (non-DSL) shader injection for the CUDA backend.
//
// The CUDA route takes `NativeShaderLanguage::CUDA_NVRTC` source (CUDA C++),
// compiles it with the backend's own NVRTC helper (`luisa_nvrtc`, built from
// cuda_nvrtc_compiler.cpp) and loads the resulting PTX with the CUDA driver
// API. HLSL and GLSL are refused with an explicit error: the CUDA backend has
// no front end for them.
//
// Reflection (see native_shader_reflection.h): PTX records the kernel's
// parameter *layout* but no resource classes, so the parameter roles are read
// from the `__global__` signature and cross-checked against the compiled
// parameter list. As documented in native_shader_ext.h, the CUDA route supports
// buffers and scalars only:
//   * every pointer parameter is one buffer binding, in declaration order
//     (`register_index` == the parameter index, space 0), read-only for
//     `const T *` and writable otherwise, and
//   * every non-pointer parameter is a scalar kernel parameter fed by the
//     launcher's `add_uniform` values, in declaration order.
// There is no `ConstantBuffer`, sampler, texture, bindless-array or
// acceleration-structure class on this route, and `load()` rejects a usage
// declaration that contradicts the reflected read-only/writable class.
//
// `load()` loads the module once (`cuModuleLoadData` + `cuModuleGetFunction`)
// and registers the instance; `destroy_shader()` (or the `NativeShader` RAII
// owner) unloads it from the CUDA context. As everywhere else in the CUDA
// backend, a `NativeShader`'s dispatches must be synchronized before it is
// destroyed, and it must be destroyed before its device.
//
// CUDA-graph capture does not include native shader dispatches: like every
// other custom command, a captured `NativeShaderDispatchCommand` is skipped by
// `CudaGraphExt`.
class CUDANativeShaderExt final : public NativeShaderExt {

private:
    CUDADevice *_device{nullptr};
    mutable std::mutex _mutex;
    luisa::unordered_map<uint64_t, CUDANativeShader *> _shaders;

private:
    [[nodiscard]] CUDANativeShader *_find(uint64_t handle) noexcept;

public:
    explicit CUDANativeShaderExt(CUDADevice *device) noexcept;
    CUDANativeShaderExt(CUDANativeShaderExt const &) = delete;
    CUDANativeShaderExt(CUDANativeShaderExt &&) = delete;
    ~CUDANativeShaderExt() noexcept;

    [[nodiscard]] NativeShaderCompileResult compile(
        const NativeShaderCompileInfo &info) noexcept override;
    [[nodiscard]] NativeShaderMetadata load(
        const NativeShaderCompileResult &result,
        luisa::span<const Usage> usage_override = {}) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    // Launches the shader of `command` on the encoder's stream.
    void encode(CUDACommandEncoder *encoder,
                NativeShaderDispatchCommand const *command) noexcept;
};

// Resolves the CUDA native-shader extension of `device` and launches `command`.
// Called from the CUDA command encoder for
// `CustomCommandUUID::NATIVE_SHADER_DISPATCH`.
void encode_native_shader_dispatch(
    CUDADevice *device, CUDACommandEncoder *encoder,
    NativeShaderDispatchCommand const *command) noexcept;

}// namespace luisa::compute::cuda
