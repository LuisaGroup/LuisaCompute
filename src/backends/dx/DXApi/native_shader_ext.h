#pragma once

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/vstl/common.h>
#include <mutex>

namespace lc::dx {
using namespace luisa;
using namespace luisa::compute;

class LCDevice;
class CommandBufferBuilder;
class ComputeShader;

// Native HLSL injection for the DirectX backend.
//
// `compile()` runs DXC through the backend's own compiler module (DXIL, no
// SPIR-V) and reflects the result with `ID3D12ShaderReflection`
// (`IDxcUtils::CreateReflection`). GLSL is refused: the DirectX backend has no
// GLSL front end, so `compile()` fails closed with an explicit error.
//
// `load()` builds a `lc::dx::ComputeShader` whose root signature is
// `ShaderSerializer::SerializeRootSig`'s one-root-parameter-per-binding layout
// (root parameter index == index in the property vector == canonical binding
// index), so the dispatch path can bind property `i` from argument `i`.
//
// Supported binding classes (this iteration):
//   * constant buffers (root CBV, `register(bN[, spaceM])`),
//   * structured buffers (root SRV/UAV),
//   * byte-address buffers (root SRV/UAV),
//   * the launcher's uniform blob (root 32-bit constants at `register(b0)`,
//     only when `NativeShaderCompileInfo::push_constant_size` is nonzero).
// Typed buffers, textures and samplers are reflected and reported, but
// `load()` rejects them with an explicit error: the DirectX binding helper
// (`Shader::set_compute_resource`) has no descriptor-table path for a
// non-bindless native shader.
class DxNativeShaderExt final : public NativeShaderExt {

public:
    struct Entry {
        ComputeShader *shader{nullptr};
        // Index of the root-32-bit-constants parameter (the launcher's uniform
        // blob), or ~0u when the shader declares no uniform block.
        uint32_t uniform_property_index{~0u};
        uint32_t uniform_size{0u};
    };

private:
    LCDevice *_lc_device{nullptr};
    mutable std::mutex _mutex;
    luisa::unordered_map<uint64_t, Entry> _shaders;

private:
    [[nodiscard]] Entry *_find(uint64_t handle) noexcept;

public:
    explicit DxNativeShaderExt(LCDevice *device) noexcept;
    DxNativeShaderExt(DxNativeShaderExt const &) = delete;
    DxNativeShaderExt(DxNativeShaderExt &&) = delete;
    ~DxNativeShaderExt() noexcept;

    [[nodiscard]] NativeShaderCompileResult compile(
        const NativeShaderCompileInfo &info) noexcept override;
    [[nodiscard]] NativeShaderMetadata load(
        const NativeShaderCompileResult &result,
        luisa::span<const Usage> usage_override = {}) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    // Encodes the dispatch of `cmd` (bind pipeline, bind every argument to its
    // root parameter, dispatch the exact thread count).
    void encode(CommandBufferBuilder *builder,
                NativeShaderDispatchCommand const *cmd) noexcept;
};

// Resolves the DirectX native-shader extension of `device` and encodes `cmd`.
// Called from the DX command-buffer visitor for
// `CustomCommandUUID::NATIVE_SHADER_DISPATCH`.
void encode_native_shader_dispatch(
    LCDevice *device, CommandBufferBuilder *builder,
    NativeShaderDispatchCommand const *cmd) noexcept;

}// namespace lc::dx
