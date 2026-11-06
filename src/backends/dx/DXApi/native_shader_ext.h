#pragma once

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/vstl/common.h>
#include <mutex>
#include <Shader/RasterShader.h>

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
// GLSL front end, so `compile()` / `compile_raster()` fail closed with an
// explicit error.
//
// `load()` builds a `lc::dx::ComputeShader` whose root signature is
// `ShaderSerializer::SerializeRootSig`'s one-root-parameter-per-binding layout
// (root parameter index == index in the property vector == canonical binding
// index), so the dispatch path can bind property `i` from argument `i`.
//
// `compile_raster()` / `load_raster()` are the raster counterpart and follow the
// same pattern one level up: each stage is compiled to DXIL by
// `hlsl::ShaderCompiler::compile_graphics_stage` (`vs_<sm>` / `ps_<sm>`), the two
// modules' DXIL reflections are merged by (space, register) into one binding
// table, and `load_raster()` turns that table into the `hlsl::Property` vector the
// existing root-signature serializer consumes. The instance is a
// `lc::dx::RasterShader` built by `RasterShader::create_native_raster_shader`, so
// the draw path (`LCCmdBuffer`) reuses its PSO cache, input layout and
// draw-state mapping unchanged; what it does not have is a DSL saved-argument
// table: the launcher packs the per-resource `Usage` declarations into the draw
// command (`DrawRasterSceneCommand::argument_usages()`), exactly like a native
// compute dispatch.
//
// Supported binding classes (this iteration, both routes):
//   * constant buffers (root CBV, `register(bN[, spaceM])`),
//   * structured buffers (root SRV/UAV),
//   * byte-address buffers (root SRV/UAV),
//   * the launcher's uniform blob (root 32-bit constants at `register(b0)`),
//     which on the raster route also carries the reserved
//     `RasterMesh::object_id` dword when `push_object_id` was requested.
// Typed buffers, textures and samplers are reflected and reported, but
// `load()` / `load_raster()` reject them with an explicit error: the DirectX
// binding helper (`Shader::set_compute_resource` / `set_raster_resource`) has no
// descriptor-table path for a non-bindless native shader.
//
// What the native raster route deliberately does not do: it has no serializer/AOT
// path (a native shader is JIT-only, so `ShaderSerializer::RasterSerialize` /
// `RasterDeSerialize` are untouched), no descriptor tables or bindless heaps (so
// no textures, samplers or acceleration structures), and no per-stage custom
// register spaces beyond what the root signature can express (one root parameter
// per binding, root constants capped at 64 dwords).
// The draw path's view of a native raster shader instance: the `RasterShader` the
// handle points at (so the PSO cache, the input layout and the draw-state mapping
// are the DSL path's, unchanged) plus the root-constant bookkeeping that only the
// native route has.
struct NativeRasterShaderState {
    RasterShader *shader{nullptr};
    // Root parameter index of the shader's root-32-bit-constants block (its
    // `cbuffer ... : register(b0)`), or ~0u when it declares none.
    uint32_t uniform_property_index{~0u};
    // The application's uniform bytes. The root-constant parameter is
    // `object_id_bytes + uniform_size` bytes wide.
    uint32_t uniform_size{0u};
    // Bytes reserved at the front of that block for `RasterMesh::object_id`
    // (0 or 4), i.e. `NativeRasterShaderCompileInfo::push_object_id`.
    uint32_t object_id_bytes{0u};
};

class DxNativeShaderExt final : public NativeShaderExt {

public:
    struct Entry {
        ComputeShader *shader{nullptr};
        // Index of the root-32-bit-constants parameter (the launcher's uniform
        // blob), or ~0u when the shader declares no uniform block.
        uint32_t uniform_property_index{~0u};
        uint32_t uniform_size{0u};
    };
    // Public name for the draw path's view of a raster instance (see
    // `NativeRasterShaderState` below, which is what the encode functions take).
    using RasterEntry = NativeRasterShaderState;

private:
    LCDevice *_lc_device{nullptr};
    mutable std::mutex _mutex;
    luisa::unordered_map<uint64_t, Entry> _shaders;
    luisa::unordered_map<uint64_t, RasterEntry> _raster_shaders;

private:
    [[nodiscard]] Entry *_find(uint64_t handle) noexcept;
    [[nodiscard]] RasterEntry *_find_raster(uint64_t handle) noexcept;

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

    [[nodiscard]] NativeRasterShaderCompileResult compile_raster(
        const NativeRasterShaderCompileInfo &info) noexcept override;
    [[nodiscard]] NativeShaderMetadata load_raster(
        const NativeRasterShaderCompileResult &result,
        luisa::span<const Usage> usage_override = {}) noexcept override;
    void destroy_raster_shader(uint64_t handle) noexcept override;

    // Fills `state` for a registered native raster instance and returns true.
    // Returns false when the handle is unknown (the map is only ever populated by
    // `load_raster()`).
    [[nodiscard]] bool find_raster(uint64_t handle,
                                   NativeRasterShaderState &state) noexcept;

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

// Resolves a native raster handle through the extension's registry. A raster
// instance is a `RasterShader` (so the draw path's PSO cache and every
// `reinterpret_cast<RasterShader *>` keep working) plus the root-constant
// bookkeeping only the native route has; keeping both together is what lets the
// encode path bind the pipeline and write the object id without re-deriving them.
// Reports and returns false for an unknown handle: a native raster draw must never
// dereference a handle it did not get from `load_raster()`.
[[nodiscard]] bool find_native_raster_shader(
    LCDevice *device, uint64_t handle, NativeRasterShaderState &state) noexcept;

// Binds the native raster pipeline of `cmd`: root signature, pipeline state, one
// root resource view per reflected binding, and the launcher's uniform blob as
// root 32-bit constants (at dword 1 when the object-id slot is reserved). Returns
// false after reporting when the draw is malformed, in which case the caller must
// skip it.
[[nodiscard]] bool encode_native_raster_draw(
    LCDevice *device, CommandBufferBuilder *builder,
    DrawRasterSceneCommand const *cmd, ID3D12PipelineState *pso) noexcept;

// Writes `object_id` into the dword the shader reserved at the front of its
// root-constant block, once per `RasterMesh`. A no-op for a shader that reserved
// no slot (`object_id_bytes() == 0`), so an application that never asked for it
// cannot have its first uniform value overwritten.
void encode_native_raster_object_id(
    LCDevice *device, CommandBufferBuilder *builder,
    DrawRasterSceneCommand const *cmd, uint32_t object_id) noexcept;

}// namespace lc::dx
