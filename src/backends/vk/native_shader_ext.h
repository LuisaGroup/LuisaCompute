#pragma once

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/vstl/common.h>
#include <volk.h>
#include <mutex>

namespace lc::vk {
using namespace luisa;
using namespace luisa::compute;

class Device;
class NativeShader;
// The raster counterpart of `NativeShader` (see native_raster_shader.h).
// Declared here as an incomplete type; like `NativeShader` above, the
// declaration also keeps `luisa::compute::NativeRasterShader` (the public RAII
// handle) out of unqualified lookups inside `lc::vk`.
class NativeRasterShader;
struct CommandBufferState;

// Native shader injection for the Vulkan backend.
//
// HLSL is compiled to SPIR-V by DXC (`shader_compiler.cpp`, `-spirv`), GLSL by
// the bundled glslang (`glslang_compiler.cpp`). In both cases the *SPIR-V module
// is the authoritative reflection source*: descriptor sets/bindings, resource
// classes, the workgroup size and the push-constant block are read back from the
// module's decorations (see `native_shader_reflection.h`), so the reflection can
// never disagree with what the driver sees (R2).
//
// `load` builds a `lc::vk::NativeShader` (the plan's "Tier B"): the backend's
// canonical descriptor-interface ABI is a DSL-only contract that a native
// shader cannot satisfy, so the pipeline layout and descriptor pool are built
// directly from the module's reflection. See `native_shader.h`.
//
// `compile_raster` / `load_raster` are the raster counterpart: they compile one
// vertex and one pixel module (per-language, like `compile`), merge the two
// reflected descriptor tables into the single table one pipeline layout can
// describe, and build a `lc::vk::NativeRasterShader` - a graphics pipeline that
// is created per draw, because its vertex input layout, render state and
// attachment formats come from the draw. See `native_raster_shader.h`.
//
// Supported binding classes (this iteration): constant buffers (uniform buffer
// descriptors), structured/byte-address/typed buffers (storage buffer
// descriptors) and the launcher's uniform block (push constants). Samplers,
// sampled images, storage images and acceleration structures are reflected but
// rejected by `load()`/`load_raster()` with an explicit error.
class VkNativeShaderExt final : public NativeShaderExt {

private:
    Device *_vk_device{nullptr};
    mutable std::mutex _mutex;
    luisa::unordered_map<uint64_t, NativeShader *> _shaders;
    // Raster instances keep their own registry: a raster handle must never be
    // resolved by `find()` (the two instance classes are unrelated), and the
    // extension destructor reports leftovers of each kind separately.
    luisa::unordered_map<uint64_t, NativeRasterShader *> _raster_shaders;

public:
    explicit VkNativeShaderExt(Device *device) noexcept;
    VkNativeShaderExt(VkNativeShaderExt const &) = delete;
    VkNativeShaderExt(VkNativeShaderExt &&) = delete;
    ~VkNativeShaderExt() noexcept;

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

    [[nodiscard]] NativeShader *find(uint64_t handle) noexcept;
    [[nodiscard]] NativeRasterShader *find_raster(uint64_t handle) noexcept;
};

// Resolves the Vulkan native-shader extension of `device` and records the
// dispatch into `cmdbuffer` (called from the Vulkan stream for
// `CustomCommandUUID::NATIVE_SHADER_DISPATCH`).
void encode_native_shader_dispatch(
    Device *device, CommandBufferState *state, VkCommandBuffer cmdbuffer,
    NativeShaderDispatchCommand const *cmd) noexcept;

// Resolves the Vulkan native-shader extension of `device`, finds the raster
// instance `cmd` names and records the draw state into `cmdbuffer` (graphics
// pipeline, descriptor sets and the application's uniform block). Called from
// the Vulkan stream for a `RASTER_DRAW_SCENE` whose
// `DrawRasterSceneCommand::is_native_shader()` is set, i.e. for a draw built by
// `NativeShaderLauncher::draw()`.
//
  // Returns the render pass the pipeline was built for - the caller must create
  // the draw's framebuffer from it - or `VK_NULL_HANDLE` when the extension is
  // missing, `cmd->handle()` is not a registered raster instance, or the pipeline
  // could not be created. Note that all of those are reported with `LUISA_ERROR`
  // (which aborts in every build configuration), exactly like the rest of this
  // backend's command encoding: the null return is a defensive fallback that
  // only becomes reachable if that changes. `shader`, when not null, receives the
  // resolved instance so the caller can push the per-mesh object ids through
  // `NativeRasterShader::push_object_id`.
[[nodiscard]] VkRenderPass encode_native_raster_draw(
    Device *device, CommandBufferState *state, VkCommandBuffer cmdbuffer,
    DrawRasterSceneCommand const *cmd, luisa::span<uint const> strides,
    NativeRasterShader **shader = nullptr) noexcept;

}// namespace lc::vk
