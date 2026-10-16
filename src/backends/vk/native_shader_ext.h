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
// Supported binding classes (this iteration): constant buffers (uniform buffer
// descriptors), structured/byte-address/typed buffers (storage buffer
// descriptors) and the launcher's uniform block (push constants). Samplers,
// sampled images, storage images and acceleration structures are reflected but
// rejected by `load()` with an explicit error.
class VkNativeShaderExt final : public NativeShaderExt {

private:
    Device *_vk_device{nullptr};
    mutable std::mutex _mutex;
    luisa::unordered_map<uint64_t, NativeShader *> _shaders;

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

    [[nodiscard]] NativeShader *find(uint64_t handle) noexcept;
};

// Resolves the Vulkan native-shader extension of `device` and records the
// dispatch into `cmdbuffer` (called from the Vulkan stream for
// `CustomCommandUUID::NATIVE_SHADER_DISPATCH`).
void encode_native_shader_dispatch(
    Device *device, CommandBufferState *state, VkCommandBuffer cmdbuffer,
    NativeShaderDispatchCommand const *cmd) noexcept;

}// namespace lc::vk
