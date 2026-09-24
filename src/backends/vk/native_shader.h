#pragma once

#include "resource.h"
#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rhi/argument.h>
#include <volk.h>

namespace lc::vk {
using namespace luisa;
using namespace luisa::compute;

class Device;
struct CommandBufferState;

// Owns a shader's descriptor pool. It is shared between the shader instance and
// every dispatch that still holds descriptor sets allocated from it, so the pool
// is destroyed only after both the shader is gone *and* the recording command
// buffers have been retired (R16: GPU-object lifetime).
struct NativeShaderDescriptorPool {
    VkDevice device{VK_NULL_HANDLE};
    VkDescriptorPool pool{VK_NULL_HANDLE};
    NativeShaderDescriptorPool(VkDevice device, VkDescriptorPool pool) noexcept
        : device{device}, pool{pool} {}
    NativeShaderDescriptorPool(NativeShaderDescriptorPool const &) = delete;
    NativeShaderDescriptorPool(NativeShaderDescriptorPool &&) = delete;
    ~NativeShaderDescriptorPool() noexcept;
};

// Minimal Vulkan compute shader built directly from a native SPIR-V module
// (native shader injection, "Tier B" of the plan's R1 decision).
//
// The backend's DSL shaders go through the canonical descriptor-interface plan
// (`descriptor_interface_plan.h`), whose layout is a persisted ABI: dense local
// bindings in set 0, the sampler heap in set 1, then the update-after-bind
// heaps, plus an argument buffer and a constant UBO the DSL codegen always
// emits. A native shader has none of that, so this class keeps the
// `ComputeShader` *pipeline-creation pattern* (shader module + pipeline cache +
// `vkCreateComputePipelines`) but builds the descriptor set layouts, the
// pipeline layout and the descriptor pool from the shader's own SPIR-V
// reflection:
//
//   * one `VkDescriptorSetLayout` per descriptor set index the module uses
//     (so that the module's `set` numbers map onto pipeline-layout indices),
//   * `VkDescriptorType::uniform_buffer` for reflected constant buffers,
//     `storage_buffer` for everything else,
//   * a `VkPushConstantRange` covering the launcher's uniform block.
//
// Descriptor sets are allocated per dispatch from a pool owned by the shader
// and released when the recording command buffer is recycled, so several
// dispatches of one shader in one command buffer never share descriptors.
//
// Textures/samplers are intentionally not supported yet: `load()` fails closed
// for them (see the header of `NativeShaderExt`).
class NativeShader final : public Resource {

public:
    struct LayoutBinding {
        uint32_t binding_index{0u};// index into the shader's reflection table
        uint32_t set{0u};
        uint32_t binding{0u};
        uint32_t descriptor_count{1u};
        VkDescriptorType descriptor_type{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    };

private:
    uint3 _block_size{0u, 0u, 0u};
    uint32_t _push_constant_size{0u};
    luisa::string _entry_point;
    luisa::vector<NativeShaderResourceBinding> _bindings;
    vstd::vector<LayoutBinding> _layout_bindings;
    vstd::vector<VkDescriptorSetLayout> _set_layouts;
    luisa::shared_ptr<NativeShaderDescriptorPool> _descriptor_pool;
    VkPipelineLayout _pipeline_layout{VK_NULL_HANDLE};
    VkPipeline _pipeline{VK_NULL_HANDLE};
    uint32_t _descriptor_set_count{0u};
    // Number of descriptor sets a single frame may allocate before the pool is
    // exhausted; the pool is refilled as frames retire.
    static constexpr auto max_in_flight_descriptor_sets = 256u;

public:
    NativeShader(
        Device *device, uint3 block_size,
        luisa::span<const NativeShaderResourceBinding> bindings,
        luisa::span<const std::byte> spirv,
        luisa::string_view entry_point,
        uint32_t push_constant_size) noexcept;
    ~NativeShader() noexcept override;
    NativeShader(NativeShader const &) = delete;
    NativeShader(NativeShader &&) = delete;

    [[nodiscard]] uint3 block_size() const noexcept { return _block_size; }
    [[nodiscard]] uint32_t push_constant_size() const noexcept { return _push_constant_size; }
    [[nodiscard]] luisa::span<const NativeShaderResourceBinding> bindings() const noexcept {
        return _bindings;
    }
    [[nodiscard]] VkPipeline pipeline() const noexcept { return _pipeline; }
    [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept { return _pipeline_layout; }

    // Records the dispatch: allocate + write one descriptor set per descriptor
    // set index, bind the pipeline/descriptors, push the uniform block and
    // dispatch the command's exact thread count. `state` owns the lifetime of
    // the descriptor sets (they are released when the command buffer is
    // recycled).
    void encode(CommandBufferState *state, VkCommandBuffer cmdbuffer,
                NativeShaderDispatchCommand const *cmd) noexcept;
};

}// namespace lc::vk
