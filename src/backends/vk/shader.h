#pragma once
#include "pipeline_ref.h"
#include "resource.h"
#include "saved_argument_contract.h"
#include "shader_interface_plan.h"
#include "layout_intern.h"
#include "descriptor_update_template.h"
#include <volk.h>
#include "../common/hlsl/shader_property.h"
#include <luisa/ast/type_registry.h>
#include <luisa/runtime/rhi/argument.h>
#include "buffer.h"
#include "texture.h"
namespace lc::vk {
using namespace luisa::compute;

class Shader : public Resource {
public:
    enum class ShaderTag : uint {
        kComputeShader,
        kRasterShader,
        kRayTracingShader
    };

protected:
    // Descriptor-set layouts and the pipeline layout are *interned by the
    // device* (plan P1.1): identical structures are shared between shaders and
    // the device owns the handles. This shader only holds references, which
    // `~Shader` gives back; it never destroys them.
    vstd::vector<VkDescriptorSetLayout> _desc_set_layout;
    VkPipelineLayout _pipeline_layout{};
    // Same order as `_desc_set_layout`: the interned identity (id +
    // structural hash) needed to release the reference and, later, to build a
    // PSO/blob cache key out of the structure instead of an address.
    vstd::vector<detail::InternedLayout> _desc_set_layout_interned;
    detail::InternedLayout _pipeline_layout_interned;
    vstd::vector<hlsl::Property> _binds;
    vstd::vector<Argument> _captured;
    vstd::vector<SavedArgument> _saved_arguments;
    vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> _printers;
    luisa::unique_ptr<class UploadBuffer> _constant_ubo;
    ShaderTag _shader_tag;
    bool _has_constant_ubo{false};
    bool _use_tex2d_bindless;
    bool _use_tex3d_bindless;
    bool _use_buffer_bindless;
    bool _uses_indirect_dispatch{false};
    // Plan P2.3: this shader's set 0 was created with
    // VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT and is therefore bound
    // by vkCmdPushDescriptorSetKHR instead of being allocated, updated and bound.
    // Latched at creation (device capability + this shader's own set-0 shape and
    // size) and never re-evaluated: the flag describes the *layout* object, so a
    // dispatch cannot switch paths mid-frame.
    bool _uses_push_descriptor_set{false};
    uint _validation_count{0};
    uint32_t _push_constant_size{32u};
    uint32_t _local_descriptor_binding_count{};
    uint32_t _resource_argument_binding_offset{};
    size_t _buffer_metadata_count{0u};
    vstd::vector<const hlsl::Property *> _local_descriptor_properties;
    detail::ShaderCodegenDialect _codegen_dialect{
        detail::ShaderCodegenDialect::HLSL_SPIRV};
    // Plan P2.2: the descriptor-update templates of this shader's set 0, keyed
    // by the write shape the encoder produced for it. Created lazily under the
    // cache's own lock (two streams may record the same shader concurrently) and
    // destroyed in `~Shader` *before* the interned descriptor-set layout is
    // handed back, which is the order Vulkan requires. Empty when the device has
    // no template support or a write shape could not be represented.
    detail::DescriptorUpdateTemplateCache _descriptor_update_templates;
public:
    auto pipeline_layout() const { return _pipeline_layout; }
    auto shader_tag() const { return _shader_tag; }
    // Stable structural identity of the interned layouts (plan P1.1). The
    // handles themselves are device-owned and shared with other shaders, so a
    // cache that must distinguish two layouts has to key on these hashes (or
    // on the interned ids), never on a handle value.
    [[nodiscard]] auto interned_desc_set_layouts() const noexcept {
        return vstd::span<const detail::InternedLayout>{_desc_set_layout_interned};
    }
    [[nodiscard]] auto interned_pipeline_layout() const noexcept {
        return _pipeline_layout_interned;
    }
    // Phase 3B: canonical identity of the interned `VkDescriptorSetLayout` of
    // descriptor set `set`, or 0 when this shader owns no interned layout for that
    // slot (native/custom shaders that carry a foreign layout). `CommandBufferState`
    // keys its descriptor-set reuse cache on this id, so it must be the *same*
    // value for two shaders that share one interned layout and a *different* value
    // for two shaders that do not - which is exactly what the interner guarantees
    // (plan P1.1: an id is only handed out for a structure no live entry holds, and
    // ids are never reused). A descriptor set may only be reused while the layout it
    // was allocated from is alive, which is why the id is paired with
    // `Device::shader_layout_destroy_generation()`: that counter changes whenever a
    // descriptor-set layout handle is destroyed, and a state that observes a change
    // drops its cache before reusing anything.
    [[nodiscard]] uint64_t desc_set_layout_id(uint32_t set) const noexcept {
        return set < _desc_set_layout_interned.size() ?
                   _desc_set_layout_interned[set].id :
                   0u;
    }
    virtual PipelineRef *pipeline_ref() const noexcept { return nullptr; }
    // Identity of the pipeline-cache *seed* this shader would have used before
    // plan P1.3 (driver-optimization flag + required subgroup size). It still
    // names the pre-P1.3 per-shader blob that is read as an add-only seed; the
    // cache itself is device-scoped now, so nothing serializes per shader.
    [[nodiscard]] virtual uint64_t pipeline_cache_identity() const noexcept { return 0u; }
    auto binds() const { return vstd::span<const hlsl::Property>{_binds}; }
    auto captured() const { return vstd::span<const Argument>{_captured}; }
    auto desc_set_layout() const { return vstd::span{_desc_set_layout}; }
    auto saved_arguments() const { return vstd::span{_saved_arguments}; }
    // Plan P2.2: the template cache of this shader's set 0. Only the command
    // encoder touches it, through `detail::update_descriptor_set`.
    [[nodiscard]] detail::DescriptorUpdateTemplateCache &
    descriptor_update_templates() const noexcept {
        return const_cast<detail::DescriptorUpdateTemplateCache &>(
            _descriptor_update_templates);
    }
    bool use_tex2d_bindless() const { return _use_tex2d_bindless; }
    bool use_tex3d_bindless() const { return _use_tex3d_bindless; }
    bool use_buffer_bindless() const { return _use_buffer_bindless; }
    bool uses_indirect_dispatch() const { return _uses_indirect_dispatch; }
    // Plan P2.3: does this shader's set 0 live in the push descriptor bank?
    // True only when the device negotiated VK_KHR_push_descriptor and this
    // shader's set-0 layout passed the shape/limit check at creation. The command
    // encoder branches on this flag alone (never on a counter or a query), so the
    // path of a given shader is fixed for its whole lifetime.
    [[nodiscard]] bool uses_push_descriptors() const noexcept {
        return _uses_push_descriptor_set;
    }
    uint validation_count() const { return _validation_count; }
    uint32_t push_constant_size() const { return _push_constant_size; }
    uint32_t local_descriptor_binding_count() const {
        return _local_descriptor_binding_count;
    }
    uint32_t resource_argument_binding_offset() const noexcept {
        return _resource_argument_binding_offset;
    }
    [[nodiscard]] size_t buffer_metadata_count() const noexcept {
        return _buffer_metadata_count;
    }
    [[nodiscard]] const hlsl::Property *local_descriptor_property(
        uint32_t binding) const noexcept {
        return binding < _local_descriptor_properties.size() ?
                   _local_descriptor_properties[binding] :
                   nullptr;
    }
    auto codegen_dialect() const noexcept { return _codegen_dialect; }
    bool has_constant_ubo() const { return _has_constant_ubo; }
    UploadBuffer const *constant_ubo() const { return _constant_ubo.get(); }
    auto printers() const { return luisa::span{_printers}; }
    Shader(
        Device *device,
        ShaderTag tag,
        vstd::vector<Argument> &&captured,
        vstd::vector<SavedArgument> &&saved_arguments,
        vstd::span<hlsl::Property const> binds,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> &&printers,
        luisa::span<const std::byte> constant_ubo_data = {},
        uint validation_count = 0,
        uint32_t push_constant_size = 32u,
        detail::ShaderCodegenDialect codegen_dialect =
            detail::ShaderCodegenDialect::HLSL_SPIRV,
        // Plan P2.3: internal kernels (bindless upload, TLAS update, indirect
        // preparation) allocate and bind their own set 0 outside the encoder, so
        // they must keep the classic layout even on a push-capable device.
        // Native-shader paths never construct a `Shader` at all.
        bool allow_push_descriptors = true);
    virtual ~Shader();
};
}// namespace lc::vk
