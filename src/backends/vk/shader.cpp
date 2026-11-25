#include "shader.h"
#include "log.h"
#include "device.h"
#include "buffer.h"
#include "upload_buffer.h"
#include "shader_binary_contract.h"
#include "descriptor_interface_plan.h"
#include "shader_interface_plan.h"
#include <limits>
namespace lc::vk {

SavedArgument::SavedArgument(luisa::compute::Type const *type) {
    auto checked_u32_size = [](size_t size, luisa::string_view description) {
        LUISA_ASSERT(
            size <= std::numeric_limits<uint>::max(),
            "Vulkan saved-argument size for '{}' exceeds the persisted "
            "32-bit ABI: {} bytes.",
            description, size);
        return static_cast<uint>(size);
    };
    tag = type->tag();
    if (!type->is_resource() && !type->is_custom() &&
        !type->is_cooperative_vector_ref() &&
        !type->is_cooperative_matrix_ref()) {
        struct_size = checked_u32_size(type->size(), type->description());
    } else if (type->is_buffer() && type->element() != nullptr) {
        // Reused by the XIR/SPIR-V runtime to choose a descriptor base whose
        // relative bias remains an exact element multiple.
        struct_size = checked_u32_size(
            type->element()->size(), type->element()->description());
    }
}
Shader::Shader(
    Device *device,
    ShaderTag tag,
    vstd::vector<Argument> &&captured,
    vstd::vector<SavedArgument> &&saved_arguments,
    vstd::span<hlsl::Property const> binds,
    bool use_tex2d_bindless,
    bool use_tex3d_bindless,
    bool use_buffer_bindless,
    vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> &&printers,
    luisa::span<const std::byte> constant_ubo_data,
    uint validation_count,
    uint32_t push_constant_size,
    detail::ShaderCodegenDialect codegen_dialect,
 bool allow_push_descriptors)
    : Resource{device}, _captured{std::move(captured)}, _saved_arguments(std::move(saved_arguments)),
      _shader_tag(tag), _use_tex2d_bindless(use_tex2d_bindless), _use_tex3d_bindless(use_tex3d_bindless), _use_buffer_bindless(use_buffer_bindless), _printers(std::move(printers)), _validation_count(validation_count), _push_constant_size(push_constant_size), _codegen_dialect(codegen_dialect) {
    auto argument_contract = plan_saved_argument_contract(
        _saved_arguments, validation_count);
    LUISA_ASSERT(
        argument_contract,
        "Vulkan shader argument-block ABI is invalid: {}.",
        saved_argument_contract_status_name(argument_contract.status));
    LUISA_ASSERT(
        _captured.size() <= _saved_arguments.size(),
        "Vulkan shader has {} captured arguments but only {} saved ABI entries.",
        _captured.size(), _saved_arguments.size());
    for (size_t i = 0u; i < _captured.size(); ++i) {
        auto runtime_tag = _captured[i].tag;
        auto saved_tag = _saved_arguments[i].tag;
        auto matches = [&]() noexcept {
            switch (runtime_tag) {
                case Argument::Tag::BUFFER:
                    return saved_tag == Type::Tag::BUFFER ||
                           saved_tag == Type::Tag::CUSTOM;
                case Argument::Tag::TEXTURE:
                    return saved_tag == Type::Tag::TEXTURE;
                case Argument::Tag::BINDLESS_ARRAY:
                    return saved_tag == Type::Tag::BINDLESS_ARRAY;
                case Argument::Tag::ACCEL:
                    return saved_tag == Type::Tag::ACCEL;
                case Argument::Tag::UNIFORM: return false;
            }
            return false;
        }();
        LUISA_ASSERT(
            matches,
            "Vulkan captured argument {} with runtime tag {} does not match "
            "saved ABI tag {}.",
            i, luisa::to_underlying(runtime_tag),
            luisa::to_underlying(saved_tag));
    }
    LUISA_ASSERT(
        detail::valid_shader_constant_payload_size(
            constant_ubo_data.size_bytes(),
            device->properties().limits.maxUniformBufferRange),
        "Vulkan shader constant UBO payload ({} bytes) exceeds the device "
        "maxUniformBufferRange ({} bytes) or the serialized payload cap "
        "({} bytes).",
        constant_ubo_data.size_bytes(),
        device->properties().limits.maxUniformBufferRange,
        detail::max_shader_constant_payload_size);
    if ((!device->enable_bindless()) && (use_tex2d_bindless || use_tex3d_bindless || use_buffer_bindless)) [[unlikely]] {
        LUISA_ERROR("Bindless not enabled, shader can not be load.");
    }
    auto interface_stage_mask = [&] {
        switch (tag) {
            case ShaderTag::kComputeShader:
                return detail::DescriptorInterfaceStageMask::COMPUTE;
            case ShaderTag::kRasterShader:
                return detail::DescriptorInterfaceStageMask::RASTER;
            case ShaderTag::kRayTracingShader:
                return detail::DescriptorInterfaceStageMask::RAY_TRACING;
        }
        return static_cast<detail::DescriptorInterfaceStageMask>(0u);
    }();
    auto runtime_interface_plan = detail::plan_shader_interface(
        {.properties = binds,
         .arguments = _saved_arguments,
         .stage_mask = interface_stage_mask,
         .dialect = codegen_dialect,
         .printer_count = static_cast<uint32_t>(_printers.size()),
         .validation_count = validation_count,
         .use_buffer_bindless = use_buffer_bindless,
         .use_tex2d_bindless = use_tex2d_bindless,
         .use_tex3d_bindless = use_tex3d_bindless,
         .has_constant_ubo_payload = !constant_ubo_data.empty(),
         .fallback_rtx = device->use_fallback_rtx()});
    LUISA_ASSERT(
        runtime_interface_plan,
        "Vulkan shader runtime descriptor interface is invalid: {} (device "
        "descriptor plan: {}). counts: argument_buffer {}, constant_ubo {}, "
        "resource {}, indirect {}, printer {}, local {} (descriptor plan {}).",
        detail::shader_interface_error_name(
            runtime_interface_plan.error),
        detail::descriptor_interface_error_name(
            runtime_interface_plan.descriptor_interface.error),
        runtime_interface_plan.argument_buffer_binding_count,
        runtime_interface_plan.constant_ubo_binding_count,
        runtime_interface_plan.resource_binding_count,
        runtime_interface_plan.indirect_binding_count,
        runtime_interface_plan.printer_binding_count,
        runtime_interface_plan.local_binding_count,
        runtime_interface_plan.descriptor_interface.local_binding_count);
    auto interface_plan = detail::plan_descriptor_interface(
        {.properties = binds,
         .stage_mask = interface_stage_mask,
         .bindless_heap_capacity = device->bindless_heap_capacity(),
         .use_buffer_bindless = use_buffer_bindless,
         .use_tex2d_bindless = use_tex2d_bindless,
         .use_tex3d_bindless = use_tex3d_bindless,
         .has_constant_ubo_payload = !constant_ubo_data.empty(),
         .acceleration_structure_available = device->enable_raytracing(),
         // A device that runs the software fallback has no
         // VkAccelerationStructureKHR at all: the shader's acceleration-
         // structure slot is the fallback's acceleration buffer, i.e. a
         // storage buffer (src/backends/common/rtx/fallback_rtx.h).
         .fallback_rtx = device->use_fallback_rtx(),
         .sampled_image_update_after_bind_enabled = device->enable_bindless(),
         .storage_buffer_update_after_bind_enabled = device->enable_bindless()},
        detail::descriptor_interface_limits_from(
            device->properties().limits,
            device->descriptor_indexing_properties(),
            device->acceleration_structure_properties()));
    LUISA_ASSERT(
        interface_plan,
        "Vulkan shader descriptor interface is invalid: {}.",
        detail::descriptor_interface_error_name(interface_plan.error));
    LUISA_ASSERT(
        interface_plan.local_binding_count ==
            runtime_interface_plan.local_binding_count,
        "Vulkan runtime interface planned {} local bindings, but the device "
        "descriptor plan produced {}.",
        runtime_interface_plan.local_binding_count,
        interface_plan.local_binding_count);
    _local_descriptor_binding_count =
        runtime_interface_plan.local_binding_count;
    _resource_argument_binding_offset =
        runtime_interface_plan.argument_buffer_binding_count +
        runtime_interface_plan.constant_ubo_binding_count;
    _uses_indirect_dispatch =
        runtime_interface_plan.indirect_binding_count != 0u;
    LUISA_ASSERT(
        push_constant_size != 0u &&
            push_constant_size % sizeof(uint32_t) == 0u &&
            push_constant_size <=
                device->properties().limits.maxPushConstantsSize,
        "Vulkan shader push-constant size {} is invalid or exceeds the device limit {}.",
        push_constant_size,
        device->properties().limits.maxPushConstantsSize);
    VkShaderStageFlagBits stage_bits = [&]() -> VkShaderStageFlagBits {
        switch (tag) {
            case ShaderTag::kComputeShader:
                return VK_SHADER_STAGE_COMPUTE_BIT;
            case ShaderTag::kRasterShader:
                return static_cast<VkShaderStageFlagBits>(
                    VK_SHADER_STAGE_VERTEX_BIT |
                    VK_SHADER_STAGE_FRAGMENT_BIT);
            case ShaderTag::kRayTracingShader:
                return static_cast<VkShaderStageFlagBits>(
                    VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                    VK_SHADER_STAGE_MISS_BIT_KHR);
            default:
                return VK_SHADER_STAGE_ALL;
        }
    }();
    vstd::vector<vstd::vector<VkDescriptorSetLayoutBinding>> bindings;
    vstd::vector<uint8_t> is_bindless;
    vstd::vector<vstd::vector<VkDescriptorBindingFlags>> binding_flags;
    bindings.resize(interface_plan.descriptor_set_count);
    is_bindless.resize(interface_plan.descriptor_set_count);
    binding_flags.resize(interface_plan.descriptor_set_count);
    LUISA_ASSERT(
        device->samplers().size() ==
            detail::descriptor_interface_sampler_count,
        "Vulkan device exposes {} immutable samplers but the shader ABI requires {}.",
        device->samplers().size(),
        detail::descriptor_interface_sampler_count);
    for (auto &&i : binds) {
        // ConstantValue is the legacy HLSL push-constant marker. It never
        // owns a Vulkan descriptor binding, even when set 0 is otherwise empty.
        if (i.type == hlsl::ShaderVariableType::ConstantValue) { continue; }
        auto &vec = bindings[i.space_index];
        vec.resize(std::max<size_t>(vec.size(), i.register_index + 1));
        auto &flags = binding_flags[i.space_index];
        flags.resize(vec.size());
        auto &v = vec[i.register_index];
        v.pImmutableSamplers = nullptr;
        v.binding = i.register_index;
        switch (i.type) {
            case hlsl::ShaderVariableType::ConstantBuffer:
            case hlsl::ShaderVariableType::ConstantValue:
            case hlsl::ShaderVariableType::CBVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case hlsl::ShaderVariableType::SRVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            case hlsl::ShaderVariableType::UAVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            case hlsl::ShaderVariableType::StructuredBuffer:
            case hlsl::ShaderVariableType::RWStructuredBuffer:
            case hlsl::ShaderVariableType::SPIRVAccelInstance:
            case hlsl::ShaderVariableType::SPIRVAccelInstanceRW:
            case hlsl::ShaderVariableType::SPIRVBindlessBufferMetadata:
            case hlsl::ShaderVariableType::SPIRVIndirectDispatch:
            case hlsl::ShaderVariableType::UAVBufferHeap:
            case hlsl::ShaderVariableType::SRVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case hlsl::ShaderVariableType::SPIRVAccel:
                v.descriptorType =
                    device->use_fallback_rtx() ?
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER :
                        VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                break;
            case hlsl::ShaderVariableType::SamplerHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                v.pImmutableSamplers = device->samplers().data();
                break;
            default:
                assert(false);
                break;
        }
        auto bindless = i.array_size == ~0u;
        is_bindless[i.space_index] |= bindless;
        if (bindless) {
            LUISA_ASSERT(device->bindless_heap_capacity() > 0u,
                         "Vulkan bindless shader layout requested without a "
                         "valid descriptor heap capacity.");
            flags[i.register_index] =
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        }
        v.descriptorCount = bindless ?
                                device->bindless_heap_capacity() :
                                i.array_size;
        v.stageFlags = (v.pImmutableSamplers != nullptr || i.array_size == ~0u) ? VK_SHADER_STAGE_ALL : stage_bits;
    }
    LUISA_ASSERT(
        bindings.size() == interface_plan.descriptor_set_count &&
            bindings[0u].size() ==
                interface_plan.local_binding_count,
        "Vulkan descriptor layout construction drifted from its validated plan.");
vstd::push_back_all(_binds, binds);
      // ---- set-0 path: push descriptors or classic (plan P2.3) -------------
      // One decision, here, for the whole lifetime of this shader: the set-0
      // layout either is a push descriptor set for good or is a classic set for
      // good, because the flag is part of the (interned)
      // VkDescriptorSetLayoutCreateInfo. Nothing re-evaluates it per dispatch, so
      // no dispatch can switch paths mid-frame.
      auto allow_push = allow_push_descriptors &&
                        device->push_descriptor_capable();
      auto push_eligibility = detail::PushDescriptorEligibility{
          .device_capable = allow_push,
          .max_push_descriptors = device->max_push_descriptors(),
          .set_binding_count = static_cast<uint32_t>(bindings[0u].size()),
          .set_descriptor_count = 0u,
          .has_binding_flags = false,
          .has_dynamic_or_inline_uniform_binding = false};
      for (auto binding_index = size_t{0u};
           binding_index < bindings[0u].size(); ++binding_index) {
          auto &&set_binding = bindings[0u][binding_index];
          push_eligibility.set_descriptor_count += set_binding.descriptorCount;
          if (binding_flags[0u][binding_index] != 0u) {
              push_eligibility.has_binding_flags = true;
          }
          switch (set_binding.descriptorType) {
              case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
              case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
              case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
                  push_eligibility.has_dynamic_or_inline_uniform_binding = true;
                  break;
              default: break;
          }
      }
      auto push_status = detail::push_descriptor_eligibility(push_eligibility);
      _uses_push_descriptor_set =
          push_status == detail::PushDescriptorStatus::kOk;
      if (device->push_descriptor_capable()) {
          // Counter-only (plan P0.2/P2.3): a creation-time decision, one per
          // shader - never a measured value, and never read back to choose a
          // path.
          submit_profile::record_push_descriptor_shader(_uses_push_descriptor_set);
          if (!_uses_push_descriptor_set && allow_push_descriptors) {
              LUISA_WARNING(
                  "Vulkan shader keeps classic set-0 binding ({}): {} bindings, "
                  "{} descriptors, maxPushDescriptors {}.",
                  detail::push_descriptor_status_name(push_status),
                  push_eligibility.set_binding_count,
                  push_eligibility.set_descriptor_count,
                  push_eligibility.max_push_descriptors);
          }
      }
      // ---- structural keys (plan P1.1) -------------------------------------
    // Every field the create infos below carry is captured in a key, so two
    // shaders whose create infos would be identical share one VkDescriptorSetLayout
    // and one VkPipelineLayout instead of each creating an identical copy.
    vstd::vector<detail::DescriptorSetLayoutKey> set_layout_keys;
    set_layout_keys.resize(interface_plan.descriptor_set_count);
    for (auto set_index = 0u;
         set_index < interface_plan.descriptor_set_count; ++set_index) {
        auto &set_bindings = bindings[set_index];
        auto &set_flags = binding_flags[set_index];
        LUISA_ASSERT(
            set_bindings.size() <=
                    detail::descriptor_interface_max_local_bindings &&
                set_flags.size() == set_bindings.size(),
            "Vulkan descriptor layout set {} exceeded its validated binding count.",
            set_index);
        auto bindless = is_bindless[set_index] != 0u;
        auto &key = set_layout_keys[set_index];
        key.flags = bindless ?
                        static_cast<VkDescriptorSetLayoutCreateFlags>(
                            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT) :
                        VkDescriptorSetLayoutCreateFlags{0u};
        // Plan P2.3: a push descriptor set is a *layout* property, so the bit
        // belongs in the key. P1.1's interning therefore cannot alias a push
        // set-0 layout with a classic one, and the pipeline layout key - which
        // compares the interned set-layout ids of its sets in order - inherits
        // that separation.
        if (set_index == 0u && _uses_push_descriptor_set) {
            LUISA_ASSERT(!bindless,
                         "Vulkan shader set 0 cannot be bindless and pushed at "
                         "the same time.");
            key.flags |= static_cast<VkDescriptorSetLayoutCreateFlags>(
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);
        }
        key.bindings.reserve(set_bindings.size());
        for (auto binding_index = size_t{0u};
             binding_index < set_bindings.size(); ++binding_index) {
            auto &&set_binding = set_bindings[binding_index];
            detail::DescriptorSetLayoutBindingKey binding_key{
                .binding = set_binding.binding,
                .descriptor_type = set_binding.descriptorType,
                .descriptor_count = set_binding.descriptorCount,
                .stage_flags = set_binding.stageFlags,
                .binding_flags = set_flags[binding_index]};
            if (set_binding.pImmutableSamplers != nullptr) {
                // Immutable samplers are baked into the layout, so their content
                // belongs to the structure (Vulkan reads exactly
                // descriptorCount of them).
                binding_key.immutable_samplers.assign(
                    set_binding.pImmutableSamplers,
                    set_binding.pImmutableSamplers + set_binding.descriptorCount);
            }
            key.bindings.emplace_back(std::move(binding_key));
        }
    }
    // Query every set before creating the first Vulkan object. This covers
    // implementation-specific layout support (including update-after-bind
    // and acceleration-structure descriptors) beyond the numeric plan. The query
    // describes exactly the layout the interning table will create.
    for (auto set_index = 0u;
         set_index < interface_plan.descriptor_set_count; ++set_index) {
        detail::DescriptorSetLayoutCreateInfoBuilder builder{
            set_layout_keys[set_index]};
        VkDescriptorSetLayoutSupport support{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT};
        vkGetDescriptorSetLayoutSupport(
            device->logic_device(), &builder.create_info(), &support);
        LUISA_ASSERT(
            support.supported == VK_TRUE,
            "Vulkan device rejected validated descriptor set layout {}.",
            set_index);
    }
    // The layouts are interned and owned by the device: this shader only holds
    // references. Should construction stop half-way (a future early return, not
    // the fatal checks used above), hand the references back instead of keeping
    // the shared handles alive for the lifetime of the device.
    auto &layout_interner = device->layout_interner();
    auto release_interned_layouts = [this](Device *owner) noexcept {
        auto vk_device = owner->logic_device();
        if (_pipeline_layout_interned) {
            owner->layout_interner().release_pipeline_layout(
                _pipeline_layout_interned.id, vk_device);
        }
        _pipeline_layout_interned = detail::InternedLayout{};
        _pipeline_layout = VK_NULL_HANDLE;
        for (auto &&interned : _desc_set_layout_interned) {
            // Phase 3B: a non-zero return means the device really destroyed the
            // native layout, so every VkDescriptorSet cached for this id is dead.
            if (owner->layout_interner().release_descriptor_set_layout(
                    interned.id, vk_device) != 0u) {
                owner->note_shader_layout_destroyed(interned.id);
            }
        }
        _desc_set_layout_interned.clear();
        _desc_set_layout.clear();
    };
    auto constructed = false;
    auto release_on_failure = vstd::scope_exit([&] {
        if (!constructed) { release_interned_layouts(device); }
    });
    _desc_set_layout.reserve(interface_plan.descriptor_set_count);
    _desc_set_layout_interned.reserve(interface_plan.descriptor_set_count);
    for (auto set_index = 0u;
         set_index < interface_plan.descriptor_set_count; ++set_index) {
        auto interned = layout_interner.acquire_descriptor_set_layout(
            set_layout_keys[set_index], device->logic_device());
        LUISA_ASSERT(
            interned,
            "Vulkan device failed to create descriptor set layout {}.",
            set_index);
        _desc_set_layout_interned.emplace_back(interned);
        _desc_set_layout.emplace_back(
            detail::uint64_to_handle<VkDescriptorSetLayout>(interned.handle));
    }
    // The pipeline layout key names the interned descriptor sets it is built
    // over - interned ids for exact equality, their structural hashes for the
    // portable identity a later PSO/blob cache may persist - plus the
    // push-constant range and the (zero) pipeline-layout flags.
    detail::PipelineLayoutKey pipeline_layout_key;
    pipeline_layout_key.flags = VkPipelineLayoutCreateFlags{0u};
    pipeline_layout_key.descriptor_set_layouts.reserve(
        _desc_set_layout_interned.size());
    pipeline_layout_key.descriptor_set_layout_hashes.reserve(
        _desc_set_layout_interned.size());
    for (auto &&interned : _desc_set_layout_interned) {
        pipeline_layout_key.descriptor_set_layouts.emplace_back(interned.id);
        pipeline_layout_key.descriptor_set_layout_hashes.emplace_back(
            interned.structural_hash);
    }
    pipeline_layout_key.push_constant_ranges.emplace_back(
        detail::PushConstantRangeKey{
            .stage_flags = VkShaderStageFlags(stage_bits),
            .offset = 0u,
            .size = push_constant_size});
    _pipeline_layout_interned = layout_interner.acquire_pipeline_layout(
        pipeline_layout_key, device->logic_device());
    LUISA_ASSERT(
        _pipeline_layout_interned,
        "Vulkan device failed to create the shader pipeline layout.");
    _pipeline_layout = detail::uint64_to_handle<VkPipelineLayout>(
        _pipeline_layout_interned.handle);

    if (!constant_ubo_data.empty()) {
        _has_constant_ubo = true;
        _constant_ubo = luisa::make_unique<UploadBuffer>(device, constant_ubo_data.size_bytes());
        _constant_ubo->copy_from(constant_ubo_data.data(), 0, constant_ubo_data.size_bytes());
        _constant_ubo->flush_host();
    }
    constructed = true;
}
Shader::~Shader() {
    // The descriptor-set and pipeline layouts are interned and owned by the
    // device (plan P1.1): hand the references back instead of destroying objects
    // that other shaders may share. The device destroys a layout once its last
    // user released it - and a release after the device drained its table (a
    // shader that outlives the device object graph, e.g. ~Device's own lazy
    // kernels) is a no-op, never a double destroy.
    auto *owner = device();
    if (owner == nullptr) {
        // Without a device there is nobody left who could destroy the
        // templates, and the logical device is already gone; dropping the
        // cache without a native call is the only safe option.
        detail::DescriptorUpdateTemplateCache &templates =
            descriptor_update_templates();
        templates.destroy(detail::DescriptorUpdateTemplateApi{}, VK_NULL_HANDLE);
        return;
    }
    auto vk_device = owner->logic_device();
    // Plan P2.2: the set-0 templates name this shader's (interned)
    // descriptor-set layout, so they must be destroyed *before* the layout
    // reference is released - a layout may only be destroyed once every
    // descriptor set and descriptor update template created from it is gone.
    descriptor_update_templates().destroy(
        owner->descriptor_update_template_api(), vk_device);
    auto &layout_interner = owner->layout_interner();
    if (_pipeline_layout_interned) {
        layout_interner.release_pipeline_layout(
            _pipeline_layout_interned.id, vk_device);
    }
    for (auto &&interned : _desc_set_layout_interned) {
        // Phase 3B: a non-zero return means the device really destroyed the
        // native layout, so every VkDescriptorSet cached for this id is dead.
        if (layout_interner.release_descriptor_set_layout(
                interned.id, vk_device) != 0u) {
            owner->note_shader_layout_destroyed(interned.id);
        }
    }
}
}// namespace lc::vk
