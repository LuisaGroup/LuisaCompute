#include "entry.h"
#include "constant_ubo_layout.h"
#include "texture_sampling.h"
#include <algorithm>
#include <luisa/core/logging.h>
#include <limits>

namespace lc::spirv {

namespace {

void add_u32_decoration(spv::Builder &builder, spv::Id target,
                        spv::Decoration decoration,
                        uint32_t literal) noexcept {
    builder.addDecoration(target, decoration,
                          std::vector<unsigned>{literal});
}

void add_u32_member_decoration(spv::Builder &builder, spv::Id target,
                               uint32_t member_index,
                               spv::Decoration decoration,
                               uint32_t literal) noexcept {
    builder.addMemberDecoration(target, member_index, decoration,
                                std::vector<unsigned>{literal});
}

[[nodiscard]] const Type *matrix_after_array_layers(
    const Type *type) noexcept {
    while (type != nullptr && type->is_array()) {
        type = type->element();
    }
    return type != nullptr && type->is_matrix() ? type : nullptr;
}

struct ConstantUBOElementLayout {
    size_t base_alignment;
    size_t occupied_size;
    size_t matrix_stride;
};

[[nodiscard]] ConstantUBOElementLayout constant_ubo_element_layout(
    const Type *type) noexcept {
    LUISA_ASSERT(type != nullptr &&
                     (type->is_scalar() || type->is_vector() ||
                      type->is_matrix()),
                 "Unsupported constant UBO element type {}.",
                 type == nullptr ? "<null>" : type->description());
    if (type->is_matrix()) {
        auto column = Type::vector(type->element(), type->dimension());
        // A std140 matrix is an array of column vectors. Array alignment is
        // rounded up to 16 bytes, so mat2 has 16-byte columns even though its
        // host columns are tightly packed at 8 bytes.
        auto column_layout = constant_ubo_element_layout(column);
        auto matrix_alignment =
            luisa::align(column_layout.base_alignment, size_t{16u});
        auto matrix_stride =
            luisa::align(column_layout.occupied_size, matrix_alignment);
        return {
            .base_alignment = matrix_alignment,
            .occupied_size = matrix_stride * type->dimension(),
            .matrix_stride = matrix_stride};
    }
    auto base_alignment = type->is_vector() ?
                              type->element()->size() *
                                  (type->dimension() == 2u ? 2u : 4u) :
                              type->size();
    return {
        .base_alignment = base_alignment,
        .occupied_size = type->size(),
        .matrix_stride = 0u};
}

[[nodiscard]] bool constant_ubo_storage_supported(
    const Type *type, const SpirvTargetFeatures &features) noexcept {
    while (type != nullptr &&
           (type->is_vector() || type->is_matrix())) {
        type = type->element();
    }
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Type::Tag::INT8:
        case Type::Tag::UINT8:
        case Type::Tag::FLOAT8_E4M3:
        case Type::Tag::FLOAT8_E5M2:
            return features.uniform_storage_buffer_8bit_access;
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
        case Type::Tag::FLOAT16:
            return features.uniform_storage_buffer_16bit_access;
        default: return true;
    }
}

}// namespace

void SpirvCodegenEntry::generate_binding(
    Function kernel,
    luisa::span<const std::pair<Variable, Usage>> argument_usages,
    const xir::KernelFunction *xir_kernel) {
    LUISA_ASSERT(
        _runtime_target_plan_installed,
        "SPIR-V descriptor binding requires a validated runtime target plan.");
    LUISA_ASSERT(xir_kernel != nullptr,
                 "SPIR-V descriptor binding received no XIR kernel.");
    _properties.clear();
    _kernel_resource_bindings.clear();
    _use_tex2d_bindless = false;
    _use_tex3d_bindless = false;
    _use_buffer_bindless = false;
    _use_buffer_bindless_metadata = false;
    _has_argument_buffer = false;
    _argument_buffer_id = spv::NoResult;
    _indirect_dispatch_buffer_id = spv::NoResult;
    _constant_ubo_var = spv::NoResult;
    _ubo_constant_member_indices.clear();
    _constant_ubo_data.clear();
    _has_constant_ubo = false;

    auto argument_usage = [&](const Variable &v) noexcept {
        for (auto &&[arg, usage] : argument_usages) {
            if (arg.uid() == v.uid()) { return usage; }
        }
        return kernel.variable_usage(v.uid());
    };
    auto effective_argument_usage = [&](const Variable &v) noexcept {
        auto usage = argument_usage(v);
        return usage == Usage::NONE ? Usage::READ : usage;
    };
    auto is_writable = [&](const Variable &v) {
        return (luisa::to_underlying(effective_argument_usage(v)) &
                luisa::to_underlying(Usage::WRITE)) != 0u;
    };
    auto has_usage = [](Usage usage, Usage flag) noexcept {
        return (luisa::to_underlying(usage) &
                luisa::to_underlying(flag)) != 0u;
    };
    auto is_indirect_dispatch_variable = [](const Variable &v) noexcept {
        return v.type() != nullptr && v.type()->is_custom() &&
               v.type()->description() == "LC_IndirectDispatchBuffer"sv;
    };
    auto is_kernel_resource_variable =
        [&](const Variable &v) noexcept {
            return v.is_resource() ||
                   is_indirect_dispatch_variable(v);
        };
    auto add_resource_binding = [&](const Variable &v) -> KernelResourceBinding & {
        LUISA_ASSERT(is_kernel_resource_variable(v),
                     "Only resource arguments have SPIR-V resource bindings.");
        auto &binding = _kernel_resource_bindings.emplace_back();
        binding.type_tag = v.type()->tag();
        binding.usage = binding.type_tag == Type::Tag::ACCEL ?
                            Usage::NONE :
                            effective_argument_usage(v);
        auto ast_index = size_t{0u};
        for (auto &&argument : argument_usages) {
            if (argument.first.uid() == v.uid()) { break; }
            ast_index++;
        }
        if (binding.type_tag == Type::Tag::ACCEL ||
            binding.type_tag == Type::Tag::BINDLESS_ARRAY) {
            auto xir_index = size_t{0u};
            const xir::Argument *xir_argument = nullptr;
            for (auto *argument : xir_kernel->arguments()) {
                if (xir_index++ == ast_index) {
                    xir_argument = argument;
                    break;
                }
            }
            auto matches = xir_argument != nullptr &&
                           xir_argument->type() != nullptr &&
                           xir_argument->type()->tag() == binding.type_tag;
            LUISA_ASSERT(
                matches,
                "SPIR-V resource binding of type {} has no matching XIR "
                "kernel argument at index {}.",
                v.type()->description(), ast_index);
            if (binding.type_tag == Type::Tag::ACCEL) {
                // Acceleration-structure descriptors are split by their exact
                // lowered XIR role. AST usage can conservatively retain a read
                // from a branch that AST-to-XIR eliminated; using it here
                // would create a descriptor with no matching emitted access.
                binding.usage = spirv_function_argument_usage_of(
                    _function_argument_usage, xir_kernel, xir_argument,
                    Usage::NONE);
                binding.requires_accel_traversal_descriptor =
                    spirv_function_argument_requires_accel_traversal_descriptor(
                        _function_argument_usage, xir_kernel, xir_argument);
                binding.requires_accel_instance_buffer =
                    spirv_function_argument_requires_accel_instance_buffer(
                        _function_argument_usage, xir_kernel, xir_argument);
            } else {
                binding.requires_bindless_buffer_metadata =
                    spirv_function_argument_requires_bindless_buffer_metadata(
                        _function_argument_usage, xir_kernel, xir_argument);
            }
        }
        return binding;
    };

    // Detect cbuffer non-empty: any argument that is not a resource or builtin.
    // Direct buffer views also require this internal argument buffer for their
    // descriptor-relative byte bias and exact logical size metadata.
    bool cbuffer_non_empty = false;
    bool has_direct_buffer = false;
    for (auto &&arg : kernel.arguments()) {
        if (is_indirect_dispatch_variable(arg)) { continue; }
        auto tag = arg.tag();
        has_direct_buffer |= tag == Variable::Tag::BUFFER;
        switch (tag) {
            case Variable::Tag::BUFFER:
            case Variable::Tag::TEXTURE:
            case Variable::Tag::BINDLESS_ARRAY:
            case Variable::Tag::ACCEL:
            case Variable::Tag::THREAD_ID:
            case Variable::Tag::BLOCK_ID:
            case Variable::Tag::DISPATCH_ID:
            case Variable::Tag::DISPATCH_SIZE:
            case Variable::Tag::KERNEL_ID:
            case Variable::Tag::WARP_LANE_COUNT:
            case Variable::Tag::WARP_LANE_ID:
            case Variable::Tag::RASTER_OBJECT_ID:
            case Variable::Tag::RASTER_BARYCENTRICS:
                break;
            default:
                cbuffer_non_empty = true;
                break;
        }
    }
    _has_argument_buffer = cbuffer_non_empty || has_direct_buffer;

    // Consume the exact frozen decision produced before any descriptor or
    // type is emitted. Do not rediscover AST/XIR usage in this layer.
    _use_buffer_bindless =
        _runtime_target_plan.bindless_resources.buffer_heap;
    _use_buffer_bindless_metadata =
        _runtime_target_plan.bindless_resources.buffer_metadata;
    LUISA_ASSERT(
        !_use_buffer_bindless || _use_buffer_bindless_metadata,
        "SPIR-V bindless buffer heap access has no per-slot metadata plan.");
    _use_tex2d_bindless =
        _runtime_target_plan.bindless_resources.texture_2d;
    _use_tex3d_bindless =
        _runtime_target_plan.bindless_resources.texture_3d;

    // Register indexer matching HLSL's RegisterType pattern.
    // For SPIR-V, all register types share the same flat counter (like SpirVRegisterIndexer),
    // but the abstraction matches HLSL's CBV/UAV/SRV structure for code consistency.
    enum class RegType : uint8_t { CBV = 0,
                                   UAV = 1,
                                   SRV = 2 };
    // Counter starts after fixed-position items (ConstantValue, SamplerHeap, CBuffer).
    // These items have hardcoded register indices matching HLSL's convention.
    uint reg_count = _has_argument_buffer ? 1u : 0u;
    auto next_reg = [&](RegType) -> uint { return reg_count++; };

    vstd::vector<const Type *> buffer_elem_types;
    vstd::vector<luisa::string> buffer_names;// per-property variable names

    // Push constant at space=0 reg=0 — does NOT go into _properties to avoid creating
    // a descriptor set layout binding (push constants are handled via vkCmdPushConstants).
    // The ConstantValue property is added only to buffer_elem_types/buffer_names for
    // correlation with _property_ids[0] in emit.cpp.
    buffer_elem_types.push_back(nullptr);
    buffer_names.emplace_back("dsp_c");

    // Sampler heap at space=1, reg=0 (fixed position, separate space).
    // Its descriptor count is the same ABI constant used to bound dynamic
    // selector indices during instruction emission.
    _properties.emplace_back(
        Property{
            ShaderVariableType::SamplerHeap,
            1u,
            0u,
            spirv_configured_sampler_heap_size});
    buffer_elem_types.push_back(nullptr);
    buffer_names.emplace_back("samplers");

    // CBuffer (global argument buffer) — fixed at reg=0, but bumps counter for subsequent args
    if (_has_argument_buffer) {
        _properties.emplace_back(
            Property{
                ShaderVariableType::StructuredBuffer,
                0,
                0u,
                1});
        buffer_elem_types.push_back(nullptr);
        buffer_names.emplace_back("_Global");
    }

    // Plan a portable std140 block before emitting any UBO-related SPIR-V.
    // Vulkan only guarantees a 16 KiB maxUniformBufferRange, so constants that
    // do not fit remain ordinary OpConstant values. Rejected members do not
    // consume prefix space, allowing a later smaller member to fit.
    struct PlannedConstantUBOMember {
        const xir::Constant *constant;
        const Type *element_type;
        ConstantUBOElementLayout element_layout;
        ConstantUBOMemberLayout member_layout;
    };
    luisa::vector<PlannedConstantUBOMember> planned_ubo_members;
    planned_ubo_members.reserve(_ubo_array_constants.size());
    ConstantUBOLayoutPlanner ubo_layout_planner;
    for (auto constant : _ubo_array_constants) {
        auto element_type = constant->type()->element();
        // UBO lowering is optional. If the physical device cannot legally
        // store this narrow scalar width in Uniform storage, retain the
        // ordinary OpConstant path instead of turning an optimization into a
        // new target-feature requirement.
        if (!constant_ubo_storage_supported(
                element_type, _target_features)) {
            continue;
        }
        auto element_layout = constant_ubo_element_layout(element_type);
        auto member_layout = ubo_layout_planner.try_append(
            element_layout.base_alignment,
            element_layout.occupied_size,
            constant->type()->dimension());
        if (!member_layout) { continue; }
        planned_ubo_members.emplace_back(PlannedConstantUBOMember{
            .constant = constant,
            .element_type = element_type,
            .element_layout = element_layout,
            .member_layout = member_layout});
    }
    auto planned_ubo_size = ubo_layout_planner.size_bytes();

    // From this point on, the list represents only constants actually lowered
    // into Uniform storage. This keeps feature/capability analysis in emit()
    // consistent with the selected ABI.
    _ubo_array_constants.clear();
    _ubo_array_constants.reserve(planned_ubo_members.size());
    for (auto &&member : planned_ubo_members) {
        _ubo_array_constants.emplace_back(member.constant);
    }

    // Lower selected large array constants to a single std140 UBO so dynamic
    // indexing hits the GPU constant cache instead of materializing the array.
    if (!planned_ubo_members.empty()) {
        _has_constant_ubo = true;
        std::vector<spv::Id> member_types;
        std::vector<size_t> member_offsets;
        std::vector<const Type *> member_matrices;
        member_types.reserve(planned_ubo_members.size());
        member_offsets.reserve(planned_ubo_members.size());
        member_matrices.reserve(planned_ubo_members.size());
        _constant_ubo_data.resize(planned_ubo_size);
        for (uint32_t member_idx = 0u;
             auto &&member : planned_ubo_members) {
            auto constant = member.constant;
            auto element_type = member.element_type;
            auto element_count = constant->type()->dimension();
            auto array_stride = member.member_layout.array_stride;
            auto member_offset = member.member_layout.member_offset;
            member_offsets.push_back(member_offset);

            spv::Id spv_elem_type;
            if (element_type->is_structure() || element_type->is_array()) {
                spv_elem_type = _convert_laid_out_type(element_type);
            } else {
                spv_elem_type = _convert_type(element_type, Usage::READ);
            }
            auto array_size_id = _builder.makeUintConstant(element_count);
            // The third-party builder uses a signed stride only to distinguish
            // explicitly laid-out array types while SPIR-V stores the literal
            // as uint32_t. Avoid routing an ABI value through that signed API.
            auto array_type = _builder.makeArrayType(
                spv_elem_type, array_size_id, 1);
            // makeArrayType marks the type "explicitly laid out" but does not emit the
            // ArrayStride decoration; SPIR-V validation requires it on Block-decorated arrays.
            add_u32_decoration(
                _builder, array_type, spv::Decoration::ArrayStride,
                static_cast<uint32_t>(array_stride));
            member_types.push_back(array_type);
            member_matrices.push_back(
                matrix_after_array_layers(element_type));

            _ubo_constant_member_indices.emplace(constant, member_idx);

            auto src = static_cast<const std::byte *>(constant->data());
            for (uint32_t i = 0u; i < element_count; ++i) {
                auto dst_element = _constant_ubo_data.data() +
                                   member_offset + i * array_stride;
                auto src_element = src + i * element_type->size();
                if (element_type->is_matrix()) {
                    auto column = Type::vector(
                        element_type->element(), element_type->dimension());
                    for (auto column_index = 0u;
                         column_index < element_type->dimension();
                         ++column_index) {
                        std::memcpy(
                            dst_element +
                                column_index *
                                    member.element_layout.matrix_stride,
                            src_element + column_index * column->size(),
                            column->size());
                    }
                } else {
                    std::memcpy(dst_element, src_element,
                                element_type->size());
                }
            }
            ++member_idx;
        }
        LUISA_ASSERT(
            _constant_ubo_data.size() == planned_ubo_size &&
                planned_ubo_size <= portable_constant_ubo_max_range,
            "SPIR-V constant UBO layout exceeded its planned portable range.");

        auto struct_type = _builder.makeStructType(member_types, {}, "_ConstantUBO", false);
        _builder.addDecoration(struct_type, spv::Decoration::Block);
        for (uint32_t i = 0u; i < member_offsets.size(); ++i) {
            add_u32_member_decoration(
                _builder, struct_type, i, spv::Decoration::Offset,
                static_cast<uint32_t>(member_offsets[i]));
            _builder.addMemberDecoration(struct_type, i, spv::Decoration::NonWritable);
            if (auto *matrix = member_matrices[i]) {
                auto *column = Type::vector(matrix->element(),
                                            matrix->dimension());
                _builder.addMemberDecoration(struct_type, i,
                                             spv::Decoration::ColMajor);
                add_u32_member_decoration(
                    _builder, struct_type, i,
                    spv::Decoration::MatrixStride,
                    static_cast<uint32_t>(
                        constant_ubo_element_layout(matrix).matrix_stride));
            }
        }
        _constant_ubo_var = _builder.createVariable(
            spv::NoPrecision, spv::StorageClass::Uniform, struct_type, "_constant_ubo");

        _properties.emplace_back(
            Property{
                ShaderVariableType::ConstantBuffer,
                0u,
                next_reg(RegType::CBV),
                1u});
        buffer_elem_types.push_back(nullptr);
        buffer_names.emplace_back("_constant_ubo");
    }

    // Bindless resources: spaces start at 2 for SPIR-V
    uint space_idx = 2;
    if (_use_buffer_bindless) {
        _properties.emplace_back(
            Property{
                ShaderVariableType::SRVBufferHeap,
                space_idx++,
                0u,
                std::numeric_limits<uint>::max()});
        buffer_elem_types.push_back(nullptr);
        buffer_names.emplace_back("bdls");
    }
    if (_use_tex2d_bindless) {
        _properties.emplace_back(
            Property{
                ShaderVariableType::SRVTextureHeap,
                space_idx++,
                0u,
                std::numeric_limits<uint>::max()});
        buffer_elem_types.push_back(nullptr);
        buffer_names.emplace_back("tex2d_heap");
    }
    if (_use_tex3d_bindless) {
        _properties.emplace_back(
            Property{
                ShaderVariableType::SRVTextureHeap,
                space_idx++,
                0u,
                std::numeric_limits<uint>::max()});
        buffer_elem_types.push_back(nullptr);
        buffer_names.emplace_back("tex3d_heap");
    }

    // Kernel arguments — use RegType matching HLSL's register type selection
    for (auto &&arg : kernel.arguments()) {
        switch (arg.type()->tag()) {
            case Type::Tag::TEXTURE: {
                auto &binding = add_resource_binding(arg);
                if (has_usage(binding.usage, Usage::READ)) {
                    binding.read_property_index = _properties.size();
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::SRVTextureHeap,
                            0,
                            next_reg(RegType::SRV),
                            1});
                    buffer_elem_types.push_back(arg.type());
                    buffer_names.emplace_back(
                        luisa::string("_tx_") + vstd::to_string(arg.uid()));
                }
                if (has_usage(binding.usage, Usage::WRITE)) {
                    binding.write_property_index = _properties.size();
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::UAVTextureHeap,
                            0,
                            next_reg(RegType::UAV),
                            1});
                    buffer_elem_types.push_back(arg.type());
                    buffer_names.emplace_back(
                        luisa::string("_tx_rw_") + vstd::to_string(arg.uid()));
                }
                break;
            }
            case Type::Tag::BUFFER: {
                auto &binding = add_resource_binding(arg);
                auto property_index = _properties.size();
                if (is_writable(arg)) {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::RWStructuredBuffer,
                            0,
                            next_reg(RegType::UAV),
                            1});
                } else {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::StructuredBuffer,
                            0,
                            next_reg(RegType::SRV),
                            1});
                }
                buffer_elem_types.push_back(arg.type());// Store full buffer type for _convert_type cache
                buffer_names.emplace_back(luisa::string("_buf_") + vstd::to_string(arg.uid()));
                if (has_usage(binding.usage, Usage::READ)) {
                    binding.read_property_index = property_index;
                }
                if (has_usage(binding.usage, Usage::WRITE)) {
                    binding.write_property_index = property_index;
                }
                break;
            }
            case Type::Tag::BINDLESS_ARRAY: {
                auto &binding = add_resource_binding(arg);
                binding.read_property_index = _properties.size();
                _properties.emplace_back(
                    Property{
                        ShaderVariableType::StructuredBuffer,
                        0,
                        next_reg(RegType::SRV),
                        1});
                buffer_elem_types.push_back(nullptr);
                buffer_names.emplace_back(luisa::string("_bdarr_") + vstd::to_string(arg.uid()));
                if (binding.requires_bindless_buffer_metadata) {
                    binding.bindless_buffer_metadata_property_index =
                        _properties.size();
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::SPIRVBindlessBufferMetadata,
                            0,
                            next_reg(RegType::SRV),
                            1});
                    buffer_elem_types.push_back(nullptr);
                    buffer_names.emplace_back(
                        luisa::string("_bdmeta_") +
                        vstd::to_string(arg.uid()));
                }
                break;
            }
            case Type::Tag::ACCEL: {
                auto &binding = add_resource_binding(arg);
                auto usage = luisa::to_underlying(binding.usage);
                auto writes = (usage & luisa::to_underlying(Usage::WRITE)) != 0u;
                if (binding.requires_accel_traversal_descriptor) {
                    binding.read_property_index = _properties.size();
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::SPIRVAccel,
                            0,
                            next_reg(RegType::SRV),
                            1});
                    buffer_elem_types.push_back(nullptr);
                    buffer_names.emplace_back(luisa::string("_accel_") + vstd::to_string(arg.uid()));
                }
                if (binding.requires_accel_instance_buffer) {
                    binding.accel_instance_property_index = _properties.size();
                    _properties.emplace_back(
                        Property{
                            writes ? ShaderVariableType::SPIRVAccelInstanceRW :
                                     ShaderVariableType::SPIRVAccelInstance,
                            0,
                            next_reg(writes ? RegType::UAV : RegType::SRV),
                            1});
                    buffer_elem_types.push_back(nullptr);
                    buffer_names.emplace_back(
                        luisa::string(writes ? "_accel_rw_" : "_accel_inst_") +
                        vstd::to_string(arg.uid()));
                }
                LUISA_ASSERT(
                    binding.requires_accel_traversal_descriptor ||
                        binding.requires_accel_instance_buffer ||
                        binding.usage == Usage::NONE,
                    "SPIR-V accel argument has nonempty usage but no exact descriptor role.");
                break;
            }
            case Type::Tag::CUSTOM:
                if (arg.type()->description() == "LC_IndirectDispatchBuffer"sv) {
                    auto &binding = add_resource_binding(arg);
                    binding.usage = Usage::WRITE;
                    binding.write_property_index = _properties.size();
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::RWStructuredBuffer,
                            0,
                            next_reg(RegType::UAV),
                            1});
                    buffer_elem_types.push_back(nullptr);
                    buffer_names.emplace_back("_indirect_dispatch");
                }
                break;
            default:
                break;
        }
    }
    if (_allow_indirect_dispatch) {
        _properties.emplace_back(
            Property{
                ShaderVariableType::SPIRVIndirectDispatch,
                0u,
                next_reg(RegType::SRV),
                1u});
        buffer_elem_types.push_back(nullptr);
        buffer_names.emplace_back("_indirect_dispatch_source");
    }
    auto resource_argument_count = std::ranges::count_if(
        kernel.arguments(), [&](const Variable &argument) noexcept {
            return is_kernel_resource_variable(argument);
        });
    LUISA_ASSERT(_kernel_resource_bindings.size() == resource_argument_count,
                 "SPIR-V planned {} bindings for {} kernel resource arguments.",
                 _kernel_resource_bindings.size(), resource_argument_count);
    auto has_property = [](size_t index) noexcept {
        return index != invalid_resource_property_index;
    };
    auto property_is = [&](size_t index, ShaderVariableType type) noexcept {
        return has_property(index) && index < _properties.size() &&
               _properties[index].type == type;
    };
    for (size_t i = 0u; i < _kernel_resource_bindings.size(); ++i) {
        auto &binding = _kernel_resource_bindings[i];
        auto reads = has_usage(binding.usage, Usage::READ);
        auto writes = has_usage(binding.usage, Usage::WRITE);
        switch (binding.type_tag) {
            case Type::Tag::TEXTURE:
                LUISA_ASSERT(
                    property_is(binding.read_property_index,
                                ShaderVariableType::SRVTextureHeap) == reads &&
                        property_is(binding.write_property_index,
                                    ShaderVariableType::UAVTextureHeap) == writes,
                    "SPIR-V texture resource {} has an inconsistent sampled/storage binding contract.",
                    i);
                break;
            case Type::Tag::BUFFER: {
                auto property_index = reads ? binding.read_property_index :
                                              binding.write_property_index;
                auto property_type = writes ?
                                         ShaderVariableType::RWStructuredBuffer :
                                         ShaderVariableType::StructuredBuffer;
                LUISA_ASSERT((reads || writes) &&
                                 property_is(property_index, property_type) &&
                                 (!reads || !writes ||
                                  binding.read_property_index ==
                                      binding.write_property_index),
                             "SPIR-V buffer resource {} has an inconsistent read/write binding contract.",
                             i);
                break;
            }
            case Type::Tag::BINDLESS_ARRAY:
                LUISA_ASSERT(
                    property_is(binding.read_property_index,
                                ShaderVariableType::StructuredBuffer) &&
                        property_is(
                            binding.bindless_buffer_metadata_property_index,
                            ShaderVariableType::SPIRVBindlessBufferMetadata) ==
                            binding.requires_bindless_buffer_metadata,
                    "SPIR-V bindless resource {} has an inconsistent "
                    "index/metadata binding contract.",
                    i);
                break;
            case Type::Tag::ACCEL:
                LUISA_ASSERT(
                    property_is(binding.read_property_index,
                                ShaderVariableType::SPIRVAccel) ==
                            binding.requires_accel_traversal_descriptor &&
                        property_is(
                            binding.accel_instance_property_index,
                            writes ?
                                ShaderVariableType::SPIRVAccelInstanceRW :
                                ShaderVariableType::SPIRVAccelInstance) ==
                            binding.requires_accel_instance_buffer,
                    "SPIR-V accel resource {} has an inconsistent traversal/instance binding contract.",
                    i);
                break;
            case Type::Tag::CUSTOM:
                LUISA_ASSERT(
                    !reads && writes &&
                        property_is(
                            binding.write_property_index,
                            ShaderVariableType::RWStructuredBuffer),
                    "SPIR-V indirect-dispatch resource {} has an inconsistent "
                    "writable binding contract.",
                    i);
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported SPIR-V kernel resource type tag {} in binding contract.",
                    static_cast<uint32_t>(binding.type_tag));
        }
    }
    auto emitted_bindless_metadata = std::any_of(
        _properties.cbegin(), _properties.cend(),
        [](const Property &property) noexcept {
            return property.type ==
                   ShaderVariableType::SPIRVBindlessBufferMetadata;
        });
    LUISA_ASSERT(
        emitted_bindless_metadata == _use_buffer_bindless_metadata,
        "SPIR-V aggregate bindless metadata plan ({}) disagrees with the "
        "exact per-argument descriptor roles ({}).",
        _use_buffer_bindless_metadata, emitted_bindless_metadata);

    // Print buffers
    if (kernel.requires_printing()) {
        _properties.emplace_back(
            Property{
                ShaderVariableType::RWStructuredBuffer,
                0,
                next_reg(RegType::UAV),
                1});
        buffer_elem_types.push_back(nullptr);
        buffer_names.emplace_back("_printCounter");
        _properties.emplace_back(
            Property{
                ShaderVariableType::RWStructuredBuffer,
                0,
                next_reg(RegType::UAV),
                1});
        buffer_elem_types.push_back(nullptr);
        buffer_names.emplace_back("_printBuffer");
    }
    // NonWritable is a statement about the memory backing a declaration, not
    // merely about the access path through that declaration. User resources
    // may be imported from externally aliased Vulkan memory, so a read-only
    // storage-buffer declaration is provably immutable only when this module
    // has no writable user-resource path at all. This deliberately includes
    // writable textures, accel instance storage, bindless buffers, and the
    // custom indirect-dispatch buffer instead of guessing aliasability from a
    // Type::Tag subset.
    auto may_alias_writable_user_resource = std::ranges::any_of(
        _kernel_resource_bindings,
        [&](const KernelResourceBinding &binding) noexcept {
            return has_usage(binding.usage, Usage::WRITE);
        });
    LUISA_ASSERT(buffer_elem_types.size() == _properties.size() + 1u &&
                     buffer_names.size() == _properties.size() + 1u,
                 "SPIR-V descriptor properties, element types, and names are out of sync.");

    // Create SPIR-V global variables and add OpDecorate for property bindings
    _property_ids.clear();
    _property_ids.reserve(_properties.size());

    auto make_typed_buffer_struct_type = [&](const Type *elem_type, bool writable, const char *name) -> spv::Id {
        if (elem_type == nullptr) {
            // Untyped buffer (e.g., cbuffer / global argument buffer)
            auto uint_type = _builder.makeUintType(32);
            auto runtime_array = _builder.makeRuntimeArray(uint_type);
            add_u32_decoration(
                _builder, runtime_array,
                spv::Decoration::ArrayStride, 4u);
            auto struct_type = _builder.makeStructType({runtime_array}, {}, name, false);
            _builder.addDecoration(struct_type, spv::Decoration::Block);
            add_u32_member_decoration(
                _builder, struct_type, 0u,
                spv::Decoration::Offset, 0u);
            if (!writable && !may_alias_writable_user_resource) {
                _builder.addMemberDecoration(struct_type, 0, spv::Decoration::NonWritable);
            } else {
                _builder.addMemberDecoration(struct_type, 0, spv::Decoration::Coherent);
            }
            return struct_type;
        }
        // Use _convert_type to ensure cached type consistency with callable parameters.
        auto buffer_type = Type::buffer(elem_type);
        return _convert_type(buffer_type, writable ? Usage::WRITE : Usage::READ);
    };
    auto make_buffer_struct_type = [&](const char *name) -> spv::Id {
        auto uint_type = _builder.makeUintType(32);
        auto runtime_array = _builder.makeRuntimeArray(uint_type);
        add_u32_decoration(
            _builder, runtime_array,
            spv::Decoration::ArrayStride, 4u);
        auto struct_type = _builder.makeStructType({runtime_array}, {}, name, false);
        _builder.addDecoration(struct_type, spv::Decoration::Block);
        add_u32_member_decoration(
            _builder, struct_type, 0u,
            spv::Decoration::Offset, 0u);
        return struct_type;
    };
    auto get_image_sampled_type_and_dim = [&](const Type *elem_type, size_t name_idx) -> std::pair<spv::Id, spv::Dim> {
        spv::Id sampled_type = _builder.makeFloatType(32);
        spv::Dim dim = spv::Dim::Dim2D;
        if (elem_type != nullptr && elem_type->tag() == Type::Tag::TEXTURE) {
            auto tex_elem = elem_type->element();
            if (tex_elem != nullptr && tex_elem->is_vector()) { tex_elem = tex_elem->element(); }
            if (tex_elem != nullptr) {
                if (tex_elem->is_float32()) {
                    sampled_type = _builder.makeFloatType(32);
                } else if (tex_elem->is_int32()) {
                    sampled_type = _builder.makeIntType(32);
                } else if (tex_elem->is_uint32()) {
                    sampled_type = _builder.makeUintType(32);
                }
            }
            dim = (elem_type->dimension() == 3) ? spv::Dim::Dim3D : spv::Dim::Dim2D;
        } else if (elem_type == nullptr) {
            if (buffer_names[name_idx + 1] == "tex3d_heap") {
                dim = spv::Dim::Dim3D;
            }
        }
        return {sampled_type, dim};
    };

    // Create push constant variable first (always at _property_ids[0])
    {
        auto uint_type = _builder.makeUintType(32);
        auto uint4_type = _builder.makeVectorType(uint_type, 4);
        auto struct_type = _builder.makeStructType(
            {uint4_type, uint4_type}, {}, "_PushConstant", false);
        _builder.addDecoration(struct_type, spv::Decoration::Block);
        add_u32_member_decoration(
            _builder, struct_type, 0u,
            spv::Decoration::Offset, 0u);
        add_u32_member_decoration(
            _builder, struct_type, 1u,
            spv::Decoration::Offset, 16u);
        auto var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::PushConstant, struct_type, "dsp_c");
        _property_ids.emplace_back(var);
    }

    for (size_t i = 0; i < _properties.size(); ++i) {
        auto &&prop = _properties[i];
        // buffer_elem_types and buffer_names still include the push constant at index 0,
        // so offset by 1 to match _properties (which no longer includes ConstantValue).
        auto elem_type = buffer_elem_types[i + 1];
        auto var_name = (i + 1) < buffer_names.size() ? buffer_names[i + 1].c_str() : "resource";
        spv::Id var = spv::NoResult;
        switch (prop.type) {
            case ShaderVariableType::ConstantBuffer: {
                var = _constant_ubo_var;
                add_u32_decoration(_builder, var,
                                   spv::Decoration::DescriptorSet,
                                   prop.space_index);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::Binding,
                                   prop.register_index);
                break;
            }
            case ShaderVariableType::SamplerHeap: {
                auto sampler_type = _builder.makeSamplerType("sampler");
                auto array_size_id = _builder.makeUintConstant(prop.array_size);
                auto array_type = _builder.makeArrayType(sampler_type, array_size_id, 0);
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, var_name);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::DescriptorSet,
                                   prop.space_index);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::Binding,
                                   prop.register_index);
                break;
            }
            case ShaderVariableType::StructuredBuffer:
            case ShaderVariableType::RWStructuredBuffer:
            case ShaderVariableType::SPIRVAccelInstance:
            case ShaderVariableType::SPIRVAccelInstanceRW:
            case ShaderVariableType::SPIRVBindlessBufferMetadata:
            case ShaderVariableType::SPIRVIndirectDispatch: {
                bool writable =
                    prop.type == ShaderVariableType::RWStructuredBuffer ||
                    prop.type == ShaderVariableType::SPIRVAccelInstanceRW;
                bool is_untyped = elem_type == nullptr;
                spv::Id struct_type;
                if (is_untyped && luisa::string_view{var_name}.starts_with("_bdarr_")) {
                    // Bindless array: use _convert_type to ensure type consistency with callable parameters
                    struct_type = _convert_type(Type::from("bindless_array"), Usage::READ);
                } else if (elem_type != nullptr && prop.type == ShaderVariableType::StructuredBuffer) {
                    // For typed read-only buffers, use _convert_type to share the cached type
                    struct_type = _convert_type(elem_type, Usage::READ);
                } else if (elem_type != nullptr && prop.type == ShaderVariableType::RWStructuredBuffer) {
                    struct_type = _convert_type(elem_type, Usage::READ_WRITE);
                } else {
                    struct_type = make_typed_buffer_struct_type(elem_type, writable, "_Buffer");
                }
                spv::StorageClass storage = spv::StorageClass::StorageBuffer;
                var = _builder.createVariable(spv::NoPrecision, storage, struct_type, var_name);
                if (luisa::string_view{var_name} == "_Global") {
                    _argument_buffer_id = var;
                } else if (prop.type ==
                           ShaderVariableType::SPIRVIndirectDispatch) {
                    _indirect_dispatch_buffer_id = var;
                }
                add_u32_decoration(_builder, var,
                                   spv::Decoration::DescriptorSet,
                                   prop.space_index);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::Binding,
                                   prop.register_index);
                // Luisa permits the same resource, or overlapping views of it,
                // to be bound through distinct kernel arguments. Under the
                // Vulkan memory model, separate memory-object declarations are
                // assumed not to alias unless every potentially aliasing
                // declaration carries Aliased.
                _builder.addDecoration(var, spv::Decoration::Aliased);
                // Only add Coherent when necessary:
                // - buffer is used with atomics, or
                // - element type contains bool (word-level storage causes false sharing).
                // Coherent forces GPU to bypass caches; for disjoint writes this is pure overhead.
                if (!writable && elem_type != nullptr &&
                    !may_alias_writable_user_resource) {
                    // Typed read-only buffer: add NonWritable so the Vulkan driver
                    // can optimize a genuinely immutable user-buffer access.
                    _builder.addDecoration(var, spv::Decoration::NonWritable);
                }
                if (writable && elem_type != nullptr) {
                    bool needs_coherent = _needs_atomic_buffer_types.contains(elem_type);
                    if (!needs_coherent) {
                        if (auto elem = elem_type->element();
                            elem != nullptr &&
                            spirv_type_contains_bool(elem)) {
                            needs_coherent = true;
                        }
                    }
                    if (needs_coherent) {
                        _builder.addDecoration(var, spv::Decoration::Coherent);
                    }
                }
                break;
            }
            case ShaderVariableType::SRVTextureHeap: {
                auto [sampled_type, dim] = get_image_sampled_type_and_dim(elem_type, i);
                auto image_type = elem_type == nullptr ?
                                      _builder.makeImageType(sampled_type, dim, false, false, false,
                                                             1, spv::ImageFormat::Unknown, "image") :
                                      _convert_type(elem_type, Usage::READ);
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    _require_target_feature(
                        target_feature::descriptor_indexing,
                        _target_features.descriptor_indexing);
                    _require_target_feature(
                        target_feature::runtime_descriptor_array,
                        _target_features.runtime_descriptor_array);
                    _require_target_feature(
                        target_feature::descriptor_binding_partially_bound,
                        _target_features.descriptor_binding_partially_bound);
                    _require_target_feature(
                        target_feature::descriptor_binding_sampled_image_update_after_bind,
                        _target_features.descriptor_binding_sampled_image_update_after_bind);
                    _builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::RuntimeDescriptorArray);
                    auto array_type = _builder.makeRuntimeArray(image_type);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, var_name);
                } else if (prop.array_size == 1) {
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, image_type, var_name);
                } else {
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(image_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, var_name);
                }
                add_u32_decoration(_builder, var,
                                   spv::Decoration::DescriptorSet,
                                   prop.space_index);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::Binding,
                                   prop.register_index);
                _builder.addDecoration(var, spv::Decoration::Aliased);
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    if (buffer_names[i + 1] == "tex2d_heap") {
                        _tex2d_heap_id = var;
                    } else if (buffer_names[i + 1] == "tex3d_heap") {
                        _tex3d_heap_id = var;
                    }
                }
                break;
            }
            case ShaderVariableType::UAVTextureHeap: {
                LUISA_ASSERT(
                    prop.array_size == 1u,
                    "Vulkan native XIR-to-SPIR-V codegen does not support "
                    "storage-image descriptor arrays; the required "
                    "StorageImageArrayNonUniformIndexing feature is not part "
                    "of the native artifact contract.");
                auto [sampled_type, dim] = get_image_sampled_type_and_dim(elem_type, i);
                auto image_type = elem_type == nullptr ?
                                      _builder.makeImageType(sampled_type, dim, false, false, false,
                                                             2, spv::ImageFormat::Unknown, "image") :
                                      _convert_type(elem_type, Usage::WRITE);
                var = _builder.createVariable(
                    spv::NoPrecision, spv::StorageClass::UniformConstant,
                    image_type, var_name);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::DescriptorSet,
                                   prop.space_index);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::Binding,
                                   prop.register_index);
                _builder.addDecoration(var, spv::Decoration::Aliased);
                break;
            }
            case ShaderVariableType::SRVBufferHeap:
            case ShaderVariableType::UAVBufferHeap: {
                bool writable = (prop.type == ShaderVariableType::UAVBufferHeap);
                auto struct_type = make_buffer_struct_type("_BindlessBuffer");
                spv::StorageClass storage = spv::StorageClass::StorageBuffer;
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    LUISA_ASSERT(
                        !writable,
                        "Vulkan native XIR-to-SPIR-V codegen does not produce "
                        "unbounded writable buffer heaps.");
                    _require_target_feature(
                        target_feature::descriptor_indexing,
                        _target_features.descriptor_indexing);
                    _require_target_feature(
                        target_feature::runtime_descriptor_array,
                        _target_features.runtime_descriptor_array);
                    _require_target_feature(
                        target_feature::descriptor_binding_partially_bound,
                        _target_features.descriptor_binding_partially_bound);
                    _require_target_feature(
                        target_feature::descriptor_binding_storage_buffer_update_after_bind,
                        _target_features.descriptor_binding_storage_buffer_update_after_bind);
                    _builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::RuntimeDescriptorArray);
                    auto array_type = _builder.makeRuntimeArray(struct_type);
                    var = _builder.createVariable(spv::NoPrecision, storage, array_type, var_name);
                } else if (prop.array_size > 1) {
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(struct_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, storage, array_type, var_name);
                } else {
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(struct_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, storage, array_type, var_name);
                }
                add_u32_decoration(_builder, var,
                                   spv::Decoration::DescriptorSet,
                                   prop.space_index);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::Binding,
                                   prop.register_index);
                _builder.addDecoration(var, spv::Decoration::Aliased);
                _buffer_heap_id = var;
                break;
            }
            case ShaderVariableType::SPIRVAccel: {
                _require_target_feature(target_feature::ray_query,
                                        _target_features.ray_query);
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant,
                                              _builder.makeAccelerationStructureType(), var_name);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::DescriptorSet,
                                   prop.space_index);
                add_u32_decoration(_builder, var,
                                   spv::Decoration::Binding,
                                   prop.register_index);
                break;
            }
            default:
                break;
        }
        _property_ids.emplace_back(var);
    }
    LUISA_ASSERT(
        !_allow_indirect_dispatch ||
            _indirect_dispatch_buffer_id != spv::NoResult,
        "SPIR-V native compute shader is missing its indirect-dispatch "
        "metadata buffer.");
}

}// namespace lc::spirv
