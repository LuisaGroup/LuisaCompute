#include "entry.h"
#include <algorithm>
#include <luisa/core/logging.h>
#include <limits>

namespace lc::spirv {

void SpirvCodegenEntry::generate_binding(Function kernel, luisa::span<const std::pair<Variable, Usage>> argument_usages) {
    _properties.clear();
    _use_tex2d_bindless = false;
    _use_tex3d_bindless = false;
    _use_buffer_bindless = false;

    auto argument_usage = [&](const Variable &v) noexcept {
        for (auto &&[arg, usage] : argument_usages) {
            if (arg.uid() == v.uid()) { return usage; }
        }
        return kernel.variable_usage(v.uid());
    };
    auto is_writable = [&](const Variable &v) {
        return (static_cast<uint>(argument_usage(v)) & static_cast<uint>(Usage::WRITE)) != 0;
    };

    // Detect cbuffer non-empty: any argument that is not a resource or builtin
    bool cbuffer_non_empty = false;
    for (auto &&arg : kernel.arguments()) {
        auto tag = arg.tag();
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

    // Detect bindless usage from propagated builtin callables
    const auto &builtins = kernel.propagated_builtin_callables();
    auto uses_buffer_bindless = [&]() noexcept -> bool {
        static constexpr CallOp ops[] = {
            CallOp::BINDLESS_BUFFER_SIZE,
            CallOp::BINDLESS_BUFFER_READ,
            CallOp::BINDLESS_BUFFER_WRITE,
            CallOp::BINDLESS_BYTE_BUFFER_READ,
            CallOp::BINDLESS_BUFFER_TYPE,
            CallOp::BINDLESS_BUFFER_ADDRESS,
            CallOp::UNIFORM_BINDLESS_BUFFER_SIZE,
            CallOp::UNIFORM_BINDLESS_BUFFER_READ,
            CallOp::UNIFORM_BINDLESS_BUFFER_WRITE,
            CallOp::UNIFORM_BINDLESS_BYTE_BUFFER_READ,
            CallOp::UNIFORM_BINDLESS_BUFFER_TYPE,
            CallOp::UNIFORM_BINDLESS_BUFFER_ADDRESS,
            CallOp::TYPED_BINDLESS_BUFFER_SIZE,
            CallOp::TYPED_BINDLESS_BUFFER_READ,
            CallOp::TYPED_BINDLESS_BUFFER_WRITE,
            CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_SIZE,
            CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_READ,
            CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_WRITE,
        };
        return std::ranges::any_of(ops, [&](auto op) { return builtins.test(op); });
    };
    auto uses_tex2d_bindless = [&]() noexcept -> bool {
        static constexpr CallOp ops[] = {
            CallOp::BINDLESS_TEXTURE2D_SAMPLE,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::BINDLESS_TEXTURE2D_READ,
            CallOp::BINDLESS_TEXTURE2D_READ_LEVEL,
            CallOp::BINDLESS_TEXTURE2D_SIZE,
            CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_READ,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_READ_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SIZE,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SIZE_LEVEL,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
        };
        return std::ranges::any_of(ops, [&](auto op) { return builtins.test(op); });
    };
    auto uses_tex3d_bindless = [&]() noexcept -> bool {
        static constexpr CallOp ops[] = {
            CallOp::BINDLESS_TEXTURE3D_SAMPLE,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::BINDLESS_TEXTURE3D_READ,
            CallOp::BINDLESS_TEXTURE3D_READ_LEVEL,
            CallOp::BINDLESS_TEXTURE3D_SIZE,
            CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_READ,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_READ_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SIZE,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SIZE_LEVEL,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,
        };
        return std::ranges::any_of(ops, [&](auto op) { return builtins.test(op); });
    };

    _use_buffer_bindless = uses_buffer_bindless();
    _use_tex2d_bindless = uses_tex2d_bindless();
    _use_tex3d_bindless = uses_tex3d_bindless();

    // Register indexer matching HLSL's RegisterType pattern.
    // For SPIR-V, all register types share the same flat counter (like SpirVRegisterIndexer),
    // but the abstraction matches HLSL's CBV/UAV/SRV structure for code consistency.
    enum class RegType : uint8_t { CBV = 0, UAV = 1, SRV = 2 };
    // Counter starts after fixed-position items (ConstantValue, SamplerHeap, CBuffer).
    // These items have hardcoded register indices matching HLSL's convention.
    uint reg_count = cbuffer_non_empty ? 1u : 0u;
    auto next_reg = [&](RegType) -> uint { return reg_count++; };

    uint bind_count = 2;
    vstd::vector<const Type *> buffer_elem_types;
    vstd::vector<luisa::string> buffer_names;  // per-property variable names

    // Push constant at space=0 reg=0 — does NOT go into _properties to avoid creating
    // a descriptor set layout binding (push constants are handled via vkCmdPushConstants).
    // The ConstantValue property is added only to buffer_elem_types/buffer_names for
    // correlation with _property_ids[0] in emit.cpp.
    buffer_elem_types.push_back(nullptr);
    buffer_names.emplace_back("dsp_c");

    // Sampler heap at space=1, reg=0, size=16 (fixed position, separate space)
    _properties.emplace_back(
        Property{
            ShaderVariableType::SamplerHeap,
            1u,
            0u,
            16u});
    buffer_elem_types.push_back(nullptr);
    buffer_names.emplace_back("samplers");

    // CBuffer (global argument buffer) — fixed at reg=0, but bumps counter for subsequent args
    if (cbuffer_non_empty) {
        _properties.emplace_back(
            Property{
                ShaderVariableType::StructuredBuffer,
                0,
                0u,
                1});
        buffer_elem_types.push_back(nullptr);
        buffer_names.emplace_back("_Global");
        bind_count += 2;
    }

    // Lower large array constants to a single std140 UBO so dynamic indexing
    // hits the GPU constant cache instead of being inlined as OpConstantComposite.
    if (!_ubo_array_constants.empty()) {
        _has_constant_ubo = true;
        std::vector<spv::Id> member_types;
        std::vector<size_t> member_offsets;
        size_t data_offset = 0u;
        for (uint32_t member_idx = 0u; auto c : _ubo_array_constants) {
            auto elem_type = c->type()->element();
            auto elem_count = c->type()->dimension();
            auto elem_size = elem_type->size();
            auto elem_align = elem_type->alignment();
            // std140 requires array stride and member alignment to be multiples of 16.
            auto array_stride = (elem_size + 15u) & ~15u;
            auto member_align = std::max<size_t>(elem_align, 16u);
            data_offset = (data_offset + member_align - 1u) & ~(member_align - 1u);
            member_offsets.push_back(data_offset);

            spv::Id spv_elem_type;
            if (elem_type->is_structure() || elem_type->is_array()) {
                spv_elem_type = _convert_laid_out_type(elem_type);
            } else {
                spv_elem_type = _convert_type(elem_type, Usage::READ);
            }
            auto array_size_id = _builder.makeUintConstant(elem_count);
            auto array_type = _builder.makeArrayType(spv_elem_type, array_size_id, array_stride);
            // makeArrayType marks the type "explicitly laid out" but does not emit the
            // ArrayStride decoration; SPIR-V validation requires it on Block-decorated arrays.
            _builder.addDecoration(array_type, spv::Decoration::ArrayStride, static_cast<int>(array_stride));
            member_types.push_back(array_type);

            _ubo_constant_member_by_hash.emplace(c->hash(), member_idx);

            _constant_ubo_data.resize(data_offset + array_stride * elem_count);
            auto src = static_cast<const std::byte *>(c->data());
            for (uint32_t i = 0u; i < elem_count; ++i) {
                std::memcpy(_constant_ubo_data.data() + data_offset + i * array_stride,
                            src + i * elem_size, elem_size);
            }
            data_offset += array_stride * elem_count;
            ++member_idx;
        }

        auto struct_type = _builder.makeStructType(member_types, {}, "_ConstantUBO", false);
        _builder.addDecoration(struct_type, spv::Decoration::Block);
        for (uint32_t i = 0u; i < member_offsets.size(); ++i) {
            _builder.addMemberDecoration(struct_type, i, spv::Decoration::Offset,
                                         static_cast<int>(member_offsets[i]));
            _builder.addMemberDecoration(struct_type, i, spv::Decoration::NonWritable);
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
        bind_count += 1u;
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
        bind_count += 1;
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
        bind_count += 1;
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
        bind_count += 1;
    }

    // Kernel arguments — use RegType matching HLSL's register type selection
    for (auto &&arg : kernel.arguments()) {
        switch (arg.type()->tag()) {
            case Type::Tag::TEXTURE:
                if (is_writable(arg)) {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::UAVTextureHeap,
                            0,
                            next_reg(RegType::UAV),
                            1});
                } else {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::SRVTextureHeap,
                            0,
                            next_reg(RegType::SRV),
                            1});
                }
                buffer_elem_types.push_back(arg.type());
                buffer_names.emplace_back(luisa::string("_tx_") + vstd::to_string(arg.uid()));
                bind_count += 1;
                break;
            case Type::Tag::BUFFER:
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
                buffer_elem_types.push_back(arg.type()); // Store full buffer type for _convert_type cache
                buffer_names.emplace_back(luisa::string("_buf_") + vstd::to_string(arg.uid()));
                bind_count += 2;
                break;
            case Type::Tag::BINDLESS_ARRAY:
                _properties.emplace_back(
                    Property{
                        ShaderVariableType::StructuredBuffer,
                        0,
                        next_reg(RegType::SRV),
                        1});
                buffer_elem_types.push_back(nullptr);
                buffer_names.emplace_back(luisa::string("_bdarr_") + vstd::to_string(arg.uid()));
                bind_count += 2;
                break;
            case Type::Tag::ACCEL:
                if (is_writable(arg)) {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::RWStructuredBuffer,
                            0,
                            next_reg(RegType::UAV),
                            1});
                    buffer_elem_types.push_back(nullptr);
                    buffer_names.emplace_back(luisa::string("_accel_rw_") + vstd::to_string(arg.uid()));
                    bind_count += 2;
                } else {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::SPIRVAccel,
                            0,
                            next_reg(RegType::SRV),
                            1});
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::StructuredBuffer,
                            0,
                            next_reg(RegType::SRV),
                            1});
                    buffer_elem_types.push_back(nullptr);
                    buffer_elem_types.push_back(nullptr);
                    buffer_names.emplace_back(luisa::string("_accel_") + vstd::to_string(arg.uid()));
                    buffer_names.emplace_back(luisa::string("_accel_inst_") + vstd::to_string(arg.uid()));
                    bind_count += 3;
                }
                break;
            case Type::Tag::CUSTOM:
                if (arg.type()->description() == "LC_IndirectDispatchBuffer"sv) {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::RWStructuredBuffer,
                            0,
                            next_reg(RegType::UAV),
                            1});
                    buffer_elem_types.push_back(nullptr);
                    buffer_names.emplace_back("_indirect_dispatch");
                    bind_count += 2;
                }
                break;
            default:
                break;
        }
    }

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
        bind_count += 4;
    }

    // Create SPIR-V global variables and add OpDecorate for property bindings
    _property_ids.clear();
    _property_ids.reserve(_properties.size());

    auto make_typed_buffer_struct_type = [&](const Type *elem_type, bool writable, const char *name) -> spv::Id {
        if (elem_type == nullptr) {
            // Untyped buffer (e.g., cbuffer / global argument buffer)
            auto uint_type = _builder.makeUintType(32);
            auto runtime_array = _builder.makeRuntimeArray(uint_type);
            _builder.addDecoration(runtime_array, spv::Decoration::ArrayStride, 4);
            auto struct_type = _builder.makeStructType({runtime_array}, {}, name, false);
            _builder.addDecoration(struct_type, spv::Decoration::Block);
            _builder.addMemberDecoration(struct_type, 0, spv::Decoration::Offset, 0);
            if (!writable) {
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
        _builder.addDecoration(runtime_array, spv::Decoration::ArrayStride, 4);
        auto struct_type = _builder.makeStructType({runtime_array}, {}, name, false);
        _builder.addDecoration(struct_type, spv::Decoration::Block);
        _builder.addMemberDecoration(struct_type, 0, spv::Decoration::Offset, 0);
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
        auto struct_type = _builder.makeStructType({uint4_type}, {}, "_PushConstant", false);
        _builder.addDecoration(struct_type, spv::Decoration::Block);
        _builder.addMemberDecoration(struct_type, 0, spv::Decoration::Offset, 0);
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
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                break;
            }
            case ShaderVariableType::SamplerHeap: {
                auto sampler_type = _builder.makeSamplerType("sampler");
                auto array_size_id = _builder.makeUintConstant(prop.array_size);
                auto array_type = _builder.makeArrayType(sampler_type, array_size_id, 0);
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, var_name);
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                break;
            }
            case ShaderVariableType::StructuredBuffer:
            case ShaderVariableType::RWStructuredBuffer: {
                bool writable = (prop.type == ShaderVariableType::RWStructuredBuffer);
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
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                _builder.addDecoration(var, spv::Decoration::Aliased);
                if (writable) {
                    _builder.addDecoration(var, spv::Decoration::Coherent);
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
                    _builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::RuntimeDescriptorArray);
                    _builder.addCapability(spv::Capability::ShaderNonUniformEXT);
                    _builder.addCapability(spv::Capability::SampledImageArrayNonUniformIndexingEXT);
                    auto array_type = _builder.makeRuntimeArray(image_type);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, var_name);
                } else if (prop.array_size == 1) {
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, image_type, var_name);
                } else {
                    _builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::ShaderNonUniformEXT);
                    _builder.addCapability(spv::Capability::SampledImageArrayNonUniformIndexingEXT);
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(image_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, var_name);
                }
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
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
                auto [sampled_type, dim] = get_image_sampled_type_and_dim(elem_type, i);
                auto image_type = elem_type == nullptr ?
                                      _builder.makeImageType(sampled_type, dim, false, false, false,
                                                             2, spv::ImageFormat::Unknown, "image") :
                                      _convert_type(elem_type, Usage::WRITE);
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    _builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::RuntimeDescriptorArray);
                    _builder.addCapability(spv::Capability::ShaderNonUniformEXT);
                    _builder.addCapability(spv::Capability::StorageImageArrayNonUniformIndexingEXT);
                    auto array_type = _builder.makeRuntimeArray(image_type);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, var_name);
                } else if (prop.array_size == 1) {
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, image_type, var_name);
                } else {
                    _builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::ShaderNonUniformEXT);
                    _builder.addCapability(spv::Capability::StorageImageArrayNonUniformIndexingEXT);
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(image_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, var_name);
                }
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    if (buffer_names[i + 1] == "tex2d_heap") {
                        _tex2d_heap_id = var;
                    } else if (buffer_names[i + 1] == "tex3d_heap") {
                        _tex3d_heap_id = var;
                    }
                }
                break;
            }
            case ShaderVariableType::SRVBufferHeap:
            case ShaderVariableType::UAVBufferHeap: {
                bool writable = (prop.type == ShaderVariableType::UAVBufferHeap);
                auto struct_type = make_buffer_struct_type("_BindlessBuffer");
                spv::StorageClass storage = spv::StorageClass::StorageBuffer;
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    _builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::RuntimeDescriptorArray);
                    _builder.addCapability(spv::Capability::ShaderNonUniformEXT);
                    _builder.addCapability(spv::Capability::StorageBufferArrayNonUniformIndexingEXT);
                    auto array_type = _builder.makeRuntimeArray(struct_type);
                    var = _builder.createVariable(spv::NoPrecision, storage, array_type, var_name);
                } else if (prop.array_size > 1) {
                    _builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::ShaderNonUniformEXT);
                    _builder.addCapability(spv::Capability::StorageBufferArrayNonUniformIndexingEXT);
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(struct_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, storage, array_type, var_name);
                } else {
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(struct_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, storage, array_type, var_name);
                }
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                _buffer_heap_id = var;
                break;
            }
            case ShaderVariableType::SPIRVAccel: {
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant,
                                              _builder.makeAccelerationStructureType(), var_name);
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                break;
            }
            default:
                break;
        }
        _property_ids.emplace_back(var);
    }
}

}// namespace lc::spirv
