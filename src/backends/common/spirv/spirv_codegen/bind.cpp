#include "entry.h"
#include <algorithm>
#include <luisa/core/logging.h>
#include <limits>

namespace lc::spirv {

void SpirvCodegenEntry::generate_binding(Function kernel) {
    _properties.clear();
    _use_tex2d_bindless = false;
    _use_tex3d_bindless = false;
    _use_buffer_bindless = false;

    auto is_writable = [&](const Variable &v) {
        return (static_cast<uint>(kernel.variable_usage(v.uid())) & static_cast<uint>(Usage::WRITE)) != 0;
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

    // Push constant / constant value at space=0 (fixed position, does not consume register slot)
    _properties.emplace_back(
        Property{
            ShaderVariableType::ConstantValue,
            0,
            0,
            1});
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
                buffer_elem_types.push_back(arg.type()->element());
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
    auto make_buffer_element_type = [&](const char *name) -> spv::Id {
        auto uint_type = _builder.makeUintType(32);
        auto runtime_array = _builder.makeRuntimeArray(uint_type);
        auto struct_type = _builder.makeStructType({runtime_array}, {}, name, false);
        // No Block decoration - this is used inside another Block-decorated struct
        return struct_type;
    };

    for (size_t i = 0; i < _properties.size(); ++i) {
        auto &&prop = _properties[i];
        auto elem_type = buffer_elem_types[i];
        auto var_name = i < buffer_names.size() ? buffer_names[i].c_str() : "resource";
        spv::Id var = spv::NoResult;
        switch (prop.type) {
            case ShaderVariableType::ConstantValue: {
                auto uint_type = _builder.makeUintType(32);
                auto uint4_type = _builder.makeVectorType(uint_type, 4);
                auto struct_type = _builder.makeStructType({uint4_type}, {}, "_PushConstant", false);
                _builder.addDecoration(struct_type, spv::Decoration::Block);
                _builder.addMemberDecoration(struct_type, 0, spv::Decoration::Offset, 0);
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::PushConstant, struct_type, var_name);
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
                spv::Id struct_type;
                if (elem_type == nullptr && luisa::string_view{var_name}.starts_with("_bdarr_")) {
                    // Bindless array: use _convert_type to ensure type consistency with callable parameters
                    struct_type = _convert_type(Type::from("bindless_array"), Usage::READ);
                } else {
                    struct_type = make_typed_buffer_struct_type(elem_type, writable, "_Buffer");
                }
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::StorageBuffer, struct_type, var_name);
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                // Align with HLSL globallycoherent: Coherent on writable buffer variables
                if (writable) {
                    _builder.addDecoration(var, spv::Decoration::Coherent);
                }
                break;
            }
            case ShaderVariableType::SRVTextureHeap: {
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
                    // Bindless heap: infer dimension from the variable name
                    if (buffer_names[i] == "tex3d_heap") {
                        dim = spv::Dim::Dim3D;
                    }
                }
                auto image_type = _builder.makeImageType(
                    sampled_type, dim, false, false, false, 1, spv::ImageFormat::Unknown, "image");
                if (prop.array_size == std::numeric_limits<uint>::max()) {
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
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    if (buffer_names[i] == "tex2d_heap") {
                        _tex2d_heap_id = var;
                    } else if (buffer_names[i] == "tex3d_heap") {
                        _tex3d_heap_id = var;
                    }
                }
                break;
            }
            case ShaderVariableType::UAVTextureHeap: {
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
                    // Bindless heap: infer dimension from the variable name
                    if (buffer_names[i] == "tex3d_heap") {
                        dim = spv::Dim::Dim3D;
                    }
                }
                auto image_type = _builder.makeImageType(
                    sampled_type, dim, false, false, false, 2, spv::ImageFormat::Unknown, "image");
                if (prop.array_size == std::numeric_limits<uint>::max()) {
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
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    _tex3d_heap_id = var;
                }
                break;
            }
            case ShaderVariableType::SRVBufferHeap:
            case ShaderVariableType::UAVBufferHeap: {
                auto struct_type = make_buffer_struct_type("_BindlessBuffer");
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    _builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                    _builder.addCapability(spv::Capability::RuntimeDescriptorArray);
                    auto array_type = _builder.makeRuntimeArray(struct_type);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::StorageBuffer, array_type, var_name);
                } else {
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(struct_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::StorageBuffer, array_type, var_name);
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
