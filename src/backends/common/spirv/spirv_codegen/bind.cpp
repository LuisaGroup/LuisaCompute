#include "entry.h"
#include <limits>

namespace lc::spirv {

void SpirvCodegenEntry::generate_binding(Function kernel) {
    _properties.clear();
    _use_tex2d_bindless = false;
    _use_tex3d_bindless = false;
    _use_buffer_bindless = false;

    auto is_writable = [&](Variable const &v) {
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
    auto const &builtins = kernel.propagated_builtin_callables();
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
        for (auto op : ops) {
            if (builtins.test(op)) return true;
        }
        return false;
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
        for (auto op : ops) {
            if (builtins.test(op)) return true;
        }
        return false;
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
        for (auto op : ops) {
            if (builtins.test(op)) return true;
        }
        return false;
    };

    _use_buffer_bindless = uses_buffer_bindless();
    _use_tex2d_bindless = uses_tex2d_bindless();
    _use_tex3d_bindless = uses_tex3d_bindless();

    uint bind_count = 2;
    uint register_count = 0;

    // SPIR-V: push-constant / constant value at space=0, reg=0
    _properties.emplace_back(
        Property{
            ShaderVariableType::ConstantValue,
            0,
            0,
            1});

    // Sampler heap at space=1, reg=0, size=16
    _properties.emplace_back(
        Property{
            ShaderVariableType::SamplerHeap,
            1u,
            0u,
            16u});

    // CBuffer (global argument buffer)
    if (cbuffer_non_empty) {
        register_count++;
        _properties.emplace_back(
            Property{
                ShaderVariableType::StructuredBuffer,
                0,
                0u,
                1});
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
        bind_count += 1;
    }
    if (_use_tex2d_bindless) {
        _properties.emplace_back(
            Property{
                ShaderVariableType::SRVTextureHeap,
                space_idx++,
                0u,
                std::numeric_limits<uint>::max()});
        bind_count += 1;
    }
    if (_use_tex3d_bindless) {
        _properties.emplace_back(
            Property{
                ShaderVariableType::SRVTextureHeap,
                space_idx++,
                0u,
                std::numeric_limits<uint>::max()});
        bind_count += 1;
    }

    // Kernel arguments
    for (auto &&arg : kernel.arguments()) {
        switch (arg.type()->tag()) {
            case Type::Tag::TEXTURE:
                if (is_writable(arg)) {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::UAVTextureHeap,
                            0,
                            register_count++,
                            1});
                    bind_count += 1;
                } else {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::SRVTextureHeap,
                            0,
                            register_count++,
                            1});
                    bind_count += 1;
                }
                break;
            case Type::Tag::BUFFER:
                if (is_writable(arg)) {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::RWStructuredBuffer,
                            0,
                            register_count++,
                            1});
                    bind_count += 2;
                } else {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::StructuredBuffer,
                            0,
                            register_count++,
                            1});
                    bind_count += 2;
                }
                break;
            case Type::Tag::BINDLESS_ARRAY:
                _properties.emplace_back(
                    Property{
                        ShaderVariableType::StructuredBuffer,
                        0,
                        register_count++,
                        1});
                bind_count += 2;
                break;
            case Type::Tag::ACCEL:
                if (is_writable(arg)) {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::RWStructuredBuffer,
                            0,
                            register_count++,
                            1});
                    bind_count += 2;
                } else {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::SPIRVAccel,
                            0,
                            register_count++,
                            1});
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::StructuredBuffer,
                            0,
                            register_count++,
                            1});
                    bind_count += 3;
                }
                break;
            case Type::Tag::CUSTOM:
                if (arg.type()->description() == "LC_IndirectDispatchBuffer"sv) {
                    _properties.emplace_back(
                        Property{
                            ShaderVariableType::RWStructuredBuffer,
                            0,
                            register_count++,
                            1});
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
                register_count++,
                1});
        _properties.emplace_back(
            Property{
                ShaderVariableType::RWStructuredBuffer,
                0,
                register_count++,
                1});
        bind_count += 4;
    }
    // Create SPIR-V global variables and add OpDecorate for property bindings
    _property_ids.clear();
    _property_ids.reserve(_properties.size());

    auto make_buffer_struct_type = [&](const char *name) -> spv::Id {
        auto uint_type = _builder.makeUintType(32);
        auto runtime_array = _builder.makeRuntimeArray(uint_type);
        auto struct_type = _builder.makeStructType({runtime_array}, {}, name, false);
        _builder.addDecoration(struct_type, spv::Decoration::Block);
        return struct_type;
    };

    for (auto &&prop : _properties) {
        spv::Id var = spv::NoResult;
        switch (prop.type) {
            case ShaderVariableType::ConstantValue: {
                auto uint_type = _builder.makeUintType(32);
                auto uint4_type = _builder.makeVectorType(uint_type, 4);
                auto struct_type = _builder.makeStructType({uint4_type}, {}, "_PushConstant", false);
                _builder.addDecoration(struct_type, spv::Decoration::Block);
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::PushConstant, struct_type, "dsp_c");
                break;
            }
            case ShaderVariableType::SamplerHeap: {
                auto sampler_type = _builder.makeSamplerType("sampler");
                auto array_size_id = _builder.makeUintConstant(prop.array_size);
                auto array_type = _builder.makeArrayType(sampler_type, array_size_id, 0);
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, "samplers");
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                break;
            }
            case ShaderVariableType::StructuredBuffer:
            case ShaderVariableType::RWStructuredBuffer: {
                auto struct_type = make_buffer_struct_type("_Buffer");
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::StorageBuffer, struct_type, "buffer");
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                break;
            }
            case ShaderVariableType::SRVTextureHeap: {
                auto image_type = _builder.makeImageType(
                    _builder.makeFloatType(32), spv::Dim::Dim2D, false, false, false, 1, spv::ImageFormat::Unknown, "image");
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    auto array_type = _builder.makeRuntimeArray(image_type);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, "textures");
                } else {
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(image_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, "textures");
                }
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                break;
            }
            case ShaderVariableType::UAVTextureHeap: {
                auto image_type = _builder.makeImageType(
                    _builder.makeFloatType(32), spv::Dim::Dim2D, false, false, false, 2, spv::ImageFormat::Unknown, "image");
                if (prop.array_size == std::numeric_limits<uint>::max()) {
                    auto array_type = _builder.makeRuntimeArray(image_type);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, "rwtextures");
                } else {
                    auto array_size_id = _builder.makeUintConstant(prop.array_size);
                    auto array_type = _builder.makeArrayType(image_type, array_size_id, 0);
                    var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant, array_type, "rwtextures");
                }
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                break;
            }
            case ShaderVariableType::SRVBufferHeap:
            case ShaderVariableType::UAVBufferHeap: {
                auto struct_type = make_buffer_struct_type("_BindlessBuffer");
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::StorageBuffer, struct_type, "bdls");
                _builder.addDecoration(var, spv::Decoration::DescriptorSet, static_cast<int>(prop.space_index));
                _builder.addDecoration(var, spv::Decoration::Binding, static_cast<int>(prop.register_index));
                break;
            }
            case ShaderVariableType::SPIRVAccel: {
                var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::UniformConstant,
                                              _builder.makeAccelerationStructureType(), "accel");
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
