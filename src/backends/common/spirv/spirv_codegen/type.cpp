#include "entry.h"
#include "buffer_layout.h"
#include <luisa/core/logging.h>

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

void decorate_matrix_layout(spv::Builder &builder, spv::Id struct_type,
                            uint32_t member_index,
                            const Type *member_type) noexcept {
    if (auto *matrix = matrix_after_array_layers(member_type)) {
        auto *column =
            Type::vector(matrix->element(), matrix->dimension());
        builder.addMemberDecoration(struct_type, member_index,
                                    spv::Decoration::ColMajor);
        add_u32_member_decoration(
            builder, struct_type, member_index,
            spv::Decoration::MatrixStride,
            static_cast<uint32_t>(column->size()));
    }
}

}// namespace

bool SpirvCodegenEntry::_buffer_uses_word_storage(const Type *type) noexcept {
    if (type == nullptr || !type->is_buffer()) { return false; }
    auto elem_type = type->element();
    if (elem_type == nullptr) { return false; }
    if (auto iter = _atomic_buffer_storage_plans.find(type);
        iter != _atomic_buffer_storage_plans.end()) {
        LUISA_ASSERT(iter->second != SpirvAtomicBufferStoragePlan::CONFLICT,
                     "Conflicting SPIR-V atomic-buffer storage plan escaped analysis.");
        return iter->second == SpirvAtomicBufferStoragePlan::WORD;
    }
    // Logical bool has no StorageBuffer representation. Some valid Luisa host
    // aggregates also cannot be expressed with Vulkan's standard typed-SSBO
    // alignment (notably a 64-bit vec3/vec4 placed at a 16-byte host offset).
    // Both cases retain the byte-exact uint32 word ABI.
    return !spirv_typed_buffer_layout_compatible(elem_type);
}

spv::Id SpirvCodegenEntry::_convert_type(const Type *type, Usage usage) noexcept {
    if (type == nullptr) { return _builder.makeVoidType(); }
    if (type->tag() == Type::Tag::TEXTURE) {
        auto &image_type_map =
            (luisa::to_underlying(usage) & luisa::to_underlying(Usage::WRITE)) != 0u ?
                _storage_image_type_map :
                _sampled_image_type_map;
        if (auto it = image_type_map.find(type); it != image_type_map.end()) { return it->second; }
    }
    if (auto it = _type_map.find(type); it != _type_map.end()) { return it->second; }
    spv::Id id = spv::NoResult;
    switch (type->tag()) {
        case Type::Tag::BOOL: id = _builder.makeBoolType(); break;
        case Type::Tag::FLOAT16:
            id = _builder.makeFloatType(16);
            break;
        case Type::Tag::FLOAT32: id = _builder.makeFloatType(32); break;
        case Type::Tag::FLOAT64:
            _require_target_feature(target_feature::shader_float64,
                                    _target_features.shader_float64);
            id = _builder.makeFloatType(64);
            break;
        case Type::Tag::INT8:
            id = _builder.makeIntType(8);
            break;
        case Type::Tag::UINT8:
            id = _builder.makeUintType(8);
            break;
        case Type::Tag::INT16:
            id = _builder.makeIntType(16);
            break;
        case Type::Tag::UINT16:
            id = _builder.makeUintType(16);
            break;
        case Type::Tag::INT32: id = _builder.makeIntType(32); break;
        case Type::Tag::UINT32: id = _builder.makeUintType(32); break;
        case Type::Tag::INT64:
            _require_target_feature(target_feature::shader_int64,
                                    _target_features.shader_int64);
            id = _builder.makeIntType(64);
            break;
        case Type::Tag::UINT64:
            _require_target_feature(target_feature::shader_int64,
                                    _target_features.shader_int64);
            id = _builder.makeUintType(64);
            break;
        case Type::Tag::VECTOR:
            id = _builder.makeVectorType(_convert_type(type->element(), usage), static_cast<int32_t>(type->dimension()));
            break;
        case Type::Tag::MATRIX:
            id = _builder.makeMatrixType(_convert_type(type->element(), usage),
                                         static_cast<int32_t>(type->dimension()),
                                         static_cast<int32_t>(type->dimension()));
            break;
        case Type::Tag::ARRAY: {
            auto elem_type = _convert_type(type->element(), usage);
            auto size_id = _builder.makeUintConstant(static_cast<uint32_t>(type->dimension()));
            id = _builder.makeArrayType(elem_type, size_id, 0);
            break;
        }
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            std::vector<spv::Id> member_types;
            member_types.reserve(members.size());
            for (auto m : members) { member_types.emplace_back(_convert_type(m, usage)); }
            std::vector<spv::StructMemberDebugInfo> member_debug;
            id = _builder.makeStructType(member_types, member_debug, "Struct", false);
            break;
        }
        case Type::Tag::BUFFER: {
            auto elem_type = type->element();
            bool use_typed = elem_type != nullptr && !_buffer_uses_word_storage(type);
            spv::Id spv_elem_type;
            if (use_typed && (elem_type->is_structure() || elem_type->is_array())) {
                spv_elem_type = _convert_laid_out_type(elem_type);
            } else if (use_typed && elem_type != nullptr && elem_type->is_bool()) {
                spv_elem_type = _builder.makeUintType(32);
            } else if (use_typed && elem_type != nullptr && elem_type->is_vector() && elem_type->element()->is_bool()) {
                spv_elem_type = _builder.makeVectorType(_builder.makeUintType(32), static_cast<int32_t>(elem_type->dimension()));
            } else {
                spv_elem_type = use_typed ? _convert_type(elem_type, usage) : _builder.makeUintType(32);
            }
            if (use_typed && elem_type != nullptr) {
                _mark_8bit_storage_usage(elem_type, spv::StorageClass::StorageBuffer);
            }
            auto runtime_array = _builder.makeRuntimeArray(spv_elem_type);
            auto struct_type = _builder.makeStructType({runtime_array}, {}, "Buffer", false);
            add_u32_decoration(
                _builder, runtime_array, spv::Decoration::ArrayStride,
                use_typed ? static_cast<uint32_t>(elem_type->size()) : 4u);
            add_u32_member_decoration(
                _builder, struct_type, 0u,
                spv::Decoration::Offset, 0u);
            // Matrix elements in Block-decorated structs require ColMajor and
            // MatrixStride on the containing member. This also applies when
            // the member reaches a matrix through one or more array layers.
            if (use_typed) {
                decorate_matrix_layout(_builder, struct_type, 0u, elem_type);
            }
            _builder.addDecoration(struct_type, spv::Decoration::Block);
            id = struct_type;
            break;
        }
        case Type::Tag::TEXTURE: {
            auto elem = type->element();
            if (elem != nullptr && elem->is_vector()) { elem = elem->element(); }
            LUISA_ASSERT(elem != nullptr && (elem->is_float32() || elem->is_int32() || elem->is_uint32()),
                         "SPIR-V texture element must be float32, int32, or uint32, got {}.",
                         type->description());
            spv::Id sampled_type;
            if (elem->is_float32()) {
                sampled_type = _builder.makeFloatType(32);
            } else if (elem->is_int32()) {
                sampled_type = _builder.makeIntType(32);
            } else {
                sampled_type = _builder.makeUintType(32);
            }
            spv::Dim dim = (type->dimension() == 3) ? spv::Dim::Dim3D : spv::Dim::Dim2D;
            bool is_writable = (static_cast<uint32_t>(usage) & static_cast<uint32_t>(Usage::WRITE)) != 0;
            uint32_t sampled = is_writable ? 2 : 1;
            spv::ImageFormat fmt = spv::ImageFormat::Unknown;
            id = _builder.makeImageType(sampled_type, dim, false, false, false,
                                        sampled, fmt, "image");
            break;
        }
        case Type::Tag::BINDLESS_ARRAY: {
            auto uint_type = _builder.makeUintType(32);
            auto runtime_array = _builder.makeRuntimeArray(uint_type);
            auto struct_type = _builder.makeStructType({runtime_array}, {}, "BindlessArray", false);
            add_u32_decoration(
                _builder, runtime_array,
                spv::Decoration::ArrayStride, 4u);
            add_u32_member_decoration(
                _builder, struct_type, 0u,
                spv::Decoration::Offset, 0u);
            _builder.addMemberDecoration(struct_type, 0, spv::Decoration::NonWritable);
            _builder.addDecoration(struct_type, spv::Decoration::Block);
            id = struct_type;
            break;
        }
        case Type::Tag::ACCEL:
            _require_target_feature(target_feature::ray_query,
                                    _target_features.ray_query);
            _builder.addExtension(spv::E_SPV_KHR_ray_query);
            _builder.addCapability(spv::Capability::RayQueryKHR);
            id = _builder.makeAccelerationStructureType();
            break;
        case Type::Tag::FLOAT8_E4M3:
            _require_target_feature(target_feature::shader_float8,
                                    _target_features.shader_float8);
            _uses_float8 = true;
            id = _builder.makeFloatE4M3Type();
            break;
        case Type::Tag::FLOAT8_E5M2:
            _require_target_feature(target_feature::shader_float8,
                                    _target_features.shader_float8);
            _uses_float8 = true;
            id = _builder.makeFloatE5M2Type();
            break;
        case Type::Tag::COOPERATIVE_VECTOR: {
            _require_target_feature(target_feature::cooperative_vector,
                                    _target_features.cooperative_vector);
            _uses_cooperative_vector = true;
            _builder.setMemoryModel(spv::AddressingModel::Logical,
                                    spv::MemoryModel::Vulkan);
            _builder.addExtension(spv::E_SPV_KHR_vulkan_memory_model);
            _builder.addCapability(spv::Capability::VulkanMemoryModel);
            _builder.addExtension(spv::E_SPV_NV_cooperative_vector);
            _builder.addCapability(spv::Capability::CooperativeVectorNV);
            auto component = _convert_type(type->element(), usage);
            auto count = _builder.makeUintConstant(static_cast<uint32_t>(type->dimension()));
            id = _builder.makeCooperativeVectorTypeNV(component, count);
            break;
        }
        case Type::Tag::CUSTOM: {
            auto desc = type->description();
            if (desc == "LC_RayQueryAll" || desc == "LC_RayQueryAny") {
                _require_target_feature(target_feature::ray_query,
                                        _target_features.ray_query);
                _builder.addExtension(spv::E_SPV_KHR_ray_query);
                _builder.addCapability(spv::Capability::RayQueryKHR);
                id = _builder.makeRayQueryType();
            } else {
                LUISA_NOT_IMPLEMENTED("SPIR-V type conversion for resource/custom type {}.", desc);
            }
            break;
        }
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to convert type {}.", type->description());
    if (type->tag() == Type::Tag::TEXTURE) {
        auto &image_type_map =
            (luisa::to_underlying(usage) & luisa::to_underlying(Usage::WRITE)) != 0u ?
                _storage_image_type_map :
                _sampled_image_type_map;
        image_type_map.emplace(type, id);
    } else {
        _type_map.emplace(type, id);
    }
    return id;
}

spv::Id SpirvCodegenEntry::_convert_laid_out_type(const Type *type) noexcept {
    if (type == nullptr) { return _builder.makeVoidType(); }
    if (auto it = _laid_out_type_map.find(type); it != _laid_out_type_map.end()) { return it->second; }
    spv::Id id = spv::NoResult;
    switch (type->tag()) {
        case Type::Tag::ARRAY: {
            auto elem_layout = _convert_laid_out_type(type->element());
            auto size_id = _builder.makeUintConstant(static_cast<uint32_t>(type->dimension()));
            auto stride = static_cast<uint32_t>(type->element()->size());
            // glslang's signed stride parameter is only an interning marker;
            // emit the actual unsigned SPIR-V literal ourselves so layouts at
            // the top of uint32_t's range cannot cross a signed boundary.
            id = _builder.makeArrayType(elem_layout, size_id, 1);
            add_u32_decoration(
                _builder, id, spv::Decoration::ArrayStride, stride);
            break;
        }
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            std::vector<spv::Id> member_types;
            member_types.reserve(members.size());
            for (auto m : members) {
                if (m->is_structure() || m->is_array()) {
                    member_types.emplace_back(_convert_laid_out_type(m));
                } else {
                    member_types.emplace_back(_convert_type(m, Usage::READ_WRITE));
                }
            }
            std::vector<spv::StructMemberDebugInfo> member_debug;
            id = _builder.makeStructType(member_types, member_debug, "Struct", false);
            size_t offset = 0u;
            for (uint32_t i = 0; i < members.size(); ++i) {
                auto m = members[i];
                offset = luisa::align(offset, m->alignment());
                add_u32_member_decoration(
                    _builder, id, i, spv::Decoration::Offset,
                    static_cast<uint32_t>(offset));
                decorate_matrix_layout(_builder, id, i, m);
                offset += m->size();
            }
            break;
        }
        case Type::Tag::BOOL:
            id = _builder.makeUintType(32);
            break;
        default:
            if (type->is_vector() && type->element()->is_bool()) {
                id = _builder.makeVectorType(_builder.makeUintType(32), static_cast<int32_t>(type->dimension()));
            } else {
                id = _convert_type(type, Usage::READ_WRITE);
            }
            break;
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to convert laid-out type {}.", type->description());
    _laid_out_type_map.emplace(type, id);
    return id;
}

void SpirvCodegenEntry::_mark_8bit_storage_usage(const Type *type, spv::StorageClass storage) noexcept {
    if (type == nullptr) { return; }
    auto mark_storage = [&]() noexcept {
        switch (storage) {
            case spv::StorageClass::StorageBuffer: _uses_8bit_storage_buffer = true; break;
            case spv::StorageClass::Uniform: _uses_8bit_uniform_storage = true; break;
            case spv::StorageClass::PushConstant: _uses_8bit_push_constant = true; break;
            default: break;
        }
    };
    if (type->tag() == Type::Tag::INT8 || type->tag() == Type::Tag::UINT8) {
        mark_storage();
    } else if (type->tag() == Type::Tag::FLOAT8_E4M3 || type->tag() == Type::Tag::FLOAT8_E5M2) {
        _uses_float8 = true;
        mark_storage();
    } else if (type->is_structure()) {
        for (auto m : type->members()) { _mark_8bit_storage_usage(m, storage); }
    } else if (type->is_array()) {
        _mark_8bit_storage_usage(type->element(), storage);
    } else if (type->is_vector() || type->is_matrix()) {
        _mark_8bit_storage_usage(type->element(), storage);
    }
}

}// namespace lc::spirv
