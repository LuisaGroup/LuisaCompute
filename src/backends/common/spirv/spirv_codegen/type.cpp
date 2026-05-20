#include "entry.h"
#include <luisa/core/logging.h>

namespace lc::spirv {
spv::Id SpirvCodegenEntry::_convert_type(const Type *type, Usage usage) noexcept {
    if (type == nullptr) { return _builder.makeVoidType(); }
    if (auto it = _type_map.find(type); it != _type_map.end()) { return it->second; }
    spv::Id id = spv::NoResult;
    switch (type->tag()) {
        case Type::Tag::BOOL: id = _builder.makeBoolType(); break;
        case Type::Tag::FLOAT16: id = _builder.makeFloatType(16); break;
        case Type::Tag::FLOAT32: id = _builder.makeFloatType(32); break;
        case Type::Tag::FLOAT64: id = _builder.makeFloatType(64); break;
        case Type::Tag::INT8: id = _builder.makeIntType(8); break;
        case Type::Tag::UINT8: id = _builder.makeUintType(8); break;
        case Type::Tag::INT16: id = _builder.makeIntType(16); break;
        case Type::Tag::UINT16: id = _builder.makeUintType(16); break;
        case Type::Tag::INT32: id = _builder.makeIntType(32); break;
        case Type::Tag::UINT32: id = _builder.makeUintType(32); break;
        case Type::Tag::INT64: id = _builder.makeIntType(64); break;
        case Type::Tag::UINT64: id = _builder.makeUintType(64); break;
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
            // Use typed arrays for all buffer element types.
            // Fall back to uint32 only for buffers that need atomic access
            // (since SPIR-V atomic ops require scalar integer types).
            bool needs_atomic = _needs_atomic_buffer_types.contains(type);
            bool use_typed = elem_type != nullptr && !needs_atomic;
            spv::Id spv_elem_type;
            if (use_typed && (elem_type->is_structure() || elem_type->is_array())) {
                spv_elem_type = _convert_laid_out_type(elem_type);
            } else {
                spv_elem_type = use_typed ? _convert_type(elem_type, usage) : _builder.makeUintType(32);
            }
            auto runtime_array = _builder.makeRuntimeArray(spv_elem_type);
            auto struct_type = _builder.makeStructType({runtime_array}, {}, "Buffer", false);
            _builder.addDecoration(runtime_array, spv::Decoration::ArrayStride, use_typed ? static_cast<int32_t>(elem_type->size()) : 4);
            _builder.addMemberDecoration(struct_type, 0, spv::Decoration::Offset, 0);
            // Matrix elements in Block-decorated structs require ColMajor and MatrixStride decorations.
            if (use_typed && elem_type->is_matrix()) {
                auto col_type = Type::vector(elem_type->element(), elem_type->dimension());
                _builder.addMemberDecoration(struct_type, 0, spv::Decoration::ColMajor);
                _builder.addMemberDecoration(struct_type, 0, spv::Decoration::MatrixStride, static_cast<int32_t>(col_type->size()));
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
            spv::ImageFormat storage_format;
            if (elem->is_float32()) {
                sampled_type = _builder.makeFloatType(32);
                storage_format = spv::ImageFormat::Rgba32f;
            } else if (elem->is_int32()) {
                sampled_type = _builder.makeIntType(32);
                storage_format = spv::ImageFormat::Rgba32i;
            } else {
                sampled_type = _builder.makeUintType(32);
                storage_format = spv::ImageFormat::Rgba32ui;
            }
            spv::Dim dim = (type->dimension() == 3) ? spv::Dim::Dim3D : spv::Dim::Dim2D;
            bool is_writable = (static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) != 0;
            uint32_t sampled = is_writable ? 2 : 1;
            spv::ImageFormat fmt = is_writable ? storage_format : spv::ImageFormat::Unknown;
            id = _builder.makeImageType(sampled_type, dim, false, false, false,
                                        sampled, fmt, "image");
            break;
        }
        case Type::Tag::BINDLESS_ARRAY: {
            auto uint_type = _builder.makeUintType(32);
            auto runtime_array = _builder.makeRuntimeArray(uint_type);
            auto struct_type = _builder.makeStructType({runtime_array}, {}, "BindlessArray", false);
            _builder.addDecoration(runtime_array, spv::Decoration::ArrayStride, 4);
            _builder.addMemberDecoration(struct_type, 0, spv::Decoration::Offset, 0);
            _builder.addMemberDecoration(struct_type, 0, spv::Decoration::NonWritable);
            _builder.addDecoration(struct_type, spv::Decoration::Block);
            id = struct_type;
            break;
        }
        case Type::Tag::ACCEL:
            _builder.addExtension(spv::E_SPV_KHR_ray_query);
            _builder.addCapability(spv::Capability::RayQueryKHR);
            id = _builder.makeAccelerationStructureType();
            break;
        case Type::Tag::CUSTOM: {
            auto desc = type->description();
            if (desc == "LC_RayQueryAll" || desc == "LC_RayQueryAny") {
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
    _type_map.emplace(type, id);
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
            auto stride = static_cast<int32_t>(type->element()->size());
            id = _builder.makeArrayType(elem_layout, size_id, stride);
            _builder.addDecoration(id, spv::Decoration::ArrayStride, stride);
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
                _builder.addMemberDecoration(id, i, spv::Decoration::Offset, static_cast<int32_t>(offset));
                if (m->is_matrix()) {
                    auto col_type = Type::vector(m->element(), m->dimension());
                    _builder.addMemberDecoration(id, i, spv::Decoration::ColMajor);
                    _builder.addMemberDecoration(id, i, spv::Decoration::MatrixStride,
                                                 static_cast<int32_t>(col_type->size()));
                }
                offset += m->size();
            }
            break;
        }
        default:
            id = _convert_type(type, Usage::READ_WRITE);
            break;
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to convert laid-out type {}.", type->description());
    _laid_out_type_map.emplace(type, id);
    return id;
}

}// namespace lc::spirv
