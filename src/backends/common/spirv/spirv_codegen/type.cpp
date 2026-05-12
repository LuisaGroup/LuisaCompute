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
            id = _builder.makeVectorType(_convert_type(type->element(), usage), static_cast<int>(type->dimension()));
            break;
        case Type::Tag::MATRIX:
            id = _builder.makeMatrixType(_convert_type(type->element(), usage),
                                         static_cast<int>(type->dimension()),
                                         static_cast<int>(type->dimension()));
            break;
        case Type::Tag::ARRAY: {
            auto elem_type = _convert_type(type->element(), usage);
            auto size_id = _builder.makeUintConstant(static_cast<unsigned>(type->dimension()));
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
            spv::Id elem_spv_type;
            auto elem = type->element();
            if (elem != nullptr && !elem->is_buffer()) {
                elem_spv_type = _convert_type(elem, usage);
            } else {
                elem_spv_type = _builder.makeUintType(32);
            }
            auto runtime_array = _builder.makeRuntimeArray(elem_spv_type);
            auto struct_type = _builder.makeStructType({runtime_array}, {}, "Buffer", false);
            size_t stride = (elem != nullptr) ? elem->size() : 4u;
            _builder.addDecoration(runtime_array, spv::Decoration::ArrayStride, static_cast<int>(stride));
            _builder.addMemberDecoration(struct_type, 0, spv::Decoration::Offset, 0);
            if ((static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) == 0) {
                _builder.addMemberDecoration(struct_type, 0, spv::Decoration::NonWritable);
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
            unsigned sampled = is_writable ? 2 : 1;
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
            id = _builder.makeAccelerationStructureType();
            break;
        case Type::Tag::CUSTOM:
            LUISA_NOT_IMPLEMENTED("SPIR-V type conversion for resource/custom type {}.", type->description());
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to convert type {}.", type->description());
    _type_map.emplace(type, id);
    return id;
}
}