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
        case Type::Tag::BUFFER:
        /*
TODO: according to 'bin\debug\shader.spv' definition of type_StructuredBuffer_uint and . generate type like 'type_RWStructuredBuffer_float' and 'type_StructuredBuffer_uint'. according to usage, if the usage no contain write,  OpMemberDecorate with NonWritable.
        */
        case Type::Tag::TEXTURE:
        /*
TODO: according to 'bin\debug\shader.spv':
- for 2 dimension, no write usage, definition like type_2d_image
- for 2 dimension, write usage, definition like type_2d_image_0
- for 3 dimension, no write usage, definition like type_3d_image
- for 3 dimension, write usage, definition like type_3d_image_0
        */
        case Type::Tag::BINDLESS_ARRAY:
/*
TODO: according to 'bin\debug\hlsl_output_bindless.hlsl' and 'bin\debug\shader_bindless.spv'
- define a global ByteAddressBuffer bindless 'bdls' at register 0, space 2
- define a global Texture2D<float4> bindless '_BindlessTex' at register 0, space 3
*/
        case Type::Tag::ACCEL:
        case Type::Tag::CUSTOM:
            LUISA_NOT_IMPLEMENTED("SPIR-V type conversion for resource/custom type {}.", type->description());
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to convert type {}.", type->description());
    _type_map.emplace(type, id);
    return id;
}
}