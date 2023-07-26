#ifdef LUISA_ENABLE_IR

#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include "metal_codegen_ir.h"

namespace luisa::compute::metal {

size_t MetalCodegenIR::type_size_bytes(const ir::Type *type) noexcept {

    auto primitive_size = [](auto p) noexcept {
        switch (p) {
            case ir::Primitive::Bool: return sizeof(bool);
            case ir::Primitive::Int16: return sizeof(int16_t);
            case ir::Primitive::Uint16: return sizeof(uint16_t);
            case ir::Primitive::Int32: return sizeof(int32_t);
            case ir::Primitive::Uint32: return sizeof(uint32_t);
            case ir::Primitive::Int64: return sizeof(int64_t);
            case ir::Primitive::Uint64: return sizeof(uint64_t);
            case ir::Primitive::Float32: return sizeof(float);
            case ir::Primitive::Float64: return sizeof(double);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION(
            "Unknown primitive type '{}'.",
            luisa::to_string(p));
    };
    switch (type->tag) {
        case ir::Type::Tag::Primitive: {
            return primitive_size(type->primitive._0);
        }
        case ir::Type::Tag::Vector: {
            auto &&[elem, n] = type->vector._0;
            LUISA_ASSERT(elem.tag == ir::VectorElementType::Tag::Scalar,
                         "Cannot get size of vector with non-scalar element type.");
            auto elem_size = primitive_size(elem.scalar._0);
            switch (n) {
                case 2u: return elem_size * 2u;
                case 3u: return elem_size * 4u;
                case 4u: return elem_size * 4u;
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION(
                "Cannot get size of vector with {} elements.", n);
        }
        case ir::Type::Tag::Matrix: {
            auto &&[elem, n] = type->matrix._0;
            LUISA_ASSERT(elem.tag == ir::VectorElementType::Tag::Scalar,
                         "Cannot get size of matrix with non-scalar element type.");
            auto elem_size = primitive_size(elem.scalar._0);
            switch (n) {
                case 2u: return elem_size * 2u * 2u;
                case 3u: return elem_size * 4u * 3u;
                case 4u: return elem_size * 4u * 4u;
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION(
                "Cannot get size of matrix with {} elements.", n);
        }
        case ir::Type::Tag::Struct: {
            auto &&[fields, alignment, size] = type->struct_._0;
            return size;
        }
        case ir::Type::Tag::Array: {
            auto &&[elem, n] = type->array._0;
            auto elem_size = type_size_bytes(elem.get());
            return elem_size * n;
        }
        default: LUISA_ERROR_WITH_LOCATION(
            "Cannot get size of type '{}'.",
            luisa::to_string(type->tag));
    }
}

}// namespace luisa::compute::metal

#endif

