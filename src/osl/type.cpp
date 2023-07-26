#include <luisa/core/logging.h>
#include <luisa/osl/type.h>

#include <utility>

namespace luisa::compute::osl {

ClosureType::ClosureType(const Type *gentype) noexcept
    : Type{Tag::CLOSURE}, _gentype{gentype} {
    LUISA_ASSERT(gentype->tag() == Tag::SIMPLE &&
                     static_cast<const SimpleType *>(gentype)->primitive() ==
                         SimpleType::Primitive::COLOR,
                 "Closure type must be color.");
}

luisa::string_view ClosureType::identifier() const noexcept {
    LUISA_ASSERT(_gentype->tag() == Tag::SIMPLE &&
                     static_cast<const SimpleType *>(_gentype)->primitive() ==
                         SimpleType::Primitive::COLOR,
                 "Closure type must be color.");
    using namespace std::string_view_literals;
    return "closure color"sv;
}

string_view SimpleType::identifier() const noexcept {
    using namespace std::string_view_literals;
    switch (_primitive) {
        case Primitive::INT: return "int"sv;
        case Primitive::FLOAT: return "float"sv;
        case Primitive::POINT: return "point"sv;
        case Primitive::NORMAL: return "normal"sv;
        case Primitive::VECTOR: return "vector"sv;
        case Primitive::COLOR: return "color"sv;
        case Primitive::MATRIX: return "matrix"sv;
        case Primitive::STRING: return "string"sv;
    }
    LUISA_ERROR_WITH_LOCATION("Unknown primitive type.");
}

StructType::StructType(luisa::string identifier, vector<StructType::Field> fields) noexcept
    : Type{Tag::STRUCT},
      _identifier{std::move(identifier)},
      _fields{std::move(fields)} {}

luisa::string StructType::dump() const noexcept {
    auto s = luisa::format("struct {} {{", _identifier);
    if (!_fields.empty()) {
        for (auto &&f : _fields) {
            s.append(luisa::format(" {} {}", f.type->identifier(), f.name));
            if (f.is_array()) {
                s.append(luisa::format("[{}]", f.array_length));
            }
            s.append(";");
        }
        s.append(" ");
    }
    s.append("};");
    return s;
}

}// namespace luisa::compute::osl
