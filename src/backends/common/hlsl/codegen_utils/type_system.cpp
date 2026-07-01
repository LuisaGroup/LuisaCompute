// Type System & Type Names

#include "../codegen_stack_data.h"
#include "../hlsl_codegen.h"

namespace lc::hlsl {

// Check if type is boolean
uint CodegenUtility::IsBool(Type const &type) {
    if (type.tag() == Type::Tag::BOOL) {
        return 1;
    } else if (type.tag() == Type::Tag::VECTOR && type.element()->tag() == Type::Tag::BOOL) {
        return type.dimension();
    }
    return 0;
};

// Register struct type for codegen
void CodegenUtility::RegistStructType(Type const *type) {
    if (type->is_structure() || type->is_array())
        opt->structTypes.try_emplace(type, opt->count++);
    else if (type->is_buffer()) {
        if (type->element())
            RegistStructType(type->element());
    }
}

// Generate HLSL type name
void CodegenUtility::GetTypeName(Type const &type, vstd::StringBuilder &str, Usage usage, bool local_var) {
    switch (type.tag()) {
        case Type::Tag::BOOL:
            if (!local_var)
                str << "int"sv;
            else
                str << "bool"sv;
            return;
        case Type::Tag::FLOAT32:
            str << "float"sv;
            return;
        case Type::Tag::INT32:
            str << "int"sv;
            return;
        case Type::Tag::COOPERATIVE_MATRIX_REF:
        case Type::Tag::COOPERATIVE_VECTOR_REF:
        case Type::Tag::UINT32:
            str << "uint"sv;
            return;
        case Type::Tag::FLOAT16:
            str << "float16_t"sv;
            return;
        case Type::Tag::FLOAT64:
            str << "float64_t"sv;
            return;
        case Type::Tag::INT8:
            opt->use_8bit = true;
            str << "int8_t"sv;
            return;
        case Type::Tag::UINT8:
            opt->use_8bit = true;
            str << "uint8_t"sv;
            return;
        case Type::Tag::INT16:
            str << "int16_t"sv;
            return;
        case Type::Tag::UINT16:
            str << "uint16_t"sv;
            return;
        case Type::Tag::INT64:
            str << "int64_t"sv;
            return;
        case Type::Tag::UINT64:
            str << "uint64_t"sv;
            return;
        case Type::Tag::COOPERATIVE_VECTOR:
            str << "vector<";
            GetTypeName(*type.element(), str, usage);
            str << luisa::format(",{}>", type.dimension());
            return;
        case Type::Tag::MATRIX: {
            GetTypeName(*type.element(), str, usage);
            vstd::to_string(type.dimension(), str);
            str << 'x';
            vstd::to_string((type.dimension() == 3) ? 4 : type.dimension(), str);
        }
            return;
        case Type::Tag::VECTOR: {
            if (type.dimension() != 3 || local_var || type.element()->is_bool()) {
                GetTypeName(*type.element(), str, usage);
                vstd::to_string((type.dimension()), str);
            } else {
                str << "_w"sv;
                GetTypeName(*type.element(), str, usage);
                vstd::to_string(3, str);
            }
        }
            return;
        case Type::Tag::STRUCTURE:
        case Type::Tag::ARRAY: {
            auto customType = opt->CreateStruct(&type);
            str << customType;
        }
            return;
        case Type::Tag::BUFFER: {
            if ((static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) != 0)
                str << "RW"sv;
            auto ele = type.element();
            // StructuredBuffer
            if (ele != nullptr) {
                bool aliasStruct = TypeIsAliased(ele);
                str << "StructuredBuffer<"sv;
                if (ele->is_matrix()) {
                    auto n = vstd::to_string(ele->dimension());
                    str << "_WrappedFloat"sv << n << 'x' << n;
                } else {
                    if (ele->is_vector() && ele->dimension() == 3) {
                        GetTypeName(*ele->element(), str, usage);
                        str << '4';
                    } else if (aliasStruct) {
                        str << opt->CreateAliasedStruct(ele).first;
                    } else {
                        GetTypeName(*ele, str, usage);
                    }
                }
                str << '>';
            }
            // ByteAddressBuffer / RWByteAddressBuffer
            else {
                str << "ByteAddressBuffer"sv;
            }
        } break;
        case Type::Tag::TEXTURE: {
            if ((static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) != 0)
                str << "RW"sv;
            str << "Texture"sv;
            vstd::to_string((type.dimension()), str);
            str << "D<"sv;
            GetTypeName(*type.element(), str, usage);
            if (type.tag() != Type::Tag::VECTOR) {
                str << '4';
            }
            str << '>';
            break;
        }
        case Type::Tag::BINDLESS_ARRAY: {
            str << "StructuredBuffer<uint>"sv;
        } break;
        case Type::Tag::ACCEL: {
            str << "RaytracingAccelerationStructure"sv;
        } break;
        case Type::Tag::CUSTOM: {
            str << '_' << type.description();
        } break;
        default:
            LUISA_ERROR("Unsupported {}", luisa::to_string(type.tag()));
            break;
    }
}

// Check if type has alias
bool CodegenUtility::TypeIsAliased(luisa::compute::Type const *t) const {
    if (t->is_array()) {
        auto i = t->element();
        if (VectorShouldBeAliased(i)) [[unlikely]]
            return true;
        if (TypeIsAliased(i)) {
            return true;
        }
    } else if (t->is_structure()) {
        for (auto &i : t->members()) {
            if (VectorShouldBeAliased(i)) [[unlikely]]
                return true;
            if (TypeIsAliased(i)) {
                return true;
            }
        }
    }
    return false;
}

// Check if vector should use aliased type
bool CodegenUtility::VectorShouldBeAliased(luisa::compute::Type const *t) const {
    return (t->is_vector() && ((opt->isSpirv && t->element()->size() > 4 && t->dimension() >= 3) || t->element()->is_bool()));
}

// Convert original type to aliased representation
void CodegenUtility::OriginToAliased(luisa::compute::Type const *t, vstd::StringBuilder &sb) {
    auto aliasedType = opt->CreateAliasedStruct(t);
    if (!aliasedType.second) {
        return;
    }
    auto funcName = luisa::format("_OriToAls{}", t->hash());
    auto set_name = vstd::scope_exit([&]() {
        sb << funcName;
    });
    if (!opt->originToAliasedTypes.emplace(t).second) {
        return;
    }
    vstd::StringBuilder str;
    str << aliasedType.first << ' ' << funcName << '(';
    GetTypeName(*t, str, Usage::NONE, false);
    str << " a){\n"sv << aliasedType.first << luisa::format(" r=({})0;\n", aliasedType.first);
    if (t->is_array()) {
        str << "for(uint i=0;i<" << luisa::format("{}", t->dimension()) << ";++i){\n";
        if (TypeIsAliased(t->element())) {
            str << "r.v[i]="sv;
            OriginToAliased(
                t->element(),
                str);
            str << "(a.v[i]);\n"sv;
        } else {
            str << "r.v[i]=to_Als"sv;
            GetTypeName(*t->element(), str, Usage::READ, true);
            str << luisa::format("(a.v[i]);\n");
        }
        str << "}\n"sv;
    } else {
        auto members = t->members();
        for (size_t i : vstd::range(members.size())) {
            auto m = members[i];
            if (TypeIsAliased(m)) {
                str << luisa::format("r.v{}=", i);
                OriginToAliased(m, str);
                str << luisa::format("(a.v{});\n", i);
            } else if (VectorShouldBeAliased(m)) [[unlikely]] {
                str << luisa::format("r.v{}=to_Als", i);
                GetTypeName(*m, str, Usage::READ, true);
                str << luisa::format("(a.v{});\n", i);
            } else {
                str << luisa::format("r.v{}=a.v{};\n", i, i);
            }
        }
    }
    str << "return r;\n}\n"sv;
    *opt->incrementalFunc << str;
}

// Convert aliased type back to original
void CodegenUtility::AliasedToOrigin(luisa::compute::Type const *t, vstd::StringBuilder &sb) {
    auto aliasedType = opt->CreateAliasedStruct(t);
    if (!aliasedType.second) {
        return;
    }
    auto funcName = luisa::format("_AlsToOri{}", t->hash());
    auto set_name = vstd::scope_exit([&]() {
        sb << funcName;
    });
    if (!opt->aliasedToOriginTypes.emplace(t).second) {
        return;
    }
    vstd::StringBuilder str;
    GetTypeName(*t, str, Usage::NONE, false);
    str << ' ' << funcName << '(' << aliasedType.first << " a){\n"sv;
    vstd::StringBuilder retTypeName;
    GetTypeName(*t, retTypeName, Usage::NONE, false);
    str << retTypeName << luisa::format(" r=({})0;", retTypeName.view());

    if (t->is_array()) {
        str << "for(uint i=0;i<" << luisa::format("{}", t->dimension()) << ";++i){\n";
        if (TypeIsAliased(t->element())) {
            str << "r.v[i]="sv;
            AliasedToOrigin(
                t->element(),
                str);
            str << "(a.v[i]);\n"sv;
        } else {
            str << "r.v[i]=to_";
            GetTypeName(*t->element(), str, Usage::READ, true);
            str << "(a.v[i]);\n"sv;
        }
        str << "}\n"sv;
    } else {
        auto members = t->members();
        for (size_t i : vstd::range(members.size())) {
            auto m = members[i];
            if (TypeIsAliased(m)) {
                str << luisa::format("r.v{}=", i);
                AliasedToOrigin(m, str);
                str << luisa::format("(a.v{});\n", i);
            } else if (VectorShouldBeAliased(m)) [[unlikely]] {
                str << luisa::format("r.v{}=to_", i);
                GetTypeName(*m, str, Usage::READ, true);
                str << luisa::format("(a.v{});\n", i);
            } else {
                str << luisa::format("r.v{}=a.v{};\n", i, i);
            }
        }
    }
    str << "return r;\n}\n"sv;
    *opt->incrementalFunc << str;
}

}// namespace lc::hlsl
