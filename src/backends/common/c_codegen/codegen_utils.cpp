#include "codegen_utils.h"
#include <luisa/core/logging.h>
#include <luisa/vstl/md5.h>
namespace luisa::compute {
using namespace std::string_view_literals;
void Clanguage_CodegenUtils::replace(char *ptr, size_t len, char src, char dst) {
    for (auto &i : vstd::ptr_range(ptr, len)) {
        if (i == src) i = dst;
    }
}
vstd::StringBuilder Clanguage_CodegenUtils::_gen_func_name(Key const &key) {
    vstd::StringBuilder r;
    r.reserve(256);
    r << "func"sv
      << vstd::to_string(key.type)
      << '_'
      << vstd::to_string(key.flag)
      << '_';
    vstd::StringBuilder sb;
    for (auto &i : key.arg_types) {
        get_type_name(sb, i);
    }
    vstd::MD5 md5{luisa::span{(uint8_t const *)sb.data(), sb.size()}};
    r << md5.to_string(false);
    return r;
}

void Clanguage_CodegenUtils::get_type_name(vstd::StringBuilder &sb, Type const *type) {
    if (!type) {
        sb << "void"sv;
        return;
    }
    switch (type->tag()) {
        case Type::Tag::BOOL:
            sb << "bool"sv;
            return;
        case Type::Tag::FLOAT32:
            sb << "float"sv;
            return;
        case Type::Tag::INT32:
            sb << "int32_t"sv;
            return;
        case Type::Tag::UINT32:
            sb << "uint32_t"sv;
            return;
        case Type::Tag::FLOAT16:
            sb << "float16_t"sv;
            return;
        case Type::Tag::FLOAT64:
            sb << "float64_t"sv;
            return;
        case Type::Tag::INT16:
            sb << "int16_t"sv;
            return;
        case Type::Tag::UINT16:
            sb << "uint16_t"sv;
            return;
        case Type::Tag::INT64:
            sb << "int64_t"sv;
            return;
        case Type::Tag::UINT64:
            sb << "uint64_t"sv;
            return;
        case Type::Tag::MATRIX: {
            get_type_name(sb, type->element());
            vstd::to_string(type->dimension(), sb);
            sb << 'x';
            vstd::to_string((type->dimension() == 3) ? 4 : type->dimension(), sb);
        }
            return;
        case Type::Tag::VECTOR: {
            get_type_name(sb, type->element());
            vstd::to_string(type->dimension(), sb);
        }
            return;
        case Type::Tag::BUFFER: {
            sb << "buffer_type";
        }
            return;
        case Type::Tag::STRUCTURE: {
            vstd::StringBuilder type_name;
            if (_get_custom_type(type_name, type)) {
                struct_sb << "typedef struct alignas(";
                vstd::to_string(type->alignment(), struct_sb);
                struct_sb << ") {\n";
                size_t idx = 0;
                for (auto &i : type->members()) {
                    get_type_name(struct_sb, i);
                    struct_sb << " v";
                    vstd::to_string(idx, struct_sb);
                    ++idx;
                }
                struct_sb << "} " << type_name << ";\n";
            }
            sb << type_name;
        }
            return;
        default:
            LUISA_ERROR("Unsupported type.");
            return;
            // TODO
    }
}
bool Clanguage_CodegenUtils::_get_custom_type(vstd::StringBuilder &sb, Type const *t) {
    auto size = _custom_types.size();
    auto iter = _custom_types.try_emplace(t);
    if (iter.second) {
        iter.first.value() = size;
    }
    sb << "_t";
    vstd::to_string(iter.first.value(), sb);
    return iter.second;
}

void Clanguage_CodegenUtils::gen_vec_function(vstd::StringBuilder &sb, vstd::string_view expr, Type const *type) {
    LUISA_ASSERT(type->is_vector(), "Type must be vector");
    static char arr[4] = {'x', 'y', 'z', 'w'};
    sb << '(';
    get_type_name(sb, type);
    sb << "){"sv;
    bool comma = false;
    for (auto i : vstd::range(type->dimension())) {
        if (comma) {
            sb << ", ";
        }
        sb << expr;
        replace(sb.data() + sb.size() - expr.size(), expr.size(), '#', arr[i]);
        comma = true;
    }
    sb << '}';
}
luisa::string_view Clanguage_CodegenUtils::gen_vec_unary(UnaryOp op, Type const *type) {
    Key key{
        .type = 0,
        .flag = luisa::to_underlying(op)};
    key.arg_types.emplace_back(type);
    return _gen_func(
        [this, type, op](luisa::string_view func_name) {
            auto type_name = get_type_name(type);
            decl_sb << "static " << type_name << ' ' << func_name << '(' << type_name << " a){ return "sv;
            luisa::string_view expr = [&]() {
                switch (op) {
                    case UnaryOp::MINUS:
                        return "-a.#"sv;
                    case UnaryOp::NOT:
                        return "!a.#"sv;
                    case UnaryOp::BIT_NOT:
                        return "!a.#"sv;
                    default: {
                        LUISA_ERROR("Bad unary op");
                        return ""sv;
                    }
                }
            }();
            gen_vec_function(decl_sb, expr, type);
            decl_sb << ";\n}\n"sv;
        },
        std::move(key));
}
luisa::string_view Clanguage_CodegenUtils::gen_make_vec(Type const *return_type, luisa::span<Type const *const> arg_types) {
    auto scalar_type = return_type->element();
    auto callop = [&]() {
        switch (scalar_type->tag()) {
            case Type::Tag::BOOL: return (CallOp)(luisa::to_underlying(CallOp::MAKE_BOOL2) + return_type->dimension() - 2);
            case Type::Tag::INT32: return (CallOp)(luisa::to_underlying(CallOp::MAKE_INT2) + return_type->dimension() - 2);
            case Type::Tag::UINT32: return (CallOp)(luisa::to_underlying(CallOp::MAKE_UINT2) + return_type->dimension() - 2);
            case Type::Tag::INT16: return (CallOp)(luisa::to_underlying(CallOp::MAKE_SHORT2) + return_type->dimension() - 2);
            case Type::Tag::UINT16: return (CallOp)(luisa::to_underlying(CallOp::MAKE_USHORT2) + return_type->dimension() - 2);
            case Type::Tag::INT8: return (CallOp)(luisa::to_underlying(CallOp::MAKE_BYTE2) + return_type->dimension() - 2);
            case Type::Tag::UINT8: return (CallOp)(luisa::to_underlying(CallOp::MAKE_UBYTE2) + return_type->dimension() - 2);
            case Type::Tag::INT64: return (CallOp)(luisa::to_underlying(CallOp::MAKE_LONG2) + return_type->dimension() - 2);
            case Type::Tag::UINT64: return (CallOp)(luisa::to_underlying(CallOp::MAKE_ULONG2) + return_type->dimension() - 2);
            case Type::Tag::FLOAT16: return (CallOp)(luisa::to_underlying(CallOp::MAKE_HALF2) + return_type->dimension() - 2);
            case Type::Tag::FLOAT32: return (CallOp)(luisa::to_underlying(CallOp::MAKE_FLOAT2) + return_type->dimension() - 2);
            case Type::Tag::FLOAT64: return (CallOp)(luisa::to_underlying(CallOp::MAKE_DOUBLE2) + return_type->dimension() - 2);
            default: LUISA_ERROR("Bad vector type.");
        }
    }();
    return gen_callop(callop, return_type, arg_types);
}
luisa::string_view Clanguage_CodegenUtils::gen_vec_binary(BinaryOp op, Type const *left_type, Type const *right_type) {
    Key key{
        .type = 1,
        .flag = luisa::to_underlying(op)};
    key.arg_types.emplace_back(left_type);
    key.arg_types.emplace_back(right_type);
    return _gen_func(
        [left_type, right_type, op, this](luisa::string_view func_name) mutable {
            auto left_type_name = get_type_name(left_type);
            auto right_type_name = get_type_name(right_type);
            luisa::string left_name = "a";
            luisa::string right_name = "b";
            vstd::StringBuilder temp_sb;
            auto make_vec = [&](Type const *&type, size_t dst_dim, vstd::StringBuilder &type_name, vstd::string &var_name) {
                auto vec_type = Type::vector(type, dst_dim);
                auto args = {(Type const *)type};
                auto make_func = gen_make_vec(vec_type, args);
                type_name.clear();
                get_type_name(type_name, vec_type);
                temp_sb << type_name << ' ' << var_name << "_vec = " << make_func << '(' << var_name << ");\n"sv;
                type = vec_type;
                var_name = var_name + "_vec";
            };
            if (left_type->is_scalar()) {
                make_vec(left_type, right_type->dimension(), right_type_name, left_name);
            } else if (right_type->is_scalar()) {
                make_vec(right_type, left_type->dimension(), left_type_name, right_name);
            }
            decl_sb << "static ";
            bool ret_is_boolvec = (luisa::to_underlying(op) >= luisa::to_underlying(BinaryOp::LESS));
            if (ret_is_boolvec) {
                get_type_name(decl_sb, Type::vector(Type::of<bool>(), left_type->dimension()));
            } else {
                decl_sb << left_type_name;
            }
            decl_sb << ' ' << func_name << '(' << left_type_name << " a, "sv << right_type_name << " b){\n"sv << temp_sb;
            luisa::string_view name;
            if (!ret_is_boolvec) {
                switch (op) {
                    case BinaryOp::ADD:
                        name = "+"sv;
                        break;
                    case BinaryOp::SUB:
                        name = "-"sv;
                        break;
                    case BinaryOp::MUL: name = "*"sv; break;
                    case BinaryOp::DIV: name = "/"sv; break;
                    case BinaryOp::MOD: name = "%"sv; break;
                    case BinaryOp::BIT_AND: name = "&"sv; break;
                    case BinaryOp::BIT_OR: name = "|"sv; break;
                    case BinaryOp::BIT_XOR: name = "^"sv; break;
                    case BinaryOp::SHL: name = "<<"sv; break;
                    case BinaryOp::SHR: name = ">>"sv; break;
                    case BinaryOp::AND: name = "&&"sv; break;
                    case BinaryOp::OR: name = "||"sv; break;
                }
                decl_sb << "return "sv;
                gen_vec_function(decl_sb, luisa::format("{}.# {} {}.#", left_name, name, right_name), left_type);
                decl_sb << ";\n}\n"sv;
            } else {
                switch (op) {
                    case BinaryOp::LESS: name = "<"sv; break;
                    case BinaryOp::GREATER: name = ">"sv; break;
                    case BinaryOp::LESS_EQUAL: name = "<="sv; break;
                    case BinaryOp::GREATER_EQUAL: name = ">="sv; break;
                    case BinaryOp::EQUAL: name = "=="sv; break;
                    case BinaryOp::NOT_EQUAL: name = "!="sv; break;
                }
                decl_sb << "return "sv;
                gen_vec_function(decl_sb, luisa::format("{}.# {} {}.#", left_name, name, right_name), Type::vector(Type::of<bool>(), left_type->dimension()));
                decl_sb << ";\n}\n"sv;
            }
        },
        std::move(key));
}
luisa::string_view Clanguage_CodegenUtils::gen_callop(CallOp op, Type const *return_type, luisa::span<Type const *const> arg_types) {
    static char swizzle[4] = {'x', 'y', 'z', 'w'};
    Key key{
        .type = 2,
        .flag = luisa::to_underlying(op)};
    vstd::push_back_all(key.arg_types, arg_types);

    return _gen_func(
        [&](luisa::string_view name) {
            auto ret_type_name = get_type_name(return_type);
            vstd::StringBuilder tmp_sb;
            tmp_sb << "static " << ret_type_name << ' ' << name << '(';
            bool comma = false;
            size_t idx = 0;
            for (auto &i : arg_types) {
                if (comma) {
                    tmp_sb << ", ";
                }
                get_type_name(tmp_sb, i);
                tmp_sb << " a";
                vstd::to_string(idx, tmp_sb);
                comma = true;
                idx++;
            }
            tmp_sb << ") {\n";
            auto test_all_scalar = [&]() {
                bool all_scalar = true;
                for (auto &i : arg_types) {
                    if (!i->is_scalar()) {
                        all_scalar = false;
                        break;
                    }
                }
                return all_scalar;
            };
            auto make_vec = [&]() {
                tmp_sb << "return (";
                get_type_name(tmp_sb, return_type);
                tmp_sb << "){";
                bool comma = false;
                if (arg_types.size() == 1 && arg_types[0]->is_scalar()) {
                    for (auto i : vstd::range(return_type->dimension())) {
                        if (comma) {
                            tmp_sb << ", ";
                        }
                        tmp_sb << "a0";
                        comma = true;
                    }
                    tmp_sb << "};";
                } else {
                    for (auto i : vstd::range(arg_types.size())) {
                        auto type = arg_types[i];
                        if (type->is_vector()) {
                            for (auto dim : vstd::range(type->dimension())) {
                                if (comma) {
                                    tmp_sb << ", ";
                                }
                                tmp_sb << 'a';
                                vstd::to_string(i, tmp_sb);
                                tmp_sb << '.' << swizzle[dim];
                                comma = true;
                            }
                        } else {
                            if (comma) {
                                tmp_sb << ", ";
                            }
                            tmp_sb << 'a';
                            vstd::to_string(i, tmp_sb);
                            comma = true;
                        }
                    }
                    tmp_sb << "};";
                }
            };
            switch (op) {
                case CallOp::ALL: {
                    auto a = arg_types[0];
                    tmp_sb << "return ";
                    bool c = false;
                    for (auto i : vstd::range(a->dimension())) {
                        if (c) {
                            tmp_sb << " && "sv;
                        }
                        c = true;
                        tmp_sb << "a0." << swizzle[i];
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ANY: {
                    auto a = arg_types[0];
                    tmp_sb << "return ";
                    bool c = false;
                    for (auto i : vstd::range(a->dimension())) {
                        if (c) {
                            tmp_sb << " || "sv;
                        }
                        c = true;
                        tmp_sb << "a0." << swizzle[i];
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::SELECT: {
                    if (test_all_scalar()) {
                        tmp_sb << "return a2 ? a1 : a0;"sv;
                    } else {
                        tmp_sb << "return ";
                        if (arg_types[2]->is_scalar()) {
                            auto args = {arg_types[2]};
                            auto vec_type = Type::vector(arg_types[2], arg_types[0]->dimension());
                            auto make_name = gen_make_vec(vec_type, args);
                            get_type_name(tmp_sb, vec_type);
                            tmp_sb << " tmp = " << make_name << "(a2);\n"
                                   << gen_vec_function("tmp.# ? a1.# : a0.#", Type::vector(Type::of<bool>(), arg_types[0]->dimension()));
                        } else {
                            tmp_sb << gen_vec_function("a2.# ? a1.# : a0.#", Type::vector(Type::of<bool>(), arg_types[0]->dimension()));
                        }
                        tmp_sb << ';';
                    }
                } break;
                case CallOp::MIN: {
                    if (test_all_scalar()) {
                        tmp_sb << "return a0 > a1 ? a1 : a0";
                    } else {
                        tmp_sb << "return " << gen_vec_function("a0.# > a1.# ? a1.# : a0.#", arg_types[0]) << ';';
                    }
                } break;
                case CallOp::MAX: {
                    if (test_all_scalar()) {
                        tmp_sb << "return a0 > a1 ? a0 : a1";
                    } else {
                        tmp_sb << "return " << gen_vec_function("a0.# > a1.# ? a0.# : a1.#", arg_types[0]) << ';';
                    }
                } break;
                case CallOp::CLAMP: {
                    auto min_args = {arg_types[0], arg_types[1]};
                    auto min_func = gen_callop(CallOp::MIN, arg_types[0], min_args);
                    auto max_func = gen_callop(CallOp::MAX, arg_types[0], min_args);
                    tmp_sb << "return " << min_func << '(' << max_func << "(a0, a1), a2);";
                } break;
                case CallOp::SATURATE: {
                    auto clamp_args = {arg_types[0], arg_types[0], arg_types[0]};
                    auto clamp_func = gen_callop(CallOp::CLAMP, arg_types[0], clamp_args);
                    tmp_sb << "return " << clamp_func << "(a0, ";
                    if (arg_types[0]->is_vector()) {
                        auto scalar_args = {arg_types[0]->element()};
                        auto make_vec = gen_make_vec(arg_types[0], scalar_args);
                        tmp_sb << make_vec << "(0), " << make_vec << "(1));";
                    } else {
                        tmp_sb << "0, 1);";
                    }
                } break;
                case CallOp::MAKE_BOOL2:
                case CallOp::MAKE_BOOL3:
                case CallOp::MAKE_BOOL4:
                case CallOp::MAKE_INT2:
                case CallOp::MAKE_INT3:
                case CallOp::MAKE_INT4:
                case CallOp::MAKE_UINT2:
                case CallOp::MAKE_UINT3:
                case CallOp::MAKE_UINT4:
                case CallOp::MAKE_FLOAT2:
                case CallOp::MAKE_FLOAT3:
                case CallOp::MAKE_FLOAT4:
                case CallOp::MAKE_SHORT2:
                case CallOp::MAKE_SHORT3:
                case CallOp::MAKE_SHORT4:
                case CallOp::MAKE_USHORT2:
                case CallOp::MAKE_USHORT3:
                case CallOp::MAKE_USHORT4:
                case CallOp::MAKE_LONG2:
                case CallOp::MAKE_LONG3:
                case CallOp::MAKE_LONG4:
                case CallOp::MAKE_ULONG2:
                case CallOp::MAKE_ULONG3:
                case CallOp::MAKE_ULONG4:
                case CallOp::MAKE_HALF2:
                case CallOp::MAKE_HALF3:
                case CallOp::MAKE_HALF4:
                case CallOp::MAKE_DOUBLE2:
                case CallOp::MAKE_DOUBLE3:
                case CallOp::MAKE_DOUBLE4:
                case CallOp::MAKE_BYTE2:
                case CallOp::MAKE_BYTE3:
                case CallOp::MAKE_BYTE4:
                case CallOp::MAKE_UBYTE2:
                case CallOp::MAKE_UBYTE3:
                case CallOp::MAKE_UBYTE4:
                    make_vec();
                    break;
                default: break;
            }

            tmp_sb << "\n}\n"sv;
            decl_sb << tmp_sb;
        },
        std::move(key));
}
Clanguage_CodegenUtils::Clanguage_CodegenUtils() = default;
Clanguage_CodegenUtils::~Clanguage_CodegenUtils() = default;
}// namespace luisa::compute