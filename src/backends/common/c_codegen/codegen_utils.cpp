#include "codegen_utils.h"
#include "codegen_visitor.h"
#include <luisa/core/logging.h>
#include <luisa/vstl/md5.h>
#include <luisa/ast/type_registry.h>
#include "codegen_visitor.h"
#include <luisa/vstl/functional.h>
namespace luisa::compute {
namespace c_codegen_detail {
static bool is_integer(Type const *t) {
    switch (t->tag()) {
        case Type::Tag::INT8:
        case Type::Tag::INT16:
        case Type::Tag::INT32:
        case Type::Tag::INT64:
        case Type::Tag::UINT8:
        case Type::Tag::UINT16:
        case Type::Tag::UINT32:
        case Type::Tag::UINT64:
            return true;
        default: return false;
    }
}
struct ExternalTable {
    using GenFunc = vstd::func_ptr_t<void(Clanguage_CodegenUtils &utils, vstd::StringBuilder &sb, vstd::string_view func_name, Type const *ret_type, luisa::span<Type const *const> args, luisa::bitvector &is_ref)>;
    struct GenFuncValue {
        uint32_t flag;
        luisa::bitvector is_ref;
        GenFunc func;
    };
    using Var =
        luisa::variant<
            // template function
            GenFuncValue,
            // fixed type validation
            vstd::func_ptr_t<bool(Type const *ret_type, luisa::span<Type const *const> args)>,
            // template call
            vstd::func_ptr_t<void(vstd::StringBuilder &sb, CodegenVisitor *visitor, Clanguage_CodegenUtils &utils, CallExpr const *call_expr)>>;

    vstd::HashMap<vstd::string, Var> map;
    ExternalTable() {
        auto ptr = +[](Type const *ret_type, luisa::span<Type const *const> args) -> bool {
            if (!ret_type->is_int32()) {
                return false;
            }
            return args.size() == 3 && is_integer(args[0]) && is_integer(args[1]) && is_integer(args[2]);
        };
        map.emplace(
            "lc_memcmp", ptr);
        ptr = +[](Type const *ret_type, luisa::span<Type const *const> args) -> bool {
            if (ret_type != Type::of<void>()) return false;
            return args.size() == 3 && is_integer(args[0]) && is_integer(args[1]) && is_integer(args[2]);
        };
        map.emplace(
            "lc_memcpy",
            ptr);
        map.emplace(
            "lc_memmove",
            ptr);
        ptr = +[](Type const *ret_type, luisa::span<Type const *const> args) -> bool {
            if (!ret_type->is_uint64()) return false;
            return args.size() == 1 && is_integer(args[0]);
        };
        map.emplace(
            "persist_malloc", ptr);
        map.emplace(
            "temp_malloc",
            ptr);
        map.emplace(
            "persist_free",
            +[](Type const *ret_type, luisa::span<Type const *const> args) -> bool {
                if (ret_type != Type::of<void>()) return false;
                return args.size() == 1 && args[0]->is_uint64();
            });
        map.emplace(
            "to_string",
            +[](vstd::StringBuilder &sb, CodegenVisitor *visitor, Clanguage_CodegenUtils &utils, CallExpr const *call_expr) {
                auto func = call_expr->external();
                sb << "to_string(";
                utils.get_type_name(sb, func->return_type());
                for (auto &i : call_expr->arguments()) {
                    sb << ", ";
                    i->accept(*visitor);
                }
                sb << ')';
            });
        uint flag = 0;
        auto add_ext = [&](luisa::string &&name, GenFunc func) {
            map.emplace(
                std::move(name),
                GenFuncValue{
                    .flag = flag,
                    .func = func});
            flag++;
        };
        map.emplace(
            "invoke",
            +[](vstd::StringBuilder &sb, CodegenVisitor *visitor, Clanguage_CodegenUtils &utils, CallExpr const *call_expr) {
                sb << "(((";
                utils.get_type_name(sb, call_expr->type());
                sb << "(*)(";
                bool comma = false;
                auto args = call_expr->arguments();
                for (auto &i : args.subspan(1)) {
                    if (comma) {
                        sb << ", ";
                    }
                    comma = true;
                    utils.get_type_name(sb, i->type());
                    if (i->tag() == Expression::Tag::REF && static_cast<RefExpr const *>(i)->variable().is_reference()) {
                        sb << '*';
                    }
                }
                sb << "))";
                args[0]->accept(*visitor);
                sb << ")(";
                comma = false;
                for (auto &i : args.subspan(1)) {
                    if (comma) {
                        sb << ", ";
                    }
                    comma = true;
                    if (i->tag() == Expression::Tag::REF && static_cast<RefExpr const *>(i)->variable().is_reference()) {
                        sb << '&';
                    }
                    i->accept(*visitor);
                }
                sb << "))";
            });
        add_ext(
            "device_log_ext",
            +[](Clanguage_CodegenUtils &utils, vstd::StringBuilder &sb, vstd::string_view func_name, Type const *ret_type, luisa::span<Type const *const> args, luisa::bitvector &is_ref) {
                sb << "inline ";
                utils.get_type_name(sb, ret_type);
                sb << ' ' << func_name << '(';
                bool comma = false;
                size_t arg_idx = 0;
                for (auto &i : args) {
                    if (comma) {
                        sb << ", ";
                    }
                    comma = true;
                    utils.get_type_name(sb, i);
                    sb << " a" << luisa::format("{}", arg_idx);
                    arg_idx++;
                }
                sb << "){\npush_str((char const*)a0.v0, a0.v1);\n";
                arg_idx = 1;
                for (auto &i : args.subspan(1)) {
                    sb << "push_";
                    utils.get_type_name(sb, i);
                    sb << "(a"
                       << luisa::format("{}", arg_idx)
                       << ");\n";
                    arg_idx++;
                }
                sb << "invoke_print();\n}\n";
            });
        add_ext(
            "rtti_call",
            +[](Clanguage_CodegenUtils &utils, vstd::StringBuilder &sb, vstd::string_view func_name, Type const *ret_type, luisa::span<Type const *const> args, luisa::bitvector &is_ref) {
                is_ref.resize(2);
                is_ref[1] = true;
                sb << "inline void " << func_name << '(';
                utils.get_type_name(sb, args[0]);
                sb << " a0, ";
                utils.get_type_name(sb, args[1]);
                sb << "* a1){\n"
                      "static const char type_desc[] = {";
                bool comma = false;
                auto desc = args[1]->description();
                for (auto &i : desc) {
                    if (comma) {
                        sb << ',';
                    }
                    comma = true;
                    sb << luisa::format("{}", (uint)i);
                }
                sb << "};\nrtti_call(a0.v0, a0.v1, type_desc, "
                   << luisa::format("{}", desc.size())
                   << ", a1);\n}\n";
            });
        map.emplace(
            "is_trivial",
            +[](vstd::StringBuilder &sb, CodegenVisitor *visitor, Clanguage_CodegenUtils &utils, CallExpr const *call_expr) {
                auto args = call_expr->arguments();
                auto type = args[0]->type();
                utils.set_dtor_type(type);
                if (utils.is_trivial_destructible(type)) {
                    sb << "true";
                    return;
                } else {
                    sb << "false";
                }
            });
        map.emplace(
            "dispose",
            +[](vstd::StringBuilder &sb, CodegenVisitor *visitor, Clanguage_CodegenUtils &utils, CallExpr const *call_expr) {
                auto args = call_expr->arguments();
                auto type = args[0]->type();
                utils.set_dtor_type(type);
                if (utils.is_trivial_destructible(type)) {
                    return;
                }
                sb << "dtor_" << vstd::MD5(type->description()).to_string(false) << "(&";
                args[0]->accept(*visitor);
                sb << ')';
            });
    }
};
static ExternalTable extern_table;
class CodegenConstantPrinter final : public ConstantDecoder {

private:
    vstd::StringBuilder &_str;

public:
    CodegenConstantPrinter(vstd::StringBuilder &str) noexcept
        : _str{str} {}

protected:
    void _decode_bool(bool x) noexcept override {
        PrintValue<bool>{}(x, _str);
    }
    void _decode_char(char x) noexcept override {
        PrintValue<luisa::byte>{}(x, _str);
    }
    void _decode_uchar(uchar x) noexcept override {
        PrintValue<luisa::ubyte>{}(x, _str);
    }
    void _decode_short(short x) noexcept override {
        PrintValue<short>{}(x, _str);
    }
    void _decode_ushort(ushort x) noexcept override {
        PrintValue<ushort>{}(x, _str);
    }
    void _decode_int(int x) noexcept override {
        PrintValue<int>{}(x, _str);
    }
    void _decode_uint(uint x) noexcept override {
        PrintValue<uint>{}(x, _str);
    }
    void _decode_long(slong x) noexcept override {
        PrintValue<slong>{}(x, _str);
    }
    void _decode_ulong(ulong x) noexcept override {
        PrintValue<ulong>{}(x, _str);
    }
    void _decode_half(half x) noexcept override {
        PrintValue<half>{}(x, _str);
    }
    void _decode_float(float x) noexcept override {
        PrintValue<float>{}(x, _str);
    }
    void _decode_double(double x) noexcept override {
        PrintValue<double>{}(x, _str);
    }
    void _vector_separator(const Type *type, uint index) noexcept override {
        LUISA_ERROR_WITH_LOCATION("Should not be called.");
    }
    void _matrix_separator(const Type *type, uint index) noexcept override {
        LUISA_ERROR_WITH_LOCATION("Should not be called.");
    }
    void _decode_vector(const Type *type, const std::byte *data) noexcept override {
#define LUISA_C_DECODE_CONST_VEC(T, N)                      \
    do {                                                    \
        if (type == Type::of<T##N>()) {                     \
            auto x = *reinterpret_cast<const T##N *>(data); \
            if constexpr (N == 3) { _str << "{"sv; }        \
            PrintValue<T##N>{}(x, _str);                    \
            if constexpr (N == 3) { _str << ",0}"sv; }      \
            return;                                         \
        }                                                   \
    } while (false)
#define LUISA_C_DECODE_CONST(T)     \
    LUISA_C_DECODE_CONST_VEC(T, 2); \
    LUISA_C_DECODE_CONST_VEC(T, 3); \
    LUISA_C_DECODE_CONST_VEC(T, 4)
        LUISA_C_DECODE_CONST(bool);
        LUISA_C_DECODE_CONST(int);
        LUISA_C_DECODE_CONST(uint);
        LUISA_C_DECODE_CONST(float);
        LUISA_ERROR_WITH_LOCATION(
            "Constant type '{}' is not supported yet.",
            type->description());
#undef LUISA_C_DECODE_CONST_VEC
#undef LUISA_C_DECODE_CONST
    }
    void _decode_matrix(const Type *type, const std::byte *data) noexcept override {
#define LUISA_C_DECODE_CONST_MAT(N)                                    \
    do {                                                               \
        using M = float##N##x##N;                                      \
        if (type == Type::of<M>()) {                                   \
            auto x = *reinterpret_cast<const M *>(data);               \
            _str << "(float" << #N "x" << (N == 3 ? "4" : #N) << "){"; \
            for (auto i = 0; i < N; i++) {                             \
                _str << "float" << (N == 3 ? "4" : #N) << "(";         \
                for (auto j = 0; j < 3; j++) {                         \
                    PrintValue<float>{}(x[i][j], _str);                \
                    if (j != N - 1) { _str << ","; }                   \
                }                                                      \
                if (N == 3) { _str << ",0"; }                          \
                _str << ")";                                           \
                if (i != N - 1) { _str << ","; }                       \
            }                                                          \
            _str << "}";                                               \
            return;                                                    \
        }                                                              \
    } while (false)
        LUISA_C_DECODE_CONST_MAT(2);
        LUISA_C_DECODE_CONST_MAT(3);
        LUISA_C_DECODE_CONST_MAT(4);
        LUISA_ERROR_WITH_LOCATION(
            "Constant type '{}' is not supported yet.",
            type->description());
#undef LUISA_C_DECODE_CONST_MAT
    }
    void _struct_separator(const Type *type, uint index) noexcept override {
        auto n = type->members().size();
        if (index == 0u) {
            _str << "{"sv;
        } else if (index == n) {
            _str << "}"sv;
        } else {
            _str << ',';
        }
    }
    void _array_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _str << "{{"sv;
        } else if (index == n) {
            _str << "}}"sv;
        } else {
            _str << ',';
        }
    }
};
}// namespace c_codegen_detail
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
void Clanguage_CodegenUtils::set_dtor_type(Type const *type) {
    struct Destructor {
        luisa::vector<std::pair<size_t, Type const *>> dtor_member_idx;
        size_t finalizer_offset{std::numeric_limits<size_t>::max()};
    };
    static Type const *finalizer_type = Type::from("struct<8,[finalizer]ulong>");
    switch (type->tag()) {
        case Type::Tag::STRUCTURE: {
            if (_destructor_types.find(type)) return;
            size_t offset = 0;
            Destructor destructor;
            for (auto &i : type->members()) {
                if (i->alignment() > 1) {
                    offset = (offset + i->alignment() - 1) & (~(i->alignment() - 1));
                }
                if (i == finalizer_type) {
                    if (destructor.finalizer_offset != std::numeric_limits<size_t>::max()) [[unlikely]] {
                        LUISA_ERROR("Struct can not have multiple destructor.");
                    }
                    destructor.finalizer_offset = offset;
                } else if (auto iter = _destructor_types.find(i)) {
                    destructor.dtor_member_idx.emplace_back(offset, i);
                }
                offset += i->size();
            }
            if (destructor.finalizer_offset != std::numeric_limits<size_t>::max() || (!destructor.dtor_member_idx.empty())) [[unlikely]] {
                _destructor_types.emplace(type);
                dtor_sb << "void dtor_" << vstd::MD5(type->description()).to_string(false) << "(void* ptr) {\nuint8_t* byte_ptr = (uint8_t*)ptr;\n";
                if (destructor.finalizer_offset != std::numeric_limits<size_t>::max()) {
                    dtor_sb << "call_dtor(((Finalizer*)(byte_ptr + " << luisa::format("{}", destructor.finalizer_offset) << "))->ptr, ptr);\n";
                }
                for (auto &mem : destructor.dtor_member_idx) {
                    dtor_sb
                        << "dtor_"
                        << vstd::MD5(mem.second->description()).to_string(false)
                        << "(byte_ptr + "
                        << luisa::format("{}", mem.first) << ");\n";
                }
                dtor_sb << "}\n";
            }
        } break;
        case Type::Tag::ARRAY: {
            if (_destructor_types.find(type)) return;
            if (auto iter = _destructor_types.find(type->element())) [[unlikely]] {
                _destructor_types.emplace(type);
                auto ele_type = get_type_name(type->element());
                dtor_sb
                    << "void dtor_"
                    << vstd::MD5(type->description()).to_string(false)
                    << "(void* ptr) {\n"
                    << ele_type
                    << "* byte_ptr = ("
                    << ele_type
                    << "*)ptr;\n"
                       "for(size_t i = 0; i < "
                    << luisa::format("{}", type->dimension())
                    << "; ++i) {\n"
                       "dtor_"
                    << vstd::MD5(type->element()->description()).to_string(false)
                    << "(byte_ptr + i);\n}\n}\n";
            }
        } break;
    }
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
        case Type::Tag::INT8:
            sb << "int8_t"sv;
            return;
        case Type::Tag::UINT8:
            sb << "uint8_t"sv;
            return;
        case Type::Tag::FLOAT16:
            LUISA_ERROR("Half not supported.");
            return;
        case Type::Tag::FLOAT64:
            sb << "double"sv;
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
        case Type::Tag::STRUCTURE: {
            vstd::StringBuilder type_name;
            if (_get_custom_type(type_name, type)) {
                vstd::StringBuilder temp_sb;
                temp_sb << "typedef struct {\n";
                size_t idx = 0;
                for (auto &i : type->members()) {
                    temp_sb << "_Alignas("
                            << luisa::format("{}", i->alignment())
                            << ") ";
                    get_type_name(temp_sb, i);
                    temp_sb << " v";
                    vstd::to_string(idx, temp_sb);
                    temp_sb << ";\n";
                    ++idx;
                }

                temp_sb << "} " << type_name << ";\n";
                struct_sb << temp_sb;
                set_dtor_type(type);
            }
            sb << type_name;
        }
            return;
        case Type::Tag::ARRAY: {
            vstd::StringBuilder type_name;
            if (_get_custom_type(type_name, type)) {
                vstd::StringBuilder temp_sb;
                temp_sb << "typedef struct {\n";
                temp_sb << "_Alignas("
                        << luisa::format("{}", type->element()->alignment())
                        << ") ";
                get_type_name(temp_sb, type->element());
                temp_sb << " v0[";
                vstd::to_string(type->dimension(), temp_sb);
                temp_sb << "];\n} " << type_name << ";\n";
                struct_sb << temp_sb;
                set_dtor_type(type);
            }
            sb << type_name;
        }
            return;
        default:
            LUISA_ERROR("Unsupported type {}.", luisa::to_string(type->tag()));
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
void Clanguage_CodegenUtils::gen_constant(vstd::StringBuilder &sb, ConstantData const &data) {
    auto type_name = get_type_name(data.type());
    if (const_set.try_emplace(data.hash()).second) {
        sb << "static " << type_name << " c" << luisa::format("{}", data.hash()) << "[] = ";
        c_codegen_detail::CodegenConstantPrinter printer{sb};
        data.decode(printer);
        sb << ";\n"sv;
    }
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
luisa::string_view Clanguage_CodegenUtils::gen_access(Type const *return_type, luisa::span<Type const *const> arg_types, bool is_self_rvalue) {
    Key key{
        .type = 5,
        .flag = is_self_rvalue ? 1u : 0u};
    key.arg_types.reserve(arg_types.size() + 1);
    key.arg_types.emplace_back(return_type);
    vstd::push_back_all(key.arg_types, arg_types);
    LUISA_ASSUME(arg_types.size() == 3);
    return _gen_func(
        [&](luisa::string_view name) {
            vstd::StringBuilder ret_name;
            get_type_name(ret_name, return_type);

            vstd::StringBuilder tmp_sb;
            tmp_sb << "inline " << ret_name << "* " << name << '(';
            get_type_name(tmp_sb, arg_types[0]);
            if (!is_self_rvalue) {
                tmp_sb << '*';
            }
            tmp_sb << " a0, ";
            get_type_name(tmp_sb, arg_types[1]);
            tmp_sb << " a1){\n";
            if (arg_types[0]->is_structure()) {
                if (is_self_rvalue) {
                    tmp_sb << "check_access(a0.v1, a1";
                } else {
                    tmp_sb << "check_access(a0->v1, a1";
                }
                // tmp_sb << "return check_access(a0.v0, " << luisa::format("{}", return_type->)
            } else {
                tmp_sb << "check_access(";
                tmp_sb << luisa::format("{}", arg_types[0]->dimension()) << ", a1";
            }
            tmp_sb << ");\nreturn ((" << ret_name << "*)";
            if (is_self_rvalue) {
                tmp_sb << "a0.v0";
            } else {
                tmp_sb << "a0->v0";
            }
            tmp_sb << ") + a1;\n}\n";
            decl_sb << tmp_sb;
        },
        std::move(key));
}

luisa::string_view Clanguage_CodegenUtils::gen_callop(CallOp op, Type const *return_type, luisa::span<Type const *const> arg_types) {
    static char swizzle[4] = {'x', 'y', 'z', 'w'};
    Key key{
        .type = 2,
        .flag = luisa::to_underlying(op)};
    key.arg_types.reserve(arg_types.size() + 1);
    key.arg_types.emplace_back(return_type);
    vstd::push_back_all(key.arg_types, arg_types);

    return _gen_func(
        [&](luisa::string_view name) {
            auto ret_type_name = get_type_name(return_type);
            vstd::StringBuilder tmp_sb;
            tmp_sb << "inline " << ret_type_name << ' ' << name << '(';
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
                            gen_vec_function(tmp_sb, "a2 ? a1.# : a0.#", return_type);
                        } else {
                            gen_vec_function(tmp_sb, "a2.# ? a1.# : a0.#", return_type);
                        }
                        tmp_sb << ';';
                    }
                } break;
                case CallOp::MIN: {
                    if (test_all_scalar()) {
                        tmp_sb << "return a0 > a1 ? a1 : a0";
                    } else {
                        tmp_sb << "return ";
                        gen_vec_function(tmp_sb, "a0.# > a1.# ? a1.# : a0.#", return_type);
                        tmp_sb << ';';
                    }
                } break;
                case CallOp::MAX: {
                    if (test_all_scalar()) {
                        tmp_sb << "return a0 > a1 ? a0 : a1";
                    } else {
                        tmp_sb << "return " << gen_vec_function("a0.# > a1.# ? a0.# : a1.#", return_type) << ';';
                    }
                } break;
                case CallOp::CLAMP: {
                    auto min_args = {arg_types[0], arg_types[1]};
                    auto min_func = gen_callop(CallOp::MIN, return_type, min_args);
                    auto max_func = gen_callop(CallOp::MAX, return_type, min_args);
                    tmp_sb << "return " << min_func << '(' << max_func << "(a0, a1), a2);";
                } break;
                case CallOp::SATURATE: {
                    auto clamp_args = {return_type, return_type, return_type};
                    auto clamp_func = gen_callop(CallOp::CLAMP, return_type, clamp_args);
                    tmp_sb << "return " << clamp_func << "(a0, ";
                    if (return_type->is_vector()) {
                        auto scalar_args = {return_type->element()};
                        auto make_vec = gen_make_vec(return_type, scalar_args);
                        tmp_sb << make_vec << "(0), " << make_vec << "(1));";
                    } else {
                        tmp_sb << "0, 1);";
                    }
                } break;
                case CallOp::LERP: {
                    if (return_type->is_vector()) {
                        auto scalar_args = {return_type->element(), return_type->element(), return_type->element()};
                        auto scalar_lerp = gen_callop(CallOp::LERP, return_type->element(), scalar_args);
                        if (arg_types[2]->is_scalar()) {
                            tmp_sb << "return ";
                            gen_vec_function(tmp_sb, luisa::format("{}(a0.#, a1.#, a2)", scalar_lerp), return_type);
                            tmp_sb << ';';
                        } else {
                            tmp_sb << "return ";
                            gen_vec_function(tmp_sb, luisa::format("{}(a0.#, a1.#, a2.#)", scalar_lerp), return_type);
                            tmp_sb << ';';
                        }
                    } else {
                        tmp_sb << "return a0 * (1.0 - a2) + a1 * a2;";
                    }
                } break;
                case CallOp::SMOOTHSTEP: {
                    if (return_type->is_vector()) {
                        auto scalar_args = {return_type->element(), return_type->element(), return_type->element()};
                        auto scalar_lerp = gen_callop(CallOp::SMOOTHSTEP, return_type->element(), scalar_args);
                        tmp_sb << "return ";
                        if (arg_types[2]->is_scalar()) {
                            gen_vec_function(tmp_sb, luisa::format("{}(a0.#, a1.#, tmp.#)", scalar_lerp), return_type);
                            tmp_sb << ';';
                        } else {
                            gen_vec_function(tmp_sb, luisa::format("{}(a0.#, a1.#, a2.#)", scalar_lerp), return_type);
                            tmp_sb << ';';
                        }
                    } else {
                        auto scalar_args = {return_type};
                        auto clamp_func = gen_callop(CallOp::SATURATE, return_type, scalar_args);
                        get_type_name(tmp_sb, return_type);
                        tmp_sb << " x = " << clamp_func
                               << "((a2 - a0) / (a1 - a0));\nreturn x * x * (3.0 - 2.0 * x);";
                    }
                } break;
                case CallOp::STEP: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        auto scalar_args = {return_type->element(), return_type->element()};
                        auto scalar_step = gen_callop(CallOp::STEP, return_type->element(), scalar_args);
                        gen_vec_function(tmp_sb, luisa::format("{}(a0.#, a1.#)", scalar_step), return_type);
                    } else {
                        tmp_sb << "(a0 >= a1) ? 1 : 0";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ABS: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "abs(a0.#)", return_type);
                    } else {
                        tmp_sb << "abs(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ISINF: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "isinf(a0.#)", return_type);
                    } else {
                        tmp_sb << "isinf(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ISNAN: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "isnan(a0.#)", return_type);
                    } else {
                        tmp_sb << "isnan(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ACOS: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "acos(a0.#)", return_type);
                    } else {
                        tmp_sb << "acos(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ACOSH: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "acosh(a0.#)", return_type);
                    } else {
                        tmp_sb << "acosh(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ASIN: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "asin(a0.#)", return_type);
                    } else {
                        tmp_sb << "asin(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ASINH: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "asinh(a0.#)", return_type);
                    } else {
                        tmp_sb << "asinh(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ATAN: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "atan(a0.#)", return_type);
                    } else {
                        tmp_sb << "atan(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ATAN2: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "atan2(a0.#)", return_type);
                    } else {
                        tmp_sb << "atan2(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ATANH: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "atanh(a0.#)", return_type);
                    } else {
                        tmp_sb << "atanh(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::COS: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "cos(a0.#)", return_type);
                    } else {
                        tmp_sb << "cos(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::COSH: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "cosh(a0.#)", return_type);
                    } else {
                        tmp_sb << "cosh(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::SIN: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "sin(a0.#)", return_type);
                    } else {
                        tmp_sb << "sin(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::SINH: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "sinh(a0.#)", return_type);
                    } else {
                        tmp_sb << "sinh(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::TAN: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "tan(a0.#)", return_type);
                    } else {
                        tmp_sb << "tan(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::TANH: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "tanh(a0.#)", return_type);
                    } else {
                        tmp_sb << "tanh(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::EXP: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "exp(a0.#)", return_type);
                    } else {
                        tmp_sb << "exp(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::EXP2: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "exp2(a0.#)", return_type);
                    } else {
                        tmp_sb << "exp2(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::EXP10: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "exp10(a0.#)", return_type);
                    } else {
                        tmp_sb << "ldexp(a0, 10)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::LOG: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "log(a0.#)", return_type);
                    } else {
                        tmp_sb << "log(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::LOG2: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "log2(a0.#)", return_type);
                    } else {
                        tmp_sb << "log2(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::LOG10: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "log10(a0.#)", return_type);
                    } else {
                        tmp_sb << "log10(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::POW: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "pow(a0.#)", return_type);
                    } else {
                        tmp_sb << "pow(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::SQRT: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "sqrt(a0.#)", return_type);
                    } else {
                        tmp_sb << "sqrt(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::RSQRT: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "rsqrt(a0.#)", return_type);
                    } else {
                        tmp_sb << "1.0 / sqrt(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::CEIL: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "ceil(a0.#)", return_type);
                    } else {
                        tmp_sb << "ceil(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::FLOOR: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "floor(a0.#)", return_type);
                    } else {
                        tmp_sb << "floor(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::FRACT: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "fract(a0.#)", return_type);
                    } else {
                        tmp_sb << "a0 - floor(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::TRUNC: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "trunc(a0.#)", return_type);
                    } else {
                        tmp_sb << "trunc(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::ROUND: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "round(a0.#)", return_type);
                    } else {
                        tmp_sb << "round(a0)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::FMA: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "a0.# * a1.# + a2.#", return_type);
                    } else {
                        tmp_sb << "a0 * a1 + a2";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::COPYSIGN: {
                    tmp_sb << "return ";
                    if (return_type->is_vector()) {
                        gen_vec_function(tmp_sb, "copysign(a0.#, a1.#)", return_type);
                    } else {
                        tmp_sb << "copysign(a0, a1)";
                    }
                    tmp_sb << ';';
                } break;
                case CallOp::CROSS: {
                    get_type_name(tmp_sb, return_type);
                    tmp_sb << " r;\n"
                              "r.x = a0.y * a1.z - a0.z * a1.y;\n"
                              "r.y = -(a0.x * a1.z - a0.z * a1.x);\n"
                              "r.z = a0.x * a1.y - a0.y * a1.x;\n"
                              "return r;";
                } break;
                case CallOp::DOT: {
                    switch (arg_types[0]->dimension()) {
                        case 2:
                            tmp_sb << "return a0.x * a1.x + a0.y * a1.y;";
                            break;
                        case 3:
                            tmp_sb << "return a0.x * a1.x + a0.y * a1.y + a0.z * a1.z;";
                            break;
                        case 4:
                            tmp_sb << "return a0.x * a1.x + a0.y * a1.y + a0.z * a1.z + a0.w * a1.w;";
                            break;
                    }
                } break;
                case CallOp::LENGTH: {
                    switch (arg_types[0]->dimension()) {
                        case 2:
                            tmp_sb << "return sqrt(a0.x * a0.x + a0.y * a0.y);";
                            break;
                        case 3:
                            tmp_sb << "return sqrt(a0.x * a0.x + a0.y * a0.y + a0.z * a0.z);";
                            break;
                        case 4:
                            tmp_sb << "return sqrt(a0.x * a0.x + a0.y * a0.y + a0.z * a0.z + a0.w * a0.w);";
                            break;
                    }
                } break;
                case CallOp::LENGTH_SQUARED: {
                    switch (arg_types[0]->dimension()) {
                        case 2:
                            tmp_sb << "return (a0.x * a0.x + a0.y * a0.y);";
                            break;
                        case 3:
                            tmp_sb << "return (a0.x * a0.x + a0.y * a0.y + a0.z * a0.z);";
                            break;
                        case 4:
                            tmp_sb << "return (a0.x * a0.x + a0.y * a0.y + a0.z * a0.z + a0.w * a0.w);";
                            break;
                    }
                } break;
                case CallOp::NORMALIZE: {
                    auto length_arg = {return_type};
                    auto length_func = gen_callop(CallOp::LENGTH, return_type->element(), length_arg);
                    get_type_name(tmp_sb, return_type->element());
                    tmp_sb
                        << " tmp = "
                        << length_func
                        << "(a0);\n";
                    tmp_sb << "return ";
                    gen_vec_function(tmp_sb, "a0.# / tmp", return_type);
                    tmp_sb << ';';
                } break;
                case CallOp::BUFFER_READ: {
                    tmp_sb << "return ((";
                    get_type_name(tmp_sb, return_type);
                    tmp_sb << "*)(a0.ptr))[a1];";
                } break;
                case CallOp::BUFFER_WRITE: {
                    tmp_sb << "((";
                    get_type_name(tmp_sb, arg_types[2]);
                    tmp_sb << "*)(a0.ptr))[a1] = a2;";
                } break;
                case CallOp::BYTE_BUFFER_READ: {
                    tmp_sb << "return *((";
                    get_type_name(tmp_sb, return_type);
                    tmp_sb << "*)(a0.ptr + a1));";
                } break;

                case CallOp::BYTE_BUFFER_WRITE: {
                    tmp_sb << "*((";
                    get_type_name(tmp_sb, arg_types[2]);
                    tmp_sb << "*)(a0.ptr + a1)) = a2;";
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
                default:
                    LUISA_ERROR("Unsupported call {}", luisa::to_string(op));
                    break;
            }

            tmp_sb << "\n}\n"sv;
            decl_sb << tmp_sb;
        },
        std::move(key));
}
luisa::string_view Clanguage_CodegenUtils::gen_vec_swizzle(luisa::span<uint const> swizzle, uint swizzle_code, Type const *arg) {
    Key key{
        .type = 3,
        .flag = swizzle_code};
    key.arg_types.emplace_back(arg);
    return _gen_func(
        [&](luisa::string_view name) {
            auto return_type = Type::vector(arg->element(), swizzle.size());
            auto ret_typename = get_type_name(return_type);
            auto ele_typename = get_type_name(arg->element());
            decl_sb << "static " << ret_typename << ' ' << name << '(';
            get_type_name(decl_sb, arg);
            decl_sb << " a0){\nreturn ("
                    << ret_typename << "){";
            bool comma = false;
            for (auto &i : swizzle) {
                if (comma) {
                    decl_sb << ", ";
                }
                comma = true;
                decl_sb << "GET(" << ele_typename << ", a0, ";
                vstd::to_string(i, decl_sb);
                decl_sb << ')';
            }
            decl_sb << "};\n}\n";
        },
        std::move(key));
}
size_t Clanguage_CodegenUtils::func_index(Function f) {
    auto size = _custom_funcs.size();
    auto iter = _custom_funcs.try_emplace(f.builder());
    if (iter.second) {
        iter.first.value() = size;
    };
    return iter.first.value();
}
void Clanguage_CodegenUtils::print_function_declare(vstd::StringBuilder &sb, Function func) {
    sb << "static ";
    get_type_name(sb, func.return_type());
    sb << " custom_" << luisa::format("{}", func_index(func))
       << '(';
    bool comma = false;
    for (auto &i : func.arguments()) {
        if (comma) {
            sb << ", ";
        }
        comma = true;
        get_type_name(sb, i.type());
        sb << ' ';
        if (i.tag() == Variable::Tag::REFERENCE) {
            sb << '*';
        }
        gen_var_name(sb, i);
    }
    sb << ')';
}
void Clanguage_CodegenUtils::print_kernel_declare(vstd::StringBuilder &sb, Function func) {
    sb << "static void builtin_c4434d750cf64f0eae3f73cca8650b16(uint32_t3 thd_id, uint32_t3 blk_id, uint32_t3 dsp_id, uint32_t3 dsp_size, uint32_t3 ker_id";
    for (auto &i : func.arguments()) {
        sb << ", ";
        get_type_name(sb, i.type());
        sb << ' ';
        if (i.tag() == Variable::Tag::REFERENCE) {
            sb << '*';
        }
        gen_var_name(sb, i);
    }
    sb << ')';
}
void Clanguage_CodegenUtils::gen_var_name(vstd::StringBuilder &sb, Variable const &var) {
    switch (var.tag()) {
        case Variable::Tag::LOCAL:
            sb << luisa::format("v{}", var.uid());
            break;
        case Variable::Tag::BUFFER:
            sb << luisa::format("b{}", var.uid());
            break;
        case Variable::Tag::THREAD_ID:
            sb << "thd_id";
            break;
        case Variable::Tag::BLOCK_ID:
            sb << "blk_id";
            break;
        case Variable::Tag::DISPATCH_ID:
            sb << "dsp_id";
            break;
        case Variable::Tag::DISPATCH_SIZE:
            sb << "dsp_size";
            break;
        case Variable::Tag::KERNEL_ID:
            sb << "ker_id";
            break;
        case Variable::Tag::OBJECT_ID:
            sb << "ker_id";
            break;
        case Variable::Tag::REFERENCE:
            sb << luisa::format("r{}", var.uid());
            break;
        default:
            LUISA_ERROR("Bad variable type. {}", luisa::to_string(var.tag()));
            break;
    }
}
void Clanguage_CodegenUtils::codegen(
    luisa::string const &path,
    luisa::string_view entry_name,
    Function func) {

    struct_sb << "#include \"header.h\"\n#define INF_d ";
    vstd::to_string(std::numeric_limits<double>::max(), struct_sb);
    struct_sb << "\n#define INF_f ";
    vstd::to_string(std::numeric_limits<float>::max(), struct_sb);
    struct_sb << "\n";
    vstd::StringBuilder sb;
    auto print_extern = [&]() {
#ifdef _MSC_VER
        sb << "__declspec(dllexport) ";
#else
        sb << "__attribute__((visibility(\"default\"))) ";
#endif
    };
    auto find_all_custom_func = [&](auto &find_all_custom_func, Function func) -> void {
        auto size = _custom_funcs.size();
        auto iter = _custom_funcs.try_emplace(func.builder());
        if (iter.second) {
            iter.first.value() = size;
            for (auto &i : func.custom_callables()) {
                find_all_custom_func(find_all_custom_func, Function(i.get()));
            }
        };
    };
    find_all_custom_func(find_all_custom_func, func);
    for (auto &i : _custom_funcs) {
        if (i.first != func.builder()) {
            print_function_declare(decl_sb, Function(i.first));
            decl_sb << ";\n";
        }
        for (auto &i : func.constants()) {
            gen_constant(sb, i);
        }
    }
    for (auto &i : _custom_funcs) {
        CodegenVisitor visitor(
            sb,
            entry_name,
            *this,
            Function(i.first));
    }
    print_extern();
    sb << "uint32_t " << entry_name << "_arg_usage_c4434d750cf64f0eae3f73cca8650b16(uint32_t idx) {\nstatic const uint32_t usages[] = {";
    if (func.arguments().empty()) {
        sb << '0';
    } else {
        bool comma = false;
        for (auto &i : func.arguments()) {
            if (comma) {
                sb << ", ";
            }
            comma = true;
            sb << luisa::format("{}u", luisa::to_underlying(func.variable_usage(i.uid())));
        }
    }
    sb << "};\nreturn usages[idx];\n}\n";

    print_extern();
    auto blk_size = func.block_size();
    sb << "uint32_t3 " << entry_name << "_block_size_c4434d750cf64f0eae3f73cca8650b16(){\nreturn (uint32_t3){"
       << luisa::format("{}, {}, {}", blk_size.x, blk_size.y, blk_size.z)
       << "};\n}\n";
    print_extern();
    sb << "uint64_t2 " << entry_name << "_args_md5_c4434d750cf64f0eae3f73cca8650b16(){\nreturn (uint64_t2){";
    luisa::vector<char> arg_decs;
    arg_decs.reserve(1024);
    for (auto &i : func.arguments()) {
        auto &&desc = i.type()->description();
        vstd::push_back_all(arg_decs, desc.data(), desc.size());
        arg_decs.emplace_back(' ');
    }
    vstd::MD5 md5{luisa::span{reinterpret_cast<uint8_t const *>(arg_decs.data()), arg_decs.size_bytes()}};
    auto &md5_data = md5.to_binary();
    PrintValue<uint64_t>{}(md5_data.data0, sb);
    sb << ", ";
    PrintValue<uint64_t>{}(md5_data.data1, sb);
    sb << "};\n}\n";
    sb << "typedef struct {\n";
    size_t arg_idx = 0;
    for (auto &i : func.arguments()) {
        sb << "_Alignas(16) ";
        get_type_name(sb, i.type());
        sb << " a" << luisa::format("{}", arg_idx)
           << ";\n";
        ++arg_idx;
    }
    if (func.arguments().empty()) {
        sb << "int8_t _a;\n";
    }
    sb << "} Args;\n";
#ifdef _MSC_VER
    sb << "__declspec(dllexport) ";
#else
    sb << "__attribute__((visibility(\"default\"))) ";
#endif

    sb << " void " << entry_name << "(uint32_t3 thd_id, uint32_t3 blk_id, uint32_t3 dsp_id, uint32_t3 dsp_size, uint32_t3 ker_id, Args* args){\nbuiltin_c4434d750cf64f0eae3f73cca8650b16(thd_id, blk_id, dsp_id, dsp_size, ker_id";
    arg_idx = 0;
    for (auto &i : func.arguments()) {
        sb << ", args->a" << luisa::format("{}", arg_idx);
        arg_idx++;
    }
    sb << ");\n}\n";
    auto f = fopen(path.c_str(), "wb");
    if (!_destructor_types.empty()) {
        struct_sb << R"(
typedef struct {
    void(*ptr)(void*);
} Finalizer;
void call_dtor(void(*func)(void*), void* ptr) {
    if(func) { func(ptr); }
}
)";
    }
    if (f) {
        if (struct_sb.size() > 0)
            fwrite(struct_sb.data(), struct_sb.size(), 1, f);
        if (dtor_sb.size() > 0)
            fwrite(dtor_sb.data(), dtor_sb.size(), 1, f);
        if (decl_sb.size() > 0)
            fwrite(decl_sb.data(), decl_sb.size(), 1, f);
        if (sb.size() > 0)
            fwrite(sb.data(), sb.size(), 1, f);
        fclose(f);
    }
    // order: struct_cb, decl_sb, sb
}
void Clanguage_CodegenUtils::call_external_func(vstd::StringBuilder &sb, CodegenVisitor *visitor, CallExpr const *expr) {
    auto func = expr->external();
    auto name = func->name();
    auto ret_type = func->return_type();
    auto arg_types = func->argument_types();
    auto iter = c_codegen_detail::extern_table.map.find(name);
    auto print_args = [&]() {
        bool comma = false;
        for (auto &i : expr->arguments()) {
            if (comma) {
                sb << ", ";
            }
            comma = true;
            i->accept(*visitor);
        }
    };
    if (!iter) {
        sb << name << '(';
        print_args();
        sb << ')';
        return;
    }
    auto &&func_generator = iter.value();
    switch (func_generator.index()) {
        case 0: {
            auto &&kv = luisa::get<0>(func_generator);
            Key key;
            key.type = 4;
            key.flag = kv.flag;
            key.arg_types.reserve(arg_types.size());
            key.arg_types.emplace_back(ret_type);
            vstd::push_back_all(key.arg_types, arg_types);
            sb << _gen_func(
                      [&](luisa::string_view func_name) {
                          kv.func(*this, decl_sb, func_name, ret_type, arg_types, kv.is_ref);
                      },
                      std::move(key))
               << '(';
            bool comma = false;
            size_t idx = 0;
            for (auto &i : expr->arguments()) {
                if (comma) {
                    sb << ", ";
                }
                if (kv.is_ref.size() > idx && kv.is_ref[idx]) {
                    sb << '&';
                }
                comma = true;
                i->accept(*visitor);
                ++idx;
            }
            sb << ')';
        }
            return;
        case 1: {
            if (!luisa::get<1>(func_generator)(ret_type, arg_types)) {
                LUISA_ERROR("Function {} argument type not match.", name);
            }
            sb << name << '(';
            print_args();
            sb << ')';
        }
            return;
        case 2: {
            auto func_ptr = luisa::get<2>(func_generator);
            func_ptr(sb, visitor, *this, expr);
        }
            return;
    }
}
Clanguage_CodegenUtils::Clanguage_CodegenUtils() = default;
Clanguage_CodegenUtils::~Clanguage_CodegenUtils() = default;
}// namespace luisa::compute