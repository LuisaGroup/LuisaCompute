#pragma once
#include <luisa/vstl/common.h>
#include <luisa/ast/type.h>
#include <luisa/ast/usage.h>
#include <luisa/ast/op.h>
#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/constant_data.h>
#include "../hlsl/string_builder.h"
#include <luisa/core/mathematics.h>
#include <luisa/core/logging.h>
namespace luisa::compute {
class CodegenVisitor;
class Clanguage_CodegenUtils {
    struct Key {
        // 0: unary
        // 1: binary
        // 2: Call op
        // 3: swizzle
        // 4: external
        // 5: access
        uint8_t type;
        uint32_t flag;
        luisa::fixed_vector<const Type *, 2> arg_types;
        uint64_t hash;
    };
    struct KeyHash {
        size_t operator()(Key const &key) const {
            return key.hash;
        }
    };
    struct KeyCompare {
        int32_t operator()(Key const &a, Key const &b) const {
            if (a.hash > b.hash) return 1;
            if (a.hash < b.hash) return -1;
            if (a.type > b.type) return 1;
            if (a.type < b.type) return -1;
            if (a.flag > b.flag) return 1;
            if (a.flag < b.flag) return -1;
            if (a.arg_types.size() > b.arg_types.size()) return 1;
            if (a.arg_types.size() < b.arg_types.size()) return -1;
            if (!a.arg_types.empty()) {
                return std::memcmp(a.arg_types.data(), b.arg_types.data(), a.arg_types.size_bytes());
            }
            return 0;
        }
    };
    using FuncMap = vstd::HashMap<Key, vstd::StringBuilder, KeyHash, KeyCompare>;
    FuncMap func_map;
    vstd::HashMap<Type const *, size_t> _custom_types;
    vstd::HashMap<detail::FunctionBuilder const *, size_t> _custom_funcs;
    bool _get_custom_type(vstd::StringBuilder &sb, Type const *t);
    vstd::StringBuilder _gen_func_name(Key const &key);
    template<typename Func>
        requires std::is_invocable_v<Func, luisa::string_view>
    luisa::string_view _gen_func(
        Func &&func,
        Key &&key) {
        key.hash = luisa::hash64(&key.flag, sizeof(key.flag), luisa::hash64_default_seed) + key.type;
        if (!key.arg_types.empty()) {
            key.hash = luisa::hash64(key.arg_types.data(), key.arg_types.size_bytes(), key.hash);
        }
        auto iter = func_map.try_emplace(std::move(key));
        if (iter.second) {
            auto func_name = _gen_func_name(iter.first.key());
            func(func_name.view());
            iter.first.value() = std::move(func_name);
        }
        return iter.first.value().view();
    }

public:
    vstd::StringBuilder struct_sb;
    vstd::StringBuilder decl_sb;
    vstd::HashMap<uint64> const_set;

    Clanguage_CodegenUtils();
    ~Clanguage_CodegenUtils();
    Clanguage_CodegenUtils(Clanguage_CodegenUtils const &) = delete;
    Clanguage_CodegenUtils(Clanguage_CodegenUtils &&) = delete;
    static void replace(char *ptr, size_t len, char src, char dst);
    void get_type_name(vstd::StringBuilder &sb, Type const *type);
    void print_function_declare(vstd::StringBuilder &sb, Function func);
    void print_kernel_declare(vstd::StringBuilder &sb, Function func);
    void call_external_func(
        vstd::StringBuilder &sb, CodegenVisitor *visitor, CallExpr const *expr);
    vstd::StringBuilder get_type_name(Type const *type) {
        vstd::StringBuilder r;
        get_type_name(r, type);
        return r;
    }
    // replace '#' into swizzle,
    // if arg is ("a.# + b.#", type(int3)), result will be "(int3){a.x + b.x, a.y + b.y, a.z + b.z}"
    void gen_vec_function(vstd::StringBuilder &sb, vstd::string_view expr, Type const *type);
    void gen_var_name(vstd::StringBuilder &sb, Variable const &var);
    vstd::StringBuilder gen_var_name(Variable const &var) {
        vstd::StringBuilder r;
        gen_var_name(r, var);
        return r;
    }
    vstd::StringBuilder gen_vec_function(vstd::string_view expr, Type const *type) {
        vstd::StringBuilder r;
        gen_vec_function(r, expr, type);
        return r;
    }
    size_t func_index(Function f);
    void gen_constant(vstd::StringBuilder &sb, ConstantData const &data);
    luisa::string_view gen_vec_swizzle(luisa::span<uint const> swizzle, uint swizzle_code, Type const *arg);
    luisa::string_view gen_vec_unary(UnaryOp op, Type const *type);
    luisa::string_view gen_vec_binary(BinaryOp op, Type const *left_type, Type const *right_type);
    luisa::string_view gen_callop(CallOp op, Type const *return_type, luisa::span<Type const *const> arg_types);
    luisa::string_view gen_access(Type const *return_type, luisa::span<Type const *const> arg_types, bool is_self_rvalue);
    luisa::string_view gen_make_vec(Type const *return_type, luisa::span<Type const *const> arg_types);
    void codegen(
        luisa::string const &path,
        luisa::string_view entry_name,
        Function func);
};
template<typename T>
struct PrintValue;

template<>
struct PrintValue<float> {
    void operator()(float const &v, vstd::StringBuilder &str) {
        if (luisa::isnan(v)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (luisa::isinf(v)) [[unlikely]] {
            str.append(v < 0.0f ? "(-_INF_f)" : "(_INF_f)");
        } else {
            vstd::to_string(v, str);
        }
    }
};

template<>
struct PrintValue<double> {
    void operator()(double const &v, vstd::StringBuilder &str) {
        if (luisa::isnan(v)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Encountered with NaN.");
        }
        if (luisa::isinf(v)) [[unlikely]] {
            str.append(v < 0.0 ? "(-_INF_d)" : "(_INF_d)");
        } else {
            str.append(luisa::format("(double){}", v));
        }
    }
};

template<>
struct PrintValue<half> {
    void operator()(half const &v, vstd::StringBuilder &str) {
        LUISA_ERROR("Half not supported by backend.");
    }
};

template<>
struct PrintValue<short> {
    void operator()(short const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("(int16_t){}", v));
    }
};

template<>
struct PrintValue<ushort> {
    void operator()(ushort const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("(uint16_t){}u", v));
    }
};

template<>
struct PrintValue<int> {
    void operator()(int const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("{}", v));
    }
};

template<>
struct PrintValue<uint> {
    void operator()(uint const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("{}u", v));
    }
};

template<>
struct PrintValue<slong> {
    void operator()(slong const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("{}ll", v));
    }
};

template<>
struct PrintValue<ulong> {
    void operator()(ulong const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("{}ull", v));
    }
};

template<>
struct PrintValue<bool> {
    void operator()(bool const &v, vstd::StringBuilder &str) {
        if (v)
            str << "true";
        else
            str << "false";
    }
};
template<>
struct PrintValue<luisa::byte> {
    void operator()(luisa::byte const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("(int8_t){}", v));
    }
};
template<>
struct PrintValue<luisa::ubyte> {
    void operator()(luisa::ubyte const &v, vstd::StringBuilder &str) {
        str.append(luisa::format("(uint8_t){}", v));
    }
};
template<typename EleType, uint64 N>
struct PrintValue<Vector<EleType, N>> {
    using T = Vector<EleType, N>;
    void print_elem(T const &v, vstd::StringBuilder &varName) {
        for (uint64 i = 0; i < N; ++i) {
            PrintValue<EleType>{}(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
    }
    void operator()(T const &v, vstd::StringBuilder &varName) {
        if constexpr (N > 1) {
            varName << '(';
            if constexpr (std::is_same_v<EleType, float>) {
                varName << "float";
            } else if constexpr (std::is_same_v<EleType, uint>) {
                varName << "uint32_t";
            } else if constexpr (std::is_same_v<EleType, int>) {
                varName << "int32_t";
            } else if constexpr (std::is_same_v<EleType, bool>) {
                varName << "bool";
            } else if constexpr (std::is_same_v<EleType, half>) {
                LUISA_ERROR("Half not supported by backend.");
            } else if constexpr (std::is_same_v<EleType, double>) {
                varName << "double";
            } else if constexpr (std::is_same_v<EleType, short>) {
                varName << "int16_t";
            } else if constexpr (std::is_same_v<EleType, ushort>) {
                varName << "uint16_t";
            } else if constexpr (std::is_same_v<EleType, slong>) {
                varName << "int64_t";
            } else if constexpr (std::is_same_v<EleType, ulong>) {
                varName << "uint64_t";
            } else {
                // static_assert(luisa::always_false_v<T>, "Unsupported type.");
                LUISA_ERROR_WITH_LOCATION("Unsupported type.");
            }
            vstd::to_string(N, varName);
            varName << "){";
            print_elem(v, varName);
            varName << '}';
        } else {
            print_elem(v, varName);
        }
    }
};

template<uint64 N>
struct PrintValue<Matrix<N>> {
    using T = Matrix<N>;
    using EleType = float;
    void operator()(T const &v, vstd::StringBuilder &varName) {
        varName << "make_float";
        auto ss = vstd::to_string(N);
        varName << ss << 'x' << ss << "_1";
        PrintValue<Vector<EleType, N>> vecPrinter;
        for (uint64 i = 0; i < N; ++i) {
            vecPrinter.print_elem(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
        varName << ')';
    }
};

}// namespace luisa::compute