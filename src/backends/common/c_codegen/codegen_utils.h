#pragma once
#include <luisa/vstl/common.h>
#include <luisa/ast/type.h>
#include <luisa/ast/usage.h>
#include <luisa/ast/op.h>
#include "../hlsl/string_builder.h"
namespace luisa::compute {
class Clanguage_CodegenUtils {
    struct Key {
        // 0: unary
        // 1: binary
        // 2: Call op
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
    static vstd::StringBuilder _gen_func_name(Key const &key);
    template<typename Func>
        requires std::is_invocable_v<Func, luisa::string_view>
    luisa::string_view _gen_func(
        Func &&func,
        Key &&key) {
        key.hash = luisa::hash64(&key.flag, sizeof(key.flag), luisa::hash64_default_seed) + key.type;
        if (!key.arg_types.empty()) {
            key.hash = luisa::hash64(key.arg_types.data(), key.arg_types.size_bytes(), key.hash);
        }
        for (auto &i : key.arg_types) {
            auto type_hs = i->hash();
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
    Clanguage_CodegenUtils();
    ~Clanguage_CodegenUtils();
    Clanguage_CodegenUtils(Clanguage_CodegenUtils const &) = delete;
    Clanguage_CodegenUtils(Clanguage_CodegenUtils &&) = delete;
    static void replace(char *ptr, size_t len, char src, char dst);
    static void get_type_name(vstd::StringBuilder &sb, Type const *type);
    static vstd::StringBuilder get_type_name(Type const *type) {
        vstd::StringBuilder r;
        get_type_name(r, type);
        return r;
    }
    // replace '#' into swizzle,
    // if arg is ("a.# + b.#", type(int3)), result will be "(int3){a.x + b.x, a.y + b.y, a.z + b.z}"
    static void gen_vec_function(vstd::StringBuilder &sb, vstd::string_view expr, Type const *type);
    static vstd::StringBuilder gen_vec_function(vstd::string_view expr, Type const *type) {
        vstd::StringBuilder r;
        gen_vec_function(r, expr, type);
        return r;
    }
    luisa::string_view gen_vec_unary(vstd::StringBuilder &decl_sb, UnaryOp op, Type const *type);
    luisa::string_view gen_vec_binary(vstd::StringBuilder &decl_sb, BinaryOp op, Type const *left_type, Type const *right_type);
    luisa::string_view gen_callop(vstd::StringBuilder &decl_sb, CallOp op, Type const *return_type, luisa::span<Type const *const> arg_types);
    luisa::string_view gen_make_vec(vstd::StringBuilder &decl_sb, Type const *return_type, luisa::span<Type const *const> arg_types);
};
}// namespace luisa::compute