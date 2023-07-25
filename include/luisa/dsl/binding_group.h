//
// Created by Mike Smith on 2023/7/23.
//

#pragma once

#include <luisa/dsl/struct.h>

#define LUISA_BINDING_GROUP_MAKE_MEMBER_VAR_DECL(m) \
    Var<member_type_##m> m;

#define LUISA_BINDING_GROUP_MAKE_MEMBER_EXPR_DECL(m) \
    Expr<member_type_##m> m;

#define LUISA_BINDING_GROUP_MAKE_MEMBER_VAR_INIT(m) \
    m(detail::ArgumentCreation{})

#define LUISA_BINDING_GROUP_MAKE_MEMBER_EXPR_INIT(m) \
    m(s.m)

#define LUISA_BINDING_GROUP_MAKE_INVOKE(m) \
    invoke << s.m;

#define LUISA_BINDING_GROUP(S, ...)                                                     \
    template<>                                                                          \
    struct luisa_compute_extension<S>;                                                  \
    namespace luisa::compute {                                                          \
    template<>                                                                          \
    struct Var<S> {                                                                     \
        using this_type = S;                                                            \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                           \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_MEMBER_VAR_DECL, __VA_ARGS__)                \
        explicit Var(detail::ArgumentCreation) noexcept                                 \
            : LUISA_MAP_LIST(LUISA_BINDING_GROUP_MAKE_MEMBER_VAR_INIT, __VA_ARGS__) {}  \
        Var(Var &&) noexcept = default;                                                 \
        Var(const Var &) noexcept = delete;                                             \
        Var &operator=(Var &&) noexcept = delete;                                       \
        Var &operator=(const Var &) noexcept = delete;                                  \
        [[nodiscard]] auto operator->() noexcept {                                      \
            return reinterpret_cast<luisa_compute_extension<this_type> *>(this);        \
        }                                                                               \
        [[nodiscard]] auto operator->() const noexcept {                                \
            return reinterpret_cast<const luisa_compute_extension<this_type> *>(this);  \
        }                                                                               \
    };                                                                                  \
    template<>                                                                          \
    struct Expr<S> {                                                                    \
        using this_type = S;                                                            \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                           \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_MEMBER_EXPR_DECL, __VA_ARGS__)               \
        Expr(const Var<S> &s) noexcept                                                  \
            : LUISA_MAP_LIST(LUISA_BINDING_GROUP_MAKE_MEMBER_EXPR_INIT, __VA_ARGS__) {} \
        Expr(Expr &&another) noexcept = default;                                        \
        Expr(const Expr &another) noexcept = default;                                   \
        Expr &operator=(Expr) noexcept = delete;                                        \
    };                                                                                  \
    namespace detail {                                                                  \
    CallableInvoke &operator<<(CallableInvoke &invoke, Expr<S> s) noexcept {            \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_INVOKE, __VA_ARGS__)                         \
        return invoke;                                                                  \
    }                                                                                   \
    ShaderInvokeBase &operator<<(ShaderInvokeBase &invoke, const S &s) noexcept {       \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_INVOKE, __VA_ARGS__)                         \
        return invoke;                                                                  \
    }                                                                                   \
    }                                                                                   \
    }                                                                                   \
    template<>                                                                          \
    struct luisa_compute_extension<S> final : luisa::compute::Var<S>
