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

#define LUISA_BINDING_GROUP_MAKE_MEMBER_ENCODE_COUNT(m) \
    (shader_argument_encode_count<member_type_##m>::value) +

#define LUISA_BINDING_GROUP_TEMPLATE_IMPL(TEMPLATE, TEMPLATE2, S, ...)                  \
    LUISA_MACRO_EVAL(TEMPLATE())                                                        \
    struct luisa_compute_extension<S>;                                                  \
    namespace luisa::compute {                                                          \
    LUISA_MACRO_EVAL(TEMPLATE())                                                        \
    struct Var<S> {                                                                     \
        using is_binding_group = void;                                                  \
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
    LUISA_MACRO_EVAL(TEMPLATE())                                                        \
    struct Expr<S> {                                                                    \
        using is_binding_group = void;                                                  \
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
    LUISA_MACRO_EVAL(TEMPLATE())                                                        \
    struct shader_argument_encode_count<S> {                                            \
        using this_type = S;                                                            \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                           \
        static constexpr uint value =                                                   \
            LUISA_MAP(LUISA_BINDING_GROUP_MAKE_MEMBER_ENCODE_COUNT, __VA_ARGS__)        \
        0u;                                                                             \
    };                                                                                  \
    LUISA_MACRO_EVAL(TEMPLATE2())                                                       \
    void callable_encode_binding_group(CallableInvoke &invoke, Expr<S> s) noexcept {    \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_INVOKE, __VA_ARGS__)                         \
    }                                                                                   \
    LUISA_MACRO_EVAL(TEMPLATE2())                                                       \
    ShaderInvokeBase &operator<<(ShaderInvokeBase &invoke, const S &s) noexcept {       \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_INVOKE, __VA_ARGS__)                         \
        return invoke;                                                                  \
    }                                                                                   \
    }                                                                                   \
    }                                                                                   \
    LUISA_MACRO_EVAL(TEMPLATE())                                                        \
    struct luisa_compute_extension<S> final : luisa::compute::Var<S>

#define LUISA_BIND_GROUP_IMPL_EMPTY_TEMPLATE() template<>
#define LUISA_BIND_GROUP_IMPL_EMPTY()

#define LUISA_BINDING_GROUP(S, ...)           \
    LUISA_BINDING_GROUP_TEMPLATE_IMPL(        \
        LUISA_BIND_GROUP_IMPL_EMPTY_TEMPLATE, \
        LUISA_BIND_GROUP_IMPL_EMPTY,          \
        S, __VA_ARGS__)

#define LUISA_BINDING_GROUP_TEMPLATE(TEMPLATE, S, ...) \
    LUISA_BINDING_GROUP_TEMPLATE_IMPL(                 \
        TEMPLATE, TEMPLATE, S, __VA_ARGS__)
