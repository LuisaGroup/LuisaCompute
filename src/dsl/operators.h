//
// Created by Mike Smith on 2021/9/11.
//

#pragma once

#include <string_view>

#include <core/basic_types.h>
#include <dsl/expr.h>

/// Define global unary operation of dsl objects
#define LUISA_MAKE_GLOBAL_DSL_UNARY_OP(op, op_tag)                                   \
    template<typename T>                                                             \
        requires luisa::compute::is_dsl_v<T>                                         \
    [[nodiscard]] inline auto operator op(T &&expr) noexcept {                       \
        using R = std::remove_cvref_t<                                               \
            decltype(op std::declval<luisa::compute::expr_value_t<T>>())>;           \
        return luisa::compute::dsl::def<R>(                                          \
            luisa::compute::detail::FunctionBuilder::current()->unary(               \
                luisa::compute::Type::of<R>(),                                       \
                luisa::compute::UnaryOp::op_tag,                                     \
                luisa::compute::detail::extract_expression(std::forward<T>(expr)))); \
    }
LUISA_MAKE_GLOBAL_DSL_UNARY_OP(+, PLUS)
LUISA_MAKE_GLOBAL_DSL_UNARY_OP(-, MINUS)
LUISA_MAKE_GLOBAL_DSL_UNARY_OP(!, NOT)
LUISA_MAKE_GLOBAL_DSL_UNARY_OP(~, BIT_NOT)
#undef LUISA_MAKE_GLOBAL_DSL_UNARY_OP

/// Define global binary operation of dsl objects
#define LUISA_MAKE_GLOBAL_DSL_BINARY_OP(op, op_tag_name)                              \
    template<typename Lhs, typename Rhs>                                              \
        requires luisa::compute::any_dsl_v<Lhs, Rhs> &&                               \
                 luisa::is_basic_v<luisa::compute::expr_value_t<Lhs>> &&              \
                 luisa::is_basic_v<luisa::compute::expr_value_t<Rhs>>                 \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {            \
        using namespace std::string_view_literals;                                    \
        static constexpr auto is_logic_op = #op == "||"sv || #op == "&&"sv;           \
        if constexpr (is_logic_op && luisa::compute::is_dsl_v<Rhs>) {                 \
            static_assert(                                                            \
                !luisa::compute::is_scalar_expr_v<Lhs> &&                             \
                    !luisa::compute::is_scalar_expr_v<Rhs>,                           \
                "Operator `&&` and `||` are not allowed for DSL scalars, "            \
                "since the short-circuit semantics are not implemented. "             \
                "Please use `&` and `|` to workaround.");                             \
        }                                                                             \
        static constexpr auto is_bit_op = #op == "&"sv ||                             \
                                          #op == "|"sv ||                             \
                                          #op == "^"sv;                               \
        static constexpr auto is_bool_lhs = luisa::compute::is_boolean_expr_v<Lhs>;   \
        static constexpr auto is_bool_rhs = luisa::compute::is_boolean_expr_v<Rhs>;   \
        if constexpr (is_bit_op) {                                                    \
            static_assert(                                                            \
                (is_bool_lhs && is_bool_rhs) ||                                       \
                    (!is_bool_lhs && !is_bool_rhs),                                   \
                "Mixing boolean and non-boolean operands "                            \
                "for `&`, `|`, and `^` is not allowed.");                             \
        }                                                                             \
        using NormalR = std::remove_cvref_t<                                          \
            decltype(std::declval<luisa::compute::expr_value_t<Lhs>>() op             \
                         std::declval<luisa::compute::expr_value_t<Rhs>>())>;         \
        using R = std::conditional_t<is_bit_op && is_bool_lhs, bool, NormalR>;        \
        return luisa::compute::dsl::def<R>(                                           \
            luisa::compute::detail::FunctionBuilder::current()->binary(               \
                luisa::compute::Type::of<R>(),                                        \
                luisa::compute::BinaryOp::op_tag_name,                                \
                luisa::compute::detail::extract_expression(std::forward<Lhs>(lhs)),   \
                luisa::compute::detail::extract_expression(std::forward<Rhs>(rhs)))); \
    }
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(+, ADD)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(-, SUB)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(*, MUL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(/, DIV)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(%, MOD)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(&, BIT_AND)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(|, BIT_OR)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(^, BIT_XOR)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(<<, SHL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(>>, SHR)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(&&, AND)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(||, OR)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(==, EQUAL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(!=, NOT_EQUAL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(<, LESS)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(<=, LESS_EQUAL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(>, GREATER)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(>=, GREATER_EQUAL)
#undef LUISA_MAKE_GLOBAL_DSL_BINARY_OP

/// Define global assign operation of dsl objects
#define LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(op)                                                               \
    template<typename T, typename U>                                                                      \
        requires requires { std::declval<T &>() op## =                                                    \
                                std::declval<luisa::compute::expr_value_t<U>>(); } \
    void operator op##=(luisa::compute::Var<T> &lhs, U &&rhs) noexcept {                                  \
        auto x = lhs op std::forward<U>(rhs);                                                             \
        luisa::compute::detail::FunctionBuilder::current()->assign(                                       \
            lhs.expression(), x.expression());                                                            \
    }
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(+)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(-)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(*)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(/)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(%)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(&)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(|)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(^)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(<<)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(>>)

#define LUISA_DISABLE_DSL_ADDRESS_OF_MESSAGE                 \
    "\n"                                                     \
    "Address-of operator is not allowed for DSL objects,\n"  \
    "as it is only valid during the AST recording phrase\n"  \
    "and will not behave as expected like a real pointer\n"  \
    "during kernel execution.\n"                             \
    "\n"                                                     \
    "For example, if allowed, the following code\n"          \
    "```\n"                                                  \
    "UInt *a = nullptr;\n"                                   \
    "$if (cond) {\n"                                         \
    "  a = &b;\n"                                            \
    "} $else {\n"                                            \
    "  a = &c;\n"                                            \
    "};\n"                                                   \
    "```\n"                                                  \
    "will make `a` **always** pointing to `c`.\n"            \
    "\n"                                                     \
    "Please use references to pass variables between\n"      \
    "functions. Or, if you fully understand the semantics\n" \
    "and effects, please use `std::addressof` instead for\n" \
    "advanced usage.\n"

// disable the address-of operator for dsl objects
template<typename T>
    requires ::luisa::compute::is_dsl_v<T>
[[nodiscard]] inline T *operator&(T &&) noexcept {
    static_assert(::luisa::always_false_v<T>,
                  LUISA_DISABLE_DSL_ADDRESS_OF_MESSAGE);
    std::abort();
}

// convenience macro to disable the address-of operator for specific types
#define LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(...)           \
    template<typename T>                                     \
        requires std::same_as<T, __VA_ARGS__>                \
    [[nodiscard]] inline T *operator&(T &&) noexcept {       \
        static_assert(::luisa::always_false_v<T>,            \
                      LUISA_DISABLE_DSL_ADDRESS_OF_MESSAGE); \
        std::abort();                                        \
    }

#undef LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP
