#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/dsl/expr_traits.h>
#include <luisa/dsl/expr.h>

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

namespace luisa::compute::detail {

template<BinaryOp op, typename Lhs, typename Rhs>
constexpr auto dsl_binary_op_return_type_helper() noexcept {

    static_assert(!any_dsl_v<Lhs, Rhs>);

    constexpr auto lhs_is_scalar = luisa::is_scalar_v<Lhs>;
    constexpr auto rhs_is_scalar = luisa::is_scalar_v<Rhs>;
    constexpr auto lhs_is_vector = luisa::is_vector_v<Lhs>;
    constexpr auto rhs_is_vector = luisa::is_vector_v<Rhs>;
    constexpr auto lhs_is_matrix = luisa::is_matrix_v<Lhs>;
    constexpr auto rhs_is_matrix = luisa::is_matrix_v<Rhs>;
    constexpr auto lhs_is_integral = luisa::is_integral_or_vector_v<Lhs>;
    constexpr auto rhs_is_integral = luisa::is_integral_or_vector_v<Rhs>;
    constexpr auto lhs_is_boolean = luisa::is_boolean_or_vector_v<Lhs>;
    constexpr auto rhs_is_boolean = luisa::is_boolean_or_vector_v<Rhs>;
    constexpr auto lhs_is_fp = is_floating_point_or_vector_v<Lhs> || is_matrix_v<Lhs>;
    constexpr auto rhs_is_fp = is_floating_point_or_vector_v<Rhs> || is_matrix_v<Rhs>;

    using lhs_elem = std::conditional_t<lhs_is_matrix, float, vector_element_t<Lhs>>;
    using rhs_elem = std::conditional_t<rhs_is_matrix, float, vector_element_t<Rhs>>;
    // we only allow implicit conversion between
    //  - scalars and vectors with the same element type
    //  - integral and floating-point scalars
    //  - equally sized integral scalars
    //  to avoid unexpected (and typically expensive) behaviors
    static_assert(
        // no conversion; or scalars and vectors with the same element type
        std::is_same_v<lhs_elem, rhs_elem> ||
            // integral and floating-point scalars
            (lhs_is_scalar && rhs_is_scalar &&
             ((lhs_is_integral && rhs_is_fp) ||
              (lhs_is_fp && rhs_is_integral) ||
              (lhs_is_integral && rhs_is_integral &&
               sizeof(lhs_elem) == sizeof(rhs_elem)))),
        "Binary operator requires operands "
        "of the same element type.");

    constexpr auto lhs = expr_value_t<Lhs>{};
    constexpr auto rhs = expr_value_t<Rhs>{};

    if constexpr (op == BinaryOp::ADD ||
                  op == BinaryOp::SUB ||
                  op == BinaryOp::MUL ||
                  op == BinaryOp::DIV) {
        static_assert((lhs_is_integral || lhs_is_fp) &&
                          (rhs_is_integral || rhs_is_fp),
                      "Arithmetic operator requires integral "
                      "or floating-point operands.");
        return decltype(lhs * rhs){};
    } else if constexpr (op == BinaryOp::MOD) {
        static_assert(lhs_is_integral && rhs_is_integral,
                      "Modulo operator requires integral operands.");
        return decltype(lhs % rhs){};
    } else if constexpr (op == BinaryOp::EQUAL ||
                         op == BinaryOp::NOT_EQUAL ||
                         op == BinaryOp::LESS ||
                         op == BinaryOp::LESS_EQUAL ||
                         op == BinaryOp::GREATER ||
                         op == BinaryOp::GREATER_EQUAL) {
        return decltype(lhs == rhs){};
    } else if constexpr (op == BinaryOp::BIT_AND ||
                         op == BinaryOp::BIT_OR ||
                         op == BinaryOp::BIT_XOR) {
        static_assert((lhs_is_integral && rhs_is_integral) ||
                          (lhs_is_scalar && lhs_is_boolean &&
                           rhs_is_scalar && rhs_is_boolean),
                      "Bitwise operator requires integral or boolean operands.");
        if constexpr (lhs_is_integral) {
            return decltype(lhs & rhs){};
        } else {
            return decltype(lhs && rhs){};
        }
    } else if constexpr (op == BinaryOp::AND ||
                         op == BinaryOp::OR) {
        static_assert(lhs_is_vector && rhs_is_vector &&
                          lhs_is_boolean && rhs_is_boolean,
                      "Logical operator requires boolean vector operands.");
        return decltype(lhs && rhs){};
    } else if constexpr (op == BinaryOp::SHL ||
                         op == BinaryOp::SHR) {
        static_assert(lhs_is_integral && rhs_is_integral,
                      "Shift operator requires integral operands.");
        return decltype(lhs << rhs){};
    } else {
        static_assert(always_false_v<Lhs>);
    }
}

template<BinaryOp op, typename Lhs, typename Rhs>
using dsl_binary_op_return_type =
    decltype(dsl_binary_op_return_type_helper<
             op, expr_value_t<Lhs>, expr_value_t<Rhs>>());

}// namespace luisa::compute::detail

/// Define global binary operation of dsl objects
#define LUISA_MAKE_GLOBAL_DSL_BINARY_OP(op, op_tag_name)                              \
    template<typename Lhs, typename Rhs>                                              \
        requires luisa::compute::any_dsl_v<Lhs, Rhs> &&                               \
                 luisa::is_basic_v<luisa::compute::expr_value_t<Lhs>> &&              \
                 luisa::is_basic_v<luisa::compute::expr_value_t<Rhs>>                 \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {            \
        using R = luisa::compute::detail::dsl_binary_op_return_type<                  \
            luisa::compute::BinaryOp::op_tag_name, Lhs, Rhs>;                         \
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

/// Define global assign operation of dsl objects; returns *const* reference to lhs
#define LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(op)                                     \
    template<typename T, typename U>                                            \
        requires std::same_as<decltype(std::declval<luisa::compute::Var<T> &>() \
                                           op std::declval<U &>()),             \
                              luisa::compute::Var<T>>                           \
    const auto &operator op##=(luisa::compute::Var<T> &lhs, U &&rhs) noexcept { \
        auto x = lhs op std::forward<U>(rhs);                                   \
        luisa::compute::detail::FunctionBuilder::current()->assign(             \
            lhs.expression(), x.expression());                                  \
        return lhs;                                                             \
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
#undef LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP

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
