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

template<typename, typename>// for better warning messages
[[deprecated(
    "\n\n"
    "Implicit conversion between floating-point and integral values detected.\n"
    "LuisaCompute DSL will automatically insert type conversions to make the\n"
    "code compile, but this is not recommended and could be error prone.\n"
    "Please consider explicitly casting the operands of the binary operator.\n"
    "\n")]]
// empty function to generate a warning
inline void
dsl_binary_op_fp_integral_implicit_conversion_detected() noexcept {}

template<BinaryOp op, typename Lhs, typename Rhs>
constexpr auto// (ret, lhs_cast, rhs_cast)
dsl_binary_op_return_type_helper() noexcept {

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
    //  - for SHL/SHR, rhs can be any integral scalar and will be cast to the same element type of lhs
    //  to avoid unexpected (and typically expensive) behaviors
    static_assert(
        // no conversion; or scalars and vectors with the same element type
        std::is_same_v<lhs_elem, rhs_elem> ||
            // integral and floating-point scalars
            (lhs_is_scalar && rhs_is_scalar &&
                 ((lhs_is_integral && rhs_is_fp) ||
                  (lhs_is_fp && rhs_is_integral) ||
                  (lhs_is_integral && rhs_is_integral &&
                   sizeof(lhs_elem) == sizeof(rhs_elem))) ||
             // for SHL/SHR, the rhs might be any unsigned integer
             ((op == BinaryOp::SHL || op == BinaryOp::SHR) &&
              lhs_is_integral && rhs_is_scalar && rhs_is_integral)),
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
        if constexpr (lhs_is_scalar && rhs_is_scalar) {
            if constexpr (lhs_is_integral && rhs_is_fp) {
                dsl_binary_op_fp_integral_implicit_conversion_detected<Lhs, Rhs>();
                return std::make_tuple(rhs, rhs, rhs);
            } else if constexpr (lhs_is_fp && rhs_is_integral) {
                dsl_binary_op_fp_integral_implicit_conversion_detected<Lhs, Rhs>();
                return std::make_tuple(lhs, lhs, lhs);
            } else {
                auto ret = decltype(lhs * rhs){};
                return std::make_tuple(ret, ret, ret);
            }
        } else {
            auto ret = decltype(lhs * rhs){};
            return std::make_tuple(ret, lhs, rhs);
        }
    } else if constexpr (op == BinaryOp::MOD) {
        static_assert(lhs_is_integral && rhs_is_integral,
                      "Modulo operator requires integral operands.");
        return std::make_tuple(decltype(lhs % rhs){}, lhs, rhs);
    } else if constexpr (op == BinaryOp::EQUAL ||
                         op == BinaryOp::NOT_EQUAL ||
                         op == BinaryOp::LESS ||
                         op == BinaryOp::LESS_EQUAL ||
                         op == BinaryOp::GREATER ||
                         op == BinaryOp::GREATER_EQUAL) {
        if constexpr (lhs_is_scalar && rhs_is_scalar) {
            if constexpr (lhs_is_integral && rhs_is_fp) {
                dsl_binary_op_fp_integral_implicit_conversion_detected<Lhs, Rhs>();
                return std::make_tuple(bool{}, rhs, rhs);
            } else if constexpr (lhs_is_fp && rhs_is_integral) {
                dsl_binary_op_fp_integral_implicit_conversion_detected<Lhs, Rhs>();
                return std::make_tuple(bool{}, lhs, lhs);
            } else {
                return std::make_tuple(bool{}, lhs, rhs);
            }
        } else {
            auto ret = decltype(lhs == rhs){};
            return std::make_tuple(ret, lhs, rhs);
        }
    } else if constexpr (op == BinaryOp::BIT_AND ||
                         op == BinaryOp::BIT_OR ||
                         op == BinaryOp::BIT_XOR) {
        static_assert((lhs_is_integral && rhs_is_integral) ||
                          (lhs_is_scalar && lhs_is_boolean &&
                           rhs_is_scalar && rhs_is_boolean),
                      "Bitwise operator requires integral or boolean operands.");
        if constexpr (lhs_is_integral) {
            return std::make_tuple(decltype(lhs & rhs){}, lhs, rhs);
        } else {
            return std::make_tuple(decltype(lhs && rhs){}, lhs, rhs);
        }
    } else if constexpr (op == BinaryOp::AND ||
                         op == BinaryOp::OR) {
        static_assert(lhs_is_vector && rhs_is_vector &&
                          lhs_is_boolean && rhs_is_boolean,
                      "Logical operator requires boolean vector operands. To perform "
                      "logical operations on scalars, please use the bitwise operators "
                      "and note that the short-circuit evaluation is not supported.");
        return std::make_tuple(decltype(lhs && rhs){}, lhs, rhs);
    } else if constexpr (op == BinaryOp::SHL ||
                         op == BinaryOp::SHR) {
        static_assert(lhs_is_integral && rhs_is_integral,
                      "Shift operator requires integral operands.");
        if constexpr (rhs_is_scalar) {
            auto rhs_cast = lhs_elem{};
            auto ret = decltype(lhs << rhs_cast){};
            return std::make_tuple(ret, lhs, rhs_cast);
        } else {
            auto ret = decltype(lhs << rhs){};
            return std::make_tuple(ret, lhs, rhs);
        }
    } else {
        static_assert(always_false_v<Lhs>);
    }
}

}// namespace luisa::compute::detail

/// Define global binary operation of dsl objects
#define LUISA_MAKE_GLOBAL_DSL_BINARY_OP(op, op_tag_name)                                     \
    template<typename LhsT, typename RhsT>                                                   \
        requires luisa::compute::any_dsl_v<LhsT, RhsT> &&                                    \
                 luisa::is_basic_v<luisa::compute::expr_value_t<LhsT>> &&                    \
                 luisa::is_basic_v<luisa::compute::expr_value_t<RhsT>>                       \
    [[nodiscard]] inline auto operator op(LhsT &&lhs, RhsT &&rhs) noexcept {                 \
        using Lhs = luisa::compute::expr_value_t<LhsT>;                                      \
        using Rhs = luisa::compute::expr_value_t<RhsT>;                                      \
        auto [ret, lhs_cast, rhs_cast] =                                                     \
            luisa::compute::detail::dsl_binary_op_return_type_helper<                        \
                luisa::compute::BinaryOp::op_tag_name, Lhs, Rhs>();                          \
        using Ret = decltype(ret);                                                           \
        using LhsCast = decltype(lhs_cast);                                                  \
        using RhsCast = decltype(rhs_cast);                                                  \
        auto lhs_expr = luisa::compute::detail::extract_expression(std::forward<LhsT>(lhs)); \
        auto rhs_expr = luisa::compute::detail::extract_expression(std::forward<RhsT>(rhs)); \
        auto fb = luisa::compute::detail::FunctionBuilder::current();                        \
        if constexpr (!std::is_same_v<LhsCast, Lhs>) {                                       \
            lhs_expr = fb->cast(luisa::compute::Type::of<LhsCast>(),                         \
                                luisa::compute::CastOp::STATIC, lhs_expr);                   \
        }                                                                                    \
        if constexpr (!std::is_same_v<RhsCast, Rhs>) {                                       \
            rhs_expr = fb->cast(luisa::compute::Type::of<RhsCast>(),                         \
                                luisa::compute::CastOp::STATIC, rhs_expr);                   \
        }                                                                                    \
        return luisa::compute::dsl::def<Ret>(fb->binary(                                     \
            luisa::compute::Type::of<Ret>(),                                                 \
            luisa::compute::BinaryOp::op_tag_name,                                           \
            lhs_expr, rhs_expr));                                                            \
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
#define LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(op)                                      \
    template<typename T, typename U>                                             \
        requires std::same_as<decltype(std::declval<luisa::compute::Var<T> &>()  \
                                           op std::declval<U &>()),              \
                              luisa::compute::Var<T>>                            \
    const auto &operator op##=(luisa::compute::Var<T> &lhs, U && rhs) noexcept { \
        auto x = lhs op std::forward<U>(rhs);                                    \
        luisa::compute::detail::FunctionBuilder::current()->assign(              \
            lhs.expression(), x.expression());                                   \
        return lhs;                                                              \
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
