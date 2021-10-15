//
// Created by Mike Smith on 2021/9/11.
//

#pragma once

#include <core/basic_types.h>
#include <dsl/expr.h>

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

#define LUISA_MAKE_GLOBAL_DSL_BINARY_OP(op, op_tag_name)                              \
    template<typename Lhs, typename Rhs>                                              \
        requires luisa::compute::any_dsl_v<Lhs, Rhs>                                  \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {            \
        using R = std::remove_cvref_t<                                                \
            decltype(std::declval<luisa::compute::expr_value_t<Lhs>>() op             \
                         std::declval<luisa::compute::expr_value_t<Rhs>>())>;         \
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

#define LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(op, op_tag_name)                                              \
    template<typename T, typename U>                                                                  \
        requires requires { std::declval<T &>() op std::declval<luisa::compute::expr_value_t<U>>(); } \
    void operator op(luisa::compute::Var<T> &lhs, U &&rhs) noexcept {                                 \
        luisa::compute::detail::FunctionBuilder::current()->assign(                                   \
            luisa::compute::AssignOp::op_tag_name,                                                    \
            lhs.expression(),                                                                         \
            luisa::compute::detail::extract_expression(std::forward<U>(rhs)));                        \
    }
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(+=, ADD_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(-=, SUB_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(*=, MUL_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(/=, DIV_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(%=, MOD_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(&=, BIT_AND_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(|=, BIT_OR_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(^=, BIT_XOR_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(<<=, SHL_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(>>=, SHR_ASSIGN)
#undef LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP
