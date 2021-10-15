//
// Created by Mike Smith on 2021/9/11.
//

#pragma once

#include <dsl/expr.h>

#define LUISA_MAKE_GLOBAL_DSL_UNARY_OP(op, op_concept, op_tag)                       \
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
LUISA_MAKE_GLOBAL_DSL_UNARY_OP(+, operator_plus, PLUS)
LUISA_MAKE_GLOBAL_DSL_UNARY_OP(-, operator_minus, MINUS)
LUISA_MAKE_GLOBAL_DSL_UNARY_OP(!, operator_not, NOT)
LUISA_MAKE_GLOBAL_DSL_UNARY_OP(~, operator_bit_not, BIT_NOT)
#undef LUISA_MAKE_GLOBAL_DSL_UNARY_OP

#define LUISA_MAKE_GLOBAL_DSL_BINARY_OP(op, op_concept_name, op_tag_name)             \
    template<typename Lhs, typename Rhs>                                              \
        requires luisa::compute::any_dsl_v<Lhs, Rhs>                                  \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept              \
        ->luisa::compute::Var<std::remove_cvref_t<                                    \
            decltype(std::declval<luisa::compute::expr_value_t<Lhs>>()                \
                         op std::declval<luisa::compute::expr_value_t<Rhs>>())>> {    \
        using R = std::remove_cvref_t<                                                \
            decltype(std::declval<luisa::compute::expr_value_t<Lhs>>()                \
                         op std::declval<luisa::compute::expr_value_t<Rhs>>())>;      \
        return luisa::compute::dsl::def<R>(                                           \
            luisa::compute::detail::FunctionBuilder::current()->binary(               \
                luisa::compute::Type::of<R>(),                                        \
                luisa::compute::BinaryOp::op_tag_name,                                \
                luisa::compute::detail::extract_expression(std::forward<Lhs>(lhs)),   \
                luisa::compute::detail::extract_expression(std::forward<Rhs>(rhs)))); \
    }
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(+, operator_add, ADD)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(-, operator_sub, SUB)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(*, operator_mul, MUL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(/, operator_div, DIV)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(%, operator_mod, MOD)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(&, operator_bit_and, BIT_AND)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(|, operator_bit_or, BIT_OR)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(^, operator_bit_Xor, BIT_XOR)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(<<, operator_shift_left, SHL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(>>, operator_shift_right, SHR)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(&&, operator_and, AND)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(||, operator_or, OR)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(==, operator_equal, EQUAL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(!=, operator_not_equal, NOT_EQUAL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(<, operator_less, LESS)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(<=, operator_less_equal, LESS_EQUAL)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(>, operator_greater, GREATER)
LUISA_MAKE_GLOBAL_DSL_BINARY_OP(>=, operator_greater_equal, GREATER_EQUAL)
#undef LUISA_MAKE_GLOBAL_DSL_BINARY_OP

#define LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(op, op_name, op_tag_name)              \
    template<typename T, typename U>                                           \
    requires luisa::concepts::op_name##able<                                   \
        T, luisa::compute::expr_value_t<U>> void                               \
    operator op(luisa::compute::Var<T> &lhs, U &&rhs) noexcept {               \
        luisa::compute::detail::FunctionBuilder::current()->assign(            \
            luisa::compute::AssignOp::op_tag_name,                             \
            lhs.expression(),                                                  \
            luisa::compute::detail::extract_expression(std::forward<U>(rhs))); \
    }
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(+=, add_assign, ADD_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(-=, sub_assign, SUB_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(*=, mul_assign, MUL_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(/=, div_assign, DIV_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(%=, mod_assign, MOD_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(&=, bit_and_assign, BIT_AND_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(|=, bit_or_assign, BIT_OR_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(^=, bit_xor_assign, BIT_XOR_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(<<=, shift_left_assign, SHL_ASSIGN)
LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP(>>=, shift_right_assign, SHR_ASSIGN)
#undef LUISA_MAKE_GLOBAL_DSL_ASSIGN_OP
