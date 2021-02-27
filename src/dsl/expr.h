//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <ast/function_builder.h>

namespace luisa::compute::dsl {

template<typename T>
class Expr;

template<typename T>
class Var;

namespace detail {

template<typename T>
class ExprBase {

private:
    const Expression *_expression;

public:
    explicit constexpr ExprBase(const Expression *expr) noexcept : _expression{expr} {}
    constexpr ExprBase(const Var<T> &variable) noexcept
        : _expression{FunctionBuilder::current()->value(variable.variable())} {}
    constexpr ExprBase(ExprBase &&) noexcept = default;
    constexpr ExprBase(const ExprBase &) noexcept = default;
    [[nodiscard]] constexpr auto expression() const noexcept { return _expression; }

#define LUISA_MAKE_EXPR_BINARY_OP(op, op_concept_name, op_tag_name)                                                  \
    template<typename U>                                                                                             \
    requires concepts::operator_##op_concept_name<T, U> [[nodiscard]] auto operator op(Expr<U> rhs) const noexcept { \
        return Expr{FunctionBuilder::current()->binary(                                                              \
            Type::of(std::declval<T>() op std::declval<U>()),                                                        \
            BinaryOp::op_tag_name, this->expression(), rhs.expression())};                                           \
    }
#define LUISA_MAKE_EXPR_BINARY_OP_FROM_TUPLE(op) LUISA_MAKE_EXPR_BINARY_OP op
    LUISA_MAP(LUISA_MAKE_EXPR_BINARY_OP_FROM_TUPLE,
              (+, add, ADD),
              (-, sub, SUB),
              (*, mul, MUL),
              (/, div, DIV),
              (%, mod, MOD),
              (&, bit_and, BIT_AND),
              (|, bit_or, BIT_OR),
              (^, bit_xor, BIT_XOR),
              (>>, shr, SHR),
              (<<, shl, SHL),
              (&&, and, AND),
              (||, or, OR),
              (==, equal, EQUAL),
              (!=, not_equal, NOT_EQUAL),
              (<, less, LESS),
              (<=, less_equal, LESS_EQUAL),
              (>, greater, GREATER),
              (>=, greater_equal, GREATER_EQUAL))
#undef LUISA_MAKE_EXPR_BINARY_OP
#undef LUISA_MAKE_EXPR_BINARY_OP_FROM_TUPLE

    template<typename U>
    requires concepts::operator_access<T, U> [[nodiscard]] auto operator[](Expr<U> index) const noexcept {
        return Expr{FunctionBuilder::current()->access(
            Type::of(std::declval<T>()[std::declval<U>()]),
            this->expression(), index.expression())};
    }

    void operator=(const ExprBase &rhs) const noexcept {
        FunctionBuilder::current()->assign(AssignOp::ASSIGN, this->expression(), rhs.expression());
    }

    void operator=(ExprBase &&rhs) const noexcept {
        FunctionBuilder::current()->assign(AssignOp::ASSIGN, this->expression(), rhs.expression());
    }

#define LUISA_MAKE_EXPR_ASSIGN_OP(op, op_concept_name, op_tag_name)                                      \
    template<typename U>                                                                                 \
    requires concepts::operator_##op_concept_name<T, U> void operator op(Expr<U> rhs) const noexcept {   \
        FunctionBuilder::current()->assign(AssignOp::op_tag_name, this->expression(), rhs.expression()); \
    }
#define LUISA_MAKE_EXPR_ASSIGN_OP_FROM_TUPLE(op) LUISA_MAKE_EXPR_ASSIGN_OP op
    LUISA_MAP(LUISA_MAKE_EXPR_ASSIGN_OP_FROM_TUPLE,
              (=, assign, ASSIGN),
              (+=, add_assign, ADD_ASSIGN),
              (-=, sub_assign, SUB_ASSIGN),
              (*=, mul_assign, MUL_ASSIGN),
              (/=, div_assign, DIV_ASSIGN),
              (%=, mod_assign, MOD_ASSIGN),
              (&=, bit_and_assign, BIT_AND_ASSIGN),
              (|=, bit_or_assign, BIT_OR_ASSIGN),
              (^=, bit_xor_assign, BIT_XOR_ASSIGN),
              (>>=, shr_assign, SHR_ASSIGN),
              (<<=, shl_assign, SHL_ASSIGN))
#undef LUISA_MAKE_EXPR_ASSIGN_OP
#undef LUISA_MAKE_EXPR_ASSIGN_OP_FROM_TUPLE
};

}// namespace detail

template<typename T>
class Expr : public detail::ExprBase<T> {

    using detail::ExprBase<T>::ExprBase;
    using detail::ExprBase<T>::operator=;
};

// deduction guides
template<typename T>
Expr(Expr<T>) -> Expr<T>;

template<typename T>
Expr(const Var<T> &) -> Expr<T>;

template<typename T>
Expr(Var<T> &&) -> Expr<T>;

template<concepts::core_data_type T>
Expr(T &&) -> Expr<std::remove_cvref_t<T>>;

}// namespace luisa::compute::dsl
