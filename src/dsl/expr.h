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
        : _expression{FunctionBuilder::current()->ref(variable.variable())} {}
    constexpr ExprBase(ExprBase &&) noexcept = default;
    constexpr ExprBase(const ExprBase &) noexcept = default;
    [[nodiscard]] constexpr auto expression() const noexcept { return _expression; }

#define LUISA_MAKE_EXPR_BINARY_OP(op, op_concept_name, op_tag_name)                                                  \
    template<typename U>                                                                                             \
    requires concepts::op_concept_name<T, U> [[nodiscard]] auto operator op(Expr<U> rhs) const noexcept { \
        return Expr{FunctionBuilder::current()->binary(                                                              \
            Type::of(std::declval<T>() op std::declval<U>()),                                                        \
            BinaryOp::op_tag_name, this->expression(), rhs.expression())};                                           \
    }
#define LUISA_MAKE_EXPR_BINARY_OP_FROM_TUPLE(op) LUISA_MAKE_EXPR_BINARY_OP op
    LUISA_MAP(LUISA_MAKE_EXPR_BINARY_OP_FROM_TUPLE,
              (+, Add, ADD),
              (-, Sub, SUB),
              (*, Mul, MUL),
              (/, Div, DIV),
              (%, Mod, MOD),
              (&, BitAnd, BIT_AND),
              (|, BitOr, BIT_OR),
              (^, BitXor, BIT_XOR),
              (<<, ShiftLeft, SHL),
              (>>, ShiftRight, SHR),
              (&&, And, AND),
              (||, Or, OR),
              (==, Equal, EQUAL),
              (!=, NotEqual, NOT_EQUAL),
              (<, Less, LESS),
              (<=, LessEqual, LESS_EQUAL),
              (>, Greater, GREATER),
              (>=, GreaterEqual, GREATER_EQUAL))
#undef LUISA_MAKE_EXPR_BINARY_OP
#undef LUISA_MAKE_EXPR_BINARY_OP_FROM_TUPLE

    template<typename U>
    requires concepts::Access<T, U> [[nodiscard]] auto operator[](Expr<U> index) const noexcept {
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
    requires concepts::op_concept_name<T, U> void operator op(Expr<U> rhs) const noexcept {   \
        FunctionBuilder::current()->assign(AssignOp::op_tag_name, this->expression(), rhs.expression()); \
    }
#define LUISA_MAKE_EXPR_ASSIGN_OP_FROM_TUPLE(op) LUISA_MAKE_EXPR_ASSIGN_OP op
    LUISA_MAP(LUISA_MAKE_EXPR_ASSIGN_OP_FROM_TUPLE,
              (=, Assign, ASSIGN),
              (+=, AddAssign, ADD_ASSIGN),
              (-=, SubAssign, SUB_ASSIGN),
              (*=, MulAssign, MUL_ASSIGN),
              (/=, DivAssign, DIV_ASSIGN),
              (%=, ModAssign, MOD_ASSIGN),
              (&=, BitAndAssign, BIT_AND_ASSIGN),
              (|=, BitOrAssign, BIT_OR_ASSIGN),
              (^=, BitXorAssign, BIT_XOR_ASSIGN),
              (<<=, ShiftLeftAssign, SHL_ASSIGN),
              (>>=, ShiftRightAssign, SHR_ASSIGN))
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

template<concepts::Native T>
Expr(T &&) -> Expr<std::remove_cvref_t<T>>;

}// namespace luisa::compute::dsl
