//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <array>
#include <string_view>
#include <ast/function_builder.h>

namespace luisa::compute::dsl::detail {

template<typename T>
struct Expr;

template<typename T>
class ExprBase {

public:
    using ValueType = T;

protected:
    const Expression *_expression;

public:
    explicit ExprBase(const Expression *expr) noexcept : _expression{expr} {}

    template<concepts::NonPointer U>// to prevent conversion from pointer to bool
    requires concepts::Constructible<T, U>
    ExprBase(U literal) noexcept : ExprBase{FunctionBuilder::current()->literal(literal)} {}

    constexpr ExprBase(ExprBase &&) noexcept = default;
    constexpr ExprBase(const ExprBase &) noexcept = default;
    [[nodiscard]] constexpr auto expression() const noexcept { return _expression; }

#define LUISA_MAKE_EXPR_BINARY_OP(op, op_concept_name, op_tag_name)                                       \
    template<typename U>                                                                                  \
    requires concepts::op_concept_name<T, U> [[nodiscard]] auto operator op(Expr<U> rhs) const noexcept { \
        using R = std::remove_cvref_t<decltype(std::declval<T>() op std::declval<U>())>;                  \
        return Expr<R>{FunctionBuilder::current()->binary(                                                \
            Type::of<R>(),                                                                                \
            BinaryOp::op_tag_name, this->expression(), rhs.expression())};                                \
    }                                                                                                     \
    template<typename U>                                                                                  \
    [[nodiscard]] auto operator op(U &&rhs) const noexcept {                                              \
        return this->operator op(Expr{std::forward<U>(rhs)});                                             \
    }
    LUISA_MAKE_EXPR_BINARY_OP(+, Add, ADD)
    LUISA_MAKE_EXPR_BINARY_OP(-, Sub, SUB)
    LUISA_MAKE_EXPR_BINARY_OP(*, Mul, MUL)
    LUISA_MAKE_EXPR_BINARY_OP(/, Div, DIV)
    LUISA_MAKE_EXPR_BINARY_OP(%, Mod, MOD)
    LUISA_MAKE_EXPR_BINARY_OP(&, BitAnd, BIT_AND)
    LUISA_MAKE_EXPR_BINARY_OP(|, BitOr, BIT_OR)
    LUISA_MAKE_EXPR_BINARY_OP(^, BitXor, BIT_XOR)
    LUISA_MAKE_EXPR_BINARY_OP(<<, ShiftLeft, SHL)
    LUISA_MAKE_EXPR_BINARY_OP(>>, ShiftRight, SHR)
    LUISA_MAKE_EXPR_BINARY_OP(&&, And, AND)
    LUISA_MAKE_EXPR_BINARY_OP(||, Or, OR)
    LUISA_MAKE_EXPR_BINARY_OP(==, Equal, EQUAL)
    LUISA_MAKE_EXPR_BINARY_OP(!=, NotEqual, NOT_EQUAL)
    LUISA_MAKE_EXPR_BINARY_OP(<, Less, LESS)
    LUISA_MAKE_EXPR_BINARY_OP(<=, LessEqual, LESS_EQUAL)
    LUISA_MAKE_EXPR_BINARY_OP(>, Greater, GREATER)
    LUISA_MAKE_EXPR_BINARY_OP(>=, GreaterEqual, GREATER_EQUAL)
#undef LUISA_MAKE_EXPR_BINARY_OP

    template<typename U>
    requires concepts::Access<T, U> [[nodiscard]] auto operator[](Expr<U> index) const noexcept {
        using R = std::remove_cvref_t<decltype(std::declval<T>()[std::declval<U>()])>;
        return Expr<R>{FunctionBuilder::current()->access(
            Type::of<R>(),
            this->expression(), index.expression())};
    }

    template<typename U>
    [[nodiscard]] auto operator[](U &&index) const noexcept { return this->operator[](Expr{std::forward<U>(index)}); }

    void operator=(const ExprBase &rhs) &noexcept {
        FunctionBuilder::current()->assign(AssignOp::ASSIGN, this->expression(), rhs.expression());
    }

    void operator=(ExprBase &&rhs) &noexcept {
        FunctionBuilder::current()->assign(AssignOp::ASSIGN, this->expression(), rhs.expression());
    }

#define LUISA_MAKE_EXPR_ASSIGN_OP(op, op_concept_name, op_tag_name)                                      \
    template<typename U>                                                                                 \
    requires concepts::op_concept_name<T, U> void operator op(Expr<U> rhs) &noexcept {                   \
        FunctionBuilder::current()->assign(AssignOp::op_tag_name, this->expression(), rhs.expression()); \
    }                                                                                                    \
    template<typename U>                                                                                 \
    void operator op(U &&rhs) &noexcept {                                                                \
        return this->operator op(Expr{std::forward<U>(rhs)});                                            \
    }
    LUISA_MAKE_EXPR_ASSIGN_OP(=, Assign, ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(+=, AddAssign, ADD_ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(-=, SubAssign, SUB_ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(*=, MulAssign, MUL_ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(/=, DivAssign, DIV_ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(%=, ModAssign, MOD_ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(&=, BitAndAssign, BIT_AND_ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(|=, BitOrAssign, BIT_OR_ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(^=, BitXorAssign, BIT_XOR_ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(<<=, ShiftLeftAssign, SHL_ASSIGN)
    LUISA_MAKE_EXPR_ASSIGN_OP(>>=, ShiftRightAssign, SHR_ASSIGN)
#undef LUISA_MAKE_EXPR_ASSIGN_OP

    // casts
    template<typename Dest>
    requires concepts::StaticConvertible<T, Dest> [[nodiscard]] auto cast() const noexcept {
        return Expr<Dest>{FunctionBuilder::current()->cast(Type::of<Dest>(), CastOp::STATIC, _expression)};
    }

    template<typename Dest>
    requires concepts::BitwiseConvertible<T, Dest> [[nodiscard]] auto as() const noexcept {
        return Expr<Dest>{FunctionBuilder::current()->cast(Type::of<Dest>(), CastOp::BITWISE, _expression)};
    }
};

template<typename T>
struct Expr : public ExprBase<T> {
    using ExprBase<T>::ExprBase;
    Expr(Expr &&another) noexcept = default;
    Expr(const Expr &another) noexcept = default;
    void operator=(Expr &&rhs) noexcept { ExprBase<T>::operator=(rhs); }
    void operator=(const Expr &rhs) noexcept { ExprBase<T>::operator=(rhs); }
};

template<typename T>
struct Expr<Vector<T, 2>> : public ExprBase<Vector<T, 2>> {
    using ExprBase<Vector<T, 2>>::ExprBase;
    Expr(Expr &&another) noexcept = default;
    Expr(const Expr &another) noexcept = default;
    void operator=(Expr &&rhs) noexcept { ExprBase<Vector<T, 2>>::operator=(rhs); }
    void operator=(const Expr &rhs) noexcept { ExprBase<Vector<T, 2>>::operator=(rhs); }
    Expr<T> x{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 2>>::_expression, 0)};
    Expr<T> y{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 2>>::_expression, 1)};
};

template<typename T>
struct Expr<Vector<T, 3>> : public ExprBase<Vector<T, 3>> {
    using ExprBase<Vector<T, 3>>::ExprBase;
    Expr(Expr &&another) noexcept = default;
    Expr(const Expr &another) noexcept = default;
    void operator=(Expr &&rhs) noexcept { ExprBase<Vector<T, 3>>::operator=(rhs); }
    void operator=(const Expr &rhs) noexcept { ExprBase<Vector<T, 3>>::operator=(rhs); }
    Expr<T> x{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 3>>::_expression, 0)};
    Expr<T> y{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 3>>::_expression, 1)};
    Expr<T> z{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 3>>::_expression, 2)};
};

template<typename T>
struct Expr<Vector<T, 4>> : public ExprBase<Vector<T, 4>> {
    using ExprBase<Vector<T, 4>>::ExprBase;
    Expr(Expr &&another) noexcept = default;
    Expr(const Expr &another) noexcept = default;
    void operator=(Expr &&rhs) noexcept { ExprBase<Vector<T, 4>>::operator=(rhs); }
    void operator=(const Expr &rhs) noexcept { ExprBase<Vector<T, 4>>::operator=(rhs); }
    Expr<T> x{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 4>>::_expression, 0)};
    Expr<T> y{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 4>>::_expression, 1)};
    Expr<T> z{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 4>>::_expression, 2)};
    Expr<T> w{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 4>>::_expression, 3)};
};

// deduction guides
template<typename T>
Expr(Expr<T>) -> Expr<T>;

template<concepts::Basic T>
Expr(T) -> Expr<T>;

template<typename T>
[[nodiscard]] const Expression *extract_expression(T &&v) noexcept {
    Expr expr{std::forward<T>(v)};
    return expr.expression();
}

template<typename T>
using expr_value_t = typename std::remove_cvref_t<decltype(Expr{std::declval<T>()})>::ValueType;

}// namespace luisa::compute::dsl::detail

#define LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(op, op_concept, op_tag)                                 \
    template<luisa::concepts::op_concept T>                                                     \
    [[nodiscard]] inline auto operator op(luisa::compute::dsl::detail::Expr<T> expr) noexcept { \
        using R = std::remove_cvref_t<decltype(op std::declval<T>())>;                          \
        return luisa::compute::dsl::detail::Expr<R>{                                            \
            luisa::compute::FunctionBuilder::current()->unary(                                  \
                luisa::compute::Type::of<R>(),                                                  \
                luisa::compute::UnaryOp::op_tag,                                                \
                expr.expression())};                                                            \
    }
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(+, Plus, PLUS)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(-, Minus, MINUS)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(!, Not, NOT)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(~, BitNot, BIT_NOT)
#undef LUISA_MAKE_GLOBAL_EXPR_UNARY_OP

#define LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(op, op_concept)                        \
    template<luisa::concepts::Basic Lhs, typename Rhs>                          \
    requires luisa::concepts::op_concept<Lhs, Rhs> [[nodiscard]] inline auto    \
    operator op(Lhs lhs, luisa::compute::dsl::detail::Expr<Rhs> rhs) noexcept { \
        return luisa::compute::dsl::detail::Expr{lhs} op rhs;                   \
    }
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(+, Add)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(-, Sub)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(*, Mul)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(/, Div)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(%, Mod)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(&, BitAnd)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(|, BitOr)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(^, BitXor)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(<<, ShiftLeft)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(>>, ShiftRight)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(&&, And)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(||, Or)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(==, Equal)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(!=, NotEqual)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(<, Less)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(<=, LessEqual)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(>, Greater)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(>=, GreaterEqual)
#undef LUISA_MAKE_GLOBAL_EXPR_BINARY_OP

namespace luisa::compute::dsl {

template<typename Dest, typename Src>
[[nodiscard]] inline auto cast(detail::Expr<Src> s) noexcept { return s.template cast<Dest>(); }

template<typename Dest, typename Src>
[[nodiscard]] inline auto as(detail::Expr<Src> s) noexcept { return s.template as<Dest>(); }

template<typename Dest, typename Src>
[[nodiscard]] inline auto reinterpret(detail::Expr<Src> s) noexcept { return s.template reinterpret<Dest>(); }

[[nodiscard]] inline auto thread_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->thread_id()};
}

[[nodiscard]] inline auto block_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->block_id()};
}

[[nodiscard]] inline auto dispatch_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->dispatch_id()};
}

}// namespace luisa::compute::dsl
