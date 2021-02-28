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
    explicit constexpr ExprBase(const Expression *expr) noexcept : _expression{expr} {}
    ExprBase(T literal) noexcept : ExprBase{FunctionBuilder::current()->literal(literal)} {}
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
#define LUISA_MAKE_EXPR_BINARY_OP_FROM_TRIPLET(op) LUISA_MAKE_EXPR_BINARY_OP op
    LUISA_MAP(LUISA_MAKE_EXPR_BINARY_OP_FROM_TRIPLET,
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
#undef LUISA_MAKE_EXPR_BINARY_OP_FROM_TRIPLET

    template<typename U>
    requires concepts::Access<T, U> [[nodiscard]] auto operator[](Expr<U> index) const noexcept {
        using R = std::remove_cvref_t<decltype(std::declval<T>()[std::declval<U>()])>;
        return Expr<R>{FunctionBuilder::current()->access(
            Type::of<R>(),
            this->expression(), index.expression())};
    }

    template<typename U>
    [[nodiscard]] auto operator[](U &&index) const noexcept { return this->operator[](Expr{std::forward<U>(index)}); }

    void operator=(const ExprBase &rhs) const noexcept {
        FunctionBuilder::current()->assign(AssignOp::ASSIGN, this->expression(), rhs.expression());
    }

    void operator=(ExprBase &&rhs) const noexcept {
        FunctionBuilder::current()->assign(AssignOp::ASSIGN, this->expression(), rhs.expression());
    }

#define LUISA_MAKE_EXPR_ASSIGN_OP(op, op_concept_name, op_tag_name)                                      \
    template<typename U>                                                                                 \
    requires concepts::op_concept_name<T, U> void operator op(Expr<U> rhs) const noexcept {              \
        FunctionBuilder::current()->assign(AssignOp::op_tag_name, this->expression(), rhs.expression()); \
    }                                                                                                    \
    template<typename U>                                                                                 \
    void operator op(U &&rhs) const noexcept {                                                           \
        return this->operator op(Expr{std::forward<U>(rhs)});                                            \
    }
#define LUISA_MAKE_EXPR_ASSIGN_OP_FROM_TRIPLET(op) LUISA_MAKE_EXPR_ASSIGN_OP op
    LUISA_MAP(LUISA_MAKE_EXPR_ASSIGN_OP_FROM_TRIPLET,
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
#undef LUISA_MAKE_EXPR_ASSIGN_OP_FROM_TRIPLET
};

template<typename T>
struct Expr : public ExprBase<T> {
    using ExprBase<T>::ExprBase;
    Expr(Expr &&another) noexcept = default;
    Expr(const Expr &another) noexcept = default;
    void operator=(Expr &&rhs) const noexcept { ExprBase<T>::operator=(rhs); }
    void operator=(const Expr &rhs) const noexcept { ExprBase<T>::operator=(rhs); }
};

template<typename T>
struct Expr<Vector<T, 2>> : public ExprBase<Vector<T, 2>> {
    using ExprBase<Vector<T, 2>>::ExprBase;
    Expr(Expr &&another) noexcept = default;
    Expr(const Expr &another) noexcept = default;
    void operator=(Expr &&rhs) const noexcept { ExprBase<Vector<T, 2>>::operator=(rhs); }
    void operator=(const Expr &rhs) const noexcept { ExprBase<Vector<T, 2>>::operator=(rhs); }
    Expr<T> x{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 2>>::_expression, 0)};
    Expr<T> y{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 2>>::_expression, 1)};
};

template<typename T>
struct Expr<Vector<T, 3>> : public ExprBase<Vector<T, 3>> {
    using ExprBase<Vector<T, 3>>::ExprBase;
    Expr(Expr &&another) noexcept = default;
    Expr(const Expr &another) noexcept = default;
    void operator=(Expr &&rhs) const noexcept { ExprBase<Vector<T, 3>>::operator=(rhs); }
    void operator=(const Expr &rhs) const noexcept { ExprBase<Vector<T, 3>>::operator=(rhs); }
    Expr<T> x{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 3>>::_expression, 0)};
    Expr<T> y{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 3>>::_expression, 1)};
    Expr<T> z{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 3>>::_expression, 2)};
};

template<typename T>
struct Expr<Vector<T, 4>> : public ExprBase<Vector<T, 4>> {
    using ExprBase<Vector<T, 4>>::ExprBase;
    Expr(Expr &&another) noexcept = default;
    Expr(const Expr &another) noexcept = default;
    void operator=(Expr &&rhs) const noexcept { ExprBase<Vector<T, 4>>::operator=(rhs); }
    void operator=(const Expr &rhs) const noexcept { ExprBase<Vector<T, 4>>::operator=(rhs); }
    Expr<T> x{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 4>>::_expression, 0)};
    Expr<T> y{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 4>>::_expression, 1)};
    Expr<T> z{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 4>>::_expression, 2)};
    Expr<T> w{FunctionBuilder::current()->member(Type::of<T>(), ExprBase<Vector<T, 4>>::_expression, 3)};
};

// deduction guides
template<typename T>
Expr(Expr<T>) -> Expr<T>;

template<concepts::Native T>
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
    template<luisa::concepts::Native Lhs, typename Rhs>                         \
    requires luisa::concepts::op_concept<Lhs, Rhs> [[nodiscard]] inline auto    \
    operator op(Lhs lhs, luisa::compute::dsl::detail::Expr<Rhs> rhs) noexcept { \
        return luisa::compute::dsl::detail::Expr{lhs} op rhs;                   \
    }
#define LUISA_MAKE_GLOBAL_EXPR_BINARY_OP_FROM_PAIR(op) LUISA_MAKE_GLOBAL_EXPR_BINARY_OP op
LUISA_MAP(LUISA_MAKE_GLOBAL_EXPR_BINARY_OP_FROM_PAIR,
          (+, Add),
          (-, Sub),
          (*, Mul),
          (/, Div),
          (%, Mod),
          (&, BitAnd),
          (|, BitOr),
          (^, BitXor),
          (<<, ShiftLeft),
          (>>, ShiftRight),
          (&&, And),
          (||, Or),
          (==, Equal),
          (!=, NotEqual),
          (<, Less),
          (<=, LessEqual),
          (>, Greater),
          (>=, GreaterEqual))
#undef LUISA_MAKE_GLOBAL_EXPR_BINARY_OP
#undef LUISA_MAKE_GLOBAL_EXPR_BINARY_OP_FROM_PAIR

// for custom structs
#undef LUISA_STRUCT// to extend it...

#define LUISA_STRUCT_MAKE_MEMBER_EXPR(m)                                    \
private:                                                                    \
    using Type_##m = std::remove_cvref_t<decltype(std::declval<This>().m)>; \
                                                                            \
public:                                                                     \
    Expr<Type_##m> m{FunctionBuilder::current()->member(                    \
        Type::of<Type_##m>(),                                               \
        ExprBase<This>::_expression,                                        \
        _member_index(#m))};

#define LUISA_STRUCT(S, ...)                                                                                     \
    LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(S, __VA_ARGS__)                                                \
    namespace luisa::compute::dsl::detail {                                                                      \
    template<>                                                                                                   \
    struct Expr<S> : public ExprBase<S> {                                                                        \
    private:                                                                                                     \
        using This = S;                                                                                          \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept {                    \
            constexpr const std::string_view member_names[]{LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};       \
            return std::find(std::begin(member_names), std::end(member_names), name) - std::begin(member_names); \
        }                                                                                                        \
                                                                                                                 \
    public:                                                                                                      \
        using ExprBase<S>::ExprBase;                                                                             \
        Expr(Expr &&another) noexcept = default;                                                                 \
        Expr(const Expr &another) noexcept = default;                                                            \
        void operator=(Expr &&rhs) const noexcept { ExprBase<S>::operator=(rhs); }                               \
        void operator=(const Expr &rhs) const noexcept { ExprBase<S>::operator=(rhs); }                          \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_EXPR, __VA_ARGS__)                                                    \
    };                                                                                                           \
    }
