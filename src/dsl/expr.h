//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <array>
#include <string_view>

#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/buffer.h>
#include <runtime/heap.h>
#include <ast/function_builder.h>

namespace luisa::compute {

template<typename T>
struct Expr;

template<typename T>
struct Ref;

template<typename T>
struct Var;

namespace detail {

template<typename T>
[[nodiscard]] inline const Expression *extract_expression(T &&v) noexcept;

template<typename T>
struct expr_value_impl {
    using type = T;
};

template<typename T>
struct expr_value_impl<Expr<T>> {
    using type = T;
};

template<typename T>
struct expr_value_impl<Ref<T>> {
    using type = T;
};

template<typename T>
struct expr_value_impl<Var<T>> {
    using type = T;
};

}// namespace detail

template<typename T>
using expr_value = detail::expr_value_impl<std::remove_cvref_t<T>>;

template<typename T>
using expr_value_t = typename expr_value<T>::type;

template<typename T>
using vector_expr_element = vector_element<expr_value_t<T>>;

template<typename T>
using vector_expr_element_t = typename vector_expr_element<T>::type;

template<typename T>
using vector_expr_dimension = vector_dimension<expr_value_t<T>>;

template<typename T>
constexpr auto vector_expr_dimension_v = vector_expr_dimension<T>::value;

template<typename... T>
using is_vector_expr_same_dimension = is_vector_same_dimension<expr_value_t<T>...>;

template<typename... T>
constexpr auto is_vector_expr_same_dimension_v = is_vector_expr_same_dimension<T...>::value;

template<typename... T>
using is_vector_expr_same_element = concepts::is_same<vector_expr_element_t<T>...>;

template<typename... T>
constexpr auto is_vector_expr_same_element_v = is_vector_expr_same_element<T...>::value;

namespace detail {
template<typename T>
struct is_dsl_impl : std::false_type {};

template<typename T>
struct is_dsl_impl<Expr<T>> : std::true_type {};

template<typename T>
struct is_dsl_impl<Ref<T>> : std::true_type {};

template<typename T>
struct is_dsl_impl<Var<T>> : std::true_type {};

}// namespace detail

template<typename T>
using is_dsl = typename detail::is_dsl_impl<std::remove_cvref_t<T>>::type;

template<typename T>
constexpr auto is_dsl_v = is_dsl<T>::value;

template<typename... T>
using any_dsl = std::disjunction<is_dsl<T>...>;

template<typename... T>
constexpr auto any_dsl_v = any_dsl<T...>::value;

template<typename... T>
using is_same_expr = concepts::is_same<expr_value_t<T>...>;

template<typename... T>
constexpr auto is_same_expr_v = is_same_expr<T...>::value;

template<typename T>
using is_integral_expr = is_integral<expr_value_t<T>>;

template<typename T>
constexpr auto is_integral_expr_v = is_integral_expr<T>::value;

template<typename T>
using is_floating_point_expr = is_floating_point<expr_value_t<T>>;

template<typename T>
constexpr auto is_floating_point_expr_v = is_floating_point_expr<T>::value;

template<typename T>
using is_scalar_expr = is_scalar<expr_value_t<T>>;

template<typename T>
constexpr auto is_scalar_expr_v = is_scalar_expr<T>::value;

template<typename T>
using is_vector_expr = is_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector_expr_v = is_vector_expr<T>::value;

template<typename T>
using is_vector2_expr = is_vector2<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector2_expr_v = is_vector2_expr<T>::value;

template<typename T>
using is_vector3_expr = is_vector3<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector3_expr_v = is_vector3_expr<T>::value;

template<typename T>
using is_vector4_expr = is_vector4<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector4_expr_v = is_vector4_expr<T>::value;

template<typename T>
using is_bool_vector_expr = is_bool_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_bool_vector_expr_v = is_bool_vector_expr<T>::value;

template<typename T>
using is_float_vector_expr = is_float_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_float_vector_expr_v = is_float_vector_expr<T>::value;

template<typename T>
using is_int_vector_expr = is_int_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_int_vector_expr_v = is_int_vector_expr<T>::value;

template<typename T>
using is_uint_vector_expr = is_uint_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_uint_vector_expr_v = is_uint_vector_expr<T>::value;

template<typename T>
using is_matrix_expr = is_matrix<expr_value_t<T>>;

template<typename T>
constexpr auto is_matrix_expr_v = is_matrix_expr<T>::value;

template<typename T>
using is_matrix2_expr = is_matrix2<expr_value_t<T>>;

template<typename T>
constexpr auto is_matrix2_expr_v = is_matrix2_expr<T>::value;

template<typename T>
using is_matrix3_expr = is_matrix3<expr_value_t<T>>;

template<typename T>
constexpr auto is_matrix3_expr_v = is_matrix3_expr<T>::value;

template<typename T>
using is_matrix4_expr = is_matrix4<expr_value_t<T>>;

template<typename T>
constexpr auto is_matrix4_expr_v = is_matrix4_expr<T>::value;

template<typename T>
using is_float_or_vector_expr = std::disjunction<
    is_floating_point_expr<T>,
    is_float_vector_expr<T>>;

template<typename T>
constexpr auto is_float_or_vector_expr_v = is_float_or_vector_expr<T>::value;

template<typename T>
using is_int_or_vector_expr = std::disjunction<
    std::is_same<expr_value_t<T>, int>,
    is_int_vector_expr<T>>;

template<typename T>
constexpr auto is_int_or_vector_expr_v = is_int_or_vector_expr<T>::value;

template<typename T>
using is_bool_or_vector_expr = std::disjunction<
    std::is_same<expr_value_t<T>, bool>,
    is_bool_vector_expr<T>>;

template<typename T>
constexpr auto is_bool_or_vector_expr_v = is_bool_or_vector_expr<T>::value;

template<typename T>
using is_uint_or_vector_expr = std::disjunction<
    std::is_same<expr_value_t<T>, uint>,
    is_uint_vector_expr<T>>;

template<typename T>
constexpr auto is_uint_or_vector_expr_v = is_uint_or_vector_expr<T>::value;

#define LUISA_EXPR_COMMON(...)                                                   \
private:                                                                         \
    const Expression *_expression;                                               \
                                                                                 \
public:                                                                          \
    explicit Expr(const Expression *expr) noexcept : _expression{expr} {}        \
    [[nodiscard]] auto expression() const noexcept { return this->_expression; } \
    Expr(Expr &&another) noexcept = default;                                     \
    Expr(const Expr &another) noexcept = default;                                \
    Expr &operator=(Expr &&) noexcept = delete;                                  \
    Expr &operator=(const Expr &) noexcept = delete;

#define LUISA_REF_COMMON(...)                                              \
private:                                                                   \
    const Expression *_expression;                                         \
                                                                           \
public:                                                                    \
    explicit Ref(const Expression *e) noexcept : _expression{e} {}         \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
    Ref(Ref &&) noexcept = default;                                        \
    Ref(const Ref &) noexcept = default;                                   \
    void operator=(Expr<__VA_ARGS__> rhs) const noexcept {                 \
        detail::FunctionBuilder::current()->assign(                        \
            AssignOp::ASSIGN,                                              \
            this->expression(),                                            \
            rhs.expression());                                             \
    }                                                                      \
    [[nodiscard]] operator Expr<__VA_ARGS__>() const noexcept {            \
        return Expr<__VA_ARGS__>{this->expression()};                      \
    }                                                                      \
    void operator=(Ref &&rhs) const noexcept {                             \
        (*this) = Expr<__VA_ARGS__>{rhs};                                  \
    }                                                                      \
    void operator=(const Ref &rhs) const noexcept {                        \
        (*this) = Expr<__VA_ARGS__>{rhs};                                  \
    }                                                                      \
    template<typename Value>                                               \
    void assign(Value &&v) const noexcept {                                \
        (*this) = std::forward<Value>(v);                                  \
    }

#define LUISA_EXPR_FROM_LITERAL(...)                             \
    template<typename U>                                         \
    requires std::same_as<U, __VA_ARGS__>                        \
    Expr(U literal)                                              \
    noexcept : Expr{detail::FunctionBuilder::current()->literal( \
        Type::of<U>(), literal)} {}

namespace detail {

template<typename T>
struct ExprEnableStaticCast;

template<typename T>
struct ExprEnableBitwiseCast;

template<typename T>
struct ExprEnableAccessOp;

template<typename T>
struct RefEnableStaticCast;

template<typename T>
struct RefEnableBitwiseCast;

template<typename T>
struct RefEnableAccessOp;

template<typename T>
struct RefEnableArithmeticAssign;

}// namespace detail

template<typename T>
struct Expr
    : detail::ExprEnableStaticCast<T>,
      detail::ExprEnableBitwiseCast<T> {
    static_assert(concepts::basic<T>);
    LUISA_EXPR_COMMON(T)
    LUISA_EXPR_FROM_LITERAL(T)
};

template<typename T, size_t N>
struct Expr<std::array<T, N>>
    : detail::ExprEnableAccessOp<std::array<T, N>> {
    LUISA_EXPR_COMMON(std::array<T, N>)
};

template<size_t N>
struct Expr<Matrix<N>>
    : detail::ExprEnableAccessOp<Matrix<N>> {
    LUISA_EXPR_COMMON(Matrix<N>)
    LUISA_EXPR_FROM_LITERAL(Matrix<N>)
};

template<typename... T>
struct Expr<std::tuple<T...>> {
    LUISA_EXPR_COMMON(std::tuple<T...>)
    template<size_t i>
    [[nodiscard]] auto member() const noexcept {
        using M = std::tuple_element_t<i, std::tuple<T...>>;
        return Expr<M>{detail::FunctionBuilder::current()->member(
            Type::of<M>(), this->expression(), i)};
    };
};

template<typename T>
struct Expr<Vector<T, 2>>
    : detail::ExprEnableStaticCast<Vector<T, 2>>,
      detail::ExprEnableBitwiseCast<Vector<T, 2>>,
      detail::ExprEnableAccessOp<Vector<T, 2>> {
    LUISA_EXPR_COMMON(Vector<T, 2>)
    LUISA_EXPR_FROM_LITERAL(Vector<T, 2>)
    Expr<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
#include <dsl/swizzle_2.inl.h>
};

template<typename T>
struct Expr<Vector<T, 3>>
    : detail::ExprEnableStaticCast<Vector<T, 3>>,
      detail::ExprEnableBitwiseCast<Vector<T, 3>>,
      detail::ExprEnableAccessOp<Vector<T, 3>> {
    LUISA_EXPR_COMMON(Vector<T, 3>)
    LUISA_EXPR_FROM_LITERAL(Vector<T, 3>)
    Expr<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Expr<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
#include <dsl/swizzle_3.inl.h>
};

template<typename T>
struct Expr<Vector<T, 4>>
    : detail::ExprEnableStaticCast<Vector<T, 4>>,
      detail::ExprEnableBitwiseCast<Vector<T, 4>>,
      detail::ExprEnableAccessOp<Vector<T, 4>> {
    LUISA_EXPR_COMMON(Vector<T, 4>)
    LUISA_EXPR_FROM_LITERAL(Vector<T, 4>)
    Expr<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Expr<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
    Expr<T> w{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x3u)};
#include <dsl/swizzle_4.inl.h>
};

template<typename T>
struct Ref
    : detail::RefEnableStaticCast<T>,
      detail::RefEnableBitwiseCast<T>,
      detail::RefEnableArithmeticAssign<T> {
    static_assert(concepts::basic<T>);
    LUISA_REF_COMMON(T)
};

template<typename T, size_t N>
struct Ref<std::array<T, N>>
    : detail::RefEnableAccessOp<std::array<T, N>>,
      detail::RefEnableArithmeticAssign<std::array<T, N>> {
    LUISA_REF_COMMON(std::array<T, N>)
};

template<size_t N>
struct Ref<Matrix<N>>
    : detail::RefEnableAccessOp<Matrix<N>>,
      detail::RefEnableArithmeticAssign<Matrix<N>> {
    LUISA_REF_COMMON(Matrix<N>)
};

template<typename... T>
struct Ref<std::tuple<T...>> {
    LUISA_REF_COMMON(std::tuple<T...>)
    template<size_t i>
    [[nodiscard]] auto member() const noexcept {
        using M = std::tuple_element_t<i, std::tuple<T...>>;
        return Ref<M>{detail::FunctionBuilder::current()->member(
            Type::of<M>(), this->expression(), i)};
    };
};

template<typename T>
struct Ref<Vector<T, 2>>
    : detail::RefEnableStaticCast<Vector<T, 2>>,
      detail::RefEnableBitwiseCast<Vector<T, 2>>,
      detail::RefEnableAccessOp<Vector<T, 2>>,
      detail::RefEnableArithmeticAssign<Vector<T, 2>> {
    LUISA_REF_COMMON(Vector<T, 2>)
    Ref<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Ref<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
#include <dsl/swizzle_2.inl.h>
};

template<typename T>
struct Ref<Vector<T, 3>>
    : detail::RefEnableStaticCast<Vector<T, 3>>,
      detail::RefEnableBitwiseCast<Vector<T, 3>>,
      detail::RefEnableAccessOp<Vector<T, 3>>,
      detail::RefEnableArithmeticAssign<Vector<T, 3>> {
    LUISA_REF_COMMON(Vector<T, 3>)
    Ref<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Ref<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Ref<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
#include <dsl/swizzle_3.inl.h>
};

template<typename T>
struct Ref<Vector<T, 4>>
    : detail::RefEnableStaticCast<Vector<T, 4>>,
      detail::RefEnableBitwiseCast<Vector<T, 4>>,
      detail::RefEnableAccessOp<Vector<T, 4>>,
      detail::RefEnableArithmeticAssign<Vector<T, 4>> {
    LUISA_REF_COMMON(Vector<T, 4>)
    Ref<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Ref<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Ref<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
    Ref<T> w{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x3u)};
#include <dsl/swizzle_4.inl.h>
};

#undef LUISA_REF_COMMON
#undef LUISA_EXPR_COMMON
#undef LUISA_EXPR_FROM_LITERAL

namespace detail {

template<typename T>
struct ExprEnableStaticCast {
    template<typename Dest>
    requires concepts::static_convertible<T, expr_value_t<Dest>>
    [[nodiscard]] auto cast() const noexcept {
        using TrueDest = expr_value_t<Dest>;
        return Expr<TrueDest>{FunctionBuilder::current()->cast(
            Type::of<TrueDest>(),
            CastOp::STATIC,
            static_cast<const Expr<T> *>(this)->expression())};
    }
};

template<typename T>
struct ExprEnableBitwiseCast {
    template<typename Dest>
    requires concepts::bitwise_convertible<T, expr_value_t<Dest>>
    [[nodiscard]] auto as() const noexcept {
        using TrueDest = expr_value_t<Dest>;
        return Expr<TrueDest>{FunctionBuilder::current()->cast(
            Type::of<TrueDest>(),
            CastOp::BITWISE,
            static_cast<const Expr<T> *>(this)->expression())};
    }
};

template<typename T>
struct ExprEnableAccessOp {
    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        using Elem = std::remove_cvref_t<decltype(std::declval<T>()[0])>;
        return Expr<Elem>{FunctionBuilder::current()->access(
            Type::of<Elem>(),
            static_cast<const Expr<T> *>(this)->expression(),
            extract_expression(std::forward<I>(index)))};
    }
};

template<typename T>
struct RefEnableArithmeticAssign {
#define LUISA_EXPR_ASSIGN_OP(op, op_concept_name, op_tag_name) \
    template<typename U>                                       \
    requires concepts::op_concept_name<                        \
        T, expr_value_t<U>>                                    \
    void operator op(U &&rhs) noexcept {                       \
        FunctionBuilder::current()->assign(                    \
            AssignOp::op_tag_name,                             \
            static_cast<const Ref<T> *>(this)->expression(),   \
            extract_expression(std::forward<U>(rhs)));         \
    }
    LUISA_EXPR_ASSIGN_OP(+=, add_assignable, ADD_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(-=, sub_assignable, SUB_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(*=, mul_assignable, MUL_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(/=, div_assignable, DIV_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(%=, mod_assignable, MOD_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(&=, bit_and_assignable, BIT_AND_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(|=, bit_or_assignable, BIT_OR_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(^=, bit_xor_assignable, BIT_XOR_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(<<=, shift_left_assignable, SHL_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(>>=, shift_right_assignable, SHR_ASSIGN)
#undef LUISA_EXPR_ASSIGN_OP
};

template<typename T>
struct RefEnableStaticCast {
    template<typename Dest>
    requires concepts::static_convertible<T, expr_value_t<Dest>>
    [[nodiscard]] auto cast() const noexcept {
        using TrueDest = expr_value_t<Dest>;
        return Expr<TrueDest>{FunctionBuilder::current()->cast(
            Type::of<TrueDest>(),
            CastOp::STATIC,
            static_cast<const Ref<T> *>(this)->expression())};
    }
};

template<typename T>
struct RefEnableBitwiseCast {
    template<typename Dest>
    requires concepts::bitwise_convertible<T, expr_value_t<Dest>>
    [[nodiscard]] auto as() const noexcept {
        using TrueDest = expr_value_t<Dest>;
        return Expr<TrueDest>{FunctionBuilder::current()->cast(
            Type::of<TrueDest>(),
            CastOp::BITWISE,
            static_cast<const Ref<T> *>(this)->expression())};
    }
};

template<typename T>
struct RefEnableAccessOp {
    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        using Elem = std::remove_cvref_t<decltype(std::declval<T>()[0])>;
        return Ref<Elem>{FunctionBuilder::current()->access(
            Type::of<Elem>(),
            static_cast<const Ref<T> *>(this)->expression(),
            extract_expression(std::forward<I>(index)))};
    }
};

template<typename>
struct BufferExprAsAtomic {};

}// namespace detail

template<typename T>
struct Expr<Buffer<T>>
    : detail::BufferExprAsAtomic<T> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}
    explicit Expr(BufferView<T> buffer) noexcept
        : _expression{detail::FunctionBuilder::current()->buffer_binding(
            Type::of<Buffer<T>>(),
            buffer.handle(), buffer.offset_bytes())} {}

    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto operator[](I &&i) const noexcept {
        return Ref<T>{detail::FunctionBuilder::current()->access(
            Type::of<T>(), _expression,
            detail::extract_expression(std::forward<I>(i)))};
    };
};

template<typename T>
struct Expr<BufferView<T>> : public Expr<Buffer<T>> {
    using Expr<Buffer<T>>::Expr;
};

template<typename T>
class AtomicRef {

private:
    const AccessExpr *_expression{nullptr};

public:
    explicit AtomicRef(const AccessExpr *expr) noexcept
        : _expression{expr} {}

    void store(Expr<T> value) const noexcept {
        detail::FunctionBuilder::current()->call(CallOp::ATOMIC_STORE, {this->_expression, value.expression()});
    }

#define LUISA_ATOMIC_NODISCARD                                           \
    [[nodiscard(                                                         \
        "Return values from atomic operations with side effects should " \
        "not be discarded. Enclose this expression with void_().")]]

    LUISA_ATOMIC_NODISCARD auto load() const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_LOAD,
            {this->_expression});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto exchange(Expr<T> desired) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_EXCHANGE,
            {this->_expression, desired.expression()});
        return Expr<T>{expr};
    }

    // stores old == compare ? val : old, returns old
    LUISA_ATOMIC_NODISCARD auto compare_exchange(Expr<T> expected, Expr<T> desired) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_COMPARE_EXCHANGE,
            {this->_expression, expected.expression(), desired.expression()});
        return Expr<T>{expr};
    }

    LUISA_ATOMIC_NODISCARD auto fetch_add(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_ADD,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_sub(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_SUB,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_and(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_AND,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_or(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_OR,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_xor(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_XOR,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_min(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MIN,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_max(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MAX,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

#undef LUISA_ATOMIC_NODISCARD
};

namespace detail {

template<>
struct BufferExprAsAtomic<int> {
    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        return AtomicRef<int>{FunctionBuilder::current()->access(
            Type::of<int>(),
            static_cast<const Expr<Buffer<int>> *>(this)->expression(),
            extract_expression(std::forward<I>(i)))};
    }
};

template<>
struct BufferExprAsAtomic<uint> {
    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        return AtomicRef<uint>{FunctionBuilder::current()->access(
            Type::of<uint>(),
            static_cast<const Expr<Buffer<uint>> *>(this)->expression(),
            extract_expression(std::forward<I>(i)))};
    }
};

}// namespace detail

template<typename T>
struct Expr<Image<T>> {

private:
    const RefExpr *_expression{nullptr};
    const Expression *_offset{nullptr};

    [[nodiscard]] auto _offset_uv(const Expression *uv) const noexcept -> const Expression * {
        if (_offset == nullptr) { return uv; }
        auto f = detail::FunctionBuilder::current();
        return f->binary(Type::of<uint2>(), BinaryOp::ADD, uv, _offset);
    }

public:
    explicit Expr(const RefExpr *expr, const Expression *offset) noexcept
        : _expression{expr}, _offset{offset} {}
    explicit Expr(ImageView<T> image) noexcept
        : _expression{detail::FunctionBuilder::current()->texture_binding(
            Type::of<Image<T>>(), image.handle())},
          _offset{any(image.offset())
                      ? detail::FunctionBuilder::current()->literal(Type::of<uint2>(), image.offset())
                      : nullptr} {}

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }

    [[nodiscard]] auto read(Expr<uint2> uv) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<Vector<T, 4>>{f->call(
            Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
            {_expression, _offset_uv(uv.expression())})};
    };

    void write(Expr<uint2> uv, Expr<Vector<T, 4>> value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::TEXTURE_WRITE,
            {_expression, _offset_uv(uv.expression()), value.expression()});
    }
};

template<typename T>
struct Expr<ImageView<T>> : public Expr<Image<T>> {
    using Expr<Image<T>>::Expr;
};

template<typename T>
struct Expr<Volume<T>> {

private:
    const RefExpr *_expression{nullptr};
    const Expression *_offset{nullptr};

    [[nodiscard]] auto _offset_uvw(const Expression *uvw) const noexcept -> const Expression * {
        if (_offset == nullptr) { return uvw; }
        auto f = detail::FunctionBuilder::current();
        return f->binary(Type::of<uint3>(), BinaryOp::ADD, uvw, _offset);
    }

public:
    explicit Expr(const RefExpr *expr, const Expression *offset) noexcept
        : _expression{expr}, _offset{offset} {}
    explicit Expr(VolumeView<T> volume) noexcept
        : _expression{detail::FunctionBuilder::current()->texture_binding(
            Type::of<Volume<T>>(), volume.handle())},
          _offset{any(volume.offset())
                      ? detail::FunctionBuilder::current()->literal(Type::of<uint3>(), volume.offset())
                      : nullptr} {}

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }

    [[nodiscard]] auto read(Expr<uint3> uvw) const noexcept {
        return Expr<Vector<T, 4>>{detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
            {_expression, _offset_uvw(uvw.expression())})};
    };

    void write(Expr<uint3> uvw, Expr<Vector<T, 4>> value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::TEXTURE_WRITE,
            {_expression, _offset_uvw(uvw.expression()), value.expression()});
    }
};

template<typename T>
struct Expr<VolumeView<T>> : public Expr<Volume<T>> {
    using Expr<Volume<T>>::Expr;
};

template<typename T>
class BufferRef {

private:
    const RefExpr *_heap{nullptr};
    const Expression *_index{nullptr};

public:
    BufferRef(const RefExpr *heap, const Expression *index) noexcept
        : _heap{heap},
          _index{index} {}

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&i) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<T>{f->call(
            Type::of<T>(), CallOp::BUFFER_HEAP_READ,
            {_heap, _index, detail::extract_expression(std::forward<I>(i))})};
    }
};

class TextureRef2D {

private:
    const RefExpr *_heap{nullptr};
    const Expression *_index{nullptr};

public:
    TextureRef2D(const RefExpr *heap, const Expression *index) noexcept
        : _heap{heap},
          _index{index} {}

    [[nodiscard]] auto sample(Expr<float2> uv) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D,
            {_heap, _index, uv.expression()})};
    }

    [[nodiscard]] auto sample(Expr<float2> uv, Expr<float> mip) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D_LEVEL,
            {_heap, _index, uv.expression(), mip.expression()})};
    }

    [[nodiscard]] auto sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D_GRAD,
            {_heap, _index, uv.expression(), dpdx.expression(), dpdy.expression()})};
    }

    [[nodiscard]] auto size() const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<uint2>{f->call(
            Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D,
            {_heap, _index})};
    }

    [[nodiscard]] auto size(Expr<int> level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<uint2>{f->call(
            Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D_LEVEL,
            {_heap, _index, level.expression()})};
    }

    [[nodiscard]] auto size(Expr<uint> level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<uint2>{f->call(
            Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D_LEVEL,
            {_heap, _index, level.expression()})};
    }

    [[nodiscard]] auto read(Expr<uint2> coord) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ2D,
            {_heap, _index, coord.expression()})};
    }

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint2> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ2D_LEVEL,
            {_heap, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))})};
    }
};

class TextureRef3D {

private:
    const RefExpr *_heap{nullptr};
    const Expression *_index{nullptr};

public:
    TextureRef3D(const RefExpr *heap, const Expression *index) noexcept
        : _heap{heap},
          _index{index} {}

    [[nodiscard]] auto sample(Expr<float3> uvw) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D,
            {_heap, _index, uvw.expression()})};
    }

    [[nodiscard]] auto sample(Expr<float3> uvw, Expr<float> mip) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D_LEVEL,
            {_heap, _index, uvw.expression(), mip.expression()})};
    }

    [[nodiscard]] auto sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D_GRAD,
            {_heap, _index, uvw.expression(), dpdx.expression(), dpdy.expression()})};
    }

    [[nodiscard]] auto size() const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<uint3>{f->call(
            Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D,
            {_heap, _index})};
    }

    [[nodiscard]] auto size(Expr<int> level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<uint3>{f->call(
            Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D_LEVEL,
            {_heap, _index, level.expression()})};
    }

    [[nodiscard]] auto size(Expr<uint> level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<uint3>{f->call(
            Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D_LEVEL,
            {_heap, _index, level.expression()})};
    }

    [[nodiscard]] auto read(Expr<uint3> coord) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ3D,
            {_heap, _index, coord.expression()})};
    }

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint3> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ3D_LEVEL,
            {_heap, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))})};
    }
};

template<>
struct Expr<Heap> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}

    explicit Expr(const Heap &heap) noexcept
        : _expression{detail::FunctionBuilder::current()->heap_binding(heap.handle())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto tex2d(I index) const noexcept {
        return TextureRef2D{_expression, detail::extract_expression(std::forward<I>(index))};
    }

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto tex3d(I &&index) const noexcept {
        return TextureRef3D{_expression, detail::extract_expression(std::forward<I>(index))};
    }

    template<typename T, typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto buffer(I &&index) const noexcept {
        return BufferRef<T>{_expression, detail::extract_expression(std::forward<I>(index))};
    }
};

// deduction guides
template<typename T>
Expr(Expr<T>) -> Expr<T>;

template<typename T>
Expr(Var<T>) -> Expr<T>;

template<typename T>
Expr(Ref<T>) -> Expr<T>;

template<concepts::basic T>
Expr(T) -> Expr<T>;

template<typename T>
Expr(const Buffer<T> &) -> Expr<Buffer<T>>;

template<typename T>
Expr(BufferView<T>) -> Expr<Buffer<T>>;

template<typename T>
Expr(const Image<T> &) -> Expr<Image<T>>;

template<typename T>
Expr(ImageView<T>) -> Expr<Image<T>>;

template<typename T>
Expr(const Volume<T> &) -> Expr<Volume<T>>;

template<typename T>
Expr(VolumeView<T>) -> Expr<Volume<T>>;

Expr(const Heap &) -> Expr<Heap>;

namespace detail {

template<typename T>
[[nodiscard]] inline const Expression *extract_expression(T &&v) noexcept {
    Expr expr{std::forward<T>(v)};
    return expr.expression();
}

}// namespace detail

template<typename I>
TextureRef2D Heap::tex2d(I &&index) const noexcept {
    return Expr<Heap>{*this}.tex2d(std::forward<I>(index));
}

template<typename I>
TextureRef2D Heap::tex3d(I &&index) const noexcept {
    return Expr<Heap>{*this}.tex3d(std::forward<I>(index));
}

template<typename T, typename I>
BufferRef<T> Heap::buffer(I &&index) const noexcept {
    return Expr<Heap>{*this}.buffer<T>(std::forward<I>(index));
}

}// namespace luisa::compute

#define LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(op, op_concept, op_tag)                      \
    template<typename T>                                                             \
    requires luisa::compute::is_dsl_v<T>                                             \
    [[nodiscard]] inline auto operator op(T &&expr) noexcept {                       \
        using R = std::remove_cvref_t<                                               \
            decltype(op std::declval<luisa::compute::expr_value_t<T>>())>;           \
        return luisa::compute::Expr<R>{                                              \
            luisa::compute::detail::FunctionBuilder::current()->unary(               \
                luisa::compute::Type::of<R>(),                                       \
                luisa::compute::UnaryOp::op_tag,                                     \
                luisa::compute::detail::extract_expression(std::forward<T>(expr)))}; \
    }
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(+, operator_plus, PLUS)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(-, operator_minus, MINUS)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(!, operator_not, NOT)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(~, operator_bit_not, BIT_NOT)
#undef LUISA_MAKE_GLOBAL_EXPR_UNARY_OP

#define LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(op, op_concept_name, op_tag_name)                         \
    template<typename Lhs, typename Rhs>                                                           \
    requires luisa::compute::any_dsl_v<Lhs, Rhs> && luisa::concepts::op_concept_name<              \
        luisa::compute::expr_value_t<Lhs>,                                                         \
        luisa::compute::expr_value_t<Rhs>>                                                         \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {                         \
        using R = std::remove_cvref_t<                                                             \
            decltype(std::declval<luisa::compute::expr_value_t<Lhs>>()                             \
                         op std::declval<luisa::compute::expr_value_t<Rhs>>())>;                   \
        return luisa::compute::Expr<R>{luisa::compute::detail::FunctionBuilder::current()->binary( \
            luisa::compute::Type::of<R>(),                                                         \
            luisa::compute::BinaryOp::op_tag_name,                                                 \
            luisa::compute::detail::extract_expression(std::forward<Lhs>(lhs)),                    \
            luisa::compute::detail::extract_expression(std::forward<Rhs>(rhs)))};                  \
    }
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(+, operator_add, ADD)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(-, operator_sub, SUB)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(*, operator_mul, MUL)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(/, operator_div, DIV)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(%, operator_mod, MOD)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(&, operator_bit_and, BIT_AND)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(|, operator_bit_or, BIT_OR)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(^, operator_bit_Xor, BIT_XOR)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(<<, operator_shift_left, SHL)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(>>, operator_shift_right, SHR)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(&&, operator_and, AND)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(||, operator_or, OR)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(==, operator_equal, EQUAL)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(!=, operator_not_equal, NOT_EQUAL)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(<, operator_less, LESS)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(<=, operator_less_equal, LESS_EQUAL)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(>, operator_greater, GREATER)
LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(>=, operator_greater_equal, GREATER_EQUAL)
#undef LUISA_MAKE_GLOBAL_EXPR_BINARY_OP
