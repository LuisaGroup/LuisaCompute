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
#include <dsl/expr_traits.h>
#include <dsl/arg.h>

namespace luisa::compute {

namespace detail {

template<typename T>
[[nodiscard]] inline const Expression *extract_expression(T &&v) noexcept;

template<typename T>
[[nodiscard]] inline Expr<T> make_var_expr(const Expression *expr) noexcept;// defined in dsl/var.h

}// namespace detail

inline namespace dsl {
template<typename Lhs, typename Rhs>
inline void assign(Lhs &&lhs, Rhs &&rhs) noexcept;// defined in dsl/func.h
}

#define LUISA_EXPR_COMMON(...)                                                   \
private:                                                                         \
    const Expression *_expression;                                               \
                                                                                 \
public:                                                                          \
    explicit Expr(const Expression *expr) noexcept : _expression{expr} {}        \
    [[nodiscard]] auto expression() const noexcept { return this->_expression; } \
    Expr(Expr &&another) noexcept = default;                                     \
    Expr(const Expr &another) noexcept = default;                                \
    Expr &operator=(Expr) noexcept = delete;

#define LUISA_REF_COMMON(...)                                              \
private:                                                                   \
    const Expression *_expression;                                         \
                                                                           \
public:                                                                    \
    explicit Ref(const Expression *e) noexcept : _expression{e} {}         \
    explicit Ref(detail::ArgumentCreation) noexcept                        \
        : Ref{detail::FunctionBuilder::current()                           \
                  ->reference(Type::of<__VA_ARGS__>())} {}                 \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
    Ref(Ref &&) noexcept = default;                                        \
    Ref(const Ref &) noexcept = default;                                   \
    template<typename Rhs>                                                 \
    void operator=(Rhs &&rhs) const noexcept {                             \
        dsl::assign(*this, std::forward<Rhs>(rhs));                        \
    }                                                                      \
    [[nodiscard]] operator Expr<__VA_ARGS__>() const noexcept {            \
        return Expr<__VA_ARGS__>{this->expression()};                      \
    }                                                                      \
    void operator=(Ref rhs) const noexcept {                               \
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

template<typename T, typename E>
struct expr_as {
    static_assert(always_false_v<T>);
};

template<typename T, typename E>
struct expr_as<Expr<T>, E> {
    using type = Expr<E>;
};

template<typename T, typename E>
struct expr_as<Ref<T>, E> {
    using type = Ref<E>;
};

template<typename T, typename E>
using expr_as_t = typename expr_as<T, E>::type;

template<typename T>
struct ExprEnableStaticCast;

template<typename T>
struct ExprEnableBitwiseCast;

template<typename T>
struct ExprEnableSubscriptAccess;

template<typename T>
struct ExprEnableArithmeticAssign;

template<typename T>
struct ExprEnableGetMemberByIndex;

template<typename... T, size_t... i>
[[nodiscard]] inline auto compose_impl(std::tuple<T...> t, std::index_sequence<i...>) noexcept {
    Var<std::tuple<expr_value_t<T>...>> v{Expr{std::get<i>(t)}...};
    return Expr{v};
}

template<typename T, size_t... i>
[[nodiscard]] inline auto decompose_impl(Expr<T> x, std::index_sequence<i...>) noexcept {
    return std::make_tuple(x.template get<i>()...);
}

}// namespace detail

inline namespace dsl {

template<typename... T>
[[nodiscard]] inline auto compose(std::tuple<T...> t) noexcept {
    return detail::compose_impl(t, std::index_sequence_for<T...>{});
}

template<typename... T>
[[nodiscard]] inline auto compose(T &&...v) noexcept {
    return compose(std::make_tuple(Expr{std::forward<T>(v)}...));
}

template<typename T>
[[nodiscard]] inline auto compose(T &&v) noexcept {
    return Expr{std::forward<T>(v)};
}

template<typename T>
[[nodiscard]] inline auto decompose(T &&t) noexcept {
    using member_tuple = struct_member_tuple_t<expr_value_t<T>>;
    using index_sequence = std::make_index_sequence<std::tuple_size_v<member_tuple>>;
    Var x{std::forward<T>(t)};// to avoid redundant evaluation
    return detail::decompose_impl(Expr{x}, index_sequence{});
}

}// namespace dsl

template<typename T>
struct Expr
    : detail::ExprEnableStaticCast<Expr<T>>,
      detail::ExprEnableBitwiseCast<Expr<T>> {
    static_assert(concepts::scalar<T>);
    LUISA_EXPR_COMMON(T)
    LUISA_EXPR_FROM_LITERAL(T)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i == 0u);
        return *this;
    }
};

template<typename T, size_t N>
struct Expr<std::array<T, N>>
    : detail::ExprEnableSubscriptAccess<Expr<std::array<T, N>>>,
      detail::ExprEnableGetMemberByIndex<Expr<std::array<T, N>>> {
    LUISA_EXPR_COMMON(std::array<T, N>)
};

template<size_t N>
struct Expr<Matrix<N>>
    : detail::ExprEnableSubscriptAccess<Expr<Matrix<N>>>,
      detail::ExprEnableGetMemberByIndex<Expr<Matrix<N>>> {
    LUISA_EXPR_COMMON(Matrix<N>)
    LUISA_EXPR_FROM_LITERAL(Matrix<N>)
};

template<typename... T>
struct Expr<std::tuple<T...>> {
    LUISA_EXPR_COMMON(std::tuple<T...>)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        using M = std::tuple_element_t<i, std::tuple<T...>>;
        return Expr<M>{detail::FunctionBuilder::current()->member(
            Type::of<M>(), this->expression(), i)};
    };
};

template<typename T>
struct Expr<Vector<T, 2>>
    : detail::ExprEnableStaticCast<Expr<Vector<T, 2>>>,
      detail::ExprEnableBitwiseCast<Expr<Vector<T, 2>>>,
      detail::ExprEnableSubscriptAccess<Expr<Vector<T, 2>>>,
      detail::ExprEnableGetMemberByIndex<Expr<Vector<T, 2>>> {
    LUISA_EXPR_COMMON(Vector<T, 2>)
    LUISA_EXPR_FROM_LITERAL(Vector<T, 2>)
    Expr<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
#include <dsl/swizzle_2.inl.h>
};

template<typename T>
struct Expr<Vector<T, 3>>
    : detail::ExprEnableStaticCast<Expr<Vector<T, 3>>>,
      detail::ExprEnableBitwiseCast<Expr<Vector<T, 3>>>,
      detail::ExprEnableSubscriptAccess<Expr<Vector<T, 3>>>,
      detail::ExprEnableGetMemberByIndex<Expr<Vector<T, 3>>> {
    LUISA_EXPR_COMMON(Vector<T, 3>)
    LUISA_EXPR_FROM_LITERAL(Vector<T, 3>)
    Expr<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Expr<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
#include <dsl/swizzle_3.inl.h>
};

template<typename T>
struct Expr<Vector<T, 4>>
    : detail::ExprEnableStaticCast<Expr<Vector<T, 4>>>,
      detail::ExprEnableBitwiseCast<Expr<Vector<T, 4>>>,
      detail::ExprEnableSubscriptAccess<Expr<Vector<T, 4>>>,
      detail::ExprEnableGetMemberByIndex<Expr<Vector<T, 4>>> {
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
    : detail::ExprEnableStaticCast<Ref<T>>,
      detail::ExprEnableBitwiseCast<Ref<T>>,
      detail::ExprEnableArithmeticAssign<Ref<T>> {
    static_assert(concepts::scalar<T>);
    LUISA_REF_COMMON(T)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i == 0u);
        return *this;
    }
};

template<typename T, size_t N>
struct Ref<std::array<T, N>>
    : detail::ExprEnableSubscriptAccess<Ref<std::array<T, N>>>,
      detail::ExprEnableArithmeticAssign<Ref<std::array<T, N>>>,
      detail::ExprEnableGetMemberByIndex<Ref<std::array<T, N>>> {
    LUISA_REF_COMMON(std::array<T, N>)
};

template<size_t N>
struct Ref<Matrix<N>>
    : detail::ExprEnableSubscriptAccess<Ref<Matrix<N>>>,
      detail::ExprEnableArithmeticAssign<Ref<Matrix<N>>>,
      detail::ExprEnableGetMemberByIndex<Ref<Matrix<N>>> {
    LUISA_REF_COMMON(Matrix<N>)
};

template<typename... T>
struct Ref<std::tuple<T...>> {
    LUISA_REF_COMMON(std::tuple<T...>)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        using M = std::tuple_element_t<i, std::tuple<T...>>;
        return Ref<M>{detail::FunctionBuilder::current()->member(
            Type::of<M>(), this->expression(), i)};
    };
};

template<typename T>
struct Ref<Vector<T, 2>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 2>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 2>>>,
      detail::ExprEnableSubscriptAccess<Ref<Vector<T, 2>>>,
      detail::ExprEnableArithmeticAssign<Ref<Vector<T, 2>>>,
      detail::ExprEnableGetMemberByIndex<Ref<Vector<T, 2>>> {
    LUISA_REF_COMMON(Vector<T, 2>)
    Ref<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Ref<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
#include <dsl/swizzle_2.inl.h>
};

template<typename T>
struct Ref<Vector<T, 3>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 3>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 3>>>,
      detail::ExprEnableSubscriptAccess<Ref<Vector<T, 3>>>,
      detail::ExprEnableArithmeticAssign<Ref<Vector<T, 3>>>,
      detail::ExprEnableGetMemberByIndex<Ref<Vector<T, 3>>> {
    LUISA_REF_COMMON(Vector<T, 3>)
    Ref<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Ref<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Ref<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
#include <dsl/swizzle_3.inl.h>
};

template<typename T>
struct Ref<Vector<T, 4>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 4>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 4>>>,
      detail::ExprEnableSubscriptAccess<Ref<Vector<T, 4>>>,
      detail::ExprEnableArithmeticAssign<Ref<Vector<T, 4>>>,
      detail::ExprEnableGetMemberByIndex<Ref<Vector<T, 4>>> {
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
    requires concepts::static_convertible<expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto cast() const noexcept {
        using TrueDest = expr_value_t<Dest>;
        return make_var_expr<TrueDest>(
            FunctionBuilder::current()->cast(
                Type::of<TrueDest>(),
                CastOp::STATIC,
                static_cast<const T *>(this)->expression()));
    }
};

template<typename T>
struct ExprEnableBitwiseCast {
    template<typename Dest>
    requires concepts::bitwise_convertible<expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto as() const noexcept {
        using TrueDest = expr_value_t<Dest>;
        return make_var_expr<TrueDest>(
            FunctionBuilder::current()->cast(
                Type::of<TrueDest>(),
                CastOp::BITWISE,
                static_cast<const T *>(this)->expression()));
    }
};

template<typename T>
struct ExprEnableSubscriptAccess {
    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        using Elem = std::remove_cvref_t<
            decltype(std::declval<expr_value_t<T>>()[0])>;
        using R = expr_as_t<T, Elem>;
        return R{FunctionBuilder::current()->access(
            Type::of<Elem>(),
            static_cast<const T *>(this)->expression(),
            extract_expression(std::forward<I>(index)))};
    }
};

template<typename T>
struct ExprEnableGetMemberByIndex {
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i < dimension_v<expr_value_t<T>>);
        return static_cast<const T *>(this)->operator[](static_cast<uint>(i));
    }
};

template<typename T>
struct ExprEnableArithmeticAssign {
#define LUISA_EXPR_ASSIGN_OP(op, op_name, op_tag_name)  \
    template<typename U>                                \
    requires concepts::op_name##able<                   \
        expr_value_t<T>,                                \
        expr_value_t<U>> void                           \
    operator op(U &&rhs) const noexcept {               \
        FunctionBuilder::current()->assign(             \
            AssignOp::op_tag_name,                      \
            static_cast<const T *>(this)->expression(), \
            extract_expression(std::forward<U>(rhs)));  \
    }                                                   \
    template<typename U>                                \
    void op_name(U &&rhs) const noexcept {              \
        (*this) op std::forward<U>(rhs);                \
    }
    LUISA_EXPR_ASSIGN_OP(+=, add_assign, ADD_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(-=, sub_assign, SUB_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(*=, mul_assign, MUL_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(/=, div_assign, DIV_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(%=, mod_assign, MOD_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(&=, bit_and_assign, BIT_AND_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(|=, bit_or_assign, BIT_OR_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(^=, bit_xor_assign, BIT_XOR_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(<<=, shift_left_assign, SHL_ASSIGN)
    LUISA_EXPR_ASSIGN_OP(>>=, shift_right_assign, SHR_ASSIGN)
#undef LUISA_EXPR_ASSIGN_OP
};

}// namespace detail

namespace detail {
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

    [[nodiscard]] auto load() const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_LOAD,
            {this->_expression});
        return detail::make_var_expr<T>(expr);
    };

    auto exchange(Expr<T> desired) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_EXCHANGE,
            {this->_expression, desired.expression()});
        return detail::make_var_expr<T>(expr);
    }

    // stores old == compare ? val : old, returns old
    auto compare_exchange(Expr<T> expected, Expr<T> desired) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_COMPARE_EXCHANGE,
            {this->_expression, expected.expression(), desired.expression()});
        return detail::make_var_expr<T>(expr);
    }

    auto fetch_add(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_ADD,
            {this->_expression, val.expression()});
        return detail::make_var_expr<T>(expr);
    };

    auto fetch_sub(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_SUB,
            {this->_expression, val.expression()});
        return detail::make_var_expr<T>(expr);
    };

    auto fetch_and(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_AND,
            {this->_expression, val.expression()});
        return detail::make_var_expr<T>(expr);
    };

    auto fetch_or(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_OR,
            {this->_expression, val.expression()});
        return detail::make_var_expr<T>(expr);
    };

    auto fetch_xor(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_XOR,
            {this->_expression, val.expression()});
        return detail::make_var_expr<T>(expr);
    };

    auto fetch_min(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MIN,
            {this->_expression, val.expression()});
        return detail::make_var_expr<T>(expr);
    };

    auto fetch_max(Expr<T> val) const noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MAX,
            {this->_expression, val.expression()});
        return detail::make_var_expr<T>(expr);
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
        return detail::make_var_expr<Vector<T, 4>>(
            f->call(
                Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
                {_expression, _offset_uv(uv.expression())}));
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
        return detail::make_var_expr<Vector<T, 4>>(
            detail::FunctionBuilder::current()->call(
                Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
                {_expression, _offset_uvw(uvw.expression())}));
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
class HeapBuffer {

private:
    const RefExpr *_heap{nullptr};
    const Expression *_index{nullptr};

public:
    HeapBuffer(const RefExpr *heap, const Expression *index) noexcept
        : _heap{heap},
          _index{index} {}

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&i) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<T>(
            f->call(
                Type::of<T>(), CallOp::BUFFER_HEAP_READ,
                {_heap, _index, detail::extract_expression(std::forward<I>(i))}));
    }
};

class HeapTexture2D {

private:
    const RefExpr *_heap{nullptr};
    const Expression *_index{nullptr};

public:
    HeapTexture2D(const RefExpr *heap, const Expression *index) noexcept
        : _heap{heap},
          _index{index} {}

    [[nodiscard]] auto sample(Expr<float2> uv) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D,
            {_heap, _index, uv.expression()}));
    }

    [[nodiscard]] auto sample(Expr<float2> uv, Expr<float> mip) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D_LEVEL,
            {_heap, _index, uv.expression(), mip.expression()}));
    }

    [[nodiscard]] auto sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D_GRAD,
            {_heap, _index, uv.expression(), dpdx.expression(), dpdy.expression()}));
    }

    [[nodiscard]] auto size() const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<uint2>(f->call(
            Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D,
            {_heap, _index}));
    }

    [[nodiscard]] auto size(Expr<int> level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<uint2>(f->call(
            Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D_LEVEL,
            {_heap, _index, level.expression()}));
    }

    [[nodiscard]] auto size(Expr<uint> level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<uint2>(f->call(
            Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D_LEVEL,
            {_heap, _index, level.expression()}));
    }

    [[nodiscard]] auto read(Expr<uint2> coord) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ2D,
            {_heap, _index, coord.expression()}));
    }

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint2> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ2D_LEVEL,
            {_heap, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))}));
    }
};

class HeapTexture3D {

private:
    const RefExpr *_heap{nullptr};
    const Expression *_index{nullptr};

public:
    HeapTexture3D(const RefExpr *heap, const Expression *index) noexcept
        : _heap{heap},
          _index{index} {}

    [[nodiscard]] auto sample(Expr<float3> uvw) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D,
            {_heap, _index, uvw.expression()}));
    }

    [[nodiscard]] auto sample(Expr<float3> uvw, Expr<float> mip) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D_LEVEL,
            {_heap, _index, uvw.expression(), mip.expression()}));
    }

    [[nodiscard]] auto sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D_GRAD,
            {_heap, _index, uvw.expression(), dpdx.expression(), dpdy.expression()}));
    }

    [[nodiscard]] auto size() const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<uint3>(f->call(
            Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D,
            {_heap, _index}));
    }

    [[nodiscard]] auto size(Expr<int> level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<uint3>(f->call(
            Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D_LEVEL,
            {_heap, _index, level.expression()}));
    }

    [[nodiscard]] auto size(Expr<uint> level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<uint3>(f->call(
            Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D_LEVEL,
            {_heap, _index, level.expression()}));
    }

    [[nodiscard]] auto read(Expr<uint3> coord) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ3D,
            {_heap, _index, coord.expression()}));
    }

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint3> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return detail::make_var_expr<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ3D_LEVEL,
            {_heap, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))}));
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
        return HeapTexture2D{_expression, detail::extract_expression(std::forward<I>(index))};
    }

    template<typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto tex3d(I &&index) const noexcept {
        return HeapTexture3D{_expression, detail::extract_expression(std::forward<I>(index))};
    }

    template<typename T, typename I>
    requires is_integral_expr_v<I>
    [[nodiscard]] auto buffer(I &&index) const noexcept {
        return HeapBuffer<T>{_expression, detail::extract_expression(std::forward<I>(index))};
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
HeapTexture2D Heap::tex2d(I &&index) const noexcept {
    return Expr<Heap>{*this}.tex2d(std::forward<I>(index));
}

template<typename I>
HeapTexture2D Heap::tex3d(I &&index) const noexcept {
    return Expr<Heap>{*this}.tex3d(std::forward<I>(index));
}

template<typename T, typename I>
HeapBuffer<T> Heap::buffer(I &&index) const noexcept {
    return Expr<Heap>{*this}.buffer<T>(std::forward<I>(index));
}

}// namespace luisa::compute

#define LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(op, op_concept, op_tag)                      \
    template<typename T>                                                             \
    requires luisa::compute::is_dsl_v<T>                                             \
    [[nodiscard]] inline auto operator op(T &&expr) noexcept {                       \
        using R = std::remove_cvref_t<                                               \
            decltype(op std::declval<luisa::compute::expr_value_t<T>>())>;           \
        return luisa::compute::detail::make_var_expr<R>(                             \
            luisa::compute::detail::FunctionBuilder::current()->unary(               \
                luisa::compute::Type::of<R>(),                                       \
                luisa::compute::UnaryOp::op_tag,                                     \
                luisa::compute::detail::extract_expression(std::forward<T>(expr)))); \
    }
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(+, operator_plus, PLUS)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(-, operator_minus, MINUS)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(!, operator_not, NOT)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(~, operator_bit_not, BIT_NOT)
#undef LUISA_MAKE_GLOBAL_EXPR_UNARY_OP

#define LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(op, op_concept_name, op_tag_name)            \
    template<typename Lhs, typename Rhs>                                              \
    requires luisa::compute::any_dsl_v<Lhs, Rhs> && luisa::concepts::op_concept_name< \
        luisa::compute::expr_value_t<Lhs>,                                            \
        luisa::compute::expr_value_t<Rhs>>                                            \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {            \
        using R = std::remove_cvref_t<                                                \
            decltype(std::declval<luisa::compute::expr_value_t<Lhs>>()                \
                         op std::declval<luisa::compute::expr_value_t<Rhs>>())>;      \
        return luisa::compute::detail::make_var_expr<R>(                              \
            luisa::compute::detail::FunctionBuilder::current()->binary(               \
                luisa::compute::Type::of<R>(),                                        \
                luisa::compute::BinaryOp::op_tag_name,                                \
                luisa::compute::detail::extract_expression(std::forward<Lhs>(lhs)),   \
                luisa::compute::detail::extract_expression(std::forward<Rhs>(rhs)))); \
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
