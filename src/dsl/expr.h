//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <array>
#include <string_view>

#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/buffer.h>
#include <runtime/bindless_array.h>
#include <ast/function_builder.h>
#include <dsl/expr_traits.h>
#include <dsl/arg.h>

namespace luisa::compute {

inline namespace dsl {

template<typename T>
[[nodiscard]] auto def(const Expression *expr) noexcept -> Var<expr_value_t<T>>;

template<typename T>
[[nodiscard]] auto def(T &&x) noexcept -> Var<expr_value_t<T>>;

}// namespace dsl

namespace detail {

/// Extract or construct expression from given data
template<typename T>
[[nodiscard]] inline auto extract_expression(T &&v) noexcept {
    if constexpr (is_dsl_v<T>) {
        return std::forward<T>(v).expression();
    } else {
        static_assert(concepts::basic<T>);
        return FunctionBuilder::current()->literal(Type::of<T>(), std::forward<T>(v));
    }
}

}// namespace detail

/// Expr common definition
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

/// Construct Expr from literal value
#define LUISA_EXPR_FROM_LITERAL(...)                                             \
    template<typename U>                                                         \
        requires std::same_as<U, __VA_ARGS__>                                    \
    Expr(U literal) noexcept : Expr{detail::FunctionBuilder::current()->literal( \
                                   Type::of<U>(), literal)} {}

namespace detail {

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
    return Var<std::tuple<expr_value_t<T>...>>{Expr{std::get<i>(t)}...};
}

template<typename T, size_t... i>
[[nodiscard]] inline auto decompose_impl(Expr<T> x, std::index_sequence<i...>) noexcept {
    return std::make_tuple(x.template get<i>()...);
}

}// namespace detail

inline namespace dsl {

/// Compose tuple of values to var of tuple
template<typename... T>
[[nodiscard]] inline auto compose(std::tuple<T...> t) noexcept {
    return detail::compose_impl(t, std::index_sequence_for<T...>{});
}

/// Do nothing
template<typename T>
[[nodiscard]] inline auto compose(T &&v) noexcept {
    return std::forward<T>(v);
}

/// Compose values to var of tuple
template<typename... T>
[[nodiscard]] inline auto compose(T &&...v) noexcept {
    return compose(std::make_tuple(Expr{std::forward<T>(v)}...));
}

/// Decompose var of tuple to tuple
template<typename T>
[[nodiscard]] inline auto decompose(T &&t) noexcept {
    using member_tuple = struct_member_tuple_t<expr_value_t<T>>;
    using index_sequence = std::make_index_sequence<std::tuple_size_v<member_tuple>>;
    return detail::decompose_impl(Expr{std::forward<T>(t)}, index_sequence{});
}

}// namespace dsl

/**
 * @brief Class of Expr<T>.
 * 
 * Member function's tparam must be 0.
 * 
 * @tparam T needs to be scalar 
 */
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

/// Class of Expr<std::array<T, N>>
template<typename T, size_t N>
struct Expr<std::array<T, N>>
    : detail::ExprEnableSubscriptAccess<Expr<std::array<T, N>>>,
      detail::ExprEnableGetMemberByIndex<Expr<std::array<T, N>>> {
    LUISA_EXPR_COMMON(std::array<T, N>)
};

/// Class of Expr<Matrix><N>>. Can be constructed from Matrix<N>
template<size_t N>
struct Expr<Matrix<N>>
    : detail::ExprEnableSubscriptAccess<Expr<Matrix<N>>>,
      detail::ExprEnableGetMemberByIndex<Expr<Matrix<N>>> {
    LUISA_EXPR_COMMON(Matrix<N>)
    LUISA_EXPR_FROM_LITERAL(Matrix<N>)
};

/// Class of Expr<std::tuple<T...>>.
/// get<i>() will return Expr of the ith member of tuple
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

/// Class of Expr<Vector<T, 2>>. Can be constructed from Vector<T, 2>.
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

/// Class of Expr<Vector<T, 3>>. Can be constructed from Vector<T, 3>.
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

/// Class of Expr<Vector<T, 4>>. Can be constructed from Vector<T, 4>.
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

#undef LUISA_EXPR_COMMON
#undef LUISA_EXPR_FROM_LITERAL

namespace detail {

/// Enable static cast to type Dest
template<typename T>
struct ExprEnableStaticCast {
    template<typename Dest>
        requires concepts::static_convertible<expr_value_t<T>,
                                              expr_value_t<Dest>> [[nodiscard]] auto
        cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using TrueDest = expr_value_t<Dest>;
        return def<TrueDest>(
            FunctionBuilder::current()->cast(
                Type::of<TrueDest>(),
                CastOp::STATIC,
                src.expression()));
    }
};

/// Enable bitwise cast to type Dest
template<typename T>
struct ExprEnableBitwiseCast {
    template<typename Dest>
        requires concepts::bitwise_convertible<expr_value_t<T>,
                                               expr_value_t<Dest>> [[nodiscard]] auto
        as() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using TrueDest = expr_value_t<Dest>;
        return def<TrueDest>(
            FunctionBuilder::current()->cast(
                Type::of<TrueDest>(),
                CastOp::BITWISE,
                src.expression()));
    }
};

/// Enable subscript access
template<typename T>
struct ExprEnableSubscriptAccess {
    template<typename I>
        requires is_integral_expr_v<I> [[nodiscard]] auto
        operator[](I &&index) const noexcept {
        auto self = def<T>(static_cast<const T *>(this)->expression());
        using Elem = std::remove_cvref_t<
            decltype(std::declval<expr_value_t<T>>()[0])>;
        return def<Elem>(FunctionBuilder::current()->access(
            Type::of<Elem>(), self.expression(),
            extract_expression(std::forward<I>(index))));
    }
};

/// Enable get member by index
template<typename T>
struct ExprEnableGetMemberByIndex {
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i < dimension_v<expr_value_t<T>>);
        return static_cast<const T *>(this)->operator[](static_cast<uint>(i));
    }
};

}// namespace detail

namespace detail {
template<typename>
struct BufferExprAsAtomic {};
}// namespace detail

/// Class of Expr<Buffer<T>>
template<typename T>
struct Expr<Buffer<T>>
    : detail::BufferExprAsAtomic<T> {

private:
    const RefExpr *_expression{nullptr};

public:
    /// Construct from RefExpr
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}
    /// Construct from BufferView. Will call buffer_binding() to bind buffer
    explicit Expr(BufferView<T> buffer) noexcept
        : _expression{detail::FunctionBuilder::current()->buffer_binding(
              Type::of<Buffer<T>>(), buffer.handle(),
              buffer.offset_bytes(), buffer.size_bytes())} {}

    /// Return RefExpr
    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }

    /// Read buffer at index
    template<typename I>
        requires is_integral_expr_v<I> [
            [nodiscard]] auto
        read(I &&index) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto expr = f->call(
            Type::of<T>(), CallOp::BUFFER_READ,
            {_expression,
             detail::extract_expression(std::forward<I>(index))});
        return def<T>(expr);
    }

    /// Write buffer at index
    template<typename I>
        requires is_integral_expr_v<I>
    void write(I &&index, Expr<T> value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::BUFFER_WRITE,
            {_expression,
             detail::extract_expression(std::forward<I>(index)),
             value.expression()});
    }
};

/// Same as Expr<Buffer<T>>
template<typename T>
struct Expr<BufferView<T>> : public Expr<Buffer<T>> {
    using Expr<Buffer<T>>::Expr;
};

/// Class of atomic reference
template<typename T>
class AtomicRef {

private:
    const AccessExpr *_expression{nullptr};

public:
    /// Construct from AccessExpr
    explicit AtomicRef(const AccessExpr *expr) noexcept
        : _expression{expr} {}
    AtomicRef(AtomicRef &&) noexcept = delete;
    AtomicRef(const AtomicRef &) noexcept = delete;
    AtomicRef &operator=(AtomicRef &&) noexcept = delete;
    AtomicRef &operator=(const AtomicRef &) noexcept = delete;

    /// Atomic exchange. Stores desired, returns old. See also CallOp::ATOMIC_EXCHANGE.
    auto exchange(Expr<T> desired) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_EXCHANGE,
            {this->_expression, desired.expression()});
        return def<T>(expr);
    }

    /// Atomic compare exchange. Stores old == expected ? desired : old, returns old. See also CallOp::ATOMIC_COMPARE_EXCHANGE.
    auto compare_exchange(Expr<T> expected, Expr<T> desired) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_COMPARE_EXCHANGE,
            {this->_expression, expected.expression(), desired.expression()});
        return def<T>(expr);
    }

    /// Atomic fetch add. Stores old + val, returns old. See also CallOp::ATOMIC_FETCH_ADD.
    auto fetch_add(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_ADD,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch sub. Stores old - val, returns old. See also CallOp::ATOMIC_FETCH_SUB.
    auto fetch_sub(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_SUB,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch and. Stores old & val, returns old. See also CallOp::ATOMIC_FETCH_AND.
    auto fetch_and(Expr<T> val) &&noexcept requires std::integral<T> {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_AND,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch or. Stores old | val, returns old. See also CallOp::ATOMIC_FETCH_OR.
    auto fetch_or(Expr<T> val) &&noexcept requires std::integral<T>  {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_OR,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch xor. Stores old ^ val, returns old. See also CallOp::ATOMIC_FETCH_XOR.
    auto fetch_xor(Expr<T> val) &&noexcept requires std::integral<T>  {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_XOR,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch min. Stores min(old, val), returns old. See also CallOp::ATOMIC_FETCH_MIN.
    auto fetch_min(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MIN,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch max. Stores max(old, val), returns old. See also CallOp::ATOMIC_FETCH_MAX.
    auto fetch_max(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MAX,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };
};

namespace detail {

/// Integer buffer expr as atomic
template<>
struct BufferExprAsAtomic<int> {
    /// Atomic access
    template<typename I>
        requires is_integral_expr_v<I>
            [[nodiscard]] auto
        atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<int>{FunctionBuilder::current()->access(
            Type::of<int>(),
            static_cast<const Expr<Buffer<int>> *>(this)->expression(),
            index.expression())};
    }
};

/// Unsigned integer buffer expr as atomic
template<>
struct BufferExprAsAtomic<uint> {
    /// Atomic access
    template<typename I>
        requires is_integral_expr_v<I>
            [[nodiscard]] auto
        atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<uint>{FunctionBuilder::current()->access(
            Type::of<uint>(),
            static_cast<const Expr<Buffer<uint>> *>(this)->expression(),
            index.expression())};
    }
};

/// Floating point buffer expr as atomic
template<>
struct BufferExprAsAtomic<float> {
    /// Atomic access
    template<typename I>
        requires is_integral_expr_v<I>
            [[nodiscard]] auto
        atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<float>{FunctionBuilder::current()->access(
            Type::of<float>(),
            static_cast<const Expr<Buffer<float>> *>(this)->expression(),
            index.expression())};
    }
};

}// namespace detail

/// Class of Expr<Image<T>>
template<typename T>
struct Expr<Image<T>> {

private:
    const RefExpr *_expression{nullptr};

public:
    /// Construct from RefExpr
    explicit Expr(const RefExpr *expr) noexcept : _expression{expr} {}
    /// Construct from ImageView. Will create texture binding.
    explicit Expr(ImageView<T> image) noexcept
        : _expression{detail::FunctionBuilder::current()->texture_binding(
              Type::of<Image<T>>(), image.handle(), image.level())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }

    /// Read at (u, v)
    [[nodiscard]] auto read(Expr<uint2> uv) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<Vector<T, 4>>(
            f->call(Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
                    {_expression, uv.expression()}));
    };

    /// Write T4 at (u, v)
    void write(Expr<uint2> uv, Expr<Vector<T, 4>> value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::TEXTURE_WRITE,
            {_expression, uv.expression(), value.expression()});
    }
};

/// Same as Expr<Image<T>>
template<typename T>
struct Expr<ImageView<T>> : public Expr<Image<T>> {
    using Expr<Image<T>>::Expr;
};

/// Class of Expr<Volume<T>>
template<typename T>
struct Expr<Volume<T>> {

private:
    const RefExpr *_expression{nullptr};

public:
    /// Construct from RefExpr
    explicit Expr(const RefExpr *expr, const Expression *offset) noexcept : _expression{expr} {}
    /// Construct from VolumeView. Will create texture binding.
    explicit Expr(VolumeView<T> volume) noexcept
        : _expression{detail::FunctionBuilder::current()->texture_binding(
              Type::of<Volume<T>>(), volume.handle(), volume.level())} {}

    [[nodiscard]] auto expression() const noexcept { return _expression; }

    /// Read at (u, v, w)
    [[nodiscard]] auto read(Expr<uint3> uvw) const noexcept {
        return def<Vector<T, 4>>(
            detail::FunctionBuilder::current()->call(
                Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
                {_expression, uvw.expression()}));
    };

    /// Write T4 at (u, v, w)
    void write(Expr<uint3> uvw, Expr<Vector<T, 4>> value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::TEXTURE_WRITE,
            {_expression, uvw.expression(), value.expression()});
    }
};

/// Same as Expr<Volume<T>>
template<typename T>
struct Expr<VolumeView<T>> : public Expr<Volume<T>> {
    using Expr<Volume<T>>::Expr;
};

/// Class of bindless buffer
template<typename T>
class BindlessBuffer {

    static_assert(is_valid_buffer_element_v<T>);

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};

public:
    /// Construct from array RefExpr and index Expression
    BindlessBuffer(const RefExpr *array, const Expression *index) noexcept
        : _array{array},
          _index{index} {}

    /// Read at index i
    template<typename I>
        requires is_integral_expr_v<I> [
            [nodiscard]] auto
        read(I &&i) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<T>(
            f->call(
                Type::of<T>(), CallOp::BINDLESS_BUFFER_READ,
                {_array, _index, detail::extract_expression(std::forward<I>(i))}));
    }
};

/// Class of bindless 2D texture
class LC_DSL_API BindlessTexture2D {

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};

public:
    /// Construct from array RefExpr and index Expression
    BindlessTexture2D(const RefExpr *array, const Expression *index) noexcept
        : _array{array},
          _index{index} {}
    /// Sample at (u, v)
    [[nodiscard]] Var<float4> sample(Expr<float2> uv) const noexcept;
    /// Sample at (u, v) at mip level
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float> mip) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy) const noexcept;
    /// Size
    [[nodiscard]] Var<uint2> size() const noexcept;
    /// Size at level
    [[nodiscard]] Var<uint2> size(Expr<int> level) const noexcept;
    /// Size at level
    [[nodiscard]] Var<uint2> size(Expr<uint> level) const noexcept;
    /// Read at coordinate
    [[nodiscard]] Var<float4> read(Expr<uint2> coord) const noexcept;

    /// Read at coordinate and mipmap level
    template<typename I>
        requires is_integral_expr_v<I> [
            [nodiscard]] auto
        read(Expr<uint2> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<float4>(f->call(
            Type::of<float4>(), CallOp::BINDLESS_TEXTURE2D_READ_LEVEL,
            {_array, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))}));
    }
};

/// Class of bindless 3D texture
class LC_DSL_API BindlessTexture3D {

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};

public:
    /// Construct from array RefExpr and index Expression
    BindlessTexture3D(const RefExpr *array, const Expression *index) noexcept
        : _array{array},
          _index{index} {}
    /// Sample at (u, v, w)
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw) const noexcept;
    /// Sample at (u, v, w) at mip level
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float> mip) const noexcept;
    /// Sample at (u, v, w) with grad dpdx, dpdy
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy) const noexcept;
    /// Size
    [[nodiscard]] Var<uint3> size() const noexcept;
    /// Size at level
    [[nodiscard]] Var<uint3> size(Expr<int> level) const noexcept;
    /// Size at level
    [[nodiscard]] Var<uint3> size(Expr<uint> level) const noexcept;
    /// Read at coordinate
    [[nodiscard]] Var<float4> read(Expr<uint3> coord) const noexcept;

    /// Read at coordinate and mipmap level
    template<typename I>
        requires is_integral_expr_v<I> [
            [nodiscard]] auto
        read(Expr<uint3> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<float4>(f->call(
            Type::of<float4>(), CallOp::BINDLESS_TEXTURE3D_READ_LEVEL,
            {_array, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))}));
    }
};

/// Class of Expr<BindlessArray>
template<>
struct Expr<BindlessArray> {

private:
    const RefExpr *_expression{nullptr};

public:
    /// Construct from RefExpr
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}

    /// Construct from BindlessArray. Will create bindless array binding
    explicit Expr(const BindlessArray &array) noexcept
        : _expression{detail::FunctionBuilder::current()->bindless_array_binding(array.handle())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }

    /// Get 2D texture at index
    template<typename I>
        requires is_integral_expr_v<I> [
            [nodiscard]] auto
        tex2d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return BindlessTexture2D{_expression, i.expression()};
    }

    /// Get 3D texture at index
    template<typename I>
        requires is_integral_expr_v<I> [
            [nodiscard]] auto
        tex3d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return BindlessTexture3D{_expression, i.expression()};
    }

    /// Get buffer at index
    template<typename T, typename I>
        requires is_integral_expr_v<I> [
            [nodiscard]] auto
        buffer(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return BindlessBuffer<T>{_expression, i.expression()};
    }
};

// deduction guides
template<typename T>
Expr(Expr<T>) -> Expr<T>;

template<typename T>
Expr(const Var<T> &) -> Expr<T>;

template<typename T>
Expr(detail::Ref<T>) -> Expr<T>;

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

Expr(const BindlessArray &) -> Expr<BindlessArray>;

/// Get 2D texture at index
template<typename I>
BindlessTexture2D BindlessArray::tex2d(I &&index) const noexcept {
    auto i = def(std::forward<I>(index));
    return Expr<BindlessArray>{*this}.tex2d(i);
}

/// Get 3D textrue at index
template<typename I>
BindlessTexture3D BindlessArray::tex3d(I &&index) const noexcept {
    auto i = def(std::forward<I>(index));
    return Expr<BindlessArray>{*this}.tex3d(i);
}

/// Get buffer at index
template<typename T, typename I>
BindlessBuffer<T> BindlessArray::buffer(I &&index) const noexcept {
    auto i = def(std::forward<I>(index));
    return Expr<BindlessArray>{*this}.buffer<T>(i);
}

}// namespace luisa::compute
