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

inline namespace dsl {

template<typename T>
[[nodiscard]] inline auto def(const Expression *expr) noexcept -> Var<expr_value_t<T>>;

template<typename T>
[[nodiscard]] inline auto def(T &&x) noexcept -> Var<expr_value_t<T>>;

}// namespace dsl

namespace detail {

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

#define LUISA_EXPR_FROM_LITERAL(...)                             \
    template<typename U>                                         \
        requires std::same_as<U, __VA_ARGS__>                    \
        Expr(U literal)                                          \
    noexcept : Expr{detail::FunctionBuilder::current()->literal( \
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

template<typename... T>
[[nodiscard]] inline auto compose(std::tuple<T...> t) noexcept {
    return detail::compose_impl(t, std::index_sequence_for<T...>{});
}

template<typename T>
[[nodiscard]] inline auto compose(T &&v) noexcept {
    return std::forward<T>(v);
}

template<typename... T>
[[nodiscard]] inline auto compose(T &&...v) noexcept {
    return compose(std::make_tuple(Expr{std::forward<T>(v)}...));
}

template<typename T>
[[nodiscard]] inline auto decompose(T &&t) noexcept {
    using member_tuple = struct_member_tuple_t<expr_value_t<T>>;
    using index_sequence = std::make_index_sequence<std::tuple_size_v<member_tuple>>;
    return detail::decompose_impl(Expr{std::forward<T>(t)}, index_sequence{});
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

#undef LUISA_EXPR_COMMON
#undef LUISA_EXPR_FROM_LITERAL

namespace detail {

template<typename T>
struct ExprEnableStaticCast {
    template<typename Dest>
        requires concepts::static_convertible<expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto cast() const noexcept {
        using TrueDest = expr_value_t<Dest>;
        return def<TrueDest>(
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
        return def<TrueDest>(
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
        return def<Elem>(FunctionBuilder::current()->access(
            Type::of<Elem>(),
            static_cast<const T *>(this)->expression(),
            extract_expression(std::forward<I>(index))));
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
    [[nodiscard]] auto &operator[](I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        auto f = detail::FunctionBuilder::current();
        auto expr = f->access(
            Type::of<T>(), _expression,
            index.expression());
        return *f->arena().create<Var<T>>(expr);
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
    AtomicRef(AtomicRef &&) noexcept = delete;
    AtomicRef(const AtomicRef &) noexcept = delete;
    AtomicRef &operator=(AtomicRef &&) noexcept = delete;
    AtomicRef &operator=(const AtomicRef &) noexcept = delete;

    void store(Expr<T> value) &&noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::ATOMIC_STORE,
            {this->_expression, value.expression()});
    }

    [[nodiscard]] auto load() &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_LOAD,
            {this->_expression});
        return def<T>(expr);
    };

    auto exchange(Expr<T> desired) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_EXCHANGE,
            {this->_expression, desired.expression()});
        return def<T>(expr);
    }

    // stores old == compare ? val : old, returns old
    auto compare_exchange(Expr<T> expected, Expr<T> desired) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_COMPARE_EXCHANGE,
            {this->_expression, expected.expression(), desired.expression()});
        return def<T>(expr);
    }

    auto fetch_add(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_ADD,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    auto fetch_sub(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_SUB,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    auto fetch_and(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_AND,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    auto fetch_or(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_OR,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    auto fetch_xor(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_XOR,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    auto fetch_min(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MIN,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };

    auto fetch_max(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MAX,
            {this->_expression, val.expression()});
        return def<T>(expr);
    };
};

namespace detail {

template<>
struct BufferExprAsAtomic<int> {
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<int>{FunctionBuilder::current()->access(
            Type::of<int>(),
            static_cast<const Expr<Buffer<int>> *>(this)->expression(),
            index.expression())};
    }
};

template<>
struct BufferExprAsAtomic<uint> {
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<uint>{FunctionBuilder::current()->access(
            Type::of<uint>(),
            static_cast<const Expr<Buffer<uint>> *>(this)->expression(),
            index.expression())};
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
          _offset{any(image.offset()) ? detail::FunctionBuilder::current()->literal(Type::of<uint2>(), image.offset()) : nullptr} {}

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }

    [[nodiscard]] auto read(Expr<uint2> uv) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<Vector<T, 4>>(
            f->call(
                Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
                {_expression, _offset_uv(uv.expression())}));
    };

    void write(Expr<uint2> uv, Expr<Vector<T, 4>> value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::TEXTURE_WRITE,
            {_expression, _offset_uv(uv.expression()), value.expression()});
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint2> uv, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<Vector<T, 4>>(
            f->call(
                Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ_LEVEL,
                {_expression,
                 _offset_uv(uv.expression()),
                 detail::extract_expression(std::forward<I>(level))}));
    };

    template<typename I>
        requires is_integral_expr_v<I>
    void write(Expr<uint2> uv, Expr<Vector<T, 4>> value, I &&level) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::TEXTURE_WRITE_LEVEL,
            {_expression,
             _offset_uv(uv.expression()),
             value.expression(),
             detail::extract_expression(std::forward<I>(level))});
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
          _offset{any(volume.offset()) ? detail::FunctionBuilder::current()->literal(Type::of<uint3>(), volume.offset()) : nullptr} {}

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }

    [[nodiscard]] auto read(Expr<uint3> uvw) const noexcept {
        return def<Vector<T, 4>>(
            detail::FunctionBuilder::current()->call(
                Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
                {_expression, _offset_uvw(uvw.expression())}));
    };

    void write(Expr<uint3> uvw, Expr<Vector<T, 4>> value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::TEXTURE_WRITE,
            {_expression, _offset_uvw(uvw.expression()), value.expression()});
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint3> uvw, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<Vector<T, 4>>(
            f->call(
                Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ_LEVEL,
                {_expression,
                 _offset_uvw(uvw.expression()),
                 detail::extract_expression(std::forward<I>(level))}));
    };

    template<typename I>
        requires is_integral_expr_v<I>
    void write(Expr<uint3> uvw, Expr<Vector<T, 4>> value, I &&level) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::TEXTURE_WRITE_LEVEL,
            {_expression,
             _offset_uvw(uvw.expression()),
             value.expression(),
             detail::extract_expression(std::forward<I>(level))});
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
        return def<T>(
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
    [[nodiscard]] Var<float4> sample(Expr<float2> uv) const noexcept;
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float> mip) const noexcept;
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy) const noexcept;
    [[nodiscard]] Var<uint2> size() const noexcept;
    [[nodiscard]] Var<uint2> size(Expr<int> level) const noexcept;
    [[nodiscard]] Var<uint2> size(Expr<uint> level) const noexcept;
    [[nodiscard]] Var<float4> read(Expr<uint2> coord) const noexcept;

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint2> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<float4>(f->call(
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
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw) const noexcept;
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float> mip) const noexcept;
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy) const noexcept;
    [[nodiscard]] Var<uint3> size() const noexcept;
    [[nodiscard]] Var<uint3> size(Expr<int> level) const noexcept;
    [[nodiscard]] Var<uint3> size(Expr<uint> level) const noexcept;
    [[nodiscard]] Var<float4> read(Expr<uint3> coord) const noexcept;

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint3> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<float4>(f->call(
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
    [[nodiscard]] auto tex2d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return HeapTexture2D{_expression, i.expression()};
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex3d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return HeapTexture3D{_expression, i.expression()};
    }

    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto buffer(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return HeapBuffer<T>{_expression, i.expression()};
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

Expr(const Heap &) -> Expr<Heap>;

template<typename I>
HeapTexture2D Heap::tex2d(I &&index) const noexcept {
    auto i = def(std::forward<I>(index));
    return Expr<Heap>{*this}.tex2d(i);
}

template<typename I>
HeapTexture2D Heap::tex3d(I &&index) const noexcept {
    auto i = def(std::forward<I>(index));
    return Expr<Heap>{*this}.tex3d(i);
}

template<typename T, typename I>
HeapBuffer<T> Heap::buffer(I &&index) const noexcept {
    auto i = def(std::forward<I>(index));
    return Expr<Heap>{*this}.buffer<T>(i);
}

}// namespace luisa::compute
