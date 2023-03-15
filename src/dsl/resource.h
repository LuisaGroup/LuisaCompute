//
// Created by Mike Smith on 2023/2/14.
//

#pragma once

#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/bindless_array.h>
#include <dsl/expr.h>
#include <dsl/var.h>

namespace luisa::compute {

namespace detail {

/// Class of atomic reference
template<typename T>
class AtomicRef {

private:
    const Expression *_range{nullptr};
    const Expression *_index{nullptr};

public:
    /// Construct from AccessExpr
    AtomicRef(const Expression *range,
              const Expression *index) noexcept
        : _range{range}, _index{index} {}
    AtomicRef(AtomicRef &&) noexcept = delete;
    AtomicRef(const AtomicRef &) noexcept = delete;
    AtomicRef &operator=(AtomicRef &&) noexcept = delete;
    AtomicRef &operator=(const AtomicRef &) noexcept = delete;

    /// Atomic exchange. Stores desired, returns old. See also CallOp::ATOMIC_EXCHANGE.
    auto exchange(Expr<T> desired) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_EXCHANGE,
            {this->_range, this->_index, desired.expression()});
        return def<T>(expr);
    }

    /// Atomic compare exchange. Stores old == expected ? desired : old, returns old. See also CallOp::ATOMIC_COMPARE_EXCHANGE.
    auto compare_exchange(Expr<T> expected, Expr<T> desired) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_COMPARE_EXCHANGE,
            {this->_range, this->_index, expected.expression(), desired.expression()});
        return def<T>(expr);
    }

    /// Atomic fetch add. Stores old + val, returns old. See also CallOp::ATOMIC_FETCH_ADD.
    auto fetch_add(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_ADD,
            {this->_range, this->_index, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch sub. Stores old - val, returns old. See also CallOp::ATOMIC_FETCH_SUB.
    auto fetch_sub(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_SUB,
            {this->_range, this->_index, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch and. Stores old & val, returns old. See also CallOp::ATOMIC_FETCH_AND.
    auto fetch_and(Expr<T> val) &&noexcept
        requires std::integral<T>
    {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_AND,
            {this->_range, this->_index, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch or. Stores old | val, returns old. See also CallOp::ATOMIC_FETCH_OR.
    auto fetch_or(Expr<T> val) &&noexcept
        requires std::integral<T>
    {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_OR,
            {this->_range, this->_index, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch xor. Stores old ^ val, returns old. See also CallOp::ATOMIC_FETCH_XOR.
    auto fetch_xor(Expr<T> val) &&noexcept
        requires std::integral<T>
    {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_XOR,
            {this->_range, this->_index, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch min. Stores min(old, val), returns old. See also CallOp::ATOMIC_FETCH_MIN.
    auto fetch_min(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MIN,
            {this->_range, this->_index, val.expression()});
        return def<T>(expr);
    };

    /// Atomic fetch max. Stores max(old, val), returns old. See also CallOp::ATOMIC_FETCH_MAX.
    auto fetch_max(Expr<T> val) &&noexcept {
        auto expr = detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MAX,
            {this->_range, this->_index, val.expression()});
        return def<T>(expr);
    };
};

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
    Expr(BufferView<T> buffer) noexcept
        : _expression{detail::FunctionBuilder::current()->buffer_binding(
              Type::of<Buffer<T>>(), buffer.handle(),
              buffer.offset_bytes(), buffer.size_bytes())} {}

    /// Return RefExpr
    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }

    /// Read buffer at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&index) const noexcept {
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

namespace detail {

/// Integer buffer expr as atomic
template<>
struct BufferExprAsAtomic<int> {
    /// Atomic access
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<int>{
            static_cast<const Expr<Buffer<int>> *>(this)->expression(),
            index.expression()};
    }
};

/// Unsigned integer buffer expr as atomic
template<>
struct BufferExprAsAtomic<uint> {
    /// Atomic access
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<uint>{
            static_cast<const Expr<Buffer<uint>> *>(this)->expression(),
            index.expression()};
    }
};

/// Floating point buffer expr as atomic
template<>
struct BufferExprAsAtomic<float> {
    /// Atomic access
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<float>{
            static_cast<const Expr<Buffer<float>> *>(this)->expression(),
            index.expression()};
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
    Expr(ImageView<T> image) noexcept
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
    Expr(VolumeView<T> volume) noexcept
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

namespace detail {

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
        : _array{array}, _index{index} {}

    /// Read at index i
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&i) const noexcept {
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
        : _array{array}, _index{index} {}
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
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint2> coord, I &&level) const noexcept {
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
        : _array{array}, _index{index} {}
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
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint3> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<float4>(f->call(
            Type::of<float4>(), CallOp::BINDLESS_TEXTURE3D_READ_LEVEL,
            {_array, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))}));
    }
};

}// namespace detail

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
    Expr(const BindlessArray &array) noexcept
        : _expression{detail::FunctionBuilder::current()->bindless_array_binding(array.handle())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }

    /// Get 2D texture at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex2d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessTexture2D{_expression, i.expression()};
    }

    /// Get 3D texture at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex3d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessTexture3D{_expression, i.expression()};
    }

    /// Get buffer at index
    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto buffer(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessBuffer<T>{_expression, i.expression()};
    }
};

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

// resource proxies
namespace detail {

template<typename T>
class BufferExprProxy {

private:
    T _buffer;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(BufferExprProxy)

public:
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        return Expr<T>{_buffer}.read(std::forward<I>(index));
    }
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write(I &&index, V &&value) const noexcept {
        return Expr<T>{_buffer}.write(std::forward<I>(index),
                                      std::forward<V>(value));
    }
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&index) const noexcept {
        return Expr<T>{_buffer}.atomic(std::forward<I>(index));
    }
};

template<typename T>
class ImageExprProxy {

private:
    T _img;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(ImageExprProxy)

public:
    [[nodiscard]] auto read(Expr<uint2> uv) const noexcept {
        return Expr<T>{_img}.read(uv);
    }
    template<typename V>
    void write(Expr<uint2> uv, V &&value) const noexcept {
        Expr<T>{_img}.write(uv, std::forward<V>(value));
    }
};

template<typename T>
class VolumeExprProxy {

private:
    T _img;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(VolumeExprProxy)

public:
    [[nodiscard]] auto read(Expr<uint3> uv) const noexcept {
        return Expr<T>{_img}.read(uv);
    }
    template<typename V>
    void write(Expr<uint3> uv, V &&value) const noexcept {
        Expr<T>{_img}.write(uv, std::forward<V>(value));
    }
};

class BindlessArrayExprProxy {

private:
    BindlessArray _array;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(BindlessArrayExprProxy)

public:
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex2d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return Expr<BindlessArray>{_array}.tex2d(i);
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex3d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return Expr<BindlessArray>{_array}.tex3d(i);
    }

    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto buffer(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return Expr<BindlessArray>{_array}.buffer<T>(i);
    }
};

}// namespace detail

template<typename T>
struct Var<Buffer<T>> : public Expr<Buffer<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Buffer<T>>{
              detail::FunctionBuilder::current()->buffer(Type::of<Buffer<T>>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<typename T>
struct Var<BufferView<T>> : public Expr<Buffer<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Buffer<T>>{
              detail::FunctionBuilder::buffer(Type::of<Buffer<T>>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<typename T>
struct Var<Image<T>> : public Expr<Image<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Image<T>>{detail::FunctionBuilder::current()->texture(Type::of<Image<T>>())} {
    }
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<typename T>
struct Var<ImageView<T>> : public Expr<Image<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Image<T>>{detail::FunctionBuilder::texture(Type::of<Image<T>>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<typename T>
struct Var<Volume<T>> : public Expr<Volume<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Volume<T>>{detail::FunctionBuilder::current()->texture(Type::of<Volume<T>>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<typename T>
struct Var<VolumeView<T>> : public Expr<Volume<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Volume<T>>{detail::FunctionBuilder::texture(Type::of<Volume<T>>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<>
struct Var<BindlessArray> : public Expr<BindlessArray> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<BindlessArray>{
              detail::FunctionBuilder::current()->bindless_array()} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<typename T>
using BufferVar = Var<Buffer<T>>;

template<typename T>
using ImageVar = Var<Image<T>>;

template<typename T>
using VolumeVar = Var<Volume<T>>;

using BindlessVar = Var<BindlessArray>;

using BufferInt = BufferVar<int>;
using BufferInt2 = BufferVar<int2>;
using BufferInt3 = BufferVar<int3>;
using BufferInt4 = BufferVar<int4>;
using BufferUInt = BufferVar<uint>;
using BufferUInt2 = BufferVar<uint2>;
using BufferUInt3 = BufferVar<uint3>;
using BufferUInt4 = BufferVar<uint4>;
using BufferFloat = BufferVar<float>;
using BufferFloat2 = BufferVar<float2>;
using BufferFloat3 = BufferVar<float3>;
using BufferFloat4 = BufferVar<float4>;
using BufferBool = BufferVar<bool>;
using BufferBool2 = BufferVar<bool2>;
using BufferBool3 = BufferVar<bool3>;
using BufferBool4 = BufferVar<bool4>;

using ImageInt = ImageVar<int>;
using ImageUInt = ImageVar<uint>;
using ImageFloat = ImageVar<float>;

using VolumeInt = VolumeVar<int>;
using VolumeUInt = VolumeVar<uint>;
using VolumeFloat = VolumeVar<float>;

}// namespace luisa::compute
