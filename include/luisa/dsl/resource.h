#pragma once

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/volume.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/dsl/expr.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/atomic.h>
#include <luisa/dsl/shared.h>
#include <luisa/dsl/builtin.h>

namespace luisa::compute {
enum struct SamplerFilter {
    POINT = 0,
    LINEAR_POINT = 1,
    LINEAR_LINEAR = 2,
    ANISOTROPIC = 3,
};

enum struct SamplerAddress {
    EDGE = 0,
    REPEAT = 1,
    MIRROR = 2,
    ZERO = 3,
};
namespace detail {

template<typename T>
struct BufferExprAsAtomic {
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        auto buffer = static_cast<const Expr<Buffer<T>> *>(this)->expression();
        return AtomicRef<T>{AtomicRefNode::create(buffer)->access(index.expression())};
    }
};

// no-op for non-atomic struct
template<typename T>
    requires is_custom_struct_v<T>
struct BufferExprAsAtomic<T> {};

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

    /// Contruct from Buffer. Will call buffer_binding() to bind buffer
    Expr(const Buffer<T> &buffer) noexcept
        : Expr{BufferView{buffer}} {}

    /// Construct from Var<Buffer<T>>.
    Expr(const Var<Buffer<T>> &buffer) noexcept
        : Expr{buffer.expression()} {}

    /// Construct from Var<BufferView<T>>.
    Expr(const Var<BufferView<T>> &buffer) noexcept
        : Expr{buffer.expression()} {}

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

    /// Volatile read buffer at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto volatile_read(I &&index) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto expr = f->call(
            Type::of<T>(), CallOp::BUFFER_VOLATILE_READ,
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

    /// Volatile write buffer at index
    template<typename I>
        requires is_integral_expr_v<I>
    void volatile_write(I &&index, Expr<T> value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::BUFFER_VOLATILE_WRITE,
            {_expression,
             detail::extract_expression(std::forward<I>(index)),
             value.expression()});
    }

    [[nodiscard]] Expr<uint64_t> device_address() const noexcept {
        return def<uint64_t>(detail::FunctionBuilder::current()->call(
            Type::of<uint64_t>(), CallOp::BUFFER_ADDRESS, {_expression}));
    }
    /// Self-pointer to unify the interfaces of the captured Buffer<T> and Expr<Buffer<T>>
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

template<>
struct Expr<ByteBuffer> {
private:
    const RefExpr *_expression{nullptr};
public:
    /// Construct from RefExpr
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}

    /// Construct from BufferView. Will call buffer_binding() to bind buffer
    Expr(const ByteBuffer &buffer) noexcept
        : _expression{detail::FunctionBuilder::current()->buffer_binding(
              Type::of<ByteBuffer>(), buffer.handle(),
              0u, buffer.size_bytes())} {}

    /// Return RefExpr
    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }

    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&byte_offset) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto expr = f->call(
            Type::of<T>(), CallOp::BYTE_BUFFER_READ,
            {_expression,
             detail::extract_expression(std::forward<I>(byte_offset))});
        return def<T>(expr);
    }
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write(I &&byte_offset, V &&value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::BYTE_BUFFER_WRITE,
            {_expression,
             detail::extract_expression(std::forward<I>(byte_offset)),
             detail::extract_expression(std::forward<V>(value))});
    }

    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto volatile_read(I &&byte_offset) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto expr = f->call(
            Type::of<T>(), CallOp::BYTE_BUFFER_VOLATILE_READ,
            {_expression,
             detail::extract_expression(std::forward<I>(byte_offset))});
        return def<T>(expr);
    }
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void volatile_write(I &&byte_offset, V &&value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::BYTE_BUFFER_VOLATILE_WRITE,
            {_expression,
             detail::extract_expression(std::forward<I>(byte_offset)),
             detail::extract_expression(std::forward<V>(value))});
    }
    [[nodiscard]] Expr<uint64_t> device_address() const noexcept {
        return def<uint64_t>(detail::FunctionBuilder::current()->call(
            Type::of<uint64_t>(), CallOp::BUFFER_ADDRESS, {_expression}));
    }
};

/// Same as Expr<Buffer<T>>
template<typename T>
struct Expr<BufferView<T>> : public Expr<Buffer<T>> {
    using Expr<Buffer<T>>::Expr;
};

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

    /// Construct from Image. Will create texture binding.
    Expr(const Image<T> &image) noexcept
        : Expr{ImageView{image}} {}

    /// Construct from Var<Image<T>>.
    Expr(const Var<Image<T>> &image) noexcept
        : Expr{image.expression()} {}

    /// Construct from Var<ImageView<T>>.
    Expr(const Var<ImageView<T>> &image) noexcept
        : Expr{image.expression()} {}

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

    /// Size
    [[nodiscard]] auto size() const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<uint2>(f->call(
            Type::of<uint2>(), CallOp::TEXTURE_SIZE, {_expression}));
    }

    /// Self-pointer to unify the interfaces of the captured Image<T> and Expr<Image<T>>
    [[nodiscard]] auto operator->() const noexcept { return this; }
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
    explicit Expr(const RefExpr *expr) noexcept : _expression{expr} {}

    /// Construct from VolumeView. Will create texture binding.
    Expr(VolumeView<T> volume) noexcept
        : _expression{detail::FunctionBuilder::current()->texture_binding(
              Type::of<Volume<T>>(), volume.handle(), volume.level())} {}

    /// Construct from Volume. Will create texture binding.
    Expr(const Volume<T> &volume) noexcept
        : Expr{VolumeView{volume}} {}

    /// Construct from Var<Volume<T>>.
    Expr(const Var<Volume<T>> &volume) noexcept
        : Expr{volume.expression()} {}

    /// Construct from Var<VolumeView<T>>.
    Expr(const Var<VolumeView<T>> &volume) noexcept
        : Expr{volume.expression()} {}

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

    [[nodiscard]] auto size() const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<uint3>(f->call(
            Type::of<uint3>(), CallOp::TEXTURE_SIZE, {_expression}));
    }

    /// Self pointer to unify the interfaces of the captured Volume<T> and Expr<Volume<T>>
    [[nodiscard]] auto operator->() const noexcept { return this; }
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
    bool _is_typed;
    bool _is_uniform;

public:
    /// Construct from array RefExpr and index Expression
    BindlessBuffer(const RefExpr *array, const Expression *index, bool is_typed, bool is_uniform) noexcept
        : _array{array}, _index{index}, _is_typed{is_typed}, _is_uniform{is_uniform} {}

    /// Read at index i
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&i) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<T>(
            f->call(
                Type::of<T>(), _is_typed ? (_is_uniform ? CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_READ : CallOp::TYPED_BINDLESS_BUFFER_READ) : (_is_uniform ? CallOp::UNIFORM_BINDLESS_BUFFER_READ : CallOp::BINDLESS_BUFFER_READ),
                {_array, _index, detail::extract_expression(std::forward<I>(i))}));
    }

    /// write at index i
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write(I &&i, V &&value) const noexcept {
        detail::FunctionBuilder::current()->call(
            _is_typed ? (_is_uniform ? CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_WRITE : CallOp::TYPED_BINDLESS_BUFFER_WRITE) : (_is_uniform ? CallOp::UNIFORM_BINDLESS_BUFFER_WRITE : CallOp::BINDLESS_BUFFER_WRITE),
            {_array, _index,
             detail::extract_expression(std::forward<I>(i)),
             detail::extract_expression(std::forward<V>(value))});
    }

    /// Number of logical T elements in the bound buffer view.
    [[nodiscard]] auto size() const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto stride = def(static_cast<uint>(sizeof(T)));
        return def<uint>(f->call(
            Type::of<uint>(),
            _is_typed ?
                (_is_uniform ?
                     CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_SIZE :
                     CallOp::TYPED_BINDLESS_BUFFER_SIZE) :
                (_is_uniform ?
                     CallOp::UNIFORM_BINDLESS_BUFFER_SIZE :
                     CallOp::BINDLESS_BUFFER_SIZE),
            {_array, _index, stride.expression()}));
    }

    /// Device address of the first byte in the bound logical buffer view.
    [[nodiscard]] auto device_address() const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<uint64_t>(f->call(
            Type::of<uint64_t>(),
            _is_typed ?
                (_is_uniform ?
                     CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_ADDRESS :
                     CallOp::TYPED_BINDLESS_BUFFER_ADDRESS) :
                (_is_uniform ?
                     CallOp::UNIFORM_BINDLESS_BUFFER_ADDRESS :
                     CallOp::BINDLESS_BUFFER_ADDRESS),
            {_array, _index}));
    }

    /// Self-pointer to unify the interfaces with Expr<Buffer<T>>
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

class LUISA_DSL_API BindlessByteBuffer {

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};
    bool _is_typed;
    bool _is_uniform;

public:
    BindlessByteBuffer(const RefExpr *array, const Expression *index, bool is_typed, bool is_uniform) noexcept
        : _array{array}, _index{index}, _is_typed{is_typed}, _is_uniform{is_uniform} {}

    template<typename T, typename I>
        requires is_valid_buffer_element_v<T> && is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&offset) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<T>(
            f->call(
                Type::of<T>(), _is_typed ? (_is_uniform ? CallOp::TYPED_UNIFORM_BINDLESS_BYTE_BUFFER_READ : CallOp::TYPED_BINDLESS_BYTE_BUFFER_READ) : (_is_uniform ? CallOp::UNIFORM_BINDLESS_BYTE_BUFFER_READ : CallOp::BINDLESS_BYTE_BUFFER_READ),
                {_array, _index, detail::extract_expression(std::forward<I>(offset))}));
    }

    /// Exact logical byte size of the bound buffer view.
    [[nodiscard]] auto size() const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto byte_stride = def(1u);
        return def<uint>(f->call(
            Type::of<uint>(),
            _is_typed ?
                (_is_uniform ?
                     CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_SIZE :
                     CallOp::TYPED_BINDLESS_BUFFER_SIZE) :
                (_is_uniform ?
                     CallOp::UNIFORM_BINDLESS_BUFFER_SIZE :
                     CallOp::BINDLESS_BUFFER_SIZE),
            {_array, _index, byte_stride.expression()}));
    }

    /// Device address of the first byte in the bound logical byte-buffer view.
    [[nodiscard]] auto device_address() const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<uint64_t>(f->call(
            Type::of<uint64_t>(),
            _is_typed ?
                (_is_uniform ?
                     CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_ADDRESS :
                     CallOp::TYPED_BINDLESS_BUFFER_ADDRESS) :
                (_is_uniform ?
                     CallOp::UNIFORM_BINDLESS_BUFFER_ADDRESS :
                     CallOp::BINDLESS_BUFFER_ADDRESS),
            {_array, _index}));
    }

    /// Self-pointer to unify the interfaces with Expr<Buffer<T>>
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

/// Class of bindless 2D texture
class LUISA_DSL_API BindlessTexture2D {

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};
    bool _is_typed;
    bool _is_uniform;

public:
    /// Construct from array RefExpr and index Expression
    BindlessTexture2D(const RefExpr *array, const Expression *index, bool is_typed, bool is_uniform) noexcept
        : _array{array}, _index{index}, _is_typed{is_typed}, _is_uniform{is_uniform} {}
    /// Sample at (u, v)
    [[nodiscard]] Var<float4> sample(Expr<float2> uv) const noexcept;
    /// Sample at (u, v) at mip level
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float> mip) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy, mip-level offset, mip-level clamp
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy, Expr<float> min_mip) const noexcept;
    /// Sample at (u, v)
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, SamplerFilter filter, SamplerAddress address) const noexcept;
    /// Sample at (u, v) at mip level
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float> mip, SamplerFilter filter, SamplerAddress address) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy, SamplerFilter filter, SamplerAddress address) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy, mip-level offset, mip-level clamp
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy, Expr<float> min_mip, SamplerFilter filter, SamplerAddress address) const noexcept;
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
            Type::of<float4>(),
            _is_typed ? (_is_uniform ? CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_READ_LEVEL : CallOp::TYPED_BINDLESS_TEXTURE2D_READ_LEVEL) : (_is_uniform ? CallOp::UNIFORM_BINDLESS_TEXTURE2D_READ_LEVEL : CallOp::BINDLESS_TEXTURE2D_READ_LEVEL),
            {_array, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))}));
    }

    /// Self-pointer to unify the interfaces with Expr<Texture2D>
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

/// Class of bindless 3D texture
class LUISA_DSL_API BindlessTexture3D {

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};
    bool _is_typed;
    bool _is_uniform;

public:
    /// Construct from array RefExpr and index Expression
    BindlessTexture3D(const RefExpr *array, const Expression *index, bool is_typed, bool is_uniform) noexcept
        : _array{array}, _index{index}, _is_typed{is_typed}, _is_uniform{is_uniform} {}
    /// Sample at (u, v, w)
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw) const noexcept;
    /// Sample at (u, v, w) at mip level
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float> mip) const noexcept;
    /// Sample at (u, v, w) with grad dpdx, dpdy
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy, mip-level offset, mip-level clamp
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy, Expr<float> min_mip) const noexcept;
    /// Sample at (u, v, w)
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, SamplerFilter filter, SamplerAddress address) const noexcept;
    /// Sample at (u, v, w) at mip level
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float> mip, SamplerFilter filter, SamplerAddress address) const noexcept;
    /// Sample at (u, v, w) with grad dpdx, dpdy
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy, SamplerFilter filter, SamplerAddress address) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy, mip-level offset, mip-level clamp
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy, Expr<float> min_mip, SamplerFilter filter, SamplerAddress address) const noexcept;
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
            Type::of<float4>(),
            _is_typed ? (_is_uniform ? CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_READ_LEVEL : CallOp::TYPED_BINDLESS_TEXTURE3D_READ_LEVEL) : (_is_uniform ? CallOp::UNIFORM_BINDLESS_TEXTURE3D_READ_LEVEL : CallOp::BINDLESS_TEXTURE3D_READ_LEVEL),
            {_array, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))}));
    }

    /// Self-pointer to unify the interfaces with Expr<Texture3D>
    [[nodiscard]] auto operator->() const noexcept { return this; }
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
    [[nodiscard]] auto tex2d(I &&index, bool is_typed = false, bool is_uniform_index = false) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessTexture2D{_expression, i.expression(), is_typed, is_uniform_index};
    }

    /// Get 3D texture at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex3d(I &&index, bool is_typed = false, bool is_uniform_index = false) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessTexture3D{_expression, i.expression(), is_typed, is_uniform_index};
    }

    /// Get buffer at index
    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto buffer(I &&index, bool is_typed = false, bool is_uniform_index = false) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessBuffer<T>{_expression, i.expression(), is_typed, is_uniform_index};
    }

    /// Get byte-address buffer at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto byte_buffer(I &&index, bool is_typed = false, bool is_uniform_index = false) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessByteBuffer{_expression, i.expression(), is_typed, is_uniform_index};
    }

    /// Self-pointer to unify the interfaces of the captured BindlessArray and Expr<BindlessArray>
    [[nodiscard]] auto *operator->() const noexcept { return this; }
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
        Expr<T>{_buffer}.write(std::forward<I>(index),
                               std::forward<V>(value));
    }
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto volatile_read(I &&index) const noexcept {
        return Expr<T>{_buffer}.volatile_read(std::forward<I>(index));
    }
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void volatile_write(I &&index, V &&value) const noexcept {
        Expr<T>{_buffer}.volatile_write(std::forward<I>(index),
                                        std::forward<V>(value));
    }
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&index) const noexcept {
        return Expr<T>{_buffer}.atomic(std::forward<I>(index));
    }

    [[nodiscard]] Expr<uint64_t> device_address() const noexcept {
        return Expr<T>{_buffer}.device_address();
    }
};

class ByteBufferExprProxy {

private:
    ByteBuffer _buffer;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(ByteBufferExprProxy)

public:
    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        return Expr<ByteBuffer>{_buffer}.template read<T, I>(std::forward<I>(index));
    }
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write(I &&index, V &&value) const noexcept {
        Expr<ByteBuffer>{_buffer}.write(std::forward<I>(index),
                                        std::forward<V>(value));
    }
    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto volatile_read(I &&index) const noexcept {
        return Expr<ByteBuffer>{_buffer}.template volatile_read<T, I>(std::forward<I>(index));
    }
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void volatile_write(I &&index, V &&value) const noexcept {
        Expr<ByteBuffer>{_buffer}.volatile_write(std::forward<I>(index),
                                                 std::forward<V>(value));
    }
    [[nodiscard]] Expr<uint64_t> device_address() const noexcept {
        return Expr<ByteBuffer>{_buffer}.device_address();
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
    [[nodiscard]] auto size() const noexcept {
        return Expr<T>{_img}.size();
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
    [[nodiscard]] auto size() const noexcept {
        return Expr<T>{_img}.size();
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
    [[nodiscard]] auto tex2d(I &&index, bool is_typed = false, bool is_uniform_index = false) const noexcept {
        return Expr<BindlessArray>{_array}.tex2d(std::forward<I>(index), is_typed, is_uniform_index);
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex3d(I &&index, bool is_typed = false, bool is_uniform_index = false) const noexcept {
        return Expr<BindlessArray>{_array}.tex3d(std::forward<I>(index), is_typed, is_uniform_index);
    }

    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto buffer(I &&index, bool is_typed = false, bool is_uniform_index = false) const noexcept {
        return Expr<BindlessArray>{_array}.buffer<T>(std::forward<I>(index), is_typed, is_uniform_index);
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto byte_buffer(I &&index, bool is_typed = false, bool is_uniform_index = false) const noexcept {
        return Expr<BindlessArray>{_array}.byte_buffer(std::forward<I>(index), is_typed, is_uniform_index);
    }
};

}// namespace detail

template<typename T>
struct Var<Buffer<T>> : public Expr<Buffer<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Buffer<T>>{detail::FunctionBuilder::current()->buffer(Type::of<Buffer<T>>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<>
struct Var<ByteBuffer> : public Expr<ByteBuffer> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<ByteBuffer>{detail::FunctionBuilder::current()->buffer(Type::of<ByteBuffer>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<typename T>
struct Var<BufferView<T>> : public Expr<Buffer<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Buffer<T>>{detail::FunctionBuilder::current()->buffer(Type::of<Buffer<T>>())} {}
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
        : Expr<Image<T>>{detail::FunctionBuilder::current()->texture(Type::of<Image<T>>())} {}
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
        : Expr<Volume<T>>{detail::FunctionBuilder::current()->texture(Type::of<Volume<T>>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<>
struct Var<BindlessArray> : public Expr<BindlessArray> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<BindlessArray>{detail::FunctionBuilder::current()->bindless_array()} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

template<typename T>
using BufferVar = Var<Buffer<T>>;
using ByteBufferVar = Var<ByteBuffer>;

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
using BufferShort = BufferVar<short>;
using BufferShort2 = BufferVar<short2>;
using BufferShort3 = BufferVar<short3>;
using BufferShort4 = BufferVar<short4>;
using BufferUShort = BufferVar<ushort>;
using BufferUShort2 = BufferVar<ushort2>;
using BufferUShort3 = BufferVar<ushort3>;
using BufferUShort4 = BufferVar<ushort4>;
using BufferSLong = BufferVar<slong>;
using BufferSLong2 = BufferVar<slong2>;
using BufferSLong3 = BufferVar<slong3>;
using BufferSLong4 = BufferVar<slong4>;
using BufferULong = BufferVar<ulong>;
using BufferULong2 = BufferVar<ulong2>;
using BufferULong3 = BufferVar<ulong3>;
using BufferULong4 = BufferVar<ulong4>;
using BufferHalf = BufferVar<half>;
using BufferHalf2 = BufferVar<half2>;
using BufferHalf3 = BufferVar<half3>;
using BufferHalf4 = BufferVar<half4>;

using BufferFloat2x2 = BufferVar<float2x2>;
using BufferFloat3x3 = BufferVar<float3x3>;
using BufferFloat4x4 = BufferVar<float4x4>;

using ImageInt = ImageVar<int>;
using ImageUInt = ImageVar<uint>;
using ImageFloat = ImageVar<float>;

using VolumeInt = VolumeVar<int>;
using VolumeUInt = VolumeVar<uint>;
using VolumeFloat = VolumeVar<float>;

inline namespace dsl {

template<typename T>
inline void pack_to(T &&x, Expr<Buffer<uint>> arr, Expr<uint> index) noexcept {
    Expr xx{std::forward<T>(x)};
    detail::FunctionBuilder::current()->call(
        CallOp::PACK,
        {xx.expression(), arr.expression(), index.expression()});
}

template<class T>
[[nodiscard]] inline auto unpack_from(Expr<Buffer<uint>> arr, Expr<uint> index) noexcept {
    using E = expr_value_t<T>;
    return def<E>(
        detail::FunctionBuilder::current()->call(
            Type::of<E>(), CallOp::UNPACK,
            {arr.expression(), index.expression()}));
}

template<typename OutType, typename InType>
inline CoopVector<OutType> cooperative_mat_mul_add(
    Expr<ByteBuffer> matrix_buffer,
    CoopMatrixRef const &matrix_offset,
    Expr<ByteBuffer> bias_buffer,
    CoopVectorRef const &vector_offset,
    Expr<CoopVector<InType>> input_vector) {
    CoopVector<OutType> var(matrix_offset.size().y);
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<OutType>(), var.size()),
            CallOp::COOPERATIVE_MUL_ADD,
            {matrix_buffer.expression(),
             matrix_offset.expression(),
             bias_buffer.expression(),
             vector_offset.expression(),
             input_vector.expression()}));
    return var;
}
template<typename OutType, typename InType>
inline CoopVector<OutType> bindless_cooperative_mat_mul_add(
    Expr<BindlessArray> bindless_array,
    Expr<uint> matrix_buffer,
    CoopMatrixRef const &matrix_offset,
    Expr<uint> bias_buffer,
    CoopVectorRef const &vector_offset,
    Expr<CoopVector<InType>> input_vector) {
    CoopVector<OutType> var(matrix_offset.size().y);
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<OutType>(), var.size()),
            CallOp::BINDLESS_COOPERATIVE_MUL_ADD,
            {bindless_array.expression(),
             matrix_buffer.expression(),
             matrix_offset.expression(),
             bias_buffer.expression(),
             vector_offset.expression(),
             input_vector.expression()}));
    return var;
}
template<typename OutType, typename InType>
inline CoopVector<OutType> typed_bindless_cooperative_mat_mul_add(
    Expr<BindlessArray> bindless_array,
    Expr<uint> matrix_buffer,
    CoopMatrixRef const &matrix_offset,
    Expr<uint> bias_buffer,
    CoopVectorRef const &vector_offset,
    Expr<CoopVector<InType>> input_vector) {
    CoopVector<OutType> var(matrix_offset.size().y);
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<OutType>(), var.size()),
            CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD,
            {bindless_array.expression(),
             matrix_buffer.expression(),
             matrix_offset.expression(),
             bias_buffer.expression(),
             vector_offset.expression(),
             input_vector.expression()}));
    return var;
}

template<typename OutType, typename InType>
inline CoopVector<OutType> cooperative_mat_mul(
    Expr<ByteBuffer> matrix_buffer,
    CoopMatrixRef const &matrix_offset,
    Expr<CoopVector<InType>> input_vector) {
    CoopVector<OutType> var(matrix_offset.size().y);
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<OutType>(), var.size()),
            CallOp::COOPERATIVE_MUL,
            {matrix_buffer.expression(),
             matrix_offset.expression(),
             input_vector.expression()}));
    return var;
}

template<typename OutType, typename InType>
inline CoopVector<OutType> bindless_cooperative_mat_mul(
    Expr<BindlessArray> bindless_array,
    Expr<uint> matrix_buffer,
    CoopMatrixRef const &matrix_offset,
    Expr<CoopVector<InType>> input_vector) {
    CoopVector<OutType> var(matrix_offset.size().y);
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<OutType>(), var.size()),
            CallOp::BINDLESS_COOPERATIVE_MUL,
            {bindless_array.expression(),
             matrix_buffer.expression(),
             matrix_offset.expression(),
             input_vector.expression()}));
    return var;
}

template<typename OutType, typename InType>
inline CoopVector<OutType> typed_bindless_cooperative_mat_mul(
    Expr<BindlessArray> bindless_array,
    Expr<uint> matrix_buffer,
    CoopMatrixRef const &matrix_offset,
    Expr<CoopVector<InType>> input_vector) {
    CoopVector<OutType> var(matrix_offset.size().y);
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<OutType>(), var.size()),
            CallOp::TYPED_BINDLESS_COOPERATIVE_MUL,
            {bindless_array.expression(),
             matrix_buffer.expression(),
             matrix_offset.expression(),
             input_vector.expression()}));
    return var;
}

template<typename VecElemType>
inline void cooperative_outer_product_accumulate(
    Expr<ByteBuffer> matrix_buffer,
    CoopMatrixRef const &matrix_offset,
    Expr<CoopVector<VecElemType>> input_vector1,
    Expr<CoopVector<VecElemType>> input_vector2) {
    detail::FunctionBuilder::current()->call(
        CallOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE,
        {matrix_buffer.expression(),
         matrix_offset.expression(),
         input_vector1.expression(),
         input_vector2.expression()});
}

template<typename VecElemType>
inline void cooperative_vector_accumulate(
    Expr<ByteBuffer> vector_buffer,
    CoopVectorRef const &vector_offset,
    Expr<CoopVector<VecElemType>> input_vector) {
    detail::FunctionBuilder::current()->call(
        CallOp::COOPERATIVE_VECTOR_ACCUMULATE,
        {vector_buffer.expression(),
         vector_offset.expression(),
         input_vector.expression()});
}

template<typename T>
inline CoopVector<T> cooperative_vector_load(
    Expr<ByteBuffer> buffer,
    CoopVectorRef const &offset) {
    CoopVector<T> var(static_cast<uint32_t>(offset.size()));
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<T>(), var.size()),
            CallOp::COOPERATIVE_VECTOR_LOAD,
            {buffer.expression(),
             offset.expression()}));
    return var;
}

template<typename T>
inline void cooperative_vector_store(
    Expr<ByteBuffer> buffer,
    CoopVectorRef const &offset,
    Expr<CoopVector<T>> value) {
    detail::FunctionBuilder::current()->call(
        CallOp::COOPERATIVE_VECTOR_STORE,
        {buffer.expression(),
         offset.expression(),
         value.expression()});
}

template<typename T>
inline CoopVector<T> bindless_cooperative_vector_load(
    Expr<BindlessArray> bindless_array,
    Expr<uint> buffer_handle,
    CoopVectorRef const &offset) {
    CoopVector<T> var(static_cast<uint32_t>(offset.size()));
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<T>(), var.size()),
            CallOp::BINDLESS_COOPERATIVE_VECTOR_LOAD,
            {bindless_array.expression(),
             buffer_handle.expression(),
             offset.expression()}));
    return var;
}

template<typename T>
inline CoopVector<T> typed_bindless_cooperative_vector_load(
    Expr<BindlessArray> bindless_array,
    Expr<uint> buffer_handle,
    CoopVectorRef const &offset) {
    CoopVector<T> var(static_cast<uint32_t>(offset.size()));
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<T>(), var.size()),
            CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD,
            {bindless_array.expression(),
             buffer_handle.expression(),
             offset.expression()}));
    return var;
}

template<typename T>
inline void bindless_cooperative_vector_store(
    Expr<BindlessArray> bindless_array,
    Expr<uint> buffer_handle,
    CoopVectorRef const &offset,
    Expr<CoopVector<T>> value) {
    detail::FunctionBuilder::current()->call(
        CallOp::BINDLESS_COOPERATIVE_VECTOR_STORE,
        {bindless_array.expression(),
         buffer_handle.expression(),
         offset.expression(),
         value.expression()});
}

template<typename T>
inline void typed_bindless_cooperative_vector_store(
    Expr<BindlessArray> bindless_array,
    Expr<uint> buffer_handle,
    CoopVectorRef const &offset,
    Expr<CoopVector<T>> value) {
    detail::FunctionBuilder::current()->call(
        CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE,
        {bindless_array.expression(),
         buffer_handle.expression(),
         offset.expression(),
         value.expression()});
}

template<typename T>
inline CoopVector<T> cooperative_vector_workgroup_load(
    const Shared<T> &shared_mem,
    Expr<uint32_t> index) {
    auto n = static_cast<uint32_t>(shared_mem.size());
    CoopVector<T> var(n);
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<T>(), n),
            CallOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD,
            {shared_mem.expression(),
             index.expression()}));
    return var;
}

template<typename T>
inline void cooperative_vector_workgroup_store(
    const Shared<T> &shared_mem,
    Expr<uint32_t> index,
    Expr<CoopVector<T>> value) {
    detail::FunctionBuilder::current()->call(
        CallOp::COOPERATIVE_VECTOR_WORKGROUP_STORE,
        {shared_mem.expression(),
         index.expression(),
         value.expression()});
}

template<typename T>
inline CoopVector<T> cooperative_vector_splat(Expr<T> scalar, uint32_t n) {
    CoopVector<T> var(n);
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<T>(), n),
            CallOp::COOPERATIVE_VECTOR_SPLAT,
            {scalar.expression()}));
    return var;
}

template<typename NewType, typename OldType>
inline CoopVector<NewType> cooperative_vector_cast(Expr<CoopVector<OldType>> v) {
    auto n = static_cast<uint32_t>(v.expression()->type()->dimension());
    CoopVector<NewType> var(n);
    detail::FunctionBuilder::current()->assign(
        var.expression(),
        detail::FunctionBuilder::current()->call(
            Type::cooperative_vector(Type::of<NewType>(), n),
            CallOp::COOPERATIVE_VECTOR_CAST,
            {v.expression()}));
    return var;
}

// Cooperative vector element-wise math operations.
// These use element-wise access + DSL math functions, which DXC compiles
// to appropriate SPIR-V GLSL.std.450 or native DX instructions.

template<typename T>
inline CoopVector<T> cooperative_vector_min(const CoopVector<T> &a, const CoopVector<T> &b) {
    auto n = static_cast<uint32_t>(a.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = min(a[i], b[i]);
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_max(const CoopVector<T> &a, const CoopVector<T> &b) {
    auto n = static_cast<uint32_t>(a.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = max(a[i], b[i]);
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_clamp(const CoopVector<T> &v, const CoopVector<T> &lo, const CoopVector<T> &hi) {
    auto n = static_cast<uint32_t>(v.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = clamp(v[i], lo[i], hi[i]);
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_exp(const CoopVector<T> &v) {
    auto n = static_cast<uint32_t>(v.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = exp(v[i]);
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_log(const CoopVector<T> &v) {
    auto n = static_cast<uint32_t>(v.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = log(v[i]);
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_tanh(const CoopVector<T> &v) {
    auto n = static_cast<uint32_t>(v.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = tanh(v[i]);
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_atan(const CoopVector<T> &v) {
    auto n = static_cast<uint32_t>(v.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = atan(v[i]);
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_fma(const CoopVector<T> &a, const CoopVector<T> &b, const CoopVector<T> &c) {
    auto n = static_cast<uint32_t>(a.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = fma(a[i], b[i], c[i]);
    }
    return result;
}

// Cooperative vector bit operations (integer types only).
// These use element-wise access + DSL bitwise operators.

template<typename T>
inline CoopVector<T> cooperative_vector_bitwise_and(const CoopVector<T> &a, const CoopVector<T> &b) {
    auto n = static_cast<uint32_t>(a.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = a[i] & b[i];
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_bitwise_or(const CoopVector<T> &a, const CoopVector<T> &b) {
    auto n = static_cast<uint32_t>(a.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = a[i] | b[i];
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_bitwise_xor(const CoopVector<T> &a, const CoopVector<T> &b) {
    auto n = static_cast<uint32_t>(a.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = a[i] ^ b[i];
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_bitwise_not(const CoopVector<T> &v) {
    auto n = static_cast<uint32_t>(v.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = ~v[i];
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_shift_left(const CoopVector<T> &v, Expr<uint32_t> bits) {
    auto n = static_cast<uint32_t>(v.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = v[i] << bits;
    }
    return result;
}

template<typename T>
inline CoopVector<T> cooperative_vector_shift_right(const CoopVector<T> &v, Expr<uint32_t> bits) {
    auto n = static_cast<uint32_t>(v.size());
    CoopVector<T> result(n);
    for (uint32_t i = 0u; i < n; ++i) {
        result[i] = v[i] >> bits;
    }
    return result;
}

}// namespace dsl

}// namespace luisa::compute
