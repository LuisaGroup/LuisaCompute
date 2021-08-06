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

namespace detail {

template<typename T>
struct Expr;

}

template<typename T>
struct Var;

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector2(detail::Expr<T> x, detail::Expr<T> y) noexcept;

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector3(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z) noexcept;

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z, detail::Expr<T> w) noexcept;

namespace detail {

#define LUISA_EXPR_COMMON()                                                      \
private:                                                                         \
    const Expression *_expression;                                               \
                                                                                 \
public:                                                                          \
    explicit Expr(const Expression *expr) noexcept : _expression{expr} {}        \
    [[nodiscard]] auto expression() const noexcept { return this->_expression; } \
    Expr(Expr &&another) noexcept = default;                                     \
    Expr(const Expr &another) noexcept = default;                                \
    void operator=(const Expr &rhs) noexcept {                                   \
        FunctionBuilder::current()->assign(                                      \
            AssignOp::ASSIGN,                                                    \
            this->expression(),                                                  \
            rhs.expression());                                                   \
    }                                                                            \
    void operator=(Expr &&rhs) noexcept {                                        \
        FunctionBuilder::current()->assign(                                      \
            AssignOp::ASSIGN,                                                    \
            this->expression(),                                                  \
            rhs.expression());                                                   \
    }

#define LUISA_EXPR_FROM_LITERAL(...)                     \
    template<typename U>                                 \
    requires std::same_as<U, __VA_ARGS__>                \
    Expr(U literal)                                      \
    noexcept : Expr{FunctionBuilder::current()->literal( \
        Type::of<U>(), literal)} {}

template<typename T>
struct ExprEnableArithmeticAssign;

template<typename T>
struct ExprEnableStaticCast;

template<typename T>
struct ExprEnableBitwiseCast;

template<typename T>
struct ExprEnableAccessOp;

template<typename T>
struct Expr
    : ExprEnableStaticCast<T>,
      ExprEnableBitwiseCast<T>,
      ExprEnableArithmeticAssign<T> {
    static_assert(concepts::basic<T>);
    using value_type = T;
    LUISA_EXPR_COMMON()
    LUISA_EXPR_FROM_LITERAL(T)
};

template<typename T, size_t N>
struct Expr<std::array<T, N>>
    : ExprEnableAccessOp<std::array<T, N>> {
    LUISA_EXPR_COMMON()
};

template<size_t N>
struct Expr<Matrix<N>>
    : ExprEnableAccessOp<Matrix<N>>,
      ExprEnableArithmeticAssign<Matrix<N>> {
    LUISA_EXPR_COMMON()
    LUISA_EXPR_FROM_LITERAL(Matrix<N>)
};

template<typename... T>
struct Expr<std::tuple<T...>> {
    LUISA_EXPR_COMMON()
    template<size_t i>
    [[nodiscard]] auto member() const noexcept {
        using M = std::tuple_element_t<i, std::tuple<T...>>;
        return Expr<M>{FunctionBuilder::current()->member(
            Type::of<M>(), this->expression(), i)};
    };
};

template<typename T>
struct Expr<Vector<T, 2>>
    : ExprEnableStaticCast<Vector<T, 2>>,
      ExprEnableBitwiseCast<Vector<T, 2>>,
      ExprEnableAccessOp<Vector<T, 2>>,
      ExprEnableArithmeticAssign<Vector<T, 3>> {
    LUISA_EXPR_COMMON()
    LUISA_EXPR_FROM_LITERAL(Vector<T, 2>)
    Expr<T> x{FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
#include <dsl/swizzle_2.inl.h>
};

template<typename T>
struct Expr<Vector<T, 3>>
    : ExprEnableStaticCast<Vector<T, 3>>,
      ExprEnableBitwiseCast<Vector<T, 3>>,
      ExprEnableAccessOp<Vector<T, 3>>,
      ExprEnableArithmeticAssign<Vector<T, 3>> {
    LUISA_EXPR_COMMON()
    LUISA_EXPR_FROM_LITERAL(Vector<T, 3>)
    Expr<T> x{FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Expr<T> z{FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
#include <dsl/swizzle_3.inl.h>
};

template<typename T>
struct Expr<Vector<T, 4>>
    : ExprEnableStaticCast<Vector<T, 4>>,
      ExprEnableBitwiseCast<Vector<T, 4>>,
      ExprEnableAccessOp<Vector<T, 4>>,
      ExprEnableArithmeticAssign<Vector<T, 4>> {
    LUISA_EXPR_COMMON()
    LUISA_EXPR_FROM_LITERAL(Vector<T, 4>)
    Expr<T> x{FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Expr<T> z{FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
    Expr<T> w{FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x3u)};
#include <dsl/swizzle_4.inl.h>
};

template<typename T>
struct ExprEnableArithmeticAssign {
#define LUISA_EXPR_ASSIGN_OP(op, op_concept_name, op_tag_name) \
    template<typename U>                                       \
    requires concepts::op_concept_name<T, U>                   \
    void operator op(Expr<U> rhs) &noexcept {                  \
        FunctionBuilder::current()->assign(                    \
            AssignOp::op_tag_name,                             \
            static_cast<const Expr<T> *>(this)->expression(),  \
            rhs.expression());                                 \
    }                                                          \
    template<typename U>                                       \
    void operator op(U &&rhs) &noexcept {                      \
        return this->operator op(Expr{std::forward<U>(rhs)});  \
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
struct ExprEnableStaticCast {
    template<typename Dest>
    [[nodiscard]] auto cast() const noexcept {
        static_assert(concepts::static_convertible<T, Dest>);
        return Expr<Dest>{FunctionBuilder::current()->cast(
            Type::of<Dest>(),
            CastOp::STATIC,
            static_cast<const Expr<T> *>(this)->expression())};
    }
};

template<typename T>
struct ExprEnableBitwiseCast {
    template<typename Dest>
    [[nodiscard]] auto as() const noexcept {
        static_assert(concepts::bitwise_convertible<T, Dest>);
        return Expr<Dest>{FunctionBuilder::current()->cast(
            Type::of<Dest>(),
            CastOp::BITWISE,
            static_cast<const Expr<T> *>(this)->expression())};
    }
};

template<typename T>
struct ExprEnableAccessOp {
    template<concepts::integral I>
    [[nodiscard]] auto operator[](Expr<I> index) const noexcept {
        using Elem = std::remove_cvref_t<decltype(std::declval<T>()[0])>;
        return Expr<Elem>{FunctionBuilder::current()->access(
            Type::of<Elem>(),
            static_cast<const Expr<T> *>(this)->expression(),
            index.expression())};
    }
    template<concepts::integral I>
    [[nodiscard]] auto operator[](I index) const noexcept {
        return (*this)[Expr<I>{index}];
    }
};

#undef LUISA_EXPR_COMMON
#undef LUISA_EXPR_FROM_LITERAL

template<typename>
struct BufferExprAsAtomic {};

template<typename T>
struct Expr<Buffer<T>>
    : BufferExprAsAtomic<T> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}
    explicit Expr(BufferView<T> buffer) noexcept
        : _expression{FunctionBuilder::current()->buffer_binding(
            Type::of<Buffer<T>>(),
            buffer.handle(), buffer.offset_bytes())} {}

    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }

    [[nodiscard]] auto operator[](Expr<uint> i) const noexcept {
        return Expr<T>{FunctionBuilder::current()->access(
            Type::of<T>(), _expression, i.expression())};
    };

    [[nodiscard]] auto operator[](Expr<int> i) const noexcept {
        return Expr<T>{FunctionBuilder::current()->access(
            Type::of<T>(), _expression, i.expression())};
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
        FunctionBuilder::current()->call(CallOp::ATOMIC_STORE, {this->_expression, value.expression()});
    }

#define LUISA_ATOMIC_NODISCARD                                           \
    [[nodiscard(                                                         \
        "Return values from atomic operations with side effects should " \
        "not be discarded. Enclose this expression with void_().")]]

    LUISA_ATOMIC_NODISCARD auto load() const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_LOAD,
            {this->_expression});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto exchange(Expr<T> desired) const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_EXCHANGE,
            {this->_expression, desired.expression()});
        return Expr<T>{expr};
    }

    // stores old == compare ? val : old, returns old
    LUISA_ATOMIC_NODISCARD auto compare_exchange(Expr<T> expected, Expr<T> desired) const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_COMPARE_EXCHANGE,
            {this->_expression, expected.expression(), desired.expression()});
        return Expr<T>{expr};
    }

    LUISA_ATOMIC_NODISCARD auto fetch_add(Expr<T> val) const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_ADD,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_sub(Expr<T> val) const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_SUB,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_and(Expr<T> val) const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_AND,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_or(Expr<T> val) const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_OR,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_xor(Expr<T> val) const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_XOR,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_min(Expr<T> val) const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MIN,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

    LUISA_ATOMIC_NODISCARD auto fetch_max(Expr<T> val) const noexcept {
        auto expr = FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATOMIC_FETCH_MAX,
            {this->_expression, val.expression()});
        return Expr<T>{expr};
    };

#undef LUISA_ATOMIC_NODISCARD
};

template<>
struct BufferExprAsAtomic<int> {
    [[nodiscard]] auto atomic(Expr<int> i) const noexcept {
        return AtomicRef<int>{FunctionBuilder::current()->access(
            Type::of<int>(),
            static_cast<const Expr<Buffer<int>> *>(this)->expression(),
            i.expression())};
    }
    [[nodiscard]] auto atomic(Expr<uint> i) const noexcept {
        return AtomicRef<int>{FunctionBuilder::current()->access(
            Type::of<int>(),
            static_cast<const Expr<Buffer<int>> *>(this)->expression(),
            i.expression())};
    }
};

template<>
struct BufferExprAsAtomic<uint> {
    [[nodiscard]] auto atomic(Expr<int> i) const noexcept {
        return AtomicRef<uint>{FunctionBuilder::current()->access(
            Type::of<uint>(),
            static_cast<const Expr<Buffer<uint>> *>(this)->expression(),
            i.expression())};
    }
    [[nodiscard]] auto atomic(Expr<uint> i) const noexcept {
        return AtomicRef<uint>{FunctionBuilder::current()->access(
            Type::of<uint>(),
            static_cast<const Expr<Buffer<uint>> *>(this)->expression(),
            i.expression())};
    }
};

template<typename T>
struct Expr<Image<T>> {

private:
    const RefExpr *_expression{nullptr};
    const Expression *_offset{nullptr};

    [[nodiscard]] auto _offset_uv(const Expression *uv) const noexcept -> const Expression * {
        if (_offset == nullptr) { return uv; }
        auto f = FunctionBuilder::current();
        return f->binary(Type::of<uint2>(), BinaryOp::ADD, uv, _offset);
    }

public:
    explicit Expr(const RefExpr *expr, const Expression *offset) noexcept
        : _expression{expr}, _offset{offset} {}
    explicit Expr(ImageView<T> image) noexcept
        : _expression{FunctionBuilder::current()->texture_binding(
            Type::of<Image<T>>(), image.handle())},
          _offset{any(image.offset())
                      ? FunctionBuilder::current()->literal(Type::of<uint2>(), image.offset())
                      : nullptr} {}

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }

    [[nodiscard]] auto read(Expr<uint2> uv) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<Vector<T, 4>>{f->call(
            Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
            {_expression, _offset_uv(uv.expression())})};
    };

    void write(Expr<uint2> uv, Expr<Vector<T, 4>> value) const noexcept {
        FunctionBuilder::current()->call(
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
        auto f = FunctionBuilder::current();
        return f->binary(Type::of<uint3>(), BinaryOp::ADD, uvw, _offset);
    }

public:
    explicit Expr(const RefExpr *expr, const Expression *offset) noexcept
        : _expression{expr}, _offset{offset} {}
    explicit Expr(VolumeView<T> volume) noexcept
        : _expression{FunctionBuilder::current()->texture_binding(
            Type::of<Volume<T>>(), volume.handle())},
          _offset{any(volume.offset())
                      ? FunctionBuilder::current()->literal(Type::of<uint3>(), volume.offset())
                      : nullptr} {}

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }

    [[nodiscard]] auto read(Expr<uint3> uvw) const noexcept {
        return Expr<Vector<T, 4>>{FunctionBuilder::current()->call(
            Type::of<Vector<T, 4>>(), CallOp::TEXTURE_READ,
            {_expression, _offset_uvw(uvw.expression())})};
    };

    void write(Expr<uint3> uvw, Expr<Vector<T, 4>> value) const noexcept {
        FunctionBuilder::current()->call(
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

    [[nodiscard]] auto read(Expr<int> i) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<T>{f->call(
            Type::of<T>(), CallOp::BUFFER_HEAP_READ,
            {_heap, _index, i.expression()})};
    }

    [[nodiscard]] auto read(Expr<uint> i) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<T>{f->call(
            Type::of<T>(), CallOp::BUFFER_HEAP_READ,
            {_heap, _index, i.expression()})};
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
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D,
            {_heap, _index, uv.expression()})};
    }

    [[nodiscard]] auto sample(Expr<float2> uv, Expr<float> mip) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D_LEVEL,
            {_heap, _index, uv.expression(), mip.expression()})};
    }

    [[nodiscard]] auto sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D_GRAD,
            {_heap, _index, uv.expression(), dpdx.expression(), dpdy.expression()})};
    }

    [[nodiscard]] auto size() const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<uint2>{f->call(
            Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D,
            {_heap, _index})};
    }

    [[nodiscard]] auto size(Expr<int> level) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<uint2>{f->call(
            Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D_LEVEL,
            {_heap, _index, level.expression()})};
    }

    [[nodiscard]] auto size(Expr<uint> level) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<uint2>{f->call(
            Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D_LEVEL,
            {_heap, _index, level.expression()})};
    }

    [[nodiscard]] auto read(Expr<uint2> coord) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ2D,
            {_heap, _index, coord.expression()})};
    }

    [[nodiscard]] auto read(Expr<uint2> coord, Expr<int> level) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ2D_LEVEL,
            {_heap, _index, coord.expression(), level.expression()})};
    }

    [[nodiscard]] auto read(Expr<uint2> coord, Expr<uint> level) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ2D_LEVEL,
            {_heap, _index, coord.expression(), level.expression()})};
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
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D,
            {_heap, _index, uvw.expression()})};
    }

    [[nodiscard]] auto sample(Expr<float3> uvw, Expr<float> mip) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D_LEVEL,
            {_heap, _index, uvw.expression(), mip.expression()})};
    }

    [[nodiscard]] auto sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D_GRAD,
            {_heap, _index, uvw.expression(), dpdx.expression(), dpdy.expression()})};
    }

    [[nodiscard]] auto size() const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<uint3>{f->call(
            Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D,
            {_heap, _index})};
    }

    [[nodiscard]] auto size(Expr<int> level) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<uint3>{f->call(
            Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D_LEVEL,
            {_heap, _index, level.expression()})};
    }

    [[nodiscard]] auto size(Expr<uint> level) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<uint3>{f->call(
            Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D_LEVEL,
            {_heap, _index, level.expression()})};
    }

    [[nodiscard]] auto read(Expr<uint3> coord) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ3D,
            {_heap, _index, coord.expression()})};
    }

    [[nodiscard]] auto read(Expr<uint3> coord, Expr<int> level) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ3D_LEVEL,
            {_heap, _index, coord.expression(), level.expression()})};
    }

    [[nodiscard]] auto read(Expr<uint3> coord, Expr<uint> level) const noexcept {
        auto f = FunctionBuilder::current();
        return Expr<float4>{f->call(
            Type::of<float4>(), CallOp::TEXTURE_HEAP_READ3D_LEVEL,
            {_heap, _index, coord.expression(), level.expression()})};
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
        : _expression{FunctionBuilder::current()->heap_binding(heap.handle())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto tex2d(Expr<int> index) const noexcept { return TextureRef2D{_expression, index.expression()}; }
    [[nodiscard]] auto tex2d(Expr<uint> index) const noexcept { return TextureRef2D{_expression, index.expression()}; }
    [[nodiscard]] auto tex3d(Expr<int> index) const noexcept { return TextureRef3D{_expression, index.expression()}; }
    [[nodiscard]] auto tex3d(Expr<uint> index) const noexcept { return TextureRef3D{_expression, index.expression()}; }

    template<typename T>
    [[nodiscard]] auto buffer(Expr<int> index) const noexcept { return BufferRef<T>{_expression, index.expression()}; }

    template<typename T>
    [[nodiscard]] auto buffer(Expr<uint> index) const noexcept { return BufferRef<T>{_expression, index.expression()}; }
};

// deduction guides
template<typename T>
Expr(Expr<T>) -> Expr<T>;

template<typename T>
Expr(Var<T>) -> Expr<T>;

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

template<typename T>
[[nodiscard]] inline const Expression *extract_expression(T &&v) noexcept {
    Expr expr{std::forward<T>(v)};
    return expr.expression();
}

template<typename T>
struct expr_value_impl {
    using type = T;
};

template<typename T>
struct expr_value_impl<Expr<T>> {
    using type = T;
};

template<typename T>
struct expr_value_impl<Var<T>> {
    using type = T;
};

template<typename T>
using expr_value = expr_value_impl<std::remove_cvref_t<T>>;

template<typename T>
using expr_value_t = typename expr_value<T>::type;

template<typename T>
struct is_expr : std::false_type {};

template<typename T>
struct is_expr<Expr<T>> : std::true_type {};

template<typename T>
struct is_expr<Var<T>> : std::true_type {};

template<typename T>
constexpr auto is_expr_v = is_expr<T>::value;

}// namespace detail

template<typename I>
detail::TextureRef2D Heap::tex2d(I &&index) const noexcept {
    return detail::Expr<Heap>{*this}.tex2d(std::forward<I>(index));
}

template<typename I>
detail::TextureRef2D Heap::tex3d(I &&index) const noexcept {
    return detail::Expr<Heap>{*this}.tex3d(std::forward<I>(index));
}

template<typename T, typename I>
detail::BufferRef<T> Heap::buffer(I &&index) const noexcept {
    return detail::Expr<Heap>{*this}.buffer<T>(std::forward<I>(index));
}

}// namespace luisa::compute

#define LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(op, op_concept, op_tag)                            \
    template<luisa::concepts::op_concept T>                                                \
    [[nodiscard]] inline auto operator op(luisa::compute::detail::Expr<T> expr) noexcept { \
        using R = std::remove_cvref_t<decltype(op std::declval<T>())>;                     \
        return luisa::compute::detail::Expr<R>{                                            \
            luisa::compute::detail::FunctionBuilder::current()->unary(                     \
                luisa::compute::Type::of<R>(),                                             \
                luisa::compute::UnaryOp::op_tag,                                           \
                expr.expression())};                                                       \
    }
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(+, operator_plus, PLUS)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(-, operator_minus, MINUS)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(!, operator_not, NOT)
LUISA_MAKE_GLOBAL_EXPR_UNARY_OP(~, operator_bit_not, BIT_NOT)
#undef LUISA_MAKE_GLOBAL_EXPR_UNARY_OP

#define LUISA_MAKE_GLOBAL_EXPR_BINARY_OP(op, op_concept_name, op_tag_name)                                                         \
    template<typename Lhs, typename Rhs>                                                                                           \
    requires luisa::concepts::op_concept_name<Lhs, Rhs>                                                                            \
    [[nodiscard]] inline auto operator op(luisa::compute::detail::Expr<Lhs> lhs, luisa::compute::detail::Expr<Rhs> rhs) noexcept { \
        using R = std::remove_cvref_t<decltype(std::declval<Lhs>() op std::declval<Rhs>())>;                                       \
        return luisa::compute::detail::Expr<R>{luisa::compute::detail::FunctionBuilder::current()->binary(                         \
            luisa::compute::Type::of<R>(),                                                                                         \
            luisa::compute::BinaryOp::op_tag_name, lhs.expression(), rhs.expression())};                                           \
    }                                                                                                                              \
    template<typename Lhs, typename Rhs>                                                                                           \
    requires luisa::concepts::basic<std::remove_cvref_t<Rhs>>                                                                      \
    [[nodiscard]] inline auto operator op(luisa::compute::detail::Expr<Lhs> lhs, Rhs &&rhs) noexcept {                             \
        return lhs op luisa::compute::detail::Expr{std::forward<Rhs>(rhs)};                                                        \
    }                                                                                                                              \
    template<typename Lhs, typename Rhs>                                                                                           \
    requires luisa::concepts::basic<std::remove_cvref_t<Lhs>>                                                                      \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, luisa::compute::detail::Expr<Rhs> rhs) noexcept {                             \
        return luisa::compute::detail::Expr{std::forward<Lhs>(lhs)} op rhs;                                                        \
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
