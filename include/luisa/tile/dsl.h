#pragma once

#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/ir.h>

namespace luisa::compute::tile {

class Axis;
class Kernel;
class Nest;
class NestRange;
template<typename T>
class Scalar;
template<typename T>
class ElementRef;
template<typename T>
class Buffer;

enum class BufferAccess : uint8_t {
    READ,
    WRITE,
    READ_WRITE
};

namespace exec {

enum class Scope : uint8_t {
    AUTOMATIC,
    DEVICE,
    GROUP,
    SUBGROUP,
    WORKER,
    VECTOR
};

}// namespace exec

struct PipelinePolicy {
    uint32_t stages{0u};
    uint32_t initiation_interval{1u};
};

namespace detail {

struct ValueSlot;
struct KernelStorage;
struct ScopeStorage;

class LUISA_TILE_API ValueHandle final {

private:
    luisa::shared_ptr<ValueSlot> _slot;

    explicit ValueHandle(luisa::shared_ptr<ValueSlot> slot) noexcept;
    void _assign(Value *value) noexcept;

    friend ValueHandle make_constant(ScalarType type, Attribute value) noexcept;
    friend ValueHandle make_elementwise_operation(ElementwiseOp op,
                                                  luisa::span<const ValueHandle> operands,
                                                  ScalarType result_type) noexcept;
    friend ValueHandle load_view(Value *view,
                                 luisa::span<const ValueHandle> indices) noexcept;
    friend luisa::vector<ValueHandle> nest_indices(
        const Nest &nest,
        const IndexSpace &space) noexcept;
    friend class ::luisa::compute::tile::Nest;
    template<typename T>
    friend class ::luisa::compute::tile::Buffer;
    template<typename T>
    friend class ::luisa::compute::tile::Scalar;

public:
    ValueHandle() noexcept = default;
    ValueHandle(const ValueHandle &other) noexcept;
    ValueHandle(ValueHandle &&other) noexcept;
    ValueHandle &operator=(const ValueHandle &other) noexcept;
    ValueHandle &operator=(ValueHandle &&other) noexcept;
    ~ValueHandle() noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] Value *value() const noexcept;
};

[[nodiscard]] LUISA_TILE_API ValueHandle make_constant(ScalarType type, Attribute value) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle make_elementwise_operation(
    ElementwiseOp op,
    luisa::span<const ValueHandle> operands,
    ScalarType result_type) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle load_view(
    Value *view,
    luisa::span<const ValueHandle> indices) noexcept;
LUISA_TILE_API void store_view(
    Value *view,
    luisa::span<const ValueHandle> indices,
    const ValueHandle &value) noexcept;

class LUISA_TILE_API CaptureGuard final {

private:
    Kernel *_kernel{nullptr};

public:
    explicit CaptureGuard(Kernel &kernel) noexcept;
    CaptureGuard(CaptureGuard &&) noexcept = delete;
    CaptureGuard(const CaptureGuard &) noexcept = delete;
    CaptureGuard &operator=(CaptureGuard &&) noexcept = delete;
    CaptureGuard &operator=(const CaptureGuard &) noexcept = delete;
    ~CaptureGuard() noexcept;
};

[[nodiscard]] LUISA_TILE_API Axis create_axis(luisa::string_view name, Extent extent) noexcept;
[[nodiscard]] LUISA_TILE_API IndexSpace make_shape(luisa::span<const Axis> axes) noexcept;
[[nodiscard]] LUISA_TILE_API Value *declare_view(
    luisa::string_view name,
    ScalarType element_type,
    const IndexSpace &space,
    BufferAccess access) noexcept;
[[nodiscard]] LUISA_TILE_API luisa::vector<ValueHandle> nest_indices(
    const Nest &nest,
    const IndexSpace &space) noexcept;
[[nodiscard]] LUISA_TILE_API NestRange make_range(
    const Nest *parent,
    OperationKind kind,
    IndexSpace domain,
    exec::Scope scope,
    PipelinePolicy policy) noexcept;

}// namespace detail

template<typename T>
struct scalar_type;

#define LUISA_TILE_DEFINE_SCALAR_TYPE(cpp_type, tile_type)   \
    template<>                                               \
    struct scalar_type<cpp_type> {                           \
        static constexpr auto value = ScalarType::tile_type; \
    }

LUISA_TILE_DEFINE_SCALAR_TYPE(bool, BOOL);
LUISA_TILE_DEFINE_SCALAR_TYPE(int8_t, INT8);
LUISA_TILE_DEFINE_SCALAR_TYPE(uint8_t, UINT8);
LUISA_TILE_DEFINE_SCALAR_TYPE(int16_t, INT16);
LUISA_TILE_DEFINE_SCALAR_TYPE(uint16_t, UINT16);
LUISA_TILE_DEFINE_SCALAR_TYPE(int32_t, INT32);
LUISA_TILE_DEFINE_SCALAR_TYPE(uint32_t, UINT32);
LUISA_TILE_DEFINE_SCALAR_TYPE(int64_t, INT64);
LUISA_TILE_DEFINE_SCALAR_TYPE(uint64_t, UINT64);
LUISA_TILE_DEFINE_SCALAR_TYPE(float, FLOAT32);
LUISA_TILE_DEFINE_SCALAR_TYPE(double, FLOAT64);

#undef LUISA_TILE_DEFINE_SCALAR_TYPE

template<typename T>
inline constexpr auto scalar_type_v = scalar_type<std::remove_cv_t<T>>::value;

template<typename T>
concept scalar_cpp_type = requires { scalar_type<std::remove_cv_t<T>>::value; };

template<typename T>
class Scalar final {

    static_assert(scalar_cpp_type<T>);

private:
    detail::ValueHandle _handle;

    explicit Scalar(detail::ValueHandle handle) noexcept
        : _handle{std::move(handle)} {}

    friend class Nest;
    template<typename U>
    friend class ElementRef;
    template<typename U>
    friend class Buffer;
    template<scalar_cpp_type U>
    friend Scalar<U> detail_make_unary(ElementwiseOp, const Scalar<U> &) noexcept;
    template<scalar_cpp_type U>
    friend Scalar<U> detail_make_binary(ElementwiseOp, const Scalar<U> &, const Scalar<U> &) noexcept;
    template<scalar_cpp_type U>
    friend Scalar<bool> detail_make_compare(ElementwiseOp, const Scalar<U> &, const Scalar<U> &) noexcept;
    template<scalar_cpp_type To, scalar_cpp_type From>
    friend Scalar<To> cast(const Scalar<From> &) noexcept;
    template<scalar_cpp_type U>
    friend Scalar<U> select(const Scalar<bool> &, const Scalar<U> &, const Scalar<U> &) noexcept;

public:
    using value_type = T;

    Scalar() noexcept = default;
    Scalar(T value) noexcept {// NOLINT(google-explicit-constructor)
        if constexpr (std::is_same_v<T, bool>) {
            _handle = detail::make_constant(scalar_type_v<T>, Attribute{value});
        } else if constexpr (std::is_floating_point_v<T>) {
            _handle = detail::make_constant(scalar_type_v<T>, Attribute{static_cast<double>(value)});
        } else if constexpr (std::is_signed_v<T>) {
            _handle = detail::make_constant(scalar_type_v<T>, Attribute{static_cast<int64_t>(value)});
        } else {
            _handle = detail::make_constant(scalar_type_v<T>, Attribute{static_cast<uint64_t>(value)});
        }
    }

    Scalar(const Scalar &) noexcept = default;
    Scalar(Scalar &&) noexcept = default;
    Scalar &operator=(const Scalar &) noexcept = default;
    Scalar &operator=(Scalar &&) noexcept = default;
    Scalar &operator=(T value) noexcept {
        return *this = Scalar{value};
    }

    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(_handle); }
    [[nodiscard]] Value *ir_value() const noexcept { return _handle.value(); }

    Scalar &operator+=(const Scalar &rhs) noexcept;
    Scalar &operator-=(const Scalar &rhs) noexcept;
    Scalar &operator*=(const Scalar &rhs) noexcept;
    Scalar &operator/=(const Scalar &rhs) noexcept;
};

template<scalar_cpp_type T>
[[nodiscard]] Scalar<T> detail_make_unary(ElementwiseOp op, const Scalar<T> &value) noexcept {
    detail::ValueHandle operands[]{value._handle};
    return Scalar<T>{detail::make_elementwise_operation(op, operands, scalar_type_v<T>)};
}

template<scalar_cpp_type T>
[[nodiscard]] Scalar<T> detail_make_binary(
    ElementwiseOp op,
    const Scalar<T> &lhs,
    const Scalar<T> &rhs) noexcept {
    detail::ValueHandle operands[]{lhs._handle, rhs._handle};
    return Scalar<T>{detail::make_elementwise_operation(op, operands, scalar_type_v<T>)};
}

template<scalar_cpp_type T>
[[nodiscard]] Scalar<bool> detail_make_compare(
    ElementwiseOp op,
    const Scalar<T> &lhs,
    const Scalar<T> &rhs) noexcept {
    detail::ValueHandle operands[]{lhs._handle, rhs._handle};
    return Scalar<bool>{detail::make_elementwise_operation(op, operands, ScalarType::BOOL)};
}

#define LUISA_TILE_DEFINE_BINARY_OPERATOR(symbol, opcode)                                                 \
    template<scalar_cpp_type T>                                                                           \
    [[nodiscard]] inline Scalar<T> operator symbol(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept { \
        return detail_make_binary(ElementwiseOp::opcode, lhs, rhs);                                       \
    }                                                                                                     \
    template<scalar_cpp_type T>                                                                           \
    [[nodiscard]] inline Scalar<T> operator symbol(const Scalar<T> &lhs, T rhs) noexcept {                \
        return lhs symbol Scalar<T>{rhs};                                                                 \
    }                                                                                                     \
    template<scalar_cpp_type T>                                                                           \
    [[nodiscard]] inline Scalar<T> operator symbol(T lhs, const Scalar<T> &rhs) noexcept {                \
        return Scalar<T>{lhs} symbol rhs;                                                                 \
    }

LUISA_TILE_DEFINE_BINARY_OPERATOR(+, ADD)
LUISA_TILE_DEFINE_BINARY_OPERATOR(-, SUB)
LUISA_TILE_DEFINE_BINARY_OPERATOR(*, MUL)
LUISA_TILE_DEFINE_BINARY_OPERATOR(/, DIV)
LUISA_TILE_DEFINE_BINARY_OPERATOR(%, MOD)

#undef LUISA_TILE_DEFINE_BINARY_OPERATOR

#define LUISA_TILE_DEFINE_COMPARE_OPERATOR(symbol, opcode)                                                   \
    template<scalar_cpp_type T>                                                                              \
    [[nodiscard]] inline Scalar<bool> operator symbol(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept { \
        return detail_make_compare(ElementwiseOp::opcode, lhs, rhs);                                         \
    }                                                                                                        \
    template<scalar_cpp_type T>                                                                              \
    [[nodiscard]] inline Scalar<bool> operator symbol(const Scalar<T> &lhs, T rhs) noexcept {                \
        return lhs symbol Scalar<T>{rhs};                                                                    \
    }                                                                                                        \
    template<scalar_cpp_type T>                                                                              \
    [[nodiscard]] inline Scalar<bool> operator symbol(T lhs, const Scalar<T> &rhs) noexcept {                \
        return Scalar<T>{lhs} symbol rhs;                                                                    \
    }

LUISA_TILE_DEFINE_COMPARE_OPERATOR(==, EQ)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(!=, NE)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(<, LT)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(<=, LE)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(>, GT)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(>=, GE)

#undef LUISA_TILE_DEFINE_COMPARE_OPERATOR

template<typename T>
Scalar<T> &Scalar<T>::operator+=(const Scalar &rhs) noexcept {
    return *this = *this + rhs;
}

template<typename T>
Scalar<T> &Scalar<T>::operator-=(const Scalar &rhs) noexcept {
    return *this = *this - rhs;
}

template<typename T>
Scalar<T> &Scalar<T>::operator*=(const Scalar &rhs) noexcept {
    return *this = *this * rhs;
}

template<typename T>
Scalar<T> &Scalar<T>::operator/=(const Scalar &rhs) noexcept {
    return *this = *this / rhs;
}

template<scalar_cpp_type T>
[[nodiscard]] inline Scalar<T> operator-(const Scalar<T> &value) noexcept {
    return detail_make_unary(ElementwiseOp::NEG, value);
}

template<scalar_cpp_type To, scalar_cpp_type From>
[[nodiscard]] Scalar<To> cast(const Scalar<From> &value) noexcept {
    detail::ValueHandle operands[]{value._handle};
    return Scalar<To>{detail::make_elementwise_operation(ElementwiseOp::CAST, operands, scalar_type_v<To>)};
}

template<scalar_cpp_type T>
[[nodiscard]] Scalar<T> select(
    const Scalar<bool> &condition,
    const Scalar<T> &true_value,
    const Scalar<T> &false_value) noexcept {
    detail::ValueHandle operands[]{condition._handle, true_value._handle, false_value._handle};
    return Scalar<T>{detail::make_elementwise_operation(ElementwiseOp::SELECT, operands, scalar_type_v<T>)};
}

template<std::floating_point T>
[[nodiscard]] inline Scalar<T> exp(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::EXP, value); }
template<std::floating_point T>
[[nodiscard]] inline Scalar<T> log(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::LOG, value); }
template<std::floating_point T>
[[nodiscard]] inline Scalar<T> sqrt(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::SQRT, value); }
template<std::floating_point T>
[[nodiscard]] inline Scalar<T> tanh(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::TANH, value); }
template<scalar_cpp_type T>
[[nodiscard]] inline Scalar<T> abs(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::ABS, value); }
template<scalar_cpp_type T>
[[nodiscard]] inline Scalar<T> min(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
    return detail_make_binary(ElementwiseOp::MIN, lhs, rhs);
}
template<scalar_cpp_type T>
[[nodiscard]] inline Scalar<T> max(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
    return detail_make_binary(ElementwiseOp::MAX, lhs, rhs);
}

class LUISA_TILE_API Axis final {

private:
    Dim _dimension;
    Extent _extent;

    Axis(Dim dimension, Extent extent) noexcept
        : _dimension{dimension}, _extent{extent} {}
    friend Axis detail::create_axis(luisa::string_view, Extent) noexcept;

public:
    Axis() noexcept = default;
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(_dimension) && _extent.is_valid(); }
    [[nodiscard]] Dim dimension() const noexcept { return _dimension; }
    [[nodiscard]] Extent extent() const noexcept { return _extent; }
};

[[nodiscard]] inline Axis axis(luisa::string_view name, uint64_t extent) noexcept {
    return detail::create_axis(name, Extent::constant(extent));
}

[[nodiscard]] inline IndexSpace shape(std::initializer_list<Axis> axes) noexcept {
    return detail::make_shape(axes);
}

template<typename... A>
    requires(sizeof...(A) > 0u && (std::same_as<std::remove_cvref_t<A>, Axis> && ...))
[[nodiscard]] IndexSpace shape(A &&...axes) noexcept {
    Axis values[]{std::forward<A>(axes)...};
    return detail::make_shape(values);
}

class LUISA_TILE_API Kernel final {

private:
    luisa::unique_ptr<detail::KernelStorage> _storage;

    explicit Kernel(luisa::string_view name) noexcept;
    friend class detail::CaptureGuard;

public:
    Kernel(Kernel &&) noexcept;
    Kernel(const Kernel &) noexcept = delete;
    Kernel &operator=(Kernel &&) noexcept;
    Kernel &operator=(const Kernel &) noexcept = delete;
    ~Kernel() noexcept;

    [[nodiscard]] Module &module() noexcept;
    [[nodiscard]] const Module &module() const noexcept;
    [[nodiscard]] Function &function() noexcept;
    [[nodiscard]] const Function &function() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] luisa::span<const luisa::string> diagnostics() const noexcept;

    template<typename F>
    [[nodiscard]] static Kernel define(luisa::string_view name, F &&body) noexcept {
        Kernel kernel{name};
        {
            detail::CaptureGuard guard{kernel};
            std::forward<F>(body)();
        }
        return kernel;
    }
};

template<typename F>
[[nodiscard]] Kernel define(luisa::string_view name, F &&body) noexcept {
    return Kernel::define(name, std::forward<F>(body));
}

class LUISA_TILE_API Nest final {

private:
    detail::ScopeStorage *_scope{nullptr};
    explicit Nest(detail::ScopeStorage *scope) noexcept : _scope{scope} {}
    friend struct detail::ScopeStorage;
    friend luisa::vector<detail::ValueHandle> detail::nest_indices(const Nest &, const IndexSpace &) noexcept;

public:
    Nest() noexcept = default;

    [[nodiscard]] Scalar<int64_t> index(const Axis &axis) const noexcept;
    [[nodiscard]] Scalar<int64_t> index() const noexcept;
    [[nodiscard]] Scalar<int64_t> operator[](const Axis &axis) const noexcept { return index(axis); }

    [[nodiscard]] NestRange parallel(IndexSpace domain, exec::Scope scope = exec::Scope::AUTOMATIC) const noexcept;
    [[nodiscard]] NestRange serial(IndexSpace domain) const noexcept;
    [[nodiscard]] NestRange reduce(IndexSpace domain) const noexcept;
    [[nodiscard]] NestRange pipeline(IndexSpace domain, PipelinePolicy policy = {}) const noexcept;

    void stage(luisa::string_view name = {}) const noexcept;
};

class LUISA_TILE_API NestIterator final {

private:
    NestRange *_range{nullptr};
    bool _done{true};

    explicit NestIterator(NestRange *range, bool done) noexcept
        : _range{range}, _done{done} {}
    friend class NestRange;

public:
    [[nodiscard]] Nest &operator*() const noexcept;
    NestIterator &operator++() noexcept;
    [[nodiscard]] bool operator!=(std::default_sentinel_t) const noexcept { return !_done; }
};

class LUISA_TILE_API NestRange final {

private:
    luisa::unique_ptr<detail::ScopeStorage> _storage;

    explicit NestRange(luisa::unique_ptr<detail::ScopeStorage> storage) noexcept;
    void _enter() noexcept;
    void _exit() noexcept;
    [[nodiscard]] Nest &_nest() noexcept;
    friend NestRange detail::make_range(const Nest *, OperationKind, IndexSpace, exec::Scope, PipelinePolicy) noexcept;
    friend class NestIterator;

public:
    NestRange(NestRange &&) noexcept;
    NestRange(const NestRange &) noexcept = delete;
    NestRange &operator=(NestRange &&) noexcept;
    NestRange &operator=(const NestRange &) noexcept = delete;
    ~NestRange() noexcept;

    [[nodiscard]] NestIterator begin() noexcept {
        _enter();
        return NestIterator{this, false};
    }
    [[nodiscard]] std::default_sentinel_t end() const noexcept { return {}; }
};

[[nodiscard]] inline NestRange parallel(IndexSpace domain, exec::Scope scope = exec::Scope::AUTOMATIC) noexcept {
    return detail::make_range(nullptr, OperationKind::PARALLEL, std::move(domain), scope, {});
}

[[nodiscard]] inline NestRange serial(IndexSpace domain) noexcept {
    return detail::make_range(nullptr, OperationKind::SERIAL, std::move(domain), exec::Scope::AUTOMATIC, {});
}

[[nodiscard]] inline NestRange reduce(IndexSpace domain) noexcept {
    return detail::make_range(nullptr, OperationKind::REDUCE, std::move(domain), exec::Scope::AUTOMATIC, {});
}

[[nodiscard]] inline NestRange pipeline(IndexSpace domain, PipelinePolicy policy = {}) noexcept {
    return detail::make_range(nullptr, OperationKind::PIPELINE, std::move(domain), exec::Scope::AUTOMATIC, policy);
}

template<typename T>
class ElementRef final {

    static_assert(scalar_cpp_type<T>);

private:
    Value *_view{nullptr};
    luisa::vector<detail::ValueHandle> _indices;

public:
    ElementRef(Value *view, luisa::vector<detail::ValueHandle> indices) noexcept
        : _view{view}, _indices{std::move(indices)} {}

    [[nodiscard]] Scalar<T> load() const noexcept {
        return Scalar<T>{detail::load_view(_view, _indices)};
    }
    void store(const Scalar<T> &value) const noexcept {
        detail::store_view(_view, _indices, value._handle);
    }
    void store(T value) const noexcept { store(Scalar<T>{value}); }

    operator Scalar<T>() const noexcept {// NOLINT(google-explicit-constructor)
        return load();
    }
    ElementRef &operator=(const Scalar<T> &value) noexcept {
        store(value);
        return *this;
    }
    ElementRef &operator=(T value) noexcept {
        store(value);
        return *this;
    }
};

template<typename T>
class Buffer final {

    static_assert(scalar_cpp_type<T>);

private:
    Value *_view{nullptr};
    IndexSpace _space;
    BufferAccess _access{BufferAccess::READ_WRITE};

    Buffer(Value *view, IndexSpace space, BufferAccess access) noexcept
        : _view{view}, _space{std::move(space)}, _access{access} {}
    template<scalar_cpp_type U>
    friend Buffer<U> input(luisa::string_view, IndexSpace) noexcept;
    template<scalar_cpp_type U>
    friend Buffer<U> output(luisa::string_view, IndexSpace) noexcept;
    template<scalar_cpp_type U>
    friend Buffer<U> inout(luisa::string_view, IndexSpace) noexcept;

public:
    Buffer() noexcept = default;
    [[nodiscard]] const IndexSpace &space() const noexcept { return _space; }
    [[nodiscard]] BufferAccess access() const noexcept { return _access; }
    [[nodiscard]] Value *ir_value() const noexcept { return _view; }

    [[nodiscard]] ElementRef<T> operator[](const Nest &nest) const noexcept {
        return ElementRef<T>{_view, detail::nest_indices(nest, _space)};
    }

    template<typename... I>
        requires(sizeof...(I) > 0u && (std::same_as<std::remove_cvref_t<I>, Scalar<int64_t>> && ...))
    [[nodiscard]] ElementRef<T> operator()(I &&...indices) const noexcept {
        detail::ValueHandle values[]{indices._handle...};
        return ElementRef<T>{_view, {std::begin(values), std::end(values)}};
    }
};

template<scalar_cpp_type T>
[[nodiscard]] Buffer<T> input(luisa::string_view name, IndexSpace space) noexcept {
    auto view = detail::declare_view(name, scalar_type_v<T>, space, BufferAccess::READ);
    return Buffer<T>{view, std::move(space), BufferAccess::READ};
}

template<scalar_cpp_type T>
[[nodiscard]] Buffer<T> output(luisa::string_view name, IndexSpace space) noexcept {
    auto view = detail::declare_view(name, scalar_type_v<T>, space, BufferAccess::WRITE);
    return Buffer<T>{view, std::move(space), BufferAccess::WRITE};
}

template<scalar_cpp_type T>
[[nodiscard]] Buffer<T> inout(luisa::string_view name, IndexSpace space) noexcept {
    auto view = detail::declare_view(name, scalar_type_v<T>, space, BufferAccess::READ_WRITE);
    return Buffer<T>{view, std::move(space), BufferAccess::READ_WRITE};
}

}// namespace luisa::compute::tile
