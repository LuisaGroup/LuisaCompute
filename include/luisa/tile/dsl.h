#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <tuple>
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
class Tile;
template<typename T, size_t Rank>
class MemoryRef;
template<typename T, size_t Rank>
class TensorView;
template<size_t Rank>
class TensorShape;
template<typename Signature, typename F>
class KernelDefinition;

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

namespace mem {

// Independent resource classes, not an ordering of memory levels.
enum class Resource : uint8_t {
    AUTOMATIC,
    PRIVATE,
    SHARED,
    CLUSTER,
    GLOBAL,
    TENSOR
};

inline constexpr auto auto_ = Resource::AUTOMATIC;
inline constexpr auto private_ = Resource::PRIVATE;
inline constexpr auto shared = Resource::SHARED;
inline constexpr auto cluster = Resource::CLUSTER;
inline constexpr auto global = Resource::GLOBAL;
inline constexpr auto tensor = Resource::TENSOR;

}// namespace mem

struct PipelinePolicy {
    uint32_t stages{0u};
    uint32_t initiation_interval{1u};
};

namespace bounds {
inline constexpr auto assume = BoundsMode::ASSUME;
inline constexpr auto zero = BoundsMode::ZERO;
}// namespace bounds

namespace detail {

struct ValueSlot;
struct DeclaredMemory;
struct KernelStorage;
struct ScopeStorage;
template<size_t Rank>
struct Coordinates;
template<size_t Rank>
struct TileSelection;

struct DeclaredTensorView {
    Value *value{nullptr};
    IndexSpace space;
};

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
                                 luisa::span<const ValueHandle> indices,
                                 const ValueHandle *predicate,
                                 const ValueHandle *fallback) noexcept;
    friend luisa::vector<ValueHandle> nest_indices(
        const Nest &nest,
        const IndexSpace &space) noexcept;
    friend class ::luisa::compute::tile::Nest;
    template<typename T>
    friend class ::luisa::compute::tile::Scalar;
    friend ValueHandle make_tile_constant(ScalarType, const IndexSpace &, Attribute) noexcept;
    friend ValueHandle make_tile_elementwise(ElementwiseOp, luisa::span<Value *const>, ScalarType) noexcept;
    friend ValueHandle make_mma(Value *, Value *, Value *) noexcept;
    friend ValueHandle load_tile(Value *, luisa::span<Value *const>, const IndexSpace &, BoundsMode, Value *) noexcept;
    friend ValueHandle extract_tile(Value *, luisa::span<Value *const>) noexcept;
    friend ValueHandle capture_tile_map(const IndexSpace &, ScalarType, const std::function<Value *(const Nest &)> &) noexcept;
    friend DeclaredMemory declare_memory(ScalarType, const IndexSpace &, mem::Resource, const IndexMap *) noexcept;
    friend ValueHandle load_memory(Value *, const ValueHandle &) noexcept;
    friend void store_memory(Value *, ValueHandle &, Value *) noexcept;

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

struct DeclaredMemory {
    Value *memory{nullptr};
    ValueHandle state;
};

[[nodiscard]] LUISA_TILE_API DeclaredMemory declare_memory(ScalarType type, const IndexSpace &space, mem::Resource resource, const IndexMap *layout = nullptr) noexcept;
[[nodiscard]] LUISA_TILE_API IndexMap make_strided_layout(const IndexSpace &space, luisa::span<const uint64_t> strides) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle load_memory(Value *memory, const ValueHandle &state) noexcept;
LUISA_TILE_API void store_memory(Value *memory, ValueHandle &state, Value *tile) noexcept;

[[nodiscard]] LUISA_TILE_API ValueHandle make_constant(ScalarType type, Attribute value) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle make_elementwise_operation(
    ElementwiseOp op,
    luisa::span<const ValueHandle> operands,
    ScalarType result_type) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle load_view(
    Value *view,
    luisa::span<const ValueHandle> indices,
    const ValueHandle *predicate = nullptr,
    const ValueHandle *fallback = nullptr) noexcept;
LUISA_TILE_API void store_view(
    Value *view,
    luisa::span<const ValueHandle> indices,
    const ValueHandle &value) noexcept;

[[nodiscard]] LUISA_TILE_API ValueHandle make_tile_constant(ScalarType type, const IndexSpace &space, Attribute value) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle make_tile_elementwise(ElementwiseOp op, luisa::span<Value *const> operands, ScalarType type) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle make_mma(Value *a, Value *b, Value *accumulator) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle load_tile(Value *view, luisa::span<Value *const> origin,
                                                   const IndexSpace &space, BoundsMode bounds, Value *fallback = nullptr) noexcept;
LUISA_TILE_API void store_tile(Value *view, luisa::span<Value *const> origin,
                               const IndexSpace &space, Value *tile, BoundsMode bounds) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle extract_tile(Value *tile, luisa::span<Value *const> indices) noexcept;
[[nodiscard]] LUISA_TILE_API ValueHandle capture_tile_map(
    const IndexSpace &space, ScalarType type, const std::function<Value *(const Nest &)> &body) noexcept;
LUISA_TILE_API void capture_error(luisa::string_view message) noexcept;

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
[[nodiscard]] LUISA_TILE_API IndexSpace make_positional_shape(luisa::span<const uint64_t> extents) noexcept;
[[nodiscard]] LUISA_TILE_API DeclaredTensorView declare_tensor_view(
    size_t argument_index,
    luisa::string_view name,
    ScalarType element_type,
    luisa::span<const uint64_t> extents) noexcept;
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
concept arithmetic_scalar_cpp_type = scalar_cpp_type<T> && !std::same_as<std::remove_cv_t<T>, bool>;

template<typename T>
concept integral_scalar_cpp_type = std::integral<T> && !std::same_as<std::remove_cv_t<T>, bool>;

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
    friend class Tile;
    template<typename U, size_t Rank>
    friend class TensorView;
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

    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(_handle); }
    [[nodiscard]] Value *ir_value() const noexcept { return _handle.value(); }

    Scalar &operator+=(const Scalar &rhs) noexcept
        requires arithmetic_scalar_cpp_type<T>;
    Scalar &operator-=(const Scalar &rhs) noexcept
        requires arithmetic_scalar_cpp_type<T>;
    Scalar &operator*=(const Scalar &rhs) noexcept
        requires arithmetic_scalar_cpp_type<T>;
    Scalar &operator/=(const Scalar &rhs) noexcept
        requires arithmetic_scalar_cpp_type<T>;
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
    template<arithmetic_scalar_cpp_type T>                                                                \
    [[nodiscard]] inline Scalar<T> operator symbol(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept { \
        return detail_make_binary(ElementwiseOp::opcode, lhs, rhs);                                       \
    }                                                                                                     \
    template<arithmetic_scalar_cpp_type T, typename U>                                                    \
        requires std::convertible_to<U, T>                                                                \
    [[nodiscard]] inline Scalar<T> operator symbol(const Scalar<T> &lhs, U rhs) noexcept {                \
        return lhs symbol Scalar<T>{static_cast<T>(rhs)};                                                 \
    }                                                                                                     \
    template<typename U, arithmetic_scalar_cpp_type T>                                                    \
        requires std::convertible_to<U, T>                                                                \
    [[nodiscard]] inline Scalar<T> operator symbol(U lhs, const Scalar<T> &rhs) noexcept {                \
        return Scalar<T>{static_cast<T>(lhs)} symbol rhs;                                                 \
    }

LUISA_TILE_DEFINE_BINARY_OPERATOR(+, ADD)
LUISA_TILE_DEFINE_BINARY_OPERATOR(-, SUB)
LUISA_TILE_DEFINE_BINARY_OPERATOR(*, MUL)
LUISA_TILE_DEFINE_BINARY_OPERATOR(/, DIV)

#undef LUISA_TILE_DEFINE_BINARY_OPERATOR

template<integral_scalar_cpp_type T>
[[nodiscard]] inline Scalar<T> operator%(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
    return detail_make_binary(ElementwiseOp::MOD, lhs, rhs);
}

template<integral_scalar_cpp_type T, typename U>
    requires std::convertible_to<U, T>
[[nodiscard]] inline Scalar<T> operator%(const Scalar<T> &lhs, U rhs) noexcept {
    return lhs % Scalar<T>{static_cast<T>(rhs)};
}

template<typename U, integral_scalar_cpp_type T>
    requires std::convertible_to<U, T>
[[nodiscard]] inline Scalar<T> operator%(U lhs, const Scalar<T> &rhs) noexcept {
    return Scalar<T>{static_cast<T>(lhs)} % rhs;
}

#define LUISA_TILE_DEFINE_COMPARE_OPERATOR(symbol, opcode)                                                   \
    template<scalar_cpp_type T>                                                                              \
    [[nodiscard]] inline Scalar<bool> operator symbol(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept { \
        return detail_make_compare(ElementwiseOp::opcode, lhs, rhs);                                         \
    }                                                                                                        \
    template<scalar_cpp_type T, typename U>                                                                  \
        requires std::convertible_to<U, T>                                                                   \
    [[nodiscard]] inline Scalar<bool> operator symbol(const Scalar<T> &lhs, U rhs) noexcept {                \
        return lhs symbol Scalar<T>{static_cast<T>(rhs)};                                                    \
    }                                                                                                        \
    template<typename U, scalar_cpp_type T>                                                                  \
        requires std::convertible_to<U, T>                                                                   \
    [[nodiscard]] inline Scalar<bool> operator symbol(U lhs, const Scalar<T> &rhs) noexcept {                \
        return Scalar<T>{static_cast<T>(lhs)} symbol rhs;                                                    \
    }

LUISA_TILE_DEFINE_COMPARE_OPERATOR(==, EQ)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(!=, NE)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(<, LT)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(<=, LE)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(>, GT)
LUISA_TILE_DEFINE_COMPARE_OPERATOR(>=, GE)

#undef LUISA_TILE_DEFINE_COMPARE_OPERATOR

template<typename T>
Scalar<T> &Scalar<T>::operator+=(const Scalar &rhs) noexcept
    requires arithmetic_scalar_cpp_type<T>
{
    return *this = *this + rhs;
}

template<typename T>
Scalar<T> &Scalar<T>::operator-=(const Scalar &rhs) noexcept
    requires arithmetic_scalar_cpp_type<T>
{
    return *this = *this - rhs;
}

template<typename T>
Scalar<T> &Scalar<T>::operator*=(const Scalar &rhs) noexcept
    requires arithmetic_scalar_cpp_type<T>
{
    return *this = *this * rhs;
}

template<typename T>
Scalar<T> &Scalar<T>::operator/=(const Scalar &rhs) noexcept
    requires arithmetic_scalar_cpp_type<T>
{
    return *this = *this / rhs;
}

template<arithmetic_scalar_cpp_type T>
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

template<scalar_cpp_type T>
[[nodiscard]] Scalar<T> select(
    const Scalar<bool> &condition,
    const Scalar<T> &true_value,
    T false_value) noexcept {
    return select(condition, true_value, Scalar<T>{false_value});
}

template<scalar_cpp_type T>
[[nodiscard]] Scalar<T> select(
    const Scalar<bool> &condition,
    T true_value,
    const Scalar<T> &false_value) noexcept {
    return select(condition, Scalar<T>{true_value}, false_value);
}

template<scalar_cpp_type T>
[[nodiscard]] Scalar<T> select(
    const Scalar<bool> &condition,
    T true_value,
    T false_value) noexcept {
    return select(condition, Scalar<T>{true_value}, Scalar<T>{false_value});
}

[[nodiscard]] inline Scalar<bool> operator&&(const Scalar<bool> &lhs, const Scalar<bool> &rhs) noexcept {
    return detail_make_binary(ElementwiseOp::LOGICAL_AND, lhs, rhs);
}

[[nodiscard]] inline Scalar<bool> operator||(const Scalar<bool> &lhs, const Scalar<bool> &rhs) noexcept {
    return detail_make_binary(ElementwiseOp::LOGICAL_OR, lhs, rhs);
}

[[nodiscard]] inline Scalar<bool> operator!(const Scalar<bool> &value) noexcept {
    return detail_make_unary(ElementwiseOp::LOGICAL_NOT, value);
}

template<std::floating_point T>
[[nodiscard]] inline Scalar<T> exp(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::EXP, value); }
template<std::floating_point T>
[[nodiscard]] inline Scalar<T> log(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::LOG, value); }
template<std::floating_point T>
[[nodiscard]] inline Scalar<T> sqrt(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::SQRT, value); }
template<std::floating_point T>
[[nodiscard]] inline Scalar<T> tanh(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::TANH, value); }
template<typename T>
    requires(std::signed_integral<T> || std::floating_point<T>)
[[nodiscard]] inline Scalar<T> abs(const Scalar<T> &value) noexcept { return detail_make_unary(ElementwiseOp::ABS, value); }
template<arithmetic_scalar_cpp_type T>
[[nodiscard]] inline Scalar<T> min(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
    return detail_make_binary(ElementwiseOp::MIN, lhs, rhs);
}
template<arithmetic_scalar_cpp_type T, typename U>
    requires std::convertible_to<U, T>
[[nodiscard]] inline Scalar<T> min(const Scalar<T> &lhs, U rhs) noexcept {
    return min(lhs, Scalar<T>{static_cast<T>(rhs)});
}
template<typename U, arithmetic_scalar_cpp_type T>
    requires std::convertible_to<U, T>
[[nodiscard]] inline Scalar<T> min(U lhs, const Scalar<T> &rhs) noexcept {
    return min(Scalar<T>{static_cast<T>(lhs)}, rhs);
}
template<arithmetic_scalar_cpp_type T>
[[nodiscard]] inline Scalar<T> max(const Scalar<T> &lhs, const Scalar<T> &rhs) noexcept {
    return detail_make_binary(ElementwiseOp::MAX, lhs, rhs);
}
template<arithmetic_scalar_cpp_type T, typename U>
    requires std::convertible_to<U, T>
[[nodiscard]] inline Scalar<T> max(const Scalar<T> &lhs, U rhs) noexcept {
    return max(lhs, Scalar<T>{static_cast<T>(rhs)});
}
template<typename U, arithmetic_scalar_cpp_type T>
    requires std::convertible_to<U, T>
[[nodiscard]] inline Scalar<T> max(U lhs, const Scalar<T> &rhs) noexcept {
    return max(Scalar<T>{static_cast<T>(lhs)}, rhs);
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

template<std::integral... E>
    requires(sizeof...(E) > 0u)
[[nodiscard]] IndexSpace shape(E... extents) noexcept {
    if (((extents < 0) || ...)) {
        detail::capture_error("Tile shape extents cannot be negative");
        return {};
    }
    uint64_t values[]{static_cast<uint64_t>(extents)...};
    return detail::make_positional_shape(values);
}

class LUISA_TILE_API Kernel final {

private:
    luisa::unique_ptr<detail::KernelStorage> _storage;

    explicit Kernel(luisa::string_view name) noexcept;
    friend class detail::CaptureGuard;
    template<typename Signature, typename F>
    friend class KernelDefinition;

    template<typename F>
    [[nodiscard]] static Kernel _capture(luisa::string_view name, F &&body) noexcept {
        Kernel kernel{name};
        {
            detail::CaptureGuard guard{kernel};
            std::forward<F>(body)();
        }
        return kernel;
    }

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
};

class LUISA_TILE_API Nest final {

private:
    detail::ScopeStorage *_scope{nullptr};
    explicit Nest(detail::ScopeStorage *scope) noexcept : _scope{scope} {}
    friend struct detail::ScopeStorage;
    friend luisa::vector<detail::ValueHandle> detail::nest_indices(const Nest &, const IndexSpace &) noexcept;

public:
    Nest() noexcept = default;

    [[nodiscard]] Scalar<int64_t> index(const Axis &axis) const noexcept;
    [[nodiscard]] Scalar<int64_t> index(Dim dimension) const noexcept;
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
    using ValueType = std::remove_cv_t<T>;

private:
    Value *_view{nullptr};
    luisa::vector<detail::ValueHandle> _indices;

public:
    ElementRef(Value *view, luisa::vector<detail::ValueHandle> indices) noexcept
        : _view{view}, _indices{std::move(indices)} {}

    ElementRef(const ElementRef &) noexcept = default;
    ElementRef(ElementRef &&) noexcept = default;
    // A reference is an address, not a value definition. Memory effects must
    // use load/store; even reference-to-reference assignment is not a copy.
    ElementRef &operator=(const ElementRef &) noexcept = delete;
    ElementRef &operator=(ElementRef &&) noexcept = delete;

    [[nodiscard]] Scalar<ValueType> load() const noexcept {
        return Scalar<ValueType>{detail::load_view(_view, _indices)};
    }
    [[nodiscard]] Scalar<ValueType> load(const Scalar<bool> &predicate, const Scalar<ValueType> &fallback) const noexcept {
        return Scalar<ValueType>{detail::load_view(_view, _indices, &predicate._handle, &fallback._handle)};
    }
    [[nodiscard]] Scalar<ValueType> load(const Scalar<bool> &predicate, ValueType fallback = ValueType{}) const noexcept {
        return load(predicate, Scalar<ValueType>{fallback});
    }
    void store(const Scalar<ValueType> &value) const noexcept
        requires(!std::is_const_v<T>)
    {
        detail::store_view(_view, _indices, value._handle);
    }
    void store(ValueType value) const noexcept
        requires(!std::is_const_v<T>)
    { store(Scalar<ValueType>{value}); }
};

// Concrete, host-side metadata used by capture/JIT. The current bridge accepts
// dense tensors; arbitrary strides will be represented by explicit View maps.
// This descriptor does not create an IR parameter or require an active capture.
template<size_t Rank>
class TensorShape final {

    static_assert(Rank > 0u);

private:
    luisa::string _name;
    std::array<uint64_t, Rank> _extents;

public:
    static constexpr auto rank = Rank;

    TensorShape(luisa::string_view name, std::array<uint64_t, Rank> extents) noexcept
        : _name{name}, _extents{extents} {}

    [[nodiscard]] luisa::string_view name() const noexcept { return _name; }
    [[nodiscard]] const auto &extents() const noexcept { return _extents; }
};

template<std::integral... E>
    requires(sizeof...(E) > 0u)
[[nodiscard]] auto tensor_shape(luisa::string_view name, E... extents) noexcept {
    return TensorShape<sizeof...(E)>{name, {static_cast<uint64_t>(extents)...}};
}

template<std::integral... E>
    requires(sizeof...(E) > 0u)
[[nodiscard]] auto tensor_shape(E... extents) noexcept {
    return tensor_shape({}, extents...);
}

// A kernel parameter is a typed projection of an external resource. Const on
// the element type forbids writes; mutable parameters are not labelled output
// or inout, because their actual effects follow from the captured uses.
template<typename T, size_t Rank>
class TensorView final {

    static_assert(scalar_cpp_type<T> && Rank > 0u);

private:
    Value *_view{nullptr};
    IndexSpace _space;
    std::array<uint64_t, Rank> _extents{};

    explicit TensorView(size_t argument_index, const TensorShape<Rank> &shape) noexcept
        : _extents{shape.extents()} {
        auto declared = detail::declare_tensor_view(
            argument_index, shape.name(), scalar_type_v<T>, _extents);
        _view = declared.value;
        _space = std::move(declared.space);
    }

    template<typename Signature, typename F>
    friend class KernelDefinition;

public:
    using value_type = std::remove_cv_t<T>;
    static constexpr auto rank = Rank;
    static constexpr auto writable = !std::is_const_v<T>;

    TensorView() noexcept = default;

    [[nodiscard]] Value *ir_value() const noexcept { return _view; }
    [[nodiscard]] const IndexSpace &space() const noexcept { return _space; }
    [[nodiscard]] uint64_t extent(size_t dimension) const noexcept { return _extents[dimension]; }

    template<size_t Dimension>
        requires(Dimension < Rank)
    [[nodiscard]] uint64_t extent() const noexcept { return _extents[Dimension]; }

    [[nodiscard]] MemoryRef<T, Rank> tile(detail::Coordinates<Rank> origin,
                                          IndexSpace space, BoundsMode bounds = bounds::zero) const noexcept;
    [[nodiscard]] MemoryRef<T, Rank> operator()(detail::Coordinates<Rank> origin,
                                                IndexSpace space, BoundsMode bounds = bounds::zero) const noexcept;
#if defined(__cpp_multidimensional_subscript) && __cpp_multidimensional_subscript >= 202110L
    [[nodiscard]] Tile<std::remove_cv_t<T>> operator[](detail::Coordinates<Rank> origin,
                                                       IndexSpace space, BoundsMode bounds = bounds::zero) const noexcept;
#else
    [[nodiscard]] Tile<std::remove_cv_t<T>> operator[](detail::TileSelection<Rank> selection) const noexcept;
#endif

    template<typename... I>
        requires(sizeof...(I) == Rank && (std::same_as<std::remove_cvref_t<I>, Scalar<int64_t>> && ...))
    [[nodiscard]] ElementRef<T> operator()(I &&...indices) const noexcept {
        detail::ValueHandle values[]{indices._handle...};
        return ElementRef<T>{_view, {std::begin(values), std::end(values)}};
    }
};

namespace detail {

template<typename T>
struct kernel_signature : kernel_signature<decltype(&T::operator())> {};

template<typename R, typename... Args>
struct kernel_signature<R(Args...)> {
    using type = R(Args...);
};

template<typename R, typename... Args>
struct kernel_signature<R (*)(Args...)> : kernel_signature<R(Args...)> {};

template<typename R, typename... Args>
struct kernel_signature<R (*)(Args...) noexcept> : kernel_signature<R(Args...)> {};

#define LUISA_TILE_KERNEL_SIGNATURE(qualifiers)           \
    template<typename R, typename C, typename... Args>    \
    struct kernel_signature<R (C::*)(Args...) qualifiers> \
        : kernel_signature<R(Args...)> {}

LUISA_TILE_KERNEL_SIGNATURE();
LUISA_TILE_KERNEL_SIGNATURE(const);
LUISA_TILE_KERNEL_SIGNATURE(noexcept);
LUISA_TILE_KERNEL_SIGNATURE(const noexcept);

#undef LUISA_TILE_KERNEL_SIGNATURE

template<typename T>
inline constexpr bool is_tensor_view_v = false;

template<typename T, size_t Rank>
inline constexpr bool is_tensor_view_v<TensorView<T, Rank>> = true;

}// namespace detail

// Like Luisa's Kernel lambda signature, but deferred until concrete argument
// metadata is available. Each capture creates a fresh TileIR specialization;
// ordinary host configuration remains an ordinary C++ lambda capture.
template<typename F, typename... Args>
class KernelDefinition<void(Args...), F> final {

    static_assert((detail::is_tensor_view_v<Args> && ...),
                  "Tile kernel parameters must be typed TensorView values.");

private:
    luisa::string _name;
    F _body;

    template<size_t... Ranks, size_t... I>
    [[nodiscard]] Kernel _capture(
        const std::tuple<TensorShape<Ranks>...> &shapes,
        std::index_sequence<I...>) noexcept {
        return Kernel::_capture(_name, [&] {
            std::tuple<Args...> arguments;
            // Comma-fold sequencing is intentional: ABI parameter order must
            // not depend on C++ function-argument evaluation order.
            ((std::get<I>(arguments) = Args{I, std::get<I>(shapes)}), ...);
            std::apply(_body, std::move(arguments));
        });
    }

public:
    KernelDefinition(luisa::string_view name, F body) noexcept
        : _name{name}, _body{std::move(body)} {}

    // Low-level entry for IR tooling and compiler bridges. A device JIT entry
    // derives these descriptors from the concrete runtime argument views.
    template<size_t... Ranks>
        requires(sizeof...(Ranks) == sizeof...(Args))
    [[nodiscard]] Kernel capture(TensorShape<Ranks>... shapes) noexcept {
        static_assert(((Args::rank == Ranks) && ...),
                      "Tensor argument rank does not match the kernel signature.");
        return _capture(std::tuple{std::move(shapes)...}, std::index_sequence_for<Args...>{});
    }
};

template<typename F>
[[nodiscard]] auto tile_kernel(luisa::string_view name, F &&body) noexcept {
    using Definition = std::decay_t<F>;
    using Signature = typename detail::kernel_signature<Definition>::type;
    return KernelDefinition<Signature, Definition>{name, std::forward<F>(body)};
}

template<typename F>
[[nodiscard]] auto tile_kernel(F &&body) noexcept {
    return tile_kernel("tile_kernel", std::forward<F>(body));
}

}// namespace luisa::compute::tile

#include <luisa/tile/value.h>
