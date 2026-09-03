#pragma once

#include <limits>
#include <luisa/tile/dsl.h>

namespace luisa::compute::tile {

namespace detail {

// Strongly typed only to constrain the pre-C++23 comma-subscript adapter.
// Neither coordinates nor selections create a memory effect.
template<size_t Rank>
struct Coordinates {
    std::array<Scalar<int64_t>, Rank> values;
};

template<size_t Rank>
struct TileSelection {
    Coordinates<Rank> origin;
    IndexSpace space;
    BoundsMode bounds{bounds::zero};
};

#if !defined(__cpp_multidimensional_subscript) || __cpp_multidimensional_subscript < 202110L
template<size_t Rank>
[[nodiscard]] TileSelection<Rank> operator,(Coordinates<Rank> origin, IndexSpace space) noexcept {
    return {std::move(origin), std::move(space), bounds::zero};
}

template<size_t Rank>
[[nodiscard]] TileSelection<Rank> operator,(TileSelection<Rank> selection, BoundsMode bounds) noexcept {
    selection.bounds = bounds;
    return selection;
}
#endif

template<typename T>
inline constexpr bool is_tile_v = false;
template<typename T>
inline constexpr bool is_tile_v<Tile<T>> = true;

template<typename T>
struct element_type {
    using type = std::remove_cvref_t<T>;
};
template<typename T>
struct element_type<Tile<T>> {
    using type = T;
};
template<typename T>
struct element_type<Scalar<T>> {
    using type = T;
};
template<typename T>
using element_type_t = typename element_type<std::remove_cvref_t<T>>::type;

template<typename A, typename B>
using binary_element_t = std::conditional_t<is_tile_v<std::remove_cvref_t<A>>, element_type_t<A>, element_type_t<B>>;

template<typename T, typename V>
concept tile_operand = std::same_as<std::remove_cvref_t<V>, Tile<T>> ||
                       std::same_as<std::remove_cvref_t<V>, Scalar<T>> ||
                       (scalar_cpp_type<std::remove_cvref_t<V>> && std::convertible_to<V, T>);

template<typename A, typename B>
concept tile_binary_operands = (is_tile_v<std::remove_cvref_t<A>> || is_tile_v<std::remove_cvref_t<B>>) &&
                               tile_operand<binary_element_t<A, B>, A> && tile_operand<binary_element_t<A, B>, B>;

template<scalar_cpp_type T, typename V>
[[nodiscard]] Value *operand_value(const V &value) noexcept {
    if constexpr (requires { value.ir_value(); }) {
        return value.ir_value();
    } else {
        return Scalar<T>{static_cast<T>(value)}.ir_value();
    }
}

}// namespace detail

template<typename... I>
    requires((std::same_as<std::remove_cvref_t<I>, Scalar<int64_t>> ||
              integral_scalar_cpp_type<std::remove_cvref_t<I>>) &&
             ...)
[[nodiscard]] auto coord(I &&...indices) noexcept {
    auto lift = []<typename V>(V &&value) noexcept {
        if constexpr (std::same_as<std::remove_cvref_t<V>, Scalar<int64_t>>) {
            return Scalar<int64_t>{std::forward<V>(value)};
        } else {
            return Scalar<int64_t>{static_cast<int64_t>(value)};
        }
    };
    return detail::Coordinates<sizeof...(I)>{{lift(std::forward<I>(indices))...}};
}

// A Tile is an SSA value, never a reference to its source memory. Assignments
// are ref-qualified: a named Tile can evolve, but A[coord, shape] = x is invalid.
template<typename T>
class Tile final {

    static_assert(scalar_cpp_type<T> && !std::is_const_v<T>);

private:
    detail::ValueHandle _handle;

public:
    using value_type = T;
    Tile() noexcept = default;
    explicit Tile(detail::ValueHandle handle) noexcept : _handle{std::move(handle)} {}
    Tile(const Tile &) noexcept = default;
    Tile(Tile &&) noexcept = default;
    Tile &operator=(const Tile &) & noexcept = default;
    Tile &operator=(Tile &&) & noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(_handle); }
    [[nodiscard]] Value *ir_value() const noexcept { return _handle.value(); }
    [[nodiscard]] const IndexSpace &space() const noexcept {
        static const IndexSpace empty;
        return valid() ? *ir_value()->type().index_space() : empty;
    }

    // Explicit descent from a whole Tile into pure element computations. This
    // is not a memory access; coordinates are relative to the Tile's origin.
    [[nodiscard]] Scalar<T> at(luisa::span<const Scalar<int64_t>> indices) const noexcept {
        luisa::vector<Value *> values;
        values.reserve(indices.size());
        for (auto &&index : indices) { values.emplace_back(index.ir_value()); }
        return Scalar<T>{detail::extract_tile(ir_value(), values)};
    }
    template<size_t Rank>
    [[nodiscard]] Scalar<T> at(const detail::Coordinates<Rank> &indices) const noexcept {
        return at(indices.values);
    }
    [[nodiscard]] Scalar<T> at(const Nest &nest) const noexcept {
        luisa::vector<Scalar<int64_t>> indices;
        for (auto &&axis : space().axes()) { indices.emplace_back(nest.index(axis.dimension)); }
        return at(indices);
    }
    [[nodiscard]] Scalar<T> scalar() const noexcept { return at(coord()); }

    template<typename U>
        requires detail::tile_operand<T, U>
    Tile &operator+=(const U &rhs) & noexcept { return *this = *this + rhs; }
    template<typename U>
        requires detail::tile_operand<T, U>
    Tile &operator-=(const U &rhs) & noexcept { return *this = *this - rhs; }
    template<typename U>
        requires detail::tile_operand<T, U>
    Tile &operator*=(const U &rhs) & noexcept { return *this = *this * rhs; }
    template<typename U>
        requires detail::tile_operand<T, U>
    Tile &operator/=(const U &rhs) & noexcept { return *this = *this / rhs; }
};

template<scalar_cpp_type T, typename F>
[[nodiscard]] Tile<T> map(const IndexSpace &space, F &&body) noexcept {
    return Tile<T>{detail::capture_tile_map(space, scalar_type_v<T>, [&](const Nest &nest) {
        Scalar<T> value = body(nest);
        if (value.valid() && value.ir_value()->type().kind() == TypeKind::INDEX) { return cast<T>(value).ir_value(); }
        return value.ir_value();
    })};
}

template<scalar_cpp_type T>
[[nodiscard]] Tile<T> full(const IndexSpace &space, T value) noexcept {
    Attribute attribute;
    if constexpr (std::same_as<T, bool>) {
        attribute = Attribute{value};
    } else if constexpr (std::floating_point<T>) {
        attribute = Attribute{static_cast<double>(value)};
    } else if constexpr (std::signed_integral<T>) {
        attribute = Attribute{static_cast<int64_t>(value)};
    } else {
        attribute = Attribute{static_cast<uint64_t>(value)};
    }
    return Tile<T>{detail::make_tile_constant(scalar_type_v<T>, space, std::move(attribute))};
}

template<scalar_cpp_type T>
[[nodiscard]] Tile<T> full(const IndexSpace &space, const Scalar<T> &value) noexcept {
    return map<T>(space, [&](const Nest &) { return value; });
}

template<scalar_cpp_type T>
[[nodiscard]] Tile<T> zeros(const IndexSpace &space) noexcept { return full<T>(space, T{}); }

template<scalar_cpp_type T>
[[nodiscard]] Tile<T> relabel(const Tile<T> &value, const IndexSpace &space) noexcept {
    if (value.space() == space) { return value; }
    if (value.space().rank() != space.rank()) {
        detail::capture_error("Tile relabel requires the same rank");
        return {};
    }
    for (auto i = 0u; i < space.rank(); i++) {
        if (value.space().axis(i).extent != space.axis(i).extent) {
            detail::capture_error("Tile relabel cannot change extents or transpose elements");
            return {};
        }
    }
    return map<T>(space, [&](const Nest &nest) {
        luisa::vector<Scalar<int64_t>> indices;
        for (auto &&axis : space.axes()) { indices.emplace_back(nest.index(axis.dimension)); }
        return value.at(indices);
    });
}

template<scalar_cpp_type T>
[[nodiscard]] Tile<T> broadcast_to(const Tile<T> &value, const IndexSpace &space) noexcept {
    if (value.space() == space) { return value; }
    for (auto &&axis : value.space().axes()) {
        auto target = space.axis_index(axis.dimension);
        if (!target || (space.axis(*target).extent != axis.extent &&
                        (!axis.extent.is_constant() || axis.extent.constant_value() != 1u))) {
            detail::capture_error("broadcast_to requires matching dimensions with equal or singleton source extents");
            return {};
        }
    }
    return map<T>(space, [&](const Nest &nest) {
        luisa::vector<Scalar<int64_t>> indices;
        for (auto &&axis : value.space().axes()) {
            indices.emplace_back(axis.extent.is_constant() && axis.extent.constant_value() == 1u ?
                                     Scalar<int64_t>{0} :
                                     nest.index(axis.dimension));
        }
        return value.at(indices);
    });
}

template<typename T, size_t Rank>
class MemoryRef final {

    using ValueType = std::remove_cv_t<T>;
    Value *_view{nullptr};
    detail::Coordinates<Rank> _origin;
    IndexSpace _space;
    BoundsMode _bounds{bounds::zero};

    [[nodiscard]] auto _indices() const noexcept {
        std::array<Value *, Rank> result;
        for (auto i = 0u; i < Rank; i++) { result[i] = _origin.values[i].ir_value(); }
        return result;
    }

public:
    MemoryRef(Value *view, detail::Coordinates<Rank> origin, IndexSpace space, BoundsMode bounds) noexcept
        : _view{view}, _origin{std::move(origin)}, _space{std::move(space)}, _bounds{bounds} {}
    MemoryRef(const MemoryRef &) noexcept = default;
    MemoryRef(MemoryRef &&) noexcept = default;
    MemoryRef &operator=(const MemoryRef &) = delete;
    MemoryRef &operator=(MemoryRef &&) = delete;

    [[nodiscard]] Tile<ValueType> load() const noexcept {
        return Tile<ValueType>{detail::load_tile(_view, _indices(), _space, _bounds)};
    }
    [[nodiscard]] Tile<ValueType> load(const Scalar<ValueType> &fallback) const noexcept {
        return Tile<ValueType>{detail::load_tile(_view, _indices(), _space, _bounds, fallback.ir_value())};
    }
    [[nodiscard]] Tile<ValueType> load(ValueType fallback) const noexcept { return load(Scalar<ValueType>{fallback}); }
    void store(const Tile<ValueType> &value) const noexcept
        requires(!std::is_const_v<T>)
    {
        if (value.space() == _space) {
            detail::store_tile(_view, _indices(), _space, value.ir_value(), _bounds);
        } else {
            auto can_broadcast = true;
            for (auto &&axis : value.space().axes()) {
                auto destination = _space.axis_index(axis.dimension);
                can_broadcast &= destination.has_value() &&
                                 (_space.axis(*destination).extent == axis.extent ||
                                  (axis.extent.is_constant() && axis.extent.constant_value() == 1u));
            }
            if (can_broadcast) {
                auto broadcast = broadcast_to(value, _space);
                detail::store_tile(_view, _indices(), _space, broadcast.ir_value(), _bounds);
                return;
            }
            auto renamed = relabel(value, _space);
            detail::store_tile(_view, _indices(), _space, renamed.ir_value(), _bounds);
        }
    }
};

template<typename T, size_t Rank>
MemoryRef<T, Rank> TensorView<T, Rank>::tile(detail::Coordinates<Rank> origin, IndexSpace space, BoundsMode bounds) const noexcept {
    return MemoryRef<T, Rank>{_view, std::move(origin), std::move(space), bounds};
}

template<typename T, size_t Rank>
MemoryRef<T, Rank> TensorView<T, Rank>::operator()(detail::Coordinates<Rank> origin, IndexSpace space, BoundsMode bounds) const noexcept {
    return tile(std::move(origin), std::move(space), bounds);
}

#if defined(__cpp_multidimensional_subscript) && __cpp_multidimensional_subscript >= 202110L
template<typename T, size_t Rank>
Tile<std::remove_cv_t<T>> TensorView<T, Rank>::operator[](detail::Coordinates<Rank> origin, IndexSpace space, BoundsMode bounds) const noexcept {
    return tile(std::move(origin), std::move(space), bounds).load();
}
#else
template<typename T, size_t Rank>
Tile<std::remove_cv_t<T>> TensorView<T, Rank>::operator[](detail::TileSelection<Rank> selection) const noexcept {
    return tile(std::move(selection.origin), std::move(selection.space), selection.bounds).load();
}
#endif

#define LUISA_TILE_BINARY(symbol, opcode)                                                                 \
    template<typename A, typename B>                                                                      \
        requires detail::tile_binary_operands<A, B>                                                       \
    [[nodiscard]] auto operator symbol(const A &a, const B &b) noexcept {                                 \
        using T = detail::binary_element_t<A, B>;                                                         \
        Value *operands[]{detail::operand_value<T>(a), detail::operand_value<T>(b)};                      \
        return Tile<T>{detail::make_tile_elementwise(ElementwiseOp::opcode, operands, scalar_type_v<T>)}; \
    }
LUISA_TILE_BINARY(+, ADD)
LUISA_TILE_BINARY(-, SUB)
LUISA_TILE_BINARY(*, MUL)
LUISA_TILE_BINARY(/, DIV)
LUISA_TILE_BINARY(%, MOD)
#undef LUISA_TILE_BINARY

#define LUISA_TILE_COMPARE(symbol, opcode)                                                                   \
    template<typename A, typename B>                                                                         \
        requires detail::tile_binary_operands<A, B>                                                          \
    [[nodiscard]] Tile<bool> operator symbol(const A &a, const B &b) noexcept {                              \
        using T = detail::binary_element_t<A, B>;                                                            \
        Value *operands[]{detail::operand_value<T>(a), detail::operand_value<T>(b)};                         \
        return Tile<bool>{detail::make_tile_elementwise(ElementwiseOp::opcode, operands, ScalarType::BOOL)}; \
    }
LUISA_TILE_COMPARE(==, EQ)
LUISA_TILE_COMPARE(!=, NE)
LUISA_TILE_COMPARE(<, LT)
LUISA_TILE_COMPARE(<=, LE)
LUISA_TILE_COMPARE(>, GT)
LUISA_TILE_COMPARE(>=, GE)
LUISA_TILE_COMPARE(&&, LOGICAL_AND)
LUISA_TILE_COMPARE(||, LOGICAL_OR)
#undef LUISA_TILE_COMPARE

#define LUISA_TILE_UNARY(name, opcode)                                                                    \
    template<scalar_cpp_type T>                                                                           \
    [[nodiscard]] Tile<T> name(const Tile<T> &value) noexcept {                                           \
        Value *operands[]{value.ir_value()};                                                              \
        return Tile<T>{detail::make_tile_elementwise(ElementwiseOp::opcode, operands, scalar_type_v<T>)}; \
    }
LUISA_TILE_UNARY(operator-, NEG)
LUISA_TILE_UNARY(operator!, LOGICAL_NOT)
LUISA_TILE_UNARY(exp, EXP)
LUISA_TILE_UNARY(log, LOG)
LUISA_TILE_UNARY(sqrt, SQRT)
LUISA_TILE_UNARY(tanh, TANH)
LUISA_TILE_UNARY(abs, ABS)
#undef LUISA_TILE_UNARY

#define LUISA_TILE_MINMAX(name, opcode)                                                                   \
    template<typename A, typename B>                                                                      \
        requires detail::tile_binary_operands<A, B>                                                       \
    [[nodiscard]] auto name(const A &a, const B &b) noexcept {                                            \
        using T = detail::binary_element_t<A, B>;                                                         \
        Value *operands[]{detail::operand_value<T>(a), detail::operand_value<T>(b)};                      \
        return Tile<T>{detail::make_tile_elementwise(ElementwiseOp::opcode, operands, scalar_type_v<T>)}; \
    }
LUISA_TILE_MINMAX(min, MIN)
LUISA_TILE_MINMAX(max, MAX)
#undef LUISA_TILE_MINMAX

template<scalar_cpp_type To, scalar_cpp_type From>
[[nodiscard]] Tile<To> cast(const Tile<From> &value) noexcept {
    Value *operands[]{value.ir_value()};
    return Tile<To>{detail::make_tile_elementwise(ElementwiseOp::CAST, operands, scalar_type_v<To>)};
}

template<typename C, typename A, typename B>
    requires(detail::is_tile_v<std::remove_cvref_t<C>> || detail::tile_binary_operands<A, B>) &&
            (std::same_as<std::remove_cvref_t<C>, Tile<bool>> || std::same_as<std::remove_cvref_t<C>, Scalar<bool>>)
[[nodiscard]] auto ite(const C &condition, const A &true_value, const B &false_value) noexcept {
    using T = detail::binary_element_t<A, B>;
    Value *operands[]{condition.ir_value(), detail::operand_value<T>(true_value), detail::operand_value<T>(false_value)};
    return Tile<T>{detail::make_tile_elementwise(ElementwiseOp::SELECT, operands, scalar_type_v<T>)};
}

template<scalar_cpp_type A, scalar_cpp_type B, scalar_cpp_type C>
[[nodiscard]] Tile<C> mma(const Tile<A> &a, const Tile<B> &b, const Tile<C> &accumulator, MmaPolicy policy = {}) noexcept {
    auto &&as = a.space();
    auto &&bs = b.space();
    auto &&cs = accumulator.space();
    auto compatible = true;
    auto contracted = false;
    for (auto &&axis : as.axes()) {
        auto bi = bs.axis_index(axis.dimension);
        auto ci = cs.axis_index(axis.dimension);
        compatible &= (bi || ci) && (!bi || bs.axis(*bi).extent == axis.extent) &&
                      (!ci || cs.axis(*ci).extent == axis.extent);
        contracted |= bi.has_value() && !ci;
    }
    for (auto &&axis : bs.axes()) {
        auto ai = as.axis_index(axis.dimension);
        auto ci = cs.axis_index(axis.dimension);
        compatible &= (ai || ci) && (!ci || cs.axis(*ci).extent == axis.extent);
    }
    for (auto &&axis : cs.axes()) { compatible &= as.contains(axis.dimension) || bs.contains(axis.dimension); }
    if (!compatible || !contracted) {
        // Ordinary positional rank-two shapes use conventional (M,K)x(K,N).
        // Explicit shared axes also support transposed operands and batches;
        // the canonical TileIR remains a checked named-dimension contraction.
        if (as.rank() != 2u || bs.rank() != 2u || cs.rank() != 2u ||
            as.axis(0).extent != cs.axis(0).extent || bs.axis(1).extent != cs.axis(1).extent ||
            as.axis(1).extent != bs.axis(0).extent) {
            detail::capture_error("mma dimensions do not form a valid contraction or positional matrix product");
            return {};
        }
        auto contracted_axis = detail::create_axis("mma.k", as.axis(1).extent);
        IndexSpace lhs, rhs;
        static_cast<void>(lhs.add(cs.axis(0).dimension, cs.axis(0).extent));
        static_cast<void>(lhs.add(contracted_axis.dimension(), contracted_axis.extent()));
        static_cast<void>(rhs.add(contracted_axis.dimension(), contracted_axis.extent()));
        static_cast<void>(rhs.add(cs.axis(1).dimension, cs.axis(1).extent));
        auto av = relabel(a, lhs);
        auto bv = relabel(b, rhs);
        return Tile<C>{detail::make_mma(av.ir_value(), bv.ir_value(), accumulator.ir_value(), policy)};
    }
    return Tile<C>{detail::make_mma(a.ir_value(), b.ir_value(), accumulator.ir_value(), policy)};
}

// Reduction policies are library values. They do not add operation kinds to
// TileIR: reductions compose tile.map, nest.reduce and pure tile extraction.
struct AddReduction {
    template<scalar_cpp_type T>
    [[nodiscard]] static constexpr T identity() noexcept { return T{}; }
    template<typename A, typename B>
    [[nodiscard]] auto operator()(const A &a, const B &b) const noexcept { return a + b; }
};
struct MaxReduction {
    template<scalar_cpp_type T>
    [[nodiscard]] static constexpr T identity() noexcept {
        if constexpr (std::floating_point<T>) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::lowest();
        }
    }
    template<typename A, typename B>
    [[nodiscard]] auto operator()(const A &a, const B &b) const noexcept { return max(a, b); }
};
struct MinReduction {
    template<scalar_cpp_type T>
    [[nodiscard]] static constexpr T identity() noexcept {
        if constexpr (std::floating_point<T>) {
            return std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::max();
        }
    }
    template<typename A, typename B>
    [[nodiscard]] auto operator()(const A &a, const B &b) const noexcept { return min(a, b); }
};
inline constexpr AddReduction add;
inline constexpr MaxReduction maximum;
inline constexpr MinReduction minimum;

template<scalar_cpp_type T, typename Reducer>
[[nodiscard]] Tile<T> reduce(const Tile<T> &value, const IndexSpace &dimensions, Reducer reducer) noexcept {
    IndexSpace output;
    for (auto &&axis : dimensions.axes()) {
        auto index = value.space().axis_index(axis.dimension);
        if (!index || value.space().axis(*index).extent != axis.extent) {
            detail::capture_error("reduction axes must be dimensions of the source Tile with matching extents");
            return {};
        }
    }
    for (auto &&axis : value.space().axes()) {
        if (!dimensions.contains(axis.dimension)) { static_cast<void>(output.add(axis.dimension, axis.extent)); }
    }
    if (dimensions.empty()) { return value; }
    return map<T>(output, [&](const Nest &nest) {
        auto accumulator = Scalar<T>{Reducer::template identity<T>()};
        for (auto &element : nest.reduce(dimensions)) { accumulator = reducer(accumulator, value.at(element)); }
        return accumulator;
    });
}

template<scalar_cpp_type T, typename Reducer>
[[nodiscard]] Tile<T> reduce(const Tile<T> &value, Axis dimension, Reducer reducer) noexcept {
    return reduce(value, shape(dimension), reducer);
}

[[nodiscard]] inline Tile<int64_t> iota(Axis axis) noexcept {
    return map<int64_t>(shape(axis), [&](const Nest &nest) { return nest.index(axis); });
}

}// namespace luisa::compute::tile
