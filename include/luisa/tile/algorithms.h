#pragma once

// Portable Tile algorithms. These compose the core value/nest operations;
// they are not additional TileIR primitives or target-specific scope kinds.
#include <luisa/tile/dsl.h>
#include <bit>

namespace luisa::compute::tile {

template<scalar_cpp_type T>
[[nodiscard]] Tile<T> reshape(const Tile<T> &value, const IndexSpace &space) noexcept {
    if (value.space() == space) { return value; }
    auto source_volume = value.space().static_volume();
    auto destination_volume = space.static_volume();
    if (!source_volume || !destination_volume || *source_volume != *destination_volume) {
        detail::capture_error("reshape requires equal static logical volumes");
        return {};
    }
    if (*source_volume == 0u) { return zeros<T>(space); }
    return map<T>(space, [&](const Nest &nest) {
        auto linear = Scalar<int64_t>{0};
        for (auto &&axis : space.axes()) {
            linear = linear * axis.extent.constant_value() + nest.index(axis.dimension);
        }
        luisa::vector<Scalar<int64_t>> indices(value.space().rank());
        for (auto i = value.space().rank(); i != 0u; i--) {
            auto extent = value.space().axis(i - 1u).extent.constant_value();
            indices[i - 1u] = linear % extent;
            linear = linear / extent;
        }
        return value.at(indices);
    });
}

template<scalar_cpp_type T, typename F>
[[nodiscard]] Tile<T> reindex(const Tile<T> &value, const IndexSpace &space, F &&coordinates) noexcept {
    return map<T>(space, [&](const Nest &nest) { return value.at(coordinates(nest)); });
}

namespace detail {

[[nodiscard]] inline luisa::vector<Scalar<int64_t>> projected_coordinates(
    const IndexSpace &space, const Nest &nest, Dim replacement_dimension, const Scalar<int64_t> &replacement) noexcept {
    luisa::vector<Scalar<int64_t>> indices;
    for (auto &&axis : space.axes()) {
        indices.emplace_back(axis.dimension == replacement_dimension ? replacement : nest.index(axis.dimension));
    }
    return indices;
}

}// namespace detail

template<scalar_cpp_type T>
[[nodiscard]] Tile<T> gather(const Tile<T> &value, const Tile<int64_t> &indices, Axis dimension,
                             T fallback = T{}) noexcept {
    auto axis_index = value.space().axis_index(dimension.dimension());
    if (!axis_index || value.space().axis(*axis_index).extent != dimension.extent()) {
        detail::capture_error("gather dimension must belong to the source Tile");
        return {};
    }
    IndexSpace output;
    for (auto &&axis : value.space().axes()) {
        if (axis.dimension != dimension.dimension()) { static_cast<void>(output.add(axis.dimension, axis.extent)); }
    }
    for (auto &&axis : indices.space().axes()) {
        if (auto existing = output.axis_index(axis.dimension)) {
            if (output.axis(*existing).extent != axis.extent) {
                detail::capture_error("gather index dimensions disagree with the source");
                return {};
            }
        } else {
            static_cast<void>(output.add(axis.dimension, axis.extent));
        }
    }
    return map<T>(output, [&](const Nest &nest) {
        auto index = indices.at(nest);
        auto coordinates = detail::projected_coordinates(value.space(), nest, dimension.dimension(), index);
        auto in_bounds = (index >= 0) && (index < dimension.extent().constant_value());
        return ite(in_bounds, value.at(coordinates), fallback);
    });
}

template<scalar_cpp_type T>
[[nodiscard]] Tile<int64_t> argmax(const Tile<T> &value, Axis dimension) noexcept {
    auto peak = reduce(value, dimension, maximum);
    auto indices = iota(dimension);
    return reduce(ite(value == peak, indices, std::numeric_limits<int64_t>::max()), dimension, minimum);
}

// The default composition remains unchanged on every backend. PACKED_FP32
// is explicit and requires non-NaN float values, a power-of-two axis and N <= 2^31.
enum class SortAlgorithm : uint8_t { DEFAULT, PACKED_FP32 };

template<scalar_cpp_type T>
struct RankedTile {
    Tile<T> values;
    Tile<int64_t> indices;
};

// Every output includes its own element. The policy is part of the IR: a
// backend may implement an unordered sum with a parallel scan, while ordered
// policies keep their declared contribution order.
template<scalar_cpp_type T>
[[nodiscard]] Tile<T> inclusive_sum(const Tile<T> &value, Axis dimension,
                                    ReductionPolicy policy = reduction::unordered_tree) noexcept {
    auto index = value.space().axis_index(dimension.dimension());
    if (!index || !dimension.extent().is_constant() ||
        dimension.extent().constant_value() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        value.space().axis(*index).extent != dimension.extent()) {
        detail::capture_error("inclusive_sum requires a source dimension with matching static extent");
        return {};
    }
    auto candidate = axis("prefix", dimension.extent().constant_value());
    return map<T>(value.space(), [&](const Nest &nest) {
        auto accumulator = Scalar<T>{T{}};
        for (auto &part : nest.reduce(shape(candidate), policy)) {
            auto coordinate = part.index(candidate);
            auto element = value.at(detail::projected_coordinates(value.space(), part, dimension.dimension(), coordinate));
            accumulator += ite(coordinate <= nest.index(dimension), element, T{});
        }
        return accumulator;
    });
}

namespace detail {

// A sorting network over a power-of-two axis. The total order is lexicographic
// (value, original index), so ties preserve both index order and value bits.
// Each stage is a pure coordinate permutation plus independent comparisons;
// targets can lower the permutation without a data-dependent register gather.
template<scalar_cpp_type T, scalar_cpp_type I>
[[nodiscard]] RankedTile<T> bitonic_sort_indexed(const Tile<T> &value, Axis dimension, bool largest) noexcept {
    auto extent = dimension.extent().constant_value();
    auto values = value;
    auto indices = cast<I>(iota(dimension)) + zeros<I>(value.space());
    auto lane = iota(dimension);
    for (auto span = uint64_t{2u}; span <= extent;) {
        for (auto stride = span / 2u; stride != 0u; stride /= 2u) {
            auto partner = [&](const Nest &nest) {
                auto i = nest.index(dimension);
                auto distance = static_cast<int64_t>(stride);
                auto group = 2 * distance;
                auto j = i / group * group + (i + distance) % group;
                return projected_coordinates(value.space(), nest, dimension.dimension(), j);
            };
            auto other_values = reindex(values, value.space(), partner);
            auto other_indices = reindex(indices, value.space(), partner);
            auto first = (lane / static_cast<int64_t>(stride) % 2 == 0) ==
                         (lane / static_cast<int64_t>(span) % 2 == 0);
            auto other_before = (largest ? other_values > values : other_values < values) ||
                                ((other_values == values) && (other_indices < indices));
            // The supported value order excludes NaNs, and original indices
            // stay unique through permutations. The two pair orders are then
            // complements, including value ties and signed zero.
            auto take_other = first == other_before;
            values = ite(take_other, other_values, values);
            indices = ite(take_other, other_indices, indices);
        }
        if (span == extent) { break; }
        span *= 2u;
    }
    return {std::move(values), cast<int64_t>(indices)};
}

// Reversible UINT64 keys carry one sorting network. Integer division and
// remainder use exact power-of-two constants; no floating arithmetic or
// pointer gather participates in either encoding or decoding.
[[nodiscard]] inline RankedTile<float> packed_bitonic_sort_fp32(const Tile<float> &value, Axis dimension, bool largest) noexcept {
    constexpr auto sign = uint32_t{0x80000000u};
    constexpr auto low_mask = uint32_t{0x7fffffffu};
    constexpr auto word_mask = uint32_t{0xffffffffu};
    constexpr auto word_scale = uint64_t{1u} << 32u;
    auto bits = bitcast<uint32_t>(value);
    auto normalized = ite(bits % sign == 0u, uint32_t{0u}, bits);
    auto negative = normalized >= sign;
    auto numerical = largest ? ite(negative, normalized, low_mask - normalized) :
                               ite(negative, word_mask - normalized, normalized + sign);
    auto indices = cast<uint64_t>(iota(dimension)) + zeros<uint64_t>(value.space());
    auto payload = indices * uint64_t{2u} + cast<uint64_t>(bits / sign);
    auto keys = cast<uint64_t>(numerical) * word_scale + payload;
    auto lane = iota(dimension);
    auto extent = dimension.extent().constant_value();
    for (auto span = uint64_t{2u}; span <= extent;) {
        for (auto stride = span / 2u; stride != 0u; stride /= 2u) {
            auto partner = [&](const Nest &nest) {
                auto i = nest.index(dimension);
                auto distance = static_cast<int64_t>(stride);
                auto group = 2 * distance;
                auto j = i / group * group + (i + distance) % group;
                return projected_coordinates(value.space(), nest, dimension.dimension(), j);
            };
            auto other = reindex(keys, value.space(), partner);
            auto first = (lane / static_cast<int64_t>(stride) % 2 == 0) ==
                         (lane / static_cast<int64_t>(span) % 2 == 0);
            // Composite keys are unique: original index precedes the sign bit.
            keys = ite(first == (other < keys), other, keys);
        }
        if (span == extent) { break; }
        span *= 2u;
    }
    auto sorted_payload = cast<uint32_t>(keys % word_scale);
    auto sorted_numerical = cast<uint32_t>(keys / word_scale);
    auto restored_normalized = largest ? ite(sorted_numerical >= sign, sorted_numerical, low_mask - sorted_numerical) :
                                         ite(sorted_numerical < sign, word_mask - sorted_numerical, sorted_numerical - sign);
    auto original_bits = ite(restored_normalized == 0u, (sorted_payload % uint32_t{2u}) * sign, restored_normalized);
    return {bitcast<float>(original_bits), cast<int64_t>(sorted_payload / uint32_t{2u})};
}

template<scalar_cpp_type T>
[[nodiscard]] RankedTile<T> bitonic_sort(const Tile<T> &value, Axis dimension, bool largest) noexcept {
    // Only the carried original indices are narrowed. Coordinates retain their
    // existing type and shape, including the native permutation recognizer.
    // The public result always has int64 indices, including small Tiles.
    if (dimension.extent().constant_value() <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return bitonic_sort_indexed<T, int32_t>(value, dimension, largest);
    }
    return bitonic_sort_indexed<T, int64_t>(value, dimension, largest);
}

}// namespace detail

// Stable total order on non-NaN values (including infinities); ties use the original index. This
// uses a bitonic network for power-of-two axes, and retains a general quadratic
// composition for other extents. Selecting a prefix never changes tie order.
template<scalar_cpp_type T>
[[nodiscard]] RankedTile<T> topk(const Tile<T> &value, Axis dimension, uint64_t count, bool largest = true,
                                 SortAlgorithm algorithm = SortAlgorithm::DEFAULT) noexcept {
    auto source_axis = value.space().axis_index(dimension.dimension());
    if (!source_axis || !dimension.extent().is_constant() ||
        dimension.extent().constant_value() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        value.space().axis(*source_axis).extent != dimension.extent() || count > dimension.extent().constant_value()) {
        detail::capture_error("topk requires a source dimension and k no greater than its extent");
        return {};
    }
    auto extent = dimension.extent().constant_value();
    if (algorithm != SortAlgorithm::DEFAULT && algorithm != SortAlgorithm::PACKED_FP32) {
        detail::capture_error("unknown Tile sort algorithm");
        return {};
    }
    if (algorithm == SortAlgorithm::PACKED_FP32 &&
        (!std::same_as<T, float> || !std::has_single_bit(extent) || extent > (uint64_t{1u} << 31u))) {
        detail::capture_error("packed sort requires float32, a power-of-two axis and N <= 2^31");
        return {};
    }
    if (std::has_single_bit(extent)) {
        auto sorted = [&] {
            if constexpr (std::same_as<T, float>) {
                if (algorithm == SortAlgorithm::PACKED_FP32) { return detail::packed_bitonic_sort_fp32(value, dimension, largest); }
            }
            return detail::bitonic_sort(value, dimension, largest);
        }();
        if (count == extent) { return sorted; }
        auto rank_axis = axis("rank", count);
        IndexSpace output;
        for (auto &&axis : value.space().axes()) {
            auto replacement = axis.dimension == dimension.dimension();
            static_cast<void>(output.add(replacement ? rank_axis.dimension() : axis.dimension,
                                         replacement ? rank_axis.extent() : axis.extent));
        }
        auto coordinates = [&](const Nest &nest) {
            return detail::projected_coordinates(value.space(), nest, dimension.dimension(), nest.index(rank_axis));
        };
        return {reindex(sorted.values, output, coordinates), reindex(sorted.indices, output, coordinates)};
    }
    auto candidate = axis("candidate", dimension.extent().constant_value());
    auto ranks = map<int64_t>(value.space(), [&](const Nest &nest) {
        auto index = nest.index(dimension);
        auto element = value.at(nest);
        auto rank = Scalar<int64_t>{0};
        for (auto &other : nest.reduce(shape(candidate))) {
            auto other_index = other.index();
            auto other_value = value.at(detail::projected_coordinates(value.space(), other, dimension.dimension(), other_index));
            auto ordered = largest ? other_value > element : other_value < element;
            rank += cast<int64_t>(ordered || ((other_value == element) && (other_index < index)));
        }
        return rank;
    });
    auto rank_axis = count == dimension.extent().constant_value() ? dimension : axis("rank", count);
    IndexSpace output;
    for (auto &&axis : value.space().axes()) {
        auto replacement = axis.dimension == dimension.dimension();
        static_cast<void>(output.add(replacement ? rank_axis.dimension() : axis.dimension,
                                     replacement ? rank_axis.extent() : axis.extent));
    }
    auto selected = map<int64_t>(output, [&](const Nest &nest) {
        auto output_rank = nest.index(rank_axis);
        auto index = Scalar<int64_t>{-1};
        for (auto &item : nest.reduce(shape(candidate))) {
            auto candidate_index = item.index();
            auto coordinates = detail::projected_coordinates(value.space(), item, dimension.dimension(), candidate_index);
            index = ite(ranks.at(coordinates) == output_rank, candidate_index, index);
        }
        return index;
    });
    // Keep the ranked axis at its original position. General gather builds a
    // broadcast union of dimensions and may place a replaced interior axis
    // after the other source axes, which differs from the indices' layout.
    auto values = map<T>(output, [&](const Nest &nest) {
        auto index = selected.at(nest);
        auto coordinates = detail::projected_coordinates(value.space(), nest, dimension.dimension(), index);
        auto in_bounds = (index >= 0) && (index < dimension.extent().constant_value());
        return ite(in_bounds, value.at(coordinates), T{});
    });
    return {std::move(values), std::move(selected)};
}

template<scalar_cpp_type T>
[[nodiscard]] RankedTile<T> sort(const Tile<T> &value, Axis dimension, bool descending = false,
                                 SortAlgorithm algorithm = SortAlgorithm::DEFAULT) noexcept {
    return topk(value, dimension, dimension.extent().constant_value(), descending, algorithm);
}

}// namespace luisa::compute::tile
