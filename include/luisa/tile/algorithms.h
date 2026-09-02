#pragma once

// Portable Tile algorithms. These compose the core value/nest operations;
// they are not additional TileIR primitives or target-specific scope kinds.
#include <luisa/tile/dsl.h>

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
        return select(in_bounds, value.at(coordinates), fallback);
    });
}

template<scalar_cpp_type T>
[[nodiscard]] Tile<int64_t> argmax(const Tile<T> &value, Axis dimension) noexcept {
    auto peak = reduce(value, dimension, maximum);
    auto indices = iota(dimension);
    return reduce(select(value == peak, indices, std::numeric_limits<int64_t>::max()), dimension, minimum);
}

template<scalar_cpp_type T>
struct RankedTile {
    Tile<T> values;
    Tile<int64_t> indices;
};

// Stable total order on finite values; ties use the original index. This
// quadratic reference composition is deliberately not a claim of a tuned
// sorting network. Specialized library implementations can replace it later.
template<scalar_cpp_type T>
[[nodiscard]] RankedTile<T> topk(const Tile<T> &value, Axis dimension, uint64_t count, bool largest = true) noexcept {
    auto source_axis = value.space().axis_index(dimension.dimension());
    if (!source_axis || !dimension.extent().is_constant() ||
        value.space().axis(*source_axis).extent != dimension.extent() || count > dimension.extent().constant_value()) {
        detail::capture_error("topk requires a source dimension and k no greater than its extent");
        return {};
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
            index = select(ranks.at(coordinates) == output_rank, candidate_index, index);
        }
        return index;
    });
    return {gather(value, selected, dimension), std::move(selected)};
}

template<scalar_cpp_type T>
[[nodiscard]] RankedTile<T> sort(const Tile<T> &value, Axis dimension, bool descending = false) noexcept {
    return topk(value, dimension, dimension.extent().constant_value(), descending);
}

}// namespace luisa::compute::tile
