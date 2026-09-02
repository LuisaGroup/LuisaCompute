#include <algorithm>
#include <limits>
#include <utility>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/tirx.h>

namespace luisa::compute::tile {

namespace detail {

[[nodiscard]] static bool decode_point(uint64_t linear, const IndexSpace &space, luisa::vector<int64_t> &point) noexcept {
    point.resize(space.rank());
    for (auto i = space.rank(); i != 0u; i--) {
        auto extent = space.axis(i - 1u).extent;
        if (!extent.is_constant() || extent.constant_value() == 0u) { return false; }
        point[i - 1u] = static_cast<int64_t>(linear % extent.constant_value());
        linear /= extent.constant_value();
    }
    return linear == 0u;
}

[[nodiscard]] static luisa::optional<uint64_t> encode_point(luisa::span<const int64_t> point, const IndexSpace &space) noexcept {
    if (point.size() != space.rank()) { return luisa::nullopt; }
    uint64_t linear = 0u;
    for (auto i = 0u; i < point.size(); i++) {
        auto extent = space.axis(i).extent;
        if (!extent.is_constant() || point[i] < 0 || static_cast<uint64_t>(point[i]) >= extent.constant_value()) {
            return luisa::nullopt;
        }
        if (extent.constant_value() != 0u && linear > std::numeric_limits<uint64_t>::max() / extent.constant_value()) {
            return luisa::nullopt;
        }
        linear *= extent.constant_value();
        if (linear > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(point[i])) { return luisa::nullopt; }
        linear += static_cast<uint64_t>(point[i]);
    }
    return linear;
}

[[nodiscard]] bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t &result) noexcept {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) { return false; }
    result = lhs + rhs;
    return true;
}

[[nodiscard]] bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t &result) noexcept {
    if (rhs != 0u && lhs > std::numeric_limits<uint64_t>::max() / rhs) { return false; }
    result = lhs * rhs;
    return true;
}

[[nodiscard]] bool same_point(luisa::span<const int64_t> lhs, luisa::span<const int64_t> rhs) noexcept {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

[[nodiscard]] luisa::string quote_python(luisa::string_view value) noexcept {
    luisa::string result{"\""};
    for (auto c : value) {
        if (c == '\\' || c == '"') { result.push_back('\\'); }
        result.push_back(c);
    }
    result.push_back('"');
    return result;
}

template<typename F>
[[nodiscard]] luisa::string render_tuple(size_t count, F &&render) noexcept {
    luisa::string result{"("};
    for (auto i = 0u; i < count; i++) {
        if (i != 0u) { result.append(", "); }
        result.append(render(i));
    }
    if (count == 1u) { result.push_back(','); }
    result.push_back(')');
    return result;
}

}// namespace detail

bool LayoutCorrespondence::verify() const noexcept {
    return _left.verify() && _right.verify() && _left.domain() == _right.domain();
}

luisa::optional<luisa::vector<luisa::vector<int64_t>>> LayoutCorrespondence::placements(
    luisa::span<const int64_t> logical_point,
    uint64_t max_fiber_points) const noexcept {
    auto fiber_volume = fiber_space().static_volume();
    if (!verify() || !fiber_volume || *fiber_volume > max_fiber_points ||
        !detail::encode_point(logical_point, logical_space())) { return luisa::nullopt; }
    luisa::vector<luisa::vector<int64_t>> result;
    luisa::vector<int64_t> fiber_point;
    for (uint64_t linear = 0u; linear < *fiber_volume; linear++) {
        if (!detail::decode_point(linear, fiber_space(), fiber_point)) { return luisa::nullopt; }
        auto logical = _left.apply(fiber_point);
        if (!logical) { return luisa::nullopt; }
        if (detail::same_point(*logical, logical_point)) {
            auto physical = _right.apply(fiber_point);
            if (!physical) { return luisa::nullopt; }
            auto duplicate = std::find(result.begin(), result.end(), *physical);
            if (duplicate == result.end()) { result.emplace_back(std::move(*physical)); }
        }
    }
    return result;
}

LayoutCorrespondenceProperties LayoutCorrespondence::analyze_finite(uint64_t max_fiber_points) const noexcept {
    LayoutCorrespondenceProperties properties;
    auto fiber_volume = fiber_space().static_volume();
    auto logical_volume = logical_space().static_volume();
    auto physical_volume = physical_space().static_volume();
    if (!verify() || !fiber_volume || !logical_volume || !physical_volume || *fiber_volume > max_fiber_points) {
        return properties;
    }
    properties.enumerated = true;
    properties.total = true;
    properties.fiber_points = *fiber_volume;
    properties.logical_points = *logical_volume;
    properties.physical_points = *physical_volume;
    luisa::vector<std::pair<uint64_t, uint64_t>> placements;
    placements.reserve(*fiber_volume);
    luisa::vector<int64_t> fiber_point;
    for (uint64_t linear = 0u; linear < *fiber_volume; linear++) {
        if (!detail::decode_point(linear, fiber_space(), fiber_point)) {
            properties.total = false;
            break;
        }
        auto logical = _left.apply(fiber_point);
        auto physical = _right.apply(fiber_point);
        if (!logical || !physical) {
            properties.total = false;
            break;
        }
        auto logical_linear = detail::encode_point(*logical, logical_space());
        auto physical_linear = detail::encode_point(*physical, physical_space());
        if (!logical_linear || !physical_linear) {
            properties.total = false;
            break;
        }
        placements.emplace_back(*logical_linear, *physical_linear);
    }
    if (properties.total) {
        std::sort(placements.begin(), placements.end());
        placements.erase(std::unique(placements.begin(), placements.end()), placements.end());
        luisa::vector<uint64_t> multiplicity(*logical_volume, 0u);
        for (auto &&placement : placements) { multiplicity[placement.first]++; }
        properties.covers_logical_space = true;
        properties.minimum_replication = std::numeric_limits<uint64_t>::max();
        for (auto count : multiplicity) {
            properties.covers_logical_space &= count != 0u;
            properties.minimum_replication = std::min(properties.minimum_replication, count);
            properties.maximum_replication = std::max(properties.maximum_replication, count);
        }
        if (multiplicity.empty()) { properties.minimum_replication = 0u; }
    }
    return properties;
}

bool TirxLayoutSpec::add_shard(uint64_t extent, int64_t stride, Dim physical_axis) noexcept {
    if (extent == 0u || stride < 0 || !physical_axis) { return false; }
    _shard.emplace_back(TirxLayoutIter{extent, stride, physical_axis});
    return true;
}

bool TirxLayoutSpec::add_replica(Dim fiber_dimension, uint64_t extent, int64_t stride, Dim physical_axis) noexcept {
    if (!fiber_dimension || extent == 0u || stride < 0 || !physical_axis ||
        _logical_space.contains(fiber_dimension)) { return false; }
    for (auto &&iter : _replica) {
        if (iter.fiber_dimension == fiber_dimension) { return false; }
    }
    _replica.emplace_back(TirxReplicaIter{fiber_dimension, extent, stride, physical_axis});
    return true;
}

bool TirxLayoutSpec::add_offset(Dim physical_axis, int64_t value) noexcept {
    if (!physical_axis || value < 0) { return false; }
    for (auto &&offset : _offsets) {
        if (offset.physical_axis == physical_axis) {
            if (offset.value > std::numeric_limits<int64_t>::max() - value) { return false; }
            offset.value += value;
            return true;
        }
    }
    _offsets.emplace_back(TirxLayoutOffset{physical_axis, value});
    return true;
}

bool TirxLayoutSpec::verify() const noexcept {
    if (!_logical_space.is_valid() || _logical_space.empty() || _shard.empty()) { return false; }
    auto context = _logical_space.axis(0u).dimension.context();
    uint64_t shard_volume = 1u;
    for (auto &&iter : _shard) {
        if (iter.extent == 0u || iter.stride < 0 || !iter.physical_axis ||
            iter.physical_axis.context() != context ||
            !detail::checked_multiply(shard_volume, iter.extent, shard_volume)) { return false; }
    }
    auto logical_volume = _logical_space.static_volume();
    if (!logical_volume || *logical_volume != shard_volume) { return false; }
    luisa::vector<Dim> replica_dimensions;
    for (auto &&iter : _replica) {
        if (!iter.fiber_dimension || iter.fiber_dimension.context() != context ||
            _logical_space.contains(iter.fiber_dimension) || iter.extent == 0u || iter.stride < 0 ||
            !iter.physical_axis || iter.physical_axis.context() != context ||
            std::find(replica_dimensions.begin(), replica_dimensions.end(), iter.fiber_dimension) != replica_dimensions.end()) {
            return false;
        }
        replica_dimensions.emplace_back(iter.fiber_dimension);
    }
    for (auto &&offset : _offsets) {
        if (!offset.physical_axis || offset.physical_axis.context() != context || offset.value < 0) { return false; }
    }
    return true;
}

luisa::optional<LayoutCorrespondence> TirxLayoutSpec::correspondence() const noexcept {
    if (!verify()) { return luisa::nullopt; }
    IndexSpace fiber = _logical_space;
    for (auto &&iter : _replica) {
        if (!fiber.add(iter.fiber_dimension, iter.extent)) { return luisa::nullopt; }
    }

    luisa::vector<Dim> physical_dimensions;
    auto collect_physical = [&physical_dimensions](Dim dimension) noexcept {
        if (std::find(physical_dimensions.begin(), physical_dimensions.end(), dimension) == physical_dimensions.end()) {
            physical_dimensions.emplace_back(dimension);
        }
    };
    for (auto &&iter : _shard) { collect_physical(iter.physical_axis); }
    for (auto &&iter : _replica) { collect_physical(iter.physical_axis); }
    for (auto &&offset : _offsets) { collect_physical(offset.physical_axis); }

    luisa::vector<uint64_t> physical_extents(physical_dimensions.size(), 1u);
    luisa::vector<IndexExpr> physical_expressions;
    physical_expressions.reserve(physical_dimensions.size());
    for (auto dimension : physical_dimensions) {
        uint64_t value = 0u;
        for (auto &&offset : _offsets) {
            if (offset.physical_axis == dimension &&
                !detail::checked_add(value, static_cast<uint64_t>(offset.value), value)) { return luisa::nullopt; }
        }
        physical_expressions.emplace_back(IndexExpr::constant(static_cast<int64_t>(value)));
        physical_extents[physical_expressions.size() - 1u] = value;
    }

    auto add_extent = [&](Dim dimension, uint64_t extent, int64_t stride) noexcept {
        auto index = static_cast<size_t>(std::find(physical_dimensions.begin(), physical_dimensions.end(), dimension) - physical_dimensions.begin());
        uint64_t span;
        uint64_t updated;
        if (!detail::checked_multiply(extent - 1u, static_cast<uint64_t>(stride), span) ||
            !detail::checked_add(physical_extents[index], span, updated)) { return false; }
        physical_extents[index] = updated;
        return true;
    };
    for (auto &&iter : _shard) {
        if (!add_extent(iter.physical_axis, iter.extent, iter.stride)) { return luisa::nullopt; }
    }
    for (auto &&iter : _replica) {
        if (!add_extent(iter.physical_axis, iter.extent, iter.stride)) { return luisa::nullopt; }
    }
    for (auto &extent : physical_extents) {
        if (extent == std::numeric_limits<uint64_t>::max() || extent >= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return luisa::nullopt;
        }
        extent++;
    }

    auto flat = IndexExpr::constant(0);
    for (auto &&axis : _logical_space.axes()) {
        flat = flat * IndexExpr::constant(static_cast<int64_t>(axis.extent.constant_value())) +
               IndexExpr::coordinate(axis.dimension);
    }
    uint64_t suffix = 1u;
    luisa::vector<uint64_t> suffixes(_shard.size(), 1u);
    for (auto i = _shard.size(); i != 0u; i--) {
        suffixes[i - 1u] = suffix;
        if (!detail::checked_multiply(suffix, _shard[i - 1u].extent, suffix) ||
            suffix > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return luisa::nullopt; }
    }
    for (auto i = 0u; i < _shard.size(); i++) {
        auto &&iter = _shard[i];
        auto physical_index = static_cast<size_t>(std::find(physical_dimensions.begin(), physical_dimensions.end(), iter.physical_axis) - physical_dimensions.begin());
        auto component = modulo(
            floor_div(flat, IndexExpr::constant(static_cast<int64_t>(suffixes[i]))),
            IndexExpr::constant(static_cast<int64_t>(iter.extent)));
        physical_expressions[physical_index] = physical_expressions[physical_index] +
                                               component * IndexExpr::constant(iter.stride);
    }
    for (auto &&iter : _replica) {
        auto physical_index = static_cast<size_t>(std::find(physical_dimensions.begin(), physical_dimensions.end(), iter.physical_axis) - physical_dimensions.begin());
        physical_expressions[physical_index] = physical_expressions[physical_index] +
                                               IndexExpr::coordinate(iter.fiber_dimension) * IndexExpr::constant(iter.stride);
    }

    IndexSpace physical;
    for (auto i = 0u; i < physical_dimensions.size(); i++) {
        if (!physical.add(physical_dimensions[i], physical_extents[i])) { return luisa::nullopt; }
    }
    luisa::vector<IndexExpr> logical_expressions;
    logical_expressions.reserve(_logical_space.rank());
    for (auto &&axis : _logical_space.axes()) { logical_expressions.emplace_back(IndexExpr::coordinate(axis.dimension)); }
    IndexMap left{fiber, _logical_space, logical_expressions};
    IndexMap right{fiber, std::move(physical), physical_expressions};
    LayoutCorrespondence result{std::move(left), std::move(right)};
    return result.verify() ? luisa::optional<LayoutCorrespondence>{std::move(result)} : luisa::nullopt;
}

TirxLayoutExport export_tirx_layout(const TirxLayoutSpec &layout, luisa::span<const TirxAxisBinding> axes) noexcept {
    TirxLayoutExport result;
    if (!layout.verify()) {
        result.error = "cannot export an invalid or non-static TIRx layout";
        return result;
    }
    auto axis_name = [&](Dim dimension) noexcept -> luisa::optional<luisa::string_view> {
        luisa::optional<luisa::string_view> found;
        for (auto &&binding : axes) {
            if (binding.physical_axis == dimension) {
                if (binding.name.empty() || found) { return luisa::nullopt; }
                found = luisa::string_view{binding.name};
            }
        }
        return found;
    };
    luisa::vector<Dim> required_axes;
    auto require_axis = [&required_axes](Dim dimension) noexcept {
        if (std::find(required_axes.begin(), required_axes.end(), dimension) == required_axes.end()) {
            required_axes.emplace_back(dimension);
        }
    };
    for (auto &&iter : layout.shard()) { require_axis(iter.physical_axis); }
    for (auto &&iter : layout.replica()) { require_axis(iter.physical_axis); }
    for (auto &&offset : layout.offsets()) { require_axis(offset.physical_axis); }
    luisa::unordered_map<uint32_t, luisa::string> names;
    luisa::vector<luisa::string_view> used_names;
    for (auto dimension : required_axes) {
        auto name = axis_name(dimension);
        if (!name) {
            result.error = "every physical layout dimension must have exactly one non-empty TIRx axis binding";
            return result;
        }
        if (std::find(used_names.begin(), used_names.end(), *name) != used_names.end()) {
            result.error = "different physical layout dimensions cannot share a TIRx axis name";
            return result;
        }
        used_names.emplace_back(*name);
        names.emplace(dimension.index(), detail::quote_python(*name));
    }
    auto render_axis_term = [&names](int64_t stride, Dim dimension) noexcept {
        return luisa::format("{} @ _TileAxis.get({})", stride, names.at(dimension.index()));
    };
    auto shard_extents = detail::render_tuple(layout.shard().size(), [&](size_t i) noexcept {
        return luisa::format("{}", layout.shard()[i].extent);
    });
    auto shard_strides = detail::render_tuple(layout.shard().size(), [&](size_t i) noexcept {
        auto &&iter = layout.shard()[i];
        return render_axis_term(iter.stride, iter.physical_axis);
    });
    auto specification = luisa::format("T.S[{} : {}]", shard_extents, shard_strides);
    if (!layout.replica().empty()) {
        auto replica_extents = detail::render_tuple(layout.replica().size(), [&](size_t i) noexcept {
            return luisa::format("{}", layout.replica()[i].extent);
        });
        auto replica_strides = detail::render_tuple(layout.replica().size(), [&](size_t i) noexcept {
            auto &&iter = layout.replica()[i];
            return render_axis_term(iter.stride, iter.physical_axis);
        });
        specification.append(luisa::format(" + T.R[{} : {}]", replica_extents, replica_strides));
    }
    for (auto &&offset : layout.offsets()) {
        if (offset.value != 0) {
            specification.append(luisa::format(" + {}", render_axis_term(offset.value, offset.physical_axis)));
        }
    }
    result.preamble = "from tvm.script import tirx as T\nfrom tvm.tirx.layout import Axis as _TileAxis\n";
    result.expression = luisa::format("T.TileLayout({})", specification);
    return result;
}

}// namespace luisa::compute::tile
