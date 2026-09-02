#include <algorithm>
#include <exception>
#include <limits>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/bridge/tirx/layout.h>

namespace luisa::compute::tile::bridge::tirx {

namespace detail {

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

}// namespace detail

bool LayoutSpec::add_shard(uint64_t extent, int64_t stride, Dim physical_axis) noexcept {
    if (extent == 0u || stride < 0 || !physical_axis) { return false; }
    _shard.emplace_back(LayoutIter{extent, stride, physical_axis});
    return true;
}

bool LayoutSpec::add_replica(Dim fiber_dimension, uint64_t extent, int64_t stride, Dim physical_axis) noexcept {
    if (!fiber_dimension || extent == 0u || stride < 0 || !physical_axis ||
        _logical_space.contains(fiber_dimension)) { return false; }
    for (auto &&iter : _replica) {
        if (iter.fiber_dimension == fiber_dimension) { return false; }
    }
    _replica.emplace_back(ReplicaIter{fiber_dimension, extent, stride, physical_axis});
    return true;
}

bool LayoutSpec::add_offset(Dim physical_axis, int64_t value) noexcept {
    if (!physical_axis || value < 0) { return false; }
    for (auto &&offset : _offsets) {
        if (offset.physical_axis == physical_axis) {
            if (offset.value > std::numeric_limits<int64_t>::max() - value) { return false; }
            offset.value += value;
            return true;
        }
    }
    _offsets.emplace_back(LayoutOffset{physical_axis, value});
    return true;
}

bool LayoutSpec::verify() const noexcept {
    if (!_logical_space.is_valid() || _logical_space.empty() || _shard.empty()) { return false; }
    auto context = _logical_space.axis(0u).dimension.context();
    uint64_t shard_volume = 1u;
    for (auto &&iter : _shard) {
        if (iter.extent == 0u || iter.extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            iter.stride < 0 || !iter.physical_axis ||
            iter.physical_axis.context() != context ||
            !detail::checked_multiply(shard_volume, iter.extent, shard_volume)) { return false; }
    }
    auto logical_volume = _logical_space.static_volume();
    if (!logical_volume || *logical_volume != shard_volume) { return false; }
    luisa::vector<Dim> replica_dimensions;
    for (auto &&iter : _replica) {
        if (!iter.fiber_dimension || iter.fiber_dimension.context() != context ||
            _logical_space.contains(iter.fiber_dimension) || iter.extent == 0u ||
            iter.extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) || iter.stride < 0 ||
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

luisa::optional<LayoutCorrespondence> LayoutSpec::correspondence() const noexcept {
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
        if (extent == std::numeric_limits<uint64_t>::max() ||
            extent >= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return luisa::nullopt; }
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
    for (auto &&axis : _logical_space.axes()) {
        logical_expressions.emplace_back(IndexExpr::coordinate(axis.dimension));
    }
    IndexMap left{fiber, _logical_space, logical_expressions};
    IndexMap right{fiber, std::move(physical), physical_expressions};
    LayoutCorrespondence result{std::move(left), std::move(right)};
    return result.verify() ? luisa::optional<LayoutCorrespondence>{std::move(result)} : luisa::nullopt;
}

NativeLayout export_layout(const LayoutSpec &layout, luisa::span<const AxisBinding> axes) noexcept {
    NativeLayout result;
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
        names.emplace(dimension.index(), luisa::string{*name});
    }
    try {
        luisa::unordered_map<uint32_t, tvm::tirx::Axis> native_axes;
        for (auto dimension : required_axes) {
            native_axes.emplace(dimension.index(), tvm::tirx::Axis::Get(names.at(dimension.index()).c_str()));
        }
        tvm::ffi::Array<tvm::tirx::Iter> shard;
        for (auto &&iter : layout.shard()) {
            shard.push_back(tvm::tirx::Iter{
                tvm::IntImm::Int64(static_cast<int64_t>(iter.extent)),
                tvm::IntImm::Int64(iter.stride),
                native_axes.at(iter.physical_axis.index())});
        }
        tvm::ffi::Array<tvm::tirx::Iter> replica;
        for (auto &&iter : layout.replica()) {
            replica.push_back(tvm::tirx::Iter{
                tvm::IntImm::Int64(static_cast<int64_t>(iter.extent)),
                tvm::IntImm::Int64(iter.stride),
                native_axes.at(iter.physical_axis.index())});
        }
        tvm::ffi::Map<tvm::tirx::Axis, tvm::PrimExpr> offsets;
        for (auto &&offset : layout.offsets()) {
            auto axis = native_axes.at(offset.physical_axis.index());
            offsets.Set(axis, tvm::IntImm::Int64(offset.value));
        }
        result.value = tvm::tirx::TileLayout{std::move(shard), std::move(replica), std::move(offsets)};
    } catch (const std::exception &error) {
        result.error = error.what();
    } catch (...) {
        result.error = "TVM TIRx rejected the native layout";
    }
    return result;
}

}// namespace luisa::compute::tile::bridge::tirx
