#pragma once

#include <limits>
#include <utility>
#include <luisa/core/mathematics.h>
#include <luisa/tile/dimension.h>

namespace luisa::compute::tile::bridge::xir::detail {

// A logical team geometry, not a target capability or whole-program admission.
// The backend must separately validate the physical packet/launch contract.
class ProgramTeamLayout final {
private:
    uint32_t _width;

    explicit ProgramTeamLayout(uint32_t width) noexcept : _width{width} {}

public:
    [[nodiscard]] static luisa::optional<ProgramTeamLayout> create(uint32_t width) noexcept {
        if (width == 0u || (width & (width - 1u)) != 0u) { return {}; }
        return ProgramTeamLayout{width};
    }
    [[nodiscard]] uint32_t width() const noexcept { return _width; }
    [[nodiscard]] friend bool operator==(ProgramTeamLayout, ProgramTeamLayout) noexcept = default;
};

// Facts supplied by the later access analysis. In particular, matching axis
// names alone do not prove OWNER_PRESERVING, and TEAM_UNIFORM means the complete
// projection, not only its distributed coordinate, is uniform. This layer does
// not establish bounds, convergence, effects, or the validity of those facts.
enum class ReadProjection : uint8_t {
    UNKNOWN,
    OWNER_PRESERVING,
    TEAM_UNIFORM
};

enum class ReadTransition : uint8_t {
    UNSUPPORTED,
    REPLICATED_LOCAL,
    OWNER_LOCAL,
    UNIFORM_BROADCAST
};

struct OwnerSlot {
    uint32_t owner;
    uint64_t local_flat;

    [[nodiscard]] friend bool operator==(const OwnerSlot &, const OwnerSlot &) noexcept = default;
};

// Immutable geometry for one SSA definition or carry slot. Dimension identities
// are borrowed from the owning TileIR context; that context must outlive this
// layout. Extents are copied, so later IndexSpace edits cannot change a plan.
// Empty static shapes are representable but have no valid coordinates/slots.
class ValueLayout final {
public:
    enum class Kind : uint8_t {
        REPLICATED,
        CYCLIC
    };

private:
    ProgramTeamLayout _team;
    IndexSpace _space;
    luisa::optional<size_t> _cyclic_axis;
    luisa::vector<uint64_t> _local_extents;
    uint64_t _logical_elements;
    uint64_t _local_elements;

    ValueLayout(ProgramTeamLayout team, const IndexSpace &space, luisa::optional<size_t> cyclic_axis,
                luisa::vector<uint64_t> local_extents, uint64_t logical_elements, uint64_t local_elements) noexcept
        : _team{team}, _space{space}, _cyclic_axis{cyclic_axis}, _local_extents{std::move(local_extents)},
          _logical_elements{logical_elements}, _local_elements{local_elements} {}

    [[nodiscard]] static luisa::optional<ValueLayout> _create(
        ProgramTeamLayout team, const IndexSpace &space, luisa::optional<Dim> cyclic_dimension) noexcept {
        if (!space.is_valid()) { return {}; }
        for (auto &axis : space.axes()) {
            if (!axis.extent.is_constant()) { return {}; }
        }
        auto logical_elements = space.static_volume();
        if (!logical_elements) { return {}; }
        luisa::optional<size_t> cyclic_axis;
        if (cyclic_dimension) {
            cyclic_axis = space.axis_index(*cyclic_dimension);
            if (!cyclic_axis) { return {}; }
        }
        luisa::vector<uint64_t> local_extents;
        local_extents.reserve(space.rank());
        auto local_elements = *logical_elements == 0u ? uint64_t{0u} : uint64_t{1u};
        for (size_t i = 0u; i < space.rank(); i++) {
            auto extent = space.axis(i).extent.constant_value();
            if (cyclic_axis && i == *cyclic_axis) { extent = ceil_div(extent, static_cast<uint64_t>(team.width())); }
            local_extents.emplace_back(extent);
            if (local_elements != 0u) {
                if (extent > std::numeric_limits<uint64_t>::max() / local_elements) { return {}; }
                local_elements *= extent;
            }
        }
        return ValueLayout{team, space, cyclic_axis, std::move(local_extents), *logical_elements, local_elements};
    }

public:
    [[nodiscard]] static luisa::optional<ValueLayout> replicated(ProgramTeamLayout team, const IndexSpace &space) noexcept {
        return _create(team, space, {});
    }
    [[nodiscard]] static luisa::optional<ValueLayout> cyclic(ProgramTeamLayout team, const IndexSpace &space, Dim axis) noexcept {
        return _create(team, space, axis);
    }
    [[nodiscard]] ProgramTeamLayout team() const noexcept { return _team; }
    [[nodiscard]] Kind kind() const noexcept { return _cyclic_axis ? Kind::CYCLIC : Kind::REPLICATED; }
    [[nodiscard]] const IndexSpace &space() const noexcept { return _space; }
    [[nodiscard]] luisa::optional<size_t> cyclic_axis_index() const noexcept { return _cyclic_axis; }
    [[nodiscard]] luisa::optional<Dim> cyclic_axis() const noexcept {
        return _cyclic_axis ? luisa::optional<Dim>{_space.axis(*_cyclic_axis).dimension} : luisa::nullopt;
    }
    [[nodiscard]] luisa::span<const uint64_t> local_extents() const noexcept { return _local_extents; }
    [[nodiscard]] uint64_t logical_elements() const noexcept { return _logical_elements; }
    [[nodiscard]] uint64_t local_elements() const noexcept { return _local_elements; }

    // Replicated values use the requesting lane's copy, not an invented unique
    // owner. Cyclic ownership is independent of the requesting lane. This is a
    // checked host mapping, not a proof about symbolic TileIR projections.
    [[nodiscard]] luisa::optional<OwnerSlot> map(luisa::span<const uint64_t> coordinates, uint32_t requesting_lane) const noexcept {
        if (requesting_lane >= _team.width() || coordinates.size() != _space.rank() || _logical_elements == 0u) { return {}; }
        OwnerSlot result{requesting_lane, 0u};
        for (size_t i = 0u; i < coordinates.size(); i++) {
            auto coordinate = coordinates[i];
            if (coordinate >= _space.axis(i).extent.constant_value()) { return {}; }
            if (_cyclic_axis && i == *_cyclic_axis) {
                result.owner = static_cast<uint32_t>(coordinate % _team.width());
                coordinate /= _team.width();
            }
            // Validated local volume and bounded coordinates make this exact.
            result.local_flat = result.local_flat * _local_extents[i] + coordinate;
        }
        return result;
    }

    // Invert a physical lane/slot pair. Padding slots on a ragged cyclic axis
    // return no logical coordinate; they must never become reduction terms.
    [[nodiscard]] luisa::optional<luisa::vector<uint64_t>> unmap(uint32_t owner, uint64_t local_flat) const noexcept {
        if (owner >= _team.width() || local_flat >= _local_elements) { return {}; }
        luisa::vector<uint64_t> coordinates(_space.rank());
        for (auto i = _space.rank(); i != 0u; i--) {
            auto coordinate = local_flat % _local_extents[i - 1u];
            local_flat /= _local_extents[i - 1u];
            if (_cyclic_axis && i - 1u == *_cyclic_axis) {
                if (coordinate > (std::numeric_limits<uint64_t>::max() - owner) / _team.width()) { return {}; }
                coordinate = coordinate * _team.width() + owner;
                if (coordinate >= _space.axis(i - 1u).extent.constant_value()) { return {}; }
            }
            coordinates[i - 1u] = coordinate;
        }
        return coordinates;
    }

    [[nodiscard]] ReadTransition read_transition(ReadProjection projection) const noexcept {
        if (!_cyclic_axis) { return ReadTransition::REPLICATED_LOCAL; }
        switch (projection) {
            case ReadProjection::OWNER_PRESERVING: return ReadTransition::OWNER_LOCAL;
            case ReadProjection::TEAM_UNIFORM: return ReadTransition::UNIFORM_BROADCAST;
            case ReadProjection::UNKNOWN: return ReadTransition::UNSUPPORTED;
        }
        return ReadTransition::UNSUPPORTED;
    }
};

}// namespace luisa::compute::tile::bridge::xir::detail
