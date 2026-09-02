#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/layout.h>

namespace luisa::compute::tile {

// A layout relation is represented by a span F and two ordinary maps:
//
//   left  : F -> logical coordinates
//   right : F -> physical coordinates
//
// Unlike an inverse layout map, this representation is exact for replication,
// broadcast, and non-bijective placement. The fiber witnesses in F are an IR
// construction detail; placements() returns a set and removes duplicate
// witnesses.
struct LayoutCorrespondenceProperties {
    bool enumerated{false};
    bool total{false};
    bool covers_logical_space{false};
    uint64_t fiber_points{0u};
    uint64_t logical_points{0u};
    uint64_t physical_points{0u};
    // Number of distinct physical placements per logical point. Duplicate
    // fiber witnesses do not count as extra replicas.
    uint64_t minimum_replication{0u};
    uint64_t maximum_replication{0u};
};

class LUISA_TILE_API LayoutCorrespondence final {

private:
    IndexMap _left;
    IndexMap _right;

public:
    LayoutCorrespondence() noexcept = default;
    LayoutCorrespondence(IndexMap left, IndexMap right) noexcept
        : _left{std::move(left)}, _right{std::move(right)} {}

    [[nodiscard]] const IndexSpace &fiber_space() const noexcept { return _left.domain(); }
    [[nodiscard]] const IndexSpace &logical_space() const noexcept { return _left.codomain(); }
    [[nodiscard]] const IndexSpace &physical_space() const noexcept { return _right.codomain(); }
    [[nodiscard]] const IndexMap &left_leg() const noexcept { return _left; }
    [[nodiscard]] const IndexMap &right_leg() const noexcept { return _right; }

    [[nodiscard]] bool verify() const noexcept;
    [[nodiscard]] LayoutCorrespondence converse() const noexcept { return LayoutCorrespondence{_right, _left}; }
    [[nodiscard]] luisa::optional<luisa::vector<luisa::vector<int64_t>>> placements(
        luisa::span<const int64_t> logical_point,
        uint64_t max_fiber_points = 1024u * 1024u) const noexcept;
    [[nodiscard]] LayoutCorrespondenceProperties analyze_finite(
        uint64_t max_fiber_points = 1024u * 1024u) const noexcept;
};

struct TirxLayoutIter {
    uint64_t extent{0u};
    int64_t stride{0};
    Dim physical_axis;
};

struct TirxReplicaIter {
    Dim fiber_dimension;
    uint64_t extent{0u};
    int64_t stride{0};
    Dim physical_axis;
};

struct TirxLayoutOffset {
    Dim physical_axis;
    int64_t value{0};
};

// The factored subset shared exactly with tvm.tirx.layout.TileLayout:
// shard iters + replica iters + per-axis offsets. More general IndexMaps stay
// legal Tile layouts, but export as explicit TIRx index arithmetic instead of
// being approximated as TileLayout.
class LUISA_TILE_API TirxLayoutSpec final {

private:
    IndexSpace _logical_space;
    luisa::vector<TirxLayoutIter> _shard;
    luisa::vector<TirxReplicaIter> _replica;
    luisa::vector<TirxLayoutOffset> _offsets;

public:
    explicit TirxLayoutSpec(IndexSpace logical_space) noexcept
        : _logical_space{std::move(logical_space)} {}

    [[nodiscard]] const IndexSpace &logical_space() const noexcept { return _logical_space; }
    [[nodiscard]] luisa::span<const TirxLayoutIter> shard() const noexcept { return _shard; }
    [[nodiscard]] luisa::span<const TirxReplicaIter> replica() const noexcept { return _replica; }
    [[nodiscard]] luisa::span<const TirxLayoutOffset> offsets() const noexcept { return _offsets; }

    [[nodiscard]] bool add_shard(uint64_t extent, int64_t stride, Dim physical_axis) noexcept;
    [[nodiscard]] bool add_replica(Dim fiber_dimension, uint64_t extent, int64_t stride, Dim physical_axis) noexcept;
    [[nodiscard]] bool add_offset(Dim physical_axis, int64_t value) noexcept;
    [[nodiscard]] bool verify() const noexcept;
    [[nodiscard]] luisa::optional<LayoutCorrespondence> correspondence() const noexcept;
};

struct TirxAxisBinding {
    Dim physical_axis;
    luisa::string name;
};

struct TirxLayoutExport {
    luisa::string preamble;
    luisa::string expression;
    luisa::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty() && !expression.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Emits the current TVM 0.26 TIRx API, not the legacy tvm.tir surface. The
// expression assumes the returned preamble has been emitted once at module
// scope. Axis bindings are intentionally external: Dim labels never acquire
// hardware meaning merely because they happen to be named "lane" or "m".
[[nodiscard]] LUISA_TILE_API TirxLayoutExport export_tirx_layout(
    const TirxLayoutSpec &layout,
    luisa::span<const TirxAxisBinding> axes) noexcept;

}// namespace luisa::compute::tile
