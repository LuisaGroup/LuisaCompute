#pragma once

#include <cstdint>
#include <utility>

#include <tvm/tirx/layout.h>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/layout.h>

namespace luisa::compute::tile::bridge::tirx {

struct LayoutIter {
    uint64_t extent{0u};
    int64_t stride{0};
    Dim physical_axis;
};

struct ReplicaIter {
    Dim fiber_dimension;
    uint64_t extent{0u};
    int64_t stride{0};
    Dim physical_axis;
};

struct LayoutOffset {
    Dim physical_axis;
    int64_t value{0};
};

// The factored subset shared exactly with tvm::tirx::TileLayout: shard iters,
// replica iters, and per-axis offsets. More general Tile IndexMaps remain legal
// and lower as explicit TIRx index arithmetic instead of being approximated.
class LUISA_TILE_TIRX_BRIDGE_API LayoutSpec final {

private:
    IndexSpace _logical_space;
    luisa::vector<LayoutIter> _shard;
    luisa::vector<ReplicaIter> _replica;
    luisa::vector<LayoutOffset> _offsets;

public:
    explicit LayoutSpec(IndexSpace logical_space) noexcept
        : _logical_space{std::move(logical_space)} {}

    [[nodiscard]] const IndexSpace &logical_space() const noexcept { return _logical_space; }
    [[nodiscard]] luisa::span<const LayoutIter> shard() const noexcept { return _shard; }
    [[nodiscard]] luisa::span<const ReplicaIter> replica() const noexcept { return _replica; }
    [[nodiscard]] luisa::span<const LayoutOffset> offsets() const noexcept { return _offsets; }

    [[nodiscard]] bool add_shard(uint64_t extent, int64_t stride, Dim physical_axis) noexcept;
    [[nodiscard]] bool add_replica(Dim fiber_dimension, uint64_t extent, int64_t stride, Dim physical_axis) noexcept;
    [[nodiscard]] bool add_offset(Dim physical_axis, int64_t value) noexcept;
    [[nodiscard]] bool verify() const noexcept;
    [[nodiscard]] luisa::optional<LayoutCorrespondence> correspondence() const noexcept;
};

struct AxisBinding {
    Dim physical_axis;
    luisa::string name;
};

struct NativeLayout {
    tvm::tirx::TileLayout value;
    luisa::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty() && value.defined(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct NativeIndices {
    tvm::ffi::Array<tvm::PrimExpr> value;
    luisa::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Lower a general IndexMap as native signed-64-bit coordinate arithmetic.
// This does not assert injectivity or pretend that XOR maps are IterSumExprs
// accepted by TIRx IndexMap inverse analysis. Memory realization separately
// requires a total, in-bounds, injective address map on its logical domain.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API NativeIndices lower_index_map(
    const IndexMap &map,
    const tvm::ffi::Array<tvm::PrimExpr> &coordinates) noexcept;

// Constructs a native tvm::tirx::TileLayout directly through TVM's public C++
// API. Axis bindings are intentionally external: Dim labels never acquire
// hardware meaning merely because they happen to be named "lane" or "m".
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API NativeLayout export_layout(
    const LayoutSpec &layout,
    luisa::span<const AxisBinding> axes) noexcept;

}// namespace luisa::compute::tile::bridge::tirx
