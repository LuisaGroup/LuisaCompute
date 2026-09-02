// Tests for the exact Tile layout correspondence -> native TVM TIRx bridge.

#include "ut/ut.hpp"

#include <limits>
#include <utility>

#include <luisa/tile/bridge/tirx/layout.h>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct TiledReplicaFixture {
    DimensionContext dimensions;
    Dim row = dimensions.create_dimension("row");
    Dim column = dimensions.create_dimension("column");
    Dim replica = dimensions.create_dimension("replica");
    Dim lane = dimensions.create_dimension("physical lane");
    Dim warp = dimensions.create_dimension("physical warp");
    Dim slot = dimensions.create_dimension("physical slot");
    IndexSpace logical;
    LayoutSpec layout;

    TiledReplicaFixture() noexcept
        : layout{[&] {
              static_cast<void>(logical.add(row, 8u));
              static_cast<void>(logical.add(column, 16u));
              return logical;
          }()} {
        // Official TIRx documentation example:
        // S[(8,2,4,2):(4@laneid,1@warpid,1@laneid,1@m)]
        // + R[2:4@warpid] + 5@warpid
        static_cast<void>(layout.add_shard(8u, 4, lane));
        static_cast<void>(layout.add_shard(2u, 1, warp));
        static_cast<void>(layout.add_shard(4u, 1, lane));
        static_cast<void>(layout.add_shard(2u, 1, slot));
        static_cast<void>(layout.add_replica(replica, 2u, 4, warp));
        static_cast<void>(layout.add_offset(warp, 5));
    }

    [[nodiscard]] luisa::vector<AxisBinding> bindings() const noexcept {
        return {{lane, "laneid"}, {warp, "warpid"}, {slot, "m"}};
    }
};

[[nodiscard]] bool contains_point(
    luisa::span<const luisa::vector<int64_t>> points,
    std::initializer_list<int64_t> expected) noexcept {
    luisa::vector<int64_t> value{expected};
    for (auto &&point : points) {
        if (point == value) { return true; }
    }
    return false;
}

void test_correspondence() {
    TiledReplicaFixture fixture;
    expect(fixture.layout.verify());
    auto correspondence = fixture.layout.correspondence();
    expect(correspondence.has_value());
    expect(correspondence->verify());

    auto properties = correspondence->analyze_finite();
    expect(properties.enumerated);
    expect(properties.total);
    expect(properties.covers_logical_space);
    expect(eq(properties.fiber_points, 256u));
    expect(eq(properties.logical_points, 128u));
    expect(eq(properties.minimum_replication, 2u));
    expect(eq(properties.maximum_replication, 2u));

    int64_t first[]{0, 0};
    auto first_placements = correspondence->placements(first);
    expect(first_placements.has_value());
    expect(eq(first_placements->size(), 2u));
    expect(contains_point(*first_placements, {0, 5, 0}));
    expect(contains_point(*first_placements, {0, 9, 0}));

    int64_t last[]{7, 15};
    auto last_placements = correspondence->placements(last);
    expect(last_placements.has_value());
    expect(eq(last_placements->size(), 2u));
    expect(contains_point(*last_placements, {31, 6, 1}));
    expect(contains_point(*last_placements, {31, 10, 1}));
}

void test_coincident_replica_witnesses() {
    DimensionContext dimensions;
    auto logical_dimension = dimensions.create_dimension("logical");
    auto replica_dimension = dimensions.create_dimension("replica");
    auto physical_dimension = dimensions.create_dimension("physical");
    IndexSpace logical;
    expect(logical.add(logical_dimension, 4u));
    LayoutSpec layout{logical};
    expect(layout.add_shard(4u, 1, physical_dimension));
    expect(layout.add_replica(replica_dimension, 3u, 0, physical_dimension));
    auto correspondence = layout.correspondence();
    expect(correspondence.has_value());
    auto properties = correspondence->analyze_finite();
    expect(properties.covers_logical_space);
    expect(eq(properties.fiber_points, 12u));
    expect(eq(properties.minimum_replication, 1u));
    expect(eq(properties.maximum_replication, 1u));
}

[[gnu::noinline]] bool verify_native_layout(const tvm::tirx::Layout &layout) {
    return layout->VerifyWellFormed();
}

[[gnu::noinline]] tvm::ffi::Map<tvm::ffi::String, tvm::PrimExpr> apply_native_layout(
    const tvm::tirx::Layout &layout,
    tvm::ffi::Array<tvm::PrimExpr> coordinate,
    tvm::ffi::Array<tvm::PrimExpr> shape) {
    return layout->Apply(std::move(coordinate), std::move(shape));
}

[[nodiscard]] int64_t native_coordinate(
    const tvm::ffi::Map<tvm::ffi::String, tvm::PrimExpr> &placement,
    const char *axis) {
    auto value = placement.Get(tvm::ffi::String{axis});
    if (!value) { return std::numeric_limits<int64_t>::min(); }
    auto constant = value->as<tvm::IntImmNode>();
    return constant == nullptr ? std::numeric_limits<int64_t>::min() : constant->value;
}

void test_native_export() {
    TiledReplicaFixture fixture;
    auto bindings = fixture.bindings();
    auto exported = export_layout(fixture.layout, bindings);
    expect(exported.ok());
    tvm::tirx::Layout native = exported.value;
    expect(verify_native_layout(native));
    auto first = apply_native_layout(
        native,
        {tvm::IntImm::Int64(0), tvm::IntImm::Int64(0)},
        {tvm::IntImm::Int64(8), tvm::IntImm::Int64(16)});
    expect(eq(native_coordinate(first, "m"), int64_t{0}));
    expect(eq(native_coordinate(first, "warpid"), int64_t{5}));
    expect(eq(native_coordinate(first, "laneid"), int64_t{0}));
    auto last = apply_native_layout(
        native,
        {tvm::IntImm::Int64(7), tvm::IntImm::Int64(15)},
        {tvm::IntImm::Int64(8), tvm::IntImm::Int64(16)});
    expect(eq(native_coordinate(last, "m"), int64_t{1}));
    expect(eq(native_coordinate(last, "warpid"), int64_t{6}));
    expect(eq(native_coordinate(last, "laneid"), int64_t{31}));

    auto missing = bindings;
    missing.pop_back();
    expect(!export_layout(fixture.layout, missing).ok());
    auto duplicate = bindings;
    duplicate.emplace_back(AxisBinding{fixture.lane, "another_lane"});
    expect(!export_layout(fixture.layout, duplicate).ok());
    auto alias = bindings;
    alias[2].name = "laneid";
    expect(!export_layout(fixture.layout, alias).ok());
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_tirx_correspondence"_test = test_correspondence;
    "tile_tirx_coincident_replica_witnesses"_test = test_coincident_replica_witnesses;
    "tile_tirx_native_export"_test = test_native_export;
}
