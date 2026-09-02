// Tests for the exact Tile layout correspondence -> TVM TIRx bridge.

#include "ut/ut.hpp"

#include <iostream>

#include <luisa/tile/tirx.h>

using namespace luisa;
using namespace luisa::compute::tile;
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
    TirxLayoutSpec layout;

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

    [[nodiscard]] luisa::vector<TirxAxisBinding> bindings() const noexcept {
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

void test_tirx_correspondence() {
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

    auto converse = correspondence->converse();
    expect(converse.verify());
    expect(converse.logical_space() == correspondence->physical_space());
    expect(converse.physical_space() == correspondence->logical_space());
}

void test_coincident_replica_witnesses() {
    DimensionContext dimensions;
    auto logical_dimension = dimensions.create_dimension("logical");
    auto replica_dimension = dimensions.create_dimension("replica");
    auto physical_dimension = dimensions.create_dimension("physical");
    IndexSpace logical;
    expect(logical.add(logical_dimension, 4u));
    TirxLayoutSpec layout{logical};
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

void test_tirx_export() {
    TiledReplicaFixture fixture;
    auto bindings = fixture.bindings();
    auto exported = export_tirx_layout(fixture.layout, bindings);
    expect(exported.ok());
    expect(exported.preamble.find("tvm.script import tirx") != luisa::string::npos);
    expect(exported.expression.find("T.TileLayout") != luisa::string::npos);
    expect(exported.expression.find("T.S[(8, 2, 4, 2)") != luisa::string::npos);
    expect(exported.expression.find("T.R[(2,)") != luisa::string::npos);
    expect(exported.expression.find("_TileAxis.get(\"laneid\")") != luisa::string::npos);
    expect(exported.expression.find("_TileAxis.get(\"warpid\")") != luisa::string::npos);

    auto missing = bindings;
    missing.pop_back();
    expect(!export_tirx_layout(fixture.layout, missing).ok());
    auto duplicate = bindings;
    duplicate.emplace_back(TirxAxisBinding{fixture.lane, "another_lane"});
    expect(!export_tirx_layout(fixture.layout, duplicate).ok());
    auto alias = bindings;
    alias[2].name = "laneid";
    expect(!export_tirx_layout(fixture.layout, alias).ok());
}

void emit_python_probe() {
    TiledReplicaFixture fixture;
    auto bindings = fixture.bindings();
    auto exported = export_tirx_layout(fixture.layout, bindings);
    if (!exported) {
        std::cerr << exported.error << '\n';
        return;
    }
    std::cout << exported.preamble
              << "layout = " << exported.expression << '\n'
              << "assert layout.verify_well_formed()\n"
              << "assert {str(k): int(v) for k, v in layout.apply(0, 0, shape=[8, 16]).items()} == "
                 "{'m': 0, 'warpid': 5, 'laneid': 0}\n"
              << "assert {str(k): int(v) for k, v in layout.apply(7, 15, shape=[8, 16]).items()} == "
                 "{'m': 1, 'warpid': 6, 'laneid': 31}\n"
              << "assert len(layout.replica) == 1\n"
              << "print('TIRx layout bridge OK:', layout)\n";
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc == 2) {
        if (argv[1] != nullptr && luisa::string_view{argv[1]} == "--emit-python") {
            emit_python_probe();
            return 0;
        }
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_tirx_correspondence"_test = test_tirx_correspondence;
    "tile_tirx_coincident_replica_witnesses"_test = test_coincident_replica_witnesses;
    "tile_tirx_export"_test = test_tirx_export;
}
