#include <array>
#include <limits>

#include <luisa/tile/bridge/tirx/layout.h>

namespace luisa::compute::tile::bridge::tirx {

namespace {

[[nodiscard]] bool valid_mapping(const MatrixWorkload &workload, MatrixDistribution distribution) noexcept {
    auto groups = static_cast<uint64_t>(distribution.subgroups_m) * distribution.subgroups_n;
    if (!distribution.rectangular() || groups > std::numeric_limits<uint32_t>::max() / 32u ||
        workload.rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        workload.columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return false; }
    // Time residency does not change the spatial relation.
    distribution.persistent_accumulator = false;
    return verify_matrix_distribution(workload, distribution, static_cast<uint32_t>(groups * 32u), 32u);
}

}// namespace

NativeLayout matrix_distribution_layout(const MatrixWorkload &workload, const MatrixDistribution &distribution) noexcept {
    if (!valid_mapping(workload, distribution)) { return {{}, "invalid rectangular matrix execution map"}; }
    DimensionContext dimensions;
    auto row = dimensions.create_dimension();
    auto column = dimensions.create_dimension();
    auto subgroup = dimensions.create_dimension();
    auto fragment = dimensions.create_dimension();
    IndexSpace atoms;
    static_cast<void>(atoms.add(row, workload.rows / 8u));
    static_cast<void>(atoms.add(column, workload.columns / 8u));
    LayoutSpec layout{atoms};
    static_cast<void>(layout.add_shard(distribution.subgroups_m, distribution.subgroups_n, subgroup));
    static_cast<void>(layout.add_shard(distribution.atom_rows, static_cast<int64_t>(distribution.atom_columns), fragment));
    static_cast<void>(layout.add_shard(distribution.subgroups_n, 1, subgroup));
    static_cast<void>(layout.add_shard(distribution.atom_columns, 1, fragment));
    // TIRx regrouping can remove extent-one shards. Keep both coordinates in
    // the exported relation even for a single subgroup or a single fragment.
    static_cast<void>(layout.add_offset(subgroup, 0));
    static_cast<void>(layout.add_offset(fragment, 0));
    std::array<AxisBinding, 2u> bindings{{{subgroup, "warpid"}, {fragment, "m"}}};
    return export_layout(layout, bindings);
}

NativeIndices matrix_atom_coordinates(const MatrixWorkload &workload, const MatrixDistribution &distribution,
                                      tvm::PrimExpr subgroup, tvm::PrimExpr fragment) noexcept {
    if (!valid_mapping(workload, distribution)) { return {{}, "invalid rectangular matrix execution map"}; }
    DimensionContext dimensions;
    auto sg = dimensions.create_dimension();
    auto local = dimensions.create_dimension();
    auto row = dimensions.create_dimension();
    auto column = dimensions.create_dimension();
    IndexSpace physical;
    static_cast<void>(physical.add(sg, static_cast<uint64_t>(distribution.subgroups_m) * distribution.subgroups_n));
    static_cast<void>(physical.add(local, distribution.atom_rows * distribution.atom_columns));
    IndexSpace logical;
    static_cast<void>(logical.add(row, workload.rows / 8u));
    static_cast<void>(logical.add(column, workload.columns / 8u));
    auto groups_n = IndexExpr::constant(distribution.subgroups_n);
    auto rows = IndexExpr::constant(static_cast<int64_t>(distribution.atom_rows));
    auto columns = IndexExpr::constant(static_cast<int64_t>(distribution.atom_columns));
    auto s = IndexExpr::coordinate(sg);
    auto f = IndexExpr::coordinate(local);
    std::array outputs{floor_div(s, groups_n) * rows + floor_div(f, columns),
                       modulo(s, groups_n) * columns + modulo(f, columns)};
    return lower_index_map(IndexMap{physical, logical, outputs}, {std::move(subgroup), std::move(fragment)});
}

}// namespace luisa::compute::tile::bridge::tirx
