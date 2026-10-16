#pragma once

#include <limits>
#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/xir/metadata/contiguous_copy.h>

namespace luisa::compute::tile::bridge::xir::detail {

// Geometry admission only. Runtime per-axis bounds are checked at the load
// definition, not inferred from flat buffer capacity. Unit axes contribute no
// varying coordinate; every other axis must have the same source and snapshot
// stride. No dimension names, consumer kind, or program packet width enters
// this decision. Existing allocation/fusion policy has precedence.
[[nodiscard]] inline luisa::optional<compute::xir::ContiguousCopyDescriptor>
native_copy_plan(const Operation &op, const LowerOptions &options, bool snapshot) noexcept {
    auto width = options.native_copy_vector_width;
    if (width < 2u || (width & (width - 1u)) != 0u ||
        options.max_unrolled_tile_elements == 0u || options.local_lanes != 1u || !snapshot ||
        op.kind() != OperationKind::VIEW_LOAD || !op.domain() || op.result_count() != 1u || op.operand_count() == 0u ||
        !op.result(0u)->type().is_tile() || op.result(0u)->type().scalar_type() != ScalarType::FLOAT32 ||
        op.operand(0u)->type().scalar_type() != ScalarType::FLOAT32) { return {}; }
    auto &source = *op.operand(0u)->type().index_space();
    auto &tile = *op.domain();
    if (source.rank() != tile.rank()) { return {}; }
    constexpr auto limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / sizeof(float);
    auto source_stride = uint64_t{1u};
    auto tile_stride = uint64_t{1u};
    for (auto i = tile.rank(); i != 0u; i--) {
        auto &s = source.axis(i - 1u).extent;
        auto &t = tile.axis(i - 1u).extent;
        if (!s.is_constant() || !t.is_constant()) { return {}; }
        auto se = s.constant_value();
        auto te = t.constant_value();
        if (te == 0u || se == 0u || te > se ||
            (te > 1u && source_stride != tile_stride) ||
            source_stride > limit / se || tile_stride > limit / te) { return {}; }
        source_stride *= se;
        tile_stride *= te;
    }
    if (tile_stride < width) { return {}; }
    return compute::xir::ContiguousCopyDescriptor{tile_stride, width};
}

}// namespace luisa::compute::tile::bridge::xir::detail
