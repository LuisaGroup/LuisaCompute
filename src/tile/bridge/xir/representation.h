#pragma once

#include <luisa/tile/ir.h>

namespace luisa::compute::tile::bridge::xir::detail {

[[nodiscard]] inline bool bounded_domain(const IndexSpace &space, uint32_t limit) noexcept {
    if (limit == 0u) { return false; }
    for (auto &axis : space.axes()) {
        if (axis.extent.is_constant() && axis.extent.constant_value() == 0u) { return false; }
    }
    uint64_t count = 1u;
    for (auto &axis : space.axes()) {
        if (!axis.extent.is_constant()) { return true; }
        auto extent = axis.extent.constant_value();
        if (extent == 0u) { return false; }
        if (count > limit / extent) { return true; }
        count *= extent;
    }
    return count > limit;
}

[[nodiscard]] inline bool bounded_tile(const Value *value, uint32_t limit) noexcept {
    return value->type().is_tile() && bounded_domain(*value->type().index_space(), limit);
}

[[nodiscard]] inline bool deferred_elementwise(const Value *value, uint32_t limit) noexcept {
    auto op = value->defining_operation();
    return op && op->kind() == OperationKind::ELEMENTWISE && value->use_count() == 1u && bounded_tile(value, limit);
}

// Small Tile maps expand their coordinates at lowering time. Bounded maps,
// loop/parallel coordinates and carried state remain runtime values. This is
// only a representation choice, never permission to move a memory effect.
[[nodiscard]] inline bool expanded_index(const Value *value, uint32_t limit = 0u, uint32_t depth = 0u) noexcept {
    if (depth > 32u) { return false; }
    if (auto block = value->argument_block()) {
        auto owner = block->parent_region()->parent_operation();
        return owner && owner->kind() == OperationKind::TILE_MAP && value->index() < owner->domain()->rank() && !bounded_domain(*owner->domain(), limit);
    }
    auto op = value->defining_operation();
    if (!op) { return false; }
    if (op->kind() == OperationKind::CONSTANT) { return true; }
    if (op->kind() != OperationKind::ELEMENTWISE || op->operand_count() != 2u) { return false; }
    switch (op->elementwise_op()) {
        case ElementwiseOp::ADD:
        case ElementwiseOp::SUB:
        case ElementwiseOp::MUL: return expanded_index(op->operand(0u), limit, depth + 1u) && expanded_index(op->operand(1u), limit, depth + 1u);
        default: return false;
    }
}

[[nodiscard]] inline bool expanded_extract(const Operation &op, uint32_t limit = 0u) noexcept {
    for (size_t i = 1u; i < op.operand_count(); i++) {
        if (!expanded_index(op.operand(i), limit)) { return false; }
    }
    return true;
}

[[nodiscard]] inline bool needs_indexable_snapshot(const Value *value, uint32_t limit = 0u) noexcept {
    if (!value->type().is_tile()) { return false; }
    for (auto use : value->use_list()) {
        auto op = use->user();
        if (use->index() == 0u && op->kind() == OperationKind::TILE_EXTRACT && !expanded_extract(*op, limit)) { return true; }
        if ((op->kind() == OperationKind::ELEMENTWISE || op->kind() == OperationKind::MMA) && bounded_tile(op->result(0u), limit)) { return true; }
    }
    return false;
}

}// namespace luisa::compute::tile::bridge::xir::detail
