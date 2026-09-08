#pragma once

#include <luisa/tile/ir.h>

namespace luisa::compute::tile::bridge::xir::detail {

// Tile maps expand their coordinates at capture-independent lowering time.
// Loop/parallel coordinates and carried state remain runtime values. This is
// only a representation choice, never permission to move a memory effect.
[[nodiscard]] inline bool expanded_index(const Value *value, uint32_t depth = 0u) noexcept {
    if (depth > 32u) { return false; }
    if (auto block = value->argument_block()) {
        auto owner = block->parent_region()->parent_operation();
        return owner && owner->kind() == OperationKind::TILE_MAP && value->index() < owner->domain()->rank();
    }
    auto op = value->defining_operation();
    if (!op) { return false; }
    if (op->kind() == OperationKind::CONSTANT) { return true; }
    if (op->kind() != OperationKind::ELEMENTWISE || op->operand_count() != 2u) { return false; }
    switch (op->elementwise_op()) {
        case ElementwiseOp::ADD:
        case ElementwiseOp::SUB:
        case ElementwiseOp::MUL: return expanded_index(op->operand(0u), depth + 1u) && expanded_index(op->operand(1u), depth + 1u);
        default: return false;
    }
}

[[nodiscard]] inline bool expanded_extract(const Operation &op) noexcept {
    for (size_t i = 1u; i < op.operand_count(); i++) {
        if (!expanded_index(op.operand(i))) { return false; }
    }
    return true;
}

[[nodiscard]] inline bool needs_indexable_snapshot(const Value *value) noexcept {
    if (!value->type().is_tile()) { return false; }
    for (auto use : value->use_list()) {
        auto op = use->user();
        if (use->index() == 0u && op->kind() == OperationKind::TILE_EXTRACT && !expanded_extract(*op)) { return true; }
    }
    return false;
}

}// namespace luisa::compute::tile::bridge::xir::detail
