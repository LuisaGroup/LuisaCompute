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

[[nodiscard]] inline bool deferred_elementwise(const Value *value, uint32_t limit, uint32_t local_lanes = 1u) noexcept {
    auto op = value->defining_operation();
    return op && op->kind() == OperationKind::ELEMENTWISE && value->use_count() == 1u &&
           (bounded_tile(value, limit) || (local_lanes > 1u && bounded_tile(value, local_lanes - 1u)));
}

struct ClosedReduction {
    const Operation *update;
    const Operation *yield;
    const Value *contribution;
    bool carry_left;
};

[[nodiscard]] inline luisa::optional<ClosedReduction> closed_reduction(const Operation &op) noexcept {
    if (op.kind() != OperationKind::REDUCE || op.reduction_policy() != reduction::unordered_tree ||
        op.result_count() != 1u || op.result(0u)->type().is_tile()) { return {}; }
    auto body = op.region(0u)->block(0u);
    auto carry = body->argument(op.domain()->rank());
    if (carry->use_count() != 1u) { return {}; }
    const Operation *yield = nullptr;
    for (auto operation : body->operations()) {
        auto kind = operation->kind();
        if (kind == OperationKind::YIELD) {
            yield = operation;
        } else if (kind != OperationKind::CONSTANT && kind != OperationKind::ELEMENTWISE && kind != OperationKind::TILE_EXTRACT) {
            return {};
        }
    }
    if (!yield || yield->operand_count() != 1u) { return {}; }
    auto update = yield->operand(0u)->defining_operation();
    if (!update || update->parent_block() != body || update->kind() != OperationKind::ELEMENTWISE ||
        update->operand_count() != 2u || update->result(0u)->use_count() != 1u) { return {}; }
    auto left = update->operand(0u) == carry;
    if (!left && update->operand(1u) != carry) { return {}; }
    auto kind = update->elementwise_op();
    if (kind != ElementwiseOp::ADD && kind != ElementwiseOp::MUL && kind != ElementwiseOp::MIN && kind != ElementwiseOp::MAX) { return {}; }
    return ClosedReduction{update, yield, update->operand(left ? 1u : 0u), left};
}

// A sufficient, exact admission contract for packet-local distribution. It is
// not a claim that all other Tile programs are dependent. Their realizations
// need more general redistribution/carry machinery and remain available via
// complete-program lanes. No names, input noalias assumptions, or FP estimates
// participate in this check.
[[nodiscard]] inline bool packet_local_program(const Function &function, uint32_t lanes) noexcept {
    luisa::optional<Dim> dimension;
    uint64_t extent = 0u;
    auto shape = [&](const IndexSpace &space) {
        uint32_t nonunit = 0u;
        for (auto &axis : space.axes()) {
            if (!axis.extent.is_constant() || axis.extent.constant_value() == 0u) { return false; }
            auto count = axis.extent.constant_value();
            if (count == 1u) { continue; }
            if (++nonunit > 1u || count < lanes || count > UINT32_MAX) { return false; }
            if (!dimension) {
                dimension = axis.dimension;
                extent = count;
            } else if (*dimension != axis.dimension || extent != count) {
                return false;
            }
        }
        return true;
    };
    auto visit = [&](auto &&self, const Block &block, bool root, bool active_axis) -> bool {
        for (auto op : block.operations()) {
            if (op->memory_layout() || op->resource_class_constraint()) { return false; }
            if (auto binding = op->execution_scope_constraint(); binding && *binding != "auto") { return false; }
            if (op->kind() == OperationKind::PARALLEL) {
                if (!root || !self(self, *op->region(0u)->block(0u), false, false)) { return false; }
                continue;
            }
            if (root) {
                if (op->kind() != OperationKind::CONSTANT && op->kind() != OperationKind::ELEMENTWISE) { return false; }
                for (size_t i = 0u; i < op->result_count(); i++) {
                    if (op->result(i)->type().is_tile()) { return false; }
                }
                continue;
            }
            if (op->domain() && !shape(*op->domain())) { return false; }
            for (size_t i = 0u; i < op->result_count(); i++) {
                auto &type = op->result(i)->type();
                if (type.is_tile() && !shape(*type.index_space())) { return false; }
                if (active_axis && type.is_tile() && bounded_domain(*type.index_space(), 1u)) { return false; }
            }
            switch (op->kind()) {
                case OperationKind::CONSTANT:
                case OperationKind::ELEMENTWISE:
                case OperationKind::YIELD: break;
                case OperationKind::VIEW_LOAD:
                case OperationKind::VIEW_STORE:
                    // Scalar side effects would be duplicated by every lane.
                    // A unit Tile store is emitted by the packet leader below.
                    if (!op->domain() || active_axis) { return false; }
                    break;
                case OperationKind::TILE_MAP:
                    if (active_axis || !self(self, *op->region(0u)->block(0u), false, bounded_domain(*op->domain(), 1u))) { return false; }
                    break;
                case OperationKind::REDUCE:
                    if (active_axis || !closed_reduction(*op) || !self(self, *op->region(0u)->block(0u), false, bounded_domain(*op->domain(), 1u))) { return false; }
                    break;
                case OperationKind::TILE_EXTRACT: {
                    auto &space = *op->operand(0u)->type().index_space();
                    for (size_t i = 0u; i < space.rank(); i++) {
                        if (space.axis(i).extent.constant_value() == 1u) { continue; }
                        auto index = op->operand(i + 1u);
                        auto owner = index->argument_block();
                        auto nest = owner ? owner->parent_region()->parent_operation() : nullptr;
                        // Every nonunit extract is owner-preserving, including
                        // a ragged tail with inactive lanes. Unit coordinates
                        // must be literal zero, so they cannot alter flat owner.
                        if (!nest || (nest->kind() != OperationKind::TILE_MAP && nest->kind() != OperationKind::REDUCE) ||
                            index->index() >= nest->domain()->rank() || nest->domain()->axis(index->index()).dimension != space.axis(i).dimension) { return false; }
                    }
                    for (size_t i = 0u; i < space.rank(); i++) {
                        if (space.axis(i).extent.constant_value() != 1u) { continue; }
                        auto index = op->operand(i + 1u);
                        auto owner = index->argument_block();
                        auto nest = owner ? owner->parent_region()->parent_operation() : nullptr;
                        if (nest && nest->domain() && index->index() < nest->domain()->rank() &&
                            nest->domain()->axis(index->index()).extent.is_constant() && nest->domain()->axis(index->index()).extent.constant_value() == 1u) { continue; }
                        auto constant = index->defining_operation();
                        auto value = constant && constant->kind() == OperationKind::CONSTANT ? constant->attribute("value") : nullptr;
                        if (!value) { return false; }
                        auto s = luisa::get_if<int64_t>(&value->value());
                        auto u = luisa::get_if<uint64_t>(&value->value());
                        if ((!s || *s != 0) && (!u || *u != 0u)) { return false; }
                    }
                    break;
                }
                default: return false;
            }
        }
        return true;
    };
    return lanes > 1u && function.body().block_count() == 1u && visit(visit, *function.body().block(0u), true, false) && dimension.has_value();
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
