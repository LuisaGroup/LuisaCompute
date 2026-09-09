#pragma once

#include <algorithm>
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

// A recipe cannot escape into a different temporal or spatial execution
// region. Pure element maps are transparent; loops/reductions/stages are not
// an invitation to repeatedly evaluate an old carried-state definition.
[[nodiscard]] inline bool map_local_use(const Value *value) noexcept {
    auto producer = value->defining_operation();
    if (!producer || value->use_count() != 1u) { return false; }
    for (auto use : value->use_list()) {
        auto consumer = use->user();
        auto anchor = consumer;
        auto block = consumer->parent_block();
        for (auto depth = 0u; block != producer->parent_block(); depth++) {
            if (!block || depth > 32u) { return false; }
            auto owner = block->parent_region()->parent_operation();
            if (!owner || owner->kind() != OperationKind::TILE_MAP) { return false; }
            anchor = owner;
            block = owner->parent_block();
        }
        auto active = false;
        for (auto operation : block->operations()) {
            if (operation == anchor) { break; }
            if (operation == producer) {
                active = true;
                continue;
            }
            if (active && operation->kind() != OperationKind::CONSTANT && operation->kind() != OperationKind::ELEMENTWISE &&
                operation->kind() != OperationKind::TILE_MAP && operation->kind() != OperationKind::VIEW_LOAD &&
                operation->kind() != OperationKind::VIEW_STORE) { return false; }
        }
        return consumer->kind() == OperationKind::ELEMENTWISE || consumer->kind() == OperationKind::TILE_EXTRACT ||
               consumer->kind() == OperationKind::VIEW_STORE;
    }
    return false;
}

[[nodiscard]] inline bool deferred_map(const Value *value, bool enabled, uint32_t lanes) noexcept {
    if (!enabled || lanes != 1u || !map_local_use(value)) { return false; }
    auto op = value->defining_operation();
    if (op->kind() != OperationKind::TILE_MAP) { return false; }
    auto count = 0u;
    for (auto child : op->region(0u)->block(0u)->operations()) {
        if (++count > 64u) { return false; }
        auto kind = child->kind();
        if (kind == OperationKind::YIELD) { continue; }
        if (kind != OperationKind::CONSTANT && kind != OperationKind::ELEMENTWISE && kind != OperationKind::TILE_EXTRACT) { return false; }
        if (child->result_count() != 1u || child->result(0u)->type().is_tile()) { return false; }
    }
    return true;
}

[[nodiscard]] inline bool deferred_expression(const Value *value, uint32_t limit, uint32_t lanes, bool maps) noexcept {
    if (deferred_elementwise(value, limit, lanes)) { return true; }
    if (!maps || lanes != 1u || !map_local_use(value)) { return false; }
    auto op = value->defining_operation();
    if (op->kind() != OperationKind::ELEMENTWISE || !value->type().is_tile()) { return false; }
    // Small index/expression Tiles otherwise become arrays solely because a
    // scalar map extracts them with a runtime coordinate (e.g. iota - offset).
    for (auto use : value->use_list()) {
        auto user = use->user();
        auto owner = user->parent_block()->parent_region()->parent_operation();
        return use->index() == 0u && user->kind() == OperationKind::TILE_EXTRACT &&
               owner && owner->kind() == OperationKind::TILE_MAP;
    }
    return false;
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

struct ReductionProducerFusion {
    const Operation *reduction;
    bool retain_snapshot;
};

// A materialized producer may join its first pointwise reduction traversal,
// computing each point once (including multi-consumer expressions). It must not cross
// any write (including through another argument), stage, or unknown effect.
// Unit maps are transparent execution wrappers, as used by library reduce().
// This is a realization admission rule, not a noalias or parallelism proof.
[[nodiscard]] inline luisa::optional<ReductionProducerFusion> reduction_producer_fusion(
    const Value *value, uint32_t limit, uint32_t lanes, uint32_t partitions,
    bool enable_loads, bool enable_expressions) noexcept {
    if (!enable_loads && !enable_expressions) { return {}; }
    auto producer = value->defining_operation();
    if (!producer || !value->type().is_tile() ||
        !(bounded_tile(value, limit) || (lanes > 1u && bounded_tile(value, lanes - 1u)))) { return {}; }
    if (producer->kind() == OperationKind::VIEW_LOAD) {
        if (!enable_loads) { return {}; }
    } else if (producer->kind() == OperationKind::ELEMENTWISE) {
        if (!enable_expressions || deferred_elementwise(value, limit, lanes)) { return {}; }
    } else {
        return {};
    }
    auto block = producer->parent_block();
    luisa::vector<const Value *> recipes{value};
    luisa::vector<const Operation *> consumers;
    for (size_t i = 0u; i < recipes.size(); i++) {
        if (recipes.size() + consumers.size() > 256u) { return {}; }
        for (auto use : recipes[i]->use_list()) {
            auto op = use->user();
            if (op->result_count() == 1u && deferred_elementwise(op->result(0u), limit, lanes)) {
                if (std::find(recipes.begin(), recipes.end(), op->result(0u)) == recipes.end()) { recipes.emplace_back(op->result(0u)); }
            } else if (std::find(consumers.begin(), consumers.end(), op) == consumers.end()) {
                consumers.emplace_back(op);
            }
        }
    }
    if (consumers.empty()) { return {}; }
    auto inside = [](const Operation *child, const Operation *parent) {
        for (auto depth = 0u; child && depth < 64u; depth++) {
            if (child == parent) { return true; }
            auto block = child->parent_block();
            child = block ? block->parent_region()->parent_operation() : nullptr;
        }
        return false;
    };
    auto unit = [](const IndexSpace &space) {
        for (auto &axis : space.axes()) {
            if (!axis.extent.is_constant() || axis.extent.constant_value() != 1u) { return false; }
        }
        return true;
    };
    auto read_only = [&](auto &&self, const Operation &op, uint32_t depth) -> bool {
        if (depth > 32u) { return false; }
        switch (op.kind()) {
            case OperationKind::CONSTANT:
            case OperationKind::ELEMENTWISE:
            case OperationKind::VIEW_LOAD:
            case OperationKind::TILE_EXTRACT:
            case OperationKind::YIELD: return true;
            case OperationKind::REDUCE: return closed_reduction(op).has_value();
            case OperationKind::TILE_MAP:
                for (auto child : op.region(0u)->block(0u)->operations()) {
                    if (!self(self, *child, depth + 1u)) { return false; }
                }
                return true;
            default: return false;
        }
    };
    bool valid = true;
    auto scan = [&](auto &&self, const Block &body, bool active, uint32_t depth) -> const Operation * {
        if (depth > 32u) {
            valid = false;
            return nullptr;
        }
        for (auto op : body.operations()) {
            if (!active) {
                active = op == producer;
                continue;
            }
            auto used = std::any_of(consumers.begin(), consumers.end(), [&](auto user) { return inside(user, op); });
            if (used) {
                if (op->kind() == OperationKind::REDUCE) { return op; }
                if (op->kind() == OperationKind::TILE_MAP && unit(*op->domain())) {
                    return self(self, *op->region(0u)->block(0u), true, depth + 1u);
                }
                valid = false;
                return nullptr;
            }
            if (!read_only(read_only, *op, 0u)) {
                valid = false;
                return nullptr;
            }
        }
        return nullptr;
    };
    auto reduction = scan(scan, *block, false, 0u);
    if (!valid || !reduction || !closed_reduction(*reduction) ||
        !((lanes > 1u && bounded_domain(*reduction->domain(), lanes - 1u)) ||
          (partitions > 1u && bounded_domain(*reduction->domain(), limit)))) { return {}; }
    auto &domain = *reduction->domain();
    // Bijection of nonunit dimensions: every producer point is visited exactly
    // once. Unit axes may be inserted/projected by a library wrapper.
    for (auto recipe : recipes) {
        auto &space = *recipe->type().index_space();
        auto matches = [](const IndexSpace &a, const IndexSpace &b) {
            for (auto &axis : a.axes()) {
                if (!axis.extent.is_constant() || axis.extent.constant_value() == 0u) { return false; }
                if (axis.extent.constant_value() == 1u) { continue; }
                auto i = b.axis_index(axis.dimension);
                if (!i || b.axis(*i).extent != axis.extent) { return false; }
            }
            return true;
        };
        if (!matches(space, domain) || !matches(domain, space)) { return {}; }
    }
    auto body = reduction->region(0u)->block(0u);
    bool retain = false;
    for (auto op : consumers) {
        if (!inside(op, reduction)) {
            retain = true;
            continue;
        }
        if (op->kind() != OperationKind::TILE_EXTRACT || op->parent_block() != body) { return {}; }
        auto &space = *op->operand(0u)->type().index_space();
        for (size_t i = 0u; i < space.rank(); i++) {
            auto index = op->operand(i + 1u);
            auto owner = index->argument_block();
            auto nest = owner ? owner->parent_region()->parent_operation() : nullptr;
            auto &axis = space.axis(i);
            if (axis.extent.constant_value() != 1u) {
                auto j = domain.axis_index(axis.dimension);
                if (!j || index != body->argument(*j)) { return {}; }
            } else {
                if (nest && nest->domain() && index->index() < nest->domain()->rank() &&
                    nest->domain()->axis(index->index()).extent.is_constant() &&
                    nest->domain()->axis(index->index()).extent.constant_value() == 1u) { continue; }
                auto constant = index->defining_operation();
                auto attr = constant && constant->kind() == OperationKind::CONSTANT ? constant->attribute("value") : nullptr;
                if (!attr) { return {}; }
                auto s = luisa::get_if<int64_t>(&attr->value());
                auto u = luisa::get_if<uint64_t>(&attr->value());
                if ((!s || *s != 0) && (!u || *u != 0u)) { return {}; }
            }
        }
    }
    return ReductionProducerFusion{reduction, retain};
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
