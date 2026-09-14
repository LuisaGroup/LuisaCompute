#pragma once

#include <algorithm>
#include <luisa/core/mathematics.h>
#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/tile/ir.h>
#include "native_mma.h"

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
    bool parallel_seen = false;
    auto shape = [&](const IndexSpace &space) {
        uint32_t nonunit = 0u;
        for (auto &axis : space.axes()) {
            if (!axis.extent.is_constant() || axis.extent.constant_value() == 0u) { return false; }
            auto count = axis.extent.constant_value();
            if (count == 1u) { continue; }
            // A short common axis has one masked owner slot per lane, not
            // replicated elements. Empty shapes still fail closed above.
            if (++nonunit > 1u || count > UINT32_MAX) { return false; }
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
                parallel_seen = true;
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
    // A unit-only program is also legal: values remain replicated and its
    // unit Tile stores are leader-only. Require an actual execution root.
    return lanes > 1u && function.body().block_count() == 1u && visit(visit, *function.body().block(0u), true, false) && parallel_seen;
}

// A saturating potential-work bound, not an emitted instruction count. Loops
// retain their trip count here because downstream constant propagation/unroll
// may duplicate their bodies after a surrounding map has been enumerated.
// Unknown shapes, malformed regions and excessive depth exceed the budget.
struct MapBodyWork {
    uint64_t work;
    bool structured;
};

[[nodiscard]] inline uint64_t capped_work_product(uint64_t a, uint64_t b, uint64_t cap) noexcept {
    return b != 0u && a > cap / b ? cap : a * b;
}

[[nodiscard]] inline uint64_t capped_work_volume(const IndexSpace &space, uint64_t cap) noexcept {
    for (auto &axis : space.axes()) {
        if (axis.extent.is_constant() && axis.extent.constant_value() == 0u) { return 0u; }
    }
    auto count = uint64_t{1u};
    for (auto &axis : space.axes()) {
        if (!axis.extent.is_constant()) { return cap; }
        count = capped_work_product(count, axis.extent.constant_value(), cap);
    }
    return count;
}

[[nodiscard]] inline MapBodyWork map_body_work(const Block &body, uint64_t cap, uint32_t depth = 0u) noexcept {
    if (depth >= 64u) { return {cap, true}; }
    MapBodyWork result{};
    for (auto op : body.operations()) {
        auto work = uint64_t{1u};
        if (op->kind() == OperationKind::YIELD || op->kind() == OperationKind::STAGE) { continue; }
        if (op->kind() == OperationKind::MMA) {
            result.structured = true;
            if (op->result_count() != 1u || !op->result(0u)->type().is_tile() ||
                op->operand_count() != 3u || !op->operand(0u)->type().is_tile()) { return {cap, true}; }
            auto &output = *op->result(0u)->type().index_space();
            work = capped_work_product(2u, capped_work_volume(output, cap), cap);
            for (auto &axis : op->operand(0u)->type().index_space()->axes()) {
                if (!output.contains(axis.dimension)) {
                    if (!axis.extent.is_constant()) { return {cap, true}; }
                    work = capped_work_product(work, axis.extent.constant_value(), cap);
                }
            }
        } else if (op->region_count() != 0u) {
            result.structured = true;
            if (op->region_count() != 1u || op->region(0u)->block_count() != 1u || !op->domain()) { return {cap, true}; }
            auto nested = map_body_work(*op->region(0u)->block(0u), cap, depth + 1u);
            work = capped_work_product(capped_work_volume(*op->domain(), cap), nested.work, cap);
        } else if (op->result_count() == 1u && op->result(0u)->type().is_tile()) {
            work = capped_work_volume(*op->result(0u)->type().index_space(), cap);
        }
        result.work += std::min(work, cap - result.work);
        if (result.work == cap && result.structured) { break; }
    }
    return result;
}

// This is the single map expansion decision used by storage, coordinate
// classification, emission and resource/cost analysis. Small simple maps keep
// their historical expansion. A unit map cannot replicate its nested body.
[[nodiscard]] inline bool map_runtime_loop(const Operation &op, uint32_t element_limit,
                                           uint32_t region_budget = LowerOptions{}.max_unrolled_region_work) noexcept {
    if (element_limit == 0u) { return false; }
    if (!op.domain() || bounded_domain(*op.domain(), element_limit)) { return true; }
    if (region_budget == 0u) { return false; }
    auto cap = static_cast<uint64_t>(region_budget) + 1u;
    auto count = capped_work_volume(*op.domain(), cap);
    if (count <= 1u) { return false; }
    if (op.region_count() != 1u || op.region(0u)->block_count() != 1u) { return true; }
    auto body = map_body_work(*op.region(0u)->block(0u), cap);
    return body.structured && capped_work_product(count, body.work, cap) > region_budget;
}

// Small Tile maps expand their coordinates at lowering time. Bounded maps,
// loop/parallel coordinates and carried state remain runtime values. This is
// only a representation choice, never permission to move a memory effect.
[[nodiscard]] inline bool expanded_index(const Value *value, uint32_t limit = 0u,
                                         uint32_t region_budget = LowerOptions{}.max_unrolled_region_work, uint32_t depth = 0u) noexcept {
    if (depth > 32u) { return false; }
    if (auto block = value->argument_block()) {
        auto owner = block->parent_region()->parent_operation();
        return owner && owner->kind() == OperationKind::TILE_MAP && value->index() < owner->domain()->rank() && !map_runtime_loop(*owner, limit, region_budget);
    }
    auto op = value->defining_operation();
    if (!op) { return false; }
    if (op->kind() == OperationKind::CONSTANT) { return true; }
    if (op->kind() != OperationKind::ELEMENTWISE || op->operand_count() != 2u) { return false; }
    switch (op->elementwise_op()) {
        case ElementwiseOp::ADD:
        case ElementwiseOp::SUB:
        case ElementwiseOp::MUL: return expanded_index(op->operand(0u), limit, region_budget, depth + 1u) && expanded_index(op->operand(1u), limit, region_budget, depth + 1u);
        default: return false;
    }
}

[[nodiscard]] inline bool expanded_extract(const Operation &op, uint32_t limit = 0u,
                                           uint32_t region_budget = LowerOptions{}.max_unrolled_region_work) noexcept {
    for (size_t i = 1u; i < op.operand_count(); i++) {
        if (!expanded_index(op.operand(i), limit, region_budget)) { return false; }
    }
    return true;
}

[[nodiscard]] inline bool needs_indexable_snapshot(const Value *value, uint32_t limit = 0u,
                                                   uint32_t region_budget = LowerOptions{}.max_unrolled_region_work) noexcept {
    if (!value->type().is_tile()) { return false; }
    for (auto use : value->use_list()) {
        auto op = use->user();
        if (use->index() == 0u && op->kind() == OperationKind::TILE_EXTRACT && !expanded_extract(*op, limit, region_budget)) { return true; }
        if ((op->kind() == OperationKind::ELEMENTWISE || op->kind() == OperationKind::MMA) && bounded_tile(op->result(0u), limit)) { return true; }
    }
    return false;
}

// These plans describe static emission, not dynamic execution or liveness.
// Resource admission and the emitter must use the same decisions: counting
// source Values once is incorrect when a map or partial reduction emits a
// source definition multiple times.
[[nodiscard]] inline bool bounded_count(uint64_t count, const LowerOptions &options) noexcept {
    return options.max_unrolled_tile_elements != 0u && count > options.max_unrolled_tile_elements;
}

[[nodiscard]] inline bool distributed_count(uint64_t count, const LowerOptions &options) noexcept {
    return options.local_lanes > 1u && count > 1u;
}

[[nodiscard]] inline uint64_t snapshot_elements(uint64_t count, const LowerOptions &options) noexcept {
    return distributed_count(count, options) ? ceil_div(count, static_cast<uint64_t>(options.local_lanes)) : count;
}

[[nodiscard]] inline bool traversal_snapshot(uint64_t count, const LowerOptions &options) noexcept {
    return bounded_count(count, options) || distributed_count(count, options);
}

struct MmaContractionPlan {
    IndexSpace domain;
    bool runtime_loop{false};
    bool additional_runtime_loop{false};
};

// Pure domain analysis: never inspect another Value's allocation plan. Both
// contraction emission and its operands' definition-time storage use this
// decision, including when the operand is a carried block argument.
[[nodiscard]] inline MmaContractionPlan mma_contraction_plan(const Operation &op, const LowerOptions &options) noexcept {
    MmaContractionPlan plan;
    auto &output = *op.result(0u)->type().index_space();
    for (auto &axis : op.operand(0u)->type().index_space()->axes()) {
        if (!output.contains(axis.dimension)) { static_cast<void>(plan.domain.add(axis.dimension, axis.extent)); }
    }
    auto inherited = bounded_domain(plan.domain, options.max_unrolled_tile_elements);
    plan.runtime_loop = options.max_unrolled_tile_elements != 0u &&
                        (inherited || bounded_domain(plan.domain, options.max_unrolled_mma_terms));
    plan.additional_runtime_loop = plan.runtime_loop && !inherited;
    return plan;
}

[[nodiscard]] inline bool mma_operand_snapshot(const Value *value, const LowerOptions &options) noexcept {
    if (options.max_unrolled_mma_terms == 0u || options.max_unrolled_tile_elements == 0u ||
        !value->type().is_tile() || bounded_tile(value, options.max_unrolled_tile_elements)) { return false; }
    // Constants do not need indexable storage for newly dynamic MMA reads.
    // Existing snapshot requirements remain intact in definition_snapshot.
    if (auto op = value->defining_operation(); op && op->kind() == OperationKind::CONSTANT) { return false; }
    for (auto use : value->use_list()) {
        auto consumer = use->user();
        if (consumer->kind() != OperationKind::MMA || use->index() >= 2u) { continue; }
        auto &output = *consumer->result(0u)->type().index_space();
        if (capped_work_volume(output, 1u) == 0u || !mma_contraction_plan(*consumer, options).additional_runtime_loop) { continue; }
        for (auto &axis : value->type().index_space()->axes()) {
            if (!output.contains(axis.dimension) && axis.extent.is_constant() && axis.extent.constant_value() > 1u) { return true; }
        }
    }
    return false;
}

[[nodiscard]] inline bool definition_snapshot(const Value *value, uint64_t elements, const LowerOptions &options) noexcept {
    return native_mma_snapshot(value, options) ||
           (elements > 1u && (needs_indexable_snapshot(value, options.max_unrolled_tile_elements, options.max_unrolled_region_work) ||
                              mma_operand_snapshot(value, options)));
}

enum class ValueRepresentation : uint8_t {
    EMITTED,
    SPLAT,
    DEFERRED_EXPRESSION,
    DEFERRED_MAP,
    REDUCTION_PRODUCER
};

struct MmaEmissionPlan {
    uint32_t output_block{1u};
    uint64_t columns{1u};
    bool broadcast_lhs{true};
    bool contraction_runtime_loop{false};
    uint32_t output_rows{1u};
    uint64_t row_extent{1u};
};

[[nodiscard]] inline MmaEmissionPlan mma_emission_plan(const Operation &op, const LowerOptions &options) noexcept;

struct ValueAllocationPlan {
    ValueRepresentation representation{ValueRepresentation::EMITTED};
    bool snapshot{false};
    luisa::optional<ReductionProducerFusion> fusion;
    // Register blocking changes scalar recurrence emission, never the chosen
    // result snapshot or its size. Analysis and emission consume this plan.
    MmaEmissionPlan mma{};
};

// count is the already validated static Tile volume (one for a scalar).
[[nodiscard]] inline ValueAllocationPlan value_allocation_plan(
    const Value *value, uint64_t count, const LowerOptions &options) noexcept {
    if (!value->type().is_tile()) { return {}; }
    auto op = value->defining_operation();
    // A native call consumes references to definition-time snapshots, never
    // deferred recipes or a re-read of mutable input memory. Include constants
    // and singleton seeds; the result is always a separate writable array.
    if (native_mma_snapshot(value, options)) {
        return {ValueRepresentation::EMITTED, true, {}, op && op->kind() == OperationKind::MMA ? mma_emission_plan(*op, options) : MmaEmissionPlan{}};
    }
    if (op && op->kind() == OperationKind::CONSTANT && traversal_snapshot(count, options)) {
        return {ValueRepresentation::SPLAT, false, {}};
    }
    if (deferred_expression(value, options.max_unrolled_tile_elements, options.local_lanes, options.enable_map_fusion)) {
        return {ValueRepresentation::DEFERRED_EXPRESSION, false, {}};
    }
    if (deferred_map(value, options.enable_map_fusion, options.local_lanes)) {
        return {ValueRepresentation::DEFERRED_MAP, false, {}};
    }
    if (auto fusion = reduction_producer_fusion(value, options.max_unrolled_tile_elements,
                                                options.local_lanes, options.reduction_partitions,
                                                options.enable_load_reduction_fusion, options.enable_expression_reduction_fusion)) {
        return {ValueRepresentation::REDUCTION_PRODUCER, fusion->retain_snapshot, fusion};
    }
    auto runtime_map = op && op->kind() == OperationKind::TILE_MAP && map_runtime_loop(*op, options.max_unrolled_tile_elements, options.max_unrolled_region_work);
    return {ValueRepresentation::EMITTED,
            runtime_map || traversal_snapshot(count, options) || definition_snapshot(value, count, options),
            {},
            op && op->kind() == OperationKind::MMA ? mma_emission_plan(*op, options) : MmaEmissionPlan{}};
}

struct TraversalEmissionPlan {
    uint64_t full_count;
    uint32_t lanes;
    uint32_t tail_lanes;
    bool runtime_loop;
    [[nodiscard]] uint64_t emitted_bodies() const noexcept {
        return (runtime_loop ? 1u : full_count) + (tail_lanes != 0u);
    }
};

[[nodiscard]] inline bool serial_runtime_loop(uint64_t count, const LowerOptions &options, bool force_loop = false) noexcept {
    return force_loop || bounded_count(count, options);
}

[[nodiscard]] inline TraversalEmissionPlan traversal_emission_plan(uint64_t count, const LowerOptions &options) noexcept {
    auto lanes = distributed_count(count, options) ? options.local_lanes : 1u;
    auto full = count / lanes;
    return {full, lanes, static_cast<uint32_t>(count % lanes),
            serial_runtime_loop(full, options, bounded_count(count, options))};
}

[[nodiscard]] inline TraversalEmissionPlan map_emission_plan(const Operation &op, uint64_t count, const LowerOptions &options) noexcept {
    auto plan = traversal_emission_plan(count, options);
    plan.runtime_loop |= map_runtime_loop(op, options.max_unrolled_tile_elements, options.max_unrolled_region_work);
    return plan;
}

// Block only the contiguous output direction of a contraction whose other
// operand broadcasts that direction. This admits ordinary row-major GEMM and
// many weighted sums, without recognizing an operator or dimension name.
[[nodiscard]] inline MmaEmissionPlan mma_emission_plan(const Operation &op, const LowerOptions &options) noexcept {
    auto &output = *op.result(0u)->type().index_space();
    auto &lhs = *op.operand(0u)->type().index_space();
    auto &rhs = *op.operand(1u)->type().index_space();
    auto contraction = mma_contraction_plan(op, options);
    MmaEmissionPlan plan;
    // Decide contraction emission before output-block admission: R1 and
    // strided output layouts have the same independent MMA unroll control.
    // The zero Tile threshold remains the fully expanded diagnostic override.
    plan.contraction_runtime_loop = contraction.runtime_loop;
    if (options.mma_output_block == 1u || options.local_lanes != 1u || options.max_unrolled_tile_elements == 0u) { return plan; }
    auto axis = output.rank();
    while (axis != 0u && output.axis(axis - 1u).extent.is_constant() && output.axis(axis - 1u).extent.constant_value() == 1u) { axis--; }
    if (axis == 0u || !output.axis(axis - 1u).extent.is_constant()) { return plan; }
    auto &inner = output.axis(axis - 1u);
    auto columns = inner.extent.constant_value();
    if (columns <= 1u) { return plan; }
    auto fits_budget = [&](uint64_t outputs) noexcept {
        if (options.max_unrolled_region_work == 0u) { return true; }
        auto cap = static_cast<uint64_t>(options.max_unrolled_region_work) + 1u;
        auto updates = plan.contraction_runtime_loop ? 1u : capped_work_volume(contraction.domain, cap);
        auto work = capped_work_product(outputs, updates, cap);
        work = capped_work_product(work, 2u + lhs.rank() + rhs.rank() + output.rank(), cap);
        return work <= options.max_unrolled_region_work;
    };
    if (options.enable_mma_2d_blocking && options.mma_output_block == 4u) {
        auto outer = axis - 1u;
        while (outer != 0u && output.axis(outer - 1u).extent.is_constant() && output.axis(outer - 1u).extent.constant_value() == 1u) { outer--; }
        if (outer != 0u) {
            auto &row = output.axis(outer - 1u);
            auto lhs_rows = lhs.contains(row.dimension) && !lhs.contains(inner.dimension) &&
                            rhs.contains(inner.dimension) && !rhs.contains(row.dimension);
            auto rhs_rows = rhs.contains(row.dimension) && !rhs.contains(inner.dimension) &&
                            lhs.contains(inner.dimension) && !lhs.contains(row.dimension);
            if (row.extent.is_constant() && row.extent.constant_value() > 1u && (lhs_rows || rhs_rows) && fits_budget(4u)) {
                plan.output_block = 2u;
                plan.columns = columns;
                plan.broadcast_lhs = lhs_rows;
                plan.output_rows = 2u;
                plan.row_extent = row.extent.constant_value();
                return plan;
            }
        }
    }
    auto unit_stride = [&](const IndexSpace &space) noexcept {
        auto index = space.axis_index(inner.dimension);
        if (!index) { return false; }
        for (auto i = *index + 1u; i < space.rank(); i++) {
            if (!space.axis(i).extent.is_constant() || space.axis(i).extent.constant_value() != 1u) { return false; }
        }
        return true;
    };
    // The grouped outputs have independent ordered accumulators. A broadcast
    // LHS can be shared even when RHS projection has a non-unit logical
    // stride (for example [output, contraction]); contiguity is not needed
    // for legality. Retain the existing unit-stride symmetric candidate.
    auto broadcast_lhs = !lhs.contains(inner.dimension) && rhs.contains(inner.dimension);
    if (!broadcast_lhs && (rhs.contains(inner.dimension) || !unit_stride(lhs))) { return plan; }
    // Potential per-block expansion, not an instruction or cycle estimate.
    if (!fits_budget(std::min<uint64_t>(options.mma_output_block, columns))) { return plan; }
    plan.output_block = options.mma_output_block;
    plan.columns = columns;
    plan.broadcast_lhs = broadcast_lhs;
    return plan;
}

struct ReductionEmissionPlan {
    ClosedReduction closed;
    uint64_t count;
    uint64_t partitions;
    uint32_t lanes;
    uint32_t tail_lanes;
    [[nodiscard]] uint64_t emitted_bodies() const noexcept {
        // One seed and one emitted runtime body per partition, followed by
        // unrolled residuals and a separately emitted masked packet tail.
        // A short axis has no full chunks/partitions and exactly one tail.
        return (partitions == 0u ? 0u : partitions * 2u + count % partitions) + (tail_lanes != 0u);
    }
    [[nodiscard]] uint64_t packet_tree_work() const noexcept {
        if (lanes <= 1u) { return 0u; }
        uint64_t levels = 0u;
        for (auto width = lanes; width > 1u; width >>= 1u) { levels++; }
        // Relative selected-operation count, not latency or final LLVM size.
        // Keep the full-packet prior: payload shuffle + combine per level,
        // then the canonical-root broadcast and the user's initial combine.
        auto full_packet_work = levels * 2u + 2u;
        if (count != 0u || tail_lanes == 0u) { return full_packet_work; }
        // The short-only validity tree additionally emits one integer
        // shuffle, one comparison, two selects, AND/OR and bool-to-u32 cast
        // per level, plus the initial lane < tail predicate. Common peer
        // addressing and CFG/PHI costs remain outside this uncalibrated prior.
        constexpr uint64_t kValidityPerLevel = 1u + 1u + 2u + 2u + 1u;
        constexpr uint64_t kTailPredicate = 1u;
        return full_packet_work + levels * kValidityPerLevel + kTailPredicate;
    }
};

[[nodiscard]] inline luisa::optional<ReductionEmissionPlan> reduction_emission_plan(
    const Operation &op, uint64_t total, const LowerOptions &options) noexcept {
    auto closed = closed_reduction(op);
    if (!closed || (!distributed_count(total, options) &&
                    (!bounded_count(total, options) || options.reduction_partitions <= 1u))) { return {}; }
    auto lanes = distributed_count(total, options) ? options.local_lanes : 1u;
    auto count = total / lanes;
    return ReductionEmissionPlan{*closed, count, std::min<uint64_t>(options.reduction_partitions, count),
                                 lanes, static_cast<uint32_t>(total % lanes)};
}

struct CarryAllocationPlan {
    bool buffered;
    bool argument_snapshot;
    bool result_snapshot;
    [[nodiscard]] uint64_t allocations() const noexcept {
        return buffered ? 2u : static_cast<uint64_t>(argument_snapshot) + result_snapshot;
    }
};

[[nodiscard]] inline CarryAllocationPlan carry_allocation_plan(
    const Value *argument, const Value *result, uint64_t count, const LowerOptions &options) noexcept {
    auto buffered = result->type().is_tile() && bounded_count(count, options);
    return {buffered, !buffered && definition_snapshot(argument, count, options),
            !buffered && definition_snapshot(result, count, options)};
}

}// namespace luisa::compute::tile::bridge::xir::detail
