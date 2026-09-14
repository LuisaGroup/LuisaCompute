#pragma once

#include "representation.h"

namespace luisa::compute::tile::bridge::xir::detail {

struct PointwiseRegion {
    const IndexSpace *domain{nullptr};
    luisa::vector<const Operation *> operations;
    luisa::vector<const Operation *> loads;
    luisa::vector<const Operation *> stores;
    // Complete buffer intervals must be disjoint at runtime. Distinct
    // resource arguments never imply noalias; the original path is retained.
    struct AliasPair {
        const Value *a;
        const Value *b;
    };
    luisa::vector<AliasPair> alias_pairs;
};

[[nodiscard]] inline luisa::optional<int64_t> pointwise_integer(const Value *value) noexcept {
    auto op = value->defining_operation();
    auto attr = op && op->kind() == OperationKind::CONSTANT ? op->attribute("value") : nullptr;
    if (!attr) { return {}; }
    if (auto number = luisa::get_if<int64_t>(&attr->value())) { return *number; }
    if (auto number = luisa::get_if<uint64_t>(&attr->value()); number && *number <= INT64_MAX) { return static_cast<int64_t>(*number); }
    return {};
}

// A sufficient coordinate-box test on one common resource. Clipping to
// buffer bounds cannot make disjoint boxes overlap. No floating-point cost
// slope, argument name, or unchecked origin+extent addition is used.
[[nodiscard]] inline bool pointwise_disjoint(const Operation &a, const Operation &b) noexcept {
    if (a.operand(0u) != b.operand(0u) || !a.domain() || !b.domain() || a.domain()->rank() != b.domain()->rank()) { return false; }
    for (size_t i = 0u; i < a.domain()->rank(); i++) {
        auto x = pointwise_integer(a.operand(i + 1u)), y = pointwise_integer(b.operand(i + 1u));
        auto &nx = a.domain()->axis(i).extent;
        auto &ny = b.domain()->axis(i).extent;
        if (!x || !y || !nx.is_constant() || !ny.is_constant()) { continue; }
        // Subtraction in uint64 is the exact nonnegative distance of ordered
        // int64 values, including opposite signs, without signed overflow.
        if (*x <= *y && static_cast<uint64_t>(*y) - static_cast<uint64_t>(*x) >= nx.constant_value()) { return true; }
        if (*y <= *x && static_cast<uint64_t>(*x) - static_cast<uint64_t>(*y) >= ny.constant_value()) { return true; }
    }
    return false;
}

[[nodiscard]] inline bool pointwise_same_store(const Operation &a, const Operation &b) noexcept {
    if (a.operand(0u) != b.operand(0u) || a.domain() != b.domain()) { return false; }
    for (size_t i = 0u; i < a.domain()->rank(); i++) {
        auto x = a.operand(i + 1u), y = b.operand(i + 1u);
        if (x == y) { continue; }
        auto cx = pointwise_integer(x), cy = pointwise_integer(y);
        if (!cx || !cy || *cx != *cy) { return false; }
    }
    return true;
}

[[nodiscard]] inline luisa::vector<PointwiseRegion> pointwise_regions(
    const Block &block, uint32_t limit, uint32_t lanes) {
    luisa::vector<PointwiseRegion> regions;
    PointwiseRegion pending;
    auto finish = [&] {
        auto candidate = std::move(pending);
        pending = {};
        if (!candidate.domain || candidate.stores.empty() ||
            !(bounded_domain(*candidate.domain, limit) || (lanes > 1u && bounded_domain(*candidate.domain, lanes - 1u)))) { return; }
        // Every internal Tile is consumed inside this exact effect interval.
        // Escaping values retain their original snapshot; scalar definitions
        // are emitted once before the versioned paths and may escape safely.
        for (auto op : candidate.operations) {
            for (size_t i = 0u; i < op->result_count(); i++) {
                auto value = op->result(i);
                if (!value->type().is_tile()) { continue; }
                for (auto use : value->use_list()) {
                    if (std::find(candidate.operations.begin(), candidate.operations.end(), use->user()) == candidate.operations.end()) { return; }
                }
            }
        }
        auto separate = [&](const Operation &a, const Operation &b, bool stores) {
            auto x = a.operand(0u), y = b.operand(0u);
            if (x == y) { return pointwise_disjoint(a, b) || (stores && pointwise_same_store(a, b)); }
            for (auto pair : candidate.alias_pairs) {
                if ((pair.a == x && pair.b == y) || (pair.a == y && pair.b == x)) { return true; }
            }
            candidate.alias_pairs.emplace_back(PointwiseRegion::AliasPair{x, y});
            return true;
        };
        for (auto load : candidate.loads) {
            for (auto store : candidate.stores) {
                if (!separate(*load, *store, false)) { return; }
            }
        }
        for (size_t i = 0u; i < candidate.stores.size(); i++) {
            for (size_t j = i + 1u; j < candidate.stores.size(); j++) {
                if (!separate(*candidate.stores[i], *candidate.stores[j], true)) { return; }
            }
        }
        regions.emplace_back(std::move(candidate));
    };
    for (auto op : block.operations()) {
        auto kind = op->kind();
        bool simple = !op->region_count() && !op->memory_layout() && !op->resource_class_constraint() && !op->execution_scope_constraint() &&
                      (kind == OperationKind::CONSTANT || kind == OperationKind::ELEMENTWISE ||
                       ((kind == OperationKind::VIEW_LOAD || kind == OperationKind::VIEW_STORE) && op->domain()));
        auto fits = [&] {
            const IndexSpace *domain = pending.domain;
            auto accept = [&](const IndexSpace &space) {
                for (auto &axis : space.axes()) {
                    if (!axis.extent.is_constant() || !axis.extent.constant_value()) { return false; }
                }
                if (domain && *domain != space) { return false; }
                domain = &space;
                return true;
            };
            if (op->domain() && !accept(*op->domain())) { return false; }
            for (size_t i = 0u; i < op->operand_count(); i++) {
                if (op->operand(i)->type().is_tile() && !accept(*op->operand(i)->type().index_space())) { return false; }
            }
            for (size_t i = 0u; i < op->result_count(); i++) {
                if (op->result(i)->type().is_tile() && !accept(*op->result(i)->type().index_space())) { return false; }
            }
            pending.domain = domain;
            return true;
        };
        if (!simple || pending.operations.size() == 256u ||
            (kind == OperationKind::VIEW_LOAD && !pending.stores.empty()) || !fits()) { finish(); }
        if (!simple || !fits()) { continue; }
        pending.operations.emplace_back(op);
        if (kind == OperationKind::VIEW_LOAD) { pending.loads.emplace_back(op); }
        if (kind == OperationKind::VIEW_STORE) { pending.stores.emplace_back(op); }
    }
    finish();
    return regions;
}

}// namespace luisa::compute::tile::bridge::xir::detail
