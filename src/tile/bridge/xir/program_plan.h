#pragma once

#include <algorithm>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include "program_team.h"
#include "representation.h"

namespace luisa::compute::tile::bridge::xir::detail {

// One executable candidate for a logical program team. Phase schedules and SSA
// placement are separate: a consumer may read a replicated value, its own
// element, or one completely uniform projection from another owner. This is
// neither a new TileIR semantic restriction nor a target cost model.
//
// The caller supplies verified TileIR and separately checks target packet and
// launch capabilities. Every nonconstant Tile has an eager snapshot; traversals
// are rolled. Broadcasts must execute convergently, including lanes whose
// output coordinate is padding. Only local reads/stores may be tail-guarded.
class ProgramTeamPlan final {
private:
    struct PlacementGroup {
        size_t parent;
        luisa::vector<const Value *> values;
        luisa::optional<Dim> axis;
        bool replicated;
        bool constrained{false};
    };

    ProgramTeamLayout _team;
    luisa::unordered_map<const Value *, ValueLayout> _layouts;
    luisa::unordered_map<const Operation *, ValueLayout> _phases;
    luisa::unordered_map<const Operation *, luisa::vector<ReadProjection>> _projections;
    // Construction-only state, discarded once the candidate is admitted.
    luisa::vector<const Operation *> _operations;
    luisa::vector<PlacementGroup> _groups;
    luisa::unordered_map<const Value *, size_t> _value_groups;
    const Operation *_root{nullptr};

    explicit ProgramTeamPlan(ProgramTeamLayout team) noexcept : _team{team} {}

    [[nodiscard]] static bool _constant(const Value *value) noexcept {
        auto op = value->defining_operation();
        return op && op->kind() == OperationKind::CONSTANT;
    }
    [[nodiscard]] static bool _shape(const IndexSpace &space) noexcept {
        if (!space.is_valid()) { return false; }
        for (auto &axis : space.axes()) {
            if (!axis.extent.is_constant() || axis.extent.constant_value() == 0u ||
                axis.extent.constant_value() > UINT32_MAX) { return false; }
        }
        auto count = space.static_volume();
        return count && *count != 0u && *count <= UINT32_MAX;
    }
    [[nodiscard]] static luisa::optional<Dim> _last_axis(const IndexSpace &space) noexcept {
        for (auto i = space.rank(); i != 0u; i--) {
            if (space.axis(i - 1u).extent.constant_value() > 1u) { return space.axis(i - 1u).dimension; }
        }
        return {};
    }
    [[nodiscard]] static bool _contains(const IndexSpace &space, Dim axis) noexcept {
        auto i = space.axis_index(axis);
        return i && space.axis(*i).extent.constant_value() > 1u;
    }
    [[nodiscard]] static bool _collective_map(const Operation &op) noexcept {
        if (op.kind() != OperationKind::TILE_MAP) { return false; }
        for (auto child : op.region(0u)->block(0u)->operations()) {
            if (child->kind() == OperationKind::REDUCE) { return true; }
        }
        return false;
    }
    [[nodiscard]] size_t _group(size_t index) const noexcept {
        while (_groups[index].parent != index) { index = _groups[index].parent; }
        return index;
    }
    [[nodiscard]] bool _add_value(const Value *value) noexcept {
        if (!value->type().is_tile()) { return true; }
        auto &space = *value->type().index_space();
        if (!_shape(space)) { return false; }
        if (_constant(value)) {
            _layouts.emplace(value, *ValueLayout::replicated(_team, space));
            return true;
        }
        if (_value_groups.contains(value)) { return true; }
        auto producer = value->defining_operation();
        auto replicated = producer && _collective_map(*producer);
        auto index = _groups.size();
        _value_groups.emplace(value, index);
        _groups.emplace_back(PlacementGroup{index, {value}, replicated ? luisa::nullopt : _last_axis(space), replicated});
        return true;
    }
    [[nodiscard]] bool _collect(const Block &block, bool root, bool element_region, uint32_t depth) noexcept {
        if (depth > 64u) { return false; }
        for (auto &argument : block.arguments()) {
            if (!_add_value(argument.get())) { return false; }
        }
        for (auto op : block.operations()) {
            if (op->memory_layout() || op->resource_class_constraint()) { return false; }
            if (auto binding = op->execution_scope_constraint(); binding && *binding != "auto") { return false; }
            if (op->domain() && !_shape(*op->domain())) { return false; }
            if (op->kind() == OperationKind::PARALLEL) {
                // The current backend launches one root grid. This does not
                // forbid nested/sibling parallel scopes in the language.
                if (!root || _root || !op->domain() || op->region_count() != 1u ||
                    op->region(0u)->block_count() != 1u) { return false; }
                _root = op;
                _operations.emplace_back(op);
                if (!_collect(*op->region(0u)->block(0u), false, false, depth + 1u)) { return false; }
                continue;
            }
            if (root && op->kind() != OperationKind::CONSTANT && op->kind() != OperationKind::ELEMENTWISE) { return false; }
            for (size_t i = 0u; i < op->result_count(); i++) {
                if ((root || element_region) && op->result(i)->type().is_tile()) { return false; }
                if (!_add_value(op->result(i))) { return false; }
            }
            _operations.emplace_back(op);
            switch (op->kind()) {
                case OperationKind::CONSTANT:
                case OperationKind::ELEMENTWISE:
                case OperationKind::TILE_EXTRACT:
                case OperationKind::YIELD:
                case OperationKind::STAGE: break;
                case OperationKind::VIEW_LOAD:
                case OperationKind::VIEW_STORE:
                    if (element_region || !op->domain()) { return false; }
                    break;
                case OperationKind::MMA:
                    if (element_region) { return false; }
                    break;
                case OperationKind::TILE_MAP:
                case OperationKind::REDUCE:
                case OperationKind::SERIAL:
                case OperationKind::PIPELINE: {
                    if (!op->domain() || op->region_count() != 1u || op->region(0u)->block_count() != 1u) { return false; }
                    auto reduction = op->kind() == OperationKind::REDUCE;
                    if (element_region && !reduction) { return false; }
                    if (reduction) {
                        auto nonunit = 0u;
                        for (auto &axis : op->domain()->axes()) { nonunit += axis.extent.constant_value() > 1u; }
                        if (nonunit > 1u || !closed_reduction(*op)) { return false; }
                    }
                    if (!_collect(*op->region(0u)->block(0u), false,
                                  reduction || op->kind() == OperationKind::TILE_MAP, depth + 1u)) { return false; }
                    break;
                }
                default: return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool _join(const Value *a, const Value *b) noexcept {
        if (!(a->type() == b->type())) { return false; }
        // A constant seed is a splat, not a physically distributed array.
        // Its initialization can populate any compatible carry placement.
        if (_constant(a) || _constant(b)) { return true; }
        auto x = _group(_value_groups.at(a)), y = _group(_value_groups.at(b));
        if (x == y) { return true; }
        _groups[y].parent = x;
        auto &left = _groups[x];
        auto &right = _groups[y];
        left.values.insert(left.values.end(), right.values.begin(), right.values.end());
        left.replicated |= right.replicated;
        if (left.replicated) { left.axis.reset(); }
        right.values.clear();
        return true;
    }
    [[nodiscard]] bool _carries() noexcept {
        for (auto op : _operations) {
            if (op->kind() != OperationKind::SERIAL && op->kind() != OperationKind::PIPELINE) { continue; }
            auto body = op->region(0u)->block(0u);
            auto yield = body->operations().back();
            for (size_t i = 0u; i < op->result_count(); i++) {
                auto result = op->result(i);
                if (!result->type().is_tile()) { continue; }
                auto argument = body->argument(op->domain()->rank() + i);
                if (!_join(argument, result) || !_join(argument, op->operand(i)) || !_join(argument, yield->operand(i))) { return false; }
            }
        }
        return true;
    }

    // Demand flows through pointwise/MMA producer edges and simultaneous carry
    // components, not through a globally preferred dimension name. Conflicting
    // demands fail this candidate; no unimplemented redistribution is invented.
    [[nodiscard]] bool _require(const Value *value, Dim axis, uint32_t depth = 0u) noexcept {
        if (depth > 128u) { return false; }
        if (_constant(value)) { return true; }
        auto index = _group(_value_groups.at(value));
        auto &group = _groups[index];
        if (group.replicated) { return true; }
        if (group.constrained) { return group.axis && *group.axis == axis; }
        if (!_contains(*value->type().index_space(), axis)) { return false; }
        group.axis = axis;
        group.constrained = true;
        for (auto member : group.values) {
            auto producer = member->defining_operation();
            if (!producer || (producer->kind() != OperationKind::ELEMENTWISE && producer->kind() != OperationKind::MMA)) { continue; }
            for (size_t i = 0u; i < producer->operand_count(); i++) {
                auto input = producer->operand(i);
                if (input->type().is_tile() && _contains(*input->type().index_space(), axis) && !_require(input, axis, depth + 1u)) { return false; }
            }
        }
        return true;
    }
    [[nodiscard]] luisa::optional<Dim> _axis(const Value *value) const noexcept {
        if (_constant(value)) { return {}; }
        return _groups[_group(_value_groups.at(value))].axis;
    }
    [[nodiscard]] luisa::optional<Dim> _phase_axis(const Operation *op) const noexcept {
        if (op->kind() == OperationKind::PARALLEL || op->kind() == OperationKind::SERIAL ||
            op->kind() == OperationKind::PIPELINE) { return {}; }
        if (op->kind() == OperationKind::REDUCE) { return _last_axis(*op->domain()); }
        if (op->result_count() == 1u && op->result(0u)->type().is_tile()) { return _axis(op->result(0u)); }
        if (op->kind() == OperationKind::VIEW_STORE) { return _last_axis(*op->domain()); }
        return {};
    }
    [[nodiscard]] luisa::optional<Dim> _coordinate_axis(const Value *value) const noexcept {
        auto block = value->argument_block();
        auto nest = block ? block->parent_region()->parent_operation() : nullptr;
        if (!nest || !nest->domain() || value->index() >= nest->domain()->rank()) { return {}; }
        auto axis = _phase_axis(nest);
        return axis && nest->domain()->axis(value->index()).dimension == *axis ? axis : luisa::nullopt;
    }
    [[nodiscard]] bool _demands() noexcept {
        // Matrix output coordinates expose an operand's owner-preserving axis.
        // E.g. rhs [n,k] of output [m,n] is distributed over n, irrespective of
        // its storage-axis order; contracted coordinates stay team-uniform.
        for (auto op : _operations) {
            if (op->kind() == OperationKind::MMA) {
                if (auto axis = _axis(op->result(0u)); axis && !_require(op->result(0u), *axis)) { return false; }
            }
        }
        // Pure map schedules can themselves change through a downstream demand.
        // Iterate until no new placement component becomes constrained.
        for (size_t iteration = 0u; iteration <= _groups.size(); iteration++) {
            auto before = std::count_if(_groups.begin(), _groups.end(), [](auto &g) { return g.constrained; });
            for (auto op : _operations) {
                if (op->kind() != OperationKind::TILE_EXTRACT) { continue; }
                auto source = op->operand(0u);
                auto &space = *source->type().index_space();
                for (size_t i = 0u; i < space.rank(); i++) {
                    auto axis = _coordinate_axis(op->operand(i + 1u));
                    if (axis && space.axis(i).dimension == *axis && space.axis(i).extent.constant_value() > 1u &&
                        !_require(source, *axis)) { return false; }
                }
            }
            auto after = std::count_if(_groups.begin(), _groups.end(), [](auto &g) { return g.constrained; });
            if (before == after) { return true; }
        }
        return false;
    }
    void _materialize_layouts() noexcept {
        for (auto &[value, index] : _value_groups) {
            auto &space = *value->type().index_space();
            auto axis = _groups[_group(index)].axis;
            _layouts.emplace(value, axis ? *ValueLayout::cyclic(_team, space, *axis) : *ValueLayout::replicated(_team, space));
        }
        for (auto op : _operations) {
            if (op->result_count() == 1u && op->result(0u)->type().is_tile() &&
                op->kind() != OperationKind::SERIAL && op->kind() != OperationKind::PIPELINE) {
                _phases.emplace(op, layout(op->result(0u)));
            } else {
                auto space = op->domain() ? *op->domain() : IndexSpace{};
                auto axis = _phase_axis(op);
                _phases.emplace(op, axis ? *ValueLayout::cyclic(_team, space, *axis) : *ValueLayout::replicated(_team, space));
            }
            _projections.emplace(op, luisa::vector<ReadProjection>(op->operand_count(), ReadProjection::UNKNOWN));
        }
    }
    [[nodiscard]] static bool _zero(const Value *value) noexcept {
        auto op = value->defining_operation();
        auto literal = op && op->kind() == OperationKind::CONSTANT ? op->attribute("value") : nullptr;
        if (!literal) { return false; }
        auto s = luisa::get_if<int64_t>(&literal->value());
        auto u = luisa::get_if<uint64_t>(&literal->value());
        return (s && *s == 0) || (u && *u == 0u);
    }
    [[nodiscard]] bool _extract(const Operation &op) noexcept {
        auto source = op.operand(0u);
        auto &space = *source->type().index_space();
        auto uniform = true, owner = false;
        for (size_t i = 0u; i < space.rank(); i++) {
            auto index = op.operand(i + 1u);
            if (_zero(index)) { continue; }
            auto block = index->argument_block();
            auto nest = block ? block->parent_region()->parent_operation() : nullptr;
            if (!nest || !nest->domain() || index->index() >= nest->domain()->rank()) { return false; }
            auto &coordinate = nest->domain()->axis(index->index());
            auto &source_axis = space.axis(i);
            // Direct, in-bounds coordinates only. Unit axes may use any known
            // unit coordinate; arithmetic and data-dependent gathers are a
            // separate capability, not silently classified as broadcasts.
            if (source_axis.extent.constant_value() == 1u) {
                if (coordinate.extent.constant_value() != 1u) { return false; }
            } else if (source_axis.dimension != coordinate.dimension || source_axis.extent != coordinate.extent) {
                return false;
            }
            if (auto varying = _coordinate_axis(index)) {
                uniform = false;
                auto distributed = layout(source).cyclic_axis();
                owner |= distributed && *distributed == source_axis.dimension && *varying == *distributed;
            }
        }
        auto projection = uniform ? ReadProjection::TEAM_UNIFORM : owner ? ReadProjection::OWNER_PRESERVING :
                                                                           ReadProjection::UNKNOWN;
        _projections.at(&op)[0u] = projection;
        return layout(source).read_transition(projection) != ReadTransition::UNSUPPORTED;
    }
    [[nodiscard]] bool _uniform(const Value *value, uint32_t depth = 0u) const noexcept {
        if (depth > 128u || value->type().is_tile()) { return false; }
        if (auto block = value->argument_block()) {
            auto parent = block->parent_region()->parent_operation();
            if (!parent) { return true; }// Kernel parameter.
            if (value->index() < parent->domain()->rank()) { return !_coordinate_axis(value); }
            // Temporal carry uniformity is an inductive invariant, checked
            // against both its initializer and its yielded value below.
            return parent->kind() == OperationKind::SERIAL || parent->kind() == OperationKind::PIPELINE;
        }
        auto op = value->defining_operation();
        if (!op) { return false; }
        switch (op->kind()) {
            case OperationKind::CONSTANT:
            case OperationKind::REDUCE:
            case OperationKind::SERIAL:
            case OperationKind::PIPELINE: return true;
            case OperationKind::TILE_EXTRACT: return extract_projection(op) == ReadProjection::TEAM_UNIFORM;
            case OperationKind::ELEMENTWISE:
                for (size_t i = 0u; i < op->operand_count(); i++) {
                    if (!_uniform(op->operand(i), depth + 1u)) { return false; }
                }
                return true;
            default: return false;
        }
    }
    [[nodiscard]] bool _project(const Operation &op, size_t operand, const IndexSpace &domain) noexcept {
        auto value = op.operand(operand);
        if (!value->type().is_tile()) {
            if (!_uniform(value)) { return false; }
            _projections.at(&op)[operand] = ReadProjection::TEAM_UNIFORM;
            return true;
        }
        auto &source = layout(value);
        auto varying = phase(&op).cyclic_axis();
        auto uniform = true;
        for (auto &axis : source.space().axes()) {
            if (axis.extent.constant_value() == 1u) { continue; }
            auto found = domain.axis_index(axis.dimension);
            if (!found || domain.axis(*found).extent != axis.extent) { return false; }
            uniform &= !varying || *varying != axis.dimension;
        }
        auto owner = varying && source.cyclic_axis() && *varying == *source.cyclic_axis();
        auto projection = uniform ? ReadProjection::TEAM_UNIFORM : owner ? ReadProjection::OWNER_PRESERVING :
                                                                           ReadProjection::UNKNOWN;
        _projections.at(&op)[operand] = projection;
        return source.read_transition(projection) != ReadTransition::UNSUPPORTED;
    }
    [[nodiscard]] static bool _integer(ScalarType type) noexcept {
        switch (type) {
            case ScalarType::INT8:
            case ScalarType::UINT8:
            case ScalarType::INT16:
            case ScalarType::UINT16:
            case ScalarType::INT32:
            case ScalarType::UINT32:
            case ScalarType::INT64:
            case ScalarType::UINT64: return true;
            default: return false;
        }
    }
    [[nodiscard]] static bool _safe_arithmetic(const Operation &op) noexcept {
        auto &result_type = op.result(0u)->type();
        if (result_type.kind() != TypeKind::INDEX && !_integer(result_type.scalar_type())) { return true; }
        if (op.elementwise_op() == ElementwiseOp::CAST) {
            // A masked input can be zero, or an unused floating recipe can be
            // NaN/Inf. Do not speculate a potentially undefined float-to-int
            // conversion until the emitter has a predicated conversion path.
            auto &source = op.operand(0u)->type();
            return source.kind() == TypeKind::INDEX || source.scalar_type() == ScalarType::BOOL || _integer(source.scalar_type());
        }
        if (op.elementwise_op() != ElementwiseOp::DIV && op.elementwise_op() != ElementwiseOp::MOD) { return true; }
        auto divisor = op.operand(1u)->defining_operation();
        auto literal = divisor && divisor->kind() == OperationKind::CONSTANT ? divisor->attribute("value") : nullptr;
        if (!literal) { return false; }
        if (auto value = luisa::get_if<uint64_t>(&literal->value())) { return *value != 0u; }
        if (auto value = luisa::get_if<int64_t>(&literal->value())) {
            // Also exclude the signed minimum / -1 overflow case.
            return *value != 0 && *value != -1;
        }
        return false;
    }
    [[nodiscard]] bool _validate() noexcept {
        // Establish all explicit projections before uniformity queries on
        // scalar recipes. Definition order is not used as a hidden proof.
        for (auto op : _operations) {
            if (op->kind() == OperationKind::TILE_EXTRACT && !_extract(*op)) { return false; }
        }
        for (auto op : _operations) {
            switch (op->kind()) {
                case OperationKind::ELEMENTWISE:
                    if (!_safe_arithmetic(*op)) { return false; }
                    if (op->result_count() == 1u && op->result(0u)->type().is_tile()) {
                        for (size_t i = 0u; i < op->operand_count(); i++) {
                            if (!_project(*op, i, phase(op).space())) { return false; }
                        }
                    }
                    break;
                case OperationKind::MMA: {
                    auto domain = phase(op).space();
                    for (size_t i = 0u; i < 2u; i++) {
                        for (auto &axis : op->operand(i)->type().index_space()->axes()) {
                            if (!domain.axis_index(axis.dimension) && !domain.add(axis.dimension, axis.extent)) { return false; }
                        }
                    }
                    for (size_t i = 0u; i < op->operand_count(); i++) {
                        if (!_project(*op, i, domain)) { return false; }
                    }
                    break;
                }
                case OperationKind::VIEW_LOAD:
                case OperationKind::VIEW_STORE:
                    for (size_t i = 1u; i < op->operand_count(); i++) {
                        if (!_project(*op, i, phase(op).space())) { return false; }
                    }
                    break;
                case OperationKind::REDUCE:
                    if (!_uniform(op->operand(0u))) { return false; }
                    break;
                case OperationKind::SERIAL:
                case OperationKind::PIPELINE: {
                    auto yield = op->region(0u)->block(0u)->operations().back();
                    for (size_t i = 0u; i < op->result_count(); i++) {
                        if (!op->result(i)->type().is_tile() &&
                            (!_uniform(op->operand(i)) || !_uniform(yield->operand(i)))) { return false; }
                    }
                    break;
                }
                default: break;
            }
        }
        return true;
    }

public:
    [[nodiscard]] static luisa::optional<ProgramTeamPlan> create(const Function &function, uint32_t width) noexcept {
        auto team = ProgramTeamLayout::create(width);
        if (!team || width <= 1u || function.body().block_count() != 1u || packet_local_program(function, width)) { return {}; }
        ProgramTeamPlan plan{*team};
        if (!plan._collect(*function.body().block(0u), true, false, 0u) || !plan._root ||
            !plan._carries() || !plan._demands()) { return {}; }
        plan._materialize_layouts();
        if (!plan._validate()) { return {}; }
        plan._operations.clear();
        // Only frozen placements, phase geometries and per-use facts survive.
        plan._groups.clear();
        plan._value_groups.clear();
        return plan;
    }
    [[nodiscard]] ProgramTeamLayout team() const noexcept { return _team; }
    [[nodiscard]] const ValueLayout &layout(const Value *value) const noexcept {
        auto found = _layouts.find(value);
        LUISA_ASSERT(found != _layouts.end(), "No program-team placement for this TileIR value");
        return found->second;
    }
    [[nodiscard]] const ValueLayout &phase(const Operation *operation) const noexcept {
        auto found = _phases.find(operation);
        LUISA_ASSERT(found != _phases.end(), "No program-team schedule for this TileIR operation");
        return found->second;
    }
    [[nodiscard]] ReadProjection operand_projection(const Operation *operation, size_t operand) const noexcept {
        auto found = _projections.find(operation);
        LUISA_ASSERT(found != _projections.end() && operand < found->second.size(), "No program-team operand projection");
        return found->second[operand];
    }
    [[nodiscard]] ReadProjection extract_projection(const Operation *operation) const noexcept { return operand_projection(operation, 0u); }
    [[nodiscard]] ReadProjection mma_operand_projection(const Operation *operation, size_t operand) const noexcept { return operand_projection(operation, operand); }
};

}// namespace luisa::compute::tile::bridge::xir::detail
