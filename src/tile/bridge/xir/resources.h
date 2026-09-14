#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/verifier.h>
#include <luisa/xir/function.h>
#include "representation.h"
#include "program_plan.h"
#include "root_mapping.h"

namespace luisa::compute::tile::bridge::xir::detail {

// A static walk of the shared allocation/emission plans, not a second XIR
// lowering and not a dynamic traffic estimate. In particular, root launch
// volume never multiplies a worker's storage; an emitted runtime loop body
// counts once, while expanded map/reduction bodies count at every emission.
class ResourceAnalyzer final {
private:
    const Function &_function;
    const LowerOptions &_options;
    ResourceAnalysis _result;
    luisa::unordered_map<const Value *, uint32_t> _recipe_depths;
    luisa::optional<ProgramTeamPlan> _program_team;

    [[nodiscard]] bool _error(luisa::string_view message) noexcept {
        if (_result.error.empty()) { _result.error = message; }
        return false;
    }
    [[nodiscard]] bool _volume(const IndexSpace &space, uint64_t &count) noexcept {
        count = 1u;
        for (auto &axis : space.axes()) {
            if (!axis.extent.is_constant() || axis.extent.constant_value() > UINT32_MAX) {
                return _error("XIR realization requires static uint32-addressable extents");
            }
            auto extent = axis.extent.constant_value();
            if (extent != 0u && count > UINT32_MAX / extent) {
                return _error("XIR realization domain exceeds uint32 range");
            }
            count *= extent;
        }
        return true;
    }
    [[nodiscard]] bool _type(const Type &type) noexcept {
        if (type.kind() == TypeKind::INDEX) { return true; }
        if (type.kind() != TypeKind::SCALAR && !type.is_tile() && !type.is_view()) {
            return _error("unsupported value type in XIR worker realization; manual Memory is not implemented");
        }
        switch (type.scalar_type()) {
            case ScalarType::BOOL:
            case ScalarType::INT8:
            case ScalarType::UINT8:
            case ScalarType::INT16:
            case ScalarType::UINT16:
            case ScalarType::INT32:
            case ScalarType::UINT32:
            case ScalarType::INT64:
            case ScalarType::UINT64:
            case ScalarType::FLOAT16:
            case ScalarType::BFLOAT16:
            case ScalarType::FLOAT32:
            case ScalarType::FLOAT64: break;
            case ScalarType::FLOAT8_E4M3FN:
            case ScalarType::FLOAT8_E5M2:
                return _error("Tile to XIR: FP8 storage/conversion legalization is not implemented; use a capable bridge or explicitly unpack to FP16/FP32");
            default: return _error("unsupported scalar type in Tile to XIR bridge");
        }
        uint64_t count = 0u;
        return !type.index_space() || _volume(*type.index_space(), count);
    }
    [[nodiscard]] bool _validate_root_mapping(const Operation &op) noexcept {
        auto &domain = *op.domain();
        auto error = root_mapping_error(domain, _options.root_axis_order, _options.root_axis_tiles);
        if (!error.empty()) { return _error(error); }
        uint64_t count = 0u;
        if (!_volume(domain, count)) { return false; }
        if (count == 0u || count > UINT32_MAX / _options.local_lanes) {
            return _error("XIR root launch must be nonempty and uint32-addressable");
        }
        return true;
    }
    [[nodiscard]] bool _validate(const Block &block, bool root, uint32_t depth = 0u) noexcept {
        if (depth > 256u) { return _error("XIR resource analysis exceeds its region nesting limit"); }
        for (auto &argument : block.arguments()) {
            if (!_type(argument->type())) { return false; }
            if (root) {
                if (!argument->type().is_view()) { return _error("XIR Tile kernels currently require buffer View arguments"); }
                uint64_t count = 0u;
                if (!_volume(*argument->type().index_space(), count)) { return false; }
                if (count == 0u || count > SIZE_MAX / scalar_type_size(argument->type().scalar_type())) {
                    return _error("invalid XIR buffer footprint");
                }
            }
        }
        uint32_t roots = 0u;
        for (auto op : block.operations()) {
            if (root && op->kind() != OperationKind::CONSTANT && op->kind() != OperationKind::ELEMENTWISE && op->kind() != OperationKind::PARALLEL) {
                return _error("root effects require one explicit parallel execution domain");
            }
            if (op->memory_layout() || op->resource_class_constraint()) {
                return _error("XIR worker realization cannot realize manual Memory");
            }
            if (auto binding = op->execution_scope_constraint(); binding && *binding != "auto" && *binding != "worker") {
                return _error("XIR worker realization cannot honor this explicit execution binding");
            }
            if (op->domain()) {
                uint64_t count = 0u;
                if (!_volume(*op->domain(), count)) { return false; }
            }
            for (size_t i = 0u; i < op->result_count(); i++) {
                if (!_type(op->result(i)->type())) { return false; }
            }
            switch (op->kind()) {
                case OperationKind::CONSTANT: {
                    if (op->result_count() != 1u || op->region_count() != 0u || op->result(0u)->type().is_view()) {
                        return _error("XIR constants require exactly one scalar, index or Tile result and no regions");
                    }
                    auto attr = op->attribute("value");
                    if (!attr) { return _error("Tile constant is missing its value"); }
                    auto &payload = attr->value();
                    if (!luisa::get_if<bool>(&payload) && !luisa::get_if<int64_t>(&payload) &&
                        !luisa::get_if<uint64_t>(&payload) && !luisa::get_if<double>(&payload)) {
                        return _error("invalid Tile constant payload");
                    }
                    if (op->result(0u)->type().scalar_type() == ScalarType::BFLOAT16 && !luisa::get_if<double>(&payload)) {
                        return _error("BF16 constant requires a floating payload");
                    }
                    break;
                }
                case OperationKind::ELEMENTWISE:
                    switch (op->elementwise_op()) {
                        case ElementwiseOp::ADD:
                        case ElementwiseOp::SUB:
                        case ElementwiseOp::MUL:
                        case ElementwiseOp::DIV:
                        case ElementwiseOp::MOD:
                        case ElementwiseOp::NEG:
                        case ElementwiseOp::MIN:
                        case ElementwiseOp::MAX:
                        case ElementwiseOp::CAST:
                        case ElementwiseOp::SELECT:
                        case ElementwiseOp::EQ:
                        case ElementwiseOp::NE:
                        case ElementwiseOp::LT:
                        case ElementwiseOp::LE:
                        case ElementwiseOp::GT:
                        case ElementwiseOp::GE:
                        case ElementwiseOp::LOGICAL_AND:
                        case ElementwiseOp::LOGICAL_OR:
                        case ElementwiseOp::LOGICAL_NOT:
                        case ElementwiseOp::EXP:
                        case ElementwiseOp::LOG:
                        case ElementwiseOp::SQRT:
                        case ElementwiseOp::TANH:
                        case ElementwiseOp::ABS: break;
                        default: return _error("unsupported Tile elementwise opcode");
                    }
                    if (op->elementwise_op() == ElementwiseOp::CAST && op->result(0u)->type().scalar_type() == ScalarType::BFLOAT16) {
                        auto &from = op->operand(0u)->type();
                        if (from.kind() == TypeKind::INDEX || from.scalar_type() == ScalarType::FLOAT64 ||
                            from.scalar_type() == ScalarType::INT64 || from.scalar_type() == ScalarType::UINT64 ||
                            from.scalar_type() == ScalarType::INT32 || from.scalar_type() == ScalarType::UINT32) {
                            return _error("Tile to XIR: wide-source BF16 conversion requires an explicit intermediate cast<float>");
                        }
                    }
                    break;
                case OperationKind::VIEW_LOAD:
                case OperationKind::VIEW_STORE:
                    if (op->operand(0u)->argument_block() != _function.body().block(0u)) {
                        return _error("XIR view access requires a direct buffer argument");
                    }
                    break;
                case OperationKind::MMA:
                    if (op->result(0u)->type().scalar_type() == ScalarType::BFLOAT16) {
                        return _error("Tile to XIR: BF16 MMA accumulation requires an explicit FP32 accumulator");
                    }
                    break;
                case OperationKind::PARALLEL:
                    if (root && (++roots != 1u || op->result_count() != 0u || !_validate_root_mapping(*op))) {
                        return _error("XIR bridge requires one independent root parallel with no escaping results");
                    }
                    [[fallthrough]];
                case OperationKind::SERIAL:
                case OperationKind::PIPELINE:
                case OperationKind::REDUCE:
                case OperationKind::TILE_MAP:
                    if (op->region_count() != 1u || op->region(0u)->block_count() != 1u) {
                        return _error("XIR structured realization requires exactly one body block");
                    }
                    if (!_validate(*op->region(0u)->block(0u), false, depth + 1u)) { return false; }
                    break;
                case OperationKind::TILE_EXTRACT:
                case OperationKind::STAGE:
                case OperationKind::YIELD: break;
                default: return _error("unsupported TileIR operation in XIR worker realization; no fallback or effect erasure");
            }
        }
        return !root || roots == 1u || _error("XIR realization requires a root parallel domain");
    }
    [[nodiscard]] bool _add(ExecutionResources &to, ExecutionResources from, uint64_t repetitions = 1u) noexcept {
        auto add = [&](uint64_t &target, uint64_t value) {
            if (repetitions != 0u && value > (UINT64_MAX - target) / repetitions) {
                return _error("XIR static snapshot resource count exceeds uint64 range");
            }
            target += value * repetitions;
            return true;
        };
        return add(to.snapshot_bytes_per_worker, from.snapshot_bytes_per_worker) &&
               add(to.snapshot_allocations, from.snapshot_allocations);
    }
    [[nodiscard]] bool _snapshot(ExecutionResources &to, const Type &type, uint64_t count, uint64_t allocations = 1u) noexcept {
        auto bytes = snapshot_elements(count, _options) * scalar_type_size(type.scalar_type());
        return _add(to, ExecutionResources{bytes, 1u}, allocations);
    }
    [[nodiscard]] bool _recipe(const Value *value, const ValueAllocationPlan &plan) noexcept {
        if (plan.representation != ValueRepresentation::DEFERRED_MAP &&
            plan.representation != ValueRepresentation::DEFERRED_EXPRESSION) { return true; }
        // Verified SSA is visited in definition order. A materialized value or
        // carried argument terminates a recipe; only deferred physical inputs
        // contribute to the maximum read depth, not their number of uses.
        // Preflight both lowering and planning before their recursive readers.
        auto depth = 0u;
        auto input = [&](const Value *dependency) noexcept {
            if (auto found = _recipe_depths.find(dependency); found != _recipe_depths.end()) {
                depth = std::max(depth, found->second);
            }
        };
        auto op = value->defining_operation();
        if (plan.representation == ValueRepresentation::DEFERRED_MAP) {
            for (auto child : op->region(0u)->block(0u)->operations()) {
                if (child->kind() == OperationKind::TILE_EXTRACT) { input(child->operand(0u)); }
            }
        } else {
            for (size_t i = 0u; i < op->operand_count(); i++) { input(op->operand(i)); }
        }
        if (depth >= 64u) { return _error("XIR deferred recipe exceeds the depth budget"); }
        _recipe_depths.insert_or_assign(value, depth + 1u);
        return true;
    }
    [[nodiscard]] bool _team_snapshot(ExecutionResources &resources, const Value *value, uint64_t allocations = 1u) noexcept {
        auto elements = _program_team->layout(value).local_elements();
        auto bytes = elements * scalar_type_size(value->type().scalar_type());
        return _add(resources, ExecutionResources{bytes, 1u}, allocations);
    }
    [[nodiscard]] bool _team_operation(const Operation &op, ExecutionResources &resources) noexcept {
        auto kind = op.kind();
        if (kind == OperationKind::PARALLEL) { return _block(*op.region(0u)->block(0u), resources); }
        if (kind == OperationKind::SERIAL || kind == OperationKind::PIPELINE || kind == OperationKind::REDUCE) {
            // Temporal Tile state has one stable layout and two arrays for
            // simultaneous current/next copies. Results alias current; neither
            // body arguments nor loop results allocate again. Admitted closed
            // reductions carry only scalars and therefore add no arrays.
            for (size_t i = 0u; i < op.result_count(); i++) {
                if (op.result(i)->type().is_tile() && !_team_snapshot(resources, op.result(i), 2u)) { return false; }
            }
            return _block(*op.region(0u)->block(0u), resources);
        }
        if (kind == OperationKind::CONSTANT || kind == OperationKind::VIEW_STORE ||
            kind == OperationKind::TILE_EXTRACT || kind == OperationKind::STAGE || kind == OperationKind::YIELD) { return true; }
        // The new phase-aware path is eager: one array per nonconstant Tile
        // definition, with per-axis padded local volume. Every runtime map
        // emits its body once. Splat constants require no arrays.
        for (size_t i = 0u; i < op.result_count(); i++) {
            if (op.result(i)->type().is_tile() && !_team_snapshot(resources, op.result(i))) { return false; }
        }
        return kind != OperationKind::TILE_MAP || _block(*op.region(0u)->block(0u), resources);
    }
    [[nodiscard]] bool _operation(const Operation &op, ExecutionResources &resources, bool root) noexcept {
        if (_program_team) { return _team_operation(op, resources); }
        switch (op.kind()) {
            case OperationKind::PARALLEL:
            case OperationKind::SERIAL:
            case OperationKind::PIPELINE:
            case OperationKind::REDUCE: {
                auto &body = *op.region(0u)->block(0u);
                if (root) { return _block(body, resources); }
                uint64_t count = 0u;
                if (!_volume(*op.domain(), count)) { return false; }
                if (auto plan = reduction_emission_plan(op, count, _options)) {
                    ExecutionResources contribution;
                    if (!_block(body, contribution, plan->closed.update, plan->closed.yield)) { return false; }
                    return _add(resources, contribution, plan->emitted_bodies());
                }
                for (size_t i = 0u; i < op.result_count(); i++) {
                    auto result = op.result(i);
                    auto argument = body.argument(op.domain()->rank() + i);
                    uint64_t elements = 1u;
                    if (result->type().is_tile() && !_volume(*result->type().index_space(), elements)) { return false; }
                    auto plan = carry_allocation_plan(argument, result, elements, _options);
                    if (plan.allocations() && !_snapshot(resources, result->type(), elements, plan.allocations())) { return false; }
                }
                // Every ordered loop emits its body once, including zero-trip
                // loops. Buffered carry outputs alias their current array;
                // the two arrays and small SSA definition snapshots are above.
                return _block(body, resources);
            }
            case OperationKind::VIEW_STORE:
            case OperationKind::TILE_EXTRACT:
            case OperationKind::STAGE:
            case OperationKind::YIELD: return true;
            default: break;
        }
        auto value = op.result(0u);
        uint64_t count = 1u;
        if (value->type().is_tile() && !_volume(*value->type().index_space(), count)) { return false; }
        auto plan = value_allocation_plan(value, count, _options);
        if (!_recipe(value, plan)) { return false; }
        if (plan.snapshot && !_snapshot(resources, value->type(), count)) { return false; }
        if (op.kind() == OperationKind::TILE_MAP && plan.representation != ValueRepresentation::DEFERRED_MAP) {
            auto repetitions = map_emission_plan(op, count, _options).emitted_bodies();
            if (repetitions == 0u) { return true; }
            ExecutionResources body;
            if (!_block(*op.region(0u)->block(0u), body)) { return false; }
            return _add(resources, body, repetitions);
        }
        return true;
    }
    [[nodiscard]] bool _block(const Block &block, ExecutionResources &resources,
                              const Operation *skip = nullptr, const Operation *end = nullptr, bool root = false) noexcept {
        // Pointwise versioning emits a scalar-only fast path and retains one
        // eager alias fallback. Counting the original operations once is exact
        // for snapshots; neither erase the fallback nor count it twice.
        // Deferred maps admit only scalar definitions, so later recipe reads
        // cannot allocate arrays. Effects/aliases never change these plans.
        for (auto op : block.operations()) {
            if (op == end || op->kind() == OperationKind::YIELD) { break; }
            if (op != skip && !_operation(*op, resources, root)) { return false; }
        }
        return true;
    }

public:
    ResourceAnalyzer(const Function &function, const LowerOptions &options) noexcept
        : _function{function}, _options{options} {}
    [[nodiscard]] ResourceAnalysis run() noexcept {
        if (_function.parent_module() == nullptr || !verify(*_function.parent_module())) {
            static_cast<void>(_error("TileIR verification failed before XIR resource analysis"));
        } else if (_function.body().block_count() != 1u ||
                   !compute::xir::KernelFunction::is_valid_block_size(luisa::make_uint3(_options.block_size, 1u, 1u)) ||
                   _options.max_expanded_values == 0u || _options.reduction_partitions == 0u || _options.reduction_partitions > 16u ||
                   (_options.mma_output_block != 1u && _options.mma_output_block != 2u && _options.mma_output_block != 4u) ||
                   (_options.native_mma_vector_width != 0u &&
                    (_options.native_mma_vector_width < 2u || (_options.native_mma_vector_width & (_options.native_mma_vector_width - 1u)) != 0u)) ||
                   (_options.native_copy_vector_width != 0u &&
                    (_options.native_copy_vector_width < 2u || (_options.native_copy_vector_width & (_options.native_copy_vector_width - 1u)) != 0u)) ||
                   _options.local_lanes == 0u || (_options.local_lanes & (_options.local_lanes - 1u)) != 0u ||
                   _options.block_size % _options.local_lanes != 0u) {
            static_cast<void>(_error("invalid XIR realization options or entry region"));
        } else if (_validate(*_function.body().block(0u), true)) {
            if (_options.local_lanes > 1u && !packet_local_program(_function, _options.local_lanes)) {
                _program_team = ProgramTeamPlan::create(_function, _options.local_lanes);
                if (!_program_team) {
                    static_cast<void>(_error("XIR packet-local realization requires a common pointwise axis or an admitted phase-aware program-team layout with supported projections"));
                }
            }
            if (_result.error.empty()) {
                static_cast<void>(_block(*_function.body().block(0u), _result.resources, nullptr, nullptr, true));
            }
        }
        return std::move(_result);
    }
};

}// namespace luisa::compute::tile::bridge::xir::detail
