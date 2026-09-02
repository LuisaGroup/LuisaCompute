#include <algorithm>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/verifier.h>

namespace luisa::compute::tile {

namespace detail {

class Verifier final {

private:
    const Module &_module;
    const TargetModel *_target;
    VerificationResult _result;
    const Function *_function{nullptr};
    luisa::unordered_set<const Value *> _values;
    luisa::unordered_set<const Use *> _uses;
    luisa::unordered_set<uint64_t> _value_ids;
    luisa::unordered_set<uint64_t> _operation_ids;

private:
    void _error(const Operation *operation, luisa::string message) noexcept {
        _result._diagnostics.emplace_back(VerificationDiagnostic{_function, operation, std::move(message)});
    }

    [[nodiscard]] bool _space_belongs_to_module(const IndexSpace &space) const noexcept {
        if (!space.is_valid()) { return false; }
        for (auto &&axis : space.axes()) {
            if (!_module.dimensions().owns(axis.dimension)) { return false; }
            if (axis.extent.is_dynamic() && !_module.dimensions().owns(axis.extent.dynamic_value())) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool _type_belongs_to_module(const Type &type) const noexcept {
        if (!type.is_valid()) { return false; }
        auto space = type.index_space();
        return space == nullptr || _space_belongs_to_module(*space);
    }

    [[nodiscard]] static luisa::optional<size_t> _operation_index(const Block *block, const Operation *operation) noexcept {
        auto &&operations = block->operations();
        for (auto i = 0u; i < operations.size(); i++) {
            if (operations[i].get() == operation) { return i; }
        }
        return luisa::nullopt;
    }

    [[nodiscard]] static bool _definition_precedes_use(const Value *value, const Operation *user) noexcept {
        auto user_block = user->parent_block();
        if (value->origin() == Value::Origin::BLOCK_ARGUMENT) {
            auto definition_block = value->argument_block();
            if (definition_block == nullptr || user_block == nullptr) { return false; }
            if (definition_block == user_block) { return true; }
            auto region = user_block->parent_region();
            const Operation *containing_operation = nullptr;
            while (region != nullptr && region != definition_block->parent_region()) {
                containing_operation = region->parent_operation();
                if (containing_operation == nullptr) { return false; }
                region = containing_operation->parent_block()->parent_region();
            }
            return region == definition_block->parent_region() && containing_operation != nullptr &&
                   containing_operation->parent_block() == definition_block;
        }
        auto definition = value->defining_operation();
        if (definition == nullptr || user_block == nullptr) { return false; }
        auto definition_block = definition->parent_block();
        if (definition_block == user_block) {
            auto definition_index = _operation_index(definition_block, definition);
            auto user_index = _operation_index(user_block, user);
            return definition_index && user_index && *definition_index < *user_index;
        }
        auto region = user_block->parent_region();
        const Operation *containing_operation = nullptr;
        while (region != nullptr && region != definition_block->parent_region()) {
            containing_operation = region->parent_operation();
            if (containing_operation == nullptr) { return false; }
            region = containing_operation->parent_block()->parent_region();
        }
        if (region != definition_block->parent_region() || containing_operation == nullptr ||
            containing_operation->parent_block() != definition_block) { return false; }
        auto definition_index = _operation_index(definition_block, definition);
        auto containing_index = _operation_index(definition_block, containing_operation);
        return definition_index && containing_index && *definition_index < *containing_index;
    }

    [[nodiscard]] static const Value *_memory_owner_of_state(const Value *state) noexcept {
        if (state == nullptr || state->type().kind() != TypeKind::MEMORY_STATE ||
            state->origin() != Value::Origin::OPERATION_RESULT) { return nullptr; }
        auto operation = state->defining_operation();
        if (operation->kind() == OperationKind::MEMORY_ALLOC && state->index() == 1u && operation->result_count() >= 2u) {
            return operation->result(0u);
        }
        if (operation->kind() == OperationKind::MEMORY_STORE && state->index() == 0u && operation->operand_count() >= 2u) {
            return operation->operand(0u);
        }
        return nullptr;
    }

    [[nodiscard]] static bool _same_axis_extent(const IndexSpace &lhs, const IndexSpace &rhs, Dim dimension) noexcept {
        auto lhs_index = lhs.axis_index(dimension);
        auto rhs_index = rhs.axis_index(dimension);
        return lhs_index && rhs_index && lhs.axis(*lhs_index).extent == rhs.axis(*rhs_index).extent;
    }

    void _verify_mma(const Operation *operation) noexcept {
        if (operation->operand_count() != 3u || operation->result_count() != 1u) {
            _error(operation, "mma requires three operands and one result");
            return;
        }
        auto a = operation->operand(0u);
        auto b = operation->operand(1u);
        auto accumulator = operation->operand(2u);
        if (a == nullptr || b == nullptr || accumulator == nullptr ||
            !a->type().is_tile() || !b->type().is_tile() || !accumulator->type().is_tile()) {
            _error(operation, "mma operands must be Tile values");
            return;
        }
        if (!(operation->result(0u)->type() == accumulator->type())) {
            _error(operation, "mma result type must equal the accumulator type");
        }
        auto &&a_space = *a->type().index_space();
        auto &&b_space = *b->type().index_space();
        auto &&c_space = *accumulator->type().index_space();
        size_t contracted_dimensions = 0u;
        for (auto &&axis : a_space.axes()) {
            auto in_b = b_space.contains(axis.dimension);
            auto in_c = c_space.contains(axis.dimension);
            if (!in_c && !in_b) {
                _error(operation, "an A dimension absent from C must be contracted with B");
            } else if ((in_b && !_same_axis_extent(a_space, b_space, axis.dimension)) ||
                       (in_c && !_same_axis_extent(a_space, c_space, axis.dimension))) {
                _error(operation, "mma extents disagree for one logical dimension");
            }
            if (in_b && !in_c) { contracted_dimensions++; }
        }
        for (auto &&axis : b_space.axes()) {
            auto in_a = a_space.contains(axis.dimension);
            auto in_c = c_space.contains(axis.dimension);
            if (!in_c && !in_a) {
                _error(operation, "a B dimension absent from C must be contracted with A");
            } else if (in_c && !_same_axis_extent(b_space, c_space, axis.dimension)) {
                _error(operation, "mma extents disagree for one logical dimension");
            }
        }
        for (auto &&axis : c_space.axes()) {
            if (!a_space.contains(axis.dimension) && !b_space.contains(axis.dimension)) {
                _error(operation, "every accumulator dimension must originate in A or B");
            }
        }
        if (contracted_dimensions == 0u) { _error(operation, "mma requires at least one contracted logical dimension"); }
    }

    void _verify_memory(const Operation *operation) noexcept {
        switch (operation->kind()) {
            case OperationKind::MEMORY_ALLOC:
                if (operation->operand_count() != 0u || operation->result_count() != 2u ||
                    !operation->result(0u)->type().is_memory() ||
                    operation->result(1u)->type().kind() != TypeKind::MEMORY_STATE) {
                    _error(operation, "memory.alloc must produce one Memory and its initial MemoryState");
                }
                break;
            case OperationKind::MEMORY_LOAD:
                if (operation->operand_count() != 2u || operation->result_count() != 1u ||
                    operation->operand(0u) == nullptr || !operation->operand(0u)->type().is_memory() ||
                    operation->operand(1u) == nullptr || operation->operand(1u)->type().kind() != TypeKind::MEMORY_STATE) {
                    _error(operation, "memory.load requires (Memory, MemoryState) and one Tile result");
                } else {
                    if (!(operation->result(0u)->type() == operation->operand(0u)->type().tile_value_type())) {
                        _error(operation, "memory.load result must match the Memory element space");
                    }
                    if (_memory_owner_of_state(operation->operand(1u)) != operation->operand(0u)) {
                        _error(operation, "memory.load state does not belong to the referenced Memory");
                    }
                }
                break;
            case OperationKind::MEMORY_STORE:
                if (operation->operand_count() != 3u || operation->result_count() != 1u ||
                    operation->operand(0u) == nullptr || !operation->operand(0u)->type().is_memory() ||
                    operation->operand(1u) == nullptr || operation->operand(1u)->type().kind() != TypeKind::MEMORY_STATE ||
                    operation->operand(2u) == nullptr || !operation->operand(2u)->type().is_tile() ||
                    operation->result(0u)->type().kind() != TypeKind::MEMORY_STATE) {
                    _error(operation, "memory.store requires (Memory, MemoryState, Tile) and a new MemoryState");
                } else {
                    if (!(operation->operand(2u)->type() == operation->operand(0u)->type().tile_value_type())) {
                        _error(operation, "memory.store Tile must match the Memory element space");
                    }
                    if (_memory_owner_of_state(operation->operand(1u)) != operation->operand(0u)) {
                        _error(operation, "memory.store state does not belong to the referenced Memory");
                    }
                }
                break;
            default: break;
        }
    }

    void _verify_core_operation(const Operation *operation) noexcept {
        auto structured = operation->kind() == OperationKind::PARALLEL ||
                          operation->kind() == OperationKind::SERIAL ||
                          operation->kind() == OperationKind::PIPELINE ||
                          operation->kind() == OperationKind::REDUCE;
        if (operation->kind() == OperationKind::CUSTOM && operation->name().empty()) {
            _error(operation, "custom operation name must not be empty");
        }
        if (structured) {
            if (!operation->domain() || !_space_belongs_to_module(*operation->domain())) {
                _error(operation, "structured operation requires a valid module-local IndexSpace");
            }
            if (operation->kind() == OperationKind::PIPELINE) {
                if (operation->region_count() == 0u) { _error(operation, "pipeline requires at least one stage region"); }
            } else if (operation->region_count() != 1u) {
                _error(operation, "parallel, serial, and reduce require exactly one body region");
            }
            if (operation->domain()) {
                for (auto &&region : operation->regions()) {
                    if (region->block_count() == 0u) {
                        _error(operation, "structured region must contain at least one block");
                        continue;
                    }
                    for (auto &&block : region->blocks()) {
                        if (block->argument_count() < operation->domain()->rank()) {
                            _error(operation, "structured region block is missing index arguments");
                            continue;
                        }
                        for (auto i = 0u; i < operation->domain()->rank(); i++) {
                            if (block->argument(i)->type().kind() != TypeKind::INDEX) {
                                _error(operation, "structured region index arguments must have index type");
                            }
                        }
                    }
                }
            }
        }
        switch (operation->kind()) {
            case OperationKind::CONSTANT:
                if (operation->operand_count() != 0u || operation->result_count() == 0u) {
                    _error(operation, "constant requires no operands and at least one result");
                }
                break;
            case OperationKind::MMA: _verify_mma(operation); break;
            case OperationKind::MEMORY_ALLOC:
            case OperationKind::MEMORY_LOAD:
            case OperationKind::MEMORY_STORE: _verify_memory(operation); break;
            case OperationKind::YIELD: {
                if (operation->result_count() != 0u) { _error(operation, "yield cannot produce results"); }
                auto region = operation->parent_block()->parent_region();
                auto parent = region->parent_operation();
                if (parent == nullptr) {
                    _error(operation, "yield must terminate an operation-owned region");
                } else {
                    if (operation->operand_count() != parent->result_count()) {
                        _error(operation, "yield operand count must match parent operation results");
                    } else {
                        for (auto i = 0u; i < operation->operand_count(); i++) {
                            if (operation->operand(i) == nullptr || !(operation->operand(i)->type() == parent->result(i)->type())) {
                                _error(operation, "yield operand type must match its parent result");
                            }
                        }
                    }
                }
                auto &&operations = operation->parent_block()->operations();
                if (operations.empty() || operations.back().get() != operation) { _error(operation, "yield must be the last operation in its block"); }
                break;
            }
            default: break;
        }
        if (operation->execution_scope_constraint() && operation->kind() != OperationKind::PARALLEL) {
            _error(operation, "only parallel carries an execution-scope binding constraint");
        }
        if (operation->resource_class_constraint() && operation->kind() != OperationKind::MEMORY_ALLOC) {
            _error(operation, "only memory.alloc carries a resource-class constraint");
        }
    }

    void _verify_access_capability(const Operation *operation, luisa::optional<ExecutionScope> active_scope) noexcept {
        if (_target == nullptr || !active_scope ||
            (operation->kind() != OperationKind::MEMORY_LOAD && operation->kind() != OperationKind::MEMORY_STORE) ||
            operation->operand_count() == 0u || operation->operand(0u) == nullptr ||
            operation->operand(0u)->origin() != Value::Origin::OPERATION_RESULT) { return; }
        auto allocation = operation->operand(0u)->defining_operation();
        if (allocation->kind() != OperationKind::MEMORY_ALLOC || !allocation->resource_class_constraint()) { return; }
        auto resource = _target->find_resource_class(*allocation->resource_class_constraint());
        if (!resource) {
            _error(operation, luisa::format("unknown target resource class '{}'", *allocation->resource_class_constraint()));
            return;
        }
        auto kind = operation->kind() == OperationKind::MEMORY_LOAD ? MemoryAccessKind::LOAD : MemoryAccessKind::STORE;
        if (!_target->can_access(*active_scope, *resource, kind)) {
            _error(operation, luisa::format("execution scope '{}' cannot {} resource class '{}'",
                                            _target->name(*active_scope),
                                            kind == MemoryAccessKind::LOAD ? "load from" : "store to",
                                            _target->name(*resource)));
        }
    }

    void _verify_operation(const Operation *operation, const Block *expected_parent, luisa::optional<ExecutionScope> active_scope) noexcept {
        if (operation->parent_block() != expected_parent) { _error(operation, "operation parent pointer is inconsistent"); }
        if (!_operation_ids.emplace(operation->id()).second) { _error(operation, "operation id is not unique within the function"); }
        for (auto i = 0u; i < operation->operand_count(); i++) {
            auto use = operation->operand_use(i);
            if (use == nullptr || use->user() != operation || use->index() != i) {
                _error(operation, "operand Use has inconsistent owner or index");
                continue;
            }
            if (!_uses.emplace(use).second) { _error(operation, "operand Use object appears more than once"); }
            auto value = use->value();
            if (value == nullptr) {
                _error(operation, "operand must not be null");
            } else if (!_definition_precedes_use(value, operation)) {
                _error(operation, "operand definition does not lexically dominate this use");
            }
        }
        for (auto i = 0u; i < operation->result_count(); i++) {
            auto value = operation->result(i);
            if (value == nullptr || value->origin() != Value::Origin::OPERATION_RESULT ||
                value->defining_operation() != operation || value->index() != i) {
                _error(operation, "operation result has inconsistent definition metadata");
                continue;
            }
            if (!_type_belongs_to_module(value->type())) { _error(operation, "operation result has invalid or foreign type"); }
            if (!_values.emplace(value).second || !_value_ids.emplace(value->id()).second) {
                _error(operation, "result value id is not unique within the function");
            }
        }
        for (auto &&attribute : operation->attributes()) {
            if (attribute.name.empty() || !attribute.value.is_valid()) { _error(operation, "attributes require non-empty names and valid values"); }
        }
        _verify_core_operation(operation);

        auto nested_scope = active_scope;
        if (_target != nullptr && operation->execution_scope_constraint()) {
            auto scope = _target->find_execution_scope(*operation->execution_scope_constraint());
            if (!scope) {
                _error(operation, luisa::format("unknown target execution scope '{}'", *operation->execution_scope_constraint()));
            } else {
                if (active_scope && !_target->contains(*active_scope, *scope)) {
                    _error(operation, luisa::format("execution scope '{}' does not contain nested scope '{}'",
                                                    _target->name(*active_scope), _target->name(*scope)));
                }
                nested_scope = scope;
            }
        }
        _verify_access_capability(operation, active_scope);
        for (auto &&region : operation->regions()) { _verify_region(region.get(), operation, nested_scope); }
    }

    void _verify_block(const Block *block, const Region *expected_parent, luisa::optional<ExecutionScope> active_scope) noexcept {
        if (block->parent_region() != expected_parent) { _error(nullptr, "block parent pointer is inconsistent"); }
        for (auto i = 0u; i < block->argument_count(); i++) {
            auto value = block->argument(i);
            if (value == nullptr || value->origin() != Value::Origin::BLOCK_ARGUMENT ||
                value->argument_block() != block || value->index() != i) {
                _error(nullptr, "block argument has inconsistent definition metadata");
                continue;
            }
            if (!_type_belongs_to_module(value->type())) { _error(nullptr, "block argument has invalid or foreign type"); }
            if (!_values.emplace(value).second || !_value_ids.emplace(value->id()).second) {
                _error(nullptr, "block argument value id is not unique within the function");
            }
        }
        for (auto &&operation : block->operations()) { _verify_operation(operation.get(), block, active_scope); }
    }

    void _verify_region(const Region *region, const Operation *expected_parent, luisa::optional<ExecutionScope> active_scope) noexcept {
        if (region->parent_function() != _function || region->parent_operation() != expected_parent) {
            _error(expected_parent, "region parent pointer is inconsistent");
        }
        for (auto &&block : region->blocks()) { _verify_block(block.get(), region, active_scope); }
    }

    void _verify_use_lists() noexcept {
        for (auto value : _values) {
            luisa::unordered_set<const Use *> listed;
            for (auto use : value->uses()) {
                if (use == nullptr || use->value() != value || !_uses.contains(use) || !listed.emplace(use).second) {
                    _error(use == nullptr ? nullptr : use->user(), "Value use-list is inconsistent with operation operands");
                }
            }
            size_t expected = 0u;
            for (auto use : _uses) {
                if (use->value() == value) { expected++; }
            }
            if (listed.size() != expected) { _error(nullptr, "Value use-list omits one or more operands"); }
        }
        for (auto use : _uses) {
            if (use->value() != nullptr && !_values.contains(use->value())) {
                _error(use->user(), "operand references a Value outside this function");
            }
        }
    }

    void _verify_function(const Function *function) noexcept {
        _function = function;
        _values.clear();
        _uses.clear();
        _value_ids.clear();
        _operation_ids.clear();
        if (function->parent_module() != &_module) { _error(nullptr, "function parent pointer is inconsistent"); }
        if (function->name().empty()) { _error(nullptr, "function name must not be empty"); }
        _verify_region(&function->body(), nullptr, luisa::nullopt);
        _verify_use_lists();
    }

public:
    Verifier(const Module &module, const TargetModel *target) noexcept
        : _module{module}, _target{target} {}

    [[nodiscard]] VerificationResult run() noexcept {
        luisa::unordered_set<uint64_t> function_ids;
        luisa::unordered_set<luisa::string_view> function_names;
        for (auto &&function : _module.functions()) {
            if (!function_ids.emplace(function->id()).second) {
                _function = function.get();
                _error(nullptr, "function id is not unique within the module");
            }
            if (!function_names.emplace(function->name()).second) {
                _function = function.get();
                _error(nullptr, "function name is not unique within the module");
            }
            _verify_function(function.get());
        }
        return std::move(_result);
    }
};

}// namespace detail

VerificationResult verify(const Module &module, const TargetModel *target) noexcept {
    return detail::Verifier{module, target}.run();
}

}// namespace luisa::compute::tile
