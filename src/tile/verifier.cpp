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
        auto i = 0u;
        for (auto candidate : operations) {
            if (candidate == operation) { return i; }
            i++;
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
        luisa::unordered_set<const Value *> visited;
        while (state != nullptr && state->type().kind() == TypeKind::MEMORY_STATE && visited.emplace(state).second) {
            const Operation *operation = nullptr;
            if (state->origin() == Value::Origin::OPERATION_RESULT) {
                operation = state->defining_operation();
            } else {
                auto block = state->argument_block();
                if (block == nullptr || block->parent_region() == nullptr) { return nullptr; }
                operation = block->parent_region()->parent_operation();
            }
            if (operation == nullptr) { return nullptr; }
            if (operation->kind() == OperationKind::MEMORY_ALLOC && state->index() == 1u && operation->result_count() >= 2u) {
                return operation->result(0u);
            }
            if (operation->kind() == OperationKind::MEMORY_STORE && state->index() == 0u && operation->operand_count() >= 2u) {
                return operation->operand(0u);
            }
            if ((operation->kind() != OperationKind::SERIAL && operation->kind() != OperationKind::PIPELINE &&
                 operation->kind() != OperationKind::REDUCE) ||
                !operation->domain()) { return nullptr; }
            auto index = state->index();
            if (state->origin() == Value::Origin::BLOCK_ARGUMENT) {
                if (index < operation->domain()->rank()) { return nullptr; }
                index -= operation->domain()->rank();
            }
            if (index >= operation->operand_count()) { return nullptr; }
            state = operation->operand(index);
        }
        return nullptr;
    }

    struct MemoryState {
        const Value *value;
        bool initialized;
    };
    using MemoryStates = luisa::unordered_map<const Value *, MemoryState>;

    // Check reaching definitions, not just token types. A state cannot fork
    // into two writes to one identity, nor can an old state authorize a read
    // after that identity has been overwritten. Loaded Tile SSA snapshots are
    // independent values and remain valid across those writes.
    [[nodiscard]] MemoryStates _verify_memory_flow(const Block *block, MemoryStates states) noexcept {
        for (auto operation : block->operations()) {
            if (operation->kind() == OperationKind::MEMORY_ALLOC) {
                states.emplace(operation->result(0u), MemoryState{operation->result(1u), false});
            } else if (operation->kind() == OperationKind::MEMORY_LOAD || operation->kind() == OperationKind::MEMORY_STORE) {
                auto memory = operation->operand(0u);
                auto state = states.find(memory);
                if (state == states.end() || state->second.value != operation->operand(1u)) {
                    _error(operation, "Memory access must consume the reaching MemoryState, not a stale or unrelated state");
                } else if (operation->kind() == OperationKind::MEMORY_LOAD) {
                    if (!state->second.initialized) { _error(operation, "Memory load requires a definite preceding store"); }
                } else {
                    state->second = MemoryState{operation->result(0u), true};
                }
            } else if (operation->kind() == OperationKind::YIELD) {
                for (auto i = 0u; i < operation->operand_count(); i++) {
                    auto value = operation->operand(i);
                    if (value->type().kind() != TypeKind::MEMORY_STATE) { continue; }
                    auto owner = _memory_owner_of_state(value);
                    auto state = states.find(owner);
                    if (state == states.end() || state->second.value != value) {
                        _error(operation, "MemoryState yield must forward the reaching state of its Memory");
                    }
                }
            }

            auto kind = operation->kind();
            if (kind != OperationKind::PARALLEL && kind != OperationKind::SERIAL && kind != OperationKind::PIPELINE &&
                kind != OperationKind::REDUCE && kind != OperationKind::TILE_MAP) { continue; }
            for (auto &&region : operation->regions()) {
                for (auto body : region->blocks()) {
                    auto incoming = states;
                    luisa::unordered_map<const Value *, const Value *> results;
                    for (auto i = 0u; i < operation->operand_count(); i++) {
                        auto value = operation->operand(i);
                        if (value->type().kind() != TypeKind::MEMORY_STATE) { continue; }
                        auto owner = _memory_owner_of_state(value);
                        auto state = incoming.find(owner);
                        if (state == incoming.end() || state->second.value != value || results.contains(owner)) {
                            _error(operation, "structured MemoryState input must be the unique reaching state of its Memory");
                            continue;
                        }
                        state->second.value = body->argument(operation->domain()->rank() + i);
                        results.emplace(owner, operation->result(i));
                    }
                    auto outgoing = _verify_memory_flow(body, incoming);
                    auto nonempty = operation->domain()->static_volume().value_or(0u) != 0u;
                    for (auto &[owner, state] : states) {
                        auto final = outgoing.at(owner);
                        if (auto result = results.find(owner); result != results.end()) {
                            auto index = result->second->index();
                            auto terminator = body->operations().empty() ? nullptr : body->operations().back();
                            if (terminator == nullptr || terminator->kind() != OperationKind::YIELD ||
                                terminator->operand(index) != final.value) {
                                _error(operation, "structured MemoryState result must yield the final state of the same Memory");
                            }
                            state = MemoryState{result->second, state.initialized || (nonempty && final.initialized)};
                        } else if (incoming.at(owner).value != final.value) {
                            _error(operation, "writes to ancestor Memory require temporal MemoryState carries; parallel whole-Memory writes are not independent");
                        }
                    }
                }
            }
        }
        return states;
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
                } else if (auto &&layout = operation->memory_layout()) {
                    if (!layout->verify() || layout->domain() != *operation->result(0u)->type().index_space() ||
                        !_space_belongs_to_module(layout->codomain())) {
                        _error(operation, "Memory layout must map its logical element space to a valid module-local storage space");
                    } else {
                        auto proof = layout->prove();
                        if (proof.is_storage_invalid()) {
                            _error(operation, "Memory layout must be total, in bounds, and injective for whole-Tile load/store");
                        }
                    }
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

    void _verify_view(const Operation *operation) noexcept {
        auto is_index = [](const Value *value) noexcept {
            return value != nullptr &&
                   (value->type().kind() == TypeKind::INDEX ||
                    (value->type().kind() == TypeKind::SCALAR &&
                     (value->type().scalar_type() == ScalarType::INT32 ||
                      value->type().scalar_type() == ScalarType::UINT32 ||
                      value->type().scalar_type() == ScalarType::INT64 ||
                      value->type().scalar_type() == ScalarType::UINT64)));
        };
        if (operation->operand_count() == 0u || operation->operand(0u) == nullptr ||
            !operation->operand(0u)->type().is_view()) {
            _error(operation, "view access requires a View as its first operand");
            return;
        }
        auto &&view_type = operation->operand(0u)->type();
        auto rank = view_type.index_space()->rank();
        if (operation->domain()) {
            auto &&space = *operation->domain();
            if (!_space_belongs_to_module(space) || space.rank() != rank) {
                _error(operation, "subtile access requires a module-local domain matching the View rank");
                return;
            }
            auto is_load = operation->kind() == OperationKind::VIEW_LOAD;
            auto expected_count = rank + (is_load ? 1u : 2u);
            auto has_fallback = is_load && operation->operand_count() == expected_count + 1u;
            if (operation->operand_count() != expected_count && !has_fallback) {
                _error(operation, "subtile access origin count must match the View rank");
                return;
            }
            for (auto i = 0u; i < rank; i++) {
                if (!is_index(operation->operand(i + 1u))) {
                    _error(operation, "subtile origins must be integer scalar values");
                }
            }
            auto tile_type = Type::tile(view_type.scalar_type(), space);
            if (is_load) {
                if (operation->result_count() != 1u || operation->result(0u)->type() != tile_type) {
                    _error(operation, "subtile load must produce a Tile with the requested domain and element type");
                }
                if (has_fallback && (operation->operand(rank + 1u) == nullptr ||
                                     operation->operand(rank + 1u)->type() != Type::scalar(view_type.scalar_type()))) {
                    _error(operation, "subtile load fallback must match the View element type");
                }
            } else if (operation->result_count() != 0u || operation->operand(rank + 1u) == nullptr ||
                       operation->operand(rank + 1u)->type() != tile_type) {
                _error(operation, "subtile store requires a Tile matching the requested domain and element type");
            }
            return;
        }
        auto ordinary_operands = rank + (operation->kind() == OperationKind::VIEW_LOAD ? 1u : 2u);
        auto masked_load = operation->kind() == OperationKind::VIEW_LOAD &&
                           operation->operand_count() == rank + 3u;
        if (operation->operand_count() != ordinary_operands && !masked_load) {
            _error(operation, "view access index count must match the View rank");
            return;
        }
        for (auto i = 0u; i < rank; i++) {
            if (!is_index(operation->operand(i + 1u))) {
                _error(operation, "view indices must have index or integer scalar type");
            }
        }
        if (operation->kind() == OperationKind::VIEW_LOAD) {
            if (operation->result_count() != 1u ||
                !(operation->result(0u)->type() == Type::scalar(view_type.scalar_type()))) {
                _error(operation, "view.load must produce one scalar matching the View element type");
            }
            if (masked_load) {
                auto predicate = operation->operand(rank + 1u);
                auto fallback = operation->operand(rank + 2u);
                if (predicate == nullptr || predicate->type() != Type::scalar(ScalarType::BOOL) ||
                    fallback == nullptr || fallback->type() != Type::scalar(view_type.scalar_type())) {
                    _error(operation, "masked view.load requires a bool predicate and matching scalar fallback");
                }
            }
        } else {
            auto value = operation->operand(rank + 1u);
            if (operation->result_count() != 0u || value == nullptr ||
                !(value->type() == Type::scalar(view_type.scalar_type()))) {
                _error(operation, "view.store requires one matching scalar value and no result");
            }
        }
    }

    [[nodiscard]] static bool _is_element_value(const Type &type) noexcept {
        return type.kind() == TypeKind::INDEX || type.kind() == TypeKind::SCALAR || type.kind() == TypeKind::TILE;
    }

    [[nodiscard]] static bool _same_element_shape(const Type &lhs, const Type &rhs) noexcept {
        auto lhs_scalar = lhs.kind() == TypeKind::INDEX || lhs.kind() == TypeKind::SCALAR;
        auto rhs_scalar = rhs.kind() == TypeKind::INDEX || rhs.kind() == TypeKind::SCALAR;
        if (lhs_scalar || rhs_scalar) { return lhs_scalar && rhs_scalar; }
        if (lhs.kind() != rhs.kind()) { return false; }
        return lhs.kind() == TypeKind::TILE && lhs.index_space() != nullptr && rhs.index_space() != nullptr &&
               *lhs.index_space() == *rhs.index_space();
    }

    [[nodiscard]] static ScalarType _element_scalar_type(const Type &type) noexcept {
        return type.kind() == TypeKind::INDEX ? ScalarType::INT64 : type.scalar_type();
    }

    [[nodiscard]] static bool _same_element_type(const Type &lhs, const Type &rhs) noexcept {
        return _same_element_shape(lhs, rhs) && _element_scalar_type(lhs) == _element_scalar_type(rhs);
    }

    [[nodiscard]] static bool _broadcasts_to(const Type &source, const Type &destination) noexcept {
        if (!source.is_tile()) { return _is_element_value(source) && _is_element_value(destination); }
        if (!destination.is_tile()) { return false; }
        for (auto &&axis : source.index_space()->axes()) {
            auto index = destination.index_space()->axis_index(axis.dimension);
            if (!index || (destination.index_space()->axis(*index).extent != axis.extent &&
                           (!axis.extent.is_constant() || axis.extent.constant_value() != 1u))) { return false; }
        }
        return true;
    }

    [[nodiscard]] static bool _is_floating(ScalarType type) noexcept {
        return type == ScalarType::FLOAT8_E4M3 || type == ScalarType::FLOAT8_E5M2 ||
               type == ScalarType::BFLOAT16 || type == ScalarType::FLOAT16 ||
               type == ScalarType::FLOAT32 || type == ScalarType::FLOAT64;
    }

    [[nodiscard]] static bool _is_integer(ScalarType type) noexcept {
        return type == ScalarType::INT8 || type == ScalarType::UINT8 ||
               type == ScalarType::INT16 || type == ScalarType::UINT16 ||
               type == ScalarType::INT32 || type == ScalarType::UINT32 ||
               type == ScalarType::INT64 || type == ScalarType::UINT64;
    }

    void _verify_elementwise(const Operation *operation) noexcept {
        auto op = operation->elementwise_op();
        auto unary = op == ElementwiseOp::NEG || op == ElementwiseOp::CAST ||
                     op == ElementwiseOp::LOGICAL_NOT ||
                     op == ElementwiseOp::EXP || op == ElementwiseOp::LOG ||
                     op == ElementwiseOp::SQRT || op == ElementwiseOp::TANH ||
                     op == ElementwiseOp::ABS;
        auto binary = op == ElementwiseOp::ADD || op == ElementwiseOp::SUB ||
                      op == ElementwiseOp::MUL || op == ElementwiseOp::DIV ||
                      op == ElementwiseOp::MOD || op == ElementwiseOp::MIN ||
                      op == ElementwiseOp::MAX || op == ElementwiseOp::EQ ||
                      op == ElementwiseOp::NE || op == ElementwiseOp::LT ||
                      op == ElementwiseOp::LE || op == ElementwiseOp::GT ||
                      op == ElementwiseOp::GE || op == ElementwiseOp::LOGICAL_AND ||
                      op == ElementwiseOp::LOGICAL_OR;
        auto arity = unary ? 1u : binary                  ? 2u :
                              op == ElementwiseOp::SELECT ? 3u :
                                                            0u;
        if (op == ElementwiseOp::INVALID || operation->operand_count() != arity ||
            operation->result_count() != 1u || operation->region_count() != 0u || operation->domain()) {
            _error(operation, "elementwise operation has an invalid opcode, arity, result count, or region");
            return;
        }
        auto &&result = operation->result(0u)->type();
        if (!_is_element_value(result)) {
            _error(operation, "elementwise result must be a scalar or Tile value");
            return;
        }
        for (auto i = 0u; i < operation->operand_count(); i++) {
            if (operation->operand(i) == nullptr || !_is_element_value(operation->operand(i)->type())) {
                _error(operation, "elementwise operands must be scalar or Tile values");
                return;
            }
        }
        if (op == ElementwiseOp::CAST) {
            if (!_same_element_shape(operation->operand(0u)->type(), result)) {
                _error(operation, "elementwise cast must preserve scalar-versus-Tile shape");
            }
            return;
        }
        if (op == ElementwiseOp::SELECT) {
            auto &&condition = operation->operand(0u)->type();
            if (condition.scalar_type() != ScalarType::BOOL ||
                !_broadcasts_to(condition, result) ||
                !_broadcasts_to(operation->operand(1u)->type(), result) ||
                !_broadcasts_to(operation->operand(2u)->type(), result) ||
                _element_scalar_type(operation->operand(1u)->type()) != _element_scalar_type(result) ||
                _element_scalar_type(operation->operand(2u)->type()) != _element_scalar_type(result)) {
                _error(operation, "elementwise select requires a shape-matched bool condition and matching values");
            }
            return;
        }
        auto logical = op == ElementwiseOp::LOGICAL_AND ||
                       op == ElementwiseOp::LOGICAL_OR ||
                       op == ElementwiseOp::LOGICAL_NOT;
        if (logical) {
            if (result.scalar_type() != ScalarType::BOOL) {
                _error(operation, "logical elementwise operation requires bool operands and result");
                return;
            }
            for (auto i = 0u; i < operation->operand_count(); i++) {
                if (!_broadcasts_to(operation->operand(i)->type(), result) ||
                    operation->operand(i)->type().scalar_type() != ScalarType::BOOL) {
                    _error(operation, "logical elementwise operation requires bool operands and result");
                    return;
                }
            }
            return;
        }
        auto comparison = op == ElementwiseOp::EQ || op == ElementwiseOp::NE ||
                          op == ElementwiseOp::LT || op == ElementwiseOp::LE ||
                          op == ElementwiseOp::GT || op == ElementwiseOp::GE;
        if (comparison) {
            auto &&lhs = operation->operand(0u)->type();
            if (_element_scalar_type(lhs) != _element_scalar_type(operation->operand(1u)->type()) ||
                result.scalar_type() != ScalarType::BOOL || !_broadcasts_to(lhs, result) ||
                !_broadcasts_to(operation->operand(1u)->type(), result)) {
                _error(operation, "elementwise comparison requires matching inputs and a shape-matched bool result");
            }
            return;
        }
        for (auto i = 0u; i < operation->operand_count(); i++) {
            if (!_broadcasts_to(operation->operand(i)->type(), result) ||
                _element_scalar_type(operation->operand(i)->type()) != _element_scalar_type(result)) {
                _error(operation, "elementwise arithmetic requires operands and result to have identical types");
                return;
            }
        }
        if ((op == ElementwiseOp::EXP || op == ElementwiseOp::LOG ||
             op == ElementwiseOp::SQRT || op == ElementwiseOp::TANH) &&
            !_is_floating(result.scalar_type())) {
            _error(operation, "transcendental elementwise operation requires a floating-point element type");
        }
        if (op == ElementwiseOp::MOD && !_is_integer(result.scalar_type())) {
            _error(operation, "elementwise modulo requires an integer element type");
        }
        if (result.scalar_type() == ScalarType::BOOL) {
            _error(operation, "elementwise arithmetic does not accept bool elements");
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
            if (operation->region_count() != 1u) {
                _error(operation, "structured operation requires exactly one body region");
            }
            if (operation->kind() == OperationKind::PARALLEL &&
                (operation->operand_count() != 0u || operation->result_count() != 0u)) {
                _error(operation, "parallel cannot carry loop state; use memory effects for observable results");
            }
            if (operation->kind() != OperationKind::PARALLEL &&
                operation->operand_count() != operation->result_count()) {
                _error(operation, "serial, pipeline, and reduce require one initial operand per result");
            }
            if (operation->domain()) {
                for (auto &&region : operation->regions()) {
                    if (region->block_count() == 0u) {
                        _error(operation, "structured region must contain at least one block");
                        continue;
                    }
                    for (auto block : region->blocks()) {
                        auto expected_arguments = operation->domain()->rank() + operation->result_count();
                        if (block->argument_count() != expected_arguments) {
                            _error(operation, "structured region block arguments must be indices followed by carried state");
                            continue;
                        }
                        for (auto i = 0u; i < operation->domain()->rank(); i++) {
                            if (block->argument(i)->type().kind() != TypeKind::INDEX) {
                                _error(operation, "structured region index arguments must have index type");
                            }
                        }
                        for (auto i = 0u; i < operation->result_count(); i++) {
                            auto argument = block->argument(operation->domain()->rank() + i);
                            if (!(argument->type() == operation->result(i)->type()) ||
                                operation->operand(i) == nullptr ||
                                !(operation->operand(i)->type() == operation->result(i)->type())) {
                                _error(operation, "structured carried-state types must agree");
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
            case OperationKind::ELEMENTWISE: _verify_elementwise(operation); break;
            case OperationKind::TILE_EXTRACT: {
                if (operation->operand_count() == 0u || operation->operand(0u) == nullptr ||
                    !operation->operand(0u)->type().is_tile()) {
                    _error(operation, "tile.extract requires a Tile operand");
                    break;
                }
                auto &&tile_type = operation->operand(0u)->type();
                if (operation->operand_count() != tile_type.index_space()->rank() + 1u ||
                    operation->result_count() != 1u ||
                    operation->result(0u)->type() != Type::scalar(tile_type.scalar_type())) {
                    _error(operation, "tile.extract index count and scalar result must match the Tile");
                    break;
                }
                for (auto i = 1u; i < operation->operand_count(); i++) {
                    auto index = operation->operand(i);
                    if (index == nullptr || (index->type().kind() != TypeKind::INDEX &&
                                             (index->type().kind() != TypeKind::SCALAR || !_is_integer(index->type().scalar_type())))) {
                        _error(operation, "tile.extract indices must be integer scalar values");
                    }
                }
                break;
            }
            case OperationKind::TILE_MAP: {
                if (operation->operand_count() != 0u || operation->result_count() != 1u ||
                    !operation->result(0u)->type().is_tile() || !operation->domain() ||
                    !_space_belongs_to_module(*operation->domain()) ||
                    *operation->result(0u)->type().index_space() != *operation->domain() ||
                    operation->region_count() != 1u || operation->region(0u)->block_count() != 1u) {
                    _error(operation, "tile.map requires a Tile result, matching domain, and one element region");
                    break;
                }
                auto block = operation->region(0u)->block(0u);
                if (block->argument_count() != operation->domain()->rank()) {
                    _error(operation, "tile.map block arguments must match its logical coordinate rank");
                }
                for (auto &&argument : block->arguments()) {
                    if (argument->type().kind() != TypeKind::INDEX) {
                        _error(operation, "tile.map coordinates must have index type");
                    }
                }
                if (block->operations().empty() || block->operations().back()->kind() != OperationKind::YIELD) {
                    _error(operation, "tile.map must end with an element yield");
                }
                break;
            }
            case OperationKind::MMA: _verify_mma(operation); break;
            case OperationKind::VIEW_LOAD:
            case OperationKind::VIEW_STORE: _verify_view(operation); break;
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
                            auto expected_type = parent->kind() == OperationKind::TILE_MAP ?
                                                     Type::scalar(parent->result(i)->type().scalar_type()) :
                                                     parent->result(i)->type();
                            if (operation->operand(i) == nullptr || !(operation->operand(i)->type() == expected_type)) {
                                _error(operation, "yield operand type must match its parent result");
                            }
                        }
                    }
                }
                auto &&operations = operation->parent_block()->operations();
                if (operations.empty() || operations.back() != operation) { _error(operation, "yield must be the last operation in its block"); }
                break;
            }
            case OperationKind::STAGE: {
                if (operation->operand_count() != 0u || operation->result_count() != 0u ||
                    operation->region_count() != 0u) {
                    _error(operation, "stage marker cannot have operands, results, or regions");
                }
                auto parent = operation->parent_block()->parent_region()->parent_operation();
                if (parent == nullptr || parent->kind() != OperationKind::PIPELINE) {
                    _error(operation, "stage marker must appear directly in a pipeline body");
                }
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
        if (operation->memory_layout() && operation->kind() != OperationKind::MEMORY_ALLOC) {
            _error(operation, "only memory.alloc carries an allocation-local Memory layout");
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
        if (operation->memory_effect() != MemoryEffect::NONE) {
            auto region = expected_parent->parent_region();
            while (auto parent = region->parent_operation()) {
                if (parent->kind() == OperationKind::TILE_MAP) {
                    _error(operation, "tile.map is pure; addressable memory effects must remain in the enclosing execution nest");
                    break;
                }
                region = parent->parent_block()->parent_region();
            }
        }
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
            } else if (!value->use_list().contains(use)) {
                _error(operation, "linked operand Use is absent from its defining Value use-list");
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
        for (auto operation : block->operations()) { _verify_operation(operation, block, active_scope); }
    }

    void _verify_region(const Region *region, const Operation *expected_parent, luisa::optional<ExecutionScope> active_scope) noexcept {
        if (region->parent_function() != _function || region->parent_operation() != expected_parent) {
            _error(expected_parent, "region parent pointer is inconsistent");
        }
        for (auto block : region->blocks()) { _verify_block(block, region, active_scope); }
    }

    void _verify_use_lists() noexcept {
        for (auto value : _values) {
            luisa::unordered_set<const Use *> listed;
            for (auto use : value->use_list()) {
                if (use == nullptr || !value->use_list().contains(use) ||
                    use->value() != value || !_uses.contains(use) || !listed.emplace(use).second) {
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
        auto diagnostic_count = _result.diagnostics().size();
        _function = function;
        _values.clear();
        _uses.clear();
        _value_ids.clear();
        _operation_ids.clear();
        if (function->parent_module() != &_module) { _error(nullptr, "function parent pointer is inconsistent"); }
        if (function->name().empty()) { _error(nullptr, "function name must not be empty"); }
        _verify_region(&function->body(), nullptr, luisa::nullopt);
        _verify_use_lists();
        // The flow walk assumes the structural/type invariants checked above.
        if (_result.diagnostics().size() == diagnostic_count) {
            for (auto block : function->body().blocks()) { static_cast<void>(_verify_memory_flow(block, {})); }
        }
    }

public:
    Verifier(const Module &module, const TargetModel *target) noexcept
        : _module{module}, _target{target} {}

    [[nodiscard]] VerificationResult run() noexcept {
        luisa::unordered_set<uint64_t> function_ids;
        luisa::unordered_set<luisa::string_view> function_names;
        for (auto function : _module.functions()) {
            if (!function_ids.emplace(function->id()).second) {
                _function = function;
                _error(nullptr, "function id is not unique within the module");
            }
            if (!function_names.emplace(function->name()).second) {
                _function = function;
                _error(nullptr, "function name is not unique within the module");
            }
            _verify_function(function);
        }
        return std::move(_result);
    }
};

}// namespace detail

VerificationResult verify(const Module &module, const TargetModel *target) noexcept {
    return detail::Verifier{module, target}.run();
}

}// namespace luisa::compute::tile
