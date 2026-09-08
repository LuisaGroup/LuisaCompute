#include "coro_discriminated_prefix.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <utility>

#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/assume.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/store.h>

#include "coro_guarded_scalar_relation.h"
#include "coro_scalar_relation_liveness.h"
#include "coro_semantic_graph.h"
#include "helpers.h"

namespace luisa::compute::xir::detail {

namespace {

struct InstructionLocation {
    size_t block;
    size_t ordinal;
};

using InstructionLocationMap =
    luisa::unordered_map<Instruction *, InstructionLocation>;

[[nodiscard]] InstructionLocationMap make_instruction_locations(
    FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept {
    InstructionLocationMap result;
    for (auto *block : definition->basic_blocks()) {
        auto ordinal = size_t{0u};
        for (auto *instruction : block->instructions()) {
            result.emplace(
                instruction,
                InstructionLocation{graph.block_id(block), ordinal++});
        }
    }
    return result;
}

[[nodiscard]] bool instruction_precedes(
    Instruction *before, Instruction *after,
    const InstructionLocationMap &locations) noexcept {
    if (before == nullptr || after == nullptr) { return false; }
    auto before_iter = locations.find(before);
    auto after_iter = locations.find(after);
    return before_iter != locations.end() &&
           after_iter != locations.end() &&
           before_iter->second.block == after_iter->second.block &&
           before_iter->second.ordinal < after_iter->second.ordinal;
}

struct ArrayUseRegion {
    bool valid{true};
    luisa::unordered_set<Value *> pointers;
    luisa::unordered_set<Instruction *> users;
    luisa::vector<BasicBlock *> blocks;
};

[[nodiscard]] ArrayUseRegion collect_array_use_region(
    AllocaInst *array, FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept {
    ArrayUseRegion result;
    luisa::unordered_set<BasicBlock *> seen_blocks;
    luisa::vector<Value *> worklist{array};
    while (!worklist.empty() && result.valid) {
        auto *pointer = worklist.back();
        worklist.pop_back();
        if (!result.pointers.emplace(pointer).second) { continue; }
        for (auto *use : pointer->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                result.valid = false;
                break;
            }
            auto *instruction = static_cast<Instruction *>(user);
            auto *block = instruction->parent_block();
            if (block == nullptr ||
                instruction->parent_function() != definition ||
                !graph.contains(block) || instruction->isa<PhiInst>()) {
                result.valid = false;
                break;
            }
            result.users.emplace(instruction);
            if (seen_blocks.emplace(block).second) {
                result.blocks.emplace_back(block);
            }
            if (instruction->isa<GEPInst>() &&
                static_cast<GEPInst *>(instruction)->base() == pointer) {
                worklist.emplace_back(instruction);
            }
        }
    }
    return result;
}

struct ActiveSlice {
    bool valid{false};
    size_t target{0u};
    luisa::vector<uint8_t> active;
    luisa::vector<size_t> blocks;
};

[[nodiscard]] ActiveSlice make_active_slice(
    BasicBlock *target, const ArrayUseRegion &region,
    const CoroSemanticGraph &graph) noexcept {
    ActiveSlice result;
    if (target == nullptr || !region.valid) { return result; }
    result.target = graph.block_id(target);
    if (result.target >= graph.block_count()) { return result; }
    result.active.assign(graph.block_count(), 0u);
    luisa::vector<size_t> worklist;
    for (auto *block : region.blocks) {
        if (!graph.dominates(target, block)) { return result; }
        auto id = graph.block_id(block);
        if (id >= graph.block_count()) { return result; }
        if (result.active[id] == 0u) {
            result.active[id] = 1u;
            worklist.emplace_back(id);
        }
    }
    if (result.active[result.target] == 0u) {
        result.active[result.target] = 1u;
        worklist.emplace_back(result.target);
    }
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto id = worklist[cursor];
        if (id == result.target) { continue; }
        for (auto predecessor : graph.predecessors(id)) {
            if (!graph.dominates(
                    target, graph.block(predecessor))) {
                return result;
            }
            if (result.active[predecessor] == 0u) {
                result.active[predecessor] = 1u;
                worklist.emplace_back(predecessor);
            }
        }
    }
    std::sort(worklist.begin(), worklist.end());
    result.blocks = std::move(worklist);
    result.valid = true;
    return result;
}

struct ScalarSlotInfo {
    bool valid{false};
    StoreInst *single_store{nullptr};
    luisa::unordered_map<LoadInst *, StoreInst *> local_reaching_stores;
};

struct ScalarSnapshotFlow {
    bool built{false};
    luisa::vector<uint8_t> block_inputs;
    luisa::unordered_map<Instruction *, bool> query_results;
};

class ScalarResolver {
private:
    const InstructionLocationMap &_locations;
    const CoroSemanticGraph &_graph;
    mutable luisa::unordered_map<AllocaInst *, ScalarSlotInfo> _slots;
    mutable luisa::unordered_map<LoadInst *, ScalarSnapshotFlow>
        _snapshot_flows;

private:
    [[nodiscard]] bool _dominates(
        Instruction *definition, Instruction *use) const noexcept {
        if (definition == nullptr || use == nullptr) { return false; }
        return definition->parent_block() == use->parent_block() ?
                   instruction_precedes(definition, use, _locations) :
                   _graph.dominates(
                       definition->parent_block(), use->parent_block());
    }

    [[nodiscard]] static bool _depends_on_slot(
        Value *value, AllocaInst *slot) noexcept {
        luisa::unordered_set<Value *> visited;
        luisa::vector<Value *> worklist{value};
        while (!worklist.empty()) {
            auto *current = worklist.back();
            worklist.pop_back();
            if (current == nullptr || !visited.emplace(current).second) {
                continue;
            }
            if (current->isa<LoadInst>() &&
                static_cast<LoadInst *>(current)->variable() == slot) {
                return true;
            }
            if (!current->isa<Instruction>()) { continue; }
            auto *instruction = static_cast<Instruction *>(current);
            for (size_t i = 0u;
                 i < instruction->operand_count(); ++i) {
                worklist.emplace_back(instruction->operand(i));
            }
        }
        return false;
    }

    [[nodiscard]] const ScalarSlotInfo &_slot_info(
        AllocaInst *slot) const noexcept {
        if (auto iter = _slots.find(slot); iter != _slots.end()) {
            return iter->second;
        }
        ScalarSlotInfo info;
        if (slot == nullptr || !slot->is_local() ||
            slot->type() == nullptr || !slot->type()->is_scalar()) {
            return _slots.emplace(slot, std::move(info)).first->second;
        }
        luisa::vector<LoadInst *> loads;
        luisa::vector<StoreInst *> stores;
        for (auto *use : slot->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                return _slots.emplace(slot, std::move(info)).first->second;
            }
            auto *instruction = static_cast<Instruction *>(user);
            if (instruction->isa<LoadInst>() &&
                static_cast<LoadInst *>(instruction)->variable() == slot) {
                loads.emplace_back(static_cast<LoadInst *>(instruction));
            } else if (instruction->isa<StoreInst>() &&
                       static_cast<StoreInst *>(instruction)->variable() ==
                           slot) {
                stores.emplace_back(static_cast<StoreInst *>(instruction));
            } else {
                return _slots.emplace(slot, std::move(info)).first->second;
            }
        }
        if (stores.empty()) {
            return _slots.emplace(slot, std::move(info)).first->second;
        }
        if (stores.size() == 1u) {
            info.valid = true;
            info.single_store = stores.front();
            return _slots.emplace(slot, std::move(info)).first->second;
        }
        for (auto *load : loads) {
            auto load_location = _locations.find(load);
            if (load_location == _locations.end()) { continue; }
            StoreInst *reaching = nullptr;
            auto ordinal = size_t{0u};
            for (auto *store : stores) {
                auto store_location = _locations.find(store);
                if (store_location == _locations.end() ||
                    store_location->second.block !=
                        load_location->second.block ||
                    store_location->second.ordinal >=
                        load_location->second.ordinal) {
                    continue;
                }
                if (reaching == nullptr ||
                    ordinal < store_location->second.ordinal) {
                    reaching = store;
                    ordinal = store_location->second.ordinal;
                }
            }
            if (reaching != nullptr &&
                !_depends_on_slot(reaching->value(), slot)) {
                info.local_reaching_stores.emplace(load, reaching);
            }
        }
        info.valid = true;
        return _slots.emplace(slot, std::move(info)).first->second;
    }

    [[nodiscard]] StoreInst *_reaching_store(
        AllocaInst *slot, LoadInst *load) const noexcept {
        auto &&info = _slot_info(slot);
        if (!info.valid) { return nullptr; }
        if (info.single_store != nullptr) { return info.single_store; }
        if (auto iter = info.local_reaching_stores.find(load);
            iter != info.local_reaching_stores.end()) {
            return iter->second;
        }
        return nullptr;
    }

public:
    ScalarResolver(const InstructionLocationMap &locations,
                   const CoroSemanticGraph &graph) noexcept
        : _locations{locations}, _graph{graph} {}

    [[nodiscard]] bool direct_scalar_slot(
        AllocaInst *slot) const noexcept {
        if (slot == nullptr || !slot->is_local() ||
            slot->type() == nullptr || !slot->type()->is_scalar()) {
            return false;
        }
        for (auto *use : slot->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                return false;
            }
            auto *instruction = static_cast<Instruction *>(user);
            if ((instruction->isa<LoadInst>() &&
                 static_cast<LoadInst *>(instruction)->variable() == slot) ||
                (instruction->isa<StoreInst>() &&
                 static_cast<StoreInst *>(instruction)->variable() == slot)) {
                continue;
            }
            return false;
        }
        return true;
    }

    [[nodiscard]] Value *resolve(
        Value *value, Instruction *use,
        size_t depth = 0u) const noexcept {
        if (value == nullptr || use == nullptr || depth >= 32u ||
            !value->isa<LoadInst>()) {
            return value;
        }
        auto *load = static_cast<LoadInst *>(value);
        auto *variable = load->variable();
        if (variable == nullptr || !variable->isa<AllocaInst>()) {
            return value;
        }
        auto *slot = static_cast<AllocaInst *>(variable);
        auto *store = _reaching_store(slot, load);
        if (store == nullptr || !_dominates(store, load) ||
            !_dominates(load, use)) {
            return value;
        }
        return resolve(store->value(), store, depth + 1u);
    }

    [[nodiscard]] bool snapshot_reaches_unchanged(
        LoadInst *load, AllocaInst *slot,
        Instruction *use) const noexcept;
};

bool ScalarResolver::snapshot_reaches_unchanged(
    LoadInst *load, AllocaInst *slot,
    Instruction *use) const noexcept {
    auto &flow = _snapshot_flows[load];
    if (auto iter = flow.query_results.find(use);
        iter != flow.query_results.end()) {
        return iter->second;
    }
    const auto finish = [&](bool value) noexcept {
        flow.query_results.emplace(use, value);
        return value;
    };
    if (load == nullptr || slot == nullptr || use == nullptr ||
        !_dominates(load, use) || !direct_scalar_slot(slot)) {
        return finish(false);
    }
    auto load_location = _locations.find(load);
    auto use_location = _locations.find(use);
    if (load_location == _locations.end() ||
        use_location == _locations.end()) {
        return finish(false);
    }
    constexpr auto clean = uint8_t{1u};
    constexpr auto dirty = uint8_t{2u};
    const auto transfer = [load, slot](
                              Instruction *instruction,
                              uint8_t state) noexcept {
        if (instruction == load) { return clean; }
        if (instruction->isa<StoreInst>() &&
            static_cast<StoreInst *>(instruction)->variable() == slot) {
            // State zero denotes a block execution not reached from a dynamic
            // execution of L. A preceding store must not manufacture a dirty
            // path before L executes.
            return state == 0u ? uint8_t{0u} : dirty;
        }
        return state;
    };
    if (!flow.built) {
        // Exact two-state quotient of the path search. Clean means that the
        // latest dynamic execution of L reaches this point without a store to
        // S; Dirty means at least one such store occurred. Transfer is
        // deterministic and join is bitwise union, so one finite fixed point
        // answers all uses of the same load.
        flow.block_inputs.assign(_graph.block_count(), 0u);
        luisa::vector<size_t> worklist{load_location->second.block};
        luisa::vector<uint8_t> queued(_graph.block_count(), 0u);
        queued[load_location->second.block] = 1u;
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            auto block_id = worklist[cursor];
            queued[block_id] = 0u;
            auto state = flow.block_inputs[block_id];
            for (auto *instruction :
                 _graph.block(block_id)->instructions()) {
                state = transfer(instruction, state);
            }
            if (state == 0u) { continue; }
            for (auto successor : _graph.successors(block_id)) {
                auto next = static_cast<uint8_t>(
                    flow.block_inputs[successor] | state);
                if (next != flow.block_inputs[successor]) {
                    flow.block_inputs[successor] = next;
                    if (queued[successor] == 0u) {
                        queued[successor] = 1u;
                        worklist.emplace_back(successor);
                    }
                }
            }
        }
        flow.built = true;
    }
    auto state = flow.block_inputs[use_location->second.block];
    for (auto *instruction : use->parent_block()->instructions()) {
        if (instruction == use) { break; }
        state = transfer(instruction, state);
    }
    return finish(state == clean);
}

[[nodiscard]] luisa::optional<uint64_t> decode_unsigned(
    Value *value) noexcept {
    uint64_t result = 0u;
    if (value == nullptr || !value->isa<Constant>() ||
        !try_decode_constant_nonnegative_integer(value, result)) {
        return luisa::nullopt;
    }
    return result;
}

[[nodiscard]] uint64_t unsigned_type_max(const Type *type) noexcept {
    if (type == nullptr) { return 0u; }
    switch (type->tag()) {
        case Type::Tag::UINT8:
            return std::numeric_limits<uint8_t>::max();
        case Type::Tag::UINT16:
            return std::numeric_limits<uint16_t>::max();
        case Type::Tag::UINT32:
            return std::numeric_limits<uint32_t>::max();
        case Type::Tag::UINT64:
            return std::numeric_limits<uint64_t>::max();
        default: return 0u;
    }
}

[[nodiscard]] bool has_store_between(
    AllocaInst *slot, Instruction *before, Instruction *after,
    const InstructionLocationMap &locations) noexcept {
    if (slot == nullptr || before == nullptr || after == nullptr ||
        before->parent_block() != after->parent_block()) {
        return true;
    }
    auto before_location = locations.find(before);
    auto after_location = locations.find(after);
    if (before_location == locations.end() ||
        after_location == locations.end() ||
        before_location->second.ordinal >=
            after_location->second.ordinal) {
        return true;
    }
    for (auto *use : slot->use_list()) {
        auto *user = use == nullptr ? nullptr : use->user();
        if (user == nullptr || !user->isa<StoreInst>()) { continue; }
        auto *store = static_cast<StoreInst *>(user);
        if (store->variable() != slot ||
            store->parent_block() != before->parent_block()) {
            continue;
        }
        auto location = locations.find(store);
        if (location != locations.end() &&
            before_location->second.ordinal < location->second.ordinal &&
            location->second.ordinal < after_location->second.ordinal) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool is_current_snapshot(
    Value *value, AllocaInst *slot, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    const auto matches = [&](Value *candidate) noexcept {
        if (candidate == nullptr || !candidate->isa<LoadInst>()) {
            return false;
        }
        auto *load = static_cast<LoadInst *>(candidate);
        return load->variable() == slot &&
               ((load->parent_block() == use->parent_block() &&
                 instruction_precedes(load, use, locations) &&
                 !has_store_between(slot, load, use, locations)) ||
                resolver.snapshot_reaches_unchanged(load, slot, use));
    };
    return matches(value) || matches(resolver.resolve(value, use));
}

[[nodiscard]] AllocaInst *current_scalar_slot(
    Value *value, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    const auto direct = [&](Value *candidate) noexcept {
        if (candidate == nullptr || !candidate->isa<LoadInst>()) {
            return static_cast<AllocaInst *>(nullptr);
        }
        auto *load = static_cast<LoadInst *>(candidate);
        auto *variable = load->variable();
        if (variable == nullptr || !variable->isa<AllocaInst>()) {
            return static_cast<AllocaInst *>(nullptr);
        }
        auto *slot = static_cast<AllocaInst *>(variable);
        return is_current_snapshot(
                   candidate, slot, use, resolver, locations) ?
                   slot :
                   nullptr;
    };
    if (auto *slot = direct(value)) { return slot; }
    return direct(resolver.resolve(value, use));
}

struct BooleanExpressionSource {
    Value *predicate{nullptr};
    bool true_when_predicate_is{true};
    luisa::optional<bool> constant;
};

// Returns the durable Boolean memory/SSA predicate denoted by `value` at
// `use`, together with its polarity. Exact single/reaching-store temporaries
// are resolved first. A genuinely mutable load remains unresolved and is
// therefore represented by its slot's current value; this distinguishes a
// value copy from a later reassignment without sacrificing copy correlation.
[[nodiscard]] BooleanExpressionSource analyze_boolean_expression(
    Value *value, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    auto true_when_base_is = true;
    for (auto depth = 0u; depth < 16u; ++depth) {
        auto *resolved = resolver.resolve(value, use);
        if (resolved != value) {
            value = resolved;
            continue;
        }
        if (auto *slot = current_scalar_slot(
                value, use, resolver, locations);
            slot != nullptr && slot->type() == Type::of<bool>()) {
            return BooleanExpressionSource{
                .predicate = slot,
                .true_when_predicate_is = true_when_base_is};
        }
        if (value != nullptr && value->isa<Constant>() &&
            value->type() == Type::of<bool>()) {
            auto base_value = static_cast<Constant *>(value)->as<bool>();
            return BooleanExpressionSource{
                .true_when_predicate_is = true_when_base_is,
                .constant = base_value == true_when_base_is};
        }
        if (value == nullptr || !value->isa<ArithmeticInst>()) {
            if (value != nullptr && value->type() == Type::of<bool>()) {
                return BooleanExpressionSource{
                    .predicate = value,
                    .true_when_predicate_is = true_when_base_is};
            }
            break;
        }
        auto *arithmetic = static_cast<ArithmeticInst *>(value);
        if (arithmetic->op() == ArithmeticOp::UNARY_BIT_NOT &&
            arithmetic->operand_count() == 1u &&
            arithmetic->operand(0u)->type() == Type::of<bool>()) {
            value = arithmetic->operand(0u);
            true_when_base_is = !true_when_base_is;
            continue;
        }
        if ((arithmetic->op() == ArithmeticOp::BINARY_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_NOT_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_BIT_XOR) &&
            arithmetic->operand_count() == 2u) {
            Value *variable = nullptr;
            luisa::optional<bool> constant;
            for (auto constant_operand = 0u;
                 constant_operand < 2u; ++constant_operand) {
                auto *operand = arithmetic->operand(constant_operand);
                if (operand != nullptr && operand->isa<Constant>() &&
                    operand->type() == Type::of<bool>()) {
                    variable = arithmetic->operand(1u - constant_operand);
                    constant = static_cast<Constant *>(operand)->as<bool>();
                    break;
                }
            }
            if (variable != nullptr && constant &&
                variable->type() == Type::of<bool>()) {
                auto invert = arithmetic->op() ==
                                      ArithmeticOp::BINARY_EQUAL ?
                                  !*constant :
                                  *constant;
                value = variable;
                true_when_base_is ^= invert;
                continue;
            }
        }
        return BooleanExpressionSource{
            .predicate = value,
            .true_when_predicate_is = true_when_base_is};
    }
    return {};
}

// Collect the durable Boolean leaves of an expression after resolving exact
// scalar temporaries. Treating an AND/OR tree as one unrelated SSA predicate
// loses the very correlation that the tree denotes; conversely, decomposing
// scalar comparisons would invent semantics not represented by this domain.
// Therefore Boolean connectives are traversed structurally and every other
// Boolean-producing value is an atomic predicate.
[[nodiscard]] luisa::vector<Value *> boolean_expression_atoms(
    Value *value, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    luisa::vector<Value *> result;
    luisa::unordered_set<Value *> visited;
    const auto visit = [&](auto &&self, Value *current,
                           size_t depth) noexcept -> void {
        if (current == nullptr || depth >= 32u) { return; }
        current = resolver.resolve(current, use);
        if (current == nullptr || !visited.emplace(current).second ||
            current->type() != Type::of<bool>() ||
            current->isa<Constant>()) {
            return;
        }
        if (current->isa<ArithmeticInst>()) {
            auto *arithmetic = static_cast<ArithmeticInst *>(current);
            if (arithmetic->op() == ArithmeticOp::UNARY_BIT_NOT &&
                arithmetic->operand_count() == 1u &&
                arithmetic->operand(0u)->type() == Type::of<bool>()) {
                self(self, arithmetic->operand(0u), depth + 1u);
                return;
            }
            auto is_boolean_binary = arithmetic->operand_count() == 2u &&
                                     arithmetic->operand(0u)->type() ==
                                         Type::of<bool>() &&
                                     arithmetic->operand(1u)->type() ==
                                         Type::of<bool>();
            if (is_boolean_binary &&
                (arithmetic->op() == ArithmeticOp::BINARY_BIT_AND ||
                 arithmetic->op() == ArithmeticOp::BINARY_BIT_OR ||
                 arithmetic->op() == ArithmeticOp::BINARY_BIT_XOR ||
                 arithmetic->op() == ArithmeticOp::BINARY_EQUAL ||
                 arithmetic->op() == ArithmeticOp::BINARY_NOT_EQUAL)) {
                self(self, arithmetic->operand(0u), depth + 1u);
                self(self, arithmetic->operand(1u), depth + 1u);
                return;
            }
        }
        if (auto *slot = current_scalar_slot(
                current, use, resolver, locations);
            slot != nullptr && slot->type() == Type::of<bool>()) {
            current = slot;
        }
        result.emplace_back(current);
    };
    visit(visit, value, 0u);
    return result;
}

[[nodiscard]] bool same_snapshot(
    Value *lhs, Value *rhs, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    if (lhs == rhs) { return true; }
    auto *resolved_lhs = resolver.resolve(lhs, use);
    auto *resolved_rhs = resolver.resolve(rhs, use);
    if (resolved_lhs == resolved_rhs) { return true; }
    auto *lhs_slot = current_scalar_slot(
        lhs, use, resolver, locations);
    auto *rhs_slot = current_scalar_slot(
        rhs, use, resolver, locations);
    return lhs_slot != nullptr && lhs_slot == rhs_slot;
}

[[nodiscard]] Value *strip_boolean_wrappers(
    Value *condition, bool &truth, Instruction *use,
    const ScalarResolver &resolver) noexcept {
    for (auto depth = 0u; depth < 16u; ++depth) {
        condition = resolver.resolve(condition, use);
        if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
            break;
        }
        auto *arithmetic = static_cast<ArithmeticInst *>(condition);
        if (arithmetic->op() == ArithmeticOp::UNARY_BIT_NOT &&
            arithmetic->operand_count() == 1u &&
            arithmetic->operand(0u)->type() == Type::of<bool>()) {
            condition = arithmetic->operand(0u);
            truth = !truth;
            continue;
        }
        if ((arithmetic->op() == ArithmeticOp::BINARY_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_NOT_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_BIT_XOR) &&
            arithmetic->operand_count() == 2u) {
            Value *variable = nullptr;
            luisa::optional<bool> constant;
            for (auto constant_operand = 0u;
                 constant_operand < 2u; ++constant_operand) {
                auto *operand = arithmetic->operand(constant_operand);
                if (operand != nullptr && operand->isa<Constant>() &&
                    operand->type() == Type::of<bool>()) {
                    variable = arithmetic->operand(1u - constant_operand);
                    constant = static_cast<Constant *>(operand)->as<bool>();
                    break;
                }
            }
            if (variable != nullptr && constant &&
                variable->type() == Type::of<bool>()) {
                auto invert = arithmetic->op() ==
                                      ArithmeticOp::BINARY_EQUAL ?
                                  !*constant :
                                  *constant;
                condition = variable;
                truth ^= invert;
                continue;
            }
        }
        break;
    }
    return condition;
}

struct MaskedScalarTest {
    AllocaInst *scalar{nullptr};
    uint64_t mask{0u};
    bool nonzero_when_condition_true{true};
};

// Recognizes the exact unsigned predicate `(S & M) != 0`, including its
// equality and Boolean-negation duals. The scalar operand must denote the
// current value of a direct local; unsupported arithmetic yields no fact.
[[nodiscard]] luisa::optional<MaskedScalarTest> match_masked_scalar_test(
    Value *condition, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    auto underlying_truth_when_condition_true = true;
    condition = strip_boolean_wrappers(
        condition, underlying_truth_when_condition_true, use, resolver);
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return luisa::nullopt;
    }
    auto *comparison = static_cast<ArithmeticInst *>(condition);
    if ((comparison->op() != ArithmeticOp::BINARY_EQUAL &&
         comparison->op() != ArithmeticOp::BINARY_NOT_EQUAL) ||
        comparison->operand_count() != 2u ||
        comparison->type() != Type::of<bool>()) {
        return luisa::nullopt;
    }
    Value *masked_value = nullptr;
    for (auto zero_operand = 0u; zero_operand < 2u; ++zero_operand) {
        auto zero = decode_unsigned(resolver.resolve(
            comparison->operand(zero_operand), use));
        if (zero && *zero == 0u) {
            masked_value = comparison->operand(1u - zero_operand);
            break;
        }
    }
    masked_value = resolver.resolve(masked_value, use);
    if (masked_value == nullptr ||
        !masked_value->isa<ArithmeticInst>()) {
        return luisa::nullopt;
    }
    auto *bit_and = static_cast<ArithmeticInst *>(masked_value);
    if (bit_and->op() != ArithmeticOp::BINARY_BIT_AND ||
        bit_and->operand_count() != 2u || bit_and->type() == nullptr ||
        !bit_and->type()->is_uint()) {
        return luisa::nullopt;
    }
    for (auto mask_operand = 0u; mask_operand < 2u; ++mask_operand) {
        auto mask = decode_unsigned(resolver.resolve(
            bit_and->operand(mask_operand), use));
        if (!mask || *mask == 0u) { continue; }
        auto *scalar = current_scalar_slot(
            bit_and->operand(1u - mask_operand), use,
            resolver, locations);
        if (scalar != nullptr && scalar->type() == bit_and->type()) {
            auto comparison_true_means_nonzero =
                comparison->op() == ArithmeticOp::BINARY_NOT_EQUAL;
            return MaskedScalarTest{
                .scalar = scalar,
                .mask = *mask,
                .nonzero_when_condition_true =
                    underlying_truth_when_condition_true ==
                    comparison_true_means_nonzero};
        }
    }
    return luisa::nullopt;
}

struct MaskedScalarWitnessSet {
    luisa::vector<CoroMaskedScalarWitness> roots;
    luisa::vector<CoroMaskedScalarWitness> tracked;
};

[[nodiscard]] MaskedScalarWitnessSet collect_masked_scalar_witnesses(
    const CoroSemanticGraph &graph,
    luisa::span<const uint8_t> active_blocks,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    MaskedScalarWitnessSet result;
    const auto insert = [&](AllocaInst *scalar, uint64_t mask) noexcept {
        if (scalar == nullptr || mask == 0u ||
            !resolver.direct_scalar_slot(scalar) ||
            std::any_of(
                result.tracked.begin(), result.tracked.end(),
                [&](auto witness) noexcept {
                    return witness.scalar == scalar &&
                           witness.mask == mask;
                })) {
            return false;
        }
        result.tracked.emplace_back(CoroMaskedScalarWitness{scalar, mask});
        return true;
    };
    const auto insert_root = [&](MaskedScalarTest test) noexcept {
        if (std::none_of(
                result.roots.begin(), result.roots.end(),
                [&](auto witness) noexcept {
                    return witness.scalar == test.scalar &&
                           witness.mask == test.mask;
                })) {
            result.roots.emplace_back(CoroMaskedScalarWitness{
                test.scalar, test.mask});
        }
        static_cast<void>(insert(test.scalar, test.mask));
    };

    // The roots are exactly the masked predicates observable in this
    // candidate's backward CFG slice. Tracking any other scalar projection
    // cannot affect a proof obligation and would only enlarge the product.
    for (size_t block_id = 0u; block_id < graph.block_count(); ++block_id) {
        if (block_id >= active_blocks.size() ||
            active_blocks[block_id] == 0u) {
            continue;
        }
        for (auto *instruction : graph.block(block_id)->instructions()) {
            Value *condition = nullptr;
            if (instruction->isa<ConditionalBranchInst>()) {
                condition = static_cast<ConditionalBranchInst *>(instruction)
                                ->condition();
            } else if (instruction->isa<AssumeInst>()) {
                condition = static_cast<AssumeInst *>(instruction)
                                ->condition();
            }
            if (condition != nullptr) {
                if (auto test = match_masked_scalar_test(
                        condition, instruction, resolver, locations)) {
                    insert_root(*test);
                }
            }
        }
    }

    // Close backwards over exact scalar expression dependencies. This is the
    // least fixed point needed to transfer a witnessed mask through lowering
    // temporaries; encountering a scalar load stops at that memory boundary,
    // whose reaching stores are processed when the new witness is visited.
    for (size_t cursor = 0u; cursor < result.tracked.size(); ++cursor) {
        auto witness = result.tracked[cursor];
        for (auto *use : witness.scalar->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(user);
            if (store->variable() != witness.scalar) { continue; }
            auto location = locations.find(store);
            if (location == locations.end() ||
                location->second.block >= active_blocks.size() ||
                active_blocks[location->second.block] == 0u) {
                continue;
            }
            luisa::unordered_set<Value *> visited;
            luisa::vector<Value *> worklist{store->value()};
            while (!worklist.empty()) {
                auto *value = worklist.back();
                worklist.pop_back();
                if (value == nullptr || !visited.emplace(value).second) {
                    continue;
                }
                if (auto *source = current_scalar_slot(
                        value, store, resolver, locations);
                    source != nullptr &&
                    source->type() == witness.scalar->type()) {
                    static_cast<void>(insert(source, witness.mask));
                    continue;
                }
                value = resolver.resolve(value, store);
                if (value == nullptr || !value->isa<Instruction>()) {
                    continue;
                }
                auto *instruction = static_cast<Instruction *>(value);
                for (size_t operand = 0u;
                     operand < instruction->operand_count(); ++operand) {
                    worklist.emplace_back(instruction->operand(operand));
                }
            }
        }
    }
    return result;
}

// Sparse forward Must analysis for the masked scalar projections that are
// actually observed by control flow. A fact Z(S,M) denotes that every
// concrete value reaching the program point has all bits in M clear. The
// lattice is a finite set of facts, transfer only removes or proves facts,
// and CFG join is intersection, so the worklist computes the unique maximal
// sound Must solution independently of traversal order.
class MaskedScalarKnownZeroAnalysis {
private:
    struct Projection {
        AllocaInst *scalar;
        uint64_t mask;
    };
    using State = luisa::vector<uint8_t>;

    const CoroSemanticGraph &_graph;
    const ScalarResolver &_resolver;
    const InstructionLocationMap &_locations;
    luisa::vector<Projection> _projections;
    luisa::unordered_map<AllocaInst *, luisa::vector<size_t>>
        _projections_by_scalar;
    luisa::vector<luisa::optional<State>> _in_states;
    luisa::vector<luisa::optional<State>> _out_states;
    luisa::optional<State> _query_state;

private:
    [[nodiscard]] bool _add_projection(
        AllocaInst *scalar, uint64_t mask) noexcept {
        if (scalar == nullptr || mask == 0u ||
            scalar->type() == nullptr || !scalar->type()->is_uint() ||
            !_resolver.direct_scalar_slot(scalar)) {
            return false;
        }
        auto &indices = _projections_by_scalar[scalar];
        if (std::any_of(
                indices.begin(), indices.end(),
                [&](size_t index) noexcept {
                    return _projections[index].mask == mask;
                })) {
            return false;
        }
        indices.emplace_back(_projections.size());
        _projections.emplace_back(Projection{scalar, mask});
        return true;
    }

    void _collect_dependencies(
        Value *value, Instruction *use, uint64_t mask,
        size_t depth = 0u) noexcept {
        if (value == nullptr || use == nullptr || depth >= 32u) { return; }
        if (auto *source = current_scalar_slot(
                value, use, _resolver, _locations);
            source != nullptr && source->type() != nullptr &&
            source->type()->is_uint()) {
            static_cast<void>(_add_projection(source, mask));
            return;
        }
        value = _resolver.resolve(value, use);
        if (value == nullptr || decode_unsigned(value) ||
            !value->isa<ArithmeticInst>()) {
            return;
        }
        auto *arithmetic = static_cast<ArithmeticInst *>(value);
        switch (arithmetic->op()) {
            case ArithmeticOp::BINARY_BIT_AND:
            case ArithmeticOp::BINARY_BIT_OR:
            case ArithmeticOp::BINARY_BIT_XOR:
                if (arithmetic->operand_count() == 2u) {
                    _collect_dependencies(
                        arithmetic->operand(0u), use, mask, depth + 1u);
                    _collect_dependencies(
                        arithmetic->operand(1u), use, mask, depth + 1u);
                }
                break;
            case ArithmeticOp::SELECT:
                if (arithmetic->operand_count() == 3u) {
                    _collect_dependencies(
                        arithmetic->operand(0u), use, mask, depth + 1u);
                    _collect_dependencies(
                        arithmetic->operand(1u), use, mask, depth + 1u);
                }
                break;
            default: break;
        }
    }

    void _close_dependencies() noexcept {
        for (size_t cursor = 0u; cursor < _projections.size(); ++cursor) {
            auto projection = _projections[cursor];
            for (auto *use : projection.scalar->use_list()) {
                auto *user = use == nullptr ? nullptr : use->user();
                if (user == nullptr || !user->isa<StoreInst>()) { continue; }
                auto *store = static_cast<StoreInst *>(user);
                if (store->variable() != projection.scalar ||
                    store->parent_block() == nullptr ||
                    !_graph.contains(store->parent_block())) {
                    continue;
                }
                _collect_dependencies(
                    store->value(), store, projection.mask);
            }
        }
    }

    [[nodiscard]] bool _state_proves_zero(
        const State &state, AllocaInst *scalar,
        uint64_t mask) const noexcept {
        if (scalar == nullptr || mask == 0u) { return false; }
        auto iter = _projections_by_scalar.find(scalar);
        if (iter == _projections_by_scalar.end()) { return false; }
        return std::any_of(
            iter->second.begin(), iter->second.end(),
            [&](size_t index) noexcept {
                auto projection = _projections[index];
                return state[index] != 0u &&
                       (mask & ~projection.mask) == 0u;
            });
    }

    [[nodiscard]] bool _value_is_zero(
        Value *value, uint64_t mask, const State &state,
        Instruction *use, size_t depth = 0u) const noexcept {
        if (value == nullptr || use == nullptr || depth >= 32u) {
            return false;
        }
        if (auto *source = current_scalar_slot(
                value, use, _resolver, _locations);
            _state_proves_zero(state, source, mask)) {
            return true;
        }
        value = _resolver.resolve(value, use);
        if (auto constant = decode_unsigned(value)) {
            return (*constant & mask) == 0u;
        }
        if (value == nullptr || !value->isa<ArithmeticInst>()) {
            return false;
        }
        auto *arithmetic = static_cast<ArithmeticInst *>(value);
        const auto operand_is_zero = [&](size_t operand) noexcept {
            return _value_is_zero(
                arithmetic->operand(operand), mask, state,
                use, depth + 1u);
        };
        switch (arithmetic->op()) {
            case ArithmeticOp::BINARY_BIT_AND:
                return arithmetic->operand_count() == 2u &&
                       (operand_is_zero(0u) || operand_is_zero(1u));
            case ArithmeticOp::BINARY_BIT_OR:
            case ArithmeticOp::BINARY_BIT_XOR:
                return arithmetic->operand_count() == 2u &&
                       operand_is_zero(0u) && operand_is_zero(1u);
            case ArithmeticOp::SELECT:
                return arithmetic->operand_count() == 3u &&
                       operand_is_zero(0u) && operand_is_zero(1u);
            default: return false;
        }
    }

    void _assume_zero(
        State &state, AllocaInst *scalar,
        uint64_t zero_mask) const noexcept {
        auto iter = _projections_by_scalar.find(scalar);
        if (iter == _projections_by_scalar.end()) { return; }
        for (auto index : iter->second) {
            if ((_projections[index].mask & ~zero_mask) == 0u) {
                state[index] = 1u;
            }
        }
    }

    [[nodiscard]] bool _transfer_instruction(
        Instruction *instruction, State &state) const noexcept {
        if (instruction->isa<AssumeInst>()) {
            auto *assume = static_cast<AssumeInst *>(instruction);
            if (auto test = match_masked_scalar_test(
                    assume->condition(), assume,
                    _resolver, _locations)) {
                if (test->nonzero_when_condition_true) {
                    if (_state_proves_zero(
                            state, test->scalar, test->mask)) {
                        return false;
                    }
                } else {
                    _assume_zero(state, test->scalar, test->mask);
                }
            }
            return true;
        }
        if (!instruction->isa<StoreInst>()) { return true; }
        auto *store = static_cast<StoreInst *>(instruction);
        auto *pointer = store->variable();
        auto *destination =
            pointer != nullptr && pointer->isa<AllocaInst>() ?
                static_cast<AllocaInst *>(pointer) :
                nullptr;
        auto iter = _projections_by_scalar.find(destination);
        if (iter == _projections_by_scalar.end()) { return true; }
        luisa::vector<uint8_t> next;
        next.reserve(iter->second.size());
        for (auto index : iter->second) {
            next.emplace_back(static_cast<uint8_t>(_value_is_zero(
                store->value(), _projections[index].mask,
                state, store)));
        }
        for (size_t i = 0u; i < iter->second.size(); ++i) {
            state[iter->second[i]] = next[i];
        }
        return true;
    }

    [[nodiscard]] bool _refine_edge(
        BasicBlock *predecessor, BasicBlock *successor,
        State &state) const noexcept {
        auto *terminator = predecessor == nullptr ?
                               nullptr : predecessor->terminator();
        if (terminator == nullptr ||
            !terminator->isa<ConditionalBranchInst>()) {
            return true;
        }
        auto *branch = static_cast<ConditionalBranchInst *>(terminator);
        bool truth;
        if (branch->true_block() == successor &&
            branch->false_block() != successor) {
            truth = true;
        } else if (branch->false_block() == successor &&
                   branch->true_block() != successor) {
            truth = false;
        } else {
            return true;
        }
        auto test = match_masked_scalar_test(
            branch->condition(), branch, _resolver, _locations);
        if (!test) { return true; }
        auto selected_nonzero =
            truth == test->nonzero_when_condition_true;
        if (selected_nonzero) {
            return !_state_proves_zero(
                state, test->scalar, test->mask);
        }
        _assume_zero(state, test->scalar, test->mask);
        return true;
    }

    [[nodiscard]] static bool _merge_into(
        luisa::optional<State> &target,
        const State &incoming) noexcept {
        if (!target) {
            target = incoming;
            return true;
        }
        auto changed = false;
        for (size_t i = 0u; i < target->size(); ++i) {
            auto next = static_cast<uint8_t>(
                (*target)[i] & incoming[i]);
            changed |= next != (*target)[i];
            (*target)[i] = next;
        }
        return changed;
    }

    [[nodiscard]] luisa::optional<State> _state_before(
        Instruction *instruction) const noexcept {
        if (instruction == nullptr ||
            instruction->parent_block() == nullptr) {
            return luisa::nullopt;
        }
        auto location = _locations.find(instruction);
        if (location == _locations.end() ||
            location->second.block >= _in_states.size() ||
            !_in_states[location->second.block]) {
            return luisa::nullopt;
        }
        auto state = *_in_states[location->second.block];
        for (auto *current :
             instruction->parent_block()->instructions()) {
            if (current == instruction) { return state; }
            if (!_transfer_instruction(current, state)) {
                return luisa::nullopt;
            }
        }
        return luisa::nullopt;
    }

    void _solve(BasicBlock *target, Instruction *instruction) noexcept {
        _in_states.resize(_graph.block_count());
        _out_states.resize(_graph.block_count());
        if (_graph.block_count() == 0u) { return; }
        _in_states[0u] = State(_projections.size(), uint8_t{0u});
        luisa::vector<size_t> worklist{0u};
        luisa::vector<uint8_t> queued(_graph.block_count(), uint8_t{0u});
        queued[0u] = 1u;
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            auto block_id = worklist[cursor];
            queued[block_id] = 0u;
            if (!_in_states[block_id]) { continue; }
            auto state = *_in_states[block_id];
            auto reachable = true;
            for (auto *current : _graph.block(block_id)->instructions()) {
                if (!_transfer_instruction(current, state)) {
                    reachable = false;
                    break;
                }
            }
            if (!reachable) { continue; }
            if (_out_states[block_id] &&
                *_out_states[block_id] == state) {
                continue;
            }
            _out_states[block_id] = state;
            for (auto successor : _graph.successors(block_id)) {
                auto edge_state = state;
                if (!_refine_edge(
                        _graph.block(block_id),
                        _graph.block(successor), edge_state)) {
                    continue;
                }
                if (_merge_into(_in_states[successor], edge_state) &&
                    queued[successor] == 0u) {
                    queued[successor] = 1u;
                    worklist.emplace_back(successor);
                }
            }
        }
        if (instruction != nullptr &&
            instruction->parent_block() == target) {
            _query_state = _state_before(instruction);
        }
    }

public:
    MaskedScalarKnownZeroAnalysis(
        const CoroSemanticGraph &graph,
        const ScalarResolver &resolver,
        const InstructionLocationMap &locations,
        luisa::span<const CoroMaskedScalarWitness> roots,
        BasicBlock *target, Instruction *instruction) noexcept
        : _graph{graph}, _resolver{resolver}, _locations{locations} {
        for (auto root : roots) {
            static_cast<void>(_add_projection(root.scalar, root.mask));
        }
        _close_dependencies();
        _solve(target, instruction);
    }

    [[nodiscard]] bool proves_zero(
        AllocaInst *scalar, uint64_t mask) const noexcept {
        return _query_state &&
               _state_proves_zero(*_query_state, scalar, mask);
    }
};

[[nodiscard]] bool selected_boolean_edge_implies_operands(
    ArithmeticInst *arithmetic, bool truth) noexcept {
    if (arithmetic == nullptr || arithmetic->type() != Type::of<bool>() ||
        arithmetic->operand_count() != 2u) {
        return false;
    }
    // true(A && B) implies A and B; false(A || B) implies !A and !B.
    // The dual cases are disjunctions and cannot justify either operand.
    return (truth &&
            arithmetic->op() == ArithmeticOp::BINARY_BIT_AND) ||
           (!truth &&
            arithmetic->op() == ArithmeticOp::BINARY_BIT_OR);
}

[[nodiscard]] GEPInst *top_array_gep(
    Value *pointer, AllocaInst *array) noexcept {
    auto *current = pointer;
    GEPInst *top = nullptr;
    while (current != nullptr && current != array &&
           current->isa<GEPInst>()) {
        top = static_cast<GEPInst *>(current);
        current = top->base();
    }
    return current == array ? top : nullptr;
}

[[nodiscard]] GEPInst *full_element_gep(
    Value *pointer, AllocaInst *array) noexcept {
    auto *gep = top_array_gep(pointer, array);
    return gep != nullptr && pointer == gep && gep->base() == array &&
                   gep->index_count() == 1u &&
                   gep->type() == array->type()->element() ?
               gep :
               nullptr;
}

[[nodiscard]] bool unsigned_scalar_array_compatible(
    AllocaInst *candidate, AllocaInst *payload) noexcept {
    if (candidate == nullptr || candidate == payload ||
        candidate->type() == nullptr || payload->type() == nullptr ||
        candidate->type()->tag() != Type::Tag::ARRAY ||
        payload->type()->tag() != Type::Tag::ARRAY ||
        candidate->type()->dimension() != payload->type()->dimension()) {
        return false;
    }
    auto *element = candidate->type()->element();
    return element != nullptr && element->is_scalar() && element->is_uint();
}

}// namespace

namespace {

struct TagSet {
    luisa::vector<uint64_t> words;
    [[nodiscard]] bool operator==(const TagSet &) const noexcept = default;
};

class TagDomain {
private:
    luisa::vector<uint64_t> _constants;
    size_t _bit_count{1u};

private:
    [[nodiscard]] size_t _constant_bit(uint64_t value) const noexcept {
        auto iter = std::lower_bound(
            _constants.begin(), _constants.end(), value);
        return iter != _constants.end() && *iter == value ?
                   static_cast<size_t>(iter - _constants.begin()) :
                   _constants.size();
    }

public:
    explicit TagDomain(luisa::vector<uint64_t> constants) noexcept
        : _constants{std::move(constants)} {
        std::sort(_constants.begin(), _constants.end());
        _constants.erase(
            std::unique(_constants.begin(), _constants.end()),
            _constants.end());
        // The final bit denotes every value not explicitly named by the XIR.
        _bit_count = _constants.size() + 1u;
    }

    [[nodiscard]] TagSet empty() const noexcept {
        return TagSet{luisa::vector<uint64_t>(
            (_bit_count + 63u) / 64u, 0u)};
    }

    [[nodiscard]] TagSet all() const noexcept {
        auto result = empty();
        std::fill(result.words.begin(), result.words.end(), ~uint64_t{0u});
        auto excess = result.words.size() * 64u - _bit_count;
        if (excess != 0u) {
            result.words.back() >>= excess;
        }
        return result;
    }

    [[nodiscard]] TagSet singleton(uint64_t value) const noexcept {
        auto result = empty();
        auto bit = _constant_bit(value);
        result.words[bit / 64u] |= uint64_t{1u} << (bit % 64u);
        return result;
    }

    [[nodiscard]] TagSet unite(
        TagSet lhs, const TagSet &rhs) const noexcept {
        for (size_t i = 0u; i < lhs.words.size(); ++i) {
            lhs.words[i] |= rhs.words[i];
        }
        return lhs;
    }

    [[nodiscard]] TagSet intersect(
        TagSet lhs, const TagSet &rhs) const noexcept {
        for (size_t i = 0u; i < lhs.words.size(); ++i) {
            lhs.words[i] &= rhs.words[i];
        }
        return lhs;
    }

    [[nodiscard]] TagSet subtract(
        TagSet lhs, const TagSet &rhs) const noexcept {
        for (size_t i = 0u; i < lhs.words.size(); ++i) {
            lhs.words[i] &= ~rhs.words[i];
        }
        return lhs;
    }

    [[nodiscard]] bool disjoint(
        const TagSet &lhs, const TagSet &rhs) const noexcept {
        for (size_t i = 0u; i < lhs.words.size(); ++i) {
            if ((lhs.words[i] & rhs.words[i]) != 0u) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool is_empty(const TagSet &set) const noexcept {
        return std::all_of(
            set.words.begin(), set.words.end(),
            [](uint64_t word) noexcept { return word == 0u; });
    }

    [[nodiscard]] bool is_all(const TagSet &set) const noexcept {
        return set == all();
    }

    [[nodiscard]] size_t population(const TagSet &set) const noexcept {
        auto count = size_t{0u};
        for (auto word : set.words) {
            count += static_cast<size_t>(std::popcount(word));
        }
        return count;
    }

    [[nodiscard]] size_t bit_count() const noexcept { return _bit_count; }
};

// For every abstract tag t, stores the Boolean valuations on which an
// undefined record carrying t may exist. Thus union is the exact May join,
// edge refinement is set intersection, and Boolean assignment is relational
// image under the assignment. The vector representation is canonical because
// each tag owns exactly one reduced BDD set; fixed-point equality is therefore
// independent of predecessor arrival order.
class GuardedTagSet {
private:
    using Set = CoroBooleanSetManager::Set;
    CoroBooleanSetManager *_manager;
    luisa::vector<Set> _unsafe;

private:
    [[nodiscard]] static bool _contains(
        const TagSet &set, size_t bit) noexcept {
        return (set.words[bit / 64u] &
                (uint64_t{1u} << (bit % 64u))) != 0u;
    }

    [[nodiscard]] Set _support() const noexcept {
        auto result = CoroBooleanSetManager::empty_set();
        for (auto set : _unsafe) {
            result = _manager->unite(result, set);
        }
        return result;
    }

public:
    GuardedTagSet(
        CoroBooleanSetManager &manager,
        const TagDomain &tags) noexcept
        : _manager{&manager},
          _unsafe(tags.bit_count(),
                  CoroBooleanSetManager::empty_set()) {}

    [[nodiscard]] bool operator==(
        const GuardedTagSet &) const noexcept = default;

    [[nodiscard]] bool empty() const noexcept {
        return std::all_of(
            _unsafe.begin(), _unsafe.end(),
            [](Set set) noexcept {
                return CoroBooleanSetManager::is_empty(set);
            });
    }

    void clear() noexcept {
        std::fill(
            _unsafe.begin(), _unsafe.end(),
            CoroBooleanSetManager::empty_set());
    }

    void assign(const TagSet &tags, Set valuations) noexcept {
        for (size_t bit = 0u; bit < _unsafe.size(); ++bit) {
            _unsafe[bit] = _contains(tags, bit) ?
                               valuations :
                               CoroBooleanSetManager::empty_set();
        }
    }

    void unite(const GuardedTagSet &incoming) noexcept {
        LUISA_DEBUG_ASSERT(
            _unsafe.size() == incoming._unsafe.size(),
            "Mismatched guarded tag domains.");
        for (size_t bit = 0u; bit < _unsafe.size(); ++bit) {
            _unsafe[bit] = _manager->unite(
                _unsafe[bit], incoming._unsafe[bit]);
        }
    }

    // Retag every represented unsafe record while preserving exactly the
    // control valuations on which such a record may exist.
    void retag(const TagSet &stored_tags) noexcept {
        assign(stored_tags, _support());
    }

    void unite_retag(const TagSet &stored_tags) noexcept {
        auto support = _support();
        for (size_t bit = 0u; bit < _unsafe.size(); ++bit) {
            if (_contains(stored_tags, bit)) {
                _unsafe[bit] = _manager->unite(
                    _unsafe[bit], support);
            }
        }
    }

    // A write through an unclassified record identity can select any
    // represented unsafe record and can assign any tag, but it cannot create
    // an unsafe record on a valuation where none existed before.
    void widen_tags_to_all(const TagDomain &tags) noexcept {
        assign(tags.all(), _support());
    }

    void filter_tags(const TagSet &selected) noexcept {
        for (size_t bit = 0u; bit < _unsafe.size(); ++bit) {
            if (!_contains(selected, bit)) {
                _unsafe[bit] = CoroBooleanSetManager::empty_set();
            }
        }
    }

    void refine_boolean(Value *predicate, bool value) noexcept {
        auto selected = _manager->literal(predicate, value);
        for (auto &set : _unsafe) {
            set = _manager->intersect(set, selected);
        }
    }

    void forget_boolean(Value *predicate) noexcept {
        for (auto &set : _unsafe) {
            set = _manager->forget(set, predicate);
        }
    }

    void forget_booleans(
        luisa::span<Value *const> predicates) noexcept {
        for (auto *predicate : predicates) {
            forget_boolean(predicate);
        }
    }

    void assign_boolean(
        Value *destination, Value *source,
        bool true_when_source_is,
        luisa::optional<bool> constant) noexcept {
        for (auto &set : _unsafe) {
            set = _manager->assign(
                set, destination, source,
                true_when_source_is, constant);
        }
    }

    [[nodiscard]] TagSet possible_tags(
        const TagDomain &tags) const noexcept {
        auto result = tags.empty();
        for (size_t bit = 0u; bit < _unsafe.size(); ++bit) {
            if (!CoroBooleanSetManager::is_empty(_unsafe[bit])) {
                result.words[bit / 64u] |=
                    uint64_t{1u} << (bit % 64u);
            }
        }
        return result;
    }

    [[nodiscard]] bool may_contain_any(
        const TagSet &tags) const noexcept {
        for (size_t bit = 0u; bit < _unsafe.size(); ++bit) {
            if (_contains(tags, bit) &&
                !CoroBooleanSetManager::is_empty(_unsafe[bit])) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool may_contain_any_on(
        const TagSet &tags,
        CoroBooleanSetManager::Set valuations) const noexcept {
        for (size_t bit = 0u; bit < _unsafe.size(); ++bit) {
            if (_contains(tags, bit) &&
                !CoroBooleanSetManager::is_empty(
                    _manager->intersect(
                        _unsafe[bit], valuations))) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] luisa::string describe_valuation(
        size_t bit) const noexcept {
        return bit < _unsafe.size() ?
                   _manager->describe(_unsafe[bit]) :
                   luisa::string{"<out-of-domain>"};
    }

    [[nodiscard]] luisa::vector<Value *> valuation_support(
        size_t bit) const noexcept {
        return bit < _unsafe.size() ?
                   _manager->support(_unsafe[bit]) :
                   luisa::vector<Value *>{};
    }
};

// A path-sensitive May fact over the same canonical Boolean valuation
// universe as GuardedTagSet. `set` contains exactly the represented
// valuations on which the property may be unsafe. Union is CFG join and edge
// refinement is intersection, so an empty set is a sound Must proof.
class GuardedMaySet {
private:
    using Set = CoroBooleanSetManager::Set;
    CoroBooleanSetManager *_manager;
    Set _unsafe;

public:
    GuardedMaySet(
        CoroBooleanSetManager &manager, Set initial) noexcept
        : _manager{&manager}, _unsafe{initial} {}

    [[nodiscard]] bool operator==(
        const GuardedMaySet &) const noexcept = default;

    void clear() noexcept {
        _unsafe = CoroBooleanSetManager::empty_set();
    }

    void assign(Set unsafe) noexcept { _unsafe = unsafe; }

    void unite(const GuardedMaySet &incoming) noexcept {
        _unsafe = _manager->unite(_unsafe, incoming._unsafe);
    }

    void refine_boolean(Value *predicate, bool value) noexcept {
        _unsafe = _manager->intersect(
            _unsafe, _manager->literal(predicate, value));
    }

    void forget_boolean(Value *predicate) noexcept {
        _unsafe = _manager->forget(_unsafe, predicate);
    }

    void forget_booleans(
        luisa::span<Value *const> predicates) noexcept {
        for (auto *predicate : predicates) {
            forget_boolean(predicate);
        }
    }

    void assign_boolean(
        Value *destination, Value *source,
        bool true_when_source_is,
        luisa::optional<bool> constant) noexcept {
        _unsafe = _manager->assign(
            _unsafe, destination, source,
            true_when_source_is, constant);
    }

    [[nodiscard]] bool may_intersect(Set valuations) const noexcept {
        return !CoroBooleanSetManager::is_empty(
            _manager->intersect(_unsafe, valuations));
    }
};

struct EqualityTest {
    Value *value{nullptr};
    uint64_t constant{0u};
    bool equal_when_condition_true{true};
};

[[nodiscard]] luisa::optional<EqualityTest> match_equality_test(
    Value *condition, Instruction *use,
    const ScalarResolver &resolver) noexcept {
    condition = resolver.resolve(condition, use);
    auto inverted = false;
    while (condition != nullptr && condition->isa<ArithmeticInst>()) {
        auto *arithmetic = static_cast<ArithmeticInst *>(condition);
        if (arithmetic->op() == ArithmeticOp::UNARY_BIT_NOT &&
            arithmetic->operand_count() == 1u &&
            arithmetic->operand(0u)->type() == Type::of<bool>()) {
            condition = resolver.resolve(arithmetic->operand(0u), use);
            inverted = !inverted;
            continue;
        }
        break;
    }
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return luisa::nullopt;
    }
    auto *comparison = static_cast<ArithmeticInst *>(condition);
    if ((comparison->op() != ArithmeticOp::BINARY_EQUAL &&
         comparison->op() != ArithmeticOp::BINARY_NOT_EQUAL) ||
        comparison->operand_count() != 2u) {
        return luisa::nullopt;
    }
    for (auto constant_operand = 0u;
         constant_operand < 2u; ++constant_operand) {
        auto constant = decode_unsigned(resolver.resolve(
            comparison->operand(constant_operand), comparison));
        if (!constant) { continue; }
        return EqualityTest{
            .value = comparison->operand(1u - constant_operand),
            .constant = *constant,
            .equal_when_condition_true =
                (comparison->op() == ArithmeticOp::BINARY_EQUAL) !=
                inverted};
    }
    return luisa::nullopt;
}

struct TagLoadAccess {
    LoadInst *load;
    Value *index;
    AllocaInst *index_slot;
};

[[nodiscard]] luisa::optional<TagLoadAccess> tag_load_access(
    Value *value, AllocaInst *tag, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    value = resolver.resolve(value, use);
    if (value == nullptr || !value->isa<LoadInst>()) {
        return luisa::nullopt;
    }
    auto *load = static_cast<LoadInst *>(value);
    auto *gep = full_element_gep(load->variable(), tag);
    if (gep == nullptr) { return luisa::nullopt; }
    return TagLoadAccess{
        .load = load,
        .index = gep->index(0u),
        .index_slot = current_scalar_slot(
            gep->index(0u), use, resolver, locations)};
}

[[nodiscard]] AllocaInst *tag_load_index_slot(
    Value *value, AllocaInst *tag, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    if (auto access = tag_load_access(
            value, tag, use, resolver, locations)) {
        return access->index_slot;
    }
    return nullptr;
}

[[nodiscard]] bool match_counter_increment(
    Value *value, AllocaInst *counter, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    value = resolver.resolve(value, use);
    if (value == nullptr || !value->isa<ArithmeticInst>()) { return false; }
    auto *add = static_cast<ArithmeticInst *>(value);
    if (add->op() != ArithmeticOp::BINARY_ADD ||
        add->operand_count() != 2u) {
        return false;
    }
    for (auto counter_operand = 0u;
         counter_operand < 2u; ++counter_operand) {
        auto one = decode_unsigned(resolver.resolve(
            add->operand(1u - counter_operand), add));
        if (one && *one == 1u &&
            is_current_snapshot(
                add->operand(counter_operand), counter,
                use, resolver, locations)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool match_counter_decrement(
    Value *value, AllocaInst *counter, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    value = resolver.resolve(value, use);
    if (value == nullptr || !value->isa<ArithmeticInst>()) { return false; }
    auto *sub = static_cast<ArithmeticInst *>(value);
    if (sub->op() != ArithmeticOp::BINARY_SUB ||
        sub->operand_count() != 2u) {
        return false;
    }
    auto one = decode_unsigned(
        resolver.resolve(sub->operand(1u), sub));
    return one && *one == 1u &&
           is_current_snapshot(
               sub->operand(0u), counter, use, resolver, locations);
}

enum class ScalarZeroClass : uint8_t {
    unreachable,
    zero,
    nonzero,
    unknown
};

[[nodiscard]] luisa::string_view scalar_zero_class_name(
    ScalarZeroClass value) noexcept {
    switch (value) {
        case ScalarZeroClass::unreachable: return "unreachable";
        case ScalarZeroClass::zero: return "zero";
        case ScalarZeroClass::nonzero: return "nonzero";
        case ScalarZeroClass::unknown: return "unknown";
    }
    return "invalid";
}

[[nodiscard]] ScalarZeroClass merge_scalar_zero_class(
    ScalarZeroClass lhs, ScalarZeroClass rhs) noexcept {
    if (lhs == ScalarZeroClass::unreachable) { return rhs; }
    if (rhs == ScalarZeroClass::unreachable) { return lhs; }
    return lhs == rhs ? lhs : ScalarZeroClass::unknown;
}

[[nodiscard]] ScalarZeroClass transfer_scalar_zero_class(
    BasicBlock *block, AllocaInst *slot,
    ScalarZeroClass state, Instruction *stop_before,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    if (state == ScalarZeroClass::unreachable) {
        return ScalarZeroClass::unreachable;
    }
    for (auto *instruction : block->instructions()) {
        if (instruction == stop_before) { break; }
        if (!instruction->isa<StoreInst>()) { continue; }
        auto *store = static_cast<StoreInst *>(instruction);
        if (store->variable() != slot) { continue; }
        if (auto constant = decode_unsigned(
                resolver.resolve(store->value(), store))) {
            state = *constant == 0u ?
                        ScalarZeroClass::zero :
                        ScalarZeroClass::nonzero;
        } else if (!is_current_snapshot(
                       store->value(), slot, store,
                       resolver, locations)) {
            state = ScalarZeroClass::unknown;
        }
    }
    return state;
}

[[nodiscard]] ScalarZeroClass scalar_zero_class_at_instruction(
    AllocaInst *slot, BasicBlock *target,
    Instruction *instruction, const CoroSemanticGraph &graph,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    if (slot == nullptr || target == nullptr || instruction == nullptr ||
        !resolver.direct_scalar_slot(slot)) {
        return ScalarZeroClass::unknown;
    }
    auto target_id = graph.block_id(target);
    if (target_id >= graph.block_count() || graph.block_count() == 0u) {
        return ScalarZeroClass::unknown;
    }

    // Concrete-set quotient for the current scalar value:
    // unreachable is bottom, zero/nonzero are incomparable singleton
    // classes, and unknown is top. Stores are monotone transfer functions;
    // predecessor union is the join. Sparse forward chaotic iteration thus
    // computes the same least fixed point as exhaustive rescanning and may
    // claim zero only when every executable semantic predecessor agrees.
    luisa::vector<ScalarZeroClass> inputs(
        graph.block_count(), ScalarZeroClass::unreachable);
    luisa::vector<ScalarZeroClass> outputs(
        graph.block_count(), ScalarZeroClass::unreachable);
    inputs[0u] = ScalarZeroClass::unknown;
    luisa::vector<size_t> worklist{0u};
    luisa::vector<uint8_t> queued(graph.block_count(), 0u);
    queued[0u] = 1u;
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto block_id = worklist[cursor];
        queued[block_id] = 0u;
        auto next_output = transfer_scalar_zero_class(
            graph.block(block_id), slot, inputs[block_id], nullptr,
            resolver, locations);
        if (next_output == outputs[block_id]) { continue; }
        outputs[block_id] = next_output;
        for (auto successor : graph.successors(block_id)) {
            auto next_input = merge_scalar_zero_class(
                inputs[successor], next_output);
            if (next_input != inputs[successor]) {
                inputs[successor] = next_input;
                if (queued[successor] == 0u) {
                    queued[successor] = 1u;
                    worklist.emplace_back(successor);
                }
            }
        }
    }
    auto result = transfer_scalar_zero_class(
        target, slot, inputs[target_id], instruction,
        resolver, locations);
    if (auto *dump = std::getenv("LUISA_CORO_DUMP_ALLOCA_SCOPE");
        dump != nullptr && luisa::string_view{dump} == "1") {
        auto location = locations.find(instruction);
        LUISA_INFO(
            "Coroutine scalar-zero fact: slot='{}' target={} ordinal={} "
            "input={} result={} predecessors={}.",
            slot->name().value_or("<unnamed>"), target_id,
            location == locations.end() ?
                ~size_t{0u} : location->second.ordinal,
            scalar_zero_class_name(inputs[target_id]),
            scalar_zero_class_name(result),
            graph.predecessors(target_id).size());
        for (auto predecessor : graph.predecessors(target_id)) {
            XIRDebugPrinter printer;
            luisa::string terminator_ir;
            auto *terminator = graph.block(predecessor)->terminator();
            if (terminator != nullptr) {
                printer.emit_instruction(terminator_ir, terminator);
            }
            LUISA_INFO(
                "Coroutine scalar-zero predecessor: slot='{}' "
                "target={} predecessor={} output={} suspend_edge={} "
                "terminator=[{}].",
                slot->name().value_or("<unnamed>"), target_id,
                predecessor,
                scalar_zero_class_name(outputs[predecessor]),
                graph.is_suspend_edge(predecessor, target_id),
                terminator_ir);
        }
    }
    return result;
}

[[nodiscard]] AllocaInst *condition_implies_less_than_counter(
    Value *condition, bool truth, AllocaInst *counter,
    Instruction *use, const ScalarResolver &resolver,
    const InstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 16u) { return nullptr; }
    condition = strip_boolean_wrappers(
        condition, truth, use, resolver);
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return nullptr;
    }
    auto *comparison = static_cast<ArithmeticInst *>(condition);
    if (comparison->operand_count() != 2u) { return nullptr; }
    if (selected_boolean_edge_implies_operands(comparison, truth)) {
        if (auto *slot = condition_implies_less_than_counter(
                comparison->operand(0u), truth, counter, use,
                resolver, locations, depth + 1u)) {
            return slot;
        }
        return condition_implies_less_than_counter(
            comparison->operand(1u), truth, counter, use,
            resolver, locations, depth + 1u);
    }
    auto *lhs = comparison->operand(0u);
    auto *rhs = comparison->operand(1u);
    auto *lhs_slot = current_scalar_slot(lhs, use, resolver, locations);
    auto *rhs_slot = current_scalar_slot(rhs, use, resolver, locations);
    const auto valid_index = [counter](AllocaInst *slot) noexcept {
        return slot != nullptr && slot != counter &&
               slot->type() == counter->type();
    };
    auto lhs_counter = is_current_snapshot(
        lhs, counter, use, resolver, locations);
    auto rhs_counter = is_current_snapshot(
        rhs, counter, use, resolver, locations);
    if (valid_index(lhs_slot) && rhs_counter &&
        ((truth && comparison->op() == ArithmeticOp::BINARY_LESS) ||
         (!truth && comparison->op() ==
                        ArithmeticOp::BINARY_GREATER_EQUAL))) {
        return lhs_slot;
    }
    if (valid_index(rhs_slot) && lhs_counter &&
        ((truth && comparison->op() == ArithmeticOp::BINARY_GREATER) ||
         (!truth && comparison->op() ==
                        ArithmeticOp::BINARY_LESS_EQUAL))) {
        return rhs_slot;
    }
    return nullptr;
}

[[nodiscard]] bool condition_implies_positive_counter(
    Value *condition, bool truth, AllocaInst *counter,
    Instruction *use, const ScalarResolver &resolver,
    const InstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 16u) { return false; }
    condition = strip_boolean_wrappers(
        condition, truth, use, resolver);
    if (condition != nullptr && condition->isa<ArithmeticInst>()) {
        auto *arithmetic = static_cast<ArithmeticInst *>(condition);
        if (selected_boolean_edge_implies_operands(arithmetic, truth)) {
            return condition_implies_positive_counter(
                       arithmetic->operand(0u), truth, counter, use,
                       resolver, locations, depth + 1u) ||
                   condition_implies_positive_counter(
                       arithmetic->operand(1u), truth, counter, use,
                       resolver, locations, depth + 1u);
        }
    }
    auto test = match_equality_test(condition, use, resolver);
    if (test && test->constant == 0u &&
        is_current_snapshot(
            test->value, counter, use, resolver, locations)) {
        auto equality_selected =
            truth == test->equal_when_condition_true;
        return !equality_selected;
    }
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return false;
    }
    auto *comparison = static_cast<ArithmeticInst *>(condition);
    if (comparison->operand_count() != 2u) { return false; }
    auto zero_rhs = decode_unsigned(
        resolver.resolve(comparison->operand(1u), comparison));
    return truth && zero_rhs && *zero_rhs == 0u &&
           comparison->op() == ArithmeticOp::BINARY_GREATER &&
           is_current_snapshot(
               comparison->operand(0u), counter, use,
               resolver, locations);
}

[[nodiscard]] AllocaInst *counter_from_current_element_store(
    StoreInst *store, AllocaInst *array,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    auto *gep = full_element_gep(store->variable(), array);
    if (gep == nullptr) { return nullptr; }
    auto *index = resolver.resolve(gep->index(0u), gep);
    if (index == nullptr || !index->isa<LoadInst>()) { return nullptr; }
    auto *variable = static_cast<LoadInst *>(index)->variable();
    if (variable == nullptr || !variable->isa<AllocaInst>()) {
        return nullptr;
    }
    auto *counter = static_cast<AllocaInst *>(variable);
    return counter->type() != nullptr && counter->type()->is_scalar() &&
                   counter->type()->is_uint() &&
                   is_current_snapshot(
                       index, counter, gep, resolver, locations) ?
               counter :
               nullptr;
}

struct CounterTransitionSummary {
    luisa::vector<StoreInst *> resets;
    bool has_publication{false};
};

[[nodiscard]] CounterTransitionSummary summarize_counter_transitions(
    AllocaInst *counter, const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    CounterTransitionSummary result;
    if (!resolver.direct_scalar_slot(counter)) { return result; }
    for (auto *use : counter->use_list()) {
        auto *user = use == nullptr ? nullptr : use->user();
        if (user == nullptr || !user->isa<StoreInst>()) { continue; }
        auto *store = static_cast<StoreInst *>(user);
        if (store->variable() != counter) { continue; }
        auto constant = decode_unsigned(
            resolver.resolve(store->value(), store));
        if (constant && *constant == 0u) {
            result.resets.emplace_back(store);
        } else if (match_counter_increment(
                       store->value(), counter, store,
                       resolver, locations)) {
            result.has_publication = true;
        }
    }
    return result;
}

[[nodiscard]] bool block_dominates_region(
    BasicBlock *block, const ArrayUseRegion &region,
    const CoroSemanticGraph &graph) noexcept {
    return block != nullptr && std::all_of(
        region.blocks.begin(), region.blocks.end(),
        [&](BasicBlock *use_block) noexcept {
            return graph.dominates(block, use_block);
        });
}

[[nodiscard]] Instruction *find_lifetime_insertion(
    BasicBlock *target, StoreInst *reset) noexcept {
    if (target == nullptr || reset == nullptr ||
        reset->parent_block() != target) {
        return nullptr;
    }
    Instruction *candidate = nullptr;
    for (auto *instruction : target->instructions()) {
        if (instruction == reset) {
            return candidate != nullptr ? candidate : reset;
        }
        if (instruction->isa<CoroResumeInst>()) {
            candidate = nullptr;
        } else if (candidate == nullptr) {
            candidate = instruction;
        }
    }
    return nullptr;
}

[[nodiscard]] Instruction *find_post_resume_block_entry(
    BasicBlock *target) noexcept {
    if (target == nullptr) { return nullptr; }
    Instruction *candidate = nullptr;
    for (auto *instruction : target->instructions()) {
        if (instruction->isa<CoroResumeInst>()) {
            // A resume is a semantic block-entry boundary. Locals belonging
            // to this continuation must not be placed before the last such
            // boundary in malformed or not-yet-canonicalized input.
            candidate = nullptr;
        } else if (candidate == nullptr) {
            candidate = instruction;
        }
    }
    return candidate;
}

}// namespace

namespace {

struct DiscriminatedState {
    bool published_prefix{false};
    bool pending_tag{false};
    bool pending_payload_defined{false};
    // Valuations on which the physical record at index C has no known tag.
    // This differs from `pending_tag`: the Boolean May set preserves the
    // correlation when published and rolled-back paths later join.
    GuardedMaySet pending_tag_unsafe;
    GuardedTagSet older_undefined;
    GuardedTagSet tail_undefined;
    GuardedTagSet pending_undefined;
    luisa::vector<uint64_t> payload_defined;
    luisa::vector<uint64_t> tag_defined;
    Value *tail_tag_source{nullptr};
    Value *pending_tag_source{nullptr};
    Instruction *older_undefined_origin{nullptr};
    Instruction *tail_undefined_origin{nullptr};
    Instruction *pending_undefined_origin{nullptr};
    CoroGuardedScalarRelationDomain relations;
    luisa::unordered_map<AllocaInst *, TagSet> tag_constraints;

    DiscriminatedState(
        CoroBooleanSetManager &sets,
        const TagDomain &tags,
        size_t payload_dimension,
        luisa::span<const CoroMaskedScalarWitness>
            masked_scalar_witnesses) noexcept
        : pending_tag_unsafe{
              sets, CoroBooleanSetManager::universe()},
          older_undefined{sets, tags},
          tail_undefined{sets, tags},
          pending_undefined{sets, tags},
          // Parentheses select vector(count, value). Braces would select
          // initializer_list and manufacture definition bits from the word
          // count instead of starting the Must domain at the empty set.
          payload_defined(
              (payload_dimension + 63u) / 64u, uint64_t{0u}),
          tag_defined(
              (payload_dimension + 63u) / 64u, uint64_t{0u}),
          relations{sets, masked_scalar_witnesses} {}

    [[nodiscard]] bool operator==(
        const DiscriminatedState &rhs) const noexcept {
        return published_prefix == rhs.published_prefix &&
               pending_tag == rhs.pending_tag &&
               pending_payload_defined == rhs.pending_payload_defined &&
               pending_tag_unsafe == rhs.pending_tag_unsafe &&
               older_undefined == rhs.older_undefined &&
               tail_undefined == rhs.tail_undefined &&
               pending_undefined == rhs.pending_undefined &&
               payload_defined == rhs.payload_defined &&
               tag_defined == rhs.tag_defined &&
               tail_tag_source == rhs.tail_tag_source &&
               pending_tag_source == rhs.pending_tag_source &&
               relations == rhs.relations &&
               tag_constraints == rhs.tag_constraints;
    }
};

struct CandidateDiagnostics {
    bool detail{false};
    bool emitted_first_older_growth{false};
};

[[nodiscard]] bool merge_state(
    DiscriminatedState &target,
    const DiscriminatedState &incoming,
    const TagDomain &tags) noexcept {
    auto before = target;
    auto target_tail_empty = target.tail_undefined.empty();
    auto incoming_tail_empty = incoming.tail_undefined.empty();
    auto target_older_empty = target.older_undefined.empty();
    auto incoming_older_empty = incoming.older_undefined.empty();
    auto target_pending_empty = target.pending_undefined.empty();
    auto incoming_pending_empty = incoming.pending_undefined.empty();
    const auto merge_source = [](
                                  auto *lhs, bool lhs_empty,
                                  auto *rhs, bool rhs_empty) noexcept {
        using Pointer = decltype(lhs);
        if (lhs_empty && rhs_empty) { return Pointer{nullptr}; }
        if (lhs_empty) { return rhs; }
        if (rhs_empty) { return lhs; }
        return lhs == rhs ? lhs : Pointer{nullptr};
    };
    auto *merged_tail_source = merge_source(
        target.tail_tag_source, target_tail_empty,
        incoming.tail_tag_source, incoming_tail_empty);
    auto *merged_pending_source = merge_source(
        target.pending_tag_source, target_pending_empty,
        incoming.pending_tag_source, incoming_pending_empty);
    auto *merged_older_origin = merge_source(
        target.older_undefined_origin, target_older_empty,
        incoming.older_undefined_origin, incoming_older_empty);
    auto *merged_tail_origin = merge_source(
        target.tail_undefined_origin, target_tail_empty,
        incoming.tail_undefined_origin, incoming_tail_empty);
    auto *merged_pending_origin = merge_source(
        target.pending_undefined_origin, target_pending_empty,
        incoming.pending_undefined_origin, incoming_pending_empty);
    target.published_prefix =
        target.published_prefix && incoming.published_prefix;
    target.pending_tag = target.pending_tag && incoming.pending_tag;
    target.pending_payload_defined =
        target.pending_payload_defined &&
        incoming.pending_payload_defined;
    target.pending_tag_unsafe.unite(incoming.pending_tag_unsafe);
    target.older_undefined.unite(incoming.older_undefined);
    target.tail_undefined.unite(incoming.tail_undefined);
    target.pending_undefined.unite(incoming.pending_undefined);
    for (size_t i = 0u; i < target.payload_defined.size(); ++i) {
        target.payload_defined[i] &= incoming.payload_defined[i];
        target.tag_defined[i] &= incoming.tag_defined[i];
    }
    target.tail_tag_source = merged_tail_source;
    target.pending_tag_source = merged_pending_source;
    target.older_undefined_origin = merged_older_origin;
    target.tail_undefined_origin = merged_tail_origin;
    target.pending_undefined_origin = merged_pending_origin;
    static_cast<void>(target.relations.merge(incoming.relations));

    luisa::vector<AllocaInst *> constrained_indices;
    constrained_indices.reserve(
        target.tag_constraints.size() +
        incoming.tag_constraints.size());
    for (auto &&[index, _] : target.tag_constraints) {
        constrained_indices.emplace_back(index);
    }
    for (auto &&[index, _] : incoming.tag_constraints) {
        if (std::find(
                constrained_indices.begin(),
                constrained_indices.end(), index) ==
            constrained_indices.end()) {
            constrained_indices.emplace_back(index);
        }
    }
    luisa::unordered_map<AllocaInst *, TagSet> merged_constraints;
    auto all = tags.all();
    for (auto *index : constrained_indices) {
        auto lhs = target.tag_constraints.contains(index) ?
                       target.tag_constraints.at(index) :
                       all;
        auto rhs = incoming.tag_constraints.contains(index) ?
                       incoming.tag_constraints.at(index) :
                       all;
        auto merged = tags.unite(std::move(lhs), rhs);
        if (!tags.is_all(merged)) {
            merged_constraints.emplace(index, std::move(merged));
        }
    }
    target.tag_constraints = std::move(merged_constraints);
    return target != before;
}

struct CandidateContext {
    AllocaInst *payload;
    AllocaInst *tag;
    AllocaInst *counter;
    size_t dimension;
    const ArrayUseRegion &payload_region;
    const ActiveSlice &slice;
    const CoroSemanticGraph &graph;
    const InstructionLocationMap &locations;
    const ScalarResolver &resolver;
    const TagDomain &tags;
    const luisa::unordered_set<AllocaInst *> &relation_slots;
    const CoroScalarRelationLiveness &relation_liveness;
    const luisa::unordered_set<Value *> &boolean_guards;
    luisa::span<const CoroMaskedScalarWitness>
        masked_scalar_witnesses;
    CandidateDiagnostics *diagnostics;
};

[[nodiscard]] CoroMaskedScalarProjection evaluate_masked_scalar_expression(
    Value *value, uint64_t mask, Instruction *use,
    const CoroGuardedScalarRelationDomain &relations,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (value == nullptr || use == nullptr || depth >= 32u) {
        return relations.masked_scalar_unknown_projection();
    }
    if (auto *source = current_scalar_slot(
            value, use, resolver, locations)) {
        return relations.masked_scalar_load_projection(source, mask);
    }
    value = resolver.resolve(value, use);
    if (auto constant = decode_unsigned(value)) {
        return relations.masked_scalar_constant_projection(
            *constant, mask);
    }
    if (value == nullptr || !value->isa<ArithmeticInst>()) {
        return relations.masked_scalar_unknown_projection();
    }
    auto *arithmetic = static_cast<ArithmeticInst *>(value);
    const auto evaluate_operand = [&](size_t operand) noexcept {
        return evaluate_masked_scalar_expression(
            arithmetic->operand(operand), mask, use, relations,
            resolver, locations, depth + 1u);
    };
    switch (arithmetic->op()) {
        case ArithmeticOp::BINARY_BIT_AND:
            if (arithmetic->operand_count() == 2u) {
                return relations.masked_scalar_projection_intersection(
                    evaluate_operand(0u), evaluate_operand(1u));
            }
            break;
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
            if (arithmetic->operand_count() == 2u) {
                return relations.masked_scalar_projection_union(
                    evaluate_operand(0u), evaluate_operand(1u));
            }
            break;
        case ArithmeticOp::SELECT:
            if (arithmetic->operand_count() == 3u) {
                // The selector is not coupled to this scalar projection, so
                // either data arm may be chosen. Union is the least sound May
                // result for both nonzero and violating states.
                return relations.masked_scalar_projection_union(
                    evaluate_operand(0u), evaluate_operand(1u));
            }
            break;
        default: break;
    }
    return relations.masked_scalar_unknown_projection();
}

void transfer_masked_scalar_store(
    StoreInst *store, CoroGuardedScalarRelationDomain &relations,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    auto *pointer = store == nullptr ? nullptr : store->variable();
    auto *destination =
        pointer != nullptr && pointer->isa<AllocaInst>() ?
            static_cast<AllocaInst *>(pointer) :
            nullptr;
    if (!relations.tracks_masked_scalar(destination)) { return; }

    // Evaluate all projections in the pre-store product, then commit them
    // together. This is simultaneous assignment even for `S = S | K`.
    auto masks = relations.masked_scalar_masks(destination);
    luisa::vector<uint8_t> implied_before;
    auto trace = false;
    if (auto *detail = std::getenv(
            "LUISA_CORO_DUMP_ALLOCA_SCOPE_DETAIL");
        detail != nullptr && luisa::string_view{detail} == "1") {
        trace = true;
        implied_before.reserve(masks.size());
        for (auto mask : masks) {
            implied_before.emplace_back(static_cast<uint8_t>(
                relations.masked_nonzero_implies_counter_positive(
                    destination, mask)));
        }
    }
    luisa::vector<CoroMaskedScalarProjection> projections;
    projections.reserve(masks.size());
    for (auto mask : masks) {
        projections.emplace_back(evaluate_masked_scalar_expression(
            store->value(), mask, store, relations,
            resolver, locations));
    }
    for (size_t i = 0u; i < masks.size(); ++i) {
        relations.assign_masked_scalar_projection(
            destination, masks[i], projections[i]);
    }
    if (trace) {
        auto location = locations.find(store);
        XIRDebugPrinter printer;
        luisa::string value_ir;
        auto *value = resolver.resolve(store->value(), store);
        if (value != nullptr && value->isa<Instruction>()) {
            printer.emit_instruction(
                value_ir, static_cast<Instruction *>(value));
        }
        for (size_t i = 0u; i < masks.size(); ++i) {
            LUISA_INFO(
                "Coroutine discriminated-prefix masked transfer: "
                "scalar='{}' mask=0x{:x} before={} after={} "
                "block={} ordinal={} value=[{}].",
                destination->name().value_or("<unnamed>"), masks[i],
                implied_before[i],
                relations.masked_nonzero_implies_counter_positive(
                    destination, masks[i]),
                location == locations.end() ? ~size_t{0u} :
                                              location->second.block,
                location == locations.end() ? ~size_t{0u} :
                                              location->second.ordinal,
                value_ir);
        }
    }
}

[[nodiscard]] TagSet possible_undefined_tags(
    const GuardedTagSet &older,
    const GuardedTagSet &tail,
    const TagDomain &tags) noexcept {
    return tags.unite(
        older.possible_tags(tags), tail.possible_tags(tags));
}

[[nodiscard]] TagSet abstract_tag_value(
    Value *value, Instruction *use,
    const ScalarResolver &resolver,
    const TagDomain &tags,
    Value *&source) noexcept {
    value = resolver.resolve(value, use);
    if (auto constant = decode_unsigned(value)) {
        source = nullptr;
        return tags.singleton(*constant);
    }
    source = value;
    return tags.all();
}

void clear_tag_constraint_for_store(
    DiscriminatedState &state, AllocaInst *slot) noexcept {
    if (slot != nullptr) { state.tag_constraints.erase(slot); }
}

[[nodiscard]] bool process_array_store(
    StoreInst *store, DiscriminatedState &state,
    const CandidateContext &context) noexcept {
    auto *payload_gep = full_element_gep(
        store->variable(), context.payload);
    auto *tag_gep = full_element_gep(
        store->variable(), context.tag);
    if (payload_gep == nullptr && tag_gep == nullptr) {
        return true;
    }
    auto *gep = payload_gep != nullptr ? payload_gep : tag_gep;
    auto *index_value = gep->index(0u);
    auto *index_slot = current_scalar_slot(
        index_value, store, context.resolver, context.locations);
    auto at_counter = is_current_snapshot(
                          index_value, context.counter, store,
                          context.resolver, context.locations) ||
                      state.relations.knows_equal(index_slot);
    auto at_last = state.relations.knows_last(index_slot);
    auto below_counter = state.relations.knows_less(index_slot);

    if (payload_gep != nullptr) {
        auto unsafe_before = context.tags.population(
            possible_undefined_tags(
                state.older_undefined, state.tail_undefined,
                context.tags));
        if (auto index = decode_unsigned(context.resolver.resolve(
                index_value, store));
            index && *index < context.dimension) {
            state.payload_defined[*index / 64u] |=
                uint64_t{1u} << (*index % 64u);
        }
        if (at_counter) {
            state.pending_payload_defined = true;
            state.pending_undefined.clear();
            state.pending_tag_source = nullptr;
            state.pending_undefined_origin = nullptr;
        } else if (at_last) {
            state.tail_undefined.clear();
            state.tail_tag_source = nullptr;
            state.tail_undefined_origin = nullptr;
        }
        if (auto *detail = std::getenv(
                "LUISA_CORO_DUMP_ALLOCA_SCOPE_DETAIL");
            detail != nullptr && luisa::string_view{detail} == "1") {
            auto location = context.locations.find(store);
            auto unsafe_after = context.tags.population(
                possible_undefined_tags(
                    state.older_undefined, state.tail_undefined,
                    context.tags));
            LUISA_INFO(
                "Coroutine discriminated-prefix payload definition: "
                "payload='{}' block={} ordinal={} index='{}' "
                "at_counter={} at_last={} below_counter={} "
                "unsafe_before={} unsafe_after={}.",
                context.payload->name().value_or("<unnamed>"),
                location == context.locations.end() ?
                    ~size_t{0u} : location->second.block,
                location == context.locations.end() ?
                    ~size_t{0u} : location->second.ordinal,
                index_slot == nullptr ? luisa::string_view{"<none>"} :
                                        index_slot->name().value_or(
                                            "<unnamed>"),
                at_counter, at_last, below_counter,
                unsafe_before, unsafe_after);
        }
        // A write to an arbitrary older record may make the proof stronger,
        // but removing a tag from the existential unsafe summary would need
        // a cardinality/index partition. Retaining it is conservative.
        return true;
    }

    if (auto index = decode_unsigned(context.resolver.resolve(
            index_value, store));
        index && *index < context.dimension) {
        state.tag_defined[*index / 64u] |=
            uint64_t{1u} << (*index % 64u);
    }
    state.tag_constraints.clear();
    Value *source = nullptr;
    auto stored_tags = abstract_tag_value(
        store->value(), store, context.resolver,
        context.tags, source);
    if (at_counter) {
        state.pending_tag = true;
        state.pending_tag_unsafe.clear();
        if (state.pending_payload_defined) {
            state.pending_undefined.clear();
            state.pending_tag_source = nullptr;
            state.pending_undefined_origin = nullptr;
        } else {
            state.pending_undefined.assign(
                stored_tags, state.relations.feasible_set());
            state.pending_tag_source = source;
            state.pending_undefined_origin = store;
        }
    } else if (at_last) {
        if (!state.tail_undefined.empty()) {
            // Every represented path with an undefined tail executes this
            // overwrite, so its old tag is replaced rather than unioned.
            state.tail_undefined.retag(stored_tags);
            state.tail_tag_source = source;
        }
    } else if (below_counter) {
        if (!state.older_undefined.empty()) {
            // The selected older record may be one of the undefined records;
            // other undefined records remain. Union is the least sound May
            // update without tracking record identity.
            state.older_undefined.unite_retag(stored_tags);
        }
        if (!state.tail_undefined.empty()) {
            // Strictly below C does not exclude C-1. Without an exact last
            // identity the write may retag the current tail as well.
            state.tail_undefined.unite_retag(stored_tags);
            state.tail_tag_source = nullptr;
        }
    } else if (!state.older_undefined.empty() ||
               !state.tail_undefined.empty()) {
        // An unclassified write may retag any undefined published record.
        state.older_undefined.widen_tags_to_all(context.tags);
        state.tail_undefined.widen_tags_to_all(context.tags);
        state.tail_tag_source = nullptr;
    }
    return true;
}

void reset_published_records(
    DiscriminatedState &state,
    const TagDomain &tags) noexcept {
    state.published_prefix = true;
    state.pending_tag = false;
    state.pending_payload_defined = false;
    state.older_undefined.clear();
    state.tail_undefined.clear();
    // C:=0 starts a new candidate lifetime. The physical record at P[C]
    // does not exist until its tag and payload are written in this epoch.
    state.pending_tag_unsafe.assign(state.relations.feasible_set());
    state.pending_undefined.assign(
        tags.all(), state.relations.feasible_set());
    state.tail_tag_source = nullptr;
    state.pending_tag_source = nullptr;
    state.older_undefined_origin = nullptr;
    state.tail_undefined_origin = nullptr;
    state.pending_undefined_origin = nullptr;
    state.tag_constraints.clear();
    state.relations.clear_relations();
    state.relations.clear_tail();
    state.relations.clear_counter_positive();
    state.relations.invalidate_counter_implications();
}

void publish_pending_record(
    DiscriminatedState &state,
    const TagDomain &tags,
    Instruction *publication) noexcept {
    state.published_prefix =
        state.published_prefix && state.pending_tag;
    auto older_was_empty = state.older_undefined.empty();
    auto tail_was_empty = state.tail_undefined.empty();
    state.older_undefined.unite(state.tail_undefined);
    if (older_was_empty && !tail_was_empty) {
        state.older_undefined_origin = state.tail_undefined_origin;
    } else if (!older_was_empty && !tail_was_empty &&
               state.older_undefined_origin !=
                   state.tail_undefined_origin) {
        state.older_undefined_origin = nullptr;
    }
    if (state.pending_tag) {
        state.tail_undefined = state.pending_undefined;
    } else {
        state.tail_undefined.assign(
            tags.all(), state.relations.feasible_set());
    }
    state.tail_tag_source = state.pending_tag ?
                                state.pending_tag_source :
                                nullptr;
    state.tail_undefined_origin = state.pending_tag ?
                                      state.pending_undefined_origin :
                                      publication;
    state.pending_tag = false;
    state.pending_payload_defined = false;
    // After C++ the just-published record is C-1; the new physical P[C] is
    // absent until the next allocation transaction starts.
    state.pending_tag_unsafe.assign(state.relations.feasible_set());
    state.pending_undefined.assign(
        tags.all(), state.relations.feasible_set());
    state.pending_tag_source = nullptr;
    state.pending_undefined_origin = nullptr;
    state.relations.advance_counter();
}

void rollback_tail_record(
    DiscriminatedState &state,
    const TagDomain &) noexcept {
    // Removing the current tail cannot create an unsafe record. The new tail
    // is one of the records already summarized by older_undefined. Keeping
    // that tag set in both summaries is an over-approximation, never a proof
    // strengthening.
    // The removed semantic tail remains physically present at the new index
    // C. Preserve it as the pending/spare record before replacing the
    // published tail summary with the older-prefix summary.
    auto removed_undefined = state.tail_undefined;
    auto *removed_tag_source = state.tail_tag_source;
    auto *removed_origin = state.tail_undefined_origin;
    state.tail_undefined = state.older_undefined;
    state.tail_tag_source = nullptr;
    state.tail_undefined_origin = state.older_undefined_origin;
    state.pending_tag = state.published_prefix;
    state.pending_payload_defined = removed_undefined.empty();
    state.pending_tag_unsafe.assign(
        state.published_prefix ?
            CoroBooleanSetManager::empty_set() :
            state.relations.feasible_set());
    state.pending_undefined = std::move(removed_undefined);
    state.pending_tag_source = removed_tag_source;
    state.pending_undefined_origin = removed_origin;
    state.relations.retreat_counter();
    state.relations.clear_tail();
}

void process_scalar_store(
    StoreInst *store, DiscriminatedState &state,
    const CandidateContext &context, bool diagnose) noexcept {
    auto *pointer = store->variable();
    auto *destination =
        pointer != nullptr && pointer->isa<AllocaInst>() ?
            static_cast<AllocaInst *>(pointer) :
            nullptr;
    if (destination == nullptr || destination->type() == nullptr ||
        !destination->type()->is_scalar()) {
        return;
    }
    transfer_masked_scalar_store(
        store, state.relations,
        context.resolver, context.locations);
    // Evaluate the source in the pre-store state. If D := S copies an exact
    // scalar index, every predicate about T[S] denotes the same array element
    // through D until either scalar is redefined or the tag array is written.
    // Capture that fact before killing D's previous version.
    auto *source = current_scalar_slot(
        store->value(), store, context.resolver, context.locations);
    luisa::optional<TagSet> copied_tag_constraint;
    if (source != nullptr) {
        if (auto iter = state.tag_constraints.find(source);
            iter != state.tag_constraints.end()) {
            copied_tag_constraint = iter->second;
        }
    }
    clear_tag_constraint_for_store(state, destination);
    if (destination == context.counter) {
        auto older_was_empty = state.older_undefined.empty();
        auto *tail_origin_before = state.tail_undefined_origin;
        auto published_before = state.published_prefix;
        auto trace_transition = diagnose ||
                                (context.diagnostics != nullptr &&
                                 context.diagnostics->detail);
        auto older_before = size_t{0u};
        auto tail_before = size_t{0u};
        auto pending_before = size_t{0u};
        auto tail_word_before = uint64_t{0u};
        auto tail_bit0_guard_before = luisa::string{};
        if (trace_transition) {
            older_before = context.tags.population(
                state.older_undefined.possible_tags(context.tags));
            auto tail_tags =
                state.tail_undefined.possible_tags(context.tags);
            tail_before = context.tags.population(tail_tags);
            tail_word_before = tail_tags.words.empty() ?
                                   0u : tail_tags.words.front();
            tail_bit0_guard_before =
                state.tail_undefined.describe_valuation(0u);
            pending_before = context.tags.population(
                state.pending_undefined.possible_tags(context.tags));
        }
        auto *tail_source_before = state.tail_tag_source;
        auto *pending_source_before = state.pending_tag_source;
        auto transition = luisa::string_view{"unsupported"};
        auto resolved = context.resolver.resolve(store->value(), store);
        auto constant = decode_unsigned(resolved);
        if (constant && *constant == 0u) {
            transition = "reset";
            reset_published_records(state, context.tags);
        } else if (is_current_snapshot(
                       store->value(), context.counter, store,
                       context.resolver, context.locations)) {
            transition = "identity";
            // Exact self-assignment preserves the transition state.
        } else if (match_counter_increment(
                       store->value(), context.counter, store,
                       context.resolver, context.locations)) {
            transition = "publish";
            publish_pending_record(state, context.tags, store);
        } else if (match_counter_decrement(
                       store->value(), context.counter, store,
                       context.resolver, context.locations) &&
                   state.relations.knows_counter_positive()) {
            transition = "rollback";
            rollback_tail_record(state, context.tags);
        } else {
            state.published_prefix = false;
            state.pending_tag = false;
            state.pending_payload_defined = false;
            state.older_undefined.assign(
                context.tags.all(), state.relations.feasible_set());
            state.tail_undefined.assign(
                context.tags.all(), state.relations.feasible_set());
            state.pending_tag_unsafe.assign(
                state.relations.feasible_set());
            state.pending_undefined.assign(
                context.tags.all(), state.relations.feasible_set());
            state.tail_tag_source = nullptr;
            state.pending_tag_source = nullptr;
            state.older_undefined_origin = store;
            state.tail_undefined_origin = store;
            state.pending_undefined_origin = store;
            state.tag_constraints.clear();
            state.relations.clear_relations();
            state.relations.clear_tail();
            state.relations.clear_counter_positive();
            state.relations.invalidate_counter_implications();
        }
        const auto trace_lost_prefix = [&]() noexcept {
            auto *detail = std::getenv(
                "LUISA_CORO_DUMP_ALLOCA_SCOPE_DETAIL");
            return detail != nullptr &&
                   luisa::string_view{detail} == "1" &&
                   published_before && !state.published_prefix;
        }();
        if (!diagnose && published_before && older_was_empty &&
            !state.older_undefined.empty() &&
            context.diagnostics != nullptr &&
            context.diagnostics->detail &&
            !context.diagnostics->emitted_first_older_growth) {
            context.diagnostics->emitted_first_older_growth = true;
            auto location = context.locations.find(store);
            XIRDebugPrinter printer;
            luisa::string block_ir;
            printer.emit_basic_block(block_ir, store->parent_block());
            LUISA_INFO(
                "Coroutine discriminated-prefix first older-undefined "
                "growth: payload='{}' tag='{}' counter='{}' block={} "
                "ordinal={} kind={} older_before={} tail_before={} "
                "tail_word_before=0x{:016x} pending_before={} "
                "tail_bit0_guard={} block_ir=[{}].",
                context.payload->name().value_or("<unnamed>"),
                context.tag->name().value_or("<unnamed>"),
                context.counter->name().value_or("<unnamed>"),
                location == context.locations.end() ?
                    ~size_t{0u} : location->second.block,
                location == context.locations.end() ?
                    ~size_t{0u} : location->second.ordinal,
                transition, older_before, tail_before, tail_word_before,
                pending_before,
                tail_bit0_guard_before,
                block_ir);
            if (tail_origin_before != nullptr) {
                auto origin_location =
                    context.locations.find(tail_origin_before);
                luisa::string origin_ir;
                printer.emit_instruction(
                    origin_ir, tail_origin_before);
                luisa::string origin_block_ir;
                printer.emit_basic_block(
                    origin_block_ir,
                    tail_origin_before->parent_block());
                LUISA_INFO(
                    "Coroutine discriminated-prefix first-growth origin: "
                    "block={} ordinal={} ir=[{}] block_ir=[{}].",
                    origin_location == context.locations.end() ?
                        ~size_t{0u} : origin_location->second.block,
                    origin_location == context.locations.end() ?
                        ~size_t{0u} : origin_location->second.ordinal,
                    origin_ir, origin_block_ir);
                if (origin_location != context.locations.end()) {
                    auto origin = origin_location->second.block;
                    auto begin = origin > 24u ? origin - 24u : 0u;
                    auto end = std::min(
                        context.graph.block_count(), origin + 32u);
                    for (auto nearby = begin; nearby < end; ++nearby) {
                        luisa::string nearby_ir;
                        printer.emit_basic_block(
                            nearby_ir, context.graph.block(nearby));
                        LUISA_INFO(
                            "Coroutine discriminated-prefix first-growth "
                            "origin-nearby block: block={} ir=[{}].",
                            nearby, nearby_ir);
                    }
                }
            }
            for (auto *predicate :
                 // The old tail was moved to older by publication.
                 state.older_undefined.valuation_support(0u)) {
                luisa::string predicate_ir;
                if (predicate != nullptr && predicate->isa<Instruction>()) {
                    printer.emit_instruction(
                        predicate_ir,
                        static_cast<Instruction *>(predicate));
                }
                LUISA_INFO(
                    "Coroutine discriminated-prefix first-growth guard: "
                    "predicate='{}' kind={} ir=[{}].",
                    predicate == nullptr ? luisa::string_view{"<null>"} :
                                           predicate->name().value_or(
                                               "<unnamed>"),
                    predicate != nullptr && predicate->isa<AllocaInst>() ?
                        "alloca" : "ssa",
                    predicate_ir);
                if (predicate == nullptr ||
                    !predicate->isa<AllocaInst>()) {
                    continue;
                }
                for (auto *use : predicate->use_list()) {
                    auto *user = use == nullptr ? nullptr : use->user();
                    if (user == nullptr || !user->isa<StoreInst>() ||
                        static_cast<StoreInst *>(user)->variable() !=
                            predicate) {
                        continue;
                    }
                    auto *definition = static_cast<StoreInst *>(user);
                    auto guard_location =
                        context.locations.find(definition);
                    luisa::string store_ir;
                    printer.emit_instruction(store_ir, definition);
                    LUISA_INFO(
                        "Coroutine discriminated-prefix first-growth guard "
                        "definition: predicate='{}' block={} ordinal={} "
                        "ir=[{}].",
                        predicate->name().value_or("<unnamed>"),
                        guard_location == context.locations.end() ?
                            ~size_t{0u} : guard_location->second.block,
                        guard_location == context.locations.end() ?
                            ~size_t{0u} : guard_location->second.ordinal,
                        store_ir);
                }
            }
            if (location != context.locations.end()) {
                auto nearby_begin = location->second.block > 32u ?
                                        location->second.block - 32u : 0u;
                for (auto nearby = nearby_begin;
                     nearby < location->second.block; ++nearby) {
                    luisa::string nearby_ir;
                    printer.emit_basic_block(
                        nearby_ir, context.graph.block(nearby));
                    LUISA_INFO(
                        "Coroutine discriminated-prefix first-growth "
                        "nearby block: block={} ir=[{}].",
                        nearby, nearby_ir);
                }
                luisa::vector<std::pair<size_t, size_t>> worklist;
                luisa::unordered_set<size_t> visited;
                worklist.emplace_back(location->second.block, 0u);
                for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
                    auto [block, depth] = worklist[cursor];
                    if (depth >= 8u) { continue; }
                    for (auto predecessor :
                         context.graph.predecessors(block)) {
                        if (!visited.emplace(predecessor).second) { continue; }
                        worklist.emplace_back(predecessor, depth + 1u);
                        auto *terminator =
                            context.graph.block(predecessor)->terminator();
                        luisa::string terminator_ir;
                        if (terminator != nullptr) {
                            printer.emit_instruction(
                                terminator_ir, terminator);
                        }
                        LUISA_INFO(
                            "Coroutine discriminated-prefix first-growth "
                            "predecessor: block={} depth={} terminator=[{}].",
                            predecessor, depth + 1u, terminator_ir);
                    }
                }
            }
        }
        if (diagnose || trace_lost_prefix) {
            auto *dump = std::getenv("LUISA_CORO_DUMP_ALLOCA_SCOPE");
            if (dump != nullptr && luisa::string_view{dump} == "1") {
                auto location = context.locations.find(store);
                XIRDebugPrinter printer;
                luisa::string store_ir;
                printer.emit_instruction(store_ir, store);
                LUISA_INFO(
                    "Coroutine discriminated-prefix counter transition: "
                    "payload='{}' tag='{}' counter='{}' block={} "
                    "ordinal={} kind={} published_before={} "
                    "published_after={} pending_tag={} loss={} "
                    "older={}->{} tail={}->{} pending={}->{} "
                    "tail_source='{}' pending_source='{}' ir=[{}].",
                    context.payload->name().value_or("<unnamed>"),
                    context.tag->name().value_or("<unnamed>"),
                    context.counter->name().value_or("<unnamed>"),
                    location == context.locations.end() ?
                        ~size_t{0u} : location->second.block,
                    location == context.locations.end() ?
                        ~size_t{0u} : location->second.ordinal,
                    transition, published_before,
                    state.published_prefix, state.pending_tag,
                    trace_lost_prefix,
                    older_before,
                    context.tags.population(
                        state.older_undefined.possible_tags(context.tags)),
                    tail_before,
                    context.tags.population(
                        state.tail_undefined.possible_tags(context.tags)),
                    pending_before,
                    context.tags.population(
                        state.pending_undefined.possible_tags(context.tags)),
                    tail_source_before == nullptr ?
                        luisa::string_view{"<none>"} :
                        tail_source_before->name().value_or("<unnamed>"),
                    pending_source_before == nullptr ?
                        luisa::string_view{"<none>"} :
                        pending_source_before->name().value_or("<unnamed>"),
                    store_ir);
            }
        }
        return;
    }
    // The abstract relation state is projected to scalar slots that can
    // reach an eventual array-index or guard obligation. The collector is
    // closed backwards over exact scalar copies, so a store outside this set
    // cannot influence any relation queried by this candidate.
    if (!context.relation_slots.contains(destination)) { return; }
    if (!destination->type()->is_uint()) {
        state.relations.erase_index(destination);
        return;
    }
    auto constant = decode_unsigned(
        context.resolver.resolve(store->value(), store));
    if (source != nullptr) {
        state.relations.assign_index_copy(
            destination, source, source == context.counter);
        if (copied_tag_constraint) {
            state.tag_constraints.emplace(
                destination, std::move(*copied_tag_constraint));
        }
    } else {
        state.relations.erase_index(destination);
    }
    // Unsigned zero is below C exactly when C>0. This is the default arm of
    // many total index selectors; it must be generated after the assignment
    // kill so the relation joins normally with bounds from other arms.
    if (constant && *constant == 0u &&
        state.relations.knows_counter_positive()) {
        state.relations.add_less(destination);
    }
    if (auto *detail = std::getenv("LUISA_CORO_DUMP_ALLOCA_SCOPE_DETAIL");
        detail != nullptr && luisa::string_view{detail} == "1") {
        auto location = context.locations.find(store);
        LUISA_INFO(
            "Coroutine discriminated-prefix scalar transfer: "
            "destination='{}' source='{}' constant={} positive={} less={} "
            "block={} ordinal={}.",
            destination->name().value_or("<unnamed>"),
            source == nullptr ? luisa::string_view{"<none>"} :
                                source->name().value_or("<unnamed>"),
            constant.value_or(~uint64_t{0u}),
            state.relations.knows_counter_positive(),
            state.relations.knows_less(destination),
            location == context.locations.end() ? ~size_t{0u} :
                                                   location->second.block,
            location == context.locations.end() ? ~size_t{0u} :
                                                   location->second.ordinal);
    }
}

[[nodiscard]] bool payload_index_is_published(
    Value *index, Instruction *read,
    const DiscriminatedState &state,
    const CandidateContext &context,
    AllocaInst *&index_slot) noexcept {
    index_slot = current_scalar_slot(
        index, read, context.resolver, context.locations);
    return state.published_prefix && index_slot != nullptr &&
           state.relations.knows_less(index_slot);
}

[[nodiscard]] bool statically_defined_array_index(
    Value *index, Instruction *read,
    luisa::span<const uint64_t> defined,
    const CandidateContext &context) noexcept {
    auto constant = decode_unsigned(
        context.resolver.resolve(index, read));
    return constant && *constant < context.dimension &&
           (defined[*constant / 64u] &
            (uint64_t{1u} << (*constant % 64u))) != 0u;
}

[[nodiscard]] bool validate_tag_index(
    Value *index, LoadInst *read,
    const DiscriminatedState &state,
    const CandidateContext &context,
    size_t depth = 0u) noexcept;

[[nodiscard]] bool discriminator_state_unchanged(
    LoadInst *load, Instruction *use,
    const CandidateContext &context) noexcept {
    if (load == nullptr || use == nullptr ||
        load->parent_block() != use->parent_block()) {
        return false;
    }
    auto load_location = context.locations.find(load);
    auto use_location = context.locations.find(use);
    if (load_location == context.locations.end() ||
        use_location == context.locations.end() ||
        load_location->second.ordinal >= use_location->second.ordinal) {
        return false;
    }
    for (auto *instruction : load->parent_block()->instructions()) {
        auto location = context.locations.find(instruction);
        if (location == context.locations.end() ||
            location->second.ordinal <= load_location->second.ordinal ||
            location->second.ordinal >= use_location->second.ordinal ||
            !instruction->isa<StoreInst>()) {
            continue;
        }
        auto *store = static_cast<StoreInst *>(instruction);
        if (store->variable() == context.counter ||
            full_element_gep(store->variable(), context.tag) != nullptr) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] AllocaInst *validated_tag_load_index_slot(
    Value *value, Instruction *use,
    const DiscriminatedState &state,
    const CandidateContext &context) noexcept {
    auto access = tag_load_access(
        value, context.tag, use,
        context.resolver, context.locations);
    if (!access || access->index_slot == nullptr ||
        !discriminator_state_unchanged(access->load, use, context) ||
        !validate_tag_index(
            access->index, access->load, state, context)) {
        return nullptr;
    }
    return access->index_slot;
}

[[nodiscard]] bool payload_tags_are_safe(
    AllocaInst *index_slot,
    const DiscriminatedState &state,
    const CandidateContext &context,
    Value *arm_condition = nullptr,
    bool arm_truth = false,
    Instruction *read = nullptr) noexcept {
    if (index_slot == nullptr) { return false; }
    auto allowed = state.tag_constraints.contains(index_slot) ?
                       state.tag_constraints.at(index_slot) :
                       context.tags.all();
    if (arm_condition != nullptr && read != nullptr) {
        if (auto equality = match_equality_test(
                arm_condition, read, context.resolver)) {
            if (validated_tag_load_index_slot(
                    equality->value, read, state, context) == index_slot) {
                auto selected =
                    arm_truth == equality->equal_when_condition_true ?
                        context.tags.singleton(equality->constant) :
                        context.tags.subtract(
                            context.tags.all(),
                            context.tags.singleton(equality->constant));
                allowed = context.tags.intersect(
                    std::move(allowed), selected);
            }
        }
    }
    return !state.older_undefined.may_contain_any(allowed) &&
           !state.tail_undefined.may_contain_any(allowed);
}

[[nodiscard]] bool zero_tag_index_is_physically_defined(
    const DiscriminatedState &state) noexcept {
    // For unsigned C, index zero is in the published prefix when C>0 and is
    // exactly the physical pending/spare record when C==0. The relation
    // domain's unsafe set may over-approximate C==0, so requiring the pending
    // tag on all of it is conservative.
    return state.published_prefix &&
           !state.pending_tag_unsafe.may_intersect(
               state.relations.counter_positive_unsafe_set());
}

[[nodiscard]] bool zero_payload_index_is_physically_defined(
    AllocaInst *index_slot,
    const DiscriminatedState &state,
    const CandidateContext &context) noexcept {
    if (!zero_tag_index_is_physically_defined(state) ||
        index_slot == nullptr) {
        return false;
    }
    auto allowed = state.tag_constraints.contains(index_slot) ?
                       state.tag_constraints.at(index_slot) :
                       context.tags.all();
    // When C>0, zero denotes a published record; when C==0 it denotes the
    // retained physical record at P[C]. Both obligations are checked under
    // the same tag constraint. May sets make this proof fail closed if the
    // numeric partition or tag correlation was lost.
    return !state.older_undefined.may_contain_any(allowed) &&
           !state.tail_undefined.may_contain_any(allowed) &&
           !state.pending_undefined.may_contain_any_on(
               allowed,
               state.relations.counter_positive_unsafe_set());
}

[[nodiscard]] bool validate_payload_index(
    Value *index, LoadInst *read,
    const DiscriminatedState &state,
    const CandidateContext &context,
    size_t depth = 0u) noexcept {
    auto *selected_index_slot = current_scalar_slot(
        index, read, context.resolver, context.locations);
    if (statically_defined_array_index(
            index, read, state.payload_defined, context)) {
        return true;
    }
    auto constant = decode_unsigned(
        context.resolver.resolve(index, read));
    if (constant && *constant == 0u &&
        zero_payload_index_is_physically_defined(
            selected_index_slot, state, context)) {
        return true;
    }
    AllocaInst *index_slot = nullptr;
    if (payload_index_is_published(
            index, read, state, context, index_slot) &&
        payload_tags_are_safe(index_slot, state, context)) {
        return true;
    }
    if (depth >= 8u) { return false; }
    auto *resolved = context.resolver.resolve(index, read);
    if (resolved == nullptr || !resolved->isa<ArithmeticInst>()) {
        return false;
    }
    auto *select = static_cast<ArithmeticInst *>(resolved);
    if (select->op() != ArithmeticOp::SELECT ||
        select->operand_count() != 3u ||
        select->operand(2u)->type() != Type::of<bool>()) {
        return false;
    }
    auto *condition = select->operand(2u);
    const auto arm_is_safe = [&](Value *arm, bool truth) noexcept {
        if (statically_defined_array_index(
                arm, read, state.payload_defined, context)) {
            return true;
        }
        auto *slot = current_scalar_slot(
            arm, read, context.resolver, context.locations);
        auto *proved_slot = condition_implies_less_than_counter(
            condition, truth, context.counter, read,
            context.resolver, context.locations);
        auto *constraint_slot = selected_index_slot != nullptr ?
                                    selected_index_slot :
                                    slot;
        if (state.published_prefix && slot != nullptr &&
            proved_slot == slot && payload_tags_are_safe(
                                      constraint_slot, state, context,
                                      condition, truth, read)) {
            return true;
        }
        return validate_payload_index(
            arm, read, state, context, depth + 1u);
    };
    // XIR SELECT is select(false_value, true_value, condition). The load is
    // defined iff the selected arm is defined on every valuation; proving
    // both guarded obligations is therefore necessary and sufficient in
    // this abstract domain.
    return arm_is_safe(select->operand(0u), false) &&
           arm_is_safe(select->operand(1u), true);
}

[[nodiscard]] bool validate_tag_index(
    Value *index, LoadInst *read,
    const DiscriminatedState &state,
    const CandidateContext &context,
    size_t depth) noexcept {
    if (statically_defined_array_index(
            index, read, state.tag_defined, context)) {
        return true;
    }
    auto constant = decode_unsigned(
        context.resolver.resolve(index, read));
    if (constant && *constant == 0u &&
        zero_tag_index_is_physically_defined(state)) {
        return true;
    }
    AllocaInst *index_slot = nullptr;
    if (payload_index_is_published(
            index, read, state, context, index_slot)) {
        return true;
    }
    if (depth >= 8u) { return false; }
    auto *resolved = context.resolver.resolve(index, read);
    if (resolved == nullptr || !resolved->isa<ArithmeticInst>()) {
        return false;
    }
    auto *select = static_cast<ArithmeticInst *>(resolved);
    if (select->op() != ArithmeticOp::SELECT ||
        select->operand_count() != 3u ||
        select->operand(2u)->type() != Type::of<bool>()) {
        return false;
    }
    auto *condition = select->operand(2u);
    const auto arm_is_safe = [&](Value *arm, bool truth) noexcept {
        if (statically_defined_array_index(
                arm, read, state.tag_defined, context)) {
            return true;
        }
        auto *slot = current_scalar_slot(
            arm, read, context.resolver, context.locations);
        auto *proved_slot = condition_implies_less_than_counter(
            condition, truth, context.counter, read,
            context.resolver, context.locations);
        return (state.published_prefix && slot != nullptr &&
                proved_slot == slot) ||
               validate_tag_index(
                   arm, read, state, context, depth + 1u);
    };
    return arm_is_safe(select->operand(0u), false) &&
           arm_is_safe(select->operand(1u), true);
}

[[nodiscard]] bool validate_payload_load(
    LoadInst *load, const DiscriminatedState &state,
    const CandidateContext &context,
    bool &used_discriminated_read) noexcept {
    auto *gep = full_element_gep(load->variable(), context.payload);
    if (gep == nullptr) { return false; }
    if (!validate_payload_index(
            gep->index(0u), load, state, context)) {
        return false;
    }
    used_discriminated_read = true;
    return true;
}

void dump_failed_read(
    LoadInst *load, const DiscriminatedState &state,
    const CandidateContext &context) noexcept {
    auto *dump = std::getenv("LUISA_CORO_DUMP_ALLOCA_SCOPE");
    if (dump == nullptr || luisa::string_view{dump} != "1") { return; }
    auto *payload_gep = full_element_gep(
        load->variable(), context.payload);
    auto *tag_gep = full_element_gep(
        load->variable(), context.tag);
    auto *index = payload_gep != nullptr ? payload_gep->index(0u) :
                  tag_gep != nullptr     ? tag_gep->index(0u) :
                                           nullptr;
    auto *resolved = index == nullptr ?
                         nullptr :
                         context.resolver.resolve(index, load);
    auto *slot = index == nullptr ?
                     nullptr :
                     current_scalar_slot(
                         index, load, context.resolver,
                         context.locations);
    auto static_defined = index != nullptr &&
                          statically_defined_array_index(
                              index, load,
                              payload_gep != nullptr ?
                                  luisa::span<const uint64_t>{
                                      state.payload_defined} :
                                  luisa::span<const uint64_t>{
                                      state.tag_defined},
                              context);
    auto unsafe = possible_undefined_tags(
        state.older_undefined, state.tail_undefined,
        context.tags);
    auto allowed = slot != nullptr &&
                           state.tag_constraints.contains(slot) ?
                       state.tag_constraints.at(slot) :
                       context.tags.all();
    AllocaInst *false_less = nullptr;
    AllocaInst *true_less = nullptr;
    auto false_static = false;
    auto true_static = false;
    if (resolved != nullptr && resolved->isa<ArithmeticInst>()) {
        auto *arithmetic = static_cast<ArithmeticInst *>(resolved);
        if (arithmetic->op() == ArithmeticOp::SELECT &&
            arithmetic->operand_count() == 3u) {
            auto defined = payload_gep != nullptr ?
                               luisa::span<const uint64_t>{
                                   state.payload_defined} :
                               luisa::span<const uint64_t>{
                                   state.tag_defined};
            false_static = statically_defined_array_index(
                arithmetic->operand(0u), load, defined, context);
            true_static = statically_defined_array_index(
                arithmetic->operand(1u), load, defined, context);
            false_less = condition_implies_less_than_counter(
                arithmetic->operand(2u), false, context.counter, load,
                context.resolver, context.locations);
            true_less = condition_implies_less_than_counter(
                arithmetic->operand(2u), true, context.counter, load,
                context.resolver, context.locations);
        }
    }
    auto location = context.locations.find(load);
    auto block = location == context.locations.end() ?
                     ~size_t{0u} :
                     location->second.block;
    auto ordinal = location == context.locations.end() ?
                       ~size_t{0u} :
                       location->second.ordinal;
    XIRDebugPrinter printer;
    luisa::string read_ir;
    printer.emit_instruction(read_ir, load);
    luisa::string index_ir;
    if (resolved != nullptr && resolved->isa<Instruction>()) {
        printer.emit_instruction(
            index_ir, static_cast<Instruction *>(resolved));
    }
    LUISA_INFO(
        "Coroutine discriminated-prefix failing {} read: payload='{}' "
        "tag='{}' counter='{}' block={} ordinal={} published={} "
        "static={} slot='{}' less={} tag_constraint={} unsafe_tags={} "
        "allowed_tags={} disjoint={} select_false_static={} "
        "select_true_static={} select_false_less='{}' "
        "select_true_less='{}' read_ir=[{}] index_ir=[{}].",
        payload_gep != nullptr ? "payload" : "tag",
        context.payload->name().value_or("<unnamed>"),
        context.tag->name().value_or("<unnamed>"),
        context.counter->name().value_or("<unnamed>"),
        block, ordinal, state.published_prefix, static_defined,
        slot == nullptr ? luisa::string_view{"<none>"} :
                          slot->name().value_or("<unnamed>"),
        state.relations.knows_less(slot),
        slot != nullptr && state.tag_constraints.contains(slot),
        context.tags.population(unsafe),
        context.tags.population(allowed),
        context.tags.disjoint(unsafe, allowed),
        false_static, true_static,
        false_less == nullptr ? luisa::string_view{"<none>"} :
                                false_less->name().value_or("<unnamed>"),
        true_less == nullptr ? luisa::string_view{"<none>"} :
                               true_less->name().value_or("<unnamed>"),
        read_ir, index_ir);

    auto *detail = std::getenv(
        "LUISA_CORO_DUMP_ALLOCA_SCOPE_DETAIL");
    if (detail == nullptr || luisa::string_view{detail} != "1") {
        return;
    }
    luisa::string block_ir;
    printer.emit_basic_block(block_ir, load->parent_block());
    LUISA_INFO(
        "Coroutine discriminated-prefix failing-read block: block={} "
        "ir=[{}].",
        block, block_ir);
    if (slot == nullptr) { return; }
    auto live_indices = context.relation_liveness.live_in(block);
    auto live_at_entry = std::find(
                             live_indices.begin(), live_indices.end(), slot) !=
                         live_indices.end();
    if (!state.relations.knows_less(slot)) {
        LUISA_INFO(
            "Coroutine discriminated-prefix missing-relation evidence: "
            "index='{}' block={} live_at_entry={} predecessors={}.",
            slot->name().value_or("<unnamed>"), block, live_at_entry,
            context.graph.predecessors(block).size());
    }
    if (payload_gep != nullptr &&
        !state.tag_constraints.contains(slot)) {
        luisa::string constraints;
        for (auto &&[constrained, _] : state.tag_constraints) {
            if (!constraints.empty()) { constraints.append(","); }
            constraints.append(
                constrained->name().value_or("<unnamed>"));
        }
        LUISA_INFO(
            "Coroutine discriminated-prefix missing-tag evidence: "
            "index='{}' block={} live_at_entry={} constraints=[{}].",
            slot->name().value_or("<unnamed>"), block,
            live_at_entry, constraints);
        for (auto *use : slot->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(user);
            if (store->variable() != slot) { continue; }
            auto store_location = context.locations.find(store);
            auto *source = current_scalar_slot(
                store->value(), store, context.resolver,
                context.locations);
            XIRDebugPrinter store_printer;
            luisa::string value_ir;
            auto *value = context.resolver.resolve(
                store->value(), store);
            if (value != nullptr && value->isa<Instruction>()) {
                store_printer.emit_instruction(
                    value_ir, static_cast<Instruction *>(value));
            }
            LUISA_INFO(
                "Coroutine discriminated-prefix index definition: "
                "index='{}' block={} ordinal={} source='{}' value=[{}].",
                slot->name().value_or("<unnamed>"),
                store_location == context.locations.end() ?
                    ~size_t{0u} : store_location->second.block,
                store_location == context.locations.end() ?
                    ~size_t{0u} : store_location->second.ordinal,
                source == nullptr ? luisa::string_view{"<none>"} :
                                    source->name().value_or("<unnamed>"),
                value_ir);
        }
    }
    luisa::vector<std::pair<size_t, size_t>> predecessor_worklist;
    luisa::unordered_set<size_t> visited_predecessors;
    predecessor_worklist.emplace_back(block, 0u);
    for (size_t cursor = 0u; cursor < predecessor_worklist.size(); ++cursor) {
        auto [current, depth] = predecessor_worklist[cursor];
        if (depth >= 6u) { continue; }
        for (auto predecessor : context.graph.predecessors(current)) {
            if (!visited_predecessors.emplace(predecessor).second) {
                continue;
            }
            predecessor_worklist.emplace_back(predecessor, depth + 1u);
        auto *terminator = context.graph.block(predecessor)->terminator();
        XIRDebugPrinter edge_printer;
        luisa::string terminator_ir;
        luisa::string condition_ir;
        if (terminator != nullptr) {
            edge_printer.emit_instruction(terminator_ir, terminator);
            if (terminator->isa<ConditionalBranchInst>()) {
                auto *branch = static_cast<ConditionalBranchInst *>(
                    terminator);
                auto *condition = context.resolver.resolve(
                    branch->condition(), branch);
                if (condition != nullptr && condition->isa<Instruction>()) {
                    edge_printer.emit_instruction(
                        condition_ir,
                        static_cast<Instruction *>(condition));
                }
            } else if (terminator->isa<IndexedBranchInst>()) {
                auto *branch = static_cast<IndexedBranchInst *>(terminator);
                auto *branch_slot = tag_load_index_slot(
                    branch->value(), context.tag, branch,
                    context.resolver, context.locations);
                condition_ir.append(
                    branch_slot == nullptr ? "<no-tag-index>" :
                                             branch_slot->name().value_or(
                                                 "<unnamed>"));
            }
        }
        LUISA_INFO(
            "Coroutine discriminated-prefix predecessor evidence: "
            "block={} depth={} terminator=[{}] resolved_condition=[{}].",
            predecessor, depth + 1u, terminator_ir, condition_ir);
        }
    }
}

[[nodiscard]] bool process_instruction(
    Instruction *instruction, DiscriminatedState &state,
    const CandidateContext &context, bool validate_reads,
    bool &used_discriminated_read,
    Instruction *&failing_read) noexcept {
    const auto finish = [&](bool result) noexcept {
        // Liveness death points can only be attached to a direct scalar
        // load/store or to the two branch kinds recorded as semantic uses by
        // collect_relation_slots. Avoid a hash lookup for every unrelated
        // arithmetic instruction in a large shader.
        if (instruction->isa<LoadInst>() ||
            instruction->isa<StoreInst>() ||
            instruction->isa<ConditionalBranchInst>() ||
            instruction->isa<IndexedBranchInst>()) {
            for (auto *slot :
                 context.relation_liveness.dead_after(instruction)) {
                state.relations.erase_index(slot);
                state.tag_constraints.erase(slot);
            }
        }
        return result;
    };
    // An SSA instruction is a fresh dynamic Boolean value on every block
    // execution. Kill the previous valuation before interpreting this
    // definition; mutable Boolean locals are killed by their stores below.
    if (instruction->type() == Type::of<bool>() &&
        context.boolean_guards.contains(instruction)) {
        state.relations.forget_boolean(instruction);
        state.older_undefined.forget_boolean(instruction);
        state.tail_undefined.forget_boolean(instruction);
        state.pending_undefined.forget_boolean(instruction);
        state.pending_tag_unsafe.forget_boolean(instruction);
    }
    if (instruction->isa<AssumeInst>()) {
        auto *assume = static_cast<AssumeInst *>(instruction);
        auto masked_test = match_masked_scalar_test(
            assume->condition(), assume,
            context.resolver, context.locations);
        if (context.diagnostics != nullptr &&
            context.diagnostics->detail) {
            auto location = context.locations.find(assume);
            XIRDebugPrinter printer;
            luisa::string condition_ir;
            auto *condition = context.resolver.resolve(
                assume->condition(), assume);
            if (condition != nullptr && condition->isa<Instruction>()) {
                printer.emit_instruction(
                    condition_ir, static_cast<Instruction *>(condition));
            }
            LUISA_INFO(
                "Coroutine discriminated-prefix assume: matched={} "
                "scalar='{}' mask=0x{:x} nonzero={} block={} "
                "ordinal={} condition=[{}].",
                masked_test.has_value(),
                masked_test ? masked_test->scalar->name().value_or(
                                  "<unnamed>") :
                              luisa::string_view{"<none>"},
                masked_test ? masked_test->mask : 0u,
                masked_test && masked_test->nonzero_when_condition_true,
                location == context.locations.end() ? ~size_t{0u} :
                                                      location->second.block,
                location == context.locations.end() ? ~size_t{0u} :
                                                      location->second.ordinal,
                condition_ir);
        }
        if (masked_test) {
            auto test = *masked_test;
            if (test.nonzero_when_condition_true) {
                if (!state.relations.refine_masked_scalar_nonzero(
                        test.scalar, test.mask)) {
                    return finish(true);
                }
                if (state.relations.
                        masked_nonzero_implies_counter_positive(
                            test.scalar, test.mask)) {
                    state.relations.add_counter_positive();
                }
            } else {
                state.relations.assume_masked_scalar_zero(
                    test.scalar, test.mask);
            }
        }
    }
    if (instruction->isa<StoreInst>()) {
        auto *store = static_cast<StoreInst *>(instruction);
        static_cast<void>(process_array_store(store, state, context));
        auto *pointer = store->variable();
        auto *destination =
            pointer != nullptr && pointer->isa<AllocaInst>() ?
                static_cast<AllocaInst *>(pointer) :
                nullptr;
        if (destination != nullptr &&
            destination->type() == Type::of<bool>() &&
            context.boolean_guards.contains(destination) &&
            !is_current_snapshot(
                store->value(), destination, store,
                context.resolver, context.locations)) {
            auto source = analyze_boolean_expression(
                store->value(), store,
                context.resolver, context.locations);
            state.relations.assign_boolean(
                destination, source.predicate,
                source.true_when_predicate_is, source.constant);
            state.older_undefined.assign_boolean(
                destination, source.predicate,
                source.true_when_predicate_is, source.constant);
            state.tail_undefined.assign_boolean(
                destination, source.predicate,
                source.true_when_predicate_is, source.constant);
            state.pending_undefined.assign_boolean(
                destination, source.predicate,
                source.true_when_predicate_is, source.constant);
            state.pending_tag_unsafe.assign_boolean(
                destination, source.predicate,
                source.true_when_predicate_is, source.constant);
        }
        process_scalar_store(store, state, context, validate_reads);
        return finish(true);
    }
    if (!instruction->isa<LoadInst>() || !validate_reads) {
        return finish(true);
    }
    auto *load = static_cast<LoadInst *>(instruction);
    auto *payload_gep = full_element_gep(
        load->variable(), context.payload);
    if (payload_gep != nullptr) {
        if (validate_payload_load(
                load, state, context, used_discriminated_read)) {
            return finish(true);
        }
        dump_failed_read(load, state, context);
        failing_read = load;
        return finish(false);
    }
    // The tag array is a discriminator, not the lifetime candidate. Its
    // unrelated reads remain unchanged when `payload` moves. Reads used as
    // proof evidence are checked at the refinement site by
    // validated_tag_load_index_slot; globally requiring every tag read to be
    // in the counted prefix would impose an obligation unrelated to this
    // transformation.
    return finish(true);
}

[[nodiscard]] AllocaInst *condition_implies_last_before_counter(
    Value *condition, bool truth, AllocaInst *counter,
    Instruction *use, const ScalarResolver &resolver,
    const InstructionLocationMap &locations,
    size_t depth = 0u) noexcept {
    if (depth >= 16u) { return nullptr; }
    condition = strip_boolean_wrappers(
        condition, truth, use, resolver);
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        return nullptr;
    }
    auto *comparison = static_cast<ArithmeticInst *>(condition);
    if (comparison->operand_count() != 2u) { return nullptr; }
    if (selected_boolean_edge_implies_operands(comparison, truth)) {
        if (auto *slot = condition_implies_last_before_counter(
                comparison->operand(0u), truth, counter, use,
                resolver, locations, depth + 1u)) {
            return slot;
        }
        return condition_implies_last_before_counter(
            comparison->operand(1u), truth, counter, use,
            resolver, locations, depth + 1u);
    }
    if (!((truth && comparison->op() == ArithmeticOp::BINARY_EQUAL) ||
          (!truth && comparison->op() ==
                         ArithmeticOp::BINARY_NOT_EQUAL))) {
        return nullptr;
    }
    const auto one_before = [&](Value *value) noexcept {
        value = resolver.resolve(value, use);
        if (value == nullptr || !value->isa<ArithmeticInst>()) {
            return static_cast<AllocaInst *>(nullptr);
        }
        auto *add = static_cast<ArithmeticInst *>(value);
        if (add->op() != ArithmeticOp::BINARY_ADD ||
            add->operand_count() != 2u) {
            return static_cast<AllocaInst *>(nullptr);
        }
        for (auto slot_operand = 0u;
             slot_operand < 2u; ++slot_operand) {
            auto one = decode_unsigned(resolver.resolve(
                add->operand(1u - slot_operand), add));
            auto *slot = current_scalar_slot(
                add->operand(slot_operand), use,
                resolver, locations);
            if (one && *one == 1u && slot != nullptr &&
                slot != counter && slot->type() == counter->type()) {
                return slot;
            }
        }
        return static_cast<AllocaInst *>(nullptr);
    };
    if (is_current_snapshot(
            comparison->operand(1u), counter, use,
            resolver, locations)) {
        return one_before(comparison->operand(0u));
    }
    if (is_current_snapshot(
            comparison->operand(0u), counter, use,
            resolver, locations)) {
        return one_before(comparison->operand(1u));
    }
    return nullptr;
}

// Returns a value only when the current abstract state proves that every
// feasible concrete state gives the same Boolean result. The proof is an
// induction over the Boolean expression tree: exact constants are leaves;
// conjunction/disjunction use their ordinary truth tables; unsigned counter
// comparisons are discharged only by relations already proved for the same
// scalar snapshots. Unknown leaves remain unknown, so this routine can only
// remove an infeasible CFG edge and can never make a relation stronger on an
// executable edge.
[[nodiscard]] luisa::optional<bool> prove_condition_value(
    Value *condition, AllocaInst *counter, Instruction *use,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations,
    const CoroGuardedScalarRelationDomain &relations,
    size_t depth = 0u) noexcept {
    if (depth >= 32u || condition == nullptr) { return luisa::nullopt; }
    condition = resolver.resolve(condition, use);
    if (condition == nullptr) { return luisa::nullopt; }
    if (condition->isa<Constant>() &&
        condition->type() == Type::of<bool>()) {
        return static_cast<Constant *>(condition)->as<bool>();
    }
    auto boolean_source = analyze_boolean_expression(
        condition, use, resolver, locations);
    if (boolean_source.constant) { return boolean_source.constant; }
    if (boolean_source.predicate != nullptr) {
        if (auto value = relations.known_boolean(
                boolean_source.predicate)) {
            return *value == boolean_source.true_when_predicate_is;
        }
    }
    if (!condition->isa<ArithmeticInst>()) { return luisa::nullopt; }
    auto *arithmetic = static_cast<ArithmeticInst *>(condition);
    if (arithmetic->op() == ArithmeticOp::UNARY_BIT_NOT &&
        arithmetic->operand_count() == 1u &&
        arithmetic->operand(0u)->type() == Type::of<bool>()) {
        if (auto value = prove_condition_value(
                arithmetic->operand(0u), counter, arithmetic,
                resolver, locations, relations, depth + 1u)) {
            return !*value;
        }
        return luisa::nullopt;
    }
    if (arithmetic->operand_count() != 2u) { return luisa::nullopt; }
    if (arithmetic->type() == Type::of<bool>() &&
        (arithmetic->op() == ArithmeticOp::BINARY_BIT_AND ||
         arithmetic->op() == ArithmeticOp::BINARY_BIT_OR)) {
        auto lhs = prove_condition_value(
            arithmetic->operand(0u), counter, arithmetic,
            resolver, locations, relations, depth + 1u);
        auto rhs = prove_condition_value(
            arithmetic->operand(1u), counter, arithmetic,
            resolver, locations, relations, depth + 1u);
        if (arithmetic->op() == ArithmeticOp::BINARY_BIT_AND) {
            if ((lhs && !*lhs) || (rhs && !*rhs)) { return false; }
            if (lhs && *lhs && rhs && *rhs) { return true; }
        } else {
            if ((lhs && *lhs) || (rhs && *rhs)) { return true; }
            if (lhs && !*lhs && rhs && !*rhs) { return false; }
        }
        return luisa::nullopt;
    }

    // C is unsigned. C>0 excludes exactly the C==0 side of an equality or
    // inequality against zero.
    if (arithmetic->op() == ArithmeticOp::BINARY_EQUAL ||
        arithmetic->op() == ArithmeticOp::BINARY_NOT_EQUAL) {
        if (auto test = match_equality_test(
                arithmetic, arithmetic, resolver);
            test && test->constant == 0u &&
            is_current_snapshot(
                test->value, counter, arithmetic,
                resolver, locations) &&
            relations.knows_counter_positive()) {
            return !test->equal_when_condition_true;
        }

        // The last-record relation includes the non-wrapping C>0 premise,
        // so I+1==C is exact in unsigned arithmetic.
        auto equality_truth =
            arithmetic->op() == ArithmeticOp::BINARY_EQUAL;
        if (auto *index = condition_implies_last_before_counter(
                arithmetic, equality_truth, counter, arithmetic,
                resolver, locations);
            index != nullptr && relations.knows_last(index)) {
            return equality_truth;
        }
    }

    auto *lhs_slot = current_scalar_slot(
        arithmetic->operand(0u), arithmetic, resolver, locations);
    auto *rhs_slot = current_scalar_slot(
        arithmetic->operand(1u), arithmetic, resolver, locations);
    auto lhs_counter = is_current_snapshot(
        arithmetic->operand(0u), counter, arithmetic,
        resolver, locations);
    auto rhs_counter = is_current_snapshot(
        arithmetic->operand(1u), counter, arithmetic,
        resolver, locations);
    if (lhs_slot != nullptr && lhs_slot != counter && rhs_counter &&
        relations.knows_less(lhs_slot)) {
        if (arithmetic->op() == ArithmeticOp::BINARY_LESS) { return true; }
        if (arithmetic->op() == ArithmeticOp::BINARY_GREATER_EQUAL) {
            return false;
        }
    }
    if (rhs_slot != nullptr && rhs_slot != counter && lhs_counter &&
        relations.knows_less(rhs_slot)) {
        if (arithmetic->op() == ArithmeticOp::BINARY_GREATER) { return true; }
        if (arithmetic->op() == ArithmeticOp::BINARY_LESS_EQUAL) {
            return false;
        }
    }
    return luisa::nullopt;
}

void refine_source_tags(
    GuardedTagSet &set, Value *source, const EqualityTest &test,
    bool truth, Instruction *terminator,
    const CandidateContext &context) noexcept {
    if (source == nullptr ||
        !same_snapshot(
            source, test.value, terminator,
            context.resolver, context.locations)) {
        return;
    }
    auto singleton = context.tags.singleton(test.constant);
    auto equality_selected =
        truth == test.equal_when_condition_true;
    auto selected = equality_selected ?
                        singleton :
                        context.tags.subtract(
                            context.tags.all(), singleton);
    set.filter_tags(selected);
}

[[nodiscard]] bool refine_conditional_edge(
    DiscriminatedState &state, BasicBlock *predecessor,
    BasicBlock *successor,
    const CandidateContext &context) noexcept {
    auto *terminator = predecessor == nullptr ?
                           nullptr : predecessor->terminator();
    if (terminator == nullptr ||
        !terminator->isa<ConditionalBranchInst>()) {
        return true;
    }
    auto *branch = static_cast<ConditionalBranchInst *>(terminator);
    bool truth;
    if (branch->true_block() == successor &&
        branch->false_block() != successor) {
        truth = true;
    } else if (branch->false_block() == successor &&
               branch->true_block() != successor) {
        truth = false;
    } else {
        return true;
    }
    if (auto proved = prove_condition_value(
            branch->condition(), context.counter, branch,
            context.resolver, context.locations, state.relations);
        proved && *proved != truth) {
        return false;
    }
    auto condition_source = analyze_boolean_expression(
        branch->condition(), branch,
        context.resolver, context.locations);
    if (condition_source.constant &&
        *condition_source.constant != truth) {
        return false;
    }
    if (condition_source.predicate != nullptr &&
        context.boolean_guards.contains(
            condition_source.predicate)) {
        auto selected_value =
            truth == condition_source.true_when_predicate_is;
        if (!state.relations.refine_boolean(
                condition_source.predicate, selected_value)) {
            return false;
        }
        state.older_undefined.refine_boolean(
            condition_source.predicate, selected_value);
        state.tail_undefined.refine_boolean(
            condition_source.predicate, selected_value);
        state.pending_undefined.refine_boolean(
            condition_source.predicate, selected_value);
        state.pending_tag_unsafe.refine_boolean(
            condition_source.predicate, selected_value);
    }
    if (auto masked_test = match_masked_scalar_test(
            branch->condition(), branch,
            context.resolver, context.locations)) {
        auto selected_nonzero =
            truth == masked_test->nonzero_when_condition_true;
        if (!selected_nonzero) {
            state.relations.assume_masked_scalar_zero(
                masked_test->scalar, masked_test->mask);
        } else if (!state.relations.refine_masked_scalar_nonzero(
                       masked_test->scalar, masked_test->mask)) {
            return false;
        } else if (state.relations.
                       masked_nonzero_implies_counter_positive(
                           masked_test->scalar, masked_test->mask)) {
            state.relations.add_counter_positive();
        }
        if (auto *detail = std::getenv(
                "LUISA_CORO_DUMP_ALLOCA_SCOPE_DETAIL");
            detail != nullptr && luisa::string_view{detail} == "1") {
            auto location = context.locations.find(branch);
            LUISA_INFO(
                "Coroutine discriminated-prefix masked edge: scalar='{}' "
                "mask=0x{:x} selected_nonzero={} implies_positive={} "
                "positive={} block={}.",
                masked_test->scalar->name().value_or("<unnamed>"),
                masked_test->mask, selected_nonzero,
                state.relations.masked_nonzero_implies_counter_positive(
                    masked_test->scalar, masked_test->mask),
                state.relations.knows_counter_positive(),
                location == context.locations.end() ? ~size_t{0u} :
                                                       location->second.block);
        }
    }
    if (auto *index = condition_implies_less_than_counter(
            branch->condition(), truth, context.counter, branch,
            context.resolver, context.locations)) {
        state.relations.erase_index(index);
        state.relations.add_less(index);
    }
    if (condition_implies_positive_counter(
            branch->condition(), truth, context.counter, branch,
            context.resolver, context.locations)) {
        state.relations.add_counter_positive();
    }
    if (auto *index = condition_implies_last_before_counter(
            branch->condition(), truth, context.counter, branch,
            context.resolver, context.locations);
        index != nullptr && state.relations.knows_counter_positive()) {
        state.relations.add_last(index);
    }
    auto equality = match_equality_test(
        branch->condition(), branch, context.resolver);
    if (!equality) { return true; }
    refine_source_tags(
        state.pending_undefined, state.pending_tag_source,
        *equality, truth, branch, context);
    refine_source_tags(
        state.tail_undefined, state.tail_tag_source,
        *equality, truth, branch, context);
    if (auto *index = validated_tag_load_index_slot(
            equality->value, branch, state, context)) {
        auto selected =
            truth == equality->equal_when_condition_true ?
                context.tags.singleton(equality->constant) :
                context.tags.subtract(
                    context.tags.all(),
                    context.tags.singleton(equality->constant));
        if (auto iter = state.tag_constraints.find(index);
            iter != state.tag_constraints.end()) {
            selected = context.tags.intersect(
                std::move(selected), iter->second);
        }
        if (context.tags.is_empty(selected)) { return false; }
        state.tag_constraints[index] = std::move(selected);
    }
    return true;
}

[[nodiscard]] TagSet selected_indexed_branch_tags(
    IndexedBranchTerminatorInstruction *branch,
    BasicBlock *successor,
    const TagDomain &tags) noexcept {
    auto selected = branch->default_block() == successor ?
                        tags.all() : tags.empty();
    if (branch->default_block() == successor) {
        for (size_t i = 0u; i < branch->case_count(); ++i) {
            selected = tags.subtract(
                std::move(selected),
                tags.singleton(branch->case_value(i)));
        }
    }
    for (size_t i = 0u; i < branch->case_count(); ++i) {
        if (branch->case_block(i) == successor) {
            selected = tags.unite(
                std::move(selected),
                tags.singleton(branch->case_value(i)));
        }
    }
    return selected;
}

[[nodiscard]] bool refine_indexed_edge(
    DiscriminatedState &state, BasicBlock *predecessor,
    BasicBlock *successor,
    const CandidateContext &context) noexcept {
    auto *terminator = predecessor == nullptr ?
                           nullptr : predecessor->terminator();
    if (terminator == nullptr ||
        !terminator->isa<IndexedBranchInst>()) {
        return true;
    }
    auto *branch = static_cast<IndexedBranchInst *>(terminator);
    auto selected = selected_indexed_branch_tags(
        branch, successor, context.tags);
    if (context.tags.is_empty(selected)) { return false; }
    auto *selector = context.resolver.resolve(branch->value(), branch);
    if (state.pending_tag_source != nullptr &&
        same_snapshot(
            selector, state.pending_tag_source, branch,
            context.resolver, context.locations)) {
        state.pending_undefined.filter_tags(selected);
    }
    if (state.tail_tag_source != nullptr &&
        same_snapshot(
            selector, state.tail_tag_source, branch,
            context.resolver, context.locations)) {
        state.tail_undefined.filter_tags(selected);
    }
    if (auto *index = validated_tag_load_index_slot(
            selector, branch, state, context)) {
        if (auto iter = state.tag_constraints.find(index);
            iter != state.tag_constraints.end()) {
            selected = context.tags.intersect(
                std::move(selected), iter->second);
        }
        if (context.tags.is_empty(selected)) { return false; }
        state.tag_constraints[index] = std::move(selected);
    }
    return true;
}

[[nodiscard]] bool refine_edge(
    DiscriminatedState &state, BasicBlock *predecessor,
    BasicBlock *successor,
    const CandidateContext &context) noexcept {
    return refine_conditional_edge(
               state, predecessor, successor, context) &&
           refine_indexed_edge(
               state, predecessor, successor, context);
}

struct RelationSlotCollection {
    luisa::vector<AllocaInst *> slots;
    luisa::unordered_set<AllocaInst *> slot_set;
    CoroScalarSemanticUses semantic_uses;
};

[[nodiscard]] luisa::unordered_set<Value *>
collect_boolean_guard_values(
    const CoroSemanticGraph &graph,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    luisa::unordered_set<Value *> result;
    luisa::vector<Value *> worklist;
    const auto insert = [&](Value *predicate) noexcept {
        if (predicate != nullptr &&
            predicate->type() == Type::of<bool>() &&
            result.emplace(predicate).second) {
            worklist.emplace_back(predicate);
        }
    };
    for (size_t block_id = 0u;
         block_id < graph.block_count(); ++block_id) {
        auto *terminator = graph.block(block_id)->terminator();
        if (terminator == nullptr ||
            !terminator->isa<ConditionalBranchInst>()) {
            continue;
        }
        auto *branch = static_cast<ConditionalBranchInst *>(terminator);
        for (auto *predicate : boolean_expression_atoms(
                 branch->condition(), branch, resolver, locations)) {
            insert(predicate);
        }
    }
    // Close backwards over exact Boolean copies. This is the smallest set of
    // predicates whose current values can be observed by a later branch;
    // every other control value can be existentially projected away.
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto *destination = worklist[cursor];
        if (!destination->isa<AllocaInst>()) { continue; }
        for (auto *use : destination->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(user);
            if (store->variable() != destination) { continue; }
            for (auto *predicate : boolean_expression_atoms(
                     store->value(), store, resolver, locations)) {
                insert(predicate);
            }
        }
    }
    return result;
}

[[nodiscard]] RelationSlotCollection collect_relation_slots(
    AllocaInst *payload, AllocaInst *tag, AllocaInst *counter,
    const ActiveSlice &slice, const CoroSemanticGraph &graph,
    const ScalarResolver &resolver,
    const InstructionLocationMap &locations) noexcept {
    RelationSlotCollection result;
    const auto insert = [&](AllocaInst *slot,
                            Instruction *semantic_use) noexcept {
        if (slot == nullptr || slot == counter ||
            slot->type() != counter->type() ||
            !slot->type()->is_uint()) {
            return;
        }
        if (result.slot_set.emplace(slot).second) {
            result.slots.emplace_back(slot);
        }
        if (semantic_use != nullptr) {
            auto &uses = result.semantic_uses[semantic_use];
            if (std::find(uses.begin(), uses.end(), slot) == uses.end()) {
                uses.emplace_back(slot);
            }
        }
    };

    // Seed exactly the scalar memory states queried by this candidate's
    // transfer and read obligations. Relations for every other scalar are a
    // product component that no observation can distinguish, and can be
    // projected away without changing the proof result.
    for (auto block_id : slice.blocks) {
        for (auto *instruction : graph.block(block_id)->instructions()) {
            Value *pointer = nullptr;
            if (instruction->isa<LoadInst>()) {
                pointer = static_cast<LoadInst *>(instruction)->variable();
            } else if (instruction->isa<StoreInst>()) {
                pointer = static_cast<StoreInst *>(instruction)->variable();
            }
            auto *element = pointer == nullptr ? nullptr :
                                full_element_gep(pointer, payload);
            if (element == nullptr && pointer != nullptr) {
                element = full_element_gep(pointer, tag);
            }
            if (element != nullptr) {
                insert(current_scalar_slot(
                           element->index(0u), instruction,
                           resolver, locations),
                       instruction);
            }

            if (instruction->isa<ConditionalBranchInst>()) {
                auto *branch = static_cast<ConditionalBranchInst *>(
                    instruction);
                for (auto truth : {false, true}) {
                    insert(condition_implies_less_than_counter(
                               branch->condition(), truth, counter, branch,
                               resolver, locations),
                           branch);
                    insert(condition_implies_last_before_counter(
                               branch->condition(), truth, counter, branch,
                               resolver, locations),
                           branch);
                }
                if (auto equality = match_equality_test(
                        branch->condition(), branch, resolver)) {
                    insert(tag_load_index_slot(
                               equality->value, tag, branch,
                               resolver, locations),
                           branch);
                }
            } else if (instruction->isa<IndexedBranchInst>()) {
                auto *branch = static_cast<IndexedBranchInst *>(instruction);
                insert(tag_load_index_slot(
                           branch->value(), tag, branch,
                           resolver, locations),
                       branch);
            }
        }
    }

    // Close backwards over exact scalar copies. If D is observable later,
    // the relation of S must remain live until D:=S executes. Unsupported
    // expressions add no source and therefore conservatively leave D
    // unknown; they never synthesize a relation.
    for (size_t cursor = 0u; cursor < result.slots.size(); ++cursor) {
        auto *destination = result.slots[cursor];
        for (auto *use : destination->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(user);
            if (store->variable() != destination) { continue; }
            insert(current_scalar_slot(
                       store->value(), store, resolver, locations),
                   store);
        }
    }
    return result;
}

void retain_tag_constraints(
    DiscriminatedState &state,
    luisa::span<AllocaInst *const> live_indices) noexcept {
    const auto is_live = [live_indices](AllocaInst *index) noexcept {
        return std::find(
                   live_indices.begin(), live_indices.end(), index) !=
               live_indices.end();
    };
    for (auto iter = state.tag_constraints.begin();
         iter != state.tag_constraints.end();) {
        if (!is_live(iter->first)) {
            iter = state.tag_constraints.erase(iter);
        } else {
            ++iter;
        }
    }
}

[[nodiscard]] bool supported_array_region(
    AllocaInst *array, const ArrayUseRegion &region) noexcept {
    if (!region.valid) { return false; }
    for (auto *instruction : region.users) {
        if (instruction->isa<GEPInst>() &&
            region.pointers.contains(
                static_cast<GEPInst *>(instruction)->base())) {
            continue;
        }
        if (instruction->isa<LoadInst>() &&
            full_element_gep(
                static_cast<LoadInst *>(instruction)->variable(),
                array) != nullptr) {
            continue;
        }
        if (instruction->isa<StoreInst>() &&
            full_element_gep(
                static_cast<StoreInst *>(instruction)->variable(),
                array) != nullptr) {
            continue;
        }
        return false;
    }
    return true;
}

[[nodiscard]] bool region_follows_insertion(
    const ArrayUseRegion &region, const ActiveSlice &slice,
    Instruction *insertion,
    const InstructionLocationMap &locations) noexcept {
    auto insertion_location = locations.find(insertion);
    if (insertion_location == locations.end() ||
        insertion_location->second.block != slice.target) {
        return false;
    }
    for (auto *user : region.users) {
        auto location = locations.find(user);
        if (location == locations.end() ||
            (location->second.block == slice.target &&
             location->second.ordinal <
                 insertion_location->second.ordinal)) {
            return false;
        }
    }
    return true;
}

}// namespace

class CoroDiscriminatedPrefixAnalysis::Impl {
private:
    FunctionDefinition *_definition;
    const CoroSemanticGraph &_graph;
    InstructionLocationMap _locations;
    ScalarResolver _resolver;
    luisa::unordered_set<Value *> _boolean_guards;
    luisa::vector<AllocaInst *> _arrays;
    mutable luisa::unordered_map<
        AllocaInst *, CounterTransitionSummary>
        _counter_transitions;

private:
    [[nodiscard]] luisa::vector<uint64_t> _tag_constants(
        AllocaInst *tag, const ActiveSlice &slice) const noexcept {
        luisa::vector<uint64_t> constants;
        const auto insert = [&](uint64_t value) noexcept {
            if (std::find(constants.begin(), constants.end(), value) ==
                constants.end()) {
                constants.emplace_back(value);
            }
        };
        for (auto block_id : slice.blocks) {
            for (auto *instruction : _graph.block(block_id)->instructions()) {
                if (instruction->isa<StoreInst>()) {
                    auto *store = static_cast<StoreInst *>(instruction);
                    if (full_element_gep(store->variable(), tag) != nullptr) {
                        if (auto value = decode_unsigned(
                                _resolver.resolve(store->value(), store))) {
                            insert(*value);
                        }
                    }
                }
                if (instruction->isa<ConditionalBranchInst>()) {
                    auto *branch = static_cast<ConditionalBranchInst *>(
                        instruction);
                    if (auto test = match_equality_test(
                            branch->condition(), branch, _resolver)) {
                        insert(test->constant);
                    }
                } else if (instruction->isa<IndexedBranchInst>()) {
                    auto *branch = static_cast<IndexedBranchInst *>(instruction);
                    for (auto value : branch->case_values()) { insert(value); }
                }
            }
        }
        return constants;
    }

    [[nodiscard]] const CounterTransitionSummary &_counter_summary(
        AllocaInst *counter) const noexcept {
        if (auto iter = _counter_transitions.find(counter);
            iter != _counter_transitions.end()) {
            return iter->second;
        }
        return _counter_transitions.emplace(
            counter,
            summarize_counter_transitions(
                counter, _resolver, _locations))
            .first->second;
    }

    [[nodiscard]] CoroDiscriminatedPrefixProofResult _run_candidate(
        AllocaInst *payload, AllocaInst *tag, AllocaInst *counter,
        const ArrayUseRegion &payload_region,
        BasicBlock *target, Instruction *insertion) const noexcept {
        CoroDiscriminatedPrefixProofResult result;
        auto *requested_insertion = insertion;
        auto starts_with_empty_prefix =
            scalar_zero_class_at_instruction(
                counter, target, insertion, _graph,
                _resolver, _locations) == ScalarZeroClass::zero;
        if (!starts_with_empty_prefix) {
            auto *earlier = find_post_resume_block_entry(target);
            if (earlier != nullptr && earlier != insertion &&
                scalar_zero_class_at_instruction(
                    counter, target, earlier, _graph,
                    _resolver, _locations) == ScalarZeroClass::zero) {
                insertion = earlier;
                starts_with_empty_prefix = true;
            }
        }
        if (auto *dump = std::getenv("LUISA_CORO_DUMP_ALLOCA_SCOPE");
            dump != nullptr && luisa::string_view{dump} == "1") {
            auto requested = _locations.find(requested_insertion);
            auto selected = _locations.find(insertion);
            LUISA_INFO(
                "Coroutine discriminated-prefix candidate start: "
                "payload='{}' tag='{}' counter='{}' target={} "
                "requested_ordinal={} selected_ordinal={} empty_prefix={}.",
                payload->name().value_or("<unnamed>"),
                tag->name().value_or("<unnamed>"),
                counter->name().value_or("<unnamed>"),
                _graph.block_id(target),
                requested == _locations.end() ?
                    ~size_t{0u} : requested->second.ordinal,
                selected == _locations.end() ?
                    ~size_t{0u} : selected->second.ordinal,
                starts_with_empty_prefix);
        }
        auto slice = make_active_slice(target, payload_region, _graph);
        auto follows_insertion =
            slice.valid && region_follows_insertion(
                               payload_region, slice,
                               insertion, _locations);
        if (!slice.valid || !follows_insertion) {
            if (auto *dump = std::getenv("LUISA_CORO_DUMP_ALLOCA_SCOPE");
                dump != nullptr && luisa::string_view{dump} == "1") {
                LUISA_INFO(
                    "Coroutine discriminated-prefix candidate rejected "
                    "before dataflow: payload='{}' tag='{}' counter='{}' "
                    "slice_valid={} follows_insertion={}.",
                    payload->name().value_or("<unnamed>"),
                    tag->name().value_or("<unnamed>"),
                    counter->name().value_or("<unnamed>"),
                    slice.valid, follows_insertion);
                if (slice.valid && !follows_insertion) {
                    auto insertion_location = _locations.find(insertion);
                    auto missing_locations = size_t{0u};
                    auto preceding_uses = size_t{0u};
                    auto emitted = size_t{0u};
                    for (auto *user : payload_region.users) {
                        auto location = _locations.find(user);
                        if (location == _locations.end()) {
                            ++missing_locations;
                            continue;
                        }
                        if (insertion_location == _locations.end() ||
                            location->second.block != slice.target ||
                            location->second.ordinal >=
                                insertion_location->second.ordinal) {
                            continue;
                        }
                        ++preceding_uses;
                        XIRDebugPrinter printer;
                        luisa::string user_ir;
                        printer.emit_instruction(user_ir, user);
                        LUISA_INFO(
                            "Coroutine discriminated-prefix preceding use: "
                            "payload='{}' block={} ordinal={} kind={} ir=[{}].",
                            payload->name().value_or("<unnamed>"),
                            location->second.block,
                            location->second.ordinal,
                            to_string(user->derived_instruction_tag()),
                            user_ir);
                        if (++emitted == 8u) { break; }
                    }
                    LUISA_INFO(
                        "Coroutine discriminated-prefix insertion evidence: "
                        "payload='{}' target={} insertion_present={} "
                        "insertion_block={} insertion_ordinal={} "
                        "missing_user_locations={} preceding_uses={}.",
                        payload->name().value_or("<unnamed>"),
                        slice.target,
                        insertion_location != _locations.end(),
                        insertion_location == _locations.end() ?
                            ~size_t{0u} : insertion_location->second.block,
                        insertion_location == _locations.end() ?
                            ~size_t{0u} : insertion_location->second.ordinal,
                        missing_locations, preceding_uses);
                }
            }
            return result;
        }
        auto constants = _tag_constants(tag, slice);
        // The domain is finite by construction. Refuse pathological IR rather
        // than allocating an unbounded bitset during JIT compilation.
        constexpr auto maximum_named_tags = size_t{4096u};
        if (constants.size() > maximum_named_tags) { return result; }
        TagDomain tags{std::move(constants)};
        auto relation_slots = collect_relation_slots(
            payload, tag, counter, slice, _graph,
            _resolver, _locations);
        CoroScalarRelationLiveness relation_liveness{
            _graph, slice.active, slice.target,
            relation_slots.slots, relation_slots.semantic_uses};
        auto masked_scalar_witnesses = collect_masked_scalar_witnesses(
            _graph, slice.active, _resolver, _locations);
        MaskedScalarKnownZeroAnalysis initially_known_zero{
            _graph, _resolver, _locations,
            masked_scalar_witnesses.roots, target, insertion};
        CandidateDiagnostics diagnostics{
            .detail = []() noexcept {
                auto *value = std::getenv(
                    "LUISA_CORO_DUMP_ALLOCA_SCOPE_DETAIL");
                return value != nullptr &&
                       luisa::string_view{value} == "1";
            }()};
        CandidateContext context{
            .payload = payload,
            .tag = tag,
            .counter = counter,
            .dimension = payload->type()->dimension(),
            .payload_region = payload_region,
            .slice = slice,
            .graph = _graph,
            .locations = _locations,
            .resolver = _resolver,
            .tags = tags,
            .relation_slots = relation_slots.slot_set,
            .relation_liveness = relation_liveness,
            .boolean_guards = _boolean_guards,
            .masked_scalar_witnesses = masked_scalar_witnesses.tracked,
            .diagnostics = &diagnostics};
        auto insertion_location = _locations.find(insertion);
        CoroBooleanSetManager boolean_sets;
        luisa::vector<luisa::optional<DiscriminatedState>> inputs(
            _graph.block_count());
        luisa::vector<luisa::optional<DiscriminatedState>> outputs(
            _graph.block_count());
        inputs[slice.target].emplace(
            boolean_sets, tags, context.dimension,
            masked_scalar_witnesses.tracked);
        // The placement point may be dominated by a scalar initialization
        // that precedes the counted scratch array's own C:=0 reset. Import
        // only facts independently proved by the whole-function sparse Must
        // analysis. This is an ordinary XIR dataflow fact, not an
        // application-declared lifetime boundary.
        for (auto witness : masked_scalar_witnesses.tracked) {
            auto known_zero = initially_known_zero.proves_zero(
                witness.scalar, witness.mask);
            if (diagnostics.detail) {
                LUISA_INFO(
                    "Coroutine discriminated-prefix masked seed: "
                    "scalar='{}' mask=0x{:x} zero={}.",
                    witness.scalar->name().value_or("<unnamed>"),
                    witness.mask, known_zero);
            }
            if (known_zero) {
                inputs[slice.target]
                    ->relations.assume_masked_scalar_zero(
                        witness.scalar, witness.mask);
            }
        }
        if (starts_with_empty_prefix) {
            // Prefix(P,0) and Prefix(T,0) are vacuously true. This seed is a
            // proven scalar memory fact at the placement point, including
            // stores that dominate it across a semantic suspend/resume edge;
            // it is not an application-declared lifetime boundary.
            reset_published_records(*inputs[slice.target], tags);
        }
        luisa::vector<size_t> worklist{slice.target};
        luisa::vector<uint8_t> queued(_graph.block_count(), 0u);
        queued[slice.target] = 1u;
        auto unused_read = false;
        Instruction *unused_failure = nullptr;
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            auto block_id = worklist[cursor];
            queued[block_id] = 0u;
            ++result.block_evaluation_count;
            auto state = *inputs[block_id];
            for (auto *instruction :
                 _graph.block(block_id)->instructions()) {
                auto location = _locations.find(instruction);
                if (block_id == slice.target &&
                    location != _locations.end() &&
                    location->second.ordinal <
                        insertion_location->second.ordinal) {
                    continue;
                }
                static_cast<void>(process_instruction(
                    instruction, state, context, false,
                    unused_read, unused_failure));
            }
            if (outputs[block_id] && *outputs[block_id] == state) { continue; }
            outputs[block_id] = state;
            for (auto successor : _graph.successors(block_id)) {
                if (successor == slice.target ||
                    slice.active[successor] == 0u) {
                    continue;
                }
                auto edge_state = state;
                if (!refine_edge(
                        edge_state, _graph.block(block_id),
                        _graph.block(successor), context)) {
                    continue;
                }
                auto live_indices = relation_liveness.live_in(successor);
                edge_state.relations.retain_indices(live_indices);
                retain_tag_constraints(edge_state, live_indices);
                // A control predicate can remain observable through the
                // correlation between two memory projections after its last
                // direct SSA use: for example a conditional payload write
                // and a flag set on the same edge. Existentially projecting
                // that predicate from each component separately destroys
                // this relation. Keep the common Boolean valuation until a
                // dynamic definition kills it in process_instruction; the
                // product remains finite and no source annotation is used.
                auto changed = false;
                if (!inputs[successor]) {
                    inputs[successor] = std::move(edge_state);
                    changed = true;
                } else {
                    changed = merge_state(
                        *inputs[successor], edge_state, tags);
                }
                if (changed && queued[successor] == 0u) {
                    queued[successor] = 1u;
                    worklist.emplace_back(successor);
                }
            }
        }

        auto used_payload_read = false;
        result.failing_read = nullptr;
        for (auto block_id : slice.blocks) {
            if (!inputs[block_id]) { continue; }
            auto state = *inputs[block_id];
            for (auto *instruction :
                 _graph.block(block_id)->instructions()) {
                auto location = _locations.find(instruction);
                if (block_id == slice.target &&
                    location != _locations.end() &&
                    location->second.ordinal <
                        insertion_location->second.ordinal) {
                    continue;
                }
                if (!process_instruction(
                        instruction, state, context, true,
                        used_payload_read, result.failing_read)) {
                    return result;
                }
            }
        }
        result.succeeded = used_payload_read;
        if (result.succeeded) {
            result.placement_block = target;
            result.placement_instruction = insertion;
        }
        return result;
    }

public:
    Impl(FunctionDefinition *definition,
         const CoroSemanticGraph &graph) noexcept
        : _definition{definition}, _graph{graph},
          _locations{make_instruction_locations(definition, graph)},
          _resolver{_locations, graph},
          _boolean_guards{collect_boolean_guard_values(
              graph, _resolver, _locations)} {
        for (auto *block : definition->basic_blocks()) {
            for (auto *instruction : block->instructions()) {
                if (instruction->isa<AllocaInst>()) {
                    auto *alloca = static_cast<AllocaInst *>(instruction);
                    if (alloca->is_local() && alloca->type() != nullptr &&
                        alloca->type()->tag() == Type::Tag::ARRAY) {
                        _arrays.emplace_back(alloca);
                    }
                }
            }
        }
    }

    [[nodiscard]] CoroDiscriminatedPrefixProofResult prove(
        AllocaInst *payload, BasicBlock *target,
        Instruction *insertion) const noexcept {
        CoroDiscriminatedPrefixProofResult result;
        if (payload == nullptr || payload->type() == nullptr ||
            payload->type()->tag() != Type::Tag::ARRAY ||
            payload->type()->dimension() == 0u || target == nullptr ||
            insertion == nullptr) {
            return result;
        }
        auto payload_region = collect_array_use_region(
            payload, _definition, _graph);
        if (!supported_array_region(payload, payload_region)) { return result; }
        for (auto *tag : _arrays) {
            if (!unsigned_scalar_array_compatible(tag, payload)) { continue; }
            auto tag_region = collect_array_use_region(
                tag, _definition, _graph);
            if (!supported_array_region(tag, tag_region)) { continue; }
            luisa::vector<AllocaInst *> counters;
            for (auto *user : tag_region.users) {
                if (!user->isa<StoreInst>()) { continue; }
                if (auto *counter = counter_from_current_element_store(
                        static_cast<StoreInst *>(user), tag,
                        _resolver, _locations);
                    counter != nullptr &&
                    std::find(counters.begin(), counters.end(), counter) ==
                        counters.end()) {
                    counters.emplace_back(counter);
                }
            }
            for (auto *counter : counters) {
                if (!_resolver.direct_scalar_slot(counter) ||
                    payload->type()->dimension() >
                        unsigned_type_max(counter->type())) {
                    continue;
                }
                auto &&counter_summary = _counter_summary(counter);
                if (counter_summary.resets.empty()) { continue; }
                // Starting from Prefix(P, 0), the only transfer that can
                // produce a non-empty published prefix in this abstract
                // machine is the recognized C := C + 1 publication. A
                // counter with no such transition cannot discharge a dynamic
                // P[i], i<C obligation; running the product dataflow for it
                // is therefore provably unproductive. This is a completeness
                // filter for an optimization, never a source-level hint.
                if (!counter_summary.has_publication) {
                    ++result.rejected_missing_publication_count;
                    continue;
                }
                ++result.candidate_count;
                auto candidate = _run_candidate(
                    payload, tag, counter, payload_region,
                    target, insertion);
                candidate.candidate_count = result.candidate_count;
                candidate.rejected_missing_publication_count =
                    result.rejected_missing_publication_count;
                result.block_evaluation_count +=
                    candidate.block_evaluation_count;
                if (candidate.succeeded) {
                    candidate.block_evaluation_count =
                        result.block_evaluation_count;
                    return candidate;
                }
                if (candidate.failing_read != nullptr) {
                    result.failing_read = candidate.failing_read;
                }
                StoreInst *deepest_reset = nullptr;
                for (auto *reset : counter_summary.resets) {
                    auto *block = reset->parent_block();
                    if (block == payload->parent_block() ||
                        !block_dominates_region(
                            block, payload_region, _graph)) {
                        continue;
                    }
                    if (deepest_reset == nullptr ||
                        _graph.dominates(
                            deepest_reset->parent_block(), block)) {
                        deepest_reset = reset;
                    }
                }
                if (deepest_reset == nullptr) { continue; }
                auto *outer_target = deepest_reset->parent_block();
                auto *outer_insertion = find_lifetime_insertion(
                    outer_target, deepest_reset);
                if (outer_insertion == nullptr ||
                    (outer_target == target &&
                     outer_insertion == insertion)) {
                    continue;
                }
                candidate = _run_candidate(
                    payload, tag, counter, payload_region,
                    outer_target, outer_insertion);
                candidate.candidate_count = result.candidate_count;
                candidate.rejected_missing_publication_count =
                    result.rejected_missing_publication_count;
                result.block_evaluation_count +=
                    candidate.block_evaluation_count;
                if (candidate.succeeded) {
                    candidate.block_evaluation_count =
                        result.block_evaluation_count;
                    return candidate;
                }
                if (candidate.failing_read != nullptr) {
                    result.failing_read = candidate.failing_read;
                }
            }
        }
        return result;
    }
};

CoroDiscriminatedPrefixAnalysis::CoroDiscriminatedPrefixAnalysis(
    FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept
    : _impl{luisa::make_unique<Impl>(definition, graph)} {}

CoroDiscriminatedPrefixAnalysis::~CoroDiscriminatedPrefixAnalysis() noexcept =
    default;
CoroDiscriminatedPrefixAnalysis::CoroDiscriminatedPrefixAnalysis(
    CoroDiscriminatedPrefixAnalysis &&) noexcept = default;
CoroDiscriminatedPrefixAnalysis &
CoroDiscriminatedPrefixAnalysis::operator=(
    CoroDiscriminatedPrefixAnalysis &&) noexcept = default;

CoroDiscriminatedPrefixProofResult
CoroDiscriminatedPrefixAnalysis::prove(
    AllocaInst *payload, BasicBlock *target,
    Instruction *insertion_instruction) noexcept {
    return _impl->prove(payload, target, insertion_instruction);
}

}// namespace luisa::compute::xir::detail
