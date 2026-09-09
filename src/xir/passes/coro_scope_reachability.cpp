#include "coro_scope_reachability.h"

#include <algorithm>
#include <deque>
#include <map>

#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>

#include "coro_guarded_scalar_relation.h"
#include "coro_scalar_relation_liveness.h"
#include "coro_semantic_graph.h"

namespace luisa::compute::xir::detail {
namespace {
using Set = CoroBooleanSetManager::Set;

[[nodiscard]] bool is_boolean(Value *value) noexcept {
    return value != nullptr && value->type() == Type::of<bool>();
}

// The first precision tier tracks exact scalar slots, not address guesses.
// An escaped address, GEP/cast or ordinary reference call disqualifies that
// slot completely. Loads through it are then unknown producers. In particular
// a callee's failure to MUST-write never means it cannot MAY-write.
[[nodiscard]] bool is_private_boolean_slot(AllocaInst *slot) noexcept {
    if (!is_boolean(slot) || slot->op() != AllocaOp::LOCAL) { return false; }
    for (auto *use : slot->use_list()) {
        auto *user = use->user();
        if (user->isa<LoadInst>()) {
            if (static_cast<LoadInst *>(user)->variable() != slot) { return false; }
        } else if (user->isa<StoreInst>()) {
            if (static_cast<StoreInst *>(user)->variable() != slot) { return false; }
        } else if (user->isa<CoroSuspendInst>()) {
            // Verified Extension pointer operands have explicit write effects
            // at the semantic suspension boundary, handled below.
            auto *suspend = static_cast<CoroSuspendInst *>(user);
            if (suspend->frame() == slot) { return false; }
            auto found = false;
            for (size_t i = 0u; i < suspend->extension_binding_value_count(); ++i) {
                found |= suspend->extension_binding_value(i) == slot;
            }
            if (!found) { return false; }
        } else { return false; }
    }
    return true;
}

struct BooleanTransfer {
    CoroBooleanSetManager &sets;
    const luisa::unordered_set<Value *> &slots;

    // A load's SSA result is a snapshot: never reinterpret that old result as
    // the memory slot after a later store. Every SSA definition below replaces
    // facts from its previous dynamic execution before adding its new value.
    [[nodiscard]] Set truth(Value *value, bool positive) noexcept {
        if (!is_boolean(value) || value->is_lvalue()) { return sets.universe(); }
        if (value->isa<Constant>()) {
            return static_cast<Constant *>(value)->as<bool>() == positive ?
                       sets.universe() : sets.empty_set();
        }
        switch (value->derived_value_tag()) {
            case DerivedValueTag::ARGUMENT:
            case DerivedValueTag::INSTRUCTION:
                return sets.literal(value, positive);
            default: return sets.universe();
        }
    }

    [[nodiscard]] Set assign_formula(Set state, Value *destination,
                                     Set yes, Set no) noexcept {
        // Formulas never mention their destination's old dynamic instance:
        // raw CFG has no Phi; store operands are SSA snapshots, not lvalues.
        auto old = sets.forget(state, destination);
        auto relation = sets.unite(
            sets.intersect(sets.literal(destination, true), yes),
            sets.intersect(sets.literal(destination, false), no));
        return sets.intersect(old, relation);
    }

    [[nodiscard]] Set apply(Set state, Instruction *instruction) noexcept {
        if (instruction->isa<AllocaInst>()) {
            return slots.contains(instruction) ? sets.forget(state, instruction) : state;
        }
        if (instruction->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(instruction);
            if (slots.contains(store->variable())) {
                return assign_formula(state, store->variable(),
                                      truth(store->value(), true), truth(store->value(), false));
            }
            return state;
        }
        if (instruction->isa<CoroSuspendInst>()) {
            auto *suspend = static_cast<CoroSuspendInst *>(instruction);
            for (auto &&extension : suspend->extensions()) {
                for (auto binding : extension->bindings()) {
                    // Boundary-only bindings may also be written by an
                    // immediate lowering handler. Storage lifetime is not a
                    // no-write contract for executable reachability.
                    if (binding.access == CoroSuspendBindingAccess::read) { continue; }
                    auto *value = suspend->extension_binding_value(binding.index);
                    if (slots.contains(value)) { state = sets.forget(state, value); }
                }
            }
            return state;
        }
        if (!is_boolean(instruction) || instruction->is_lvalue()) { return state; }
        if (instruction->isa<LoadInst>()) {
            auto *load = static_cast<LoadInst *>(instruction);
            if (slots.contains(load->variable())) {
                return sets.assign(state, instruction, load->variable(), true, luisa::nullopt);
            }
        } else if (instruction->isa<ArithmeticInst>()) {
            auto *arithmetic = static_cast<ArithmeticInst *>(instruction);
            auto count = arithmetic->operand_count();
            if (count == 1u && is_boolean(arithmetic->operand(0u)) &&
                arithmetic->op() == ArithmeticOp::UNARY_BIT_NOT) {
                return assign_formula(state, instruction, truth(arithmetic->operand(0u), false),
                                      truth(arithmetic->operand(0u), true));
            }
            if (count == 2u && is_boolean(arithmetic->operand(0u)) &&
                is_boolean(arithmetic->operand(1u))) {
                auto a = truth(arithmetic->operand(0u), true);
                auto na = truth(arithmetic->operand(0u), false);
                auto b = truth(arithmetic->operand(1u), true);
                auto nb = truth(arithmetic->operand(1u), false);
                switch (arithmetic->op()) {
                    case ArithmeticOp::BINARY_BIT_AND:
                        return assign_formula(state, instruction, sets.intersect(a, b), sets.unite(na, nb));
                    case ArithmeticOp::BINARY_BIT_OR:
                        return assign_formula(state, instruction, sets.unite(a, b), sets.intersect(na, nb));
                    case ArithmeticOp::BINARY_BIT_XOR:
                    case ArithmeticOp::BINARY_EQUAL:
                    case ArithmeticOp::BINARY_NOT_EQUAL: {
                        auto same = sets.unite(sets.intersect(a, b), sets.intersect(na, nb));
                        auto different = sets.unite(sets.intersect(a, nb), sets.intersect(na, b));
                        auto equal = arithmetic->op() == ArithmeticOp::BINARY_EQUAL;
                        return assign_formula(state, instruction, equal ? same : different,
                                              equal ? different : same);
                    }
                    default: break;
                }
            }
        }
        // A fresh unknown value includes both outcomes. This includes resource
        // reads, calls, comparisons outside the Boolean domain and special ops.
        return sets.forget(state, instruction);
    }
};
}// namespace

CoroScopeReachability analyze_coro_scope_reachability(
    FunctionDefinition *definition) noexcept {
    CoroScopeReachability result;
    CoroSemanticGraph graph{definition};
    if (!graph.valid()) { return result; }

    luisa::unordered_set<Value *> slots;
    luisa::vector<Value *> predicates;
    luisa::unordered_map<size_t, uint32_t> resume_tokens;
    for (size_t block_id = 0u; block_id < graph.block_count(); ++block_id) {
        for (auto *instruction : graph.block(block_id)->instructions()) {
            if (instruction->isa<CoroResumeInst>()) {
                resume_tokens.emplace(block_id, static_cast<CoroResumeInst *>(instruction)->token());
            }
            if (instruction->isa<AllocaInst>()) {
                if (is_private_boolean_slot(static_cast<AllocaInst *>(instruction))) {
                    slots.emplace(instruction);
                    predicates.emplace_back(instruction);
                }
            } else if (is_boolean(instruction) && !instruction->is_lvalue()) {
                predicates.emplace_back(instruction);
            }
        }
    }
    for (auto *argument : definition->arguments()) {
        if (is_boolean(argument) && !argument->is_lvalue()) { predicates.emplace_back(argument); }
    }

    CoroBooleanSemanticValues uses, definitions;
    for (size_t block_id = 0u; block_id < graph.block_count(); ++block_id) {
        for (auto *instruction : graph.block(block_id)->instructions()) {
            if (is_boolean(instruction) && !instruction->is_lvalue()) {
                definitions[instruction].emplace_back(instruction);
            }
            for (auto *use : instruction->operand_uses()) {
                auto *operand = use->value();
                if (is_boolean(operand) && !operand->is_lvalue() && !operand->isa<Constant>()) {
                    uses[instruction].emplace_back(operand);
                }
            }
            if (instruction->isa<AllocaInst>() && slots.contains(instruction)) {
                definitions[instruction].emplace_back(instruction);
            } else if (instruction->isa<LoadInst>()) {
                auto *pointer = static_cast<LoadInst *>(instruction)->variable();
                if (slots.contains(pointer)) { uses[instruction].emplace_back(pointer); }
            } else if (instruction->isa<StoreInst>()) {
                auto *pointer = static_cast<StoreInst *>(instruction)->variable();
                if (slots.contains(pointer)) { definitions[instruction].emplace_back(pointer); }
            } else if (instruction->isa<CoroSuspendInst>()) {
                auto *suspend = static_cast<CoroSuspendInst *>(instruction);
                for (auto &&extension : suspend->extensions()) {
                    for (auto binding : extension->bindings()) {
                        if (binding.access != CoroSuspendBindingAccess::read) {
                            auto *value = suspend->extension_binding_value(binding.index);
                            if (slots.contains(value)) { definitions[instruction].emplace_back(value); }
                        }
                    }
                }
            }
        }
    }
    luisa::vector<uint8_t> active(graph.block_count(), 1u);
    CoroBooleanPredicateLiveness liveness{
        graph, active, graph.block_count(), predicates, uses, definitions};
    CoroBooleanSetManager sets;
    BooleanTransfer transfer{sets, slots};
    struct State {
        uint32_t owner;
        size_t block;
        Set incoming{CoroBooleanSetManager::empty_set()};
        uint8_t branch_arms{0u};
        bool queued{false};
    };
    std::map<std::pair<uint32_t, size_t>, size_t> state_ids;
    luisa::vector<State> states;
    std::deque<size_t> worklist;
    auto enqueue = [&](uint32_t owner, size_t block, Set incoming) noexcept {
        if (CoroBooleanSetManager::is_empty(incoming)) { return; }
        if (auto resume = resume_tokens.find(block); resume != resume_tokens.end()) {
            owner = resume->second;
        }
        // Remove branch history only after its outgoing states were refined.
        // This also projects values that die on only this incoming edge.
        auto live = liveness.live_in(block);
        for (auto *predicate : sets.support(incoming)) {
            if (std::find(live.begin(), live.end(), predicate) == live.end()) {
                incoming = sets.forget(incoming, predicate);
            }
        }
        auto [iter, inserted] = state_ids.try_emplace({owner, block}, states.size());
        if (inserted) { states.push_back({owner, block}); }
        auto id = iter->second;
        auto joined = sets.unite(states[id].incoming, incoming);
        if (joined != states[id].incoming) {
            states[id].incoming = joined;
            if (!states[id].queued) {
                states[id].queued = true;
                worklist.emplace_back(id);
            }
        }
    };
    enqueue(0u, graph.block_id(definition->body_block()), sets.universe());
    while (!worklist.empty()) {
        auto id = worklist.front();
        worklist.pop_front();
        states[id].queued = false;
        auto owner = states[id].owner;
        auto block_id = states[id].block;
        auto *block = graph.block(block_id);
        auto outgoing = states[id].incoming;
        for (auto *instruction : block->instructions()) {
            outgoing = transfer.apply(outgoing, instruction);
            // Existential projection after the final use preserves every
            // concrete valuation without retaining unrelated long-block
            // history. Keep terminator conditions through branch refinement.
            if (!instruction->is_terminator()) {
                for (auto *predicate : liveness.dead_after(instruction)) {
                    outgoing = sets.forget(outgoing, predicate);
                }
            }
        }
        auto *term = block->terminator();
        if (term->isa<ConditionalBranchInst>()) {
            auto *branch = static_cast<ConditionalBranchInst *>(term);
            for (auto positive : {true, false}) {
                auto next = sets.intersect(outgoing, transfer.truth(branch->condition(), positive));
                if (!sets.is_empty(next)) {
                    states[id].branch_arms |= positive ? 1u : 2u;
                    auto *successor = positive ? branch->true_block() : branch->false_block();
                    enqueue(owner, graph.block_id(successor), next);
                }
            }
        } else {
            for (auto successor : graph.successors(block_id)) { enqueue(owner, successor, outgoing); }
        }
    }
    for (const auto &state : states) {
        auto *block = graph.block(state.block);
        auto &scope = result.scopes[state.owner];
        scope.blocks.emplace(block);
        if (state.branch_arms == 1u || state.branch_arms == 2u) {
            auto *branch = static_cast<ConditionalBranchInst *>(block->terminator());
            scope.selected_successors.emplace(
                block, state.branch_arms == 1u ? branch->true_block() : branch->false_block());
        }
    }
    result.valid = true;
    result.widened = sets.widened();
    result.state_count = states.size();
    return result;
}
}// namespace luisa::compute::xir::detail
