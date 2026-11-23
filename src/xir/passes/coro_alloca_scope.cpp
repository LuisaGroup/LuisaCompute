#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <queue>
#include <utility>

#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/coro_alloca_scope.h>
#include <luisa/xir/passes/pointer_usage.h>
#include <luisa/core/stl/algorithm.h>

#include "coro_frame_access.h"
#include "coro_discriminated_prefix.h"
#include "coro_initialized_prefix.h"
#include "coro_predicate_analysis.h"
#include "coro_semantic_graph.h"

namespace luisa::compute::xir {

namespace detail {

namespace coro_alloca_scope_detail {

struct AllocaUseRegion {
    bool valid{true};
    bool contains_phi_use{false};
    luisa::unordered_set<Value *> pointers;
    luisa::unordered_set<Instruction *> users;
    luisa::vector<BasicBlock *> blocks;
};

struct InstructionLocation {
    size_t block_id;
    size_t ordinal;
};

using InstructionLocationMap =
    luisa::unordered_map<Instruction *, InstructionLocation>;

[[nodiscard]] InstructionLocationMap make_instruction_locations(
    FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept {
    InstructionLocationMap locations;
    for (auto *block : definition->basic_blocks()) {
        auto block_id = graph.block_id(block);
        auto ordinal = size_t{0u};
        for (auto *instruction : block->instructions()) {
            locations.emplace(
                instruction,
                InstructionLocation{block_id, ordinal++});
        }
    }
    return locations;
}

[[nodiscard]] AllocaUseRegion collect_alloca_use_region(
    AllocaInst *alloca, FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept {
    AllocaUseRegion result;
    luisa::unordered_set<BasicBlock *> seen_blocks;
    luisa::vector<Value *> worklist{alloca};
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
                !graph.contains(block)) {
                result.valid = false;
                break;
            }
            result.users.emplace(instruction);
            if (seen_blocks.emplace(block).second) {
                result.blocks.emplace_back(block);
            }
            if (instruction->isa<PhiInst>()) {
                result.contains_phi_use = true;
                continue;
            }
            if (instruction->isa<GEPInst>() &&
                static_cast<GEPInst *>(instruction)->base() == pointer) {
                worklist.emplace_back(instruction);
            }
        }
    }
    return result;
}

struct InsertionPoint {
    Instruction *instruction{nullptr};
    bool follows_alloca{false};
    bool has_gap_after_alloca{false};
};

[[nodiscard]] bool instruction_strictly_precedes_linear(
    Instruction *before, Instruction *after) noexcept {
    if (before == nullptr || after == nullptr ||
        before->parent_block() != after->parent_block()) {
        return false;
    }
    for (auto *instruction :
         before->parent_block()->instructions()) {
        if (instruction == before) { return true; }
        if (instruction == after) { return false; }
    }
    return false;
}

[[nodiscard]] bool instruction_immediately_precedes(
    Instruction *before, Instruction *after) noexcept {
    if (before == nullptr || after == nullptr ||
        before->parent_block() != after->parent_block()) {
        return false;
    }
    return before->next() == after;
}

[[nodiscard]] InsertionPoint find_latest_insertion_point_linear(
    BasicBlock *target, AllocaInst *alloca,
    const luisa::unordered_set<Instruction *> &users) noexcept {
    InsertionPoint result;
    auto saw_alloca = false;
    for (auto *instruction : target->instructions()) {
        if (instruction == alloca) {
            saw_alloca = true;
            continue;
        }
        if (users.contains(instruction)) {
            result.instruction = instruction;
            result.follows_alloca = saw_alloca;
            return result;
        }
        if (instruction->is_terminator()) {
            result.instruction = instruction;
            result.follows_alloca = saw_alloca;
            return result;
        }
        if (saw_alloca) { result.has_gap_after_alloca = true; }
    }
    return result;
}

// The pass freezes its candidate set before mutation and moves only the
// candidate alloca and, when proved unique, its whole-object defining store.
// Neither instruction can be an observation of a different valid candidate.
// Consequently the relative order of every pair queried for an unprocessed
// candidate is invariant under earlier contractions. Snapshot ordinals answer
// strict-order queries exactly; current intrusive-list adjacency answers the
// one property that can change when unrelated nodes are inserted in a gap.
[[nodiscard]] bool instruction_strictly_precedes(
    Instruction *before, Instruction *after,
    const InstructionLocationMap &locations,
    bool verify_instruction_order,
    size_t &instruction_order_query_count) noexcept {
    ++instruction_order_query_count;
    auto result = false;
    if (before != nullptr && after != nullptr &&
        before->parent_block() == after->parent_block()) {
        auto before_iter = locations.find(before);
        auto after_iter = locations.find(after);
        result = before_iter != locations.end() &&
                 after_iter != locations.end() &&
                 before_iter->second.block_id ==
                     after_iter->second.block_id &&
                 before_iter->second.ordinal <
                     after_iter->second.ordinal;
    }
    if (verify_instruction_order) {
        LUISA_ASSERT(
            result == instruction_strictly_precedes_linear(
                          before, after),
            "Snapshot instruction order disagrees with the current "
            "intrusive list.");
    }
    return result;
}

[[nodiscard]] InsertionPoint find_latest_insertion_point(
    BasicBlock *target, AllocaInst *alloca,
    const luisa::unordered_set<Instruction *> &users,
    const InstructionLocationMap &locations,
    bool verify_instruction_order,
    size_t &instruction_order_query_count,
    size_t &placement_user_inspection_count) noexcept {
    ++instruction_order_query_count;
    InsertionPoint result;
    if (target == nullptr) { return result; }
    auto *insertion = static_cast<Instruction *>(
        target->terminator());
    auto insertion_iter = locations.find(insertion);
    if (insertion == nullptr ||
        insertion_iter == locations.end()) {
        return result;
    }
    auto insertion_ordinal = insertion_iter->second.ordinal;
    for (auto *user : users) {
        ++placement_user_inspection_count;
        if (user == nullptr || user->parent_block() != target) {
            continue;
        }
        auto iter = locations.find(user);
        if (iter == locations.end() ||
            iter->second.block_id !=
                insertion_iter->second.block_id) {
            continue;
        }
        if (iter->second.ordinal < insertion_ordinal) {
            insertion = user;
            insertion_ordinal = iter->second.ordinal;
        }
    }
    result.instruction = insertion;
    if (alloca != nullptr && alloca->parent_block() == target) {
        auto alloca_iter = locations.find(alloca);
        result.follows_alloca =
            alloca_iter != locations.end() &&
            alloca_iter->second.block_id ==
                insertion_iter->second.block_id &&
            alloca_iter->second.ordinal < insertion_ordinal;
        result.has_gap_after_alloca =
            result.follows_alloca && alloca->next() != insertion;
    }
    if (verify_instruction_order) {
        auto oracle = find_latest_insertion_point_linear(
            target, alloca, users);
        LUISA_ASSERT(
            result.instruction == oracle.instruction &&
                result.follows_alloca == oracle.follows_alloca &&
                result.has_gap_after_alloca ==
                    oracle.has_gap_after_alloca,
            "Snapshot insertion-point query disagrees with the current "
            "intrusive list.");
    }
    return result;
}

struct FirstDefinitionPlan {
    StoreInst *definition{nullptr};
    BasicBlock *target{nullptr};
    Instruction *insertion_instruction{nullptr};
};

[[nodiscard]] FirstDefinitionPlan plan_first_definition_delay(
    AllocaInst *alloca, const AllocaUseRegion &region,
    const CoroSemanticGraph &graph,
    const InstructionLocationMap &instruction_locations,
    bool verify_instruction_order,
    size_t &instruction_order_query_count,
    size_t &placement_user_inspection_count) noexcept {
    FirstDefinitionPlan plan;
    StoreInst *definition = nullptr;
    luisa::unordered_set<Instruction *> observations;
    luisa::vector<BasicBlock *> observation_blocks;
    luisa::unordered_set<BasicBlock *> seen_blocks;

    // Formal single-definition domain:
    //   * exactly one full-object store defines the local;
    //   * every other pointer use is a typed projection or observation;
    //   * no reference/atomic/unknown operation can expose store timing.
    // Moving that store with the lifetime start may execute it more often in
    // a loop, but with no second write version every load still observes the
    // same dominating SSA value as in the original program.
    for (auto *user : region.users) {
        if (user->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(user);
            if (definition != nullptr || store->variable() != alloca) {
                return plan;
            }
            definition = store;
            continue;
        }
        if (!user->isa<GEPInst>() && !user->isa<LoadInst>()) {
            return plan;
        }
        observations.emplace(user);
        if (seen_blocks.emplace(user->parent_block()).second) {
            observation_blocks.emplace_back(user->parent_block());
        }
    }
    if (definition == nullptr || observations.empty()) { return plan; }

    auto *target = graph.nearest_common_dominator(
        luisa::span{observation_blocks});
    auto *source = alloca->parent_block();
    auto *definition_block = definition->parent_block();
    if (target == nullptr || source == nullptr ||
        definition_block == nullptr ||
        !graph.dominates(source, target) ||
        !graph.dominates(definition_block, target)) {
        return plan;
    }
    auto insertion = find_latest_insertion_point(
        target, alloca, observations, instruction_locations,
        verify_instruction_order, instruction_order_query_count,
        placement_user_inspection_count);
    if (insertion.instruction == nullptr) { return plan; }
    if (definition_block == target &&
        !instruction_strictly_precedes(
            definition, insertion.instruction,
            instruction_locations, verify_instruction_order,
            instruction_order_query_count)) {
        return plan;
    }
    if (source == target &&
        instruction_immediately_precedes(alloca, definition) &&
        instruction_immediately_precedes(
            definition, insertion.instruction)) {
        return plan;
    }

    // The store is moved, not recomputed. Its SSA operand must therefore be
    // available at the new point. Function arguments and constants are
    // globally available; an instruction operand must dominate the target
    // and textually precede the insertion point when defined there.
    auto *value = definition->value();
    if (value == nullptr) { return plan; }
    if (value->isa<Instruction>()) {
        auto *value_instruction = static_cast<Instruction *>(value);
        auto *value_block = value_instruction->parent_block();
        if (value_block == nullptr ||
            !graph.dominates(value_block, target) ||
            (value_block == target &&
             !instruction_strictly_precedes(
                 value_instruction, insertion.instruction,
                 instruction_locations, verify_instruction_order,
                 instruction_order_query_count))) {
            return plan;
        }
    } else if (value->derived_value_tag() !=
                   DerivedValueTag::ARGUMENT &&
               value->derived_value_tag() !=
                   DerivedValueTag::CONSTANT &&
               value->derived_value_tag() !=
                   DerivedValueTag::UNDEFINED) {
        // A special register is evaluated at its use site and may change at a
        // continuation boundary (notably the current coroutine token). It is
        // not an SSA snapshot unless materialized by an instruction.
        return plan;
    }
    plan.definition = definition;
    plan.target = target;
    plan.insertion_instruction = insertion.instruction;
    return plan;
}

void apply_first_definition_plan(
    AllocaInst *alloca,
    const FirstDefinitionPlan &plan) noexcept {
    auto alloca_owner = alloca->remove_self();
    auto definition_owner = plan.definition->remove_self();
    auto *moved_alloca = plan.insertion_instruction->insert_before_self(
        std::move(alloca_owner));
    auto *moved_definition =
        plan.insertion_instruction->insert_before_self(
            std::move(definition_owner));
    LUISA_DEBUG_ASSERT(
        moved_alloca == alloca &&
            moved_definition == plan.definition,
        "First-definition delay changed XIR instruction identity.");
}

enum class LifetimeEventKind : uint8_t {
    redefine_pointer,
    store,
    read
};

struct LifetimeEvent {
    LifetimeEventKind kind;
    Value *pointer;
    Instruction *instruction;
};

struct LifetimeFactLayout {
    static constexpr size_t inline_capacity = 8u;
    std::array<size_t, inline_capacity> inline_atoms{};
    std::array<Value *, inline_capacity> inline_pointers{};
    size_t inline_atom_count{0u};
    size_t inline_pointer_count{0u};
    luisa::unordered_map<size_t, size_t> atom_facts_map;
    luisa::unordered_map<Value *, size_t> pointer_facts_map;
    size_t atom_count{0u};
    size_t pointer_count{0u};
    size_t fact_count{0u};

    [[nodiscard]] size_t find_atom(size_t atom) const noexcept {
        if (inline_atom_count != 0u) {
            for (size_t i = 0u; i < inline_atom_count; ++i) {
                if (inline_atoms[i] == atom) { return i; }
            }
            return ~size_t{0u};
        }
        if (atom_facts_map.empty()) { return ~size_t{0u}; }
        auto iter = atom_facts_map.find(atom);
        return iter == atom_facts_map.end() ? ~size_t{0u} : iter->second;
    }

    [[nodiscard]] size_t find_pointer(Value *pointer) const noexcept {
        if (inline_pointer_count != 0u) {
            for (size_t i = 0u; i < inline_pointer_count; ++i) {
                if (inline_pointers[i] == pointer) { return atom_count + i; }
            }
            return ~size_t{0u};
        }
        if (pointer_facts_map.empty()) { return ~size_t{0u}; }
        auto iter = pointer_facts_map.find(pointer);
        return iter == pointer_facts_map.end() ? ~size_t{0u} : iter->second;
    }
};

struct LifetimeProofResult {
    bool succeeded{false};
    bool guarded{false};
    bool guarded_budget_exhausted{false};
    Instruction *failing_read{nullptr};
    size_t failing_predicate_count{0u};
    size_t block_evaluation_count{0u};
    size_t guarded_state_evaluation_count{0u};
    size_t predicate_widening_count{0u};
};

struct LifetimeProofTimings {
    double problem_ms{0.0};
    double slice_ms{0.0};
    double layout_ms{0.0};
    double events_ms{0.0};
    double event_order_ms{0.0};
    double event_transfer_ms{0.0};
    double unconditional_ms{0.0};
    double guarded_ms{0.0};
};

class LifetimeFactState {
private:
    uint64_t _inline_bits{0u};
    luisa::vector<uint64_t> _overflow;
    uint32_t _count{0u};

    [[nodiscard]] static uint64_t _full_mask(size_t bits) noexcept {
        if (bits == 0u) { return 0u; }
        if (bits >= 64u) { return ~uint64_t{0u}; }
        return (uint64_t{1u} << bits) - uint64_t{1u};
    }

public:
    LifetimeFactState() noexcept = default;
    explicit LifetimeFactState(size_t count, bool fill_ones = false) noexcept
        : _count{static_cast<uint32_t>(count)} {
        if (count <= 64u) {
            _inline_bits = fill_ones ? _full_mask(count) : uint64_t{0u};
        } else {
            const auto words = (count + 63u) / 64u;
            _overflow.assign(words, fill_ones ? ~uint64_t{0u} : uint64_t{0u});
            if (fill_ones && (count & 63u) != 0u) {
                _overflow.back() = _full_mask(count & 63u);
            }
        }
    }

    [[nodiscard]] size_t size() const noexcept { return _count; }

    [[nodiscard]] bool get(size_t i) const noexcept {
        if (_count <= 64u) {
            return ((_inline_bits >> i) & uint64_t{1u}) != 0u;
        }
        return ((_overflow[i >> 6u] >> (i & 63u)) & uint64_t{1u}) != 0u;
    }

    void set(size_t i) noexcept {
        if (_count <= 64u) {
            _inline_bits |= (uint64_t{1u} << i);
        } else {
            _overflow[i >> 6u] |= (uint64_t{1u} << (i & 63u));
        }
    }

    void clear(size_t i) noexcept {
        if (_count <= 64u) {
            _inline_bits &= ~(uint64_t{1u} << i);
        } else {
            _overflow[i >> 6u] &= ~(uint64_t{1u} << (i & 63u));
        }
    }

    void set_from_mask(uint64_t mask) noexcept {
        LUISA_DEBUG_ASSERT(_count <= 64u, "set_from_mask requires inline fact count.");
        _inline_bits = mask & _full_mask(_count);
    }

    [[nodiscard]] bool intersect_with(const LifetimeFactState &other) noexcept {
        LUISA_DEBUG_ASSERT(_count == other._count,
                           "Mismatched lifetime fact domains.");
        if (_count <= 64u) {
            const auto next = _inline_bits & other._inline_bits;
            const auto changed = next != _inline_bits;
            _inline_bits = next;
            return changed;
        }
        auto changed = false;
        for (size_t w = 0u; w < _overflow.size(); ++w) {
            const auto next = _overflow[w] & other._overflow[w];
            changed |= next != _overflow[w];
            _overflow[w] = next;
        }
        return changed;
    }

    [[nodiscard]] bool operator==(const LifetimeFactState &other) const noexcept {
        if (_count != other._count) { return false; }
        if (_count <= 64u) { return _inline_bits == other._inline_bits; }
        return _overflow == other._overflow;
    }

    [[nodiscard]] bool operator!=(const LifetimeFactState &other) const noexcept {
        return !(*this == other);
    }
};

struct LifetimeProofWorkspace {
    struct SccDfsFrame {
        size_t node;
        size_t edge_idx;
    };

    luisa::vector<uint32_t> active_epoch;
    luisa::vector<size_t> active_index;
    luisa::vector<uint32_t> queued_epoch;
    luisa::vector<uint32_t> pred_epoch;
    luisa::vector<uint32_t> pred_count;
    uint32_t active_gen{0u};
    uint32_t queued_gen{0u};
    uint32_t pred_gen{0u};
    luisa::unordered_map<size_t, luisa::vector<uint8_t>> scc_in_cycle_cache;
    luisa::vector<size_t> scc_index;
    luisa::vector<size_t> scc_lowlink;
    luisa::vector<uint8_t> scc_on_stack;
    luisa::vector<size_t> scc_tarjan_stack;
    luisa::vector<SccDfsFrame> scc_dfs_stack;

    LifetimeProofWorkspace(size_t block_count,
                           size_t predicate_term_count) noexcept
        : active_epoch(block_count, 0u),
          active_index(block_count, ~size_t{0u}),
          queued_epoch(block_count, 0u),
          pred_epoch(predicate_term_count, 0u),
          pred_count(predicate_term_count, 0u) {}

    [[nodiscard]] uint32_t next_active_gen() noexcept {
        if (++active_gen == 0u) {
            std::fill(active_epoch.begin(), active_epoch.end(), 0u);
            active_gen = 1u;
        }
        return active_gen;
    }

    [[nodiscard]] uint32_t next_queued_gen() noexcept {
        if (++queued_gen == 0u) {
            std::fill(queued_epoch.begin(), queued_epoch.end(), 0u);
            queued_gen = 1u;
        }
        return queued_gen;
    }

    [[nodiscard]] uint32_t next_pred_gen() noexcept {
        if (++pred_gen == 0u) {
            std::fill(pred_epoch.begin(), pred_epoch.end(), 0u);
            pred_gen = 1u;
        }
        return pred_gen;
    }

    [[nodiscard]] bool is_active(size_t block_id) const noexcept {
        return active_epoch[block_id] == active_gen;
    }

    [[nodiscard]] luisa::span<const uint8_t>
    block_in_cycle_under_target(size_t target_id,
                                const CoroSemanticGraph &graph) noexcept {
        const auto n = graph.block_count();
        if (target_id >= n) { return {}; }
        if (auto iter = scc_in_cycle_cache.find(target_id);
            iter != scc_in_cycle_cache.end()) {
            return iter->second;
        }
        auto &in_cycle = scc_in_cycle_cache[target_id];
        in_cycle.assign(n, uint8_t{0u});
        scc_index.assign(n, ~size_t{0u});
        scc_lowlink.assign(n, ~size_t{0u});
        scc_on_stack.assign(n, uint8_t{0u});
        scc_tarjan_stack.clear();
        scc_dfs_stack.clear();
        auto next_index = size_t{0u};
        for (size_t start = 0u; start < n; ++start) {
            if (scc_index[start] != ~size_t{0u} ||
                start == target_id ||
                !graph.dominates_id(target_id, start)) {
                continue;
            }
            scc_index[start] = next_index;
            scc_lowlink[start] = next_index;
            ++next_index;
            scc_on_stack[start] = 1u;
            scc_tarjan_stack.emplace_back(start);
            scc_dfs_stack.emplace_back(SccDfsFrame{start, 0u});
            while (!scc_dfs_stack.empty()) {
                const auto u = scc_dfs_stack.back().node;
                const auto succs = graph.successors(u);
                if (scc_dfs_stack.back().edge_idx < succs.size()) {
                    const auto v = succs[scc_dfs_stack.back().edge_idx++];
                    if (v == u) {
                        in_cycle[u] = 1u;
                        continue;
                    }
                    if (v == target_id || !graph.dominates_id(target_id, v)) {
                        continue;
                    }
                    if (scc_index[v] == ~size_t{0u}) {
                        scc_index[v] = next_index;
                        scc_lowlink[v] = next_index;
                        ++next_index;
                        scc_on_stack[v] = 1u;
                        scc_tarjan_stack.emplace_back(v);
                        scc_dfs_stack.emplace_back(SccDfsFrame{v, 0u});
                    } else if (scc_on_stack[v] != 0u) {
                        scc_lowlink[u] = std::min(scc_lowlink[u], scc_index[v]);
                    }
                } else {
                    if (scc_lowlink[u] == scc_index[u]) {
                        const auto first = scc_tarjan_stack.back();
                        scc_tarjan_stack.pop_back();
                        scc_on_stack[first] = 0u;
                        if (first != u) {
                            in_cycle[first] = 1u;
                            for (;;) {
                                const auto w = scc_tarjan_stack.back();
                                scc_tarjan_stack.pop_back();
                                scc_on_stack[w] = 0u;
                                in_cycle[w] = 1u;
                                if (w == u) { break; }
                            }
                        }
                    }
                    scc_dfs_stack.pop_back();
                    if (!scc_dfs_stack.empty()) {
                        const auto parent = scc_dfs_stack.back().node;
                        scc_lowlink[parent] =
                            std::min(scc_lowlink[parent], scc_lowlink[u]);
                    }
                }
            }
        }
        return in_cycle;
    }
};

struct BlockEventRange {
    uint32_t begin{0u};
    uint32_t count{0u};
};

struct LifetimeProofProblem {
    bool valid{false};
    bool distributive_facts{false};
    size_t target_id{0u};
    LifetimeFactLayout layout;
    luisa::vector<size_t> active_blocks;
    // Parallel to active_blocks. Keeping the event domain sparse is
    // essential after SROA: a large coroutine may contain thousands of CFG
    // blocks while one scalar local touches only a handful of them.
    luisa::vector<LifetimeEvent> all_events;
    luisa::vector<BlockEventRange> block_events;

    [[nodiscard]] luisa::span<const LifetimeEvent>
    events_for_active(size_t active_idx) const noexcept {
        const auto range = block_events[active_idx];
        return range.count == 0u ?
                   luisa::span<const LifetimeEvent>{} :
                   luisa::span<const LifetimeEvent>{
                       all_events.data() + range.begin, range.count};
    }
};

struct ReferenceArgumentEffect {
    // True iff some execution can observe the incoming object before a
    // complete definition of the observed fields. This is a May property.
    bool may_read_prior_value{true};
    // True iff every normally returning execution has completely defined the
    // object. This is a Must property. Non-returning exits do not weaken it.
    bool fully_defines_on_return{false};
};

class ReferenceArgumentEffectAnalysis {
private:
    luisa::unordered_map<Argument *, ReferenceArgumentEffect> _effects;
    luisa::unordered_set<FunctionDefinition *> _analyzed;
    size_t _analysis_count{0u};
    double _analysis_ms{0.0};

private:
    void _analyze(FunctionDefinition *definition) noexcept {
        if (definition == nullptr || definition->body_block() == nullptr ||
            !_analyzed.emplace(definition).second) {
            return;
        }
        auto begin = std::chrono::steady_clock::now();
        ++_analysis_count;
        PointerUsageAnalysisInfo analysis_info;

        const auto analyze = [&]() noexcept {
            // PointerUsageAnalysis is a field-sensitive pair of finite
            // dataflow problems. At function entry, LIVE is exactly the set
            // of aggregate leaves that may be read before a definite
            // overwrite. At each normal return, KILL is exactly the set
            // definitely written on every path to that return. Calls inside
            // the callee remain conservative opaque read/writes, so an
            // unsupported or recursive dependency can only make this summary
            // less precise, never unsound.
            luisa::vector<Value *> reference_arguments;
            for (auto *argument : definition->arguments()) {
                if (argument->is_reference()) {
                    _effects.try_emplace(
                        argument, ReferenceArgumentEffect{});
                    reference_arguments.emplace_back(argument);
                }
            }
            if (reference_arguments.empty()) { return; }

            // Pointer usage is a product lattice over pointer views. Reference
            // summaries query only formal-reference coordinates, so solving
            // those coordinates is exactly equivalent to solving every local
            // alloca/GEP view and projecting afterward. Pointer discovery and
            // malformed-use validation remain whole-function and therefore
            // fail closed.
            PointerUsageAnalysis analysis;
            analysis_info = analysis.analyze(
                definition, luisa::span<Value *const>{reference_arguments});
            if (!analysis_info.succeeded()) { return; }
            // The summary pass below is read-only. Validate the captured IR
            // version once, then query the immutable block-result table
            // directly; validating the whole instruction snapshot for every
            // argument and return block would turn extraction into
            // O(queries * instructions).
            LUISA_ASSERT(
                analysis.is_current(),
                "Fresh pointer-usage analysis is unexpectedly stale.");

            for (auto *argument : definition->arguments()) {
                if (!argument->is_reference()) { continue; }
                auto *entry_block = analysis.current_block_usage(
                    definition->body_block());
                auto entry_iter = entry_block == nullptr ?
                                      PointerUsageMap::const_iterator{} :
                                      entry_block->in.find(argument);
                if (entry_block == nullptr ||
                    entry_iter == entry_block->in.end()) {
                    continue;
                }
                auto *entry = entry_iter->second.get();

                auto has_normal_return = false;
                auto defines_at_every_return = true;
                for (auto *block : definition->basic_blocks()) {
                    if (!block->is_terminated() ||
                        !block->terminator()->isa<ReturnInst>()) {
                        continue;
                    }
                    auto *block_usage =
                        analysis.current_block_usage(block);
                    auto usage_iter = block_usage == nullptr ?
                                          PointerUsageMap::const_iterator{} :
                                          block_usage->out.find(argument);
                    if (block_usage == nullptr ||
                        usage_iter == block_usage->out.end()) {
                        continue;
                    }
                    auto *usage = usage_iter->second.get();
                    has_normal_return = true;
                    defines_at_every_return &= usage->kill.access().all();
                }
                _effects[argument] = ReferenceArgumentEffect{
                    .may_read_prior_value = entry->live.access().any(),
                    .fully_defines_on_return =
                        has_normal_return && defines_at_every_return};
            }
        };
        analyze();
        auto elapsed_ms =
            std::chrono::duration<double, std::milli>{
                std::chrono::steady_clock::now() - begin}
                .count();
        _analysis_ms += elapsed_ms;
        if (auto profile = std::getenv(
                "LUISA_CORO_PROFILE_COMPILATION");
            profile != nullptr && luisa::string_view{profile} == "1") {
            LUISA_INFO(
                "Coroutine reference-effect summary: function='{}' "
                "tracked={} materialized={} blocks={} conservative={} "
                "invalid={} time={:.3f} ms.",
                definition->name().value_or("<unnamed>"),
                analysis_info.tracked_pointer_count,
                analysis_info.materialized_pointer_count,
                analysis_info.analyzed_block_count,
                analysis_info.conservative_access_count,
                analysis_info.invalid_access_count,
                elapsed_ms);
        }
    }

public:
    [[nodiscard]] ReferenceArgumentEffect effect(
        Argument *argument) noexcept {
        if (argument == nullptr || !argument->is_reference()) {
            return {};
        }
        _analyze(argument->parent_function()->definition());
        if (auto iter = _effects.find(argument); iter != _effects.end()) {
            return iter->second;
        }
        return {};
    }

    [[nodiscard]] size_t analysis_count() const noexcept {
        return _analysis_count;
    }

    [[nodiscard]] double analysis_ms() const noexcept {
        return _analysis_ms;
    }
};

[[nodiscard]] LifetimeFactLayout make_lifetime_fact_layout(
    luisa::span<const size_t> atom_indices,
    const AllocaUseRegion &region) noexcept {
    LifetimeFactLayout layout;
    if (atom_indices.size() <= LifetimeFactLayout::inline_capacity) {
        for (auto atom : atom_indices) {
            if (layout.find_atom(atom) == ~size_t{0u}) {
                layout.inline_atoms[layout.inline_atom_count++] = atom;
                ++layout.atom_count;
                ++layout.fact_count;
            }
        }
    } else {
        layout.atom_facts_map.reserve(atom_indices.size());
        for (auto atom : atom_indices) {
            if (layout.atom_facts_map.emplace(atom, layout.fact_count).second) {
                ++layout.atom_count;
                ++layout.fact_count;
            }
        }
    }
    if (region.pointers.size() <= LifetimeFactLayout::inline_capacity) {
        for (auto *pointer : region.pointers) {
            if (layout.find_pointer(pointer) == ~size_t{0u}) {
                layout.inline_pointers[layout.inline_pointer_count++] = pointer;
                ++layout.pointer_count;
                ++layout.fact_count;
            }
        }
    } else {
        layout.pointer_facts_map.reserve(region.pointers.size());
        for (auto *pointer : region.pointers) {
            if (layout.pointer_facts_map.emplace(pointer, layout.fact_count).second) {
                ++layout.pointer_count;
                ++layout.fact_count;
            }
        }
    }
    return layout;
}

void redefine_pointer(Value *pointer, LifetimeFactState &state,
                      const LifetimeFactLayout &layout) noexcept {
    if (auto idx = layout.find_pointer(pointer); idx != ~size_t{0u}) {
        state.clear(idx);
    }
}

void define_pointer(Value *pointer, LifetimeFactState &state,
                    const LifetimeFactLayout &layout,
                    const CoroFrameAtomDomain &domain) noexcept {
    if (auto idx = layout.find_pointer(pointer); idx != ~size_t{0u}) {
        // A typed XIR store overwrites the complete object denoted by this
        // dynamic pointer version, even when no fixed aggregate leaf is a
        // Must target (for example array[i]).
        state.set(idx);
    }
    for (auto access : domain.memory_accesses(pointer)) {
        if (!access.covers_atom) { continue; }
        if (auto idx = layout.find_atom(access.atom_index);
            idx != ~size_t{0u}) {
            state.set(idx);
        }
    }
}

[[nodiscard]] bool pointer_is_defined(
    Value *pointer, const LifetimeFactState &state,
    const LifetimeFactLayout &layout,
    const CoroFrameAtomDomain &domain) noexcept {
    if (auto idx = layout.find_pointer(pointer);
        idx != ~size_t{0u} && state.get(idx)) {
        return true;
    }
    auto has_relevant_atom = false;
    for (auto access : domain.memory_accesses(pointer)) {
        if (auto idx = layout.find_atom(access.atom_index);
            idx != ~size_t{0u}) {
            has_relevant_atom = true;
            if (!state.get(idx)) { return false; }
        }
    }
    // A memory observation with no fact would mean the typed projection
    // analysis failed to represent a reachable use. Reject rather than
    // treating an empty conjunction as proof.
    return has_relevant_atom;
}

[[nodiscard]] bool apply_lifetime_events(
    luisa::span<const LifetimeEvent> events,
    LifetimeFactState &state,
    const LifetimeFactLayout &layout,
    const CoroFrameAtomDomain &domain,
    bool validate_reads,
    Instruction **failing_read = nullptr) noexcept {
    for (auto event : events) {
        switch (event.kind) {
            case LifetimeEventKind::redefine_pointer:
                // A static GEP instruction may execute repeatedly with a new
                // runtime index. Its previous exact-address fact therefore
                // cannot cross this definition.
                redefine_pointer(event.pointer, state, layout);
                break;
            case LifetimeEventKind::store:
                define_pointer(event.pointer, state, layout, domain);
                break;
            case LifetimeEventKind::read:
                if (validate_reads &&
                    !pointer_is_defined(
                        event.pointer, state, layout, domain)) {
                    if (failing_read != nullptr) {
                        *failing_read = event.instruction;
                    }
                    return false;
                }
                break;
        }
    }
    return true;
}

[[nodiscard]] bool append_call_lifetime_events(
    CallInst *call, const AllocaUseRegion &region,
    ReferenceArgumentEffectAnalysis &reference_effects,
    luisa::vector<LifetimeEvent> &events) noexcept {
    auto *callee = call == nullptr ? nullptr : call->callee();
    luisa::vector<Argument *> formals;
    if (callee != nullptr) {
        for (auto *argument : callee->arguments()) {
            formals.emplace_back(argument);
        }
    }
    auto signature_matches =
        callee != nullptr && formals.size() == call->argument_count();

    // A call is one unordered aliasing event at this abstraction level. Emit
    // every possible old-value observation before any Must definition. This
    // preserves soundness when the same object (or overlapping projections) is
    // bound to multiple reference formals: a write-only formal must not hide a
    // read-before-write through an aliasing formal merely because it appears
    // earlier in the signature.
    luisa::vector<Value *> reads;
    luisa::vector<Value *> definitions;
    luisa::unordered_set<Value *> seen_reads;
    luisa::unordered_set<Value *> seen_definitions;
    auto found_pointer_operand = false;
    for (size_t i = 0u; i < call->argument_count(); ++i) {
        auto *actual = call->argument(i);
        if (!region.pointers.contains(actual)) { continue; }
        found_pointer_operand = true;
        auto effect = ReferenceArgumentEffect{};
        if (signature_matches && formals[i]->is_reference() &&
            formals[i]->type() == actual->type()) {
            effect = reference_effects.effect(formals[i]);
        }
        if (effect.may_read_prior_value &&
            seen_reads.emplace(actual).second) {
            reads.emplace_back(actual);
        }
        if (effect.fully_defines_on_return &&
            seen_definitions.emplace(actual).second) {
            definitions.emplace_back(actual);
        }
    }
    for (auto *pointer : reads) {
        events.emplace_back(
            LifetimeEventKind::read, pointer, call);
    }
    for (auto *pointer : definitions) {
        events.emplace_back(
            LifetimeEventKind::store, pointer, call);
    }
    return found_pointer_operand;
}

[[nodiscard]] bool append_ray_query_pipeline_lifetime_events(
    RayQueryPipelineInst *pipeline, const AllocaUseRegion &region,
    ReferenceArgumentEffectAnalysis &reference_effects,
    luisa::vector<LifetimeEvent> &events) noexcept {
    if (pipeline == nullptr) { return false; }

    luisa::unordered_set<Value *> seen_reads;
    auto append_read = [&](Value *pointer) noexcept {
        if (region.pointers.contains(pointer) &&
            seen_reads.emplace(pointer).second) {
            events.emplace_back(
                LifetimeEventKind::read, pointer, pipeline);
            return true;
        }
        return false;
    };

    // The query object is state consumed by the traversal itself. Candidate
    // callbacks, however, execute zero or more times in backend-defined
    // candidate order. Therefore a capture may begin a fresh lifetime iff no
    // possible handler reads its incoming value before defining it. Callback
    // writes are deliberately not Must definitions of the pipeline: a query
    // with no matching candidate executes neither handler.
    auto found_pointer_operand = append_read(pipeline->query_object());
    auto capture_count = pipeline->captured_argument_count();
    auto handlers = std::array{
        pipeline->on_surface_function(),
        pipeline->on_procedural_function()};
    luisa::vector<luisa::vector<Argument *>> handler_formals;
    handler_formals.reserve(handlers.size());
    for (auto *handler : handlers) {
        auto &formals = handler_formals.emplace_back();
        if (handler != nullptr) {
            for (auto *argument : handler->arguments()) {
                formals.emplace_back(argument);
            }
        }
    }

    for (size_t capture_index = 0u;
         capture_index < capture_count; ++capture_index) {
        auto *actual = pipeline->captured_argument(capture_index);
        if (!region.pointers.contains(actual)) { continue; }
        found_pointer_operand = true;
        auto may_read_prior_value = false;
        for (size_t handler_index = 0u;
             handler_index < handlers.size(); ++handler_index) {
            auto *handler = handlers[handler_index];
            auto &formals = handler_formals[handler_index];
            auto signature_matches =
                handler != nullptr &&
                formals.size() == capture_count + 1u;
            if (!signature_matches) {
                may_read_prior_value = true;
                break;
            }
            auto *formal = formals[capture_index + 1u];
            if (!formal->is_reference() || formal->type() != actual->type() ||
                reference_effects.effect(formal).may_read_prior_value) {
                may_read_prior_value = true;
                break;
            }
        }
        if (may_read_prior_value) {
            static_cast<void>(append_read(actual));
        }
    }
    return found_pointer_operand;
}

[[nodiscard]] bool append_instruction_lifetime_events(
    Instruction *instruction, const AllocaUseRegion &region,
    ReferenceArgumentEffectAnalysis &reference_effects,
    luisa::vector<LifetimeEvent> &events) noexcept {
    if (instruction->isa<GEPInst>()) {
        events.emplace_back(LifetimeEvent{
            LifetimeEventKind::redefine_pointer, instruction,
            instruction});
        return true;
    }
    if (instruction->isa<LoadInst>()) {
        auto *pointer =
            static_cast<LoadInst *>(instruction)->variable();
        if (!region.pointers.contains(pointer)) { return false; }
        events.emplace_back(LifetimeEvent{
            LifetimeEventKind::read, pointer, instruction});
        return true;
    }
    if (instruction->isa<StoreInst>()) {
        auto *pointer =
            static_cast<StoreInst *>(instruction)->variable();
        if (!region.pointers.contains(pointer)) { return false; }
        events.emplace_back(LifetimeEvent{
            LifetimeEventKind::store, pointer, instruction});
        return true;
    }
    if (instruction->isa<CallInst>()) {
        return append_call_lifetime_events(
            static_cast<CallInst *>(instruction), region,
            reference_effects, events);
    }
    if (instruction->isa<RayQueryPipelineInst>()) {
        return append_ray_query_pipeline_lifetime_events(
            static_cast<RayQueryPipelineInst *>(instruction),
            region, reference_effects, events);
    }
    auto found_pointer_operand = false;
    luisa::unordered_set<Value *> seen_pointers;
    for (auto *operand_use : instruction->operand_uses()) {
        auto *pointer =
            operand_use == nullptr ? nullptr : operand_use->value();
        if (!region.pointers.contains(pointer) ||
            !seen_pointers.emplace(pointer).second) {
            continue;
        }
        found_pointer_operand = true;
        // Atomics and unknown pointer operations may observe the old value
        // before any possible write. Ordinary reference calls are handled
        // above by their field-sensitive callee summary.
        events.emplace_back(LifetimeEvent{
            LifetimeEventKind::read, pointer, instruction});
    }
    return found_pointer_operand;
}

struct OrderedLifetimeUser {
    Instruction *instruction;
    InstructionLocation location;
};

struct PreparedLifetimeProblem {
    LifetimeProofProblem problem;
    bool fast_succeeded{false};
    size_t fast_block_evaluations{0u};
};

[[nodiscard]] PreparedLifetimeProblem prepare_lifetime_proof_problem(
    BasicBlock *target, Instruction *insertion_instruction,
    const AllocaUseRegion &region,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    ReferenceArgumentEffectAnalysis &reference_effects,
    const InstructionLocationMap &instruction_locations,
    luisa::span<const size_t> atom_indices,
    LifetimeProofWorkspace &workspace,
    LifetimeProofTimings *timings) noexcept {
    using Clock = std::chrono::steady_clock;
    auto phase_begin = timings == nullptr ?
                           Clock::time_point{} :
                           Clock::now();
    PreparedLifetimeProblem out;
    auto &problem = out.problem;
    if (target == nullptr || insertion_instruction == nullptr) {
        return out;
    }
    auto target_id = graph.block_id(target);
    if (target_id >= graph.block_count()) {
        return out;
    }
    problem.target_id = target_id;

    auto insertion_iter = instruction_locations.find(insertion_instruction);
    if (insertion_iter == instruction_locations.end() ||
        insertion_iter->second.block_id != target_id) {
        return out;
    }
    for (auto *block : region.blocks) {
        if (!graph.dominates(target, block)) { return out; }
        auto id = graph.block_id(block);
        if (id >= graph.block_count()) { return out; }
    }

    problem.layout = make_lifetime_fact_layout(atom_indices, region);
    if (problem.layout.fact_count == 0u) { return out; }
    problem.distributive_facts = (problem.layout.pointer_count == 1u);
    if (!problem.distributive_facts) {
        auto all_covering = !region.pointers.empty();
        for (auto *pointer : region.pointers) {
            auto accesses = domain.memory_accesses(pointer);
            if (accesses.empty()) {
                all_covering = false;
                break;
            }
            for (auto access : accesses) {
                if (!access.covers_atom) {
                    all_covering = false;
                    break;
                }
            }
            if (!all_covering) { break; }
        }
        problem.distributive_facts = all_covering;
    }
    if (timings != nullptr) {
        auto now = Clock::now();
        timings->layout_ms +=
            std::chrono::duration<double, std::milli>{
                now - phase_begin}
                .count();
        phase_begin = now;
    }

    luisa::vector<OrderedLifetimeUser> ordered_users;
    auto event_order_elapsed = 0.0;
    ordered_users.reserve(region.users.size());
    for (auto *instruction : region.users) {
        auto iter = instruction_locations.find(instruction);
        if (iter == instruction_locations.end() ||
            iter->second.block_id >= graph.block_count()) {
            return out;
        }
        if (iter->second.block_id == target_id &&
            iter->second.ordinal < insertion_iter->second.ordinal) {
            return out;
        }
        ordered_users.emplace_back(
            OrderedLifetimeUser{instruction, iter->second});
    }
    luisa::sort(
        ordered_users.begin(), ordered_users.end(),
        [](auto lhs, auto rhs) noexcept {
            return lhs.location.block_id < rhs.location.block_id ||
                   (lhs.location.block_id == rhs.location.block_id &&
                    lhs.location.ordinal < rhs.location.ordinal);
        });
    if (timings != nullptr) {
        auto now = Clock::now();
        event_order_elapsed =
            std::chrono::duration<double, std::milli>{
                now - phase_begin}
                .count();
        timings->event_order_ms += event_order_elapsed;
        phase_begin = now;
    }

    luisa::vector<std::pair<size_t, BlockEventRange>> user_block_ranges;
    problem.all_events.reserve(ordered_users.size());
    for (size_t i = 0u; i < ordered_users.size();) {
        const auto block_id = ordered_users[i].location.block_id;
        const auto ev_begin = static_cast<uint32_t>(problem.all_events.size());
        while (i < ordered_users.size() &&
               ordered_users[i].location.block_id == block_id) {
            if (!append_instruction_lifetime_events(
                    ordered_users[i].instruction, region,
                    reference_effects, problem.all_events)) {
                return out;
            }
            ++i;
        }
        const auto ev_count =
            static_cast<uint32_t>(problem.all_events.size()) - ev_begin;
        if (ev_count != 0u) {
            user_block_ranges.emplace_back(
                block_id, BlockEventRange{ev_begin, ev_count});
        }
    }
    if (timings != nullptr) {
        auto now = Clock::now();
        auto elapsed =
            std::chrono::duration<double, std::milli>{
                now - phase_begin}
                .count();
        timings->event_transfer_ms += elapsed;
        timings->events_ms += event_order_elapsed + elapsed;
        phase_begin = now;
    }

    // Fast path 1 & 2: evaluate every user block from `bottom`.
    // Blocks that locally validate every read from `bottom` never need incoming
    // facts (though their stores remain available as generators for downstream
    // `needs_incoming` blocks).
    luisa::vector<std::pair<size_t, BlockEventRange>> needs_incoming;
    needs_incoming.reserve(user_block_ranges.size());
    LifetimeFactState target_out_state;
    auto has_valid_target_out = false;
    auto target_failed_locally = false;
    for (auto [ub_id, range] : user_block_ranges) {
        LifetimeFactState state{problem.layout.fact_count, false};
        luisa::span<const LifetimeEvent> evs{
            problem.all_events.data() + range.begin, range.count};
        if (!apply_lifetime_events(
                evs, state, problem.layout, domain, true)) {
            needs_incoming.emplace_back(ub_id, range);
            if (ub_id == target_id) { target_failed_locally = true; }
        } else if (ub_id == target_id) {
            target_out_state = std::move(state);
            has_valid_target_out = true;
        }
    }

    if (needs_incoming.empty()) {
        out.fast_succeeded = true;
        out.fast_block_evaluations = user_block_ranges.size();
        return out;
    }

    if (!target_failed_locally && has_valid_target_out) {
        auto covered_by_target = true;
        for (auto [ub_id, range] : needs_incoming) {
            static_cast<void>(ub_id);
            luisa::span<const LifetimeEvent> evs{
                problem.all_events.data() + range.begin, range.count};
            for (auto ev : evs) {
                if (ev.kind != LifetimeEventKind::read) { continue; }
                auto has_relevant_atom = false;
                for (auto access : domain.memory_accesses(ev.pointer)) {
                    if (auto idx = problem.layout.find_atom(access.atom_index);
                        idx != ~size_t{0u}) {
                        has_relevant_atom = true;
                        if (!target_out_state.get(idx)) {
                            covered_by_target = false;
                            break;
                        }
                    }
                }
                if (!has_relevant_atom) { covered_by_target = false; }
                if (!covered_by_target) { break; }
            }
            if (!covered_by_target) { break; }
        }
        if (covered_by_target) {
            out.fast_succeeded = true;
            out.fast_block_evaluations = 1u;
            return out;
        }
    }

    // Restrict the proof to the reverse slice from only `needs_incoming`
    // blocks back to the proposed lifetime start `target`.
    const auto a_gen = workspace.next_active_gen();
    luisa::vector<size_t> worklist;
    if (target_failed_locally) {
        workspace.active_epoch[target_id] = a_gen;
        worklist.emplace_back(target_id);
    } else {
        for (auto [ub_id, range] : needs_incoming) {
            static_cast<void>(range);
            if (workspace.active_epoch[ub_id] != a_gen) {
                workspace.active_epoch[ub_id] = a_gen;
                worklist.emplace_back(ub_id);
            }
        }
        if (workspace.active_epoch[target_id] != a_gen) {
            workspace.active_epoch[target_id] = a_gen;
            worklist.emplace_back(target_id);
        }
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            auto block_id = worklist[cursor];
            if (block_id == target_id) { continue; }
            for (auto predecessor : graph.predecessors(block_id)) {
                auto *predecessor_block = graph.block(predecessor);
                if (!graph.dominates(target, predecessor_block)) {
                    return out;
                }
                if (workspace.active_epoch[predecessor] != a_gen) {
                    workspace.active_epoch[predecessor] = a_gen;
                    worklist.emplace_back(predecessor);
                }
            }
        }
    }
    if (timings != nullptr) {
        timings->slice_ms +=
            std::chrono::duration<double, std::milli>{
                Clock::now() - phase_begin}
                .count();
    }

    // The fixed-point equations are order independent. Sort the sparse
    // reverse slice by semantic block id to retain the deterministic
    // traversal order without materializing one empty vector per CFG block.
    luisa::sort(worklist.begin(), worklist.end());
    problem.active_blocks = std::move(worklist);
    for (size_t active_idx = 0u;
         active_idx < problem.active_blocks.size(); ++active_idx) {
        workspace.active_index[problem.active_blocks[active_idx]] = active_idx;
    }
    problem.block_events.assign(
        problem.active_blocks.size(), BlockEventRange{});
    for (auto [ub_id, range] : user_block_ranges) {
        if (ub_id < workspace.active_epoch.size() &&
            workspace.is_active(ub_id)) {
            problem.block_events[workspace.active_index[ub_id]] = range;
        }
    }
    problem.valid = true;
    return out;
}

[[nodiscard]] LifetimeProofResult prove_unconditional_fresh_lifetime(
    const LifetimeProofProblem &problem,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    LifetimeProofWorkspace &workspace) noexcept {
    LifetimeProofResult result;
    if (!problem.valid) { return result; }

    const auto active_count = problem.active_blocks.size();
    if (problem.layout.fact_count <= 64u) {
        const uint64_t all_ones =
            problem.layout.fact_count == 64u ?
                ~uint64_t{0u} :
                (problem.layout.fact_count == 0u ?
                     uint64_t{0u} :
                     ((uint64_t{1u} << problem.layout.fact_count) - uint64_t{1u}));
        luisa::vector<uint64_t> keep_masks(active_count, all_ones);
        luisa::vector<uint64_t> gen_masks(active_count, uint64_t{0u});
        for (size_t active_idx = 0u; active_idx < active_count; ++active_idx) {
            const auto events = problem.events_for_active(active_idx);
            if (events.empty()) { continue; }
            auto keep = all_ones;
            auto gen = uint64_t{0u};
            for (auto ev : events) {
                if (ev.kind == LifetimeEventKind::redefine_pointer) {
                    if (auto idx = problem.layout.find_pointer(ev.pointer);
                        idx != ~size_t{0u}) {
                        const auto bit = uint64_t{1u} << idx;
                        keep &= ~bit;
                        gen &= ~bit;
                    }
                } else if (ev.kind == LifetimeEventKind::store) {
                    auto store_mask = uint64_t{0u};
                    if (auto idx = problem.layout.find_pointer(ev.pointer);
                        idx != ~size_t{0u}) {
                        store_mask |= uint64_t{1u} << idx;
                    }
                    for (auto access : domain.memory_accesses(ev.pointer)) {
                        if (!access.covers_atom) { continue; }
                        if (auto idx = problem.layout.find_atom(access.atom_index);
                            idx != ~size_t{0u}) {
                            store_mask |= uint64_t{1u} << idx;
                        }
                    }
                    gen |= store_mask;
                }
            }
            keep_masks[active_idx] = keep;
            gen_masks[active_idx] = gen;
        }

        luisa::vector<uint64_t> in_states(active_count, all_ones);
        luisa::vector<uint64_t> out_states(active_count, all_ones);
        const auto q_gen = workspace.next_queued_gen();
        luisa::vector<size_t> worklist;
        worklist.reserve(active_count);
        for (size_t active_idx = 0u; active_idx < active_count; ++active_idx) {
            auto block_id = problem.active_blocks[active_idx];
            workspace.queued_epoch[block_id] = q_gen;
            worklist.emplace_back(active_idx);
        }
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            auto active_idx = worklist[cursor];
            auto block_id = problem.active_blocks[active_idx];
            workspace.queued_epoch[block_id] = 0u;
            ++result.block_evaluation_count;
            uint64_t next_in = uint64_t{0u};
            if (block_id != problem.target_id) {
                auto first_predecessor = true;
                for (auto predecessor : graph.predecessors(block_id)) {
                    if (!workspace.is_active(predecessor)) {
                        return result;
                    }
                    const auto pred_out =
                        out_states[workspace.active_index[predecessor]];
                    if (first_predecessor) {
                        next_in = pred_out;
                        first_predecessor = false;
                    } else {
                        next_in &= pred_out;
                    }
                }
                if (first_predecessor) { return result; }
            }
            const uint64_t next_out =
                (next_in & keep_masks[active_idx]) | gen_masks[active_idx];
            const auto out_changed = out_states[active_idx] != next_out;
            if (in_states[active_idx] != next_in || out_changed) {
                in_states[active_idx] = next_in;
                out_states[active_idx] = next_out;
                if (out_changed) {
                    for (auto successor : graph.successors(block_id)) {
                        if (successor != problem.target_id &&
                            workspace.is_active(successor) &&
                            workspace.queued_epoch[successor] != q_gen) {
                            workspace.queued_epoch[successor] = q_gen;
                            worklist.emplace_back(workspace.active_index[successor]);
                        }
                    }
                }
            }
        }

        LifetimeFactState state{problem.layout.fact_count, false};
        for (size_t active_idx = 0u; active_idx < active_count; ++active_idx) {
            const auto events = problem.events_for_active(active_idx);
            if (events.empty()) { continue; }
            state.set_from_mask(in_states[active_idx]);
            if (!apply_lifetime_events(
                    events, state,
                    problem.layout, domain, true,
                    &result.failing_read)) {
                return result;
            }
        }
        result.succeeded = true;
        return result;
    }

    auto top = LifetimeFactState(problem.layout.fact_count, true);
    auto bottom = LifetimeFactState(problem.layout.fact_count, false);
    luisa::vector<LifetimeFactState> in_states(active_count, top);
    luisa::vector<LifetimeFactState> out_states(active_count, top);

    // This is the greatest fixed point of the forward Must equations:
    //   IN[target] = empty
    //   IN[b]      = intersection OUT[p]
    //   OUT[b]     = GEN[b] union (IN[b] - exact-GEP-redefinitions[b]).
    // Starting at top makes every update descending; running a sparse worklist
    // seeded in RPO order reaches the exact fixed point without rescanning
    // unchanged blocks.
    const auto q_gen = workspace.next_queued_gen();
    luisa::vector<size_t> worklist;
    worklist.reserve(active_count);
    for (size_t active_idx = 0u; active_idx < active_count; ++active_idx) {
        auto block_id = problem.active_blocks[active_idx];
        workspace.queued_epoch[block_id] = q_gen;
        worklist.emplace_back(active_idx);
    }
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto active_idx = worklist[cursor];
        auto block_id = problem.active_blocks[active_idx];
        workspace.queued_epoch[block_id] = 0u;
        ++result.block_evaluation_count;
        LifetimeFactState next_in;
        if (block_id == problem.target_id) {
            next_in = bottom;
        } else {
            auto first_predecessor = true;
            for (auto predecessor : graph.predecessors(block_id)) {
                if (!workspace.is_active(predecessor)) {
                    return result;
                }
                const auto &pred_out = out_states[workspace.active_index[predecessor]];
                if (first_predecessor) {
                    next_in = pred_out;
                    first_predecessor = false;
                } else {
                    static_cast<void>(next_in.intersect_with(pred_out));
                }
            }
            if (first_predecessor) { return result; }
        }
        auto next_out = next_in;
        if (const auto events = problem.events_for_active(active_idx);
            !events.empty()) {
            static_cast<void>(apply_lifetime_events(
                events, next_out,
                problem.layout, domain, false));
        }
        auto out_changed = out_states[active_idx] != next_out;
        if (in_states[active_idx] != next_in || out_changed) {
            in_states[active_idx] = std::move(next_in);
            out_states[active_idx] = std::move(next_out);
            if (out_changed) {
                for (auto successor : graph.successors(block_id)) {
                    if (successor != problem.target_id &&
                        workspace.is_active(successor) &&
                        workspace.queued_epoch[successor] != q_gen) {
                        workspace.queued_epoch[successor] = q_gen;
                        worklist.emplace_back(workspace.active_index[successor]);
                    }
                }
            }
        }
    }

    for (size_t active_idx = 0u; active_idx < active_count; ++active_idx) {
        const auto events = problem.events_for_active(active_idx);
        if (events.empty()) { continue; }
        auto state = in_states[active_idx];
        if (!apply_lifetime_events(
                events, state,
                problem.layout, domain, true,
                &result.failing_read)) {
            return result;
        }
    }
    result.succeeded = true;
    return result;
}

struct PredicateAssignment {
    size_t predicate;
    bool value;
};

constexpr auto max_predicates_per_cube = 12u;
constexpr auto max_guarded_states_per_block = 64u;

struct PredicateCube {
    std::array<PredicateAssignment, max_predicates_per_cube> data{};
    uint8_t count{0u};

    [[nodiscard]] bool empty() const noexcept { return count == 0u; }
    [[nodiscard]] size_t size() const noexcept { return count; }
    void clear() noexcept { count = 0u; }

    [[nodiscard]] PredicateAssignment *begin() noexcept { return data.data(); }
    [[nodiscard]] PredicateAssignment *end() noexcept { return data.data() + count; }
    [[nodiscard]] const PredicateAssignment *begin() const noexcept { return data.data(); }
    [[nodiscard]] const PredicateAssignment *end() const noexcept { return data.data() + count; }
    [[nodiscard]] const PredicateAssignment &operator[](size_t i) const noexcept {
        return data[i];
    }

    void insert(PredicateAssignment *pos, PredicateAssignment value) noexcept {
        LUISA_DEBUG_ASSERT(count < max_predicates_per_cube,
                           "PredicateCube capacity exceeded.");
        const auto idx = static_cast<size_t>(pos - data.data());
        for (size_t i = count; i > idx; --i) {
            data[i] = data[i - 1u];
        }
        data[idx] = value;
        ++count;
    }

    void erase_killed(luisa::span<const size_t> killed) noexcept {
        if (count == 0u || killed.empty()) { return; }
        uint8_t write_idx = 0u;
        for (uint8_t i = 0u; i < count; ++i) {
            if (!luisa::binary_search(
                    killed.begin(), killed.end(),
                    data[i].predicate)) {
                data[write_idx++] = data[i];
            }
        }
        count = write_idx;
    }
};

struct GuardedLifetimeState {
    PredicateCube cube;
    LifetimeFactState facts;
    bool dirty{true};
};

[[nodiscard]] bool cube_subsumes(
    const PredicateCube &general,
    const PredicateCube &specific) noexcept {
    auto j = size_t{0u};
    for (auto literal : general) {
        while (j < specific.size() &&
               specific[j].predicate < literal.predicate) {
            ++j;
        }
        if (j == specific.size() ||
            specific[j].predicate != literal.predicate ||
            specific[j].value != literal.value) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool intersect_facts(
    LifetimeFactState &target,
    const LifetimeFactState &incoming) noexcept {
    return target.intersect_with(incoming);
}

void kill_predicates(PredicateCube &cube,
                     luisa::span<const size_t> killed) noexcept {
    cube.erase_killed(killed);
}

[[nodiscard]] bool refine_cube(
    PredicateCube &cube,
    CoroPredicateLiteral literal) noexcept {
    auto iter = std::lower_bound(
        cube.begin(), cube.end(), literal.predicate,
        [](auto existing, size_t predicate) noexcept {
            return existing.predicate < predicate;
        });
    if (iter != cube.end() &&
        iter->predicate == literal.predicate) {
        return iter->value == literal.value;
    }
    // Forgetting a new literal is a conservative widening: both edges remain
    // feasible and later Must intersections can only lose facts.
    if (cube.size() >= max_predicates_per_cube) { return true; }
    cube.insert(iter, PredicateAssignment{
                          literal.predicate, literal.value});
    return true;
}

[[nodiscard]] bool merge_guarded_state(
    luisa::vector<GuardedLifetimeState> &states,
    GuardedLifetimeState incoming,
    size_t &widening_count) noexcept {
    auto covered = false;
    auto changed = false;
    for (auto &state : states) {
        if (cube_subsumes(state.cube, incoming.cube)) {
            covered = true;
            if (intersect_facts(state.facts, incoming.facts)) {
                state.dirty = true;
                changed = true;
            }
        }
    }
    if (covered) { return changed; }

    for (size_t i = 0u; i < states.size();) {
        if (cube_subsumes(incoming.cube, states[i].cube)) {
            static_cast<void>(intersect_facts(
                incoming.facts, states[i].facts));
            states.erase(states.begin() + i);
        } else {
            ++i;
        }
    }
    if (states.size() >= max_guarded_states_per_block) {
        for (auto &&state : states) {
            static_cast<void>(intersect_facts(
                incoming.facts, state.facts));
        }
        states.clear();
        incoming.cube.clear();
        ++widening_count;
    }
    incoming.dirty = true;
    states.emplace_back(std::move(incoming));
    return true;
}

struct GuardedTransferEvent {
    Instruction *instruction{nullptr};
    luisa::span<const size_t> killed_predicates;
    const LifetimeEvent *lifetimes{nullptr};
    size_t lifetime_count{0u};
};

[[nodiscard]] luisa::vector<GuardedTransferEvent>
build_block_guarded_transfers(
    size_t active_index,
    const LifetimeProofProblem &problem,
    const CoroPredicateAnalysis &predicates,
    const InstructionLocationMap &instruction_locations) noexcept {
    luisa::vector<GuardedTransferEvent> out;
    const auto block_id = problem.active_blocks[active_index];
    const auto events = problem.events_for_active(active_index);
    const auto kills = predicates.block_kills(block_id);
    if (events.empty() && kills.empty()) { return out; }

    auto min_kill_ordinal = size_t{0u};
    if (block_id == problem.target_id && !events.empty()) {
        auto iter = instruction_locations.find(events.front().instruction);
        if (iter != instruction_locations.end()) {
            min_kill_ordinal = iter->second.ordinal;
        }
    }

    size_t kill_idx = 0u;
    while (kill_idx < kills.size() &&
           kills[kill_idx].ordinal < min_kill_ordinal) {
        ++kill_idx;
    }

    size_t ev_idx = 0u;
    out.reserve(kills.size() - kill_idx + events.size());
    while (kill_idx < kills.size() || ev_idx < events.size()) {
        if (ev_idx >= events.size()) {
            const auto &k = kills[kill_idx++];
            out.emplace_back(GuardedTransferEvent{
                k.instruction, k.predicates, nullptr, 0u});
            continue;
        }
        auto *ev_inst = events[ev_idx].instruction;
        size_t ev_count = 1u;
        while (ev_idx + ev_count < events.size() &&
               events[ev_idx + ev_count].instruction == ev_inst) {
            ++ev_count;
        }
        auto ev_ord = instruction_locations.at(ev_inst).ordinal;
        if (kill_idx >= kills.size() || ev_ord < kills[kill_idx].ordinal) {
            out.emplace_back(GuardedTransferEvent{
                ev_inst, {}, &events[ev_idx], ev_count});
            ev_idx += ev_count;
        } else if (kills[kill_idx].ordinal < ev_ord) {
            const auto &k = kills[kill_idx++];
            out.emplace_back(GuardedTransferEvent{
                k.instruction, k.predicates, nullptr, 0u});
        } else {
            const auto &k = kills[kill_idx++];
            out.emplace_back(GuardedTransferEvent{
                ev_inst, k.predicates, &events[ev_idx], ev_count});
            ev_idx += ev_count;
        }
    }
    return out;
}

[[nodiscard]] bool apply_guarded_transfer(
    luisa::span<const GuardedTransferEvent> events,
    GuardedLifetimeState &state,
    const LifetimeFactLayout &layout,
    const CoroFrameAtomDomain &domain,
    bool validate_reads,
    Instruction **failing_read = nullptr) noexcept {
    for (auto event : events) {
        if (!event.killed_predicates.empty()) {
            kill_predicates(state.cube, event.killed_predicates);
        }
        if (event.lifetime_count != 0u &&
            !apply_lifetime_events(
                luisa::span<const LifetimeEvent>{
                    event.lifetimes, event.lifetime_count},
                state.facts,
                layout, domain, validate_reads,
                failing_read)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] LifetimeProofResult prove_guarded_fresh_lifetime(
    const LifetimeProofProblem &problem,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    const CoroPredicateAnalysis &predicates,
    const InstructionLocationMap &instruction_locations,
    LifetimeProofWorkspace &workspace,
    size_t max_state_evaluations) noexcept {
    LifetimeProofResult result;
    result.guarded = true;
    if (!problem.valid || predicates.predicate_count() == 0u) {
        return result;
    }
    auto consume_state_evaluation = [&]() noexcept {
        if (max_state_evaluations != 0u &&
            result.guarded_state_evaluation_count >= max_state_evaluations) {
            // No partial fixed point is a proof. Keep succeeded=false and
            // let the caller retain this lifetime or try another sound proof.
            result.guarded_budget_exhausted = true;
            return false;
        }
        ++result.guarded_state_evaluation_count;
        return true;
    };
    const auto active_count = problem.active_blocks.size();
    luisa::unordered_map<size_t, luisa::vector<GuardedTransferEvent>>
        event_block_transfers;
    auto apply_block_transfer = [&](size_t active_idx,
                                    size_t block_id,
                                    GuardedLifetimeState &state,
                                    Instruction **failing_read) noexcept -> bool {
        if (problem.block_events[active_idx].count == 0u) {
            if (state.cube.empty()) { return true; }
            for (const auto &k : predicates.block_kills(block_id)) {
                kill_predicates(state.cube, k.predicates);
                if (state.cube.empty()) { break; }
            }
            return true;
        }
        auto [iter, inserted] = event_block_transfers.try_emplace(active_idx);
        if (inserted) {
            iter->second = build_block_guarded_transfers(
                active_idx, problem, predicates, instruction_locations);
        }
        return apply_guarded_transfer(
            iter->second, state, problem.layout, domain, true, failing_read);
    };
    luisa::vector<luisa::vector<GuardedLifetimeState>>
        in_states(active_count);
    in_states[workspace.active_index[problem.target_id]].emplace_back(
        GuardedLifetimeState{
            .facts = LifetimeFactState(
                problem.layout.fact_count, false),
            .dirty = true});

    const auto skip_unobservable_predicates = problem.distributive_facts;
    auto p_gen = uint32_t{0u};
    luisa::span<const uint8_t> in_cycle;
    if (skip_unobservable_predicates) {
        p_gen = workspace.next_pred_gen();
        for (const auto block_id : problem.active_blocks) {
            const auto pred = predicates.block_predicate(block_id);
            if (pred == ~size_t{0u} || pred >= workspace.pred_epoch.size()) {
                continue;
            }
            auto has_active_literal_edge = false;
            for (const auto succ : graph.successors(block_id)) {
                if (succ != problem.target_id &&
                    workspace.is_active(succ) &&
                    predicates.literal_on_edge_by_id(block_id, succ).has_value()) {
                    has_active_literal_edge = true;
                    break;
                }
            }
            if (has_active_literal_edge) {
                if (workspace.pred_epoch[pred] != p_gen) {
                    workspace.pred_epoch[pred] = p_gen;
                    workspace.pred_count[pred] = 1u;
                } else {
                    ++workspace.pred_count[pred];
                }
            }
        }
        in_cycle = workspace.block_in_cycle_under_target(
            problem.target_id, graph);
    }

    const auto q_gen = workspace.next_queued_gen();
    std::priority_queue<size_t, luisa::vector<size_t>, std::greater<size_t>>
        worklist;
    worklist.push(problem.target_id);
    luisa::vector<GuardedLifetimeState> current_states;
    workspace.queued_epoch[problem.target_id] = q_gen;
    while (!worklist.empty()) {
        auto block_id = worklist.top();
        worklist.pop();
        workspace.queued_epoch[block_id] = 0u;
        const auto active_idx = workspace.active_index[block_id];
        current_states.clear();
        for (auto &s : in_states[active_idx]) {
            if (s.dirty) {
                s.dirty = false;
                current_states.emplace_back(s);
            }
        }
        for (auto &state : current_states) {
            if (!consume_state_evaluation()) { return result; }
            // The guarded domain is a forward May partition of paths whose
            // fact component is a Must property. Merging states only
            // intersects facts, predicate widening only admits more paths,
            // and no later fixed-point update can turn a missing fact at an
            // already reached read into a definite fact. Therefore a failed
            // read is a monotone proof failure and may terminate the analysis
            // immediately. This is not a concrete counterexample after
            // widening, but rejecting the contraction remains conservative.
            // Deferring the check until convergence needlessly explores the
            // complete downstream predicate product for known-bad lifetimes.
            if (!apply_block_transfer(
                    active_idx, block_id, state, &result.failing_read)) {
                result.failing_predicate_count = state.cube.size();
                return result;
            }
            for (auto successor : graph.successors(block_id)) {
                if (successor == problem.target_id ||
                    !workspace.is_active(successor)) {
                    continue;
                }
                auto next = state;
                if (auto literal = predicates.literal_on_edge_by_id(
                        block_id, successor)) {
                    const auto skip =
                        skip_unobservable_predicates &&
                        literal->predicate < workspace.pred_epoch.size() &&
                        workspace.pred_epoch[literal->predicate] == p_gen &&
                        workspace.pred_count[literal->predicate] <= 1u &&
                        ((block_id < in_cycle.size() && in_cycle[block_id] == 0u) ||
                         predicates.block_predicate_self_killed(block_id));
                    if (!skip && !refine_cube(next.cube, *literal)) {
                        continue;
                    }
                }
                if (merge_guarded_state(
                        in_states[workspace.active_index[successor]],
                        std::move(next),
                        result.predicate_widening_count) &&
                    workspace.queued_epoch[successor] != q_gen) {
                    workspace.queued_epoch[successor] = q_gen;
                    worklist.push(successor);
                }
            }
        }
    }

    for (size_t active_idx = 0u;
         active_idx < active_count; ++active_idx) {
        if (problem.block_events[active_idx].count == 0u) { continue; }
        const auto block_id = problem.active_blocks[active_idx];
        for (auto state : in_states[active_idx]) {
            if (!consume_state_evaluation()) { return result; }
            if (!apply_block_transfer(
                    active_idx, block_id, state, &result.failing_read)) {
                result.failing_predicate_count =
                    state.cube.size();
                return result;
            }
        }
    }
    result.succeeded = true;
    return result;
}

[[nodiscard]] LifetimeProofResult prove_fresh_lifetime(
    BasicBlock *target, Instruction *insertion_instruction,
    const AllocaUseRegion &region,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    const CoroPredicateAnalysis &predicates,
    ReferenceArgumentEffectAnalysis &reference_effects,
    const InstructionLocationMap &instruction_locations,
    luisa::span<const size_t> atom_indices,
    size_t max_guarded_state_evaluations,
    LifetimeProofWorkspace &workspace,
    LifetimeProofTimings *timings) noexcept {
    using Clock = std::chrono::steady_clock;
    auto begin = timings == nullptr ?
                     Clock::time_point{} :
                     Clock::now();
    auto prepared = prepare_lifetime_proof_problem(
        target, insertion_instruction, region, graph,
        domain, reference_effects, instruction_locations,
        atom_indices, workspace, timings);
    if (timings != nullptr) {
        auto now = Clock::now();
        timings->problem_ms +=
            std::chrono::duration<double, std::milli>{now - begin}.count();
        begin = now;
    }
    if (prepared.fast_succeeded) {
        LifetimeProofResult result;
        result.succeeded = true;
        result.block_evaluation_count = prepared.fast_block_evaluations;
        return result;
    }
    const auto &problem = prepared.problem;
    auto unconditional = prove_unconditional_fresh_lifetime(
        problem, graph, domain, workspace);
    if (timings != nullptr) {
        auto now = Clock::now();
        timings->unconditional_ms +=
            std::chrono::duration<double, std::milli>{now - begin}.count();
        begin = now;
    }
    if (unconditional.succeeded) { return unconditional; }
    auto guarded = prove_guarded_fresh_lifetime(
        problem, graph, domain, predicates,
        instruction_locations, workspace,
        max_guarded_state_evaluations);
    if (timings != nullptr) {
        timings->guarded_ms +=
            std::chrono::duration<double, std::milli>{
                Clock::now() - begin}
                .count();
    }
    guarded.block_evaluation_count =
        unconditional.block_evaluation_count;
    return guarded;
}

}

}// namespace detail

CoroAllocaScopeInfo coro_alloca_scope_pass_run_on_function(
    Function *function,
    const CoroAllocaScopeOptions &options) noexcept {
    using namespace detail::coro_alloca_scope_detail;
    CoroAllocaScopeInfo info;
    auto *definition =
        function == nullptr ? nullptr : function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return info;
    }
    detail::CoroSemanticGraph graph{definition};
    if (!graph.valid()) {
        info.invalid_semantic_cfg_count = 1u;
        return info;
    }
    info.semantic_block_count = graph.block_count();
    info.semantic_edge_count = graph.edge_count();
    detail::CoroPredicateAnalysis predicates{graph};
    detail::coro_alloca_scope_detail::ReferenceArgumentEffectAnalysis reference_effects;
    auto instruction_locations =
        detail::coro_alloca_scope_detail::make_instruction_locations(definition, graph);
    const auto dump_scope_rejections = []() noexcept {
        if (auto value = std::getenv(
                "LUISA_CORO_DUMP_ALLOCA_SCOPE")) {
            return luisa::string_view{value} == "1";
        }
        return false;
    }();
    const auto profile_compilation = []() noexcept {
        if (auto value = std::getenv(
                "LUISA_CORO_PROFILE_COMPILATION")) {
            return luisa::string_view{value} == "1";
        }
        return false;
    }();
    using ProfileClock = std::chrono::steady_clock;
    using ProfileTick = ProfileClock::time_point;
    const auto profile_begin = [profile_compilation]() noexcept {
        return profile_compilation ?
                   ProfileClock::now() :
                   ProfileTick{};
    };
    const auto profile_elapsed_ms =
        [profile_compilation](ProfileTick begin) noexcept {
            return profile_compilation ?
                       std::chrono::duration<double, std::milli>{
                           ProfileClock::now() - begin}
                           .count() :
                       0.0;
        };
    auto collect_region_ms = 0.0;
    auto first_definition_ms = 0.0;
    auto placement_ms = 0.0;
    auto proof_ms = 0.0;
    auto mutation_ms = 0.0;
    detail::coro_alloca_scope_detail::LifetimeProofTimings proof_timings;

    // Reuse the same type-shaped May/Must partition as coroutine liveness.
    // The definite-initialization proof and the eventual frame transfer must
    // agree on exactly which static subaggregates a store covers.
    detail::CoroFrameAtomDomain frame_domain{definition};
    luisa::unordered_map<AllocaInst *, luisa::vector<size_t>> alloca_atoms;
    for (size_t i = 0u; i < frame_domain.size(); ++i) {
        auto *root = frame_domain.atom(i).root;
        if (root != nullptr && root->isa<AllocaInst>()) {
            alloca_atoms[static_cast<AllocaInst *>(root)].emplace_back(i);
        }
    }

    // Freeze the candidate set before moving intrusive-list nodes.
    luisa::vector<AllocaInst *> allocas;
    for (auto *block : definition->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (instruction->isa<AllocaInst>() &&
                static_cast<AllocaInst *>(instruction)->is_local()) {
                allocas.emplace_back(
                    static_cast<AllocaInst *>(instruction));
            }
        }
    }
    detail::coro_alloca_scope_detail::LifetimeProofWorkspace proof_workspace{
        graph.block_count(), predicates.term_count()};
    luisa::optional<detail::CoroInitializedPrefixAnalysis> initialized_prefixes;
    luisa::optional<detail::CoroDiscriminatedPrefixAnalysis> discriminated_prefixes;
    auto log_rejection = [&](AllocaInst *alloca,
                             BasicBlock *source,
                             BasicBlock *target,
                             Instruction *failing_read,
                             size_t failing_predicate_count,
                             size_t atom_count,
                             size_t pointer_count,
                             size_t user_count,
                             size_t region_block_count) noexcept {
        if (!dump_scope_rejections) { return; }
        const auto alloca_name = alloca->name().value_or("<unnamed>");
        const auto read_name =
            failing_read == nullptr ?
                luisa::string_view{"<none>"} :
                failing_read->name().value_or("<unnamed>");
        const auto read_kind =
            failing_read == nullptr ?
                luisa::string_view{"<none>"} :
                to_string(failing_read->derived_instruction_tag());
        const auto callee_name =
            failing_read != nullptr &&
                    failing_read->isa<CallInst>() &&
                    static_cast<CallInst *>(failing_read)->callee() != nullptr ?
                static_cast<CallInst *>(failing_read)
                    ->callee()
                    ->name()
                    .value_or("<unnamed>") :
                luisa::string_view{"<none>"};
        LUISA_INFO(
            "Coroutine alloca lifetime rejection: name='{}' "
            "type={} source_block={} target_block={} atoms={} "
            "pointers={} users={} use_blocks={} failing_read='{}' "
            "failing_kind={} callee='{}' guard_predicates={}.",
            alloca_name,
            alloca->type()->description(),
            graph.block_id(source),
            graph.block_id(target),
            atom_count, pointer_count, user_count, region_block_count,
            read_name, read_kind, callee_name, failing_predicate_count);
    };
    for (auto *alloca : allocas) {
        ++info.scanned_local_alloca_count;
        auto phase_begin = profile_begin();
        auto region = detail::coro_alloca_scope_detail::collect_alloca_use_region(
            alloca, definition, graph);
        collect_region_ms += profile_elapsed_ms(phase_begin);
        if (!region.valid) {
            ++info.rejected_unreachable_use_count;
            continue;
        }
        if (region.contains_phi_use) {
            ++info.rejected_phi_use_count;
            continue;
        }
        if (region.blocks.empty()) { continue; }

        phase_begin = profile_begin();
        auto first_definition =
            detail::coro_alloca_scope_detail::plan_first_definition_delay(
                alloca, region, graph, instruction_locations,
                options.verify_instruction_order,
                info.instruction_order_query_count,
                info.placement_user_inspection_count);
        first_definition_ms += profile_elapsed_ms(phase_begin);
        if (first_definition.definition != nullptr) {
            auto *source = alloca->parent_block();
            phase_begin = profile_begin();
            detail::coro_alloca_scope_detail::apply_first_definition_plan(
                alloca, first_definition);
            auto moved_bool_store =
                alloca->type() != nullptr && alloca->type() == Type::of<bool>();
            if (initialized_prefixes) {
                initialized_prefixes->notify_target_mutated(
                    first_definition.target, true, moved_bool_store);
            }
            if (discriminated_prefixes) {
                discriminated_prefixes->notify_target_mutated(
                    first_definition.target, true, moved_bool_store);
            }
            mutation_ms += profile_elapsed_ms(phase_begin);
            ++info.contracted_alloca_count;
            ++info.delayed_first_definition_count;
            if (source == first_definition.target) {
                ++info.intra_block_contraction_count;
                ++info.intra_block_first_definition_delay_count;
            } else {
                ++info.cross_block_contraction_count;
                ++info.cross_block_first_definition_delay_count;
            }
            continue;
        }

        phase_begin = profile_begin();
        auto *target = graph.nearest_common_dominator(
            luisa::span{region.blocks});
        auto *source = alloca->parent_block();
        if (target == nullptr || source == nullptr ||
            !graph.dominates(source, target)) {
            ++info.rejected_non_dominating_alloca_count;
            continue;
        }
        auto insertion = find_latest_insertion_point(
            target, alloca, region.users,
            instruction_locations,
            options.verify_instruction_order,
            info.instruction_order_query_count,
            info.placement_user_inspection_count);
        placement_ms += profile_elapsed_ms(phase_begin);
        if (insertion.instruction == nullptr) {
            ++info.rejected_unreachable_use_count;
            continue;
        }
        if (source == target) {
            if (!insertion.follows_alloca) {
                ++info.rejected_non_dominating_alloca_count;
                continue;
            }
            if (!insertion.has_gap_after_alloca) { continue; }
        } else {
            auto atom_iter = alloca_atoms.find(alloca);
            auto atom_indices = atom_iter == alloca_atoms.end() ?
                                    luisa::span<const size_t>{} :
                                    luisa::span<const size_t>{
                                        atom_iter->second};
            phase_begin = profile_begin();
            auto proof = prove_fresh_lifetime(
                target, insertion.instruction, region, graph,
                frame_domain, predicates, reference_effects,
                instruction_locations,
                atom_indices,
                options.max_guarded_state_evaluations,
                proof_workspace,
                profile_compilation ? &proof_timings : nullptr);
            proof_ms += profile_elapsed_ms(phase_begin);
            info.definite_initialization_block_evaluation_count +=
                proof.block_evaluation_count;
            info.guarded_initialization_state_evaluation_count +=
                proof.guarded_state_evaluation_count;
            info.guarded_initialization_budget_exhausted_count +=
                proof.guarded_budget_exhausted ? 1u : 0u;
            info.predicate_widening_count +=
                proof.predicate_widening_count;
            if (!proof.succeeded && alloca->type() != nullptr &&
                alloca->type()->tag() == Type::Tag::ARRAY &&
                alloca->type()->dimension() != 0u) {
                phase_begin = profile_begin();
                if (!initialized_prefixes) {
                    initialized_prefixes.emplace(
                        definition, graph, frame_domain);
                }
                auto prefix_proof = initialized_prefixes->prove(
                    alloca, target, insertion.instruction);
                proof_ms += profile_elapsed_ms(phase_begin);
                info.initialized_prefix_block_evaluation_count +=
                    prefix_proof.block_evaluation_count;
                if (prefix_proof.succeeded) {
                    proof.succeeded = true;
                    proof.guarded = false;
                    proof.failing_read = nullptr;
                    target = prefix_proof.placement_block;
                    insertion.instruction =
                        prefix_proof.placement_instruction;
                    ++info.initialized_prefix_proof_count;
                } else if (prefix_proof.failing_read != nullptr) {
                    proof.failing_read = prefix_proof.failing_read;
                }
            }
            if (!proof.succeeded && alloca->type() != nullptr &&
                alloca->type()->tag() == Type::Tag::ARRAY &&
                alloca->type()->dimension() != 0u) {
                phase_begin = profile_begin();
                if (!discriminated_prefixes) {
                    discriminated_prefixes.emplace(definition, graph);
                }
                auto discriminated_proof = discriminated_prefixes->prove(
                    alloca, target, insertion.instruction);
                proof_ms += profile_elapsed_ms(phase_begin);
                info.discriminated_prefix_candidate_count +=
                    discriminated_proof.candidate_count;
                info.discriminated_prefix_rejected_missing_publication_count +=
                    discriminated_proof.rejected_missing_publication_count;
                info.discriminated_prefix_block_evaluation_count +=
                    discriminated_proof.block_evaluation_count;
                if (discriminated_proof.succeeded) {
                    proof.succeeded = true;
                    proof.guarded = false;
                    proof.failing_read = nullptr;
                    target = discriminated_proof.placement_block;
                    insertion.instruction =
                        discriminated_proof.placement_instruction;
                    ++info.discriminated_prefix_proof_count;
                } else if (discriminated_proof.failing_read != nullptr) {
                    proof.failing_read =
                        discriminated_proof.failing_read;
                }
            }
            if (!proof.succeeded) {
                log_rejection(
                    alloca, source, target,
                    proof.failing_read, proof.failing_predicate_count,
                    atom_indices.size(), region.pointers.size(),
                    region.users.size(), region.blocks.size());
                ++info.rejected_prior_lifetime_observation_count;
                continue;
            }
            ++info.definite_initialization_proof_count;
            if (proof.guarded) {
                ++info.guarded_initialization_proof_count;
            }
        }

        phase_begin = profile_begin();
        auto owned = alloca->remove_self();
        auto *moved = insertion.instruction->insert_before_self(
            std::move(owned));
        LUISA_DEBUG_ASSERT(moved == alloca,
                           "Alloca scope contraction changed identity.");
        if (initialized_prefixes) {
            initialized_prefixes->notify_target_mutated(target, false, false);
        }
        if (discriminated_prefixes) {
            discriminated_prefixes->notify_target_mutated(target, false, false);
        }
        mutation_ms += profile_elapsed_ms(phase_begin);
        ++info.contracted_alloca_count;
        if (source == target) {
            ++info.intra_block_contraction_count;
        } else {
            ++info.cross_block_contraction_count;
        }
    }

    // `store local, undef` anchors a fresh lexical lifetime for the analysis
    // above; it is not a required device write. Once final alloca placement
    // records that boundary, leaving the previous physical bits untouched is
    // one valid refinement of the arbitrary value. Erase the marker here so
    // aggregate seeds can never turn into zero-fill loops in a direct backend.
    // This scan deliberately happens after every candidate proof: removing
    // instructions earlier would invalidate the immutable location snapshot.
    luisa::vector<StoreInst *> undefined_lifetime_seeds;
    for (auto *block : definition->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (!instruction->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(instruction);
            auto *destination = store->variable();
            if (destination != nullptr &&
                destination->isa<AllocaInst>() &&
                static_cast<AllocaInst *>(destination)->is_local() &&
                store->value() != nullptr &&
                store->value()->derived_value_tag() ==
                    DerivedValueTag::UNDEFINED) {
                undefined_lifetime_seeds.emplace_back(store);
            }
        }
    }
    for (auto *seed : undefined_lifetime_seeds) {
        static_cast<void>(seed->remove_self());
        ++info.removed_undefined_lifetime_seed_count;
    }
    if (profile_compilation) {
        LUISA_INFO(
            "Coroutine alloca lifetime timing: region={:.3f} ms "
            "first_definition={:.3f} ms placement={:.3f} ms "
            "order_queries={} user_inspections={} "
            "proof={:.3f} ms (problem={:.3f} ms: slice={:.3f} ms "
            "layout={:.3f} ms events={:.3f} ms (order={:.3f} ms "
            "transfer={:.3f} ms, reference_effects={} functions/{:.3f} ms); "
            "unconditional={:.3f} ms "
            "guarded={:.3f} ms) mutation={:.3f} ms.",
            collect_region_ms, first_definition_ms, placement_ms,
            info.instruction_order_query_count,
            info.placement_user_inspection_count,
            proof_ms, proof_timings.problem_ms, proof_timings.slice_ms,
            proof_timings.layout_ms, proof_timings.events_ms,
            proof_timings.event_order_ms,
            proof_timings.event_transfer_ms,
            reference_effects.analysis_count(),
            reference_effects.analysis_ms(),
            proof_timings.unconditional_ms, proof_timings.guarded_ms,
            mutation_ms);
    }
    return info;
}

}// namespace luisa::compute::xir
