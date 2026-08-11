#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/coro_alloca_scope.h>

#include "coro_frame_access.h"
#include "coro_semantic_graph.h"

namespace luisa::compute::xir {

namespace detail {

namespace {

struct AllocaUseRegion {
    bool valid{true};
    bool contains_phi_use{false};
    luisa::unordered_set<Value *> pointers;
    luisa::unordered_set<Instruction *> users;
    luisa::vector<BasicBlock *> blocks;
};

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

[[nodiscard]] InsertionPoint find_latest_insertion_point(
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

enum class LifetimeEventKind : uint8_t {
    redefine_pointer,
    store,
    read
};

struct LifetimeEvent {
    LifetimeEventKind kind;
    Value *pointer;
};

struct LifetimeFactLayout {
    luisa::unordered_map<size_t, size_t> atom_facts;
    luisa::unordered_map<Value *, size_t> pointer_facts;
    size_t fact_count{0u};
};

struct LifetimeProofResult {
    bool succeeded{false};
    size_t block_evaluation_count{0u};
};

using LifetimeFactState = luisa::vector<uint8_t>;

[[nodiscard]] LifetimeFactLayout make_lifetime_fact_layout(
    luisa::span<const size_t> atom_indices,
    const AllocaUseRegion &region) noexcept {
    LifetimeFactLayout layout;
    layout.atom_facts.reserve(atom_indices.size());
    layout.pointer_facts.reserve(region.pointers.size());
    for (auto atom : atom_indices) {
        layout.atom_facts.emplace(atom, layout.fact_count++);
    }
    for (auto *pointer : region.pointers) {
        layout.pointer_facts.emplace(pointer, layout.fact_count++);
    }
    return layout;
}

void redefine_pointer(Value *pointer, LifetimeFactState &state,
                      const LifetimeFactLayout &layout) noexcept {
    if (auto iter = layout.pointer_facts.find(pointer);
        iter != layout.pointer_facts.end()) {
        state[iter->second] = 0u;
    }
}

void define_pointer(Value *pointer, LifetimeFactState &state,
                    const LifetimeFactLayout &layout,
                    const CoroFrameAtomDomain &domain) noexcept {
    if (auto iter = layout.pointer_facts.find(pointer);
        iter != layout.pointer_facts.end()) {
        // A typed XIR store overwrites the complete object denoted by this
        // dynamic pointer version, even when no fixed aggregate leaf is a
        // Must target (for example array[i]).
        state[iter->second] = 1u;
    }
    for (auto access : domain.memory_accesses(pointer)) {
        if (!access.covers_atom) { continue; }
        if (auto iter = layout.atom_facts.find(access.atom_index);
            iter != layout.atom_facts.end()) {
            state[iter->second] = 1u;
        }
    }
}

[[nodiscard]] bool pointer_is_defined(
    Value *pointer, const LifetimeFactState &state,
    const LifetimeFactLayout &layout,
    const CoroFrameAtomDomain &domain) noexcept {
    if (auto iter = layout.pointer_facts.find(pointer);
        iter != layout.pointer_facts.end() && state[iter->second] != 0u) {
        return true;
    }
    auto has_relevant_atom = false;
    for (auto access : domain.memory_accesses(pointer)) {
        if (auto iter = layout.atom_facts.find(access.atom_index);
            iter != layout.atom_facts.end()) {
            has_relevant_atom = true;
            if (state[iter->second] == 0u) { return false; }
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
    bool validate_reads) noexcept {
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
                    return false;
                }
                break;
        }
    }
    return true;
}

[[nodiscard]] LifetimeProofResult prove_fresh_lifetime(
    BasicBlock *target, Instruction *insertion_instruction,
    const AllocaUseRegion &region,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    luisa::span<const size_t> atom_indices) noexcept {
    LifetimeProofResult result;
    auto target_id = graph.block_id(target);
    if (target_id >= graph.block_count() ||
        insertion_instruction == nullptr) {
        return result;
    }

    // Restrict the proof to the reverse slice from every pointer use to the
    // proposed lifetime start. Since target dominates every use, this slice
    // contains every executable path that can affect an observation before
    // target is reached again; target itself is a reset boundary.
    luisa::vector<uint8_t> active(graph.block_count(), 0u);
    luisa::vector<size_t> worklist;
    for (auto *block : region.blocks) {
        if (!graph.dominates(target, block)) { return result; }
        auto id = graph.block_id(block);
        if (id >= graph.block_count()) { return result; }
        if (active[id] == 0u) {
            active[id] = 1u;
            worklist.emplace_back(id);
        }
    }
    active[target_id] = 1u;
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto block_id = worklist[cursor];
        if (block_id == target_id) { continue; }
        for (auto predecessor : graph.predecessors(block_id)) {
            auto *predecessor_block = graph.block(predecessor);
            if (!graph.dominates(target, predecessor_block)) {
                return result;
            }
            if (active[predecessor] == 0u) {
                active[predecessor] = 1u;
                worklist.emplace_back(predecessor);
            }
        }
    }

    auto layout = make_lifetime_fact_layout(atom_indices, region);
    if (layout.fact_count == 0u) { return result; }
    luisa::vector<luisa::vector<LifetimeEvent>> events(
        graph.block_count());
    luisa::vector<size_t> active_blocks;
    for (size_t block_id = 0u;
         block_id < graph.block_count(); ++block_id) {
        if (active[block_id] == 0u) { continue; }
        active_blocks.emplace_back(block_id);
        auto *block = graph.block(block_id);
        auto after_lifetime_start = block_id != target_id;
        auto found_lifetime_start = after_lifetime_start;
        for (auto *instruction : block->instructions()) {
            if (!after_lifetime_start) {
                if (instruction != insertion_instruction) { continue; }
                after_lifetime_start = true;
                found_lifetime_start = true;
            }
            if (!region.users.contains(instruction)) { continue; }
            auto &block_events = events[block_id];
            if (instruction->isa<GEPInst>()) {
                block_events.emplace_back(LifetimeEvent{
                    LifetimeEventKind::redefine_pointer, instruction});
                continue;
            }
            if (instruction->isa<LoadInst>()) {
                auto *pointer =
                    static_cast<LoadInst *>(instruction)->variable();
                if (!region.pointers.contains(pointer)) { return result; }
                block_events.emplace_back(LifetimeEvent{
                    LifetimeEventKind::read, pointer});
                continue;
            }
            if (instruction->isa<StoreInst>()) {
                auto *pointer =
                    static_cast<StoreInst *>(instruction)->variable();
                if (!region.pointers.contains(pointer)) { return result; }
                block_events.emplace_back(LifetimeEvent{
                    LifetimeEventKind::store, pointer});
                continue;
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
                // Atomics, reference calls, and unknown pointer operations
                // may observe the old value before any possible write.
                block_events.emplace_back(LifetimeEvent{
                    LifetimeEventKind::read, pointer});
            }
            if (!found_pointer_operand) { return result; }
        }
        if (!found_lifetime_start) { return result; }
    }

    auto top = LifetimeFactState(layout.fact_count, uint8_t{1u});
    auto bottom = LifetimeFactState(layout.fact_count, uint8_t{0u});
    luisa::vector<LifetimeFactState> in_states(graph.block_count());
    luisa::vector<LifetimeFactState> out_states(graph.block_count());
    for (auto block_id : active_blocks) {
        in_states[block_id] = top;
        out_states[block_id] = top;
    }

    // This is the greatest fixed point of the forward Must equations:
    //   IN[target] = empty
    //   IN[b]      = intersection OUT[p]
    //   OUT[b]     = GEN[b] union (IN[b] - exact-GEP-redefinitions[b]).
    // Starting at top makes every update descending, so loop convergence is
    // finite and no traversal order can invent a definite initialization.
    for (;;) {
        auto changed = false;
        for (auto block_id : active_blocks) {
            ++result.block_evaluation_count;
            LifetimeFactState next_in;
            if (block_id == target_id) {
                next_in = bottom;
            } else {
                auto first_predecessor = true;
                for (auto predecessor : graph.predecessors(block_id)) {
                    if (active[predecessor] == 0u) { return result; }
                    if (first_predecessor) {
                        next_in = out_states[predecessor];
                        first_predecessor = false;
                    } else {
                        for (size_t i = 0u; i < layout.fact_count; ++i) {
                            next_in[i] &= out_states[predecessor][i];
                        }
                    }
                }
                if (first_predecessor) { return result; }
            }
            auto next_out = next_in;
            static_cast<void>(apply_lifetime_events(
                events[block_id], next_out, layout, domain, false));
            if (in_states[block_id] != next_in ||
                out_states[block_id] != next_out) {
                in_states[block_id] = std::move(next_in);
                out_states[block_id] = std::move(next_out);
                changed = true;
            }
        }
        if (!changed) { break; }
    }

    for (auto block_id : active_blocks) {
        auto state = in_states[block_id];
        if (!apply_lifetime_events(
                events[block_id], state, layout, domain, true)) {
            return result;
        }
    }
    result.succeeded = true;
    return result;
}

}

}// namespace detail

CoroAllocaScopeInfo coro_alloca_scope_pass_run_on_function(
    Function *function) noexcept {
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

    for (auto *alloca : allocas) {
        ++info.scanned_local_alloca_count;
        auto region = detail::collect_alloca_use_region(
            alloca, definition, graph);
        if (!region.valid) {
            ++info.rejected_unreachable_use_count;
            continue;
        }
        if (region.contains_phi_use) {
            ++info.rejected_phi_use_count;
            continue;
        }
        if (region.blocks.empty()) { continue; }
        auto *target = graph.nearest_common_dominator(
            luisa::span{region.blocks});
        auto *source = alloca->parent_block();
        if (target == nullptr || source == nullptr ||
            !graph.dominates(source, target)) {
            ++info.rejected_non_dominating_alloca_count;
            continue;
        }
        auto insertion = detail::find_latest_insertion_point(
            target, alloca, region.users);
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
            auto proof = detail::prove_fresh_lifetime(
                target, insertion.instruction, region, graph,
                frame_domain, atom_indices);
            info.definite_initialization_block_evaluation_count +=
                proof.block_evaluation_count;
            if (!proof.succeeded) {
                ++info.rejected_prior_lifetime_observation_count;
                continue;
            }
            ++info.definite_initialization_proof_count;
        }

        auto owned = alloca->remove_self();
        auto *moved = insertion.instruction->insert_before_self(
            std::move(owned));
        LUISA_DEBUG_ASSERT(moved == alloca,
                           "Alloca scope contraction changed identity.");
        ++info.contracted_alloca_count;
        if (source == target) {
            ++info.intra_block_contraction_count;
        } else {
            ++info.cross_block_contraction_count;
        }
    }
    return info;
}

}// namespace luisa::compute::xir
