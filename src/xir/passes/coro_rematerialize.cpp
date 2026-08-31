#include <algorithm>
#include <cstdint>

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/op.h>
#include <luisa/xir/passes/coro_rematerialize.h>

#include "coro_replayable.h"
#include "coro_semantic_graph.h"

namespace luisa::compute::xir {

namespace detail {

namespace {

struct LoadProjection {
    LoadInst *load{nullptr};
    luisa::vector<Value *> indices;
    Value *reaching_value{nullptr};
    bool reaches_across_suspend{false};
};

struct LocalStateCandidate {
    AllocaInst *alloca{nullptr};
    luisa::vector<StoreInst *> stores;
    luisa::vector<LoadProjection> loads;
    size_t replay_cost{0u};
    bool replay_stored_values{false};
};

enum class ReachingValueTag : uint8_t {
    PENDING,
    UNDEFINED,
    UNIQUE,
    CONFLICT,
};

struct ReachingValue {
    ReachingValueTag tag{ReachingValueTag::PENDING};
    Value *value{nullptr};
    bool crossed_suspend{false};

    [[nodiscard]] static ReachingValue undefined() noexcept {
        return ReachingValue{
            ReachingValueTag::UNDEFINED, nullptr, false};
    }

    [[nodiscard]] static ReachingValue unique(
        Value *value, bool crossed_suspend = false) noexcept {
        return ReachingValue{
            ReachingValueTag::UNIQUE, value, crossed_suspend};
    }

    [[nodiscard]] static ReachingValue conflict() noexcept {
        return ReachingValue{
            ReachingValueTag::CONFLICT, nullptr, false};
    }

    [[nodiscard]] friend bool operator==(
        ReachingValue lhs, ReachingValue rhs) noexcept {
        return lhs.tag == rhs.tag && lhs.value == rhs.value &&
               lhs.crossed_suspend == rhs.crossed_suspend;
    }
};

[[nodiscard]] ReachingValue meet_reaching_values(
    ReachingValue lhs, ReachingValue rhs) noexcept {
    if (lhs.tag == ReachingValueTag::PENDING) { return rhs; }
    if (rhs.tag == ReachingValueTag::PENDING) { return lhs; }
    if (lhs.tag == ReachingValueTag::UNIQUE &&
        rhs.tag == ReachingValueTag::UNIQUE &&
        lhs.value == rhs.value) {
        return ReachingValue::unique(
            lhs.value,
            lhs.crossed_suspend || rhs.crossed_suspend);
    }
    if (lhs == rhs) { return lhs; }
    return ReachingValue::conflict();
}

// NAME, LOCATION, and COMMENT metadata describe presentation rather than
// runtime semantics, so they cannot invalidate an otherwise exact load
// substitution. Other kinds fail closed: REG2MEM_SPILL, for example, carries
// a structural compiler contract, and future kinds require a separate audit.
[[nodiscard]] bool has_only_nonsemantic_metadata(
    const MetadataListMixin &value) noexcept {
    for (auto *metadata : value.metadata_list()) {
        switch (metadata->derived_metadata_tag()) {
            case DerivedMetadataTag::NAME:
            case DerivedMetadataTag::LOCATION:
            case DerivedMetadataTag::COMMENT:
                break;
            case DerivedMetadataTag::CURVE_BASIS:
            case DerivedMetadataTag::SIGNATURE_CONSTRAINT:
            case DerivedMetadataTag::REG2MEM_SPILL: return false;
        }
    }
    return true;
}// namespace

[[nodiscard]] bool collect_projection_indices(
    AllocaInst *alloca, Value *pointer,
    luisa::vector<Value *> &indices) noexcept {
    luisa::vector<Value *> reversed_indices;
    luisa::unordered_set<Value *> visited;
    while (pointer != alloca) {
        if (pointer == nullptr || !pointer->isa<GEPInst>() ||
            !visited.emplace(pointer).second) {
            return false;
        }
        auto *gep = static_cast<GEPInst *>(pointer);
        auto index_uses = gep->index_uses();
        for (auto iter = index_uses.rbegin();
             iter != index_uses.rend(); ++iter) {
            auto *index = (*iter)->value();
            if (index == nullptr) { return false; }
            reversed_indices.emplace_back(index);
        }
        pointer = gep->base();
    }
    indices.assign(reversed_indices.rbegin(), reversed_indices.rend());
    return true;
}

[[nodiscard]] bool collect_local_state_candidate(
    AllocaInst *alloca, LocalStateCandidate &candidate) noexcept {
    if (alloca == nullptr || !alloca->is_local() ||
        !has_only_nonsemantic_metadata(*alloca)) {
        return false;
    }
    candidate.alloca = alloca;
    luisa::vector<Value *> worklist;
    luisa::unordered_set<Value *> visited;
    worklist.emplace_back(alloca);
    while (!worklist.empty()) {
        auto *pointer = worklist.back();
        worklist.pop_back();
        if (!visited.emplace(pointer).second) { continue; }
        for (auto *use : pointer->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                return false;
            }
            auto *instruction = static_cast<Instruction *>(user);
            if (instruction->isa<GEPInst>()) {
                auto *gep = static_cast<GEPInst *>(instruction);
                if (gep->base() != pointer ||
                    !has_only_nonsemantic_metadata(*gep)) {
                    return false;
                }
                worklist.emplace_back(gep);
                continue;
            }
            if (instruction->isa<LoadInst>()) {
                auto *load = static_cast<LoadInst *>(instruction);
                if (load->variable() != pointer ||
                    !has_only_nonsemantic_metadata(*load)) {
                    return false;
                }
                LoadProjection projection{.load = load};
                if (!collect_projection_indices(
                        alloca, pointer, projection.indices)) {
                    return false;
                }
                candidate.loads.emplace_back(std::move(projection));
                continue;
            }
            if (instruction->isa<StoreInst>()) {
                auto *store = static_cast<StoreInst *>(instruction);
                if (store->variable() != pointer || pointer != alloca ||
                    !has_only_nonsemantic_metadata(*store)) {
                    return false;
                }
                candidate.stores.emplace_back(store);
                continue;
            }
            return false;
        }
    }
    if (candidate.stores.empty() || candidate.loads.empty()) {
        return false;
    }
    for (auto *store : candidate.stores) {
        if (store->value() == nullptr ||
            store->value()->type() != alloca->type() ||
            store->value()->is_lvalue()) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] size_t resolve_reaching_values_dense_oracle(
    const CoroSemanticGraph &graph,
    const luisa::unordered_map<Instruction *, size_t> &instruction_indices,
    LocalStateCandidate &candidate,
    size_t &block_evaluation_count) noexcept {
    auto block_count = graph.block_count();
    // The transfer function depends only on this candidate's stores. Index
    // its sparse event stream once instead of rescanning every instruction in
    // every block on each fixed-point evaluation.
    luisa::vector<luisa::vector<Instruction *>> block_events(block_count);
    for (auto *store : candidate.stores) {
        auto block_id = graph.block_id(store->parent_block());
        if (block_id < block_count) {
            block_events[block_id].emplace_back(store);
        }
    }
    for (auto &projection : candidate.loads) {
        auto block_id = graph.block_id(
            projection.load->parent_block());
        if (block_id < block_count) {
            block_events[block_id].emplace_back(projection.load);
        }
    }
    luisa::vector<StoreInst *> last_stores(block_count, nullptr);
    for (size_t block_id = 0u; block_id < block_count; ++block_id) {
        auto &events = block_events[block_id];
        std::sort(
            events.begin(), events.end(),
            [&](Instruction *lhs, Instruction *rhs) noexcept {
                return instruction_indices.at(lhs) <
                       instruction_indices.at(rhs);
            });
        for (auto *event : events) {
            if (event->isa<StoreInst>()) {
                last_stores[block_id] =
                    static_cast<StoreInst *>(event);
            }
        }
    }
    luisa::vector<ReachingValue> block_inputs(block_count);
    luisa::vector<ReachingValue> block_outputs(block_count);
    luisa::vector<uint8_t> queued(block_count, 0u);
    luisa::vector<size_t> worklist;
    worklist.emplace_back(0u);
    queued.front() = 1u;
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto block_id = worklist[cursor];
        ++block_evaluation_count;
        queued[block_id] = 0u;
        auto incoming = block_id == 0u ?
                            ReachingValue::undefined() :
                            ReachingValue{};
        for (auto predecessor : graph.predecessors(block_id)) {
            auto predecessor_output = block_outputs[predecessor];
            if (predecessor_output.tag ==
                    ReachingValueTag::UNIQUE &&
                graph.is_suspend_edge(predecessor, block_id)) {
                predecessor_output.crossed_suspend = true;
            }
            incoming = meet_reaching_values(
                incoming, predecessor_output);
        }
        block_inputs[block_id] = incoming;
        auto outgoing = last_stores[block_id] == nullptr ?
                            incoming :
                            ReachingValue::unique(
                                last_stores[block_id]->value());
        if (outgoing == block_outputs[block_id]) { continue; }
        block_outputs[block_id] = outgoing;
        for (auto successor : graph.successors(block_id)) {
            if (queued[successor] == 0u) {
                queued[successor] = 1u;
                worklist.emplace_back(successor);
            }
        }
    }

    luisa::unordered_map<LoadInst *, size_t> load_indices;
    load_indices.reserve(candidate.loads.size());
    for (size_t i = 0u; i < candidate.loads.size(); ++i) {
        load_indices.emplace(candidate.loads[i].load, i);
    }
    auto unresolved = size_t{0u};
    for (size_t block_id = 0u; block_id < block_count; ++block_id) {
        auto state = block_inputs[block_id];
        for (auto *instruction : block_events[block_id]) {
            if (instruction->isa<StoreInst>()) {
                auto *store = static_cast<StoreInst *>(instruction);
                if (store->variable() == candidate.alloca) {
                    state = ReachingValue::unique(store->value());
                }
            } else if (instruction->isa<LoadInst>()) {
                auto *load = static_cast<LoadInst *>(instruction);
                if (auto iter = load_indices.find(load);
                    iter != load_indices.end()) {
                    if (state.tag == ReachingValueTag::UNIQUE) {
                        candidate.loads[iter->second].reaching_value =
                            state.value;
                        candidate.loads[iter->second].reaches_across_suspend =
                            state.crossed_suspend;
                    }
                }
            }
        }
    }
    // A use outside the executable semantic graph is conservatively
    // unresolved. Pre-distill CFG cleanup normally removes such blocks, but
    // the transform must remain atomic for malformed or diagnostic input.
    for (auto &projection : candidate.loads) {
        if (projection.reaching_value == nullptr) {
            ++unresolved;
        }
    }
    return unresolved;
}

struct ReachingEvent {
    Instruction *instruction{nullptr};
    LoadProjection *load{nullptr};
};

// Reused by all candidate coordinates. The reaching-value equations form a
// product lattice: one local alloca is one independent coordinate. Reusing
// the storage keeps projection cost proportional to the blocks actually
// touched by that coordinate instead of repeatedly constructing B vectors.
struct ReachingValueWorkspace {
    luisa::vector<luisa::vector<ReachingEvent>> block_events;
    luisa::vector<StoreInst *> last_stores;
    luisa::vector<ReachingValue> block_inputs;
    luisa::vector<ReachingValue> block_outputs;
    luisa::vector<uint8_t> event_block_touched;
    luisa::vector<uint8_t> active;
    luisa::vector<uint8_t> queued;
    luisa::vector<size_t> touched_event_blocks;
    luisa::vector<size_t> active_blocks;
    luisa::vector<size_t> reverse_worklist;
    luisa::vector<size_t> forward_worklist;

    explicit ReachingValueWorkspace(size_t block_count) noexcept
        : block_events{block_count},
          last_stores(block_count, nullptr),
          block_inputs(block_count),
          block_outputs(block_count),
          event_block_touched(block_count, 0u),
          active(block_count, 0u),
          queued(block_count, 0u) {}

    void reset() noexcept {
        for (auto block_id : touched_event_blocks) {
            block_events[block_id].clear();
            last_stores[block_id] = nullptr;
            event_block_touched[block_id] = 0u;
        }
        for (auto block_id : active_blocks) {
            block_inputs[block_id] = ReachingValue{};
            block_outputs[block_id] = ReachingValue{};
            active[block_id] = 0u;
            queued[block_id] = 0u;
        }
        touched_event_blocks.clear();
        active_blocks.clear();
        reverse_worklist.clear();
        forward_worklist.clear();
    }

    void add_event(size_t block_id, ReachingEvent event) noexcept {
        if (event_block_touched[block_id] == 0u) {
            event_block_touched[block_id] = 1u;
            touched_event_blocks.emplace_back(block_id);
        }
        block_events[block_id].emplace_back(event);
    }

    [[nodiscard]] bool mark_active(size_t block_id) noexcept {
        if (active[block_id] != 0u) { return false; }
        active[block_id] = 1u;
        active_blocks.emplace_back(block_id);
        reverse_worklist.emplace_back(block_id);
        return true;
    }
};

[[nodiscard]] size_t resolve_reaching_values_projected(
    const CoroSemanticGraph &graph,
    const luisa::unordered_map<Instruction *, size_t> &instruction_indices,
    ReachingValueWorkspace &workspace,
    LocalStateCandidate &candidate,
    size_t &active_block_count,
    size_t &block_evaluation_count) noexcept {
    workspace.reset();
    auto block_count = graph.block_count();
    for (auto *store : candidate.stores) {
        auto block_id = graph.block_id(store->parent_block());
        if (block_id < block_count) {
            workspace.add_event(
                block_id, ReachingEvent{store, nullptr});
        }
    }
    for (auto &projection : candidate.loads) {
        auto block_id = graph.block_id(
            projection.load->parent_block());
        if (block_id < block_count) {
            workspace.add_event(
                block_id,
                ReachingEvent{projection.load, &projection});
        }
    }

    // A load after a store in the same block is resolved locally. Only loads
    // before the first store demand the block input. A predecessor containing
    // a store is a boundary: its output is the last stored value, independent
    // of its input, so the backward projection stops there.
    for (auto block_id : workspace.touched_event_blocks) {
        auto &events = workspace.block_events[block_id];
        std::sort(
            events.begin(), events.end(),
            [&](const ReachingEvent &lhs,
                const ReachingEvent &rhs) noexcept {
                return instruction_indices.at(lhs.instruction) <
                       instruction_indices.at(rhs.instruction);
            });
        auto seen_store = false;
        for (auto event : events) {
            if (event.load == nullptr) {
                auto *store = static_cast<StoreInst *>(event.instruction);
                workspace.last_stores[block_id] = store;
                seen_store = true;
            } else if (!seen_store) {
                static_cast<void>(workspace.mark_active(block_id));
            }
        }
    }
    for (size_t cursor = 0u;
         cursor < workspace.reverse_worklist.size(); ++cursor) {
        auto block_id = workspace.reverse_worklist[cursor];
        // Match the original boundary condition exactly: entry is undefined
        // even if malformed input manufactures a predecessor edge to it.
        if (block_id == 0u) { continue; }
        for (auto predecessor : graph.predecessors(block_id)) {
            if (workspace.last_stores[predecessor] != nullptr) {
                continue;
            }
            static_cast<void>(workspace.mark_active(predecessor));
        }
    }
    active_block_count += workspace.active_blocks.size();

    // Semantic graph IDs are reverse postorder. Sorting the projected subset
    // preserves that order: every acyclic predecessor is evaluated before its
    // consumer, while ordinary worklist revisits retain exact loop semantics.
    std::sort(
        workspace.active_blocks.begin(),
        workspace.active_blocks.end());
    workspace.forward_worklist = workspace.active_blocks;
    for (auto block_id : workspace.active_blocks) {
        workspace.queued[block_id] = 1u;
        if (auto *store = workspace.last_stores[block_id]) {
            workspace.block_outputs[block_id] =
                ReachingValue::unique(store->value());
        }
    }
    for (size_t cursor = 0u;
         cursor < workspace.forward_worklist.size(); ++cursor) {
        auto block_id = workspace.forward_worklist[cursor];
        workspace.queued[block_id] = 0u;
        ++block_evaluation_count;
        auto incoming = block_id == 0u ?
                            ReachingValue::undefined() :
                            ReachingValue{};
        if (block_id != 0u) {
            for (auto predecessor : graph.predecessors(block_id)) {
                auto predecessor_output = ReachingValue{};
                if (auto *store = workspace.last_stores[predecessor]) {
                    predecessor_output =
                        ReachingValue::unique(store->value());
                } else {
                    LUISA_DEBUG_ASSERT(
                        workspace.active[predecessor] != 0u,
                        "Store-free predecessor must belong to the "
                        "backward-closed reaching-value projection.");
                    predecessor_output =
                        workspace.block_outputs[predecessor];
                }
                if (predecessor_output.tag ==
                        ReachingValueTag::UNIQUE &&
                    graph.is_suspend_edge(
                        predecessor, block_id)) {
                    predecessor_output.crossed_suspend = true;
                }
                incoming = meet_reaching_values(
                    incoming, predecessor_output);
            }
        }
        workspace.block_inputs[block_id] = incoming;
        auto outgoing = workspace.last_stores[block_id] == nullptr ?
                            incoming :
                            ReachingValue::unique(
                                workspace.last_stores[block_id]->value());
        if (outgoing == workspace.block_outputs[block_id]) {
            continue;
        }
        workspace.block_outputs[block_id] = outgoing;
        for (auto successor : graph.successors(block_id)) {
            if (workspace.active[successor] != 0u &&
                workspace.queued[successor] == 0u) {
                workspace.queued[successor] = 1u;
                workspace.forward_worklist.emplace_back(successor);
            }
        }
    }

    // Replay only sparse event blocks after convergence. A non-active event
    // block has no pre-store load by construction, so its initial pending
    // state is unobservable before the first overwriting store.
    for (auto block_id : workspace.touched_event_blocks) {
        auto state = workspace.active[block_id] != 0u ?
                         workspace.block_inputs[block_id] :
                         ReachingValue{};
        for (auto event : workspace.block_events[block_id]) {
            if (event.load == nullptr) {
                auto *store = static_cast<StoreInst *>(event.instruction);
                state = ReachingValue::unique(store->value());
            } else if (state.tag == ReachingValueTag::UNIQUE) {
                event.load->reaching_value = state.value;
                event.load->reaches_across_suspend =
                    state.crossed_suspend;
            }
        }
    }
    auto unresolved = size_t{0u};
    for (auto &projection : candidate.loads) {
        if (projection.reaching_value == nullptr) {
            ++unresolved;
        }
    }
    return unresolved;
}

[[nodiscard]] size_t resolve_reaching_values(
    const CoroSemanticGraph &graph,
    const luisa::unordered_map<Instruction *, size_t> &instruction_indices,
    ReachingValueWorkspace &workspace,
    LocalStateCandidate &candidate,
    bool verify_dense_oracle,
    size_t &active_block_count,
    size_t &block_evaluation_count) noexcept {
    LocalStateCandidate oracle_candidate;
    auto oracle_unresolved = size_t{0u};
    if (verify_dense_oracle) {
        oracle_candidate = candidate;
        auto oracle_evaluations = size_t{0u};
        oracle_unresolved = resolve_reaching_values_dense_oracle(
            graph, instruction_indices, oracle_candidate,
            oracle_evaluations);
    }
    auto unresolved = resolve_reaching_values_projected(
        graph, instruction_indices, workspace, candidate,
        active_block_count, block_evaluation_count);
    if (verify_dense_oracle) {
        LUISA_ASSERT(
            unresolved == oracle_unresolved &&
                candidate.loads.size() == oracle_candidate.loads.size(),
            "Projected and dense reaching-value analyses disagree on "
            "the unresolved load count.");
        for (size_t i = 0u; i < candidate.loads.size(); ++i) {
            auto &projected = candidate.loads[i];
            auto &dense = oracle_candidate.loads[i];
            LUISA_ASSERT(
                projected.load == dense.load &&
                    projected.reaching_value == dense.reaching_value &&
                    projected.reaches_across_suspend ==
                        dense.reaches_across_suspend,
                "Projected and dense reaching-value analyses disagree "
                "at load {}.",
                i);
        }
    }
    return unresolved;
}

[[nodiscard]] size_t resolve_single_store_loads(
    const CoroSemanticGraph &graph,
    const luisa::unordered_map<Instruction *, size_t> &instruction_indices,
    LocalStateCandidate &candidate) noexcept {
    LUISA_DEBUG_ASSERT(candidate.stores.size() == 1u,
                       "Single-store resolver requires one store.");
    auto *store = candidate.stores.front();
    auto *store_block = store->parent_block();
    auto unresolved = size_t{0u};
    for (auto &projection : candidate.loads) {
        auto *load_block = projection.load->parent_block();
        auto resolved = store_block != nullptr && load_block != nullptr &&
                        graph.contains(store_block) &&
                        graph.contains(load_block);
        if (resolved && store_block == load_block) {
            resolved = instruction_indices.at(store) <
                       instruction_indices.at(projection.load);
        } else if (resolved) {
            resolved = graph.dominates(store_block, load_block);
        }
        if (resolved) {
            projection.reaching_value = store->value();
            projection.reaches_across_suspend =
                graph.crosses_suspend_without_reentering(
                    store_block, load_block);
        } else {
            ++unresolved;
        }
    }
    return unresolved;
}

}

}// namespace detail

CoroRematerializeInfo
coro_rematerialize_local_state_pass_run_on_function(
    Function *function,
    const CoroRematerializeOptions &options) noexcept {
    CoroRematerializeInfo info;
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

    luisa::unordered_map<Instruction *, size_t> instruction_indices;
    luisa::vector<AllocaInst *> allocas;
    for (auto *block : definition->basic_blocks()) {
        auto instruction_index = size_t{0u};
        for (auto *instruction : block->instructions()) {
            instruction_indices.emplace(
                instruction, instruction_index++);
            if (instruction->isa<AllocaInst>() &&
                static_cast<AllocaInst *>(instruction)->is_local()) {
                allocas.emplace_back(
                    static_cast<AllocaInst *>(instruction));
            }
        }
    }

    detail::CoroReplayableValueAnalysis replayable;
    detail::ReachingValueWorkspace reaching_workspace{
        graph.block_count()};
    luisa::vector<detail::LocalStateCandidate> accepted;
    for (auto *alloca : allocas) {
        ++info.scanned_alloca_count;
        detail::LocalStateCandidate candidate;
        if (!detail::collect_local_state_candidate(
                alloca, candidate)) {
            continue;
        }
        auto all_store_values_replayable = true;
        for (auto *store : candidate.stores) {
            if (!replayable.detect(store->value())) {
                all_store_values_replayable = false;
                break;
            }
        }
        candidate.replay_stored_values =
            all_store_values_replayable;
        if (all_store_values_replayable) {
            if (candidate.stores.size() == 1u) {
                ++info.replayable_single_store_count;
            } else {
                ++info.replayable_multi_store_count;
            }
        } else {
            ++info.nonreplayable_candidate_count;
        }

        // Keep aggregate access-path state in memory unless it is replayable:
        // frame distillation can spill only the observed leaf, while direct
        // SSA forwarding would retain the complete non-replayable aggregate.
        if (!candidate.replay_stored_values &&
            std::any_of(
                candidate.loads.begin(), candidate.loads.end(),
                [](const auto &projection) noexcept {
                    return !projection.indices.empty();
                })) {
            ++info.rejected_nonreplayable_projection_count;
            continue;
        }

        // A non-replayable local is useful to version only if at least one
        // store/load pair can be separated by a semantic transfer edge. The
        // graph precomputes reverse reachability into every suspend and
        // forward reachability out of its resume, including self-resuming
        // loops. This necessary condition rejects scope-local temporaries
        // before the per-candidate reaching-value fixed point scans the IR.
        if (!candidate.replay_stored_values) {
            auto may_cross_suspend = false;
            for (auto *store : candidate.stores) {
                for (auto &projection : candidate.loads) {
                    if (graph.may_cross_suspend_between(
                            store->parent_block(),
                            projection.load->parent_block())) {
                        may_cross_suspend = true;
                        break;
                    }
                }
                if (may_cross_suspend) { break; }
            }
            if (!may_cross_suspend) {
                ++info.rejected_nonreplayable_scope_local_count;
                continue;
            }
        }

        auto unresolved = size_t{0u};
        if (candidate.stores.size() == 1u) {
            unresolved = detail::resolve_single_store_loads(
                graph, instruction_indices, candidate);
        } else {
            ++info.reaching_dataflow_alloca_count;
            unresolved = detail::resolve_reaching_values(
                graph, instruction_indices, reaching_workspace,
                candidate, options.verify_dense_reaching_values,
                info.reaching_dataflow_active_block_count,
                info.reaching_dataflow_block_evaluation_count);
        }
        info.unresolved_load_count += unresolved;
        if (unresolved != 0u) { continue; }

        // A non-replayable value may still replace a direct local load. It is
        // computed exactly once at the original store and subsequently becomes
        // ordinary SSA frame state. This is live-range splitting, not
        // rematerialization. Do not forward projected aggregate loads here:
        // inserting EXTRACT at the load would carry the complete aggregate
        // across the suspension, whereas the original access-path analysis can
        // carry only the observed leaf.
        if (!candidate.replay_stored_values) {
            auto crosses_suspend = std::any_of(
                candidate.loads.begin(), candidate.loads.end(),
                [](const auto &projection) noexcept {
                    return projection.reaches_across_suspend;
                });
            if (!crosses_suspend) {
                ++info.rejected_nonreplayable_scope_local_count;
                continue;
            }
            accepted.emplace_back(std::move(candidate));
            continue;
        }

        // A projected scalar can have a smaller replay budget than its stored
        // aggregate. Prove every exact EXTRACT expression affordable before
        // mutating anything; otherwise promotion could merely replace one
        // local frame value with a newly live SSA expression.
        auto projected_expressions_replayable = true;
        for (auto &projection : candidate.loads) {
            LUISA_DEBUG_ASSERT(
                projection.reaching_value != nullptr,
                "Resolved load must have one reaching value.");
            if (projection.indices.empty()) { continue; }
            auto budget = detail::CoroReplayableValueAnalysis::
                instruction_budget(projection.load->type());
            auto cost = size_t{1u};// the inserted EXTRACT
            auto add_operand_cost = [&](Value *operand) noexcept {
                if (!replayable.detect(operand)) { return false; }
                auto operand_cost = replayable.instruction_cost(operand);
                if (operand_cost > budget || cost > budget - operand_cost) {
                    return false;
                }
                cost += operand_cost;
                return true;
            };
            if (!add_operand_cost(projection.reaching_value)) {
                projected_expressions_replayable = false;
                break;
            }
            for (auto *index : projection.indices) {
                if (!add_operand_cost(index)) {
                    projected_expressions_replayable = false;
                    break;
                }
            }
            if (!projected_expressions_replayable) { break; }
        }
        if (!projected_expressions_replayable) {
            ++info.rejected_projected_replay_cost_count;
            continue;
        }

        luisa::unordered_set<Value *> charged_values;
        for (auto &projection : candidate.loads) {
            if (charged_values.emplace(
                                  projection.reaching_value)
                    .second) {
                candidate.replay_cost += replayable.instruction_cost(
                    projection.reaching_value);
            }
        }
        accepted.emplace_back(std::move(candidate));
    }

    // Reaching values can themselves be loads that another accepted local is
    // about to eliminate. A naive candidate-by-candidate rewrite can then
    // retain a pointer to the already detached load. Treat simultaneous load
    // substitution as a dependency graph: for a direct equation L = D, where
    // D is another eliminated load, rewrite L before D. Subsequent RAUW of D
    // then updates the uses introduced for L. A projected replacement is not
    // an edge here: all EXTRACT nodes are created before any load is detached,
    // so their operands participate in ordinary use-list rewriting.
    luisa::vector<detail::LoadProjection *> projections;
    luisa::unordered_map<LoadInst *, size_t> projection_indices;
    for (auto &candidate : accepted) {
        for (auto &projection : candidate.loads) {
            projection_indices.emplace(
                projection.load, projections.size());
            projections.emplace_back(&projection);
        }
    }
    constexpr auto no_successor = static_cast<size_t>(-1);
    luisa::vector<size_t> direct_successors(
        projections.size(), no_successor);
    luisa::vector<size_t> indegrees(projections.size(), 0u);
    for (size_t i = 0u; i < projections.size(); ++i) {
        auto &projection = *projections[i];
        if (!projection.indices.empty() ||
            projection.reaching_value == nullptr ||
            !projection.reaching_value->isa<LoadInst>()) {
            continue;
        }
        auto *dependency = static_cast<LoadInst *>(
            projection.reaching_value);
        if (auto iter = projection_indices.find(dependency);
            iter != projection_indices.end()) {
            direct_successors[i] = iter->second;
            ++indegrees[iter->second];
        }
    }
    luisa::vector<size_t> rewrite_order;
    rewrite_order.reserve(projections.size());
    for (size_t i = 0u; i < projections.size(); ++i) {
        if (indegrees[i] == 0u) {
            rewrite_order.emplace_back(i);
        }
    }
    for (size_t cursor = 0u; cursor < rewrite_order.size(); ++cursor) {
        auto successor = direct_successors[rewrite_order[cursor]];
        if (successor != no_successor &&
            --indegrees[successor] == 0u) {
            rewrite_order.emplace_back(successor);
        }
    }
    if (rewrite_order.size() != projections.size()) {
        // A cycle cannot represent a valid exact SSA substitution. Preserve
        // the original memory program atomically instead of partially
        // rewriting a malformed or otherwise unsupported value graph.
        info.rejected_forwarding_cycle_count =
            projections.size() - rewrite_order.size();
        return info;
    }

    luisa::vector<Value *> replacements(
        projections.size(), nullptr);
    luisa::vector<Value *> extract_arguments;
    for (size_t i = 0u; i < projections.size(); ++i) {
        auto &projection = *projections[i];
        auto *load = projection.load;
        auto *replacement = projection.reaching_value;
        if (!projection.indices.empty()) {
            extract_arguments.clear();
            extract_arguments.reserve(
                1u + projection.indices.size());
            extract_arguments.emplace_back(
                projection.reaching_value);
            extract_arguments.insert(
                extract_arguments.end(),
                projection.indices.begin(), projection.indices.end());
            XIRBuilder builder;
            builder.set_insertion_point(load);
            replacement = builder.call(
                load->type(), ArithmeticOp::EXTRACT,
                extract_arguments);
            ++info.inserted_extract_count;
        }
        replacements[i] = replacement;
    }

    luisa::vector<ManagedPtr<Instruction>> removed_loads;
    removed_loads.reserve(projections.size());
    for (auto i : rewrite_order) {
        auto *load = projections[i]->load;
        load->replace_all_uses_with(replacements[i]);
        removed_loads.emplace_back(load->remove_self());
        ++info.replaced_load_count;
    }
    for (auto &candidate : accepted) {
        ++info.promoted_alloca_count;
        if (!candidate.replay_stored_values) {
            ++info.promoted_nonreplayable_alloca_count;
        }
        if (candidate.stores.size() > 1u) {
            ++info.promoted_multi_store_alloca_count;
        }
        info.initializer_replay_instruction_cost += candidate.replay_cost;
        info.promoted_state_bytes += candidate.alloca->type()->size();
    }
    return info;
}

}// namespace luisa::compute::xir
