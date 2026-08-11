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
#include <luisa/xir/metadata/name.h>
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

[[nodiscard]] bool has_only_debug_name_metadata(
    const MetadataListMixin &value) noexcept {
    for (auto *metadata : value.metadata_list()) {
        if (!metadata->isa<NameMD>()) { return false; }
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
        !has_only_debug_name_metadata(*alloca)) {
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
                    !has_only_debug_name_metadata(*gep)) {
                    return false;
                }
                worklist.emplace_back(gep);
                continue;
            }
            if (instruction->isa<LoadInst>()) {
                auto *load = static_cast<LoadInst *>(instruction);
                if (load->variable() != pointer ||
                    !load->metadata_list().empty()) {
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
                    !store->metadata_list().empty()) {
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

[[nodiscard]] size_t resolve_reaching_values(
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
    Function *function) noexcept {
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
                graph, instruction_indices, candidate,
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
                    projection.reaching_value).second) {
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
