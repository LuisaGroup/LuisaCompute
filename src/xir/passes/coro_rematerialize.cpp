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
};

struct LocalStateCandidate {
    AllocaInst *alloca{nullptr};
    luisa::vector<StoreInst *> stores;
    luisa::vector<LoadProjection> loads;
    size_t replay_cost{0u};
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

    [[nodiscard]] static ReachingValue undefined() noexcept {
        return ReachingValue{ReachingValueTag::UNDEFINED, nullptr};
    }

    [[nodiscard]] static ReachingValue unique(Value *value) noexcept {
        return ReachingValue{ReachingValueTag::UNIQUE, value};
    }

    [[nodiscard]] static ReachingValue conflict() noexcept {
        return ReachingValue{ReachingValueTag::CONFLICT, nullptr};
    }

    [[nodiscard]] friend bool operator==(
        ReachingValue lhs, ReachingValue rhs) noexcept {
        return lhs.tag == rhs.tag && lhs.value == rhs.value;
    }
};

[[nodiscard]] ReachingValue meet_reaching_values(
    ReachingValue lhs, ReachingValue rhs) noexcept {
    if (lhs.tag == ReachingValueTag::PENDING) { return rhs; }
    if (rhs.tag == ReachingValueTag::PENDING) { return lhs; }
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

[[nodiscard]] ReachingValue transfer_reaching_value(
    BasicBlock *block, AllocaInst *alloca,
    ReachingValue incoming) noexcept {
    auto state = incoming;
    for (auto *instruction : block->instructions()) {
        if (instruction->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(instruction);
            if (store->variable() == alloca) {
                state = ReachingValue::unique(store->value());
            }
        }
    }
    return state;
}

[[nodiscard]] size_t resolve_reaching_values(
    const CoroSemanticGraph &graph,
    LocalStateCandidate &candidate,
    size_t &block_evaluation_count) noexcept {
    auto block_count = graph.block_count();
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
            incoming = meet_reaching_values(
                incoming, block_outputs[predecessor]);
        }
        block_inputs[block_id] = incoming;
        auto outgoing = transfer_reaching_value(
            graph.block(block_id), candidate.alloca, incoming);
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
        for (auto *instruction : graph.block(block_id)->instructions()) {
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
        if (!all_store_values_replayable) { continue; }
        if (candidate.stores.size() == 1u) {
            ++info.replayable_single_store_count;
        } else {
            ++info.replayable_multi_store_count;
        }

        auto unresolved = size_t{0u};
        if (candidate.stores.size() == 1u) {
            unresolved = detail::resolve_single_store_loads(
                graph, instruction_indices, candidate);
        } else {
            ++info.reaching_dataflow_alloca_count;
            unresolved = detail::resolve_reaching_values(
                graph, candidate,
                info.reaching_dataflow_block_evaluation_count);
        }
        info.unresolved_load_count += unresolved;
        if (unresolved != 0u) { continue; }

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

    luisa::vector<ManagedPtr<Instruction>> removed_loads;
    luisa::vector<Value *> extract_arguments;
    for (auto &candidate : accepted) {
        for (auto &projection : candidate.loads) {
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
            load->replace_all_uses_with(replacement);
            removed_loads.emplace_back(load->remove_self());
            ++info.replaced_load_count;
        }
        ++info.promoted_alloca_count;
        if (candidate.stores.size() > 1u) {
            ++info.promoted_multi_store_alloca_count;
        }
        info.initializer_replay_instruction_cost += candidate.replay_cost;
        info.promoted_state_bytes += candidate.alloca->type()->size();
    }
    return info;
}

}// namespace luisa::compute::xir
