#include <algorithm>

#include <luisa/ast/type.h>
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
};

struct LocalStateCandidate {
    AllocaInst *alloca{nullptr};
    StoreInst *store{nullptr};
    luisa::vector<LoadProjection> loads;
    size_t replay_cost{0u};
};

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
                    candidate.store != nullptr ||
                    !store->metadata_list().empty()) {
                    return false;
                }
                candidate.store = store;
                continue;
            }
            return false;
        }
    }
    return candidate.store != nullptr && !candidate.loads.empty() &&
           candidate.store->value() != nullptr &&
           candidate.store->value()->type() == alloca->type() &&
           !candidate.store->value()->is_lvalue();
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
        auto index = size_t{0u};
        for (auto *instruction : block->instructions()) {
            instruction_indices.emplace(instruction, index++);
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
        if (!replayable.detect(candidate.store->value())) { continue; }
        candidate.replay_cost =
            replayable.instruction_cost(candidate.store->value());
        ++info.replayable_single_store_count;

        // A projected scalar can have a smaller replay budget than its stored
        // aggregate. Prove the exact EXTRACT expression affordable before
        // mutating anything; otherwise promotion could merely replace one
        // local frame value with a newly live SSA expression.
        auto projected_expressions_replayable = true;
        for (auto &projection : candidate.loads) {
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
            if (!add_operand_cost(candidate.store->value())) {
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

        auto *store_block = candidate.store->parent_block();
        auto dominates_every_load = store_block != nullptr &&
                                    graph.contains(store_block);
        for (auto &projection : candidate.loads) {
            auto *load = projection.load;
            auto *load_block = load->parent_block();
            if (!dominates_every_load || load_block == nullptr ||
                !graph.contains(load_block)) {
                dominates_every_load = false;
                break;
            }
            if (store_block == load_block) {
                if (instruction_indices.at(candidate.store) >=
                    instruction_indices.at(load)) {
                    dominates_every_load = false;
                    break;
                }
            } else if (!graph.dominates(store_block, load_block)) {
                dominates_every_load = false;
                break;
            }
        }
        if (dominates_every_load) {
            accepted.emplace_back(std::move(candidate));
        }
    }

    luisa::vector<ManagedPtr<Instruction>> removed_loads;
    luisa::vector<Value *> extract_arguments;
    for (auto &candidate : accepted) {
        auto *stored_value = candidate.store->value();
        for (auto &projection : candidate.loads) {
            auto *load = projection.load;
            auto *replacement = stored_value;
            if (!projection.indices.empty()) {
                extract_arguments.clear();
                extract_arguments.reserve(
                    1u + projection.indices.size());
                extract_arguments.emplace_back(stored_value);
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
        info.initializer_replay_instruction_cost += candidate.replay_cost;
        info.promoted_state_bytes += candidate.alloca->type()->size();
    }
    return info;
}

}// namespace luisa::compute::xir
