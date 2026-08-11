#include <algorithm>

#include <luisa/ast/type.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/passes/defer_local_aggregate_load.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

namespace {

struct AggregateProjection {
    ArithmeticInst *extract{nullptr};
    LoadInst *load{nullptr};
    luisa::vector<Value *> indices;
    luisa::vector<uint64_t> decoded_indices;
};

struct AggregateLoadGroup {
    LoadInst *load{nullptr};
    luisa::vector<size_t> projection_indices;
};

struct MaterializedProjection {
    const Type *type{nullptr};
    luisa::vector<uint64_t> indices;
    LoadInst *load{nullptr};
};

void clone_metadata(const MetadataListMixin &source,
                    MetadataListMixin &target) noexcept {
    for (auto *metadata : source.metadata_list()) {
        target.metadata_list().push_front(metadata->clone());
    }
}

[[nodiscard]] bool has_static_local_pointer_path(
    Value *pointer) noexcept {
    while (pointer != nullptr && pointer->isa<Instruction>()) {
        auto *instruction = static_cast<Instruction *>(pointer);
        if (instruction->isa<AllocaInst>()) {
            return static_cast<AllocaInst *>(instruction)->is_local();
        }
        if (!instruction->isa<GEPInst>()) { return false; }
        auto *gep = static_cast<GEPInst *>(instruction);
        for (auto *index_use : gep->index_uses()) {
            uint64_t index = 0u;
            if (!try_decode_constant_nonnegative_integer(
                    index_use->value(), index)) {
                return false;
            }
        }
        pointer = gep->base();
    }
    return false;
}

[[nodiscard]] bool collect_projection(
    ArithmeticInst *extract,
    AggregateProjection &projection) noexcept {
    if (extract == nullptr || extract->op() != ArithmeticOp::EXTRACT) {
        return false;
    }

    luisa::vector<ArithmeticInst *> chain;
    auto *cursor = static_cast<Value *>(extract);
    while (cursor != nullptr && cursor->isa<ArithmeticInst>()) {
        auto *current = static_cast<ArithmeticInst *>(cursor);
        if (current->op() != ArithmeticOp::EXTRACT ||
            current->operand_count() < 2u) {
            break;
        }
        // An annotated intermediate value has a unique semantic/debug owner.
        // Project it one-to-one, but do not bypass it from a descendant.
        if (current != extract && !current->metadata_list().empty()) {
            return false;
        }
        chain.emplace_back(current);
        cursor = current->operand(0u);
    }
    if (chain.empty() || cursor == nullptr ||
        !cursor->isa<LoadInst>()) {
        return false;
    }
    auto *load = static_cast<LoadInst *>(cursor);
    if (load->type() == nullptr || load->type()->is_scalar() ||
        !has_static_local_pointer_path(load->variable())) {
        return false;
    }

    projection.extract = extract;
    projection.load = load;
    for (auto iter = chain.rbegin(); iter != chain.rend(); ++iter) {
        auto *current = *iter;
        for (size_t i = 1u; i < current->operand_count(); ++i) {
            auto *index_value = current->operand(i);
            uint64_t decoded_index = 0u;
            if (!try_decode_constant_nonnegative_integer(
                    index_value, decoded_index)) {
                return false;
            }
            projection.indices.emplace_back(index_value);
            projection.decoded_indices.emplace_back(decoded_index);
        }
    }
    return !projection.indices.empty();
}

[[nodiscard]] bool same_projection(
    const MaterializedProjection &materialized,
    const AggregateProjection &candidate) noexcept {
    return materialized.type == candidate.extract->type() &&
           materialized.indices == candidate.decoded_indices;
}

[[nodiscard]] bool has_nonprojection_user(
    const AggregateProjection &projection,
    const luisa::unordered_set<ArithmeticInst *> &candidates) noexcept {
    for (auto *use : projection.extract->use_list()) {
        auto *user = use == nullptr ? nullptr : use->user();
        if (user == nullptr || !user->isa<ArithmeticInst>() ||
            !candidates.contains(static_cast<ArithmeticInst *>(user))) {
            return true;
        }
    }
    return false;
}

void remove_projection_dag(
    luisa::span<AggregateProjection> projections) noexcept {
    luisa::unordered_set<ArithmeticInst *> candidates;
    candidates.reserve(projections.size());
    for (auto &projection : projections) {
        candidates.emplace(projection.extract);
    }

    luisa::vector<ArithmeticInst *> work;
    work.reserve(projections.size());
    for (auto &projection : projections) {
        if (projection.extract->use_list().empty()) {
            work.emplace_back(projection.extract);
        }
    }
    luisa::vector<ManagedPtr<Instruction>> removed;
    removed.reserve(projections.size());
    while (!work.empty()) {
        auto *extract = work.back();
        work.pop_back();
        if (!extract->is_linked() ||
            !extract->use_list().empty()) {
            continue;
        }
        auto *source = extract->operand(0u);
        auto *source_extract =
            source != nullptr && source->isa<ArithmeticInst>() ?
                static_cast<ArithmeticInst *>(source) : nullptr;
        removed.emplace_back(extract->remove_self());
        if (source_extract != nullptr &&
            candidates.contains(source_extract) &&
            source_extract->is_linked() &&
            source_extract->use_list().empty()) {
            work.emplace_back(source_extract);
        }
    }
#ifndef NDEBUG
    for (auto &projection : projections) {
        LUISA_DEBUG_ASSERT(
            !projection.extract->is_linked(),
            "Defer-local-aggregate-load left a candidate extract linked.");
    }
#endif
}

void run_on_function(Function *function,
                     DeferLocalAggregateLoadInfo &info) noexcept {
    auto *definition =
        function == nullptr ? nullptr : function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return;
    }

    luisa::vector<AggregateProjection> projections;
    luisa::vector<AggregateLoadGroup> groups;
    luisa::unordered_map<LoadInst *, size_t> load_to_group;
    // CoroSuspend deliberately has no ordinary CFG successor. Resume roots
    // are nevertheless owned executable blocks in the coroutine state
    // machine, so a reachability traversal from body_block() would silently
    // optimize only the entry scope. Iterating the function's ownership list
    // is also total for ordinary functions (where it simply includes dead
    // blocks that a later CFG cleanup may remove).
    for (auto *block : definition->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (!instruction->isa<ArithmeticInst>()) { continue; }
            auto *extract = static_cast<ArithmeticInst *>(instruction);
            if (extract->op() != ArithmeticOp::EXTRACT) { continue; }
            AggregateProjection projection;
            if (!collect_projection(extract, projection)) { continue; }
            auto projection_index = projections.size();
            auto *load = projection.load;
            projections.emplace_back(std::move(projection));
            auto [iter, inserted] = load_to_group.try_emplace(
                load, groups.size());
            if (inserted) {
                groups.emplace_back(AggregateLoadGroup{.load = load});
            }
            groups[iter->second].projection_indices.emplace_back(
                projection_index);
        }
    }
    if (projections.empty()) { return; }

    info.aggregate_load_count += groups.size();
    info.candidate_extract_count += projections.size();
    luisa::unordered_set<ArithmeticInst *> candidate_set;
    candidate_set.reserve(projections.size());
    for (auto &projection : projections) {
        candidate_set.emplace(projection.extract);
    }

    XIRBuilder builder;
    for (auto &group : groups) {
        builder.set_insertion_point(group.load);
        luisa::vector<MaterializedProjection> materialized;
        materialized.reserve(group.projection_indices.size());
        for (auto projection_index : group.projection_indices) {
            auto &projection = projections[projection_index];
            if (!has_nonprojection_user(projection, candidate_set)) {
                continue;
            }
            LoadInst *replacement = nullptr;
            if (projection.extract->metadata_list().empty()) {
                for (auto &previous : materialized) {
                    if (same_projection(previous, projection)) {
                        replacement = previous.load;
                        ++info.reused_projection_count;
                        break;
                    }
                }
            }
            if (replacement == nullptr) {
                auto *pointer = builder.gep(
                    projection.extract->type(),
                    group.load->variable(), projection.indices);
                replacement = builder.load(
                    projection.extract->type(), pointer);
                clone_metadata(*group.load, *replacement);
                clone_metadata(*projection.extract, *replacement);
                if (projection.extract->metadata_list().empty()) {
                    materialized.emplace_back(MaterializedProjection{
                        .type = projection.extract->type(),
                        .indices = projection.decoded_indices,
                        .load = replacement});
                }
                ++info.inserted_gep_count;
                ++info.inserted_load_count;
            }
            projection.extract->replace_all_uses_with(replacement);
        }
    }

    remove_projection_dag(projections);
    info.rewritten_extract_count += projections.size();
    for (auto &group : groups) {
        if (group.load->use_list().empty()) {
            static_cast<void>(group.load->remove_self());
            ++info.removed_aggregate_load_count;
        }
    }
}

}// namespace

}// namespace detail

DeferLocalAggregateLoadInfo
defer_local_aggregate_load_pass_run_on_function(
    Function *function) noexcept {
    DeferLocalAggregateLoadInfo info;
    detail::run_on_function(function, info);
    return info;
}

DeferLocalAggregateLoadInfo
defer_local_aggregate_load_pass_run_on_module(
    Module *module, PassReport *report) noexcept {
    DeferLocalAggregateLoadInfo info;
    if (module != nullptr) {
        for (auto *function : module->function_list()) {
            detail::run_on_function(function, info);
        }
    }
    if (report != nullptr) {
        report->set("aggregate_load", info.aggregate_load_count);
        report->set("candidate_extract", info.candidate_extract_count);
        report->set("rewritten_extract", info.rewritten_extract_count);
        report->set("inserted_gep", info.inserted_gep_count);
        report->set("inserted_load", info.inserted_load_count);
        report->set("reused_projection", info.reused_projection_count);
        report->set("removed_aggregate_load",
                    info.removed_aggregate_load_count);
    }
    return info;
}

}// namespace luisa::compute::xir
