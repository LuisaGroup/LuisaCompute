#include <cstdint>
#include <limits>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_irreducible_cfg.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/verifier.h>

#include "irreducible_cfg_analysis.h"

namespace luisa::compute::xir {

namespace {

struct IrreducibleEntryPlan {
    size_t block_index{0u};
    luisa::vector<size_t> predecessor_indices;
    bool is_function_body{false};
};

struct IrreducibleRegionPlan {
    luisa::vector<IrreducibleEntryPlan> entries;
};

[[nodiscard]] bool raw_terminator_targets(
    Instruction *terminator,
    BasicBlock *target) noexcept {
    if (terminator == nullptr || target == nullptr) {
        return false;
    }
    switch (terminator->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH:
            return static_cast<BranchInst *>(terminator)
                       ->target_block() == target;
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto *branch = static_cast<ConditionalBranchInst *>(
                terminator);
            return branch->true_block() == target ||
                   branch->false_block() == target;
        }
        case DerivedInstructionTag::INDEXED_BRANCH: {
            auto *branch = static_cast<IndexedBranchInst *>(
                terminator);
            if (branch->default_block() == target) {
                return true;
            }
            for (auto index = size_t{0u};
                 index < branch->case_count(); ++index) {
                if (branch->case_block(index) == target) {
                    return true;
                }
            }
            return false;
        }
        default: return false;
    }
}

[[nodiscard]] bool retarget_raw_terminator(
    Instruction *terminator, BasicBlock *from,
    BasicBlock *to) noexcept {
    if (terminator == nullptr || from == nullptr ||
        to == nullptr) {
        return false;
    }
    auto changed = false;
    switch (terminator->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH: {
            auto *branch = static_cast<BranchInst *>(terminator);
            if (branch->target_block() == from) {
                branch->set_target_block(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto *branch = static_cast<ConditionalBranchInst *>(
                terminator);
            if (branch->true_block() == from) {
                branch->set_true_target(to);
                changed = true;
            }
            if (branch->false_block() == from) {
                branch->set_false_target(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::INDEXED_BRANCH: {
            auto *branch = static_cast<IndexedBranchInst *>(
                terminator);
            if (branch->default_block() == from) {
                branch->set_default_block(to);
                changed = true;
            }
            for (auto index = size_t{0u};
                 index < branch->case_count(); ++index) {
                if (branch->case_block(index) == from) {
                    branch->set_case_block(index, to);
                    changed = true;
                }
            }
            break;
        }
        default: break;
    }
    return changed;
}

[[nodiscard]] IrreducibleRegionPlan
build_lowering_plan(
    FunctionDefinition *definition,
    const detail::CFGStronglyConnectedComponents &analysis,
    const detail::CFGIrreducibleRegion &region,
    size_t &error_count) noexcept {
    IrreducibleRegionPlan plan;
    plan.entries.reserve(region.entry_nodes.size());
    for (auto entry_index : region.entry_nodes) {
        auto &entry = plan.entries.emplace_back();
        entry.block_index = entry_index;
        entry.is_function_body =
            analysis.blocks[entry_index] ==
            definition->body_block();
        luisa::unordered_set<size_t> seen_predecessors;
        for (auto predecessor_index :
             analysis.predecessors[entry_index]) {
            if (!seen_predecessors.emplace(
                                      predecessor_index)
                     .second) {
                continue;
            }
            auto *predecessor =
                analysis.blocks[predecessor_index];
            if (!predecessor->is_terminated() ||
                !raw_terminator_targets(
                    predecessor->terminator(),
                    analysis.blocks[entry_index])) {
                ++error_count;
                continue;
            }
            entry.predecessor_indices.emplace_back(
                predecessor_index);
        }
    }
    return plan;
}

struct LoweringPreflight {
    bool needs_lowering{false};
    size_t irreducible_region_count{0u};
    size_t error_count{0u};
    size_t boundary_verifier_count{0u};
};

[[nodiscard]] LoweringPreflight preflight_lowering(
    Function *function,
    bool verify_boundary) noexcept {
    LoweringPreflight preflight;
    if (function == nullptr ||
        function->definition() == nullptr ||
        function->definition()->body_block() == nullptr) {
        return preflight;
    }
    if (verify_boundary) {
        ++preflight.boundary_verifier_count;
        auto verification = xir_verify_function(
            function, {.require_no_phi = true});
        if (!verification.succeeded()) {
            ++preflight.error_count;
            return preflight;
        }
    }

    auto *definition = function->definition();
    auto analysis =
        detail::analyze_cfg_strongly_connected_components(
            definition);
    preflight.irreducible_region_count =
        analysis.irreducible_region_count();
    if (analysis.irreducible_regions.empty()) {
        return preflight;
    }
    preflight.needs_lowering = true;

    // The enclosing transaction mode elides only the generic verifier. Keep
    // every transform-specific precondition local and explicit: selector
    // dispatch cannot preserve SSA Phi edge semantics, so Phi input is
    // rejected before the first mutation in both verification scopes.
    for (auto *block : analysis.blocks) {
        for (auto *instruction : block->instructions()) {
            if (instruction->isa<PhiInst>()) {
                ++preflight.error_count;
            }
        }
    }

    // The pass may expose nested irreducible regions after lowering an outer
    // one. Require the entire reachable successor relation to be raw before
    // the first mutation, not merely the entry edges visible in the first
    // region. This makes every later iteration preflight-complete.
    for (auto *block : analysis.blocks) {
        auto has_successor = false;
        if (block->is_terminated()) {
            block->traverse_successors(
                false, [&](BasicBlock *) noexcept {
                    has_successor = true;
                });
        }
        if (!has_successor) { continue; }
        auto tag = block->terminator()->derived_instruction_tag();
        if (tag != DerivedInstructionTag::BRANCH &&
            tag != DerivedInstructionTag::CONDITIONAL_BRANCH &&
            tag != DerivedInstructionTag::INDEXED_BRANCH) {
            ++preflight.error_count;
        }
    }
    // A uint selector represents entry ordinals. Reject an unrepresentable
    // function before mutation; every nested region is a subset of this
    // original reachable block domain.
    if (analysis.blocks.size() >
        static_cast<size_t>(
            std::numeric_limits<uint32_t>::max())) {
        ++preflight.error_count;
    }
    auto plan = build_lowering_plan(
        definition, analysis,
        analysis.irreducible_regions.front(),
        preflight.error_count);
    if (plan.entries.size() <= 1u) {
        ++preflight.error_count;
    }
    return preflight;
}

void apply_lowering_plan(
    FunctionDefinition *definition,
    const detail::CFGStronglyConnectedComponents &analysis,
    const IrreducibleRegionPlan &plan,
    LowerIrreducibleCFGInfo &info) noexcept {
    LUISA_ASSERT(
        plan.entries.size() > 1u,
        "Irreducible lowering plan lost its entry set.");
    auto *module = definition->parent_module();
    auto *original_body = definition->body_block();
    auto body_is_entry = false;
    for (auto &&entry : plan.entries) {
        body_is_entry |= entry.is_function_body;
    }

    BasicBlock *selector_definition_block = original_body;
    if (body_is_entry) {
        selector_definition_block =
            definition->create_basic_block();
        definition->set_body_block(
            selector_definition_block);
    }

    XIRBuilder builder;
    builder.set_insertion_point(
        selector_definition_block->instructions()
            .head_sentinel());
    auto *selector = builder.alloca_local(
        Type::of<uint32_t>());
    auto *dispatcher = definition->create_basic_block();

    luisa::vector<Constant *> selector_constants;
    selector_constants.reserve(plan.entries.size());
    for (auto selector_value = size_t{0u};
         selector_value < plan.entries.size();
         ++selector_value) {
        auto value = static_cast<uint32_t>(selector_value);
        selector_constants.emplace_back(
            module->create_constant(
                Type::of<uint32_t>(), &value));
    }

    for (auto selector_value = size_t{0u};
         selector_value < plan.entries.size();
         ++selector_value) {
        auto &&entry = plan.entries[selector_value];
        auto *entry_block =
            analysis.blocks[entry.block_index];
        for (auto predecessor_index :
             entry.predecessor_indices) {
            auto *predecessor =
                analysis.blocks[predecessor_index];
            auto *edge_block =
                definition->create_basic_block();
            builder.set_insertion_point(edge_block);
            builder.store(
                selector,
                selector_constants[selector_value]);
            builder.br(dispatcher);
            LUISA_ASSERT(
                retarget_raw_terminator(
                    predecessor->terminator(),
                    entry_block, edge_block),
                "Irreducible lowering preflight and mutation "
                "disagreed on an entry edge.");
            ++info.created_edge_block_count;
        }
        if (entry.is_function_body) {
            builder.set_insertion_point(
                selector_definition_block);
            builder.store(
                selector,
                selector_constants[selector_value]);
            builder.br(dispatcher);
        }
    }

    builder.set_insertion_point(dispatcher);
    auto *selector_value = builder.load(
        Type::of<uint32_t>(), selector);
    auto *dispatch = builder.indexed_branch(
        selector_value);
    dispatch->set_default_block(
        analysis.blocks[plan.entries.front().block_index]);
    for (auto index = size_t{1u};
         index < plan.entries.size(); ++index) {
        dispatch->add_case(
            index,
            analysis.blocks[plan.entries[index].block_index]);
    }
    ++info.lowered_region_count;
    ++info.created_dispatch_block_count;
}

[[nodiscard]] LowerIrreducibleCFGInfo run_lowering(
    Function *function,
    bool verify_boundary) noexcept {
    LowerIrreducibleCFGInfo info;
    auto *definition = function->definition();
    auto initial_analysis =
        detail::analyze_cfg_strongly_connected_components(
            definition);
    auto iteration_limit = initial_analysis.blocks.size();
    while (true) {
        auto analysis =
            detail::analyze_cfg_strongly_connected_components(
                definition);
        if (analysis.irreducible_regions.empty()) { break; }
        if (info.lowered_region_count >= iteration_limit) {
            ++info.error_count;
            info.remaining_irreducible_region_count =
                analysis.irreducible_region_count();
            return info;
        }
        auto plan = build_lowering_plan(
            definition, analysis,
            analysis.irreducible_regions.front(),
            info.error_count);
        if (info.error_count != 0u ||
            plan.entries.size() <= 1u) {
            if (plan.entries.size() <= 1u) {
                ++info.error_count;
            }
            info.remaining_irreducible_region_count =
                analysis.irreducible_region_count();
            return info;
        }
        apply_lowering_plan(
            definition, analysis, plan, info);
    }

    auto lowered_analysis =
        detail::analyze_cfg_strongly_connected_components(
            definition);
    info.remaining_irreducible_region_count =
        lowered_analysis.irreducible_region_count();
    if (verify_boundary) {
        ++info.boundary_verifier_count;
        auto output_verification = xir_verify_function(
            function, {.require_no_phi = true});
        if (!output_verification.succeeded()) {
            ++info.error_count;
        }
    }
    return info;
}

void accumulate_info(
    LowerIrreducibleCFGInfo &total,
    const LowerIrreducibleCFGInfo &info) noexcept {
    total.lowered_region_count +=
        info.lowered_region_count;
    total.created_dispatch_block_count +=
        info.created_dispatch_block_count;
    total.created_edge_block_count +=
        info.created_edge_block_count;
    total.remaining_irreducible_region_count +=
        info.remaining_irreducible_region_count;
    total.error_count += info.error_count;
    total.boundary_verifier_count +=
        info.boundary_verifier_count;
}

}// namespace

LowerIrreducibleCFGInfo
lower_irreducible_cfg_pass_run_on_function(
    Function *function,
    const LowerIrreducibleCFGOptions &options) noexcept {
    LowerIrreducibleCFGInfo info;
    if (function == nullptr ||
        function->definition() == nullptr ||
        function->definition()->body_block() == nullptr) {
        return info;
    }
    auto verify_boundaries =
        xir_pass_has_standalone_verification(
            options.verification_transaction,
            function);
    auto preflight = preflight_lowering(
        function, verify_boundaries);
    info.error_count = preflight.error_count;
    info.remaining_irreducible_region_count =
        preflight.irreducible_region_count;
    info.boundary_verifier_count =
        preflight.boundary_verifier_count;
    if (info.error_count != 0u ||
        !preflight.needs_lowering) {
        return info;
    }
    auto lowered = run_lowering(
        function, verify_boundaries);
    lowered.boundary_verifier_count +=
        preflight.boundary_verifier_count;
    return lowered;
}

LowerIrreducibleCFGInfo
lower_irreducible_cfg_pass_run_on_module(
    Module *module, PassReport *report) noexcept {
    LowerIrreducibleCFGInfo total;
    if (module != nullptr) {
        luisa::vector<Function *> functions_to_lower;
        for (auto *function : module->function_list()) {
            auto preflight = preflight_lowering(
                function, true);
            total.error_count += preflight.error_count;
            total.remaining_irreducible_region_count +=
                preflight.irreducible_region_count;
            total.boundary_verifier_count +=
                preflight.boundary_verifier_count;
            if (preflight.needs_lowering) {
                functions_to_lower.emplace_back(function);
            }
        }
        // Module lowering is transactional with respect to all supported-input
        // checks: one rejected function prevents mutation of every function.
        if (total.error_count == 0u) {
            total.remaining_irreducible_region_count = 0u;
            for (auto *function : functions_to_lower) {
                accumulate_info(total, run_lowering(function, true));
            }
        }
    }
    if (report != nullptr) {
        report->set(
            "lowered_region",
            total.lowered_region_count);
        report->set(
            "created_dispatch_block",
            total.created_dispatch_block_count);
        report->set(
            "created_edge_block",
            total.created_edge_block_count);
        report->set(
            "remaining_irreducible_region",
            total.remaining_irreducible_region_count);
        report->set("error", total.error_count);
        report->set(
            "boundary_verifier",
            total.boundary_verifier_count);
    }
    return total;
}

}// namespace luisa::compute::xir
