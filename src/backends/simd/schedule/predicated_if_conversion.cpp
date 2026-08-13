#include "predicated_if_conversion.h"

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>

#include "warp_uniformity.h"

namespace luisa::compute::simd::schedule {

namespace {

[[nodiscard]] bool is_varying_candidate(
    const xir::ConditionalBranchInst *branch,
    const void *context) noexcept {
    auto *uniformity = static_cast<
        const WarpUniformityAnalysis *>(context);
    return branch != nullptr && uniformity != nullptr &&
           uniformity->classify(branch->condition()) ==
               ValueClass::varying;
}

[[nodiscard]] xir::BasicBlock *single_predecessor(
    xir::BasicBlock *block) noexcept {
    auto *result = static_cast<xir::BasicBlock *>(nullptr);
    auto count = size_t{0u};
    block->traverse_predecessors(
        false, [&](xir::BasicBlock *predecessor) noexcept {
            result = predecessor;
            count++;
        });
    return count == 1u ? result : nullptr;
}

[[nodiscard]] luisa::unordered_set<const xir::ArithmeticInst *>
collect_selects(xir::Function *function) noexcept {
    luisa::unordered_set<const xir::ArithmeticInst *> result;
    if (function == nullptr || !function->is_definition()) {
        return result;
    }
    function->definition()->traverse_instructions(
        [&](xir::Instruction *instruction) noexcept {
            if (instruction->isa<xir::ArithmeticInst>()) {
                auto *arithmetic = static_cast<
                    xir::ArithmeticInst *>(instruction);
                if (arithmetic->op() == xir::ArithmeticOp::SELECT) {
                    result.emplace(arithmetic);
                }
            }
        });
    return result;
}

// if_conversion deliberately retains the merge Phi so it can preserve the
// logical variable's metadata. This can leave the next enclosing diamond one
// transparent block away from canonical form:
//
//   P: select ...; br F       F: named_phi [select, P]; br M
//
// Collapse only that generated forwarding shape. Name metadata moves to the
// unique select value that now represents the logical variable; every other
// metadata kind, non-unique value, or annotated CFG node fails closed.
[[nodiscard]] bool collapse_select_phi_forwarder(
    xir::Function *function,
    const luisa::unordered_set<const xir::ArithmeticInst *> &
        generated_selects,
    PredicatedIfConversionInfo &info) noexcept {
    if (function == nullptr || !function->is_definition()) {
        return false;
    }
    auto *definition = function->definition();
    auto *candidate = static_cast<xir::BasicBlock *>(nullptr);
    auto *predecessor = static_cast<xir::BasicBlock *>(nullptr);
    auto *target = static_cast<xir::BasicBlock *>(nullptr);
    luisa::vector<xir::PhiInst *> phis;
    definition->traverse_basic_blocks(
        [&](xir::BasicBlock *block) noexcept {
            if (candidate != nullptr || block == nullptr ||
                !block->metadata_list().empty() ||
                !block->is_terminated()) {
                return;
            }
            auto *terminator = block->terminator();
            if (!terminator->isa<xir::BranchInst>() ||
                !terminator->metadata_list().empty()) {
                return;
            }
            auto *pred = single_predecessor(block);
            if (pred == nullptr || !pred->is_terminated() ||
                !pred->terminator()->isa<xir::BranchInst>() ||
                static_cast<xir::BranchInst *>(pred->terminator())
                        ->target_block() != block) {
                return;
            }
            auto *next = static_cast<xir::BranchInst *>(terminator)
                             ->target_block();
            if (next == nullptr || next == block || next == pred) {
                return;
            }
            auto next_predecessor_count = size_t{0u};
            auto next_candidate_edge_count = size_t{0u};
            next->traverse_predecessors(
                false, [&](xir::BasicBlock *next_predecessor) noexcept {
                    next_predecessor_count++;
                    next_candidate_edge_count +=
                        next_predecessor == block;
                });
            // A straight-line forwarding block does not expose an enclosing
            // diamond and is outside this measured refinement. Require a
            // sibling edge at the target reconvergence.
            if (next_predecessor_count < 2u ||
                next_candidate_edge_count != 1u) {
                return;
            }
            luisa::vector<xir::PhiInst *> local_phis;
            auto has_generated_select = false;
            for (auto *instruction : block->instructions()) {
                if (instruction == terminator) { continue; }
                if (!instruction->isa<xir::PhiInst>()) { return; }
                auto *phi = static_cast<xir::PhiInst *>(instruction);
                if (phi->incoming_count() != 1u) { return; }
                auto incoming = phi->incoming(0u);
                if (incoming.block != pred || incoming.value == nullptr ||
                    incoming.value == phi ||
                    incoming.value->type() != phi->type()) {
                    return;
                }
                for (auto *metadata : phi->metadata_list()) {
                    if (metadata->derived_metadata_tag() !=
                        xir::DerivedMetadataTag::NAME) {
                        return;
                    }
                }
                if (!phi->metadata_list().empty()) {
                    if (!incoming.value->isa<xir::Instruction>() ||
                        incoming.value->use_list().count_size() != 1u) {
                        return;
                    }
                    for (auto *metadata :
                         incoming.value->metadata_list()) {
                        if (metadata->derived_metadata_tag() !=
                            xir::DerivedMetadataTag::NAME) {
                            return;
                        }
                    }
                    auto phi_name = phi->name();
                    auto value_name = incoming.value->name();
                    if (value_name.has_value() &&
                        (!phi_name.has_value() ||
                         *value_name != *phi_name)) {
                        return;
                    }
                }
                if (incoming.value->isa<xir::ArithmeticInst>() &&
                    generated_selects.contains(
                        static_cast<xir::ArithmeticInst *>(
                            incoming.value)) &&
                    static_cast<xir::Instruction *>(incoming.value)
                            ->parent_block() == pred) {
                    has_generated_select = true;
                }
                local_phis.emplace_back(phi);
            }
            if (local_phis.empty() || !has_generated_select) { return; }
            for (auto *instruction : next->instructions()) {
                if (!instruction->isa<xir::PhiInst>()) { continue; }
                auto *phi = static_cast<xir::PhiInst *>(instruction);
                auto from_candidate = size_t{0u};
                auto from_predecessor = size_t{0u};
                for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
                    auto incoming = phi->incoming(i);
                    from_candidate += incoming.block == block;
                    from_predecessor += incoming.block == pred;
                }
                if (from_candidate != 1u || from_predecessor != 0u) {
                    return;
                }
            }
            candidate = block;
            predecessor = pred;
            target = next;
            phis = std::move(local_phis);
        });
    if (candidate == nullptr) { return false; }

    luisa::vector<ManagedPtr<xir::Instruction>> removed_phis;
    removed_phis.reserve(phis.size());
    for (auto *phi : phis) {
        auto *value = phi->incoming(0u).value;
        if (value->metadata_list().empty()) {
            for (auto *metadata : phi->metadata_list()) {
                value->metadata_list().push_front(metadata->clone());
            }
        }
        phi->replace_all_uses_with(value);
        removed_phis.emplace_back(phi->remove_self());
        info.forwarded_phi_count++;
    }
    for (auto *instruction : target->instructions()) {
        if (!instruction->isa<xir::PhiInst>()) { continue; }
        auto *phi = static_cast<xir::PhiInst *>(instruction);
        for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
            auto incoming = phi->incoming(i);
            if (incoming.block == candidate) {
                phi->set_incoming(
                    i, incoming.value, predecessor);
            }
        }
    }
    static_cast<xir::BranchInst *>(predecessor->terminator())
        ->set_target_block(target);
    auto removed_branch = candidate->terminator()->remove_self();
    auto removed_block = candidate->remove_self();
    info.removed_forwarding_block_count++;
    return true;
}

}// namespace

PredicatedIfConversionInfo predicate_small_varying_diamonds(
    xir::Function *function, bool enable_refinement) noexcept {
    PredicatedIfConversionInfo result;
    static constexpr auto max_refinement_rounds = size_t{8u};
    for (auto round = size_t{0u};
         round < max_refinement_rounds; round++) {
        WarpUniformityAnalysis uniformity;
        uniformity.analyze(function);
        auto previous_selects = collect_selects(function);
        auto converted = xir::if_conversion_pass_run_on_function(
            function,
            {.max_arm_instruction_count = 4u,
             .max_total_instruction_count = 6u,
             .max_live_out_register_units = 4u,
             .max_speculation_cost = 12u,
             .candidate_filter = is_varying_candidate,
             .candidate_filter_context = &uniformity});
        result.if_conversion.converted_diamond_count +=
            converted.converted_diamond_count;
        result.if_conversion.hoisted_inst_count +=
            converted.hoisted_inst_count;
        result.if_conversion.replaced_phi_count +=
            converted.replaced_phi_count;
        result.if_conversion.structured_cfg_error_count +=
            converted.structured_cfg_error_count;
        if (!converted.changed()) { break; }
        if (!enable_refinement) { break; }
        auto generated_selects = collect_selects(function);
        for (auto *select : previous_selects) {
            generated_selects.erase(select);
        }
        auto collapsed = false;
        for (auto forwarding = size_t{0u};
             forwarding < max_refinement_rounds &&
             collapse_select_phi_forwarder(
                 function, generated_selects, result);
             forwarding++) {
            collapsed = true;
        }
        if (!collapsed) { break; }
        result.refinement_round_count++;
    }
    result.select_factoring = result.if_conversion.changed() ?
                                  xir::select_factor_pass_run_on_function(
                                      function) :
                                  xir::SelectFactorInfo{};
    return result;
}

}// namespace luisa::compute::simd::schedule
