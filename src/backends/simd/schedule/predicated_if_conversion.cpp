#include "predicated_if_conversion.h"

#include <luisa/ast/type.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>

#include <algorithm>

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

[[nodiscard]] size_t instruction_count(
    const xir::BasicBlock *block) noexcept {
    auto count = size_t{0u};
    if (block != nullptr) {
        for (auto *instruction : block->instructions()) {
            count += !instruction->is_terminator();
        }
    }
    return count;
}

[[nodiscard]] bool is_widened_update_candidate(
    const xir::ConditionalBranchInst *branch,
    const void *context) noexcept {
    if (!is_varying_candidate(branch, context)) { return false; }
    auto true_count = instruction_count(branch->true_block());
    auto false_count = instruction_count(branch->false_block());
    auto empty_count = std::min(true_count, false_count);
    auto update_count = std::max(true_count, false_count);
    if (empty_count != 0u ||
        update_count < 5u || update_count > 6u) {
        return false;
    }

    auto *true_block = branch->true_block();
    auto *false_block = branch->false_block();
    if (true_block == nullptr || false_block == nullptr ||
        !true_block->is_terminated() ||
        !false_block->is_terminated() ||
        !true_block->terminator()->isa<xir::BranchInst>() ||
        !false_block->terminator()->isa<xir::BranchInst>()) {
        return false;
    }
    auto *true_merge = static_cast<const xir::BranchInst *>(
                           true_block->terminator())
                           ->target_block();
    auto *false_merge = static_cast<const xir::BranchInst *>(
                            false_block->terminator())
                            ->target_block();
    if (true_merge == nullptr || true_merge != false_merge) {
        return false;
    }
    auto updated_phi_count = size_t{0u};
    for (auto *instruction : true_merge->instructions()) {
        if (!instruction->isa<xir::PhiInst>()) { continue; }
        auto *phi = static_cast<const xir::PhiInst *>(instruction);
        auto *true_value = static_cast<const xir::Value *>(nullptr);
        auto *false_value = static_cast<const xir::Value *>(nullptr);
        for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
            auto incoming = phi->incoming(i);
            if (incoming.block == true_block) {
                true_value = incoming.value;
            } else if (incoming.block == false_block) {
                false_value = incoming.value;
            }
        }
        updated_phi_count += true_value != nullptr &&
                             false_value != nullptr &&
                             true_value != false_value;
    }
    return updated_phi_count >= 2u;
}

[[nodiscard]] bool is_wide_select_ladder_candidate(
    const xir::ConditionalBranchInst *branch,
    const void *context) noexcept {
    if (!is_varying_candidate(branch, context)) { return false; }
    auto *true_block = branch->true_block();
    auto *false_block = branch->false_block();
    if (true_block == nullptr || false_block == nullptr) { return false; }
    auto true_count = instruction_count(true_block);
    auto false_count = instruction_count(false_block);
    if (std::min(true_count, false_count) != 0u ||
        std::max(true_count, false_count) != 6u) {
        return false;
    }
    auto *wide_block = true_count == 6u ? true_block : false_block;
    auto equality_count = size_t{0u};
    auto select_count = size_t{0u};
    for (auto *instruction : wide_block->instructions()) {
        if (instruction->is_terminator()) { continue; }
        if (!instruction->isa<xir::ArithmeticInst>()) { return false; }
        auto *arithmetic = static_cast<const xir::ArithmeticInst *>(
            instruction);
        if (arithmetic->op() == xir::ArithmeticOp::BINARY_EQUAL &&
            arithmetic->type() != nullptr &&
            arithmetic->type()->is_bool()) {
            equality_count++;
            continue;
        }
        auto *type = arithmetic->type();
        if (arithmetic->op() == xir::ArithmeticOp::SELECT &&
            type != nullptr && type->is_vector() &&
            type->dimension() == 3u &&
            type->element()->is_float32()) {
            select_count++;
            continue;
        }
        return false;
    }
    if (equality_count != 3u || select_count != 3u ||
        !true_block->is_terminated() ||
        !false_block->is_terminated() ||
        !true_block->terminator()->isa<xir::BranchInst>() ||
        !false_block->terminator()->isa<xir::BranchInst>()) {
        return false;
    }
    auto *true_merge = static_cast<const xir::BranchInst *>(
                           true_block->terminator())
                           ->target_block();
    auto *false_merge = static_cast<const xir::BranchInst *>(
                            false_block->terminator())
                            ->target_block();
    if (true_merge == nullptr || true_merge != false_merge) {
        return false;
    }
    auto differing_float3_phis = size_t{0u};
    for (auto *instruction : true_merge->instructions()) {
        if (!instruction->isa<xir::PhiInst>()) { continue; }
        auto *phi = static_cast<const xir::PhiInst *>(instruction);
        auto *true_value = static_cast<const xir::Value *>(nullptr);
        auto *false_value = static_cast<const xir::Value *>(nullptr);
        for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
            auto incoming = phi->incoming(i);
            if (incoming.block == true_block) {
                true_value = incoming.value;
            } else if (incoming.block == false_block) {
                false_value = incoming.value;
            }
        }
        if (true_value == nullptr || false_value == nullptr ||
            true_value == false_value) {
            continue;
        }
        auto *type = phi->type();
        if (type == nullptr || !type->is_vector() ||
            type->dimension() != 3u ||
            !type->element()->is_float32()) {
            return false;
        }
        differing_float3_phis++;
    }
    return differing_float3_phis == 1u;
}

[[nodiscard]] const xir::RayQueryObjectReadInst *
triangle_candidate_hit_root(const xir::Value *value) noexcept {
    while (value != nullptr && value->isa<xir::ArithmeticInst>()) {
        auto *arithmetic = static_cast<
            const xir::ArithmeticInst *>(value);
        if (arithmetic->op() != xir::ArithmeticOp::EXTRACT ||
            arithmetic->operand_count() < 2u ||
            arithmetic->operand(0u) == nullptr) {
            break;
        }
        auto *current_type = arithmetic->operand(0u)->type();
        for (auto i = size_t{1u};
             i < arithmetic->operand_count(); i++) {
            auto index = uint64_t{0u};
            if (current_type == nullptr ||
                !xir::try_decode_constant_nonnegative_integer(
                    arithmetic->operand(i), index)) {
                return nullptr;
            }
            switch (current_type->tag()) {
                case Type::Tag::ARRAY:
                case Type::Tag::VECTOR:
                    if (index >= current_type->dimension()) {
                        return nullptr;
                    }
                    current_type = current_type->element();
                    break;
                case Type::Tag::MATRIX:
                    if (index >= current_type->dimension()) {
                        return nullptr;
                    }
                    current_type = Type::vector(
                        current_type->element(),
                        current_type->dimension());
                    break;
                case Type::Tag::STRUCTURE:
                    if (index >= current_type->members().size()) {
                        return nullptr;
                    }
                    current_type = current_type->members()[static_cast<size_t>(index)];
                    break;
                default: return nullptr;
            }
        }
        if (current_type != arithmetic->type()) { return nullptr; }
        value = arithmetic->operand(0u);
    }
    if (value == nullptr ||
        !value->isa<xir::RayQueryObjectReadInst>()) {
        return nullptr;
    }
    auto *read = static_cast<
        const xir::RayQueryObjectReadInst *>(value);
    return read->op() ==
                   xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT ?
               read :
               nullptr;
}

[[nodiscard]] const xir::RayQueryObjectReadInst *
triangle_candidate_instance_root(const xir::Value *value) noexcept {
    if (value == nullptr || !value->isa<xir::ArithmeticInst>()) {
        return nullptr;
    }
    auto *extract = static_cast<const xir::ArithmeticInst *>(value);
    auto member = uint64_t{0u};
    if (extract->op() != xir::ArithmeticOp::EXTRACT ||
        extract->operand_count() != 2u ||
        extract->type() == nullptr ||
        !extract->type()->is_uint32() ||
        !xir::try_decode_constant_nonnegative_integer(
            extract->operand(1u), member) ||
        member != 0u) {
        return nullptr;
    }
    // SurfaceHit member zero is the public instance identifier. Requiring a
    // direct extraction prevents an equal-shaped prim/bary/t predicate from
    // accidentally selecting this workload-specific cost policy.
    auto *root = extract->operand(0u);
    if (root == nullptr ||
        !root->isa<xir::RayQueryObjectReadInst>()) {
        return nullptr;
    }
    return triangle_candidate_hit_root(root);
}

[[nodiscard]] bool is_ray_query_filter_candidate(
    const xir::ConditionalBranchInst *branch,
    const void *context) noexcept {
    if (!is_varying_candidate(branch, context)) { return false; }
    auto *condition = branch->condition();
    if (condition == nullptr ||
        !condition->isa<xir::ArithmeticInst>()) {
        return false;
    }
    auto *condition_arithmetic = static_cast<
        const xir::ArithmeticInst *>(condition);
    if (condition_arithmetic->op() !=
            xir::ArithmeticOp::BINARY_EQUAL ||
        condition_arithmetic->operand_count() != 2u) {
        return false;
    }
    auto *query_hit = triangle_candidate_instance_root(
        condition_arithmetic->operand(0u));
    auto *constant = condition_arithmetic->operand(1u);
    if (query_hit == nullptr) {
        query_hit = triangle_candidate_instance_root(
            condition_arithmetic->operand(1u));
        constant = condition_arithmetic->operand(0u);
    }
    if (query_hit == nullptr || constant == nullptr ||
        !constant->isa<xir::Constant>()) {
        return false;
    }

    auto *true_block = branch->true_block();
    auto *false_block = branch->false_block();
    if (true_block == nullptr || false_block == nullptr ||
        !true_block->is_terminated() ||
        !false_block->is_terminated() ||
        !true_block->terminator()->isa<xir::BranchInst>() ||
        !false_block->terminator()->isa<xir::BranchInst>()) {
        return false;
    }
    auto *true_merge = static_cast<const xir::BranchInst *>(
                           true_block->terminator())
                           ->target_block();
    auto *false_merge = static_cast<const xir::BranchInst *>(
                            false_block->terminator())
                            ->target_block();
    if (true_merge == nullptr || true_merge != false_merge) {
        return false;
    }
    auto true_count = instruction_count(true_block);
    auto false_count = instruction_count(false_block);
    auto smaller = std::min(true_count, false_count);
    auto larger = std::max(true_count, false_count);
    if (!((smaller == 0u && larger == 5u) ||
          (smaller == 5u && larger == 8u))) {
        return false;
    }

    auto saw_fract = false;
    auto validate_arm = [&](const xir::BasicBlock *arm) noexcept {
        for (auto *instruction : arm->instructions()) {
            if (instruction->is_terminator()) { continue; }
            if (!instruction->isa<xir::ArithmeticInst>()) {
                return false;
            }
            auto *arithmetic = static_cast<
                const xir::ArithmeticInst *>(instruction);
            switch (arithmetic->op()) {
                case xir::ArithmeticOp::EXTRACT:
                    if (triangle_candidate_hit_root(arithmetic) !=
                        query_hit) {
                        return false;
                    }
                    break;
                case xir::ArithmeticOp::BINARY_MUL:
                case xir::ArithmeticOp::BINARY_LESS:
                case xir::ArithmeticOp::BINARY_EQUAL:
                case xir::ArithmeticOp::SELECT: break;
                case xir::ArithmeticOp::FRACT:
                    saw_fract = true;
                    break;
                default: return false;
            }
        }
        return true;
    };
    if (!validate_arm(true_block) ||
        !validate_arm(false_block) || !saw_fract) {
        return false;
    }

    auto differing_bool_phis = size_t{0u};
    for (auto *instruction : true_merge->instructions()) {
        if (!instruction->isa<xir::PhiInst>()) { continue; }
        auto *phi = static_cast<const xir::PhiInst *>(instruction);
        auto *true_value = static_cast<const xir::Value *>(nullptr);
        auto *false_value = static_cast<const xir::Value *>(nullptr);
        for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
            auto incoming = phi->incoming(i);
            if (incoming.block == true_block) {
                true_value = incoming.value;
            } else if (incoming.block == false_block) {
                false_value = incoming.value;
            }
        }
        if (true_value == nullptr || false_value == nullptr ||
            true_value == false_value) {
            continue;
        }
        if (phi->type() == nullptr || !phi->type()->is_bool()) {
            return false;
        }
        differing_bool_phis++;
    }
    return differing_bool_phis == 1u;
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
    xir::Function *function, bool enable_refinement,
    size_t max_speculation_cost,
    bool enable_widened_updates,
    bool enable_wide_select_ladder,
    bool enable_ray_query_filter) noexcept {
    PredicatedIfConversionInfo result;
    auto accumulate = [&](const xir::IfConversionInfo &converted) noexcept {
        result.if_conversion.converted_diamond_count +=
            converted.converted_diamond_count;
        result.if_conversion.hoisted_inst_count +=
            converted.hoisted_inst_count;
        result.if_conversion.replaced_phi_count +=
            converted.replaced_phi_count;
        result.if_conversion.structured_cfg_error_count +=
            converted.structured_cfg_error_count;
    };
    if (enable_ray_query_filter) {
        static constexpr auto max_ray_query_filter_rounds = size_t{3u};
        for (auto round = size_t{0u};
             round < max_ray_query_filter_rounds; round++) {
            WarpUniformityAnalysis uniformity;
            uniformity.analyze(function);
            auto previous_selects = collect_selects(function);
            auto converted = xir::if_conversion_pass_run_on_function(
                function,
                {.max_arm_instruction_count = 8u,
                 .max_total_instruction_count = 13u,
                 .max_live_out_register_units = 1u,
                 .max_speculation_cost = 64u,
                 .allow_speculative_static_extract = true,
                 .candidate_filter = is_ray_query_filter_candidate,
                 .candidate_filter_context = &uniformity});
            accumulate(converted);
            result.ray_query_filter_diamond_count +=
                converted.converted_diamond_count;
            if (!converted.changed()) { break; }
            auto generated_selects = collect_selects(function);
            for (auto *select : previous_selects) {
                generated_selects.erase(select);
            }
            auto collapsed = false;
            for (auto forwarding = size_t{0u};
                 forwarding < max_ray_query_filter_rounds * 2u &&
                 collapse_select_phi_forwarder(
                     function, generated_selects, result);
                 forwarding++) {
                collapsed = true;
            }
            result.refinement_round_count += collapsed;
            if (!collapsed) { break; }
        }
    }
    if (enable_widened_updates) {
        WarpUniformityAnalysis uniformity;
        uniformity.analyze(function);
        auto converted = xir::if_conversion_pass_run_on_function(
            function,
            {.max_arm_instruction_count = 6u,
             .max_total_instruction_count = 6u,
             .max_live_out_register_units = 6u,
             .max_speculation_cost = 58u,
             .allow_speculative_float_division = true,
             .candidate_filter = is_widened_update_candidate,
             .candidate_filter_context = &uniformity});
        accumulate(converted);
        result.widened_update_diamond_count =
            converted.converted_diamond_count;
    }
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
             .max_speculation_cost = max_speculation_cost,
             .candidate_filter = is_varying_candidate,
             .candidate_filter_context = &uniformity});
        accumulate(converted);
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
    if (enable_wide_select_ladder) {
        WarpUniformityAnalysis uniformity;
        uniformity.analyze(function);
        auto previous_selects = collect_selects(function);
        auto converted = xir::if_conversion_pass_run_on_function(
            function,
            {.max_arm_instruction_count = 6u,
             .max_total_instruction_count = 6u,
             .max_live_out_register_units = 4u,
             .max_speculation_cost = 19u,
             .candidate_filter = is_wide_select_ladder_candidate,
             .candidate_filter_context = &uniformity});
        accumulate(converted);
        result.wide_select_ladder_diamond_count =
            converted.converted_diamond_count;
        if (converted.changed()) {
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
            result.refinement_round_count += collapsed;
        }
    }
    result.select_factoring = result.if_conversion.changed() ?
                                  xir::select_factor_pass_run_on_function(
                                      function) :
                                  xir::SelectFactorInfo{};
    return result;
}

}// namespace luisa::compute::simd::schedule
