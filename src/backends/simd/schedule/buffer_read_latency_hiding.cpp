#include "buffer_read_latency_hiding.h"

#include <luisa/ast/type.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/if_conversion.h>

#include <utility>

#include "../../../xir/passes/helpers.h"
#include "../../../xir/passes/natural_loop.h"
#include "warp_uniformity.h"

namespace luisa::compute::simd::schedule {

namespace {

struct BufferReadLatencyCandidate {
    xir::BasicBlock *load_block{nullptr};
    xir::BasicBlock *bridge_block{nullptr};
    xir::BasicBlock *step_block{nullptr};
    xir::BasicBlock *true_block{nullptr};
    xir::BasicBlock *false_block{nullptr};
    xir::BasicBlock *merge_block{nullptr};
    xir::ResourceReadInst *read{nullptr};
    xir::ConditionalBranchInst *load_branch{nullptr};
    xir::Instruction *step_metadata_owner{nullptr};
    size_t generated_select_count{0u};
};

[[nodiscard]] xir::BasicBlock *single_predecessor(
    xir::BasicBlock *block) noexcept {
    auto *result = static_cast<xir::BasicBlock *>(nullptr);
    auto count = size_t{0u};
    if (block != nullptr) {
        block->traverse_predecessors(
            false, [&](xir::BasicBlock *predecessor) noexcept {
                result = predecessor;
                count++;
            });
    }
    return count == 1u ? result : nullptr;
}

[[nodiscard]] size_t nonterminator_count(
    const xir::BasicBlock *block) noexcept {
    auto count = size_t{0u};
    if (block != nullptr) {
        for (auto *instruction : block->instructions()) {
            count += !instruction->is_terminator();
        }
    }
    return count;
}

[[nodiscard]] bool is_narrow_scalar_type(
    const Type *type) noexcept {
    return type != nullptr && type->is_scalar() &&
           type->size() <= sizeof(uint32_t);
}

// This is intentionally narrower than generic if-conversion's total-operation
// set. It describes the measured DDA continuation shape and avoids silently
// growing this latency-hiding policy into general speculation.
[[nodiscard]] bool is_bounded_total_instruction(
    const xir::Instruction *instruction) noexcept {
    if (instruction == nullptr || instruction->is_terminator() ||
        !is_narrow_scalar_type(instruction->type())) {
        return false;
    }
    if (instruction->isa<xir::CastInst>()) {
        auto *cast = static_cast<const xir::CastInst *>(instruction);
        auto *source = cast->value() == nullptr ?
                           nullptr :
                           cast->value()->type();
        if (!is_narrow_scalar_type(source)) { return false; }
        if (cast->op() == xir::CastOp::BITWISE_CAST) {
            return source->size() == instruction->type()->size();
        }
        if (cast->op() != xir::CastOp::STATIC_CAST) { return false; }
        return !(source->is_float() &&
                 (instruction->type()->is_int() ||
                  instruction->type()->is_uint()));
    }
    if (!instruction->isa<xir::ArithmeticInst>()) { return false; }
    switch (static_cast<const xir::ArithmeticInst *>(instruction)->op()) {
        case xir::ArithmeticOp::BINARY_ADD:
        case xir::ArithmeticOp::BINARY_SUB:
        case xir::ArithmeticOp::BINARY_BIT_AND:
        case xir::ArithmeticOp::BINARY_BIT_OR:
        case xir::ArithmeticOp::BINARY_BIT_XOR:
        case xir::ArithmeticOp::BINARY_LESS:
        case xir::ArithmeticOp::BINARY_LESS_EQUAL:
        case xir::ArithmeticOp::BINARY_GREATER:
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
        case xir::ArithmeticOp::BINARY_EQUAL:
        case xir::ArithmeticOp::BINARY_NOT_EQUAL:
        case xir::ArithmeticOp::SELECT: return true;
        default: return false;
    }
}

[[nodiscard]] bool is_direct_read_comparison(
    xir::ArithmeticInst *comparison,
    xir::ResourceReadInst *&read) noexcept {
    if (comparison == nullptr || comparison->operand_count() != 2u ||
        comparison->type() == nullptr ||
        !comparison->type()->is_bool()) {
        return false;
    }
    switch (comparison->op()) {
        case xir::ArithmeticOp::BINARY_LESS:
        case xir::ArithmeticOp::BINARY_LESS_EQUAL:
        case xir::ArithmeticOp::BINARY_GREATER:
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
        case xir::ArithmeticOp::BINARY_EQUAL:
        case xir::ArithmeticOp::BINARY_NOT_EQUAL: break;
        default: return false;
    }
    auto *constant = static_cast<xir::Value *>(nullptr);
    for (auto i = size_t{0u}; i < comparison->operand_count(); i++) {
        auto *operand = comparison->operand(i);
        if (operand != nullptr && operand->isa<xir::ResourceReadInst>()) {
            if (read != nullptr) { return false; }
            read = static_cast<xir::ResourceReadInst *>(operand);
        } else {
            constant = operand;
        }
    }
    if (read == nullptr || constant == nullptr ||
        !constant->isa<xir::Constant>() ||
        read->op() != xir::ResourceReadOp::BUFFER_READ ||
        read->operand_count() != 2u ||
        read->type() == nullptr || !read->type()->is_scalar() ||
        read->type()->size() != sizeof(uint32_t) ||
        !(read->type()->is_int32() || read->type()->is_uint32() ||
          read->type()->is_float32())) {
        return false;
    }
    auto *buffer = read->operand(0u);
    return buffer != nullptr && buffer->type() != nullptr &&
           buffer->type()->is_buffer() &&
           buffer->type()->element() == read->type();
}

[[nodiscard]] bool can_rehome_debug_metadata(
    const xir::Instruction *source,
    const xir::Instruction *owner) noexcept {
    auto location_count = size_t{0u};
    for (auto *metadata : source->metadata_list()) {
        switch (metadata->derived_metadata_tag()) {
            case xir::DerivedMetadataTag::COMMENT: break;
            case xir::DerivedMetadataTag::LOCATION:
                location_count++;
                break;
            default: return false;
        }
    }
    return source->metadata_list().empty() ||
           (owner != nullptr && location_count <= 1u &&
            (location_count == 0u ||
             owner->find_metadata(
                 xir::DerivedMetadataTag::LOCATION) == nullptr));
}

void rehome_debug_metadata(
    xir::Instruction *source,
    xir::Instruction *owner) noexcept {
    auto *metadata = source->metadata_list().head();
    while (metadata != nullptr) {
        auto *next = metadata->next();
        owner->metadata_list().push_front(metadata->remove_self());
        metadata = next;
    }
}

[[nodiscard]] bool inspect_side(
    xir::BasicBlock *side, xir::BasicBlock *parent,
    xir::BasicBlock *&merge, size_t &count) noexcept {
    if (side == nullptr || !side->metadata_list().empty() ||
        single_predecessor(side) != parent || !side->is_terminated() ||
        !side->terminator()->isa<xir::BranchInst>()) {
        return false;
    }
    auto *target = static_cast<xir::BranchInst *>(side->terminator())
                       ->target_block();
    if (target == nullptr || (merge != nullptr && merge != target)) {
        return false;
    }
    merge = target;
    count = 0u;
    auto *metadata_owner = static_cast<xir::Instruction *>(nullptr);
    for (auto *instruction : side->instructions()) {
        if (instruction->is_terminator()) { continue; }
        if (!is_bounded_total_instruction(instruction)) { return false; }
        metadata_owner = instruction;
        count++;
    }
    return can_rehome_debug_metadata(
        side->terminator(), metadata_owner);
}

void rehome_side_exit_debug_metadata(
    xir::BasicBlock *side) noexcept {
    auto *terminator = side->terminator();
    auto *owner = static_cast<xir::Instruction *>(nullptr);
    for (auto *instruction : side->instructions()) {
        if (instruction != terminator) { owner = instruction; }
    }
    rehome_debug_metadata(terminator, owner);
}

[[nodiscard]] bool inspect_merge_phis(
    xir::BasicBlock *merge, xir::BasicBlock *true_block,
    xir::BasicBlock *false_block,
    size_t &generated_select_count) noexcept {
    generated_select_count = 0u;
    auto phi_count = size_t{0u};
    for (auto *instruction : merge->instructions()) {
        if (!instruction->isa<xir::PhiInst>()) { continue; }
        auto *phi = static_cast<xir::PhiInst *>(instruction);
        if (!is_narrow_scalar_type(phi->type())) { return false; }
        auto *true_value = static_cast<xir::Value *>(nullptr);
        auto *false_value = static_cast<xir::Value *>(nullptr);
        auto true_count = size_t{0u};
        auto false_count = size_t{0u};
        for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
            auto incoming = phi->incoming(i);
            if (incoming.block == true_block) {
                true_value = incoming.value;
                true_count++;
            } else if (incoming.block == false_block) {
                false_value = incoming.value;
                false_count++;
            }
        }
        if (true_count != 1u || false_count != 1u ||
            true_value == nullptr || false_value == nullptr ||
            true_value->type() != phi->type() ||
            false_value->type() != phi->type()) {
            return false;
        }
        generated_select_count += true_value != false_value;
        phi_count++;
    }
    return phi_count != 0u && generated_select_count != 0u &&
           generated_select_count <= 6u;
}

[[nodiscard]] bool all_operands_available_before_read(
    const luisa::unordered_set<xir::Instruction *> &movable,
    xir::BasicBlock *load_block, xir::ResourceReadInst *read,
    const xir::DomTree &dominance) noexcept {
    luisa::unordered_set<xir::Instruction *> available;
    for (auto *instruction : load_block->instructions()) {
        if (instruction == read) { break; }
        available.emplace(instruction);
    }
    for (auto *instruction : movable) {
        for (auto *operand_use : instruction->operand_uses()) {
            auto *operand = operand_use->value();
            if (operand == nullptr ||
                !operand->isa<xir::Instruction>()) {
                continue;
            }
            auto *operand_instruction =
                static_cast<xir::Instruction *>(operand);
            if (movable.contains(operand_instruction)) { continue; }
            auto *parent = operand_instruction->parent_block();
            if (parent == load_block) {
                if (!available.contains(operand_instruction)) {
                    return false;
                }
            } else if (parent == nullptr ||
                       !dominance.dominates(parent, load_block)) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool pattern_is_in_innermost_loop(
    const BufferReadLatencyCandidate &candidate,
    luisa::span<const xir::NaturalLoop> loops) noexcept {
    auto *innermost = static_cast<const xir::NaturalLoop *>(nullptr);
    for (auto &&loop : loops) {
        if (!loop.contains(candidate.step_block)) { continue; }
        if (innermost == nullptr ||
            loop.body_blocks.size() < innermost->body_blocks.size()) {
            innermost = &loop;
        }
    }
    return innermost != nullptr &&
           innermost->contains(candidate.load_block) &&
           innermost->contains(candidate.bridge_block) &&
           innermost->contains(candidate.true_block) &&
           innermost->contains(candidate.false_block) &&
           innermost->contains(candidate.merge_block);
}

[[nodiscard]] bool inspect_candidate(
    xir::BasicBlock *step_block,
    const WarpUniformityAnalysis &uniformity,
    const xir::DomTree &dominance,
    luisa::span<const xir::NaturalLoop> loops,
    BufferReadLatencyCandidate &candidate) noexcept {
    if (step_block == nullptr ||
        !step_block->metadata_list().empty() ||
        !step_block->is_terminated() ||
        !step_block->terminator()->isa<xir::ConditionalBranchInst>()) {
        return false;
    }
    auto *step_branch = static_cast<xir::ConditionalBranchInst *>(
        step_block->terminator());
    if (uniformity.classify(step_branch->condition()) !=
        ValueClass::varying) {
        return false;
    }
    auto step_count = nonterminator_count(step_block);
    if (step_count == 0u || step_count > 4u) { return false; }
    for (auto *instruction : step_block->instructions()) {
        if (!instruction->is_terminator() &&
            !is_bounded_total_instruction(instruction)) {
            return false;
        }
    }
    auto *step_metadata_owner = step_branch->condition() != nullptr &&
                                        step_branch->condition()
                                            ->isa<xir::Instruction>() ?
                                    static_cast<xir::Instruction *>(
                                        step_branch->condition()) :
                                    nullptr;
    if (step_metadata_owner == nullptr ||
        step_metadata_owner->parent_block() != step_block ||
        !can_rehome_debug_metadata(
            step_branch, step_metadata_owner)) {
        return false;
    }

    auto *true_block = step_branch->true_block();
    auto *false_block = step_branch->false_block();
    auto *merge_block = static_cast<xir::BasicBlock *>(nullptr);
    auto true_count = size_t{0u};
    auto false_count = size_t{0u};
    if (true_block == false_block ||
        !inspect_side(true_block, step_block, merge_block, true_count) ||
        !inspect_side(false_block, step_block, merge_block, false_count) ||
        true_count > 11u || false_count > 11u ||
        true_count + false_count > 14u) {
        return false;
    }
    auto generated_select_count = size_t{0u};
    if (!inspect_merge_phis(
            merge_block, true_block, false_block,
            generated_select_count) ||
        step_count + true_count + false_count +
                generated_select_count >
            24u) {
        return false;
    }
    auto load_is_merge_predecessor = false;

    auto *bridge_block = single_predecessor(step_block);
    if (bridge_block == nullptr ||
        !bridge_block->metadata_list().empty() ||
        nonterminator_count(bridge_block) != 0u ||
        !bridge_block->is_terminated() ||
        !bridge_block->terminator()->isa<xir::BranchInst>() ||
        !bridge_block->terminator()->metadata_list().empty() ||
        static_cast<xir::BranchInst *>(bridge_block->terminator())
                ->target_block() != step_block) {
        return false;
    }
    auto *load_block = single_predecessor(bridge_block);
    if (load_block == nullptr || !load_block->is_terminated() ||
        !load_block->terminator()->isa<xir::ConditionalBranchInst>()) {
        return false;
    }
    auto *load_branch = static_cast<xir::ConditionalBranchInst *>(
        load_block->terminator());
    merge_block->traverse_predecessors(
        false, [&](xir::BasicBlock *predecessor) noexcept {
            load_is_merge_predecessor |= predecessor == load_block;
        });
    auto bridge_is_true = load_branch->true_block() == bridge_block;
    auto bridge_is_false = load_branch->false_block() == bridge_block;
    if (bridge_is_true == bridge_is_false ||
        load_is_merge_predecessor ||
        uniformity.classify(load_branch->condition()) !=
            ValueClass::varying ||
        load_branch->condition() == nullptr ||
        !load_branch->condition()->isa<xir::ArithmeticInst>()) {
        return false;
    }
    auto *read = static_cast<xir::ResourceReadInst *>(nullptr);
    if (!is_direct_read_comparison(
            static_cast<xir::ArithmeticInst *>(
                load_branch->condition()),
            read) ||
        read->parent_block() != load_block ||
        uniformity.classify(read) != ValueClass::varying) {
        return false;
    }
    auto saw_read = false;
    auto post_read_nonterminator_count = size_t{0u};
    for (auto *instruction : load_block->instructions()) {
        if (instruction == read) {
            saw_read = true;
            continue;
        }
        if (saw_read && !instruction->is_terminator()) {
            post_read_nonterminator_count++;
            if (instruction != load_branch->condition()) {
                return false;
            }
        }
    }
    if (!saw_read || post_read_nonterminator_count != 1u) {
        return false;
    }

    luisa::unordered_set<xir::Instruction *> movable;
    auto collect = [&](xir::BasicBlock *block) noexcept {
        for (auto *instruction : block->instructions()) {
            if (!instruction->is_terminator()) {
                movable.emplace(instruction);
            }
        }
    };
    collect(step_block);
    collect(true_block);
    collect(false_block);
    if (!all_operands_available_before_read(
            movable, load_block, read, dominance)) {
        return false;
    }

    candidate = {
        .load_block = load_block,
        .bridge_block = bridge_block,
        .step_block = step_block,
        .true_block = true_block,
        .false_block = false_block,
        .merge_block = merge_block,
        .read = read,
        .load_branch = load_branch,
        .step_metadata_owner = step_metadata_owner,
        .generated_select_count = generated_select_count};
    return pattern_is_in_innermost_loop(candidate, loops);
}

[[nodiscard]] bool select_exact_step_diamond(
    const xir::ConditionalBranchInst *branch,
    const void *context) noexcept {
    auto *step_block = static_cast<const xir::BasicBlock *>(context);
    return branch != nullptr && branch->parent_block() == step_block;
}

[[nodiscard]] bool find_candidate(
    xir::Function *function,
    BufferReadLatencyCandidate &candidate) noexcept {
    if (function == nullptr || !function->is_definition() ||
        function->definition()->body_block() == nullptr ||
        xir::contains_structured_control_flow(function->definition())) {
        return false;
    }
    WarpUniformityAnalysis uniformity;
    uniformity.analyze(function);
    auto dominance = xir::compute_dom_tree(
        function, {.compute_dominance_frontiers = false});
    auto loops = xir::discover_natural_loops(
        function->definition(), dominance);
    if (loops.empty()) { return false; }
    auto found = false;
    function->definition()->traverse_basic_blocks(
        [&](xir::BasicBlock *block) noexcept {
            if (!found && inspect_candidate(
                              block, uniformity, dominance,
                              luisa::span{loops}, candidate)) {
                found = true;
            }
        });
    return found;
}

}// namespace

BufferReadLatencyHidingInfo hide_innermost_buffer_read_latency(
    xir::Function *function) noexcept {
    BufferReadLatencyHidingInfo result;
    static constexpr auto max_transform_count = size_t{8u};
    for (auto round = size_t{0u}; round < max_transform_count; round++) {
        BufferReadLatencyCandidate candidate;
        if (!find_candidate(function, candidate)) { break; }
        // Generic if-conversion correctly refuses to delete annotated arm
        // exits because it has no universal metadata merge rule. This exact
        // policy accepts diagnostic comment/location metadata only and moves
        // it to the last surviving instruction from the same arm.
        rehome_side_exit_debug_metadata(candidate.true_block);
        rehome_side_exit_debug_metadata(candidate.false_block);
        auto converted = xir::if_conversion_pass_run_on_function(
            function,
            {.max_arm_instruction_count = 11u,
             .max_total_instruction_count = 14u,
             .max_live_out_register_units = 6u,
             .max_speculation_cost = 24u,
             .candidate_filter = select_exact_step_diamond,
             .candidate_filter_context = candidate.step_block});
        if (!converted.succeeded() ||
            converted.converted_diamond_count != 1u ||
            candidate.step_block->terminator() == nullptr ||
            !candidate.step_block->terminator()->isa<xir::BranchInst>() ||
            static_cast<xir::BranchInst *>(
                candidate.step_block->terminator())
                    ->target_block() != candidate.merge_block) {
            break;
        }

        auto moved_count = size_t{0u};
        auto *step_terminator = candidate.step_block->terminator();
        while (!candidate.step_block->instructions().empty()) {
            auto *instruction =
                candidate.step_block->instructions().front();
            if (instruction == step_terminator) { break; }
            auto removed = instruction->remove_self();
            candidate.read->insert_before_self(std::move(removed));
            moved_count++;
        }
        rehome_debug_metadata(
            candidate.step_block->terminator(),
            candidate.step_metadata_owner);
        for (auto *instruction :
             candidate.merge_block->instructions()) {
            if (!instruction->isa<xir::PhiInst>()) { continue; }
            auto *phi = static_cast<xir::PhiInst *>(instruction);
            for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
                auto incoming = phi->incoming(i);
                if (incoming.block == candidate.step_block) {
                    phi->set_incoming(
                        i, incoming.value, candidate.load_block);
                }
            }
        }
        if (candidate.load_branch->true_block() ==
            candidate.bridge_block) {
            candidate.load_branch->set_true_target(
                candidate.merge_block);
        } else {
            candidate.load_branch->set_false_target(
                candidate.merge_block);
        }
        [[maybe_unused]] auto removed_bridge_terminator =
            candidate.bridge_block->terminator()->remove_self();
        [[maybe_unused]] auto removed_bridge =
            candidate.bridge_block->remove_self();
        [[maybe_unused]] auto removed_step_terminator =
            candidate.step_block->terminator()->remove_self();
        [[maybe_unused]] auto removed_step =
            candidate.step_block->remove_self();

        result.hidden_diamond_count++;
        result.moved_instruction_count += moved_count;
        result.generated_select_count +=
            candidate.generated_select_count;
    }
    return result;
}

}// namespace luisa::compute::simd::schedule
