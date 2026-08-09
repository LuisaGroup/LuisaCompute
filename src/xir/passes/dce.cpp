#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static void eliminate_dead_code_in_function(Function *function, DCEInfo &info) noexcept {
    if (auto definition = function->definition()) {
        // Let R be the removable instructions and U(i) the instruction users
        // of i. The old repeated scan computed the least fixed point
        //
        //   D = {i in R | U(i) is a subset of D}
        //
        // from D = empty. Solve the identical equation by reverse-use graph
        // peeling: value-number R, count each candidate's users not yet in D,
        // seed zero-count candidates, and decrement their operand definitions
        // when a user enters D. Every instruction and Use is inspected O(1)
        // times. A removable cycle without a dead sink remains live, exactly
        // as it did under the least-fixed-point scan.
        luisa::vector<Instruction *> instructions;
        definition->traverse_instructions(
            [&](Instruction *inst) noexcept {
                instructions.emplace_back(inst);
            });
        info.dead_code_instruction_scan_count +=
            instructions.size();

        luisa::vector<Instruction *> candidates;
        candidates.reserve(instructions.size());
        luisa::vector<size_t> live_use_counts;
        live_use_counts.reserve(instructions.size());
        luisa::unordered_map<Instruction *, size_t>
            candidate_ids;
        candidate_ids.reserve(instructions.size());
        for (auto *inst : instructions) {
            auto removable =
                get_memory_info(inst)
                    .is_removable_if_unused();
            if (!removable &&
                inst->derived_instruction_tag() ==
                    DerivedInstructionTag::AUTODIFF_INTRINSIC) {
                auto *intrinsic =
                    static_cast<AutodiffIntrinsicInst *>(inst);
                removable =
                    intrinsic->op() ==
                    AutodiffIntrinsicOp::AUTODIFF_GRADIENT;
            }
            if (!removable) { continue; }
            auto id = candidates.size();
            candidates.emplace_back(inst);
            candidate_ids.emplace(inst, id);
            auto live_use_count = size_t{0u};
            for (auto &&use : inst->use_list()) {
                auto *user = use->user();
                live_use_count +=
                    user != nullptr &&
                    user->isa<Instruction>();
            }
            live_use_counts.emplace_back(live_use_count);
        }

        luisa::vector<uint8_t> dead(
            candidates.size(), uint8_t{0u});
        luisa::vector<size_t> work;
        work.reserve(candidates.size());
        for (auto id = size_t{0u};
             id < candidates.size(); ++id) {
            if (live_use_counts[id] == 0u) {
                work.emplace_back(id);
            }
        }
        luisa::vector<Instruction *> dead_in_order;
        dead_in_order.reserve(candidates.size());
        while (!work.empty()) {
            auto id = work.back();
            work.pop_back();
            if (dead[id] != 0u) { continue; }
            LUISA_DEBUG_ASSERT(
                live_use_counts[id] == 0u,
                "DCE worklist candidate still has live users.");
            dead[id] = uint8_t{1u};
            ++info.dead_code_worklist_pop_count;
            auto *inst = candidates[id];
            dead_in_order.emplace_back(inst);
            for (auto *operand_use : inst->operand_uses()) {
                auto *operand = operand_use->value();
                if (operand == nullptr ||
                    !operand->isa<Instruction>()) {
                    continue;
                }
                auto iter = candidate_ids.find(
                    static_cast<Instruction *>(operand));
                if (iter == candidate_ids.end() ||
                    dead[iter->second] != 0u) {
                    continue;
                }
                auto &live_use_count =
                    live_use_counts[iter->second];
                LUISA_DEBUG_ASSERT(
                    live_use_count != 0u,
                    "DCE live-use count underflow.");
                --live_use_count;
                if (live_use_count == 0u) {
                    work.emplace_back(iter->second);
                }
            }
        }
        // Keep every unlinked instruction alive until the whole dead
        // subgraph is detached. Destroying a definition before a still-linked
        // dead user would leave that user's operand Use pointing into freed
        // storage.
        luisa::vector<ManagedPtr<Instruction>> removed;
        removed.reserve(dead_in_order.size());
        for (auto *inst : dead_in_order) {
            removed.emplace_back(inst->remove_self());
            info.removed_inst_count++;
        }
    }
}

[[nodiscard]] static bool is_pointer_write_only(luisa::unordered_set<Instruction *> &known, Instruction *inst) noexcept {
    if (known.contains(inst)) { return true; }
    for (auto &&use : inst->use_list()) {
        if (auto user = use->user()) {
            if (!user->isa<Instruction>()) { return false; }
            switch (auto user_inst = static_cast<Instruction *>(user);
                    user_inst->derived_instruction_tag()) {
                case DerivedInstructionTag::STORE: {
                    break;
                }
                case DerivedInstructionTag::GEP: {
                    if (!is_pointer_write_only(known, user_inst)) { return false; }
                    break;
                }
                default: {
                    return false;
                }
            }
        }
    }
    known.emplace(inst);
    return true;
}

static void collect_inst_and_users_recursive(Instruction *inst, luisa::unordered_set<Instruction *> &collected) noexcept {
    if (collected.emplace(inst).second) {
        for (auto &&use : inst->use_list()) {
            if (auto user = use->user()) {
                LUISA_ASSERT(user->isa<Instruction>(), "Only instruction can be user.");
                collect_inst_and_users_recursive(static_cast<Instruction *>(user), collected);
            }
        }
    }
}

static void eliminate_dead_alloca_in_function(Function *function, DCEInfo &info) noexcept {
    if (auto definition = function->definition()) {
        luisa::unordered_set<Instruction *> dead;
        luisa::unordered_set<Instruction *> known_write_only;
        definition->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<AllocaInst>() && !dead.contains(inst) && is_pointer_write_only(known_write_only, inst)) {
                collect_inst_and_users_recursive(inst, dead);
            }
        });
        // See eliminate_dead_code_in_function: write-only pointer subgraphs
        // have the same user-before-definition lifetime requirement.
        luisa::vector<ManagedPtr<Instruction>> removed;
        removed.reserve(dead.size());
        for (auto *inst : dead) {
            removed.emplace_back(inst->remove_self());
            info.removed_inst_count++;
        }
    }
}

void eliminate_instructions_in_unreachable_blocks(const luisa::unordered_set<BasicBlock *> &blocks, DCEInfo &info) noexcept {
    luisa::vector<ManagedPtr<Instruction>> removed_instructions;
    for (auto b : blocks) {
        auto already_unreachable = false;
        if (!b->instructions().empty()) {
            if (auto inst = b->instructions().front(); inst->isa<UnreachableInst>()) {
                already_unreachable = inst->next() == b->instructions().tail_sentinel();
            }
        }
        if (already_unreachable) { continue; }
        while (!b->instructions().empty()) {
            auto inst = b->instructions().back();
            removed_instructions.emplace_back(inst->remove_self());
            info.removed_inst_count++;
        }
        XIRBuilder builder;
        builder.set_insertion_point(b);
        builder.unreachable_();
        LUISA_DEBUG_ASSERT(b->terminator()->isa<UnreachableInst>(),
                           "Block should be terminated by UnreachableInst.");
    }
}

void remove_phi_incomings_from_blocks(FunctionDefinition *definition,
                                      const luisa::unordered_set<BasicBlock *> &blocks) noexcept {
    if (blocks.empty()) { return; }
    for (auto block : definition->basic_blocks()) {
        if (!blocks.contains(block)) {
            block->traverse_instructions([&](Instruction *inst) noexcept {
                if (!inst->isa<PhiInst>()) { return; }
                auto *phi = static_cast<PhiInst *>(inst);
                for (size_t i = phi->incoming_count(); i-- > 0u;) {
                    if (blocks.contains(phi->incoming(i).block)) {
                        phi->remove_incoming(i);
                    }
                }
            });
        }
    }
}

[[nodiscard]] static luisa::optional<bool> try_evaluate_static_branch_condition(Value *cond) noexcept {
    LUISA_DEBUG_ASSERT(cond != nullptr, "Branch condition must not be null.");
    if (!cond->isa<Constant>()) { return luisa::nullopt; }
    auto static_cond = static_cast<Constant *>(cond);
    LUISA_DEBUG_ASSERT(static_cond->type()->is_bool(), "Branch condition must be a boolean constant.");
    return static_cond->as<bool>();
}

[[nodiscard]] static luisa::optional<
    IndexedBranchTerminatorInstruction::case_value_type>
try_evaluate_static_switch_condition(Value *cond) noexcept {
    LUISA_DEBUG_ASSERT(cond != nullptr, "Switch condition must not be null.");
    if (!cond->isa<Constant>()) { return luisa::nullopt; }
    auto static_cond = static_cast<Constant *>(cond);
    switch (auto t = static_cond->type(); t->tag()) {
        case Type::Tag::BOOL: return static_cond->as<bool>();
        case Type::Tag::INT8: return luisa::bit_cast<uint8_t>(static_cond->as<int8_t>());
        case Type::Tag::UINT8: return static_cond->as<uint8_t>();
        case Type::Tag::INT16: return luisa::bit_cast<uint16_t>(static_cond->as<int16_t>());
        case Type::Tag::UINT16: return static_cond->as<uint16_t>();
        case Type::Tag::INT32: return luisa::bit_cast<uint32_t>(static_cond->as<int32_t>());
        case Type::Tag::UINT32: return static_cond->as<uint32_t>();
        case Type::Tag::INT64: return luisa::bit_cast<uint64_t>(static_cond->as<int64_t>());
        case Type::Tag::UINT64: return static_cond->as<uint64_t>();
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid switch condition type.");
}

void canonicalize_static_unstructured_branches_in_function(
    FunctionDefinition *definition, DCEInfo &info) noexcept {
    // If/Switch/Loop terminators define lexical scopes for source codegen.
    // A constant-true canonical Loop.prepare may become Branch(body), but a
    // constant-false prepare must retain ConditionalBranch(body, merge):
    // Branch(merge) is not a valid structured-loop prepare form.
    luisa::unordered_map<BasicBlock *, LoopInst *> loop_prepares;
    for (auto block : definition->basic_blocks()) {
        if (block->is_terminated() && block->terminator()->isa<LoopInst>()) {
            auto loop = static_cast<LoopInst *>(block->terminator());
            if (loop->prepare_block() != nullptr) {
                loop_prepares.emplace(loop->prepare_block(), loop);
            }
        }
    }
    luisa::vector<std::pair<ConditionalBranchInst *, BasicBlock *>> replacements;
    for (auto block : definition->basic_blocks()) {
        if (!block->is_terminated()) { continue; }
        if (auto terminator = block->terminator(); terminator->isa<ConditionalBranchInst>()) {
            auto cond_br = static_cast<ConditionalBranchInst *>(terminator);
            if (auto static_cond = try_evaluate_static_branch_condition(cond_br->condition())) {
                if (!*static_cond) {
                    auto iter = loop_prepares.find(block);
                    if (iter != loop_prepares.end() &&
                        cond_br->true_block() == iter->second->body_block() &&
                        cond_br->false_block() == iter->second->merge_block()) {
                        continue;
                    }
                }
                if (auto taken = *static_cond ? cond_br->true_block() : cond_br->false_block()) {
                    replacements.emplace_back(cond_br, taken);
                }
            }
        }
    }
    for (auto [cond_br, taken] : replacements) {
        auto block = cond_br->parent_block();
        auto removed = cond_br->remove_self();
        XIRBuilder builder;
        builder.set_insertion_point(block);
        auto branch = builder.br(taken);
        for (auto metadata : removed->metadata_list()) {
            branch->metadata_list().push_front(metadata->clone());
        }
        info.removed_inst_count++;
    }
}

template<typename Visit>
void traverse_executable_successors(BasicBlock *block, Visit &&visit) noexcept {
    auto terminator = block->terminator();
    switch (terminator->derived_instruction_tag()) {
        case DerivedInstructionTag::IF: [[fallthrough]];
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto cond_br = static_cast<ConditionalBranchTerminatorInstruction *>(terminator);
            if (auto static_cond = try_evaluate_static_branch_condition(cond_br->condition())) {
                visit(*static_cond ? cond_br->true_block() : cond_br->false_block());
            } else {
                visit(cond_br->true_block());
                visit(cond_br->false_block());
            }
            break;
        }
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::INDEXED_BRANCH: {
            auto indexed_branch = static_cast<
                IndexedBranchTerminatorInstruction *>(terminator);
            if (auto static_cond =
                    try_evaluate_static_switch_condition(
                        indexed_branch->value())) {
                auto taken = indexed_branch->default_block();
                for (size_t i = 0u;
                     i < indexed_branch->case_count(); i++) {
                    if (indexed_branch->case_value(i) == *static_cond) {
                        taken = indexed_branch->case_block(i);
                        break;
                    }
                }
                visit(taken);
            } else {
                visit(indexed_branch->default_block());
                for (size_t i = 0u;
                     i < indexed_branch->case_count(); i++) {
                    visit(indexed_branch->case_block(i));
                }
            }
            break;
        }
        default: {
            block->traverse_successors(false, [&](BasicBlock *successor) noexcept {
                visit(successor);
            });
            break;
        }
    }
}

[[nodiscard]] luisa::unordered_set<BasicBlock *> collect_exec_reachable_blocks(
    FunctionDefinition *definition) noexcept {
    luisa::unordered_set<BasicBlock *> reachable;
    luisa::unordered_set<BasicBlock *> owned;
    for (auto block : definition->basic_blocks()) { owned.emplace(block); }
    luisa::vector<BasicBlock *> work_list;
    auto add_to_work_list = [&](BasicBlock *block) noexcept {
        if (block != nullptr && owned.contains(block) && reachable.emplace(block).second) {
            work_list.emplace_back(block);
        }
    };
    add_to_work_list(definition->body_block());
    while (!work_list.empty()) {
        auto block = work_list.back();
        work_list.pop_back();
        if (block->is_terminated()) {
            traverse_executable_successors(block, [&](BasicBlock *successor) noexcept {
                add_to_work_list(successor);
            });
        }
    }
    return reachable;
}

[[nodiscard]] luisa::unordered_set<BasicBlock *> collect_structural_shell_blocks(
    FunctionDefinition *definition,
    const luisa::unordered_set<BasicBlock *> &exec_reachable) noexcept {
    // Loop body/update are structural raw pointers rather than CFG operands. Keep
    // their blocks as unreachable shells when the loop itself is executable so
    // source codegen retains the lexical loop frame.
    luisa::unordered_set<BasicBlock *> shells;
    for (auto block : definition->basic_blocks()) {
        if (!exec_reachable.contains(block) || !block->is_terminated()) { continue; }
        auto terminator = block->terminator();
        if (auto merge = terminator->control_flow_merge()) {
            if (auto merge_block = merge->merge_block()) {
                shells.emplace(merge_block);
            }
        }
        if (terminator->isa<LoopInst>()) {
            auto loop = static_cast<LoopInst *>(terminator);
            if (auto body = loop->body_block()) { shells.emplace(body); }
            if (auto update = loop->update_block()) { shells.emplace(update); }
        }
    }
    return shells;
}

void eliminate_unreachable_blocks_in_function(
    FunctionDefinition *definition, const luisa::unordered_set<BasicBlock *> &exec_reachable,
    DCEInfo &info, luisa::vector<ManagedPtr<BasicBlock>> &removed_blocks) noexcept {
    auto structural_shells = collect_structural_shell_blocks(definition, exec_reachable);
    luisa::unordered_set<BasicBlock *> unreachable;
    for (auto block : definition->basic_blocks()) {
        if (!exec_reachable.contains(block)) { unreachable.emplace(block); }
    }
    remove_phi_incomings_from_blocks(definition, unreachable);
    eliminate_instructions_in_unreachable_blocks(unreachable, info);
    luisa::vector<BasicBlock *> to_remove;
    for (auto block : definition->basic_blocks()) {
        if (block != definition->body_block() && unreachable.contains(block) &&
            !structural_shells.contains(block) && block->use_list().empty()) {
            to_remove.emplace_back(block);
        }
    }
    for (auto block : to_remove) {
        removed_blocks.emplace_back(block->remove_self());
        info.removed_block_count++;
    }
}

void fix_phi_nodes_in_function(Function *function, luisa::vector<PhiInst *> &phi_nodes) noexcept {
    if (auto definition = function->definition()) {
        luisa::unordered_set<BasicBlock *> predecessors;
        definition->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<PhiInst>()) {
                predecessors.clear();
                auto phi = static_cast<PhiInst *>(inst);
                phi_nodes.emplace_back(phi);
                phi->parent_block()->traverse_predecessors(false, [&](auto block) noexcept {
                    predecessors.emplace(block);
                });
                for (size_t i = phi->incoming_count(); i-- > 0u;) {
                    if (auto incoming = phi->incoming(i); !predecessors.contains(incoming.block)) {
                        phi->remove_incoming(i);
                    }
                }
                LUISA_ASSERT(phi->incoming_count() == predecessors.size());
            }
        });
    }
}

static void eliminate_redundant_phi_nodes(luisa::vector<PhiInst *> &phi_nodes, DCEInfo &info) noexcept {
    for (;;) {
        auto prev_size = phi_nodes.size();
        phi_nodes.erase(std::remove_if(phi_nodes.begin(), phi_nodes.end(),
                                       remove_redundant_phi_instruction),
                        phi_nodes.end());
        auto removed = prev_size - phi_nodes.size();
        if (removed == 0u) { break; }
        info.removed_inst_count += removed;
    }
}

void run_dce_pass_on_function(Function *function, DCEInfo &info) noexcept {
    if (function == nullptr) { return; }
    if (auto *definition = function->definition();
        definition != nullptr && definition->body_block() == nullptr) {
        return;
    }
    if (auto definition = function->definition()) {
        for (auto block : definition->basic_blocks()) {
            if (!block->is_terminated()) {
                XIRBuilder builder;
                builder.set_insertion_point(block);
                builder.unreachable_();
                ++info.inserted_terminator_count;
            }
        }
    }
    luisa::vector<ManagedPtr<BasicBlock>> removed_blocks;
    for (;;) {
        auto prev_count = info.removed_inst_count + info.removed_block_count;
        if (auto definition = function->definition()) {
            canonicalize_static_unstructured_branches_in_function(definition, info);
            auto exec_reachable = collect_exec_reachable_blocks(definition);
            eliminate_unreachable_blocks_in_function(definition, exec_reachable, info, removed_blocks);
            {
                luisa::vector<PhiInst *> phi_nodes;
                fix_phi_nodes_in_function(function, phi_nodes);
                eliminate_redundant_phi_nodes(phi_nodes, info);
            }
        }
        eliminate_dead_code_in_function(function, info);
        eliminate_dead_alloca_in_function(function, info);
        if (info.removed_inst_count + info.removed_block_count == prev_count) { return; }
    }
}

}// namespace detail

DCEInfo dce_pass_run_on_function(Function *function) noexcept {
    DCEInfo info;
    detail::run_dce_pass_on_function(function, info);
    return info;
}

DCEInfo dce_pass_run_on_module(Module *module, PassReport *report) noexcept {
    DCEInfo info;
    if (module == nullptr) {
        if (report != nullptr) {
            report->set("removed_inst", 0u);
            report->set("removed_block", 0u);
            report->set("inserted_terminator", 0u);
            report->set("dead_code_instruction_scan", 0u);
            report->set("dead_code_worklist_pop", 0u);
        }
        return info;
    }
    for (auto f : module->function_list()) {
        detail::run_dce_pass_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("removed_inst", info.removed_inst_count);
        report->set("removed_block", info.removed_block_count);
        report->set(
            "inserted_terminator",
            info.inserted_terminator_count);
        report->set(
            "dead_code_instruction_scan",
            info.dead_code_instruction_scan_count);
        report->set(
            "dead_code_worklist_pop",
            info.dead_code_worklist_pop_count);
    }
    return info;
}

}// namespace luisa::compute::xir
