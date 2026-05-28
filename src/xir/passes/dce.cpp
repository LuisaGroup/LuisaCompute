#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static void eliminate_dead_code_in_function(Function *function, DCEInfo &info) noexcept {
    if (auto definition = function->definition()) {
        luisa::unordered_set<Instruction *> dead;
        auto all_users_dead = [&](Instruction *inst) noexcept {
            for (auto &&use : inst->use_list()) {
                auto user = use->user();
                if (user != nullptr && user->isa<Instruction>() &&
                    !dead.contains(static_cast<Instruction *>(user))) {
                    return false;// at least one user is still live
                }
            }
            return true;// no user or all users are dead
        };
        auto collect_if_dead = [&](Instruction *inst) noexcept {
            if (all_users_dead(inst)) {
                dead.emplace(inst);
            }
        };
        // collect all dead instructions
        for (;;) {
            auto prev_size = dead.size();
            definition->traverse_instructions([&](Instruction *inst) noexcept {
                if (!dead.contains(inst)) {
                    auto mem = get_memory_info(inst);
                    if (mem.is_removable_if_unused()) {
                        collect_if_dead(inst);
                    } else if (inst->derived_instruction_tag() == DerivedInstructionTag::AUTODIFF_INTRINSIC) {
                        auto intrinsic = static_cast<AutodiffIntrinsicInst *>(inst);
                        if (intrinsic->op() == AutodiffIntrinsicOp::AUTODIFF_GRADIENT) {
                            collect_if_dead(inst);
                        }
                    }
                }
            });
            if (dead.size() == prev_size) { break; }
        }
        // remove dead instructions
        for (auto inst : dead) {
            inst->remove_self();
            info.removed_inst_count++;
        }
    }
}

// if we find a pointer is only written to and never read, we can remove it
[[nodiscard]] static bool is_pointer_write_only(luisa::unordered_set<Instruction *> &known, Instruction *inst) noexcept {
    if (known.contains(inst)) { return true; }
    for (auto &&use : inst->use_list()) {
        if (auto user = use->user()) {
            if (!user->isa<Instruction>()) { return false; }
            switch (auto user_inst = static_cast<Instruction *>(user);
                    user_inst->derived_instruction_tag()) {
                case DerivedInstructionTag::STORE: {
                    // store is good
                    break;
                }
                case DerivedInstructionTag::GEP: {
                    // if the GEP is ever read, we can't remove the pointer
                    if (!is_pointer_write_only(known, user_inst)) { return false; }
                    // otherwise we are safe
                    break;
                }
                default: {
                    // just be conservative and assume the pointer is read
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
        for (auto &&inst : dead) {
            inst->remove_self();
            info.removed_inst_count++;
        }
    }
}

[[nodiscard]] static bool is_block_terminated_by_unreachable(BasicBlock *block) noexcept {
    return block->terminator()->isa<UnreachableInst>();
}

void eliminate_instructions_in_unreachable_blocks(const luisa::unordered_set<BasicBlock *> &blocks, DCEInfo &info) noexcept {
    for (auto b : blocks) {
        while (!b->instructions().empty()) {
            auto inst = b->instructions().front();
            if (inst->isa<UnreachableInst>()) { break; }
            inst->remove_self();
            info.removed_inst_count++;
        }
        if (!b->is_terminated()) {
            XIRBuilder builder;
            builder.set_insertion_point(b);
            builder.unreachable_();
        }
        LUISA_DEBUG_ASSERT(b->terminator()->isa<UnreachableInst>(),
                           "Block should be terminated by UnreachableInst.");
    }
}

void propagate_unreachable_marks_in_function(Function *function, DCEInfo &info) noexcept {
    // run a backward dataflow analysis to propagate unreachable marks:
    // we should mark a block as unreachable if all its successors are marked as unreachable
    if (auto definition = function->definition()) {
        luisa::vector<BasicBlock *> postorder;
        definition->traverse_basic_blocks(BasicBlockTraversalOrder::POST_ORDER, [&](BasicBlock *block) noexcept {
            postorder.emplace_back(block);
        });
        luisa::unordered_set<BasicBlock *> unreachable;
        for (;;) {
            auto prev_reachable_count = unreachable.size();
            for (auto block : postorder) {
                if (!unreachable.contains(block)) {
                    if (is_block_terminated_by_unreachable(block)) {
                        unreachable.emplace(block);
                    } else {
                        auto has_any_successor = false;
                        auto all_successors_unreachable = true;
                        block->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                            has_any_successor = true;
                            if (succ != block && !unreachable.contains(succ) &&
                                !is_block_terminated_by_unreachable(succ)) {
                                all_successors_unreachable = false;
                            }
                        });
                        if (has_any_successor && all_successors_unreachable) {
                            unreachable.emplace(block);
                        }
                    }
                }
            }
            if (unreachable.size() == prev_reachable_count) { break; }
        }
        // eliminate unreachable blocks
        eliminate_instructions_in_unreachable_blocks(unreachable, info);
    }
}

[[nodiscard]] static luisa::optional<bool> try_evaluate_static_branch_condition(Value *cond) noexcept {
    LUISA_DEBUG_ASSERT(cond != nullptr, "Branch condition must not be null.");
    if (!cond->isa<Constant>()) { return luisa::nullopt; }
    auto static_cond = static_cast<Constant *>(cond);
    LUISA_DEBUG_ASSERT(static_cond->type()->is_bool(), "Branch condition must be a boolean constant.");
    return static_cond->as<bool>();
}

[[nodiscard]] static luisa::optional<SwitchInst::case_value_type> try_evaluate_static_switch_condition(Value *cond) noexcept {
    LUISA_DEBUG_ASSERT(cond != nullptr, "Switch condition must not be null.");
    if (!cond->isa<Constant>()) { return luisa::nullopt; }
    return [static_cond = static_cast<Constant *>(cond)]() noexcept -> SwitchInst::case_value_type {
        switch (auto t = static_cond->type(); t->tag()) {
            case Type::Tag::BOOL: return static_cond->as<bool>();
            case Type::Tag::INT8: return static_cond->as<int8_t>();
            case Type::Tag::UINT8: return static_cond->as<uint8_t>();
            case Type::Tag::INT16: return static_cond->as<int16_t>();
            case Type::Tag::UINT16: return static_cond->as<uint16_t>();
            case Type::Tag::INT32: return static_cond->as<int32_t>();
            case Type::Tag::UINT32: return static_cast<SwitchInst::case_value_type>(static_cond->as<uint32_t>());
            case Type::Tag::INT64: return static_cast<SwitchInst::case_value_type>(static_cond->as<int64_t>());
            case Type::Tag::UINT64: return static_cast<SwitchInst::case_value_type>(static_cond->as<uint64_t>());
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid switch condition type.");
    }();
}

void eliminate_unreachable_blocks_in_function(Function *function, DCEInfo &info, luisa::vector<ManagedPtr<BasicBlock>> &removed_blocks) noexcept {
    if (auto definition = function->definition()) {
        luisa::unordered_set<BasicBlock *> reachable;
        luisa::unordered_set<BasicBlock *> visited;
        luisa::unordered_set<BasicBlock *> unreachable;
        luisa::vector<BasicBlock *> work_list;
        auto add_to_work_list = [&](BasicBlock *block) noexcept {
            if (block != nullptr && visited.emplace(block).second) {
                work_list.emplace_back(block);
            }
        };
        // Seed reachability from the entry block.
        reachable.emplace(definition->body_block());
        add_to_work_list(definition->body_block());
        // Forward reachability analysis: propagate from entry block via successors.
        while (!work_list.empty()) {
            auto b = work_list.back();
            work_list.pop_back();
            // Traverse successors to discover transitively reachable blocks.
            b->traverse_successors(true, [&](auto succ) noexcept {
                if (reachable.emplace(succ).second) {
                    add_to_work_list(succ);
                }
            });
            // Also fold constant branches to identify statically dead successors.
            switch (auto terminator = b->terminator(); terminator->derived_instruction_tag()) {
                case DerivedInstructionTag::IF: [[fallthrough]];
                case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                    auto cond_br_inst = static_cast<ConditionalBranchTerminatorInstruction *>(terminator);
                    if (auto static_cond = try_evaluate_static_branch_condition(cond_br_inst->condition())) {
                        unreachable.emplace(*static_cond ? cond_br_inst->false_block() : cond_br_inst->true_block());
                    }
                    break;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto switch_inst = static_cast<SwitchInst *>(terminator);
                    if (auto static_cond = try_evaluate_static_switch_condition(switch_inst->value())) {
                        auto any_match = false;
                        for (size_t i = 0; i < switch_inst->case_count(); i++) {
                            if (switch_inst->case_value(i) == *static_cond) {
                                any_match = true;
                            } else {
                                LUISA_DEBUG_ASSERT(switch_inst->case_block(i) != nullptr, "Switch case block must not be null.");
                                unreachable.emplace(switch_inst->case_block(i));
                            }
                        }
                        if (any_match) {
                            LUISA_DEBUG_ASSERT(switch_inst->default_block() != nullptr, "Switch default block must not be null.");
                            unreachable.emplace(switch_inst->default_block());
                        }
                    }
                    break;
                }
                default: break;
            }
        }
        // Any block not reached from the entry is unreachable.
        definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            if (!reachable.contains(block)) {
                unreachable.emplace(block);
            }
        });
        // Eliminate instructions in unreachable blocks.
        eliminate_instructions_in_unreachable_blocks(unreachable, info);
        // now we can remove non-entry blocks without users from the function
        {
            // Collect blocks referenced as structural merge targets; these must survive
            // for backends that require structured control flow (e.g., SPIR-V OpLoopMerge / OpSelectionMerge),
            // even though merge pointers don't participate in the use-list.
            luisa::unordered_set<BasicBlock *> structural_merges;
            definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
                if (auto merge = block->terminator()->control_flow_merge()) {
                    if (auto mb = merge->merge_block()) { structural_merges.emplace(mb); }
                }
            });
            work_list.clear();
            for (auto block : definition->basic_blocks()) {
                if (block != definition->body_block() && block->use_list().empty() &&
                    !structural_merges.contains(block)) {
                    work_list.emplace_back(block);
                }
            }
            for (auto block : work_list) {
                removed_blocks.emplace_back(block->remove_self());
                info.removed_block_count++;
            }
        }
    }
}

void fix_phi_nodes_in_function(Function *function, luisa::vector<PhiInst *> &phi_nodes) noexcept {
    if (auto definition = function->definition()) {
        luisa::vector<PhiIncoming> valid_incomings;
        luisa::unordered_set<BasicBlock *> predecessors;
        definition->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<PhiInst>()) {
                valid_incomings.clear();
                predecessors.clear();
                auto phi = static_cast<PhiInst *>(inst);
                phi_nodes.emplace_back(phi);
                phi->parent_block()->traverse_predecessors(false, [&](auto block) noexcept {
                    predecessors.emplace(block);
                });
                for (size_t i = 0; i < phi->incoming_count(); i++) {
                    if (auto incoming = phi->incoming(i); predecessors.contains(incoming.block)) {
                        valid_incomings.emplace_back(incoming);
                    }
                }
                LUISA_ASSERT(valid_incomings.size() == predecessors.size());
                phi->set_incoming_count(valid_incomings.size());
                for (size_t i = 0; i < valid_incomings.size(); i++) {
                    auto incoming = valid_incomings[i];
                    phi->set_incoming(i, incoming.value, incoming.block);
                }
            }
        });
    }
}

void fix_control_flow_merges_in_function(Function *function) noexcept {
    if (auto definition = function->definition()) {
        definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            auto term = block->terminator();
            if (auto merge = term->control_flow_merge()) {
                // Loop terminators require a merge block for structured codegen (e.g., SPIR-V OpLoopMerge),
                // even when the merge is only reachable via an `unreachable` terminator (infinite loops).
                if (term->isa<LoopInst>() || term->isa<SimpleLoopInst>()) { return; }
                if (auto merge_block = merge->merge_block();
                    merge_block != nullptr && is_block_terminated_by_unreachable(merge_block)) {
                    merge->set_merge_block(nullptr);
                }
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
    luisa::vector<ManagedPtr<BasicBlock>> removed_blocks;
    for (;;) {
        auto prev_count = info.removed_inst_count + info.removed_block_count;
        propagate_unreachable_marks_in_function(function, info);
        eliminate_unreachable_blocks_in_function(function, info, removed_blocks);
        fix_control_flow_merges_in_function(function);
        {
            luisa::vector<PhiInst *> phi_nodes;
            fix_phi_nodes_in_function(function, phi_nodes);
            eliminate_redundant_phi_nodes(phi_nodes, info);
        }
        eliminate_dead_code_in_function(function, info);
        eliminate_dead_alloca_in_function(function, info);
        // if we didn't remove any instruction, we are done
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
    for (auto f : module->function_list()) {
        detail::run_dce_pass_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("removed_inst", info.removed_inst_count);
        report->set("removed_block", info.removed_block_count);
    }
    return info;
}

}// namespace luisa::compute::xir
