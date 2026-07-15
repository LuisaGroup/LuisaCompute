#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
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
        luisa::unordered_set<Instruction *> dead;
        auto all_users_dead = [&](Instruction *inst) noexcept {
            for (auto &&use : inst->use_list()) {
                auto user = use->user();
                if (user != nullptr && user->isa<Instruction>() &&
                    !dead.contains(static_cast<Instruction *>(user))) {
                    return false;
                }
            }
            return true;
        };
        auto collect_if_dead = [&](Instruction *inst) noexcept {
            if (all_users_dead(inst)) {
                dead.emplace(inst);
            }
        };
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
        for (auto inst : dead) {
            inst->remove_self();
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
        for (auto &&inst : dead) {
            inst->remove_self();
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

void canonicalize_static_unstructured_branches_in_function(
    FunctionDefinition *definition, DCEInfo &info) noexcept {
    // If/Switch/Loop terminators define lexical scopes for source codegen and must
    // stay structured. Only a genuinely unstructured conditional branch can be
    // replaced with an unconditional branch here.
    luisa::vector<std::pair<ConditionalBranchInst *, BasicBlock *>> replacements;
    for (auto block : definition->basic_blocks()) {
        if (!block->is_terminated()) { continue; }
        if (auto terminator = block->terminator(); terminator->isa<ConditionalBranchInst>()) {
            auto cond_br = static_cast<ConditionalBranchInst *>(terminator);
            if (auto static_cond = try_evaluate_static_branch_condition(cond_br->condition())) {
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
        case DerivedInstructionTag::SWITCH: {
            auto switch_inst = static_cast<SwitchInst *>(terminator);
            if (auto static_cond = try_evaluate_static_switch_condition(switch_inst->value())) {
                auto taken = switch_inst->default_block();
                for (size_t i = 0u; i < switch_inst->case_count(); i++) {
                    if (switch_inst->case_value(i) == *static_cond) {
                        taken = switch_inst->case_block(i);
                        break;
                    }
                }
                visit(taken);
            } else {
                visit(switch_inst->default_block());
                for (size_t i = 0u; i < switch_inst->case_count(); i++) {
                    visit(switch_inst->case_block(i));
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

void repair_dead_control_flow_merges(
    FunctionDefinition *definition,
    const luisa::unordered_set<BasicBlock *> &exec_reachable) noexcept {
    for (auto block : definition->basic_blocks()) {
        if (!exec_reachable.contains(block) || !block->is_terminated()) { continue; }
        auto terminator = block->terminator();
        if (auto merge = terminator->control_flow_merge()) {
            if (auto merge_block = merge->merge_block();
                merge_block != nullptr && !exec_reachable.contains(merge_block)) {
                merge->set_merge_block(nullptr);
            }
        }
    }
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
        if (auto terminator = block->terminator(); terminator->isa<LoopInst>()) {
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
    if (auto definition = function->definition()) {
        for (auto block : definition->basic_blocks()) {
            if (!block->is_terminated()) {
                XIRBuilder builder;
                builder.set_insertion_point(block);
                builder.unreachable_();
            }
        }
    }
    luisa::vector<ManagedPtr<BasicBlock>> removed_blocks;
    for (;;) {
        auto prev_count = info.removed_inst_count + info.removed_block_count;
        if (auto definition = function->definition()) {
            canonicalize_static_unstructured_branches_in_function(definition, info);
            auto exec_reachable = collect_exec_reachable_blocks(definition);
            repair_dead_control_flow_merges(definition, exec_reachable);
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
