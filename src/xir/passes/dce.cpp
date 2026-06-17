#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/loop.h>

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

[[nodiscard]] static bool is_block_terminated_by_unreachable(BasicBlock *block) noexcept {
    if (!block->is_terminated()) { return false; }
    return block->terminator()->isa<UnreachableInst>();
}

[[nodiscard]] static BasicBlock *find_owned_block(Value *value, const luisa::unordered_set<BasicBlock *> &owned) noexcept {
    if (value == nullptr) { return nullptr; }
    for (auto block : owned) {
        if (static_cast<Value *>(block) == value) { return block; }
    }
    return nullptr;
}

template<typename Visit>
void traverse_structural_successors(BasicBlock *block, const luisa::unordered_set<BasicBlock *> &owned,
                                    Visit &&visit) noexcept {
    if (block == nullptr || !block->is_terminated()) { return; }
    auto *term = block->terminator();
    for (auto use : term->operand_uses()) {
        if (auto *succ = find_owned_block(use->value(), owned)) { visit(succ); }
    }
    if (auto *merge = term->control_flow_merge(); merge != nullptr) {
        if (auto *merge_block = merge->merge_block(); merge_block != nullptr && owned.contains(merge_block)) {
            visit(merge_block);
        }
    }
    if (term->isa<LoopInst>()) {
        auto *loop = static_cast<LoopInst *>(term);
        if (auto *body = loop->body_block(); body != nullptr && owned.contains(body)) { visit(body); }
        if (auto *update = loop->update_block(); update != nullptr && owned.contains(update)) { visit(update); }
    }
}

luisa::unordered_set<BasicBlock *> collect_structurally_reachable_blocks(FunctionDefinition *definition) noexcept {
    luisa::unordered_set<BasicBlock *> reachable;
    luisa::unordered_set<BasicBlock *> owned;
    for (auto block : definition->basic_blocks()) { owned.emplace(block); }
    luisa::vector<BasicBlock *> work_list;
    auto add_to_work_list = [&](BasicBlock *block) noexcept {
        if (block != nullptr && reachable.emplace(block).second) {
            work_list.emplace_back(block);
        }
    };
    add_to_work_list(definition->body_block());
    while (!work_list.empty()) {
        auto *block = work_list.back();
        work_list.pop_back();
        traverse_structural_successors(block, owned, add_to_work_list);
    }
    return reachable;
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
    definition->traverse_instructions([&](Instruction *inst) noexcept {
        if (!inst->isa<PhiInst>()) { return; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (size_t i = phi->incoming_count(); i-- > 0u;) {
            if (blocks.contains(phi->incoming(i).block)) {
                phi->remove_incoming(i);
            }
        }
    });
}

void propagate_unreachable_marks_in_function(Function *function, DCEInfo &info) noexcept {
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
        remove_phi_incomings_from_blocks(definition, unreachable);
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

void collect_unreachable_region(BasicBlock *entry, BasicBlock *merge,
                                const luisa::unordered_set<BasicBlock *> &owned,
                                luisa::unordered_set<BasicBlock *> &unreachable) noexcept {
    luisa::vector<BasicBlock *> work_list;
    auto add_to_work_list = [&](BasicBlock *block) noexcept {
        if (block != nullptr && block != merge && owned.contains(block) && unreachable.emplace(block).second) {
            work_list.emplace_back(block);
        }
    };
    add_to_work_list(entry);
    while (!work_list.empty()) {
        auto block = work_list.back();
        work_list.pop_back();
        traverse_structural_successors(block, owned, add_to_work_list);
    }
}

void eliminate_unreachable_blocks_in_function(Function *function, DCEInfo &info, luisa::vector<ManagedPtr<BasicBlock>> &removed_blocks) noexcept {
    if (auto definition = function->definition()) {
        auto reachable = collect_structurally_reachable_blocks(definition);
        luisa::unordered_set<BasicBlock *> owned;
        for (auto *block : definition->basic_blocks()) { owned.emplace(block); }
        luisa::unordered_set<BasicBlock *> unreachable;
        for (auto *block : reachable) {
            if (!block->is_terminated()) { continue; }
            switch (auto terminator = block->terminator(); terminator->derived_instruction_tag()) {
                case DerivedInstructionTag::IF: [[fallthrough]];
                case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                    auto cond_br_inst = static_cast<ConditionalBranchTerminatorInstruction *>(terminator);
                    if (auto static_cond = try_evaluate_static_branch_condition(cond_br_inst->condition())) {
                        auto dead = *static_cond ? cond_br_inst->false_block() : cond_br_inst->true_block();
                        auto merge = terminator->control_flow_merge();
                        if (merge != nullptr && merge->merge_block() != nullptr) {
                            collect_unreachable_region(dead, merge->merge_block(), owned, unreachable);
                        } else if (dead != nullptr && owned.contains(dead)) {
                            unreachable.emplace(dead);
                        }
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
                                collect_unreachable_region(switch_inst->case_block(i), switch_inst->merge_block(), owned, unreachable);
                            }
                        }
                        if (any_match) {
                            LUISA_DEBUG_ASSERT(switch_inst->default_block() != nullptr, "Switch default block must not be null.");
                            collect_unreachable_region(switch_inst->default_block(), switch_inst->merge_block(), owned, unreachable);
                        }
                    }
                    break;
                }
                default: break;
            }
        }
        for (auto *block : definition->basic_blocks()) {
            if (!reachable.contains(block)) {
                unreachable.emplace(block);
            }
        }
        remove_phi_incomings_from_blocks(definition, unreachable);
        eliminate_instructions_in_unreachable_blocks(unreachable, info);
        {
            luisa::vector<BasicBlock *> work_list;
            for (auto block : definition->basic_blocks()) {
                if (block != definition->body_block() && block->use_list().empty() &&
                    !reachable.contains(block)) {
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

void fix_control_flow_merges_in_function(Function *function) noexcept {
    if (auto definition = function->definition()) {
        definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            auto term = block->terminator();
            if (auto merge = term->control_flow_merge()) {
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
    if (auto definition = function->definition()) {
        definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            if (!block->is_terminated()) {
                XIRBuilder builder;
                builder.set_insertion_point(block);
                builder.unreachable_();
            }
        });
    }
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
