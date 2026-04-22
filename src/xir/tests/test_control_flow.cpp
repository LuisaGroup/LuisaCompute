#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/Canonicalize_Control_Flow.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>

using namespace luisa;
using namespace luisa::compute;

namespace {

struct ReachableIR {
    luisa::unordered_set<xir::BasicBlock *> blocks_seen;
    luisa::unordered_set<xir::Instruction *> instructions_seen;
    luisa::vector<xir::BasicBlock *> blocks;
    luisa::vector<xir::Instruction *> instructions;
};

[[noreturn]] void fail_with_dump(const xir::Module *module, luisa::string_view message) {
    auto dump = xir::xir_to_text_translate(module, true);
    LUISA_ERROR_WITH_LOCATION("{}\n\nXIR dump:\n{}", message, dump);
}

void require(const xir::Module *module, bool condition, luisa::string_view message) {
    if (!condition) {
        fail_with_dump(module, message);
    }
}

void collect_reachable_block(xir::BasicBlock *block, ReachableIR &reachable) {
    if (block == nullptr || !reachable.blocks_seen.emplace(block).second) {
        return;
    }
    reachable.blocks.emplace_back(block);
    for (auto inst : block->instructions()) {
        if (reachable.instructions_seen.emplace(inst).second) {
            reachable.instructions.emplace_back(inst);
        }
        switch (inst->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF: {
                auto if_inst = static_cast<xir::IfInst *>(inst);
                collect_reachable_block(if_inst->true_block(), reachable);
                collect_reachable_block(if_inst->false_block(), reachable);
                collect_reachable_block(if_inst->merge_block(), reachable);
                break;
            }
            case xir::DerivedInstructionTag::SWITCH: {
                auto switch_inst = static_cast<xir::SwitchInst *>(inst);
                for (auto i = 0u; i < switch_inst->case_count(); i++) {
                    collect_reachable_block(switch_inst->case_block(i), reachable);
                }
                collect_reachable_block(switch_inst->default_block(), reachable);
                collect_reachable_block(switch_inst->merge_block(), reachable);
                break;
            }
            case xir::DerivedInstructionTag::LOOP: {
                auto loop_inst = static_cast<xir::LoopInst *>(inst);
                collect_reachable_block(loop_inst->prepare_block(), reachable);
                collect_reachable_block(loop_inst->body_block(), reachable);
                collect_reachable_block(loop_inst->update_block(), reachable);
                collect_reachable_block(loop_inst->merge_block(), reachable);
                break;
            }
            case xir::DerivedInstructionTag::SIMPLE_LOOP: {
                auto simple_loop = static_cast<xir::SimpleLoopInst *>(inst);
                collect_reachable_block(simple_loop->body_block(), reachable);
                collect_reachable_block(simple_loop->merge_block(), reachable);
                break;
            }
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto branch = static_cast<xir::ConditionalBranchInst *>(inst);
                collect_reachable_block(branch->true_block(), reachable);
                collect_reachable_block(branch->false_block(), reachable);
                break;
            }
            case xir::DerivedInstructionTag::BRANCH:
            case xir::DerivedInstructionTag::BREAK:
            case xir::DerivedInstructionTag::CONTINUE: {
                auto branch = static_cast<xir::BranchTerminatorInstruction *>(inst);
                collect_reachable_block(branch->target_block(), reachable);
                break;
            }
            default: break;
        }
    }
}

ReachableIR collect_reachable_ir(xir::FunctionDefinition *definition) {
    ReachableIR reachable;
    collect_reachable_block(definition->body_block(), reachable);
    return reachable;
}

size_t count_tag(const ReachableIR &reachable, xir::DerivedInstructionTag tag) {
    auto count = size_t{0u};
    for (auto inst : reachable.instructions) {
        if (inst->derived_instruction_tag() == tag) {
            count++;
        }
    }
    return count;
}

luisa::vector<xir::Instruction *> collect_by_tag(const ReachableIR &reachable, xir::DerivedInstructionTag tag) {
    luisa::vector<xir::Instruction *> instructions;
    for (auto inst : reachable.instructions) {
        if (inst->derived_instruction_tag() == tag) {
            instructions.emplace_back(inst);
        }
    }
    return instructions;
}

bool block_contains_instruction(const xir::BasicBlock *block, const xir::Instruction *instruction) {
    for (auto inst : block->instructions()) {
        if (inst == instruction) {
            return true;
        }
    }
    return false;
}

size_t count_non_terminator_instructions(const xir::BasicBlock *block) {
    auto count = size_t{0u};
    for (auto inst : block->instructions()) {
        if (!inst->is_terminator()) {
            count++;
        }
    }
    return count;
}

void verify_operand_use_consistency(const xir::Module *module, const ReachableIR &reachable) {
    for (auto block : reachable.blocks) {
        for (auto use : block->use_list()) {
            require(module, use->value() == block, "Basic block use-list contains a mismatched value.");
            auto user = use->user();
            require(module, user != nullptr, "Basic block use-list contains a null user.");
            auto found = false;
            for (auto operand_use : user->operand_uses()) {
                if (operand_use == use) {
                    found = true;
                    break;
                }
            }
            require(module, found, "Basic block use-list is not mirrored by the user operand list.");
        }
    }
    for (auto inst : reachable.instructions) {
        for (auto operand_use : inst->operand_uses()) {
            require(module, operand_use->user() == inst, "Instruction operand use points to the wrong user.");
            if (auto value = operand_use->value()) {
                auto found = false;
                for (auto listed_use : value->use_list()) {
                    if (listed_use == operand_use) {
                        found = true;
                        break;
                    }
                }
                require(module, found, "Operand use is missing from the value use-list.");
            }
        }
        for (auto use : inst->use_list()) {
            require(module, use->value() == inst, "Instruction use-list contains a mismatched value.");
            auto user = use->user();
            require(module, user != nullptr, "Instruction use-list contains a null user.");
            auto found = false;
            for (auto operand_use : user->operand_uses()) {
                if (operand_use == use) {
                    found = true;
                    break;
                }
            }
            require(module, found, "Instruction use-list is not mirrored by the user operand list.");
        }
    }
}

size_t count_tag_in_block(const xir::BasicBlock *block, xir::DerivedInstructionTag tag) {
    auto count = size_t{0u};
    for (auto inst : block->instructions()) {
        if (inst->derived_instruction_tag() == tag) {
            count++;
        }
    }
    return count;
}

void require_branch_target(const xir::Module *module,
                           const xir::BasicBlock *block,
                           const xir::BasicBlock *expected_target,
                           luisa::string_view message) {
    require(module, block != nullptr, "Expected a non-null block.");
    require(module, block->is_terminated(), "Expected the inspected block to be terminated.");
    require(module, block->terminator()->isa<xir::BranchInst>(), message);
    auto branch = static_cast<const xir::BranchInst *>(block->terminator());
    require(module, branch->target_block() == expected_target, message);
}

void require_conditional_branch_targets_include(const xir::Module *module,
                                                const xir::BasicBlock *block,
                                                const xir::BasicBlock *expected_target,
                                                luisa::string_view message) {
    require(module, block != nullptr, "Expected a non-null block.");
    require(module, block->is_terminated(), "Expected the inspected block to be terminated.");
    require(module, block->terminator()->isa<xir::ConditionalBranchInst>(), message);
    auto branch = static_cast<const xir::ConditionalBranchInst *>(block->terminator());
    require(module,
            branch->true_block() == expected_target || branch->false_block() == expected_target,
            message);
}

[[nodiscard]] const xir::BasicBlock *find_other_conditional_target(const xir::BasicBlock *block,
                                                                   const xir::BasicBlock *known_target) {
    if (block == nullptr || !block->is_terminated() || !block->terminator()->isa<xir::ConditionalBranchInst>()) {
        return nullptr;
    }
    auto branch = static_cast<const xir::ConditionalBranchInst *>(block->terminator());
    if (branch->true_block() == known_target) {
        return branch->false_block();
    }
    if (branch->false_block() == known_target) {
        return branch->true_block();
    }
    return nullptr;
}

[[nodiscard]] xir::BasicBlock *find_enclosing_if_merge_block(const ReachableIR &reachable,
                                                             const xir::BasicBlock *branch_block) {
    for (auto inst : reachable.instructions) {
        if (!inst->isa<xir::IfInst>()) {
            continue;
        }
        auto if_inst = static_cast<xir::IfInst *>(inst);
        if (if_inst->true_block() == branch_block || if_inst->false_block() == branch_block) {
            return if_inst->merge_block();
        }
    }
    return nullptr;
}

[[nodiscard]] xir::ReturnInst *find_final_return_instruction(xir::FunctionDefinition *definition) {
    if (definition == nullptr) {
        return nullptr;
    }
    for (auto block = definition->body_block(); block != nullptr;) {
        auto terminator = block->terminator();
        if (terminator->isa<xir::ReturnInst>()) {
            return static_cast<xir::ReturnInst *>(terminator);
        }
        auto merge = terminator->control_flow_merge();
        if (merge == nullptr || merge->merge_block() == nullptr) {
            return nullptr;
        }
        block = merge->merge_block();
    }
    return nullptr;
}

void verify_common_invariants(xir::Module *module,
                              const xir::Canonicalize_Control_Flow_Info &info,
                              size_t expected_lowered_loops,
                              size_t expected_simple_loops) {
    auto function = module->function_list().front();
    auto definition = function->definition();
    require(module, definition != nullptr, "The translated module must contain a function definition.");

    auto reachable = collect_reachable_ir(definition);
    require(module, info.lowered_loop_count == expected_lowered_loops, "Unexpected lowered loop count.");
    require(module, info.skipped_loop_count == 0u, "The pass should not skip well-formed AST2XIR for-loops.");
    require(module, count_tag(reachable, xir::DerivedInstructionTag::LOOP) == 0u, "Reachable LoopInst should be gone after canonicalization.");
    require(module, count_tag(reachable, xir::DerivedInstructionTag::SIMPLE_LOOP) == expected_simple_loops, "Unexpected reachable SimpleLoopInst count.");
    require(module, count_tag(reachable, xir::DerivedInstructionTag::BREAK) == 0u, "Reachable BreakInst should be gone after canonicalization.");
    require(module, count_tag(reachable, xir::DerivedInstructionTag::CONTINUE) == 0u, "Reachable ContinueInst should be gone after canonicalization.");
    require(module, count_tag(reachable, xir::DerivedInstructionTag::RETURN) == 1u, "Canonicalized control flow should retain exactly one reachable ReturnInst.");

    for (auto block : reachable.blocks) {
        require(module, block->is_terminated(), "Every reachable block must remain terminated.");
    }
    verify_operand_use_consistency(module, reachable);
}

void test_basic_for_loop() {
    Kernel1D kernel = []() noexcept {
        auto idx = dispatch_id().x;
        Int x = cast<int>(idx);
        $for (i, 0, 10) {
            x += i;
        };
        x += cast<int>(idx);
    };

    auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
    auto definition = module->function_list().front()->definition();
    auto before = collect_reachable_ir(definition);
    require(module.get(), count_tag(before, xir::DerivedInstructionTag::LOOP) == 1u, "Expected one LoopInst before canonicalization.");

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    verify_common_invariants(module.get(), info, 1u, 1u);
}

void test_prepare_instructions_are_moved() {
    Kernel1D kernel = []() noexcept {
        auto idx = dispatch_id().x;
        Int start = cast<int>(idx) + 1;
        Int stop = start + 6;
        Int sum = start;
        $for (i, start - 1, stop + 2) {
            sum += i;
        };
        sum += stop;
    };

    auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
    auto definition = module->function_list().front()->definition();
    auto before = collect_reachable_ir(definition);
    auto original_loop = static_cast<xir::LoopInst *>(collect_by_tag(before, xir::DerivedInstructionTag::LOOP).front());
    auto original_body = original_loop->body_block();
    auto original_update = original_loop->update_block();
    auto original_merge = original_loop->merge_block();
    luisa::vector<xir::Instruction *> moved_prepare_insts;
    for (auto inst : original_loop->prepare_block()->instructions()) {
        if (!inst->is_terminator()) {
            moved_prepare_insts.emplace_back(inst);
        }
    }

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    verify_common_invariants(module.get(), info, 1u, 1u);

    auto after = collect_reachable_ir(definition);
    auto simple_loop = static_cast<xir::SimpleLoopInst *>(collect_by_tag(after, xir::DerivedInstructionTag::SIMPLE_LOOP).front());
    auto simple_body = simple_loop->body_block();
    require(module.get(), simple_body->terminator()->isa<xir::IfInst>(), "The simple loop body should end with the canonical guard if.");
    auto guard = static_cast<xir::IfInst *>(simple_body->terminator());

    require(module.get(), count_non_terminator_instructions(simple_body) >= moved_prepare_insts.size(), "The canonical body should still contain the moved prepare instructions.");
    for (auto inst : moved_prepare_insts) {
        require(module.get(), inst->parent_block() == simple_body, "A prepare instruction was not moved into the simple loop body.");
        require(module.get(), block_contains_instruction(simple_body, inst), "A moved prepare instruction is missing from the simple loop body.");
    }
    require(module.get(), guard->true_block() == original_body, "The loop body block should be reused as the guard true branch.");
    require(module.get(), guard->merge_block() == original_update, "The update block should be reused as the guard merge block.");
    require(module.get(), simple_loop->merge_block() == original_merge, "The original merge block should become the simple loop merge block.");
}

void test_continue_targets_update_path() {
    Kernel1D kernel = []() noexcept {
        Int x = 0;
        $for (i, 0, 8) {
            $if ((i & 1) == 0) {
                $continue;
            };
            x += i;
        };
        x += 1;
    };

    auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
    auto definition = module->function_list().front()->definition();
    auto before = collect_reachable_ir(definition);
    auto continue_inst = static_cast<xir::ContinueInst *>(collect_by_tag(before, xir::DerivedInstructionTag::CONTINUE).front());
    auto continue_block = continue_inst->parent_block();
    auto continue_target = continue_inst->target_block();
    auto continue_merge_before = find_enclosing_if_merge_block(before, continue_block);
    require(module.get(), continue_merge_before != nullptr, "Expected to find the enclosing if-merge for the continue block.");

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    verify_common_invariants(module.get(), info, 1u, 1u);

    auto after = collect_reachable_ir(definition);
    auto simple_loop = static_cast<xir::SimpleLoopInst *>(collect_by_tag(after, xir::DerivedInstructionTag::SIMPLE_LOOP).front());
    auto guard = static_cast<xir::IfInst *>(simple_loop->body_block()->terminator());
    require_branch_target(module.get(), continue_block, continue_merge_before, "Continue should branch to the enclosing merge before the guarded suffix.");
    require_conditional_branch_targets_include(module.get(), continue_merge_before, continue_target, "The guarded merge should still be able to skip directly to the update path.");
    require(module.get(), continue_target == guard->merge_block(), "Continue should still ultimately target the update path after canonicalization.");
}

void test_break_targets_simple_loop_merge() {
    Kernel1D kernel = []() noexcept {
        Int x = 0;
        $for (i, 0, 8) {
            $if (i > 4) {
                $break;
            };
            x += i;
        };
        x += 1;
    };

    auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
    auto definition = module->function_list().front()->definition();
    auto before = collect_reachable_ir(definition);
    auto break_inst = static_cast<xir::BreakInst *>(collect_by_tag(before, xir::DerivedInstructionTag::BREAK).front());
    auto break_block = break_inst->parent_block();
    auto break_target = break_inst->target_block();
    auto break_merge_before = find_enclosing_if_merge_block(before, break_block);
    require(module.get(), break_merge_before != nullptr, "Expected to find the enclosing if-merge for the break block.");

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    verify_common_invariants(module.get(), info, 1u, 1u);

    auto after = collect_reachable_ir(definition);
    auto simple_loop = static_cast<xir::SimpleLoopInst *>(collect_by_tag(after, xir::DerivedInstructionTag::SIMPLE_LOOP).front());
    auto guard = static_cast<xir::IfInst *>(simple_loop->body_block()->terminator());
    require(module.get(), break_block != guard->false_block(), "The original break should remain distinct from the guard false-path break.");
    require_branch_target(module.get(), break_block, break_merge_before, "Break should branch to the enclosing merge before the guarded loop exit.");
    require_conditional_branch_targets_include(module.get(), break_merge_before, guard->merge_block(), "The guarded merge should still be able to skip the remaining body and reach the update path.");
    require_conditional_branch_targets_include(module.get(), guard->merge_block(), break_target, "The guarded update path should still be able to exit to the simple loop merge.");
    require(module.get(), break_target == simple_loop->merge_block(), "Break should still target the simple loop merge block.");
}

void test_update_backedge_retargeted() {
    Kernel1D kernel = []() noexcept {
        Int x = 0;
        $for (i, 0, 5) {
            x += i;
        };
        x += 2;
    };

    auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
    auto definition = module->function_list().front()->definition();
    auto before = collect_reachable_ir(definition);
    auto original_loop = static_cast<xir::LoopInst *>(collect_by_tag(before, xir::DerivedInstructionTag::LOOP).front());
    auto original_update = original_loop->update_block();

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    verify_common_invariants(module.get(), info, 1u, 1u);

    auto after = collect_reachable_ir(definition);
    auto simple_loop = static_cast<xir::SimpleLoopInst *>(collect_by_tag(after, xir::DerivedInstructionTag::SIMPLE_LOOP).front());
    require_conditional_branch_targets_include(module.get(), original_update, simple_loop->merge_block(), "The canonicalized update block should guard the loop exit.");
    auto guarded_backedge = find_other_conditional_target(original_update, simple_loop->merge_block());
    require(module.get(), guarded_backedge != nullptr, "Expected the guarded update block to retain a backedge path.");
    require_branch_target(module.get(), guarded_backedge, simple_loop->body_block(), "The guarded update backedge must point to the new simple loop body.");
}

void test_nested_for_loops() {
    Kernel1D kernel = []() noexcept {
        Int x = 0;
        $for (i, 0, 3) {
            $for (j, 0, 4) {
                x += i + j;
            };
        };
        x += 1;
    };

    auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
    auto definition = module->function_list().front()->definition();
    auto before = collect_reachable_ir(definition);
    auto loops_before = collect_by_tag(before, xir::DerivedInstructionTag::LOOP);
    require(module.get(), loops_before.size() == 2u, "Expected two LoopInst before canonicalization.");
    auto outer_update = static_cast<xir::LoopInst *>(loops_before[0])->update_block();
    auto inner_update = static_cast<xir::LoopInst *>(loops_before[1])->update_block();

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    verify_common_invariants(module.get(), info, 2u, 2u);

    auto after = collect_reachable_ir(definition);
    auto simple_loops = collect_by_tag(after, xir::DerivedInstructionTag::SIMPLE_LOOP);
    auto matched_outer = false;
    auto matched_inner = false;
    for (auto inst : simple_loops) {
        auto simple_loop = static_cast<xir::SimpleLoopInst *>(inst);
        auto guard = static_cast<xir::IfInst *>(simple_loop->body_block()->terminator());
        if (guard->merge_block() == outer_update) {
            matched_outer = true;
            require_conditional_branch_targets_include(module.get(), outer_update, simple_loop->merge_block(), "Outer loop update should be guarded before exiting the loop.");
            auto guarded_backedge = find_other_conditional_target(outer_update, simple_loop->merge_block());
            require(module.get(), guarded_backedge != nullptr, "Expected the outer guarded update block to retain a backedge path.");
            require_branch_target(module.get(), guarded_backedge, simple_loop->body_block(), "Outer loop update must still target the outer simple body.");
        }
        if (guard->merge_block() == inner_update) {
            matched_inner = true;
            require_conditional_branch_targets_include(module.get(), inner_update, simple_loop->merge_block(), "Inner loop update should be guarded before exiting the loop.");
            auto guarded_backedge = find_other_conditional_target(inner_update, simple_loop->merge_block());
            require(module.get(), guarded_backedge != nullptr, "Expected the inner guarded update block to retain a backedge path.");
            require_branch_target(module.get(), guarded_backedge, simple_loop->body_block(), "Inner loop update must still target the inner simple body.");
        }
    }
    require(module.get(), matched_outer, "Failed to find the lowered outer loop.");
    require(module.get(), matched_inner, "Failed to find the lowered inner loop.");
}

void test_for_with_inner_if() {
    Kernel1D kernel = []() noexcept {
        Int x = 0;
        $for (i, 0, 6) {
            $if (i > 2) {
                x += i;
            } $else {
                x -= i;
            };
        };
        x += 1;
    };

    auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
    auto definition = module->function_list().front()->definition();
    auto before = collect_reachable_ir(definition);
    require(module.get(), count_tag(before, xir::DerivedInstructionTag::IF) >= 1u, "Expected at least one body IfInst before canonicalization.");

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    verify_common_invariants(module.get(), info, 1u, 1u);

    auto after = collect_reachable_ir(definition);
    require(module.get(), count_tag(after, xir::DerivedInstructionTag::IF) >= 2u, "Expected both the loop guard and the original body if after canonicalization.");
}

void test_void_early_return_is_lowered_to_single_return() {
    Kernel1D kernel = []() noexcept {
        Int x = 0;
        $if (dispatch_id().x > 2u) {
            $return();
        };
        x += 1;
    };

    auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
    auto definition = module->function_list().front()->definition();
    auto before = collect_reachable_ir(definition);
    require(module.get(), count_tag(before, xir::DerivedInstructionTag::RETURN) == 2u, "Expected one early return and one final return before canonicalization.");

    auto final_return_before = find_final_return_instruction(definition);
    require(module.get(), final_return_before != nullptr, "Expected to find the final return before canonicalization.");
    auto final_return_block_before = final_return_before->parent_block();
    auto early_return_block_before = static_cast<xir::BasicBlock *>(nullptr);
    for (auto inst : collect_by_tag(before, xir::DerivedInstructionTag::RETURN)) {
        if (inst != final_return_before) {
            early_return_block_before = inst->parent_block();
            break;
        }
    }
    require(module.get(), early_return_block_before != nullptr, "Expected to find the early return block before canonicalization.");

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    verify_common_invariants(module.get(), info, 0u, 0u);

    auto after = collect_reachable_ir(definition);
    auto common_return = static_cast<xir::ReturnInst *>(collect_by_tag(after, xir::DerivedInstructionTag::RETURN).front());
    auto common_return_block = common_return->parent_block();
    require(module.get(), early_return_block_before->terminator()->isa<xir::BranchInst>(), "The early return should be lowered to a branch.");
    require(module.get(), static_cast<xir::BranchInst *>(early_return_block_before->terminator())->target_block() != common_return_block, "The early return should first branch to its local follow block.");
    require_conditional_branch_targets_include(module.get(), final_return_block_before, common_return_block, "The former final return block should become a guarded continuation that can skip to the common return block.");
}

void test_nonvoid_early_return_uses_common_return_slot() {
    Callable<int()> callable = []() noexcept {
        Int sum = 0;
        $for (i, 0, 5) {
            $if (i > 2) {
                $return(sum + i);
            };
            sum += i;
        };
        return sum + 3;
    };

    auto module = xir::ast_to_xir_translate(callable.function(), {});
    auto definition = module->function_list().front()->definition();
    auto before = collect_reachable_ir(definition);
    require(module.get(), count_tag(before, xir::DerivedInstructionTag::RETURN) == 2u, "Expected one early return and one final return before canonicalization.");

    auto final_return_before = find_final_return_instruction(definition);
    require(module.get(), final_return_before != nullptr, "Expected to find the final return before canonicalization.");
    auto final_return_block_before = final_return_before->parent_block();
    auto early_return_block_before = static_cast<xir::BasicBlock *>(nullptr);
    for (auto inst : collect_by_tag(before, xir::DerivedInstructionTag::RETURN)) {
        if (inst != final_return_before) {
            early_return_block_before = inst->parent_block();
            break;
        }
    }
    require(module.get(), early_return_block_before != nullptr, "Expected to find the early return block before canonicalization.");

    auto info = xir::Canoinicalize_Control_Flow_pass_run_on_Module(module.get());
    verify_common_invariants(module.get(), info, 1u, 1u);

    auto after = collect_reachable_ir(definition);
    auto common_return = static_cast<xir::ReturnInst *>(collect_by_tag(after, xir::DerivedInstructionTag::RETURN).front());
    auto common_return_block = common_return->parent_block();
    require(module.get(), early_return_block_before->terminator()->isa<xir::BranchInst>(), "The early return should be lowered to a branch.");
    require(module.get(), static_cast<xir::BranchInst *>(early_return_block_before->terminator())->target_block() != common_return_block, "The early return should first branch to its local follow block.");
    require_conditional_branch_targets_include(module.get(), final_return_block_before, common_return_block, "The original final return block should become a guarded continuation that can skip to the common return block.");
    require(module.get(), count_tag_in_block(common_return_block, xir::DerivedInstructionTag::LOAD) == 1u, "The common non-void return block should load the shared return slot exactly once.");
    require(module.get(), count_tag_in_block(early_return_block_before, xir::DerivedInstructionTag::STORE) >= 1u, "The lowered early return path should store into the shared return slot.");
    require(module.get(), count_tag(after, xir::DerivedInstructionTag::STORE) >= 2u, "The lowered non-void return should still store into the shared return slot along both early and final paths.");
}

}// namespace

int main() {
    test_basic_for_loop();
    test_prepare_instructions_are_moved();
    test_continue_targets_update_path();
    test_break_targets_simple_loop_merge();
    test_update_backedge_retargeted();
    test_nested_for_loops();
    test_for_with_inner_if();
    test_void_early_return_is_lowered_to_single_return();
    test_nonvoid_early_return_uses_common_return_slot();
    LUISA_INFO("All control_flow tests passed.");
    return 0;
}
