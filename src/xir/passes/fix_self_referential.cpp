#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/fix_self_referential.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace detail {

static void fix_self_referential_instructions_on_function(Function *function, FixSelfReferentialInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr || def->body_block() == nullptr) { return; }
    auto dom_tree = compute_dom_tree(function);
    struct PendingFix {
        Instruction *instruction;
        Value *target;
        luisa::vector<size_t> operand_indices;
    };
    luisa::vector<PendingFix> pending;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        luisa::vector<size_t> self_operands;
        for (size_t i = 0; i < inst->operand_count(); ++i) {
            if (inst->operand(i) == inst) { self_operands.emplace_back(i); }
        }
        if (self_operands.empty()) { return; }
        if (!inst->isa<ArithmeticInst>() ||
            static_cast<ArithmeticInst *>(inst)->op() != ArithmeticOp::INSERT) {
            info.unresolved_count += self_operands.size();
            return;
        }
        Value *target = nullptr;
        auto ambiguous = false;
        for (auto &&use : inst->use_list()) {
            auto *user = use->user();
            if (user == nullptr || !user->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(user);
            if (store->value() != inst) { continue; }
            auto *candidate = store->variable();
            if (candidate == nullptr || !candidate->is_lvalue() ||
                candidate->type() != inst->type()) {
                ambiguous = true;
                break;
            }
            if (target == nullptr) {
                target = candidate;
            } else if (target != candidate) {
                ambiguous = true;
                break;
            }
        }
        if (target != nullptr && target->isa<Instruction>()) {
            auto *target_inst = static_cast<Instruction *>(target);
            auto *target_block = target_inst->parent_block();
            auto *inst_block = inst->parent_block();
            if (target_block == inst_block) {
                auto seen_target = false;
                for (auto *ordered : inst_block->instructions()) {
                    if (ordered == target_inst) { seen_target = true; }
                    if (ordered == inst) { break; }
                }
                ambiguous |= !seen_target;
            } else {
                ambiguous |= !dom_tree.contains(target_block) ||
                             !dom_tree.contains(inst_block) ||
                             !dom_tree.dominates(target_block, inst_block);
            }
        }
        if (target == nullptr || ambiguous) {
            info.unresolved_count += self_operands.size();
            return;
        }
        pending.emplace_back(PendingFix{inst, target, std::move(self_operands)});
    });
    for (auto &&fix : pending) {
        auto replacement = luisa::make_managed<LoadInst>(
            fix.instruction->parent_block(), fix.instruction->type(), fix.target);
        auto *load = static_cast<LoadInst *>(
            fix.instruction->insert_before_self(std::move(replacement)));
        for (auto operand_index : fix.operand_indices) {
            fix.instruction->set_operand(operand_index, load);
            info.fixed_count++;
        }
    }
}

}// namespace detail

FixSelfReferentialInfo fix_self_referential_pass_run_on_function(Function *function) noexcept {
    FixSelfReferentialInfo info;
    detail::fix_self_referential_instructions_on_function(function, info);
    return info;
}

FixSelfReferentialInfo fix_self_referential_pass_run_on_module(Module *module, PassReport *report) noexcept {
    FixSelfReferentialInfo info;
    for (auto f : module->function_list()) {
        detail::fix_self_referential_instructions_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("fixed_inst", info.fixed_count);
        report->set("unresolved_inst", info.unresolved_count);
    }
    return info;
}

}// namespace luisa::compute::xir
