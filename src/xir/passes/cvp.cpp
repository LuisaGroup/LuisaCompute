#include <luisa/xir/passes/cvp.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/constant.h>

namespace luisa::compute::xir {

namespace detail {

static void cvp_pass_on_function(FunctionDefinition *def, CVPInfo &info) noexcept {
    if (def->body_block() == nullptr) { return; }

    auto dom_tree = compute_dom_tree(def);

    // Collect all IfInst terminators.
    luisa::vector<IfInst *> if_insts;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<IfInst>()) {
            if_insts.push_back(static_cast<IfInst *>(inst));
        }
    });

    for (auto if_inst : if_insts) {
        auto cond = if_inst->condition();
        if (cond == nullptr || !cond->isa<ArithmeticInst>()) { continue; }

        auto arith = static_cast<ArithmeticInst *>(cond);
        auto op = arith->op();

        if (op != ArithmeticOp::BINARY_EQUAL &&
            op != ArithmeticOp::BINARY_NOT_EQUAL) {
            continue;
        }
        if (arith->operand_count() < 2) { continue; }

        auto op0 = arith->operand(0);
        auto op1 = arith->operand(1);

        // Find the variable operand and the constant operand.
        Value *var = nullptr;
        Constant *constant_val = nullptr;

        if (op1->isa<Constant>()) {
            var = op0;
            constant_val = static_cast<Constant *>(op1);
        } else if (op0->isa<Constant>()) {
            var = op1;
            constant_val = static_cast<Constant *>(op0);
        }

        if (var == nullptr || constant_val == nullptr) { continue; }
        if (!var->isa<Instruction>()) { continue; }

        // Determine the target block where var == constant_val.
        BasicBlock *target_block = nullptr;
        if (op == ArithmeticOp::BINARY_EQUAL) {
            target_block = if_inst->true_block();
        } else {
            // BINARY_NOT_EQUAL: in the false block, var == constant_val
            target_block = if_inst->false_block();
        }

        if (target_block == nullptr) { continue; }

        // Collect uses of var that are in blocks dominated by target_block.
        luisa::vector<Use *> uses_to_replace;
        for (auto &&use : var->use_list()) {
            auto user = use->user();
            if (user == nullptr || !user->isa<Instruction>()) { continue; }
            auto user_block = static_cast<Instruction *>(user)->parent_block();
            if (dom_tree.dominates(target_block, user_block)) {
                // Don't replace the condition itself — it's in the IfInst's
                // parent block which is not dominated by target_block anyway.
                uses_to_replace.push_back(use);
            }
        }

        // Perform the replacements.
        for (auto use : uses_to_replace) {
            auto owned_use = use->remove_self();
            owned_use->set_value(constant_val);
            constant_val->use_list().push_front(std::move(owned_use));
            info.replaced_inst_count++;
        }
    }
}

}// namespace detail

CVPInfo cvp_pass_run_on_function(FunctionDefinition *def) noexcept {
    CVPInfo info;
    detail::cvp_pass_on_function(def, info);
    return info;
}

CVPInfo cvp_pass_run_on_module(Module *module, PassReport *report) noexcept {
    CVPInfo info;
    for (auto f : module->function_list()) {
        detail::cvp_pass_on_function(f->definition(), info);
    }
    if (report != nullptr) {
        report->set("replaced_inst", info.replaced_inst_count);
    }
    return info;
}

}// namespace luisa::compute::xir
