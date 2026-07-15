#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/passes/fix_self_referential.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static void fix_self_referential_instructions_on_function(Function *function, FixSelfReferentialInfo &info) noexcept {
    if (auto def = function->definition(); def != nullptr) {
        def->traverse_instructions([&](Instruction *inst) noexcept {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == inst) {
                    // Found a self-referential operand
                    info.fixed_count++;
                    
                    // For INSERT instructions, try to find the alloca they're stored to
                    // and replace the self-reference with a load from that alloca.
                    // This restores the intended semantics when a load was incorrectly
                    // eliminated by an optimization pass.
                    if (auto arith = inst->isa<ArithmeticInst>() ? static_cast<ArithmeticInst *>(inst) : nullptr;
                        arith != nullptr && arith->op() == ArithmeticOp::INSERT) {
                        AllocaInst *target_alloca = nullptr;
                        // First try use_list (fast path)
                        for (auto &&use : inst->use_list()) {
                            if (auto user = use->user(); user != nullptr && user->isa<StoreInst>()) {
                                auto store = static_cast<StoreInst *>(user);
                                if (auto base = trace_pointer_base_local_alloca_inst(store->variable())) {
                                    target_alloca = base;
                                    break;
                                }
                            }
                        }
                        // Fallback: search all instructions in the function
                        if (target_alloca == nullptr) {
                            def->traverse_instructions([&](Instruction *other_inst) noexcept {
                                if (target_alloca != nullptr) { return; }
                                if (other_inst->isa<StoreInst>()) {
                                    auto store = static_cast<StoreInst *>(other_inst);
                                    if (store->value() == inst) {
                                        if (auto base = trace_pointer_base_local_alloca_inst(store->variable())) {
                                            target_alloca = base;
                                        }
                                    }
                                }
                            });
                        }
                        if (target_alloca != nullptr) {
                            auto replacement = luisa::make_managed<LoadInst>(inst->parent_block(), inst->type(), target_alloca);
                            auto load = static_cast<LoadInst *>(inst->insert_before_self(std::move(replacement)));
                            inst->set_operand(i, load);
                            continue;
                        }
                    }
                    
                    // Fallback: replace with undef
                    if (auto m = function->parent_module()) {
                        auto undef = m->create_undefined(inst->type());
                        inst->set_operand(i, undef);
                    }
                }
            }
        });
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
    }
    return info;
}

}// namespace luisa::compute::xir
