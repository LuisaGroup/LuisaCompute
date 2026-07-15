#include <luisa/xir/passes/indvar_simplify.h>
#include <luisa/xir/passes/scalar_evolution.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/module.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/gep.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace {

[[nodiscard]] bool is_iv_increment(Instruction *inst, PhiInst *phi) noexcept {
    if (!inst->isa<ArithmeticInst>()) { return false; }
    auto *arith = static_cast<ArithmeticInst *>(inst);
    if (arith->op() != ArithmeticOp::BINARY_ADD) { return false; }
    for (size_t i = 0; i < arith->operand_count(); ++i) {
        if (arith->operand(i) == phi) { return true; }
    }
    return false;
}

struct IVInfo {
    PhiInst *phi;
    Instruction *increment;
    Value *stride_val;
    bool has_constant_stride;
    int64_t stride_const;
};

[[nodiscard]] bool try_build_iv_info(PhiInst *phi, LoopInst *loop, IVInfo &out) noexcept {
    auto *update = loop->update_block();
    if (!update) { return false; }

    Value *start_val = nullptr;
    Value *recur_val = nullptr;
    for (size_t i = 0; i < phi->incoming_count(); ++i) {
        auto inc = phi->incoming(i);
        if (inc.block == update) {
            recur_val = inc.value;
        } else {
            start_val = inc.value;
        }
    }
    if (!start_val || !recur_val) { return false; }
    if (!recur_val->isa<Instruction>()) { return false; }
    auto *recur_inst = static_cast<Instruction *>(recur_val);
    if (!is_iv_increment(recur_inst, phi)) { return false; }

    auto *arith = static_cast<ArithmeticInst *>(recur_inst);
    Value *stride_val = nullptr;
    for (size_t i = 0; i < arith->operand_count(); ++i) {
        if (arith->operand(i) != phi) { stride_val = arith->operand(i); }
    }
    if (!stride_val) { return false; }

    bool has_const_stride = false;
    int64_t stride_const = 0;
    if (stride_val->isa<Constant>()) {
        auto *c = static_cast<Constant *>(stride_val);
        auto *ty = c->type();
        if (ty->is_int32()) {
            has_const_stride = true;
            stride_const = static_cast<int64_t>(c->as<int32_t>());
        } else if (ty->is_uint32()) {
            has_const_stride = true;
            stride_const = static_cast<int64_t>(c->as<uint32_t>());
        } else if (ty->is_int64()) {
            has_const_stride = true;
            stride_const = c->as<int64_t>();
        }
    }

    out.phi = phi;
    out.increment = recur_inst;
    out.stride_val = stride_val;
    out.has_constant_stride = has_const_stride;
    out.stride_const = stride_const;
    return true;
}

void remove_dead_iv(IVInfo &iv, IndVarSimplifyInfo &info) noexcept {
    for (auto *use : iv.phi->use_list()) {
        auto *user = use->user();
        if (user == iv.increment) { continue; }
        if (user == iv.phi) { continue; }
        return;
    }
    // The increment is also a value. If anything other than the recurrence
    // phi observes it, deleting the apparent phi/increment cycle would leave
    // that user dangling.
    for (auto *use : iv.increment->use_list()) {
        if (use->user() != iv.phi) { return; }
    }
    // Break the recurrence through the public phi API before deleting either
    // instruction so both use lists stay valid throughout the mutation.
    for (size_t i = iv.phi->incoming_count(); i-- > 0u;) {
        if (iv.phi->incoming(i).value == iv.increment) {
            iv.phi->remove_incoming(i);
        }
    }
    iv.increment->remove_self();
    iv.phi->remove_self();
    info.removed_dead_iv_count++;
}

void process_loop(LoopInst *loop, IndVarSimplifyInfo &info) noexcept {
    auto *prepare = loop->prepare_block();
    auto *update = loop->update_block();
    if (!prepare || !update) { return; }

    auto *prep_term = prepare->terminator();
    if (!prep_term || !prep_term->isa<ConditionalBranchInst>()) { return; }

    luisa::vector<IVInfo> ivs;
    for (auto *inst : prepare->instructions()) {
        if (inst->is_terminator()) { break; }
        if (!inst->isa<PhiInst>()) { continue; }
        auto *phi = static_cast<PhiInst *>(inst);
        IVInfo iv;
        if (try_build_iv_info(phi, loop, iv)) {
            ivs.push_back(iv);
        }
    }

    for (auto &iv : ivs) {
        remove_dead_iv(iv, info);
    }
}

void indvar_simplify_on_function_def(FunctionDefinition *def, IndVarSimplifyInfo &info) noexcept {
    static_cast<void>(scev_pass_run_on_function(def));

    luisa::vector<LoopInst *> loops;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<LoopInst>()) {
            loops.push_back(static_cast<LoopInst *>(inst));
        }
    });

    for (auto *loop : loops) {
        process_loop(loop, info);
    }
}

}// namespace

IndVarSimplifyInfo indvar_simplify_pass_run_on_function(FunctionDefinition *def) noexcept {
    IndVarSimplifyInfo info;
    indvar_simplify_on_function_def(def, info);
    return info;
}

IndVarSimplifyInfo indvar_simplify_pass_run_on_module(Module *module, PassReport *report) noexcept {
    IndVarSimplifyInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            indvar_simplify_on_function_def(def, info);
        }
    }
    if (report != nullptr) {
        report->set("simplified_iv", info.simplified_iv_count);
        report->set("removed_dead_iv", info.removed_dead_iv_count);
    }
    return info;
}

}// namespace luisa::compute::xir
