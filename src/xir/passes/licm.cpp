#include <luisa/xir/passes/licm.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace {

[[nodiscard]] bool is_operand_invariant(Value *op,
                                        const luisa::unordered_set<Instruction *> &invariant) noexcept {
    if (op->isa<Constant>() || op->isa<Argument>()) { return true; }
    if (op->isa<Instruction>()) { return invariant.contains(static_cast<Instruction *>(op)); }
    return false;
}

[[nodiscard]] bool all_operands_invariant(Instruction *inst,
                                          const luisa::unordered_set<Instruction *> &invariant) noexcept {
    for (size_t i = 0; i < inst->operand_count(); ++i) {
        if (!is_operand_invariant(inst->operand(i), invariant)) { return false; }
    }
    return true;
}

[[nodiscard]] bool is_safe_to_speculate(Instruction *inst) noexcept {
    if (!inst->isa<ArithmeticInst>()) { return true; }
    // LICM moves body instructions before the loop condition, so even a loop
    // with zero iterations will evaluate them. Keep operations with undefined
    // operands in place until the IR has explicit poison/trap semantics.
    switch (static_cast<ArithmeticInst *>(inst)->op()) {
        case ArithmeticOp::BINARY_DIV:
        case ArithmeticOp::BINARY_MOD:
        case ArithmeticOp::BINARY_SHIFT_LEFT:
        case ArithmeticOp::BINARY_SHIFT_RIGHT:
            return false;
        default: return true;
    }
}

void collect_loop_body_blocks(BasicBlock *body, BasicBlock *update,
                              BasicBlock *prepare, BasicBlock *merge,
                              luisa::unordered_set<BasicBlock *> &out) noexcept {
    luisa::vector<BasicBlock *> worklist;
    auto enqueue = [&](BasicBlock *bb) {
        if (bb && bb != prepare && bb != merge && out.insert(bb).second) {
            worklist.push_back(bb);
        }
    };
    if (body) { enqueue(body); }
    if (update) { enqueue(update); }
    while (!worklist.empty()) {
        auto *bb = worklist.back();
        worklist.pop_back();
        bb->traverse_successors(false, [&](BasicBlock *succ) { enqueue(succ); });
    }
}

void licm_on_loop(LoopInst *loop, LICMInfo &info, DomTree &dom_tree) noexcept {
    auto *prepare = loop->prepare_block();
    auto *body = loop->body_block();
    auto *update = loop->update_block();
    auto *merge = loop->merge_block();
    if (!prepare || !body || !update || !merge) { return; }

    auto *prep_term = prepare->terminator();
    if (!prep_term || !prep_term->isa<ConditionalBranchInst>()) { return; }

    luisa::unordered_set<BasicBlock *> loop_body_blocks;
    collect_loop_body_blocks(body, update, prepare, merge, loop_body_blocks);
    if (loop_body_blocks.empty()) { return; }

    luisa::unordered_set<Instruction *> invariant;
    loop->parent_function()->definition()->traverse_instructions([&](Instruction *inst) {
        auto *bb = inst->parent_block();
        if (loop_body_blocks.contains(bb)) { return; }
        if (bb == prepare) { return; }
        invariant.insert(inst);
    });

    bool changed = true;
    while (changed) {
        changed = false;
        auto try_mark = [&](Instruction *inst) {
            if (invariant.contains(inst)) { return; }
            if (inst->is_terminator()) { return; }
            if (inst->isa<PhiInst>()) { return; }
            if (inst->isa<AllocaInst>()) { return; }
            if (!all_operands_invariant(inst, invariant)) { return; }
            auto *bb = inst->parent_block();
            auto mem = get_memory_info(inst);
            // Moving a memory read from the body into prepare speculates it before
            // the loop condition and also requires alias/clobber analysis. Keep
            // memory operations in place until the pass can prove both properties.
            bool can_hoist = bb == prepare ||
                             (mem.is_pure() && is_safe_to_speculate(inst));
            if (can_hoist) {
                invariant.insert(inst);
                changed = true;
            }
        };
        for (auto *inst : prepare->instructions()) { try_mark(inst); }
        for (auto *bb : loop_body_blocks) {
            for (auto *inst : bb->instructions()) { try_mark(inst); }
        }
    }

    // Collect candidates in the stable function-owned block/instruction order,
    // then topologically order their intra-set dependencies. Generic CFG
    // traversal can omit temporarily disconnected or raw structured child
    // blocks, while iterating loop_body_blocks directly is nondeterministic.
    luisa::vector<Instruction *> candidates;
    auto *definition = loop->parent_function()->definition();
    for (auto *bb : definition->basic_blocks()) {
        if (!loop_body_blocks.contains(bb)) { continue; }
        for (auto *inst : bb->instructions()) {
            if (invariant.contains(inst) && !inst->is_terminator()) {
                candidates.emplace_back(inst);
            }
        }
    }

    luisa::unordered_set<Instruction *> remaining;
    remaining.reserve(candidates.size());
    for (auto *inst : candidates) { remaining.emplace(inst); }
    luisa::vector<Instruction *> to_hoist;
    to_hoist.reserve(candidates.size());
    while (!remaining.empty()) {
        auto made_progress = false;
        for (auto *inst : candidates) {
            if (!remaining.contains(inst)) { continue; }
            auto ready = true;
            for (size_t i = 0u; i < inst->operand_count(); ++i) {
                auto *operand = inst->operand(i);
                if (operand->isa<Instruction>() &&
                    remaining.contains(static_cast<Instruction *>(operand))) {
                    ready = false;
                    break;
                }
            }
            if (ready) {
                remaining.erase(inst);
                to_hoist.emplace_back(inst);
                made_progress = true;
            }
        }
        // Valid SSA cannot contain a non-phi instruction dependency cycle. Do
        // not partially mutate malformed input if one is nevertheless present.
        if (!made_progress) { return; }
    }

    if (to_hoist.empty()) { return; }

    for (auto *inst : to_hoist) {
        auto m = inst->remove_self();
        prep_term->insert_before_self(std::move(m));
        info.hoisted_count++;
    }
}

}// namespace

namespace detail {

static void licm_pass_on_function_def(FunctionDefinition *def, LICMInfo &info) noexcept {
    bool changed = true;
    while (changed) {
        changed = false;
        luisa::vector<LoopInst *> loops;
        def->traverse_instructions([&](Instruction *inst) {
            if (inst->isa<LoopInst>()) { loops.push_back(static_cast<LoopInst *>(inst)); }
        });
        if (loops.empty()) { return; }

        auto dom_tree = compute_dom_tree(def);

        for (auto *loop : loops) {
            auto before = info.hoisted_count;
            licm_on_loop(loop, info, dom_tree);
            if (info.hoisted_count > before) { changed = true; }
        }
    }
}

}// namespace detail

LICMInfo licm_pass_run_on_function(FunctionDefinition *def) noexcept {
    LICMInfo info;
    detail::licm_pass_on_function_def(def, info);
    return info;
}

LICMInfo licm_pass_run_on_module(Module *module, PassReport *report) noexcept {
    LICMInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) { detail::licm_pass_on_function_def(def, info); }
    }
    if (report != nullptr) { report->set("hoisted", info.hoisted_count); }
    return info;
}

}// namespace luisa::compute::xir
