#include <luisa/xir/passes/licm.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/alloca.h>

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
            // Allow hoisting of:
            //   - instructions in the prepare block (runs once before loop)
            //   - pure instructions (no memory effects)
            //   - read-only global memory accesses (buffer reads) with invariant operands
            bool can_hoist = bb == prepare ||
                             mem.is_pure() ||
                             (mem.scope == MemoryScope::GLOBAL &&
                              mem.reads_memory() &&
                              !mem.writes_memory() &&
                              !mem.is_volatile);
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

    luisa::vector<Instruction *> to_hoist;
    for (auto *bb : loop_body_blocks) {
        for (auto *inst : bb->instructions()) {
            if (invariant.contains(inst) && !inst->is_terminator()) {
                to_hoist.push_back(inst);
            }
        }
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
