#include <luisa/core/stl/algorithm.h>
#include <luisa/xir/passes/licm.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/resource.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace {

[[nodiscard]] bool is_safe_to_speculate(Instruction *inst) noexcept {
    // A bitwise cast is a total, representation-preserving operation. Numeric
    // casts are not: in particular, an out-of-range float-to-integer cast can
    // be undefined in source/LLVM backends. Do not make such a cast execute on
    // a zero-trip loop path.
    if (inst->isa<CastInst>()) {
        return static_cast<CastInst *>(inst)->op() ==
               CastOp::BITWISE_CAST;
    }
    // XIR GEP permits dynamic aggregate indices. Materializing an out-of-range
    // address can be undefined even if the loop body (and the eventual
    // dereference) never executes. A future version may hoist a GEP after
    // proving every index in range.
    if (inst->isa<GEPInst>()) { return false; }
    if (inst->isa<ResourceQueryInst>()) {
        // Only direct-resource metadata queries are total under the XIR
        // resource-binding contract. Bindless queries can index an invalid
        // slot, and sampling queries can carry derivative/control-flow
        // requirements; neither may be speculated onto a zero-trip path merely
        // because the operation has no memory side effect.
        switch (static_cast<ResourceQueryInst *>(inst)->op()) {
            case ResourceQueryOp::BUFFER_SIZE:
            case ResourceQueryOp::BYTE_BUFFER_SIZE:
            case ResourceQueryOp::TEXTURE2D_SIZE:
            case ResourceQueryOp::TEXTURE3D_SIZE:
            case ResourceQueryOp::BUFFER_DEVICE_ADDRESS:
                return true;
            default: return false;
        }
    }
    // Keep the speculative set explicit. A future side-effect-free
    // instruction is not automatically total on every operand.
    if (!inst->isa<ArithmeticInst>()) { return false; }
    // LICM moves body instructions before the loop condition, so even a loop
    // with zero iterations will evaluate them. Keep operations with undefined
    // operands in place until the IR has explicit poison/trap semantics.
    return is_arithmetic_op_safe_to_speculate(
        static_cast<ArithmeticInst *>(inst)->op());
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

struct PreparedLoop {
    BasicBlock *prepare{nullptr};
    Instruction *prep_term{nullptr};
    luisa::unordered_set<BasicBlock *> body_blocks;
    luisa::vector<BasicBlock *> ordered_body_blocks;
};

void licm_on_loop(const PreparedLoop &loop,
                  const luisa::unordered_set<BasicBlock *> &reachable_blocks,
                  LICMInfo &info) noexcept {
    auto *prepare = loop.prepare;
    auto *prep_term = loop.prep_term;
    auto &loop_body_blocks = loop.body_blocks;
    auto &ordered_body_blocks = loop.ordered_body_blocks;
    if (ordered_body_blocks.empty()) { return; }

    // Instructions in reachable blocks outside `prepare` and `loop_body_blocks`
    // are loop-invariant without materializing an O(|function|) set per loop.
    luisa::unordered_set<Instruction *> local_invariant;
    auto is_operand_invariant = [&](Value *op) noexcept -> bool {
        if (op->isa<Constant>() || op->isa<Argument>()) { return true; }
        if (op->isa<Instruction>()) {
            auto *op_inst = static_cast<Instruction *>(op);
            auto *op_bb = op_inst->parent_block();
            if (op_bb != prepare && !loop_body_blocks.contains(op_bb)) {
                return op_bb != nullptr && reachable_blocks.contains(op_bb);
            }
            return local_invariant.contains(op_inst);
        }
        return false;
    };
    auto all_operands_invariant = [&](Instruction *inst) noexcept -> bool {
        for (size_t i = 0u; i < inst->operand_count(); ++i) {
            if (!is_operand_invariant(inst->operand(i))) { return false; }
        }
        return true;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        auto try_mark = [&](Instruction *inst) {
            if (local_invariant.contains(inst)) { return; }
            if (inst->is_terminator()) { return; }
            if (inst->isa<PhiInst>()) { return; }
            if (inst->isa<AllocaInst>()) { return; }
            if (!all_operands_invariant(inst)) { return; }
            auto *bb = inst->parent_block();
            auto mem = get_memory_info(inst);
            // Moving a memory read from the body into prepare speculates it before
            // the loop condition and also requires alias/clobber analysis. Keep
            // memory operations in place until the pass can prove both properties.
            bool can_hoist = bb == prepare ||
                             (mem.is_pure() && is_safe_to_speculate(inst));
            if (can_hoist) {
                local_invariant.insert(inst);
                changed = true;
            }
        };
        for (auto *inst : prepare->instructions()) { try_mark(inst); }
        for (auto *bb : ordered_body_blocks) {
            for (auto *inst : bb->instructions()) { try_mark(inst); }
        }
    }

    if (local_invariant.empty()) { return; }

    // Collect candidates in the stable function-owned block/instruction order,
    // then topologically order their intra-set dependencies.
    luisa::vector<Instruction *> candidates;
    for (auto *bb : ordered_body_blocks) {
        for (auto *inst : bb->instructions()) {
            if (local_invariant.contains(inst) && !inst->is_terminator()) {
                candidates.emplace_back(inst);
            }
        }
    }
    if (candidates.empty()) { return; }

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
    if (def == nullptr || def->body_block() == nullptr) { return; }

    // LICM only hoists non-terminator instructions and never mutates the CFG,
    // so the reachable block set, stable block order, and loop bodies are
    // invariant across outer fixed-point iterations.
    luisa::unordered_set<BasicBlock *> reachable_blocks;
    luisa::vector<LoopInst *> raw_loops;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        reachable_blocks.emplace(bb);
        for (auto *inst : bb->instructions()) {
            if (inst->isa<LoopInst>()) {
                raw_loops.emplace_back(static_cast<LoopInst *>(inst));
            }
        }
    });
    if (raw_loops.empty()) { return; }

    luisa::unordered_map<BasicBlock *, size_t> block_order;
    size_t order_idx = 0u;
    for (auto *bb : def->basic_blocks()) {
        block_order.emplace(bb, order_idx++);
    }

    luisa::vector<PreparedLoop> prepared_loops;
    prepared_loops.reserve(raw_loops.size());
    for (auto *loop : raw_loops) {
        auto *prepare = loop->prepare_block();
        auto *body = loop->body_block();
        auto *update = loop->update_block();
        auto *merge = loop->merge_block();
        if (!prepare || !body || !update || !merge) { continue; }
        auto *prep_term = prepare->terminator();
        if (!prep_term || !prep_term->isa<ConditionalBranchInst>()) { continue; }
        PreparedLoop prepared;
        prepared.prepare = prepare;
        prepared.prep_term = prep_term;
        collect_loop_body_blocks(body, update, prepare, merge, prepared.body_blocks);
        if (prepared.body_blocks.empty()) { continue; }
        prepared.ordered_body_blocks.reserve(prepared.body_blocks.size());
        for (auto *bb : prepared.body_blocks) {
            if (block_order.contains(bb)) {
                prepared.ordered_body_blocks.emplace_back(bb);
            }
        }
        luisa::sort(prepared.ordered_body_blocks.begin(),
                    prepared.ordered_body_blocks.end(),
                    [&](BasicBlock *lhs, BasicBlock *rhs) noexcept {
                        return block_order.at(lhs) < block_order.at(rhs);
                    });
        if (!prepared.ordered_body_blocks.empty()) {
            prepared_loops.emplace_back(std::move(prepared));
        }
    }
    if (prepared_loops.empty()) { return; }

    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto &loop : prepared_loops) {
            auto before = info.hoisted_count;
            licm_on_loop(loop, reachable_blocks, info);
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
    if (module != nullptr) {
        for (auto f : module->function_list()) {
            if (auto def = f->definition()) {
                detail::licm_pass_on_function_def(def, info);
            }
        }
    }
    if (report != nullptr) { report->set("hoisted", info.hoisted_count); }
    return info;
}

}// namespace luisa::compute::xir
