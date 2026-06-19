#include <luisa/xir/passes/loop_fusion.h>
#include <luisa/xir/passes/scalar_evolution.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/gep.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

struct FusionCandidate {
    LoopInst *loop{nullptr};
    BasicBlock *preheader{nullptr};// prepare_block
    BasicBlock *header{nullptr};   // prepare_block (same in structured loops)
    BasicBlock *body{nullptr};     // body_block
    BasicBlock *update{nullptr};   // update_block
    BasicBlock *exit_block{nullptr};// merge_block
    luisa::vector<Instruction *> mem_reads;
    luisa::vector<Instruction *> mem_writes;
    bool valid{false};

    // Collect memory accesses in the loop's prepare, body and update blocks.
    void collect_memory_accesses() noexcept {
        mem_reads.clear();
        mem_writes.clear();
        auto collect = [&](BasicBlock *bb) noexcept {
            if (bb == nullptr) { return; }
            for (auto *inst : bb->instructions()) {
                auto info = get_memory_info(inst);
                if (info.reads_memory()) { mem_reads.push_back(inst); }
                if (info.writes_memory()) { mem_writes.push_back(inst); }
            }
        };
        collect(preheader);
        collect(body);
        collect(update);
    }

    // Check if the loop is in the expected structured form with a computable trip count.
    [[nodiscard]] bool is_eligible(FunctionDefinition *def) noexcept {
        if (loop == nullptr || preheader == nullptr || body == nullptr ||
            update == nullptr || exit_block == nullptr) {
            return false;
        }
        // Prepare block must terminate with cond_br(body, exit).
        auto *prep_term = preheader->terminator();
        if (prep_term == nullptr || !prep_term->isa<ConditionalBranchInst>()) { return false; }
        auto *cond_br = static_cast<ConditionalBranchInst *>(prep_term);
        if (cond_br->true_block() != body || cond_br->false_block() != exit_block) { return false; }
        // Body block must terminate with an unconditional branch to update.
        auto *body_term = body->terminator();
        if (body_term == nullptr || !body_term->isa<BranchInst>() ||
            static_cast<BranchInst *>(body_term)->target_block() != update) {
            return false;
        }
        // Update block must terminate with an unconditional branch back to prepare.
        auto *update_term = update->terminator();
        if (update_term == nullptr || !update_term->isa<BranchInst>() ||
            static_cast<BranchInst *>(update_term)->target_block() != preheader) {
            return false;
        }
        return has_computable_trip_count();
    }

    [[nodiscard]] bool has_computable_trip_count() const noexcept {
        auto *prep_term = preheader->terminator();
        if (prep_term == nullptr || !prep_term->isa<ConditionalBranchInst>()) { return false; }
        auto *cond_br = static_cast<ConditionalBranchInst *>(prep_term);
        auto *cond = cond_br->condition();
        if (cond == nullptr || !cond->isa<ArithmeticInst>()) { return false; }
        auto *cmp = static_cast<ArithmeticInst *>(cond);
        auto op = cmp->op();
        if (op != ArithmeticOp::BINARY_LESS && op != ArithmeticOp::BINARY_LESS_EQUAL) { return false; }
        if (!cmp->operand(1)->isa<Constant>()) { return false; }
        auto *induction = cmp->operand(0);
        if (induction == nullptr || !induction->isa<PhiInst>()) { return false; }
        auto *phi = static_cast<PhiInst *>(induction);
        if (phi->parent_block() != preheader) { return false; }
        return true;
    }
};

[[nodiscard]] int64_t get_constant_int_value(Constant *c) noexcept {
    if (c == nullptr) { return 0; }
    if (c->type()->is_int32()) { return static_cast<int64_t>(c->as<int32_t>()); }
    if (c->type()->is_uint32()) { return static_cast<int64_t>(c->as<uint32_t>()); }
    if (c->type()->is_int64()) { return c->as<int64_t>(); }
    return 0;
}

[[nodiscard]] bool extract_trip_count(LoopInst *loop, int64_t &out_count) noexcept {
    auto *prepare = loop->prepare_block();
    if (prepare == nullptr) { return false; }
    auto *prep_term = prepare->terminator();
    if (prep_term == nullptr || !prep_term->isa<ConditionalBranchInst>()) { return false; }
    auto *cond_br = static_cast<ConditionalBranchInst *>(prep_term);
    auto *cond = cond_br->condition();
    if (cond == nullptr || !cond->isa<ArithmeticInst>()) { return false; }
    auto *cmp = static_cast<ArithmeticInst *>(cond);
    auto op = cmp->op();
    if (op != ArithmeticOp::BINARY_LESS && op != ArithmeticOp::BINARY_LESS_EQUAL) { return false; }
    if (!cmp->operand(1)->isa<Constant>()) { return false; }
    auto *bound = static_cast<Constant *>(cmp->operand(1));
    auto *induction = cmp->operand(0);
    if (induction == nullptr || !induction->isa<PhiInst>()) { return false; }
    auto *phi = static_cast<PhiInst *>(induction);
    if (phi->parent_block() != prepare) { return false; }

    auto *scev = scev_get_for_value(phi);
    if (scev == nullptr || scev->kind() != SCEV::Kind::ADD_REC) { return false; }
    auto *add_rec = static_cast<const SCEVAddRec *>(scev);
    if (add_rec->start()->kind() != SCEV::Kind::CONSTANT ||
        add_rec->stride()->kind() != SCEV::Kind::CONSTANT) {
        return false;
    }
    auto *start_c = static_cast<const SCEVConstant *>(add_rec->start())->constant();
    auto *step_c = static_cast<const SCEVConstant *>(add_rec->stride())->constant();

    int64_t start = get_constant_int_value(start_c);
    int64_t step = get_constant_int_value(step_c);
    int64_t bound_val = get_constant_int_value(bound);
    if (step <= 0) { return false; }

    int64_t trips = (bound_val - start + (op == ArithmeticOp::BINARY_LESS_EQUAL ? 1 : 0) + step - 1) / step;
    if (trips < 0) { return false; }
    out_count = trips;
    return true;
}

[[nodiscard]] AllocaInst *get_memory_base(Instruction *inst) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::LOAD: {
            auto *load = static_cast<LoadInst *>(inst);
            return trace_pointer_base_local_alloca_inst(load->variable());
        }
        case DerivedInstructionTag::STORE: {
            auto *store = static_cast<StoreInst *>(inst);
            return trace_pointer_base_local_alloca_inst(store->variable());
        }
        default: break;
    }
    return nullptr;
}

// Two LoopInst are adjacent if L1 lives in L0's merge block and that merge block
// is only reachable from L0's prepare block (via the cond_br false edge).
[[nodiscard]] bool are_adjacent(LoopInst *l0, LoopInst *l1, FunctionDefinition *def) noexcept {
    auto *m0 = l0->merge_block();
    auto *l1_parent = l1->parent_block();
    if (m0 == nullptr || l1_parent == nullptr) { return false; }
    if (l1_parent != m0) { return false; }
    size_t pred_count = 0;
    bool pred_is_prepare = false;
    m0->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
        pred_count++;
        if (pred == l0->prepare_block()) { pred_is_prepare = true; }
    });
    return pred_count == 1 && pred_is_prepare;
}

[[nodiscard]] bool trip_counts_equal(LoopInst *l0, LoopInst *l1, FunctionDefinition *def) noexcept {
    int64_t t0 = 0;
    int64_t t1 = 0;
    if (!extract_trip_count(l0, t0)) { return false; }
    if (!extract_trip_count(l1, t1)) { return false; }
    return t0 == t1;
}

[[nodiscard]] bool dependences_allow_fusion(FusionCandidate &fc0, FusionCandidate &fc1, FunctionDefinition *def) noexcept {
    // Reject any memory dependence on the same local alloca base between the two loops.
    // Different local allocas or non-local accesses on both sides are OK; mixing local
    // and non-local accesses is rejected because we cannot analyze aliasing.
    for (auto *w0 : fc0.mem_writes) {
        for (auto *r1 : fc1.mem_reads) {
            auto *base0 = get_memory_base(w0);
            auto *base1 = get_memory_base(r1);
            if (base0 == nullptr && base1 == nullptr) { continue; }
            if (base0 != nullptr && base1 != nullptr && base0 != base1) { continue; }
            return false;
        }
    }
    // Anti-dependences: reads in L0 followed by writes in L1 to the same base are not OK.
    for (auto *r0 : fc0.mem_reads) {
        for (auto *w1 : fc1.mem_writes) {
            auto *base0 = get_memory_base(r0);
            auto *base1 = get_memory_base(w1);
            if (base0 == nullptr || base1 == nullptr) { return false; }
            if (base0 == base1) { return false; }
        }
    }
    // Output dependences: writes in both loops to the same base are not OK.
    for (auto *w0 : fc0.mem_writes) {
        for (auto *w1 : fc1.mem_writes) {
            auto *base0 = get_memory_base(w0);
            auto *base1 = get_memory_base(w1);
            if (base0 == nullptr || base1 == nullptr) { return false; }
            if (base0 == base1) { return false; }
        }
    }
    // Cross-loop def-use: reject if any instruction in L1 uses a value defined in L0.
    luisa::unordered_set<BasicBlock *> l0_blocks;
    l0_blocks.emplace(fc0.preheader);
    l0_blocks.emplace(fc0.body);
    l0_blocks.emplace(fc0.update);
    for (auto *bb : {fc1.preheader, fc1.body, fc1.update}) {
        if (bb == nullptr) { continue; }
        for (auto *inst : bb->instructions()) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                auto *op = inst->operand(i);
                if (op != nullptr && op->isa<Instruction>()) {
                    auto *op_inst = static_cast<Instruction *>(op);
                    if (l0_blocks.contains(op_inst->parent_block())) { return false; }
                }
            }
        }
    }
    return true;
}

void perform_fusion(FusionCandidate &fc0, FusionCandidate &fc1) noexcept {
    auto *l0 = fc0.loop;
    auto *l1 = fc1.loop;
    auto *p0 = fc0.preheader;
    auto *u0 = fc0.update;
    auto *p1 = fc1.preheader;
    auto *u1 = fc1.update;
    auto *m1 = fc1.exit_block;
    if (l0 == nullptr || l1 == nullptr || p0 == nullptr || u0 == nullptr ||
        p1 == nullptr || u1 == nullptr || m1 == nullptr) {
        return;
    }

    // Step A: L0's update block now branches to L1's prepare block.
    auto *u0_term = u0->terminator();
    if (u0_term != nullptr && u0_term->isa<BranchInst>()) {
        static_cast<BranchInst *>(u0_term)->set_target_block(p1);
    }

    // Step B: L1's update block now branches back to L0's prepare block.
    auto *u1_term = u1->terminator();
    if (u1_term != nullptr && u1_term->isa<BranchInst>()) {
        static_cast<BranchInst *>(u1_term)->set_target_block(p0);
    }

    // Step C: L0's merge block becomes L1's merge block, and its cond_br exits there.
    l0->set_merge_block(m1);
    auto *p0_term = p0->terminator();
    if (p0_term != nullptr && p0_term->isa<ConditionalBranchInst>()) {
        static_cast<ConditionalBranchInst *>(p0_term)->set_false_target(m1);
    }

    // P0 now has an additional predecessor (U1). Replicate the existing U0 incoming
    // value for the new U1 edge so loop-carried values remain valid.
    for (auto *inst : p0->instructions()) {
        if (inst->is_terminator()) { break; }
        if (!inst->isa<PhiInst>()) { continue; }
        auto *phi = static_cast<PhiInst *>(inst);
        Value *u0_value = nullptr;
        for (size_t i = 0; i < phi->incoming_count(); ++i) {
            if (phi->incoming(i).block == u0) {
                u0_value = phi->incoming(i).value;
                break;
            }
        }
        if (u0_value != nullptr) {
            phi->add_incoming(u0_value, u1);
        }
    }

    // P1 now has an additional predecessor (U0). Replicate the existing preheader
    // incoming value for the new U0 edge.
    auto *l1_parent = l1->parent_block();
    if (l1_parent != nullptr) {
        for (auto *inst : p1->instructions()) {
            if (inst->is_terminator()) { break; }
            if (!inst->isa<PhiInst>()) { continue; }
            auto *phi = static_cast<PhiInst *>(inst);
            Value *preheader_value = nullptr;
            for (size_t i = 0; i < phi->incoming_count(); ++i) {
                if (phi->incoming(i).block == l1_parent) {
                    preheader_value = phi->incoming(i).value;
                    break;
                }
            }
            if (preheader_value != nullptr) {
                phi->add_incoming(preheader_value, u0);
            }
        }
    }

    // Remove L1 from its parent block and seal that block with a branch to M1.
    l1->remove_self();
    if (l1_parent != nullptr && !l1_parent->is_terminated()) {
        XIRBuilder builder;
        builder.set_insertion_point(l1_parent);
        builder.br(m1);
    }
}

}// namespace detail

LoopFusionInfo loop_fusion_pass_run_on_function(Function *function) noexcept {
    auto *def = function->definition();
    if (def == nullptr) { return {}; }

    // Populate SCEV data for trip count comparison.
    scev_pass_run_on_function(def);

    // Collect all LoopInst in the function in traversal order.
    luisa::vector<LoopInst *> loops;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<LoopInst>()) {
            loops.push_back(static_cast<LoopInst *>(inst));
        }
    });

    // Build FusionCandidates for eligible loops.
    luisa::vector<detail::FusionCandidate> candidates;
    candidates.reserve(loops.size());
    for (auto *loop : loops) {
        detail::FusionCandidate fc;
        fc.loop = loop;
        fc.preheader = loop->prepare_block();
        fc.header = loop->prepare_block();
        fc.body = loop->body_block();
        fc.update = loop->update_block();
        fc.exit_block = loop->merge_block();
        fc.collect_memory_accesses();
        fc.valid = fc.is_eligible(def);
        if (fc.valid) {
            candidates.push_back(std::move(fc));
        }
    }

    // Try to fuse each eligible loop with the next adjacent one in traversal order.
    LoopFusionInfo info;
    for (size_t i = 0; i + 1 < candidates.size(); ++i) {
        auto &fc0 = candidates[i];
        auto &fc1 = candidates[i + 1];
        if (!detail::are_adjacent(fc0.loop, fc1.loop, def)) { continue; }
        if (!detail::trip_counts_equal(fc0.loop, fc1.loop, def)) { continue; }
        if (!detail::dependences_allow_fusion(fc0, fc1, def)) { continue; }
        detail::perform_fusion(fc0, fc1);
        info.fused_loop_count++;
        ++i;// skip the fused partner
    }

    return info;
}

LoopFusionInfo loop_fusion_pass_run_on_module(Module *module, PassReport *report) noexcept {
    LoopFusionInfo info;
    for (auto *func : module->function_list()) {
        if (auto *def = func->definition()) {
            auto fi = loop_fusion_pass_run_on_function(func);
            info.fused_loop_count += fi.fused_loop_count;
        }
    }
    if (report != nullptr) {
        report->set("fused_loop_count", info.fused_loop_count);
    }
    return info;
}

}// namespace luisa::compute::xir
