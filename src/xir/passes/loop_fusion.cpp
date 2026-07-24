#include <luisa/xir/passes/loop_fusion.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/resource.h>

#include "helpers.h"
#include "natural_loop.h"

namespace luisa::compute::xir {

namespace detail {

namespace {

[[nodiscard]] bool same_constant_value(const Value *a, const Value *b) noexcept {
    if (a == b) { return true; }
    if (a == nullptr || b == nullptr ||
        !a->isa<Constant>() || !b->isa<Constant>()) {
        return false;
    }
    int64_t va = 0;
    int64_t vb = 0;
    auto decode = [](const Value *v, int64_t &out) noexcept {
        auto *c = static_cast<const Constant *>(v);
        auto *t = c->type();
        if (t->is_int32()) {
            out = c->as<int32_t>();
        } else if (t->is_uint32()) {
            out = c->as<uint32_t>();
        } else if (t->is_int64()) {
            out = c->as<int64_t>();
        } else if (t->is_uint64()) {
            auto u = c->as<uint64_t>();
            if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return false; }
            out = static_cast<int64_t>(u);
        } else if (t->is_int16()) {
            out = c->as<int16_t>();
        } else if (t->is_uint16()) {
            out = c->as<uint16_t>();
        } else if (t->is_int8()) {
            out = c->as<int8_t>();
        } else if (t->is_uint8()) {
            out = c->as<uint8_t>();
        } else {
            return false;
        }
        return true;
    };
    return decode(a, va) && decode(b, vb) && va == vb;
}

struct LoopMemoryFootprint {
    luisa::unordered_set<const Value *> read_buffers;
    luisa::unordered_set<const Value *> written_buffers;
    bool has_calls_or_atomics{false};
};

[[nodiscard]] LoopMemoryFootprint collect_memory_footprint(const NaturalLoop &loop) noexcept {
    LoopMemoryFootprint footprint;
    auto scan = [&](BasicBlock *block) noexcept {
        block->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<CallInst>() || inst->isa<AtomicInst>()) {
                footprint.has_calls_or_atomics = true;
                return;
            }
            if (inst->isa<ResourceReadInst>()) {
                auto *read = static_cast<ResourceReadInst *>(inst);
                if (read->operand_count() != 0u) {
                    footprint.read_buffers.emplace(read->operand(0u));
                }
            } else if (inst->isa<ResourceWriteInst>()) {
                auto *write = static_cast<ResourceWriteInst *>(inst);
                if (write->operand_count() != 0u) {
                    footprint.written_buffers.emplace(write->operand(0u));
                }
            }
        });
    };
    scan(loop.header);
    for (auto *block : loop.body_blocks) { scan(block); }
    return footprint;
}

[[nodiscard]] bool has_cross_value_flow(const NaturalLoop &from,
                                        const NaturalLoop &to) noexcept {
    auto found = false;
    auto check = [&](BasicBlock *block) noexcept {
        block->traverse_instructions([&](Instruction *inst) noexcept {
            if (found) { return; }
            for (auto *use : inst->use_list()) {
                auto *user = use->user();
                if (user != nullptr && user->isa<Instruction>()) {
                    auto *user_block = static_cast<Instruction *>(user)->parent_block();
                    if (user_block != nullptr && to.contains(user_block)) {
                        found = true;
                        return;
                    }
                }
            }
        });
    };
    check(from.header);
    for (auto *block : from.body_blocks) { check(block); }
    return found;
}

[[nodiscard]] bool block_is_trivial_branch(BasicBlock *block, BasicBlock *target) noexcept {
    auto count = 0u;
    auto ok = true;
    block->traverse_instructions([&](Instruction *inst) noexcept {
        count++;
        if (inst != block->terminator()) { ok = false; }
    });
    if (!ok || count != 1u) { return false; }
    auto *term = block->terminator();
    return term != nullptr && term->isa<BranchInst>() &&
           static_cast<BranchInst *>(term)->target_block() == target;
}

[[nodiscard]] bool try_fuse_pair(FunctionDefinition *def,
                                 const NaturalLoop &first,
                                 const NaturalLoop &second) noexcept {
    // Canonical shapes on both sides.
    if (first.preheader == nullptr || second.preheader == nullptr ||
        first.latches.size() != 1u || second.latches.size() != 1u ||
        first.exit_blocks.size() != 1u || second.exit_blocks.size() != 1u) {
        return false;
    }
    // Adjacency: the first loop's exit is the second loop's preheader.
    if (first.exit_blocks.front() != second.preheader) { return false; }
    // The shared block must be a trivial branch into the second header.
    if (!block_is_trivial_branch(second.preheader, second.header)) { return false; }

    auto bounds1 = analyze_loop_bounds(first);
    auto bounds2 = analyze_loop_bounds(second);
    if (!bounds1.is_valid() || !bounds2.is_valid() ||
        !bounds1.stride_is_constant || !bounds2.stride_is_constant ||
        bounds1.stride != bounds2.stride ||
        bounds1.induction_phi->type() != bounds2.induction_phi->type() ||
        !same_constant_value(bounds1.start_value, bounds2.start_value) ||
        !same_constant_value(bounds1.bound_value, bounds2.bound_value)) {
        return false;
    }

    // Dependence: no buffer written by one loop may be accessed by the other,
    // no calls/atomics on either side, and no SSA values flowing from the
    // first loop into the second (their per-iteration meaning would change).
    auto mem1 = collect_memory_footprint(first);
    auto mem2 = collect_memory_footprint(second);
    if (mem1.has_calls_or_atomics || mem2.has_calls_or_atomics) { return false; }
    for (auto *buffer : mem1.written_buffers) {
        if (mem2.read_buffers.contains(buffer) ||
            mem2.written_buffers.contains(buffer)) {
            return false;
        }
    }
    for (auto *buffer : mem2.written_buffers) {
        if (mem1.read_buffers.contains(buffer)) { return false; }
    }
    if (has_cross_value_flow(first, second)) { return false; }

    auto *header1 = first.header;
    auto *latch1 = first.latches.front();
    auto *preheader1 = first.preheader;
    auto *header2 = second.header;
    auto *latch2 = second.latches.front();
    auto *preheader2 = second.preheader;// == first exit
    auto *exit2 = second.exit_blocks.front();

    // The second header's phis must be replaceable: the induction phi maps
    // to the first loop's phi; every other phi moves to the first header
    // with its entry edge retargeted to the first preheader.
    luisa::vector<PhiInst *> movable_phis;
    for (auto *inst : header2->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        if (phi == bounds2.induction_phi) { continue; }
        if (phi->incoming_count() != 2u) { return false; }
        Value *start = nullptr;
        Value *recur = nullptr;
        for (auto i = 0u; i < 2u; ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block == preheader2) { start = incoming.value; }
            if (incoming.block == latch2) { recur = incoming.value; }
        }
        if (start == nullptr || recur == nullptr) { return false; }
        // The start value must be defined before the first loop (it must
        // dominate the first preheader).
        if (start->isa<Instruction>() &&
            (first.contains(static_cast<Instruction *>(start)->parent_block()) ||
             static_cast<Instruction *>(start)->parent_block() == preheader2)) {
            return false;
        }
        movable_phis.emplace_back(phi);
    }
    // Exit phis on the second exit may only consume the second loop's phis
    // or loop-invariant values from the second header's edge.
    for (auto *inst : exit2->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block != header2) { return false; }
            auto *value = incoming.value;
            if (value == bounds2.induction_phi) { continue; }
            auto is_movable = false;
            for (auto *movable : movable_phis) {
                if (movable == value) { is_movable = true; }
            }
            if (is_movable) { continue; }
            if (value->isa<Instruction>() &&
                second.contains(static_cast<Instruction *>(value)->parent_block())) {
                return false;
            }
        }
    }

    // Both latches must close their loops unconditionally.
    auto *latch1_term = latch1->terminator();
    auto *latch2_term = latch2->terminator();
    if (latch1_term == nullptr || !latch1_term->isa<BranchInst>() ||
        static_cast<BranchInst *>(latch1_term)->target_block() != header1 ||
        latch2_term == nullptr || !latch2_term->isa<BranchInst>() ||
        static_cast<BranchInst *>(latch2_term)->target_block() != header2) {
        return false;
    }

    // Fuse. The first loop's header now exits to the second loop's exit.
    auto *header1_branch = static_cast<ConditionalBranchInst *>(header1->terminator());
    if (header1_branch->true_block() == preheader2) {
        header1_branch->set_true_target(exit2);
    } else {
        LUISA_ASSERT(header1_branch->false_block() == preheader2,
                     "Loop fusion lost the first loop's exit edge.");
        header1_branch->set_false_target(exit2);
    }
    // The first latch falls into the second body.
    auto *latch1_branch = static_cast<BranchInst *>(latch1->terminator());
    // The second header's in-loop successor is the second body entry.
    auto *header2_branch = static_cast<ConditionalBranchInst *>(header2->terminator());
    auto *body2_entry = header2_branch->true_block() == exit2 ?
                            header2_branch->false_block() :
                            header2_branch->true_block();
    latch1_branch->set_target_block(body2_entry);
    // The second latch closes the fused loop.
    auto *latch2_branch = static_cast<BranchInst *>(latch2->terminator());
    latch2_branch->set_target_block(header1);

    // The fused loop's back-edge now comes from the second latch, so the
    // first header's phis retarget their recurrence edge accordingly.
    for (auto *inst : header1->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            if (phi->incoming(i).block == latch1) {
                phi->set_incoming(i, phi->incoming(i).value, latch2);
            }
        }
    }
    // Replace the second induction phi with the first.
    bounds2.induction_phi->replace_all_uses_with(bounds1.induction_phi);
    // Move the remaining second-header phis into the first header.
    Instruction *last_phi1 = nullptr;
    for (auto *inst : header1->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        last_phi1 = inst;
    }
    for (auto *phi : movable_phis) {
        // Retarget the entry edge before moving the instruction.
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            if (phi->incoming(i).block == preheader2) {
                phi->set_incoming(i, phi->incoming(i).value, preheader1);
            }
        }
        auto owned = phi->remove_self();
        if (last_phi1 != nullptr) {
            auto *inserted = last_phi1->insert_after_self(std::move(owned));
            last_phi1 = inserted;
        } else {
            // Insert at the top of the header: after the head sentinel,
            // which is the predecessor of the first real instruction.
            auto *first_inst = header1->instructions().front();
            auto *inserted = first_inst->insert_before_self(std::move(owned));
            last_phi1 = inserted;
        }
    }
    // Exit phis on the shared exit now take their values from the first
    // header's edge.
    for (auto *inst : exit2->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            if (phi->incoming(i).block == header2) {
                auto *value = phi->incoming(i).value;
                if (value == bounds2.induction_phi) {
                    value = bounds1.induction_phi;
                }
                phi->set_incoming(i, value, header1);
            }
        }
    }
    // Remove the now-dead shared preheader and second header.
    static_cast<void>(preheader2->remove_self());
    static_cast<void>(header2->remove_self());
    return true;
}

}// namespace

static void run(Function *function, LoopFusionInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto *def = function->definition();
    if (def == nullptr) { return; }
    if (contains_structured_control_flow(def)) {
        ++info.structured_cfg_error_count;
        LUISA_WARNING_WITH_LOCATION(
            "Loop fusion rejected structured CFG; run destructure_cfg first. "
            "IR was left unchanged.");
        return;
    }
    // Fusion removes blocks; rediscover loops after every success.
    for (;;) {
        auto dom_tree = compute_dom_tree(def);
        auto loops = discover_natural_loops(def, dom_tree);
        auto any_fused = false;
        for (auto i = 0u; i < loops.size() && !any_fused; ++i) {
            for (auto j = 0u; j < loops.size() && !any_fused; ++j) {
                if (i == j) { continue; }
                if (try_fuse_pair(def, loops[i], loops[j])) {
                    ++info.fused_loop_count;
                    any_fused = true;
                }
            }
        }
        if (!any_fused) { break; }
    }
}

}// namespace detail

LoopFusionInfo loop_fusion_pass_run_on_function(Function *function) noexcept {
    LoopFusionInfo info;
    detail::run(function, info);
    return info;
}

LoopFusionInfo loop_fusion_pass_run_on_module(Module *module,
                                              PassReport *report) noexcept {
    LoopFusionInfo info;
    for (auto *function : module->function_list()) {
        detail::run(function, info);
    }
    if (report != nullptr) {
        report->set("fused_loop_count", info.fused_loop_count);
        report->set("structured_cfg_error_count", info.structured_cfg_error_count);
    }
    return info;
}

}// namespace luisa::compute::xir
