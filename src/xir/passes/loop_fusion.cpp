#include <luisa/xir/passes/loop_fusion.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/store.h>

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
    if (a->type() != b->type()) { return false; }
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
    luisa::unordered_set<const Value *> local_reads;
    luisa::unordered_set<const Value *> local_writes;
    bool has_global_read{false};
    bool has_global_write{false};
    bool has_unmodeled_effect{false};
};

[[nodiscard]] LoopMemoryFootprint collect_memory_footprint(const NaturalLoop &loop) noexcept {
    LoopMemoryFootprint footprint;
    auto scan = [&](BasicBlock *block) noexcept {
        block->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->is_terminator() || inst->isa<PhiInst>()) { return; }
            auto memory = get_memory_info(inst);
            if (memory.is_volatile || inst->isa<CallInst>() ||
                inst->isa<AtomicInst>()) {
                footprint.has_unmodeled_effect = true;
                return;
            }
            if (!memory.reads_memory() && !memory.writes_memory()) { return; }
            if (inst->isa<LoadInst>() || inst->isa<StoreInst>()) {
                auto *pointer = inst->isa<LoadInst>() ?
                                    static_cast<LoadInst *>(inst)->variable() :
                                    static_cast<StoreInst *>(inst)->variable();
                auto *base = trace_pointer_base_value(pointer);
                if (base == nullptr || !base->isa<AllocaInst>()) {
                    footprint.has_unmodeled_effect = true;
                    return;
                }
                if (memory.reads_memory()) {
                    footprint.local_reads.emplace(base);
                }
                if (memory.writes_memory()) {
                    footprint.local_writes.emplace(base);
                }
                return;
            }
            if (inst->isa<ResourceReadInst>()) {
                footprint.has_global_read = true;
                return;
            }
            if (inst->isa<ResourceWriteInst>()) {
                footprint.has_global_write = true;
                return;
            }
            // Clocks, ray-query state, thread-group operations, and future
            // memory instructions do not have a proven fusion dependence
            // model.
            footprint.has_unmodeled_effect = true;
        });
    };
    scan(loop.header);
    for (auto *block : loop.body_blocks) { scan(block); }
    return footprint;
}

[[nodiscard]] bool memory_footprints_are_independent(
    const LoopMemoryFootprint &first,
    const LoopMemoryFootprint &second) noexcept {
    if (first.has_unmodeled_effect || second.has_unmodeled_effect) {
        return false;
    }
    // Distinct resource SSA values may still be bound to overlapping runtime
    // views. Without an explicit no-alias contract, any global write conflicts
    // with every global access in the other loop.
    if ((first.has_global_write &&
         (second.has_global_read || second.has_global_write)) ||
        (second.has_global_write &&
         (first.has_global_read || first.has_global_write))) {
        return false;
    }
    for (auto *base : first.local_writes) {
        if (second.local_reads.contains(base) ||
            second.local_writes.contains(base)) {
            return false;
        }
    }
    for (auto *base : second.local_writes) {
        if (first.local_reads.contains(base) ||
            first.local_writes.contains(base)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool canonical_header_for_fusion(
    const LoopBoundsInfo &bounds) noexcept {
    auto *header = bounds.induction_phi->parent_block();
    auto *terminator = header->terminator();
    if (bounds.comparison_inst == nullptr || terminator == nullptr ||
        !terminator->isa<ConditionalBranchInst>()) {
        return false;
    }
    for (auto *inst : header->instructions()) {
        if (inst->isa<PhiInst>() || inst == bounds.comparison_inst ||
            inst == terminator) {
            continue;
        }
        return false;
    }
    // The comparison disappears with the second header, so it may only feed
    // that header's branch.
    for (auto *use : bounds.comparison_inst->use_list()) {
        if (use->user() != terminator) { return false; }
    }
    return true;
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

[[nodiscard]] bool block_has_only_predecessor(
    BasicBlock *block, BasicBlock *expected) noexcept {
    auto saw_expected = false;
    auto saw_other = false;
    block->traverse_predecessors(false, [&](BasicBlock *predecessor) noexcept {
        saw_expected |= predecessor == expected;
        saw_other |= predecessor != expected;
    });
    return saw_expected && !saw_other;
}

[[nodiscard]] bool try_fuse_pair(FunctionDefinition *def,
                                 const NaturalLoop &first,
                                 const NaturalLoop &second) noexcept {
    // Canonical shapes on both sides.
    if (first.preheader == nullptr || second.preheader == nullptr ||
        first.latches.size() != 1u || second.latches.size() != 1u ||
        first.exit_blocks.size() != 1u || second.exit_blocks.size() != 1u ||
        first.exit_edges.size() != 1u || second.exit_edges.size() != 1u) {
        return false;
    }
    // Adjacency: the first loop's exit is the second loop's preheader.
    if (first.exit_blocks.front() != second.preheader) { return false; }
    // The shared block must be a trivial branch into the second header.
    if (!block_is_trivial_branch(second.preheader, second.header)) { return false; }
    // This block is deleted by the transform. NaturalLoop::preheader only
    // constrains predecessors of the second header, so independently reject
    // a bypass edge entering the shared preheader itself.
    if (!block_has_only_predecessor(
            second.preheader, first.exit_edges.front().first)) {
        return false;
    }

    auto bounds1 = analyze_loop_bounds(first);
    auto bounds2 = analyze_loop_bounds(second);
    if (!bounds1.is_valid() || !bounds2.is_valid() ||
        !bounds1.stride_is_constant || !bounds2.stride_is_constant ||
        !bounds1.trip_count_is_constant ||
        !bounds2.trip_count_is_constant ||
        bounds1.constant_trip_count != bounds2.constant_trip_count ||
        bounds1.stride != bounds2.stride ||
        bounds1.induction_phi->type() != bounds2.induction_phi->type() ||
        bounds1.comparison != bounds2.comparison ||
        bounds1.induction_is_lhs != bounds2.induction_is_lhs ||
        bounds1.continue_on_true != bounds2.continue_on_true ||
        !same_constant_value(bounds1.start_value, bounds2.start_value) ||
        !same_constant_value(bounds1.bound_value, bounds2.bound_value) ||
        bounds1.exit_block != second.preheader ||
        bounds2.exit_block != second.exit_blocks.front() ||
        !canonical_header_for_fusion(bounds1) ||
        !canonical_header_for_fusion(bounds2)) {
        return false;
    }

    // Dependence: no buffer written by one loop may be accessed by the other,
    // no calls/atomics on either side, and no SSA values flowing from the
    // first loop into the second (their per-iteration meaning would change).
    auto mem1 = collect_memory_footprint(first);
    auto mem2 = collect_memory_footprint(second);
    if (!memory_footprints_are_independent(mem1, mem2)) { return false; }
    if (has_cross_value_flow(first, second)) { return false; }

    auto *header1 = first.header;
    auto *latch1 = first.latches.front();
    auto *preheader1 = first.preheader;
    auto *header2 = second.header;
    auto *latch2 = second.latches.front();
    auto *preheader2 = second.preheader;// == first exit
    auto *exit2 = second.exit_blocks.front();

    // These two blocks are deleted by fusion. Movable non-induction Phis
    // retain their metadata because the instructions themselves are moved,
    // but block metadata and metadata on the eliminated transition/header
    // instructions have no unique semantics-preserving destination.
    if (!preheader2->metadata_list().empty() ||
        !header2->metadata_list().empty() ||
        !preheader2->terminator()->metadata_list().empty() ||
        !bounds2.induction_phi->metadata_list().empty() ||
        !bounds2.comparison_inst->metadata_list().empty() ||
        !header2->terminator()->metadata_list().empty()) {
        return false;
    }

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
    auto *body2_entry = bounds2.body_entry;
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
    // The second body entry used to be reached from its header. It is now
    // reached from the first latch, so edge-indexed Phi inputs must follow
    // the rewritten predecessor edge before the old header is removed.
    for (auto *inst : body2_entry->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block == header2) {
                phi->set_incoming(i, incoming.value, latch1);
            }
        }
    }
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

static void loop_fusion_run(Function *function, LoopFusionInfo &info) noexcept {
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

[[nodiscard]] static bool loop_fusion_preflight_module(
    Module *module, LoopFusionInfo &info) noexcept {
    if (module == nullptr) { return true; }
    for (auto *function : module->function_list()) {
        auto *def = function == nullptr ? nullptr : function->definition();
        if (def != nullptr && contains_structured_control_flow(def)) {
            ++info.structured_cfg_error_count;
        }
    }
    if (info.structured_cfg_error_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "Loop fusion rejected a module containing structured CFG; "
            "run destructure_cfg first. The entire module was left unchanged.");
        return false;
    }
    return true;
}

}// namespace detail

LoopFusionInfo loop_fusion_pass_run_on_function(Function *function) noexcept {
    LoopFusionInfo info;
    detail::loop_fusion_run(function, info);
    return info;
}

LoopFusionInfo loop_fusion_pass_run_on_module(Module *module,
                                              PassReport *report) noexcept {
    LoopFusionInfo info;
    if (detail::loop_fusion_preflight_module(module, info)) {
        if (module != nullptr) {
            for (auto *function : module->function_list()) {
                detail::loop_fusion_run(function, info);
            }
        }
    }
    if (report != nullptr) {
        report->set("fused_loop_count", info.fused_loop_count);
        report->set("structured_cfg_error_count", info.structured_cfg_error_count);
    }
    return info;
}

}// namespace luisa::compute::xir
