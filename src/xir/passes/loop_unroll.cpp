#include <luisa/xir/passes/loop_unroll.h>
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

#include "helpers.h"
#include "natural_loop.h"

namespace luisa::compute::xir {

namespace detail {

namespace {

class UnrollValueResolver final : public InstructionCloneValueResolver {
    luisa::unordered_map<const Value *, Value *> _map;

public:
    void emplace(const Value *from, Value *to) noexcept { _map.emplace(from, to); }
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        if (auto it = _map.find(value); it != _map.end()) { return it->second; }
        return const_cast<Value *>(value);
    }
};

// Breadth-first order of the loop body starting at the body entry, restricted
// to in-loop blocks. Returns false when the body contains an inner back-edge
// (a nested loop), which full unrolling cannot expand.
[[nodiscard]] bool collect_body_blocks_in_order(
    const NaturalLoop &loop, BasicBlock *body_entry,
    luisa::vector<BasicBlock *> &ordered) noexcept {
    ordered.clear();
    luisa::unordered_set<BasicBlock *> visited;
    luisa::unordered_set<BasicBlock *> finished;
    luisa::vector<BasicBlock *> worklist{body_entry};
    visited.emplace(body_entry);
    while (!worklist.empty()) {
        auto *block = worklist.back();
        worklist.pop_back();
        ordered.emplace_back(block);
        finished.emplace(block);
        if (!block->is_terminated()) { return false; }
        block->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (succ == loop.header || !loop.contains(succ)) { return; }
            if (finished.contains(succ)) {
                // An edge back to an already-processed block is an inner
                // back-edge only when it targets a proper ancestor; edges to
                // siblings processed earlier are fine in a DAG-shaped body.
                // We conservatively treat any cycle as a nested loop.
                return;
            }
            if (visited.emplace(succ).second) {
                worklist.emplace_back(succ);
            }
        });
    }
    // Cycle detection: every in-loop successor of every body block must
    // appear no earlier in the order than its source, except the latch edge
    // back to the header (which is not in the body order).
    luisa::unordered_map<BasicBlock *, size_t> position;
    for (auto i = 0u; i < ordered.size(); ++i) { position.emplace(ordered[i], i); }
    for (auto *block : ordered) {
        auto has_back_edge = false;
        block->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (succ == loop.header || !loop.contains(succ)) { return; }
            auto it = position.find(succ);
            if (it != position.end() && it->second <= position[block]) {
                has_back_edge = true;
            }
        });
        if (has_back_edge) { return false; }
    }
    return true;
}

[[nodiscard]] bool loop_body_has_side_effect(const NaturalLoop &loop) noexcept {
    auto side_effect = false;
    auto check = [&](BasicBlock *block) noexcept {
        block->traverse_instructions([&](Instruction *inst) noexcept {
            if (side_effect) { return; }
            if (inst->isa<CallInst>()) {
                side_effect = true;
                return;
            }
            if (inst->is_terminator()) { return; }
            auto memory = get_memory_info(inst);
            if (memory.writes_memory() || memory.is_volatile) {
                side_effect = true;
            }
        });
    };
    check(loop.header);
    for (auto *block : loop.body_blocks) { check(block); }
    return side_effect;
}

[[nodiscard]] bool try_full_unroll(FunctionDefinition *def, const NaturalLoop &loop,
                                   const LoopUnrollOptions &options) noexcept {
    if (loop.preheader == nullptr || loop.latches.size() != 1u ||
        loop.exit_blocks.size() != 1u) {
        return false;
    }
    auto *header = loop.header;
    auto *preheader = loop.preheader;
    auto *latch = loop.latches.front();
    auto *exit_block = loop.exit_blocks.front();

    auto bounds = analyze_loop_bounds(loop);
    if (!bounds.is_valid() || !bounds.trip_count_is_constant) { return false; }
    auto trip_count = bounds.constant_trip_count;
    if (trip_count == 0u || trip_count > options.max_trip_count) { return false; }

    // Identify the body entry (the header's in-loop successor) and require
    // that the header is the only block with an exit edge.
    auto *terminator = header->terminator();
    if (terminator == nullptr || !terminator->isa<ConditionalBranchInst>()) {
        return false;
    }
    auto *header_branch = static_cast<ConditionalBranchInst *>(terminator);
    BasicBlock *body_entry = nullptr;
    if (header_branch->true_block() == exit_block) {
        body_entry = header_branch->false_block();
    } else if (header_branch->false_block() == exit_block) {
        body_entry = header_branch->true_block();
    } else {
        return false;
    }
    if (!loop.contains(body_entry)) { return false; }
    {
        auto extra_exit = false;
        auto check_exits = [&](BasicBlock *block) noexcept {
            if (block == header) { return; }
            block->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (!loop.contains(succ)) { extra_exit = true; }
            });
        };
        for (auto *block : loop.body_blocks) { check_exits(block); }
        if (extra_exit) { return false; }
    }
    // The latch must branch back unconditionally.
    if (latch != header) {
        auto *latch_terminator = latch->terminator();
        if (latch_terminator == nullptr || !latch_terminator->isa<BranchInst>() ||
            static_cast<BranchInst *>(latch_terminator)->target_block() != header) {
            return false;
        }
    } else {
        // A single-block loop is its own latch; handled by the same code.
    }
    if (options.unroll_pure_only && loop_body_has_side_effect(loop)) { return false; }
    // Body uses of header-computed (non-phi) values are handled by cloning
    // the header's scalar instructions into every iteration's entry block.
    // Body phis must not take incomings from the header (it is removed).
    {
        auto invalid_body_phi = false;
        for (auto *block : loop.body_blocks) {
            block->traverse_instructions([&](Instruction *inst) noexcept {
                if (invalid_body_phi || !inst->isa<PhiInst>()) { return; }
                auto *phi = static_cast<PhiInst *>(inst);
                for (auto i = 0u; i < phi->incoming_count(); ++i) {
                    if (phi->incoming(i).block == header) {
                        invalid_body_phi = true;
                        return;
                    }
                }
            });
        }
        if (invalid_body_phi) { return false; }
    }
    // Values defined in the loop may only escape through exit-block phi
    // incomings on the header edge (those are rewritten to the final
    // iteration's recurrence below).
    {
        auto invalid_escape = false;
        auto check_escape = [&](BasicBlock *block) noexcept {
            block->traverse_instructions([&](Instruction *inst) noexcept {
                if (invalid_escape) { return; }
                for (auto *use : inst->use_list()) {
                    auto *user = use->user();
                    if (user == nullptr || !user->isa<Instruction>()) { continue; }
                    auto *user_inst = static_cast<Instruction *>(user);
                    auto *user_block = user_inst->parent_block();
                    if (user_block == nullptr || loop.contains(user_block)) { continue; }
                    auto is_exit_phi = user_block == exit_block && user_inst->isa<PhiInst>();
                    if (!is_exit_phi) {
                        invalid_escape = true;
                        return;
                    }
                }
            });
        };
        check_escape(header);
        for (auto *block : loop.body_blocks) { check_escape(block); }
        if (invalid_escape) { return false; }
    }
    // Calls are only allowed when the caller accepts side effects.
    if (!options.unroll_pure_only) {
        // still fine: calls are cloned like any other instruction
    }

    // Header non-phi instructions (e.g. the recurrence add when it is
    // computed in the header) are cloned into every iteration's entry block.
    luisa::vector<Instruction *> header_scalar_insts;
    {
        auto past_phis = false;
        for (auto *inst : header->instructions()) {
            if (!past_phis && inst->isa<PhiInst>()) { continue; }
            past_phis = true;
            if (inst->is_terminator()) { continue; }
            header_scalar_insts.emplace_back(inst);
        }
    }

    // Header phis: every phi needs one preheader and one latch incoming.
    luisa::vector<PhiInst *> header_phis;
    for (auto *inst : header->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        Value *from_preheader = nullptr;
        Value *from_latch = nullptr;
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block == preheader) { from_preheader = incoming.value; }
            if (incoming.block == latch) { from_latch = incoming.value; }
        }
        if (from_preheader == nullptr || from_latch == nullptr) { return false; }
        header_phis.emplace_back(phi);
    }

    // Exit phis may only consume header phis (their final value is the
    // last iteration's recurrence) or loop-invariant values, and only from
    // the header edge.
    for (auto *inst : exit_block->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (!loop.contains(incoming.block)) { continue; }
            if (incoming.block != header) { return false; }
            auto *value = incoming.value;
            if (value != nullptr && value->isa<Instruction>()) {
                auto *def_inst = static_cast<Instruction *>(value);
                if (def_inst->parent_block() == header && !def_inst->isa<PhiInst>()) {
                    return false;
                }
                if (def_inst->parent_block() != header && loop.contains(def_inst->parent_block())) {
                    return false;
                }
            }
        }
    }

    luisa::vector<BasicBlock *> ordered_body;
    if (!collect_body_blocks_in_order(loop, body_entry, ordered_body)) {
        return false;
    }
    // The latch must be the last block in the order for a clean fall-through.
    if (ordered_body.empty() || ordered_body.back() != latch) { return false; }

    // Everything validated. Clone the body trip_count times.
    luisa::vector<luisa::unordered_map<BasicBlock *, BasicBlock *>> iteration_blocks;
    iteration_blocks.reserve(trip_count);
    for (auto i = 0u; i < trip_count; ++i) {
        luisa::unordered_map<BasicBlock *, BasicBlock *> block_map;
        for (auto *block : ordered_body) {
            block_map.emplace(block, def->create_basic_block());
        }
        iteration_blocks.emplace_back(std::move(block_map));
    }

    // Per-iteration phi substitutions: iteration 0 uses the preheader
    // incoming, later iterations use the previous iteration's recurrence.
    luisa::unordered_map<const Value *, Value *> phi_values;
    for (auto *phi : header_phis) {
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            if (phi->incoming(i).block == preheader) {
                phi_values.emplace(phi, phi->incoming(i).value);
            }
        }
    }

    XIRBuilder builder;
    BasicBlock *previous_latch_clone = nullptr;
    luisa::unordered_map<const Value *, Value *> last_iteration_map;
    for (auto i = 0u; i < trip_count; ++i) {
        UnrollValueResolver resolver;
        for (auto &[from, to] : iteration_blocks[i]) { resolver.emplace(from, to); }
        for (auto &[phi, value] : phi_values) { resolver.emplace(phi, value); }
        // Clone the header's scalar instructions (recurrence expressions
        // computed in the header) into this iteration's entry block first,
        // then the body blocks in order.
        {
            auto *entry_clone = iteration_blocks[i][body_entry];
            builder.set_insertion_point(entry_clone);
            for (auto *inst : header_scalar_insts) {
                auto *cloned = inst->clone_with_metadata(builder, resolver);
                resolver.emplace(inst, cloned);
            }
        }
        // Clone instructions block by block.
        for (auto *block : ordered_body) {
            auto *clone = iteration_blocks[i][block];
            builder.set_insertion_point(clone);
            for (auto *inst : block->instructions()) {
                auto *cloned = inst->clone_with_metadata(builder, resolver);
                resolver.emplace(inst, cloned);
            }
        }
        // Compute the next iteration's phi values from this iteration's
        // latch incoming clones.
        luisa::unordered_map<const Value *, Value *> next_phi_values;
        for (auto *phi : header_phis) {
            for (auto idx = 0u; idx < phi->incoming_count(); ++idx) {
                if (phi->incoming(idx).block == latch) {
                    next_phi_values.emplace(
                        phi, resolver.resolve(phi->incoming(idx).value));
                }
            }
        }
        phi_values = std::move(next_phi_values);
        // Record the last iteration's value mapping for exit-phi fixups.
        if (i + 1u == trip_count) {
            for (auto *block : ordered_body) {
                for (auto *inst : block->instructions()) {
                    auto *mapped = resolver.resolve(inst);
                    last_iteration_map.emplace(inst, mapped);
                }
            }
            for (auto *phi : header_phis) {
                // phi_values now holds the recurrence after the last
                // iteration, i.e. the final loop value.
                last_iteration_map.emplace(phi, phi_values[phi]);
            }
        }
        // Rewire the previous latch clone to this iteration's entry.
        if (previous_latch_clone != nullptr) {
            auto *prev_terminator = previous_latch_clone->terminator();
            static_cast<BranchInst *>(prev_terminator)
                ->set_target_block(iteration_blocks[i][body_entry]);
        }
        previous_latch_clone = iteration_blocks[i][latch];
    }
    // The last latch clone falls through to the exit.
    {
        auto *last_terminator = previous_latch_clone->terminator();
        static_cast<BranchInst *>(last_terminator)->set_target_block(exit_block);
    }
    // The preheader enters the first iteration.
    {
        auto *preheader_terminator = preheader->terminator();
        LUISA_ASSERT(preheader_terminator != nullptr &&
                         preheader_terminator->isa<BranchInst>(),
                     "Loop unroll requires an unconditional preheader branch.");
        static_cast<BranchInst *>(preheader_terminator)
            ->set_target_block(iteration_blocks.front()[body_entry]);
    }
    // Fix exit phis: the header edge is replaced by the last latch clone.
    for (auto *inst : exit_block->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block != header) { continue; }
            auto *value = incoming.value;
            if (auto it = last_iteration_map.find(value);
                it != last_iteration_map.end()) {
                value = it->second;
            }
            phi->set_incoming(i, value, previous_latch_clone);
        }
    }
    // Remove the original loop blocks.
    luisa::vector<ManagedPtr<BasicBlock>> removed;
    for (auto *block : loop.body_blocks) {
        removed.emplace_back(block->remove_self());
    }
    removed.emplace_back(header->remove_self());
    return true;
}

// Partial unrolling (peeling): emit the first `factor` iterations as
// straight-line clones and keep the remaining loop. The header phis' entry
// incomings are replaced by the peeled recurrence values, which advances the
// induction start and thereby reduces the remaining trip count by `factor`.
[[nodiscard]] bool try_partial_unroll(FunctionDefinition *def, const NaturalLoop &loop,
                                      const LoopUnrollOptions &options) noexcept {
    auto factor = options.partial_unroll_factor;
    if (factor < 2u) { return false; }
    if (loop.preheader == nullptr || loop.latches.size() != 1u ||
        loop.exit_blocks.size() != 1u) {
        return false;
    }
    auto *header = loop.header;
    auto *preheader = loop.preheader;
    auto *latch = loop.latches.front();
    auto *exit_block = loop.exit_blocks.front();
    // Single-block loops are their own latch; full unrolling owns those.
    if (latch == header) { return false; }

    auto bounds = analyze_loop_bounds(loop);
    if (!bounds.is_valid() || !bounds.trip_count_is_constant) { return false; }
    auto trip_count = bounds.constant_trip_count;
    // Keep at least one iteration in the remaining loop.
    if (trip_count <= factor) { return false; }

    // Identify the body entry (the header's in-loop successor) and require
    // that the header is the only block with an exit edge, so the exit phis
    // (untouched by peeling) keep consuming from the header edge only.
    auto *terminator = header->terminator();
    if (terminator == nullptr || !terminator->isa<ConditionalBranchInst>()) {
        return false;
    }
    auto *header_branch = static_cast<ConditionalBranchInst *>(terminator);
    BasicBlock *body_entry = nullptr;
    if (header_branch->true_block() == exit_block) {
        body_entry = header_branch->false_block();
    } else if (header_branch->false_block() == exit_block) {
        body_entry = header_branch->true_block();
    } else {
        return false;
    }
    if (!loop.contains(body_entry)) { return false; }
    {
        auto extra_exit = false;
        auto check_exits = [&](BasicBlock *block) noexcept {
            if (block == header) { return; }
            block->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (!loop.contains(succ)) { extra_exit = true; }
            });
        };
        for (auto *block : loop.body_blocks) { check_exits(block); }
        if (extra_exit) { return false; }
    }
    // The latch must branch back unconditionally.
    {
        auto *latch_terminator = latch->terminator();
        if (latch_terminator == nullptr || !latch_terminator->isa<BranchInst>() ||
            static_cast<BranchInst *>(latch_terminator)->target_block() != header) {
            return false;
        }
    }
    // The preheader must enter the header unconditionally; validated up front
    // because peeling mutates the CFG and cannot roll back.
    {
        auto *preheader_terminator = preheader->terminator();
        if (preheader_terminator == nullptr ||
            !preheader_terminator->isa<BranchInst>() ||
            static_cast<BranchInst *>(preheader_terminator)->target_block() != header) {
            return false;
        }
    }
    if (options.unroll_pure_only && loop_body_has_side_effect(loop)) { return false; }
    // Body phis must not take incomings from the header: peel clones after
    // the first would keep referencing the original header edge.
    {
        auto invalid_body_phi = false;
        for (auto *block : loop.body_blocks) {
            block->traverse_instructions([&](Instruction *inst) noexcept {
                if (invalid_body_phi || !inst->isa<PhiInst>()) { return; }
                auto *phi = static_cast<PhiInst *>(inst);
                for (auto i = 0u; i < phi->incoming_count(); ++i) {
                    if (phi->incoming(i).block == header) {
                        invalid_body_phi = true;
                        return;
                    }
                }
            });
        }
        if (invalid_body_phi) { return false; }
    }

    // Header non-phi instructions are cloned into every peeled iteration's
    // entry block, mirroring full unrolling.
    luisa::vector<Instruction *> header_scalar_insts;
    {
        auto past_phis = false;
        for (auto *inst : header->instructions()) {
            if (!past_phis && inst->isa<PhiInst>()) { continue; }
            past_phis = true;
            if (inst->is_terminator()) { continue; }
            header_scalar_insts.emplace_back(inst);
        }
    }
    // Header phis: every phi needs one preheader and one latch incoming.
    luisa::vector<PhiInst *> header_phis;
    for (auto *inst : header->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        Value *from_preheader = nullptr;
        Value *from_latch = nullptr;
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block == preheader) { from_preheader = incoming.value; }
            if (incoming.block == latch) { from_latch = incoming.value; }
        }
        if (from_preheader == nullptr || from_latch == nullptr) { return false; }
        header_phis.emplace_back(phi);
    }

    luisa::vector<BasicBlock *> ordered_body;
    if (!collect_body_blocks_in_order(loop, body_entry, ordered_body)) {
        return false;
    }
    if (ordered_body.empty() || ordered_body.back() != latch) { return false; }

    // Everything validated. Clone the body `factor` times.
    luisa::vector<luisa::unordered_map<BasicBlock *, BasicBlock *>> iteration_blocks;
    iteration_blocks.reserve(factor);
    for (auto i = 0u; i < factor; ++i) {
        luisa::unordered_map<BasicBlock *, BasicBlock *> block_map;
        for (auto *block : ordered_body) {
            block_map.emplace(block, def->create_basic_block());
        }
        iteration_blocks.emplace_back(std::move(block_map));
    }
    luisa::unordered_map<const Value *, Value *> phi_values;
    for (auto *phi : header_phis) {
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            if (phi->incoming(i).block == preheader) {
                phi_values.emplace(phi, phi->incoming(i).value);
            }
        }
    }

    XIRBuilder builder;
    BasicBlock *previous_latch_clone = nullptr;
    for (auto i = 0u; i < factor; ++i) {
        UnrollValueResolver resolver;
        for (auto &[from, to] : iteration_blocks[i]) { resolver.emplace(from, to); }
        for (auto &[phi, value] : phi_values) { resolver.emplace(phi, value); }
        {
            auto *entry_clone = iteration_blocks[i][body_entry];
            builder.set_insertion_point(entry_clone);
            for (auto *inst : header_scalar_insts) {
                auto *cloned = inst->clone_with_metadata(builder, resolver);
                resolver.emplace(inst, cloned);
            }
        }
        for (auto *block : ordered_body) {
            auto *clone = iteration_blocks[i][block];
            builder.set_insertion_point(clone);
            for (auto *inst : block->instructions()) {
                auto *cloned = inst->clone_with_metadata(builder, resolver);
                resolver.emplace(inst, cloned);
            }
        }
        luisa::unordered_map<const Value *, Value *> next_phi_values;
        for (auto *phi : header_phis) {
            for (auto idx = 0u; idx < phi->incoming_count(); ++idx) {
                if (phi->incoming(idx).block == latch) {
                    next_phi_values.emplace(
                        phi, resolver.resolve(phi->incoming(idx).value));
                }
            }
        }
        phi_values = std::move(next_phi_values);
        if (previous_latch_clone != nullptr) {
            auto *prev_terminator = previous_latch_clone->terminator();
            static_cast<BranchInst *>(prev_terminator)
                ->set_target_block(iteration_blocks[i][body_entry]);
        }
        previous_latch_clone = iteration_blocks[i][latch];
    }
    // The last peeled latch re-enters the original header; the preheader
    // enters the first peeled iteration.
    {
        auto *last_terminator = previous_latch_clone->terminator();
        static_cast<BranchInst *>(last_terminator)->set_target_block(header);
    }
    {
        auto *preheader_terminator = preheader->terminator();
        static_cast<BranchInst *>(preheader_terminator)
            ->set_target_block(iteration_blocks.front()[body_entry]);
    }
    // Header phis now enter from the last peeled latch with the peeled
    // recurrence values (the induction start is advanced by `factor`).
    for (auto *phi : header_phis) {
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            if (phi->incoming(i).block == preheader) {
                phi->set_incoming(i, phi_values[phi], previous_latch_clone);
            }
        }
    }
    return true;
}

}// namespace

static void run(Function *function, LoopUnrollInfo &info,
                const LoopUnrollOptions &options) noexcept {
    if (function == nullptr || function->definition() == nullptr) { return; }
    auto *def = function->definition();
    if (contains_structured_control_flow(def)) {
        ++info.structured_cfg_error_count;
        LUISA_WARNING_WITH_LOCATION(
            "Loop unroll rejected structured CFG; run destructure_cfg first. "
            "IR was left unchanged.");
        return;
    }
    // Unrolling removes blocks, invalidating every other discovered loop,
    // so loops are re-discovered after each successful unroll. Inner loops
    // come first, which lets a constant-trip outer loop collapse after its
    // inner loop has been expanded.
    for (;;) {
        auto dom_tree = compute_dom_tree(def);
        auto loops = discover_natural_loops(def, dom_tree);
        auto any_unrolled = false;
        for (auto &loop : loops) {
            if (try_full_unroll(def, loop, options)) {
                ++info.unrolled_loop_count;
                any_unrolled = true;
                break;
            }
            if (try_partial_unroll(def, loop, options)) {
                ++info.partially_unrolled_loop_count;
                any_unrolled = true;
                break;
            }
        }
        if (!any_unrolled) { break; }
    }
}

}// namespace detail

LoopUnrollInfo loop_unroll_pass_run_on_function(Function *function,
                                                LoopUnrollOptions options) noexcept {
    LoopUnrollInfo info;
    detail::run(function, info, options);
    return info;
}

LoopUnrollInfo loop_unroll_pass_run_on_module(Module *module,
                                              LoopUnrollOptions options) noexcept {
    LoopUnrollInfo info;
    for (auto *function : module->function_list()) {
        detail::run(function, info, options);
    }
    return info;
}

}// namespace luisa::compute::xir
