#include "restructure_cfg_loop_boundary.h"

#include "restructure_cfg_selection_merge.h"

#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/raster_discard.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/passes/dom_tree.h>

#include <cstdint>
#include <utility>

namespace luisa::compute::xir {
namespace {

template<typename Visitor>
void restructure_cfg_loop_boundary_traverse_structured_successors(
    BasicBlock *block, Visitor &&visit) noexcept {
    if (block == nullptr || !block->is_terminated()) { return; }
    auto *terminator = block->terminator();
    for (auto *operand_use : terminator->operand_uses()) {
        auto *operand = operand_use->value();
        if (operand != nullptr && operand->isa<BasicBlock>()) {
            visit(static_cast<BasicBlock *>(operand));
        }
    }
    if (auto *merge = terminator->control_flow_merge();
        merge != nullptr && merge->merge_block() != nullptr) {
        visit(merge->merge_block());
    }
    if (terminator->isa<LoopInst>()) {
        auto *loop = static_cast<LoopInst *>(terminator);
        if (loop->body_block() != nullptr) {
            visit(loop->body_block());
        }
        if (loop->update_block() != nullptr) {
            visit(loop->update_block());
        }
    }
}

[[nodiscard]] bool restructure_cfg_loop_boundary_has_only_terminator(
    BasicBlock *block) noexcept {
    if (block == nullptr || !block->is_terminated()) {
        return false;
    }
    auto iter = block->instructions().begin();
    return iter != block->instructions().end() &&
           *iter == block->terminator();
}

[[nodiscard]] BasicBlock *restructure_cfg_loop_boundary_trivial_branch_target(
    BasicBlock *block) noexcept {
    if (!restructure_cfg_loop_boundary_has_only_terminator(block) ||
        !block->terminator()->isa<BranchInst>()) {
        return nullptr;
    }
    return static_cast<BranchInst *>(block->terminator())
        ->target_block();
}

[[nodiscard]] BasicBlock *restructure_cfg_loop_boundary_structured_statement_merge(
    Instruction *terminator) noexcept {
    if (terminator == nullptr ||
        (!terminator->isa<LoopInst>() &&
         !terminator->isa<SimpleLoopInst>() &&
         !terminator->isa<SwitchInst>())) {
        return nullptr;
    }
    auto *merge = terminator->control_flow_merge();
    return merge == nullptr ? nullptr : merge->merge_block();
}

// This is intentionally the exact precondition accepted by the rewrite
// helpers in restructure_cfg.cpp. BREAK/CONTINUE terminators are already
// structured and must never be reinterpreted as raw CFG edges here.
[[nodiscard]] bool retargetable_terminator_targets(
    BasicBlock *block, BasicBlock *target) noexcept {
    if (block == nullptr || target == nullptr ||
        !block->is_terminated()) {
        return false;
    }
    auto *terminator = block->terminator();
    switch (terminator->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH:
            return static_cast<BranchInst *>(terminator)
                       ->target_block() == target;
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto *branch = static_cast<
                ConditionalBranchInst *>(terminator);
            return branch->true_block() == target ||
                   branch->false_block() == target;
        }
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::INDEXED_BRANCH: {
            auto *branch = static_cast<
                IndexedBranchTerminatorInstruction *>(terminator);
            if (branch->default_block() == target) { return true; }
            for (auto i = size_t{0u}; i < branch->case_count(); ++i) {
                if (branch->case_block(i) == target) { return true; }
            }
            return false;
        }
        default: return false;
    }
}

}// namespace

class LoopContinueBatchAnalysis::Impl {
private:
    FunctionDefinition *_definition;
    const DomTree &_dominance;
    luisa::unordered_set<BasicBlock *> _function_blocks;
    luisa::unordered_set<BasicBlock *>
        _structured_forwarding_barriers;
    luisa::vector<LoopContinueRewrite> _rewrites;
    LoopContinueBatchStats _stats;

private:
    [[nodiscard]] bool _is_live(
        BasicBlock *block) const noexcept {
        return block != nullptr &&
               _function_blocks.contains(block);
    }

    void _append(size_t site_index,
                 BasicBlock *block,
                 BasicBlock *from,
                 BasicBlock *target,
                 LoopContinueRewriteKind kind) noexcept {
        // Preserve the immutable-version proof obligation in the plan itself:
        // an action exists iff its raw source edge exists in the CFG version
        // seen by this analysis. Application may reject an edge consumed by an
        // earlier action, but can never accept an edge created by one.
        if (!retargetable_terminator_targets(block, from)) { return; }
        _rewrites.emplace_back(LoopContinueRewrite{
            .block = block,
            .from = from,
            .target = target,
            .site_index = site_index,
            .kind = kind});
    }

public:
    Impl(FunctionDefinition *definition,
         const DomTree &dominance) noexcept
        : _definition{definition}, _dominance{dominance} {
        if (_definition == nullptr) { return; }
        for (auto *block : _definition->basic_blocks()) {
            _function_blocks.emplace(block);
            if (!block->is_terminated()) { continue; }
            auto *terminator = block->terminator();
            if (auto *merge = terminator->control_flow_merge();
                merge != nullptr && merge->merge_block() != nullptr) {
                _structured_forwarding_barriers.emplace(
                    merge->merge_block());
            }
            if (terminator->isa<LoopInst>()) {
                auto *loop = static_cast<LoopInst *>(terminator);
                _structured_forwarding_barriers.emplace(
                    loop->prepare_block());
                _structured_forwarding_barriers.emplace(
                    loop->body_block());
                _structured_forwarding_barriers.emplace(
                    loop->update_block());
            } else if (terminator->isa<SimpleLoopInst>()) {
                _structured_forwarding_barriers.emplace(
                    static_cast<SimpleLoopInst *>(terminator)
                        ->body_block());
            }
        }
    }

    void plan(size_t site_index,
              BasicBlock *loop_entry,
              BasicBlock *body,
              BasicBlock *continue_target,
              BasicBlock *merge) noexcept {
        if (!_is_live(loop_entry) || !_is_live(body) ||
            !_is_live(continue_target) || !_is_live(merge)) {
            return;
        }
        auto allow_loop_entry_in_region =
            loop_entry == body && body == continue_target;
        luisa::unordered_set<BasicBlock *> loop_region;
        luisa::vector<std::pair<BasicBlock *, BasicBlock *>>
            nested_switch_scopes;
        luisa::vector<BasicBlock *> work;
        auto enqueue = [&](BasicBlock *block) noexcept {
            if (!_is_live(block) || block == merge) { return; }
            // Reachability alone is not lexical loop membership. The entry
            // dominance predicate is queried only while this CFG version is
            // immutable; all resulting rewrites are applied later.
            if (!_dominance.contains(loop_entry) ||
                !_dominance.contains(block) ||
                !_dominance.dominates(loop_entry, block)) {
                return;
            }
            if (!allow_loop_entry_in_region &&
                block == loop_entry) {
                return;
            }
            if (loop_region.emplace(block).second) {
                work.emplace_back(block);
                ++_stats.region_block_visit_count;
            }
        };
        enqueue(body);
        enqueue(continue_target);
        while (!work.empty()) {
            auto *block = work.back();
            work.pop_back();
            if (!_is_live(block) ||
                !block->is_terminated()) {
                continue;
            }
            auto *terminator = block->terminator();
            if (terminator->isa<LoopInst>() ||
                terminator->isa<SimpleLoopInst>()) {
                if (auto *nested_merge =
                        restructure_cfg_loop_boundary_structured_statement_merge(terminator);
                    nested_merge != nullptr &&
                    nested_merge != merge) {
                    enqueue(nested_merge);
                }
                continue;
            }
            // A nested loop is opaque because it installs both a nearer
            // break and a nearer continue scope. A Switch is different: it
            // installs only a nearer break scope, so Continue(U) to this
            // enclosing loop remains the canonical spelling in every case
            // region. Contracting a Switch directly to its merge therefore
            // loses exactly the raw case-to-U edges this analysis must
            // normalize.
            //
            // Walk all Switch successors here. A raw edge to this loop's
            // merge may temporarily become Break(M); the subsequent lexical
            // context pass proves whether a nearer Switch is active and, in
            // that case, converts only that non-local Break back to an
            // ordinary Branch. Thus the two independent scope dimensions are
            // handled without guessing from reachability.
            if (terminator->isa<SwitchInst>()) {
                auto *switch_merge =
                    restructure_cfg_loop_boundary_structured_statement_merge(
                        terminator);
                if (switch_merge != nullptr && switch_merge != merge) {
                    nested_switch_scopes.emplace_back(
                        block, switch_merge);
                }
            }
            restructure_cfg_loop_boundary_traverse_structured_successors(
                block, [&](BasicBlock *successor) noexcept {
                    ++_stats.region_edge_visit_count;
                    if (!_is_live(successor) ||
                        successor == merge) {
                        return;
                    }
                    if (!allow_loop_entry_in_region &&
                        successor == loop_entry) {
                        return;
                    }
                    enqueue(successor);
                });
        }

        // Break denotes the nearest lexical Loop-or-Switch scope, whereas
        // Continue denotes the nearest Loop scope. Compute, in one sparse
        // dominator-tree event walk, the blocks for which a nested Switch is
        // active. Outer-loop break edges at those blocks must remain ordinary
        // Branch edges; converting them to Break would name the nearer Switch
        // and the context canonicalizer would immediately undo the rewrite.
        //
        // A Switch activates at its header and is suspended for the complete
        // dominator subtree rooted at its merge. This is exactly the predicate
        //
        //   switch.header dom B && !(switch.merge dom B),
        //
        // evaluated for all B in O(V + S) rather than O(V*S).
        luisa::unordered_set<BasicBlock *>
            nested_switch_break_forbidden;
        if (!nested_switch_scopes.empty() &&
            _dominance.root() != nullptr) {
            luisa::unordered_map<BasicBlock *, size_t>
                switch_starts;
            luisa::unordered_map<BasicBlock *, size_t>
                switch_ends;
            for (auto [header, switch_merge] :
                 nested_switch_scopes) {
                ++switch_starts[header];
                ++switch_ends[switch_merge];
            }
            struct DomFrame {
                const DomTreeNode *node{nullptr};
                size_t next_child{0u};
                size_t activated{0u};
                size_t suspended{0u};
            };
            luisa::vector<DomFrame> dom_work;
            dom_work.emplace_back(
                DomFrame{.node = _dominance.root()});
            auto active_switch_count = size_t{0u};
            while (!dom_work.empty()) {
                auto &frame = dom_work.back();
                if (frame.next_child == 0u) {
                    auto *block = frame.node->block();
                    if (auto iter = switch_ends.find(block);
                        iter != switch_ends.end()) {
                        frame.suspended =
                            iter->second < active_switch_count ?
                                iter->second :
                                active_switch_count;
                        active_switch_count -= frame.suspended;
                    }
                    if (auto iter = switch_starts.find(block);
                        iter != switch_starts.end()) {
                        frame.activated = iter->second;
                        active_switch_count += frame.activated;
                    }
                    if (active_switch_count != 0u &&
                        loop_region.contains(block)) {
                        nested_switch_break_forbidden.emplace(
                            block);
                    }
                }
                auto children = frame.node->children();
                if (frame.next_child < children.size()) {
                    dom_work.emplace_back(DomFrame{
                        .node = children[frame.next_child++]});
                    continue;
                }
                active_switch_count -= frame.activated;
                active_switch_count += frame.suspended;
                dom_work.pop_back();
            }
        }

        // Boundary equivalence is the least reverse closure of a boundary
        // under payload-free Branch edges, with every declared structured
        // role as a quotient barrier. Planning from that closure, rather than
        // testing only literal target identity, handles the canonical latch
        // proxy left by natural-loop recovery:
        //
        //   arm -> L; L -> loop.entry
        //
        // Every path through L has exactly the same executable effect as a
        // Continue, so both incoming edges can be normalized without cloning
        // or a target-state dispatch. A merge/update/body role may itself be
        // normalized, but the reverse closure never crosses it: doing so would
        // make the role unreachable and replace lexical structure with an
        // OpUnreachable block. The immutable region and finite visited set
        // give O(V + E) work and a termination proof.
        auto plan_boundary_predecessors =
            [&](BasicBlock *boundary,
                BasicBlock *replacement_target,
                LoopContinueRewriteKind kind,
                BasicBlock *excluded_forwarder) noexcept {
                if (!_is_live(boundary)) { return; }
                luisa::unordered_set<BasicBlock *> aliases;
                luisa::vector<BasicBlock *> alias_work;
                aliases.emplace(boundary);
                alias_work.emplace_back(boundary);
                while (!alias_work.empty()) {
                    auto *alias = alias_work.back();
                    alias_work.pop_back();
                    alias->traverse_predecessors(
                        false, [&](BasicBlock *predecessor) noexcept {
                            if (!loop_region.contains(predecessor) ||
                                predecessor == excluded_forwarder) {
                                return;
                            }
                            if (kind ==
                                    LoopContinueRewriteKind::BREAK &&
                                nested_switch_break_forbidden.contains(
                                    predecessor)) {
                                return;
                            }
                            _append(site_index, predecessor, alias,
                                    replacement_target, kind);
                            if (restructure_cfg_loop_boundary_has_only_terminator(
                                    predecessor) &&
                                predecessor->terminator()->isa<BranchInst>() &&
                                static_cast<BranchInst *>(
                                    predecessor->terminator())
                                        ->target_block() == alias &&
                                !_structured_forwarding_barriers.contains(
                                    predecessor) &&
                                aliases.emplace(predecessor).second) {
                                alias_work.emplace_back(predecessor);
                            }
                        });
                }
            };
        plan_boundary_predecessors(
            continue_target, continue_target,
            LoopContinueRewriteKind::CONTINUE, nullptr);
        if (loop_entry != continue_target) {
            // The update-to-prepare edge is the LoopInst backedge. Rewriting
            // it as Continue(update) would create a self-cycle, so it is the
            // unique excluded predecessor in the prepare-boundary closure.
            plan_boundary_predecessors(
                loop_entry, continue_target,
                LoopContinueRewriteKind::CONTINUE,
                continue_target);
        }
        plan_boundary_predecessors(
            merge, merge, LoopContinueRewriteKind::BREAK,
            nullptr);
        if (auto *merge_successor =
                restructure_cfg_loop_boundary_trivial_branch_target(merge);
            merge_successor != nullptr) {
            // Destructured CFGs may have contracted the declared merge and
            // branch directly to its forwarding successor. Preserve the
            // existing quotient rule while excluding the merge itself.
            plan_boundary_predecessors(
                merge_successor, merge,
                LoopContinueRewriteKind::BREAK, merge);
        }
    }

    [[nodiscard]] size_t rewrite_count() const noexcept {
        return _rewrites.size();
    }

    [[nodiscard]] const LoopContinueRewrite &rewrite(
        size_t index) const noexcept {
        return _rewrites[index];
    }

    [[nodiscard]] const LoopContinueBatchStats &stats()
        const noexcept {
        return _stats;
    }
};

// The classification is a finite forward reachability problem over BREAK,
// CONTINUE, and INVALID facts. The reverse graph is immutable and stored as
// dense CSR. Each loop solve is restricted to a successor-closed induced
// region, except for its explicitly seeded boundaries, so its least fixed
// point equals the restriction of the whole-CFG solution.
class LoopBoundaryPathDataflow::Impl {
private:
    static constexpr auto break_bit = uint8_t{1u};
    static constexpr auto continue_bit = uint8_t{2u};
    static constexpr auto invalid_bit = uint8_t{4u};

    luisa::vector<BasicBlock *> _blocks;
    luisa::unordered_map<BasicBlock *, size_t> _block_ids;
    luisa::vector<size_t> _reverse_edge_offsets;
    luisa::vector<size_t> _reverse_edge_sources;
    luisa::vector<BasicBlock *> _canonical_targets;
    luisa::vector<uint8_t> _facts;
    luisa::vector<uint8_t> _terminal;
    luisa::vector<uint32_t> _visit_epochs;
    luisa::vector<uint32_t> _fact_epochs;
    uint32_t _visit_epoch{0u};
    luisa::vector<size_t> _work;
    luisa::vector<size_t> _region;
    luisa::vector<size_t> _active;
    size_t _edge_visit_count{0u};

private:
    void _begin_visit() noexcept {
        ++_visit_epoch;
        if (_visit_epoch == 0u) {
            std::fill(
                _visit_epochs.begin(),
                _visit_epochs.end(), 0u);
            std::fill(
                _fact_epochs.begin(),
                _fact_epochs.end(), 0u);
            _visit_epoch = 1u;
        }
        _work.clear();
        _region.clear();
        _active.clear();
        _edge_visit_count = 0u;
    }

public:
    explicit Impl(FunctionDefinition *definition) noexcept {
        if (definition == nullptr) { return; }
        for (auto *block : definition->basic_blocks()) {
            auto id = _blocks.size();
            _blocks.emplace_back(block);
            _block_ids.emplace(block, id);
        }
        _canonical_targets.reserve(_blocks.size());
        for (auto *block : _blocks) {
            _canonical_targets.emplace_back(
                detail::canonical_trivial_branch_chain_target(
                    block));
        }

        luisa::vector<std::pair<size_t, size_t>> reverse_edges;
        auto add_edge = [&](size_t source,
                            BasicBlock *target) noexcept {
            if (auto iter = _block_ids.find(target);
                iter != _block_ids.end()) {
                reverse_edges.emplace_back(
                    iter->second, source);
            }
        };
        for (auto source = size_t{0u};
             source < _blocks.size(); ++source) {
            auto *block = _blocks[source];
            if (!block->is_terminated()) { continue; }
            auto *terminator = block->terminator();
            if (terminator->isa<BreakInst>() ||
                terminator->isa<ContinueInst>() ||
                terminator->isa<ReturnInst>() ||
                terminator->isa<UnreachableInst>() ||
                terminator->isa<RasterDiscardInst>()) {
                continue;
            }
            if (terminator->isa<LoopInst>() ||
                terminator->isa<SimpleLoopInst>()) {
                if (auto *merge =
                        terminator->control_flow_merge();
                    merge != nullptr) {
                    add_edge(source, merge->merge_block());
                }
                continue;
            }
            restructure_cfg_loop_boundary_traverse_structured_successors(
                block, [&](BasicBlock *successor) noexcept {
                    add_edge(source, successor);
                });
        }
        _reverse_edge_offsets.resize(
            _blocks.size() + 1u, 0u);
        for (auto [target, source] : reverse_edges) {
            static_cast<void>(source);
            ++_reverse_edge_offsets[target + 1u];
        }
        for (auto i = size_t{1u};
             i < _reverse_edge_offsets.size(); ++i) {
            _reverse_edge_offsets[i] +=
                _reverse_edge_offsets[i - 1u];
        }
        _reverse_edge_sources.resize(reverse_edges.size());
        auto cursors = _reverse_edge_offsets;
        for (auto [target, source] : reverse_edges) {
            _reverse_edge_sources[cursors[target]++] = source;
        }
        _facts.resize(_blocks.size(), uint8_t{0u});
        _terminal.resize(_blocks.size(), uint8_t{0u});
        _visit_epochs.resize(_blocks.size(), 0u);
        _fact_epochs.resize(_blocks.size(), 0u);
        _work.reserve(_blocks.size());
        _region.reserve(_blocks.size());
        _active.reserve(_blocks.size());
    }

    void evaluate(BasicBlock *body,
                  BasicBlock *continue_target,
                  BasicBlock *loop_entry,
                  BasicBlock *merge) noexcept {
        _begin_visit();
        auto enqueue_region = [&](BasicBlock *block,
                                  bool allow_loop_entry) noexcept {
            if (block == nullptr || block == merge ||
                (!allow_loop_entry && block == loop_entry)) {
                return;
            }
            auto iter = _block_ids.find(block);
            if (iter == _block_ids.end()) { return; }
            auto id = iter->second;
            if (_visit_epochs[id] == _visit_epoch) { return; }
            _visit_epochs[id] = _visit_epoch;
            _work.emplace_back(id);
        };
        enqueue_region(body, true);
        while (!_work.empty()) {
            auto id = _work.back();
            _work.pop_back();
            _region.emplace_back(id);
            restructure_cfg_loop_boundary_traverse_structured_successors(
                _blocks[id], [&](BasicBlock *successor) noexcept {
                    enqueue_region(successor, false);
                });
        }

        auto activate_id = [&](size_t id) noexcept {
            if (_fact_epochs[id] == _visit_epoch) { return; }
            _fact_epochs[id] = _visit_epoch;
            _facts[id] = uint8_t{0u};
            _terminal[id] = uint8_t{0u};
            _active.emplace_back(id);
        };
        for (auto id : _region) { activate_id(id); }
        auto activate_block = [&](BasicBlock *block) noexcept {
            if (auto iter = _block_ids.find(block);
                iter != _block_ids.end()) {
                activate_id(iter->second);
            }
        };
        activate_block(continue_target);
        activate_block(loop_entry);
        activate_block(merge);

        auto *canonical_merge =
            detail::canonical_trivial_branch_chain_target(merge);
        for (auto id : _active) {
            auto *block = _blocks[id];
            auto fact = uint8_t{0u};
            if (block == merge ||
                _canonical_targets[id] == canonical_merge) {
                fact = break_bit;
            } else if (block == continue_target ||
                       block == loop_entry) {
                fact = continue_bit;
            } else if (!block->is_terminated()) {
                fact = invalid_bit;
            } else {
                auto *terminator = block->terminator();
                if (terminator->isa<BreakInst>()) {
                    fact = static_cast<BreakInst *>(terminator)
                                       ->target_block() == merge ?
                               break_bit :
                               invalid_bit;
                } else if (terminator->isa<ContinueInst>()) {
                    auto *target =
                        static_cast<ContinueInst *>(terminator)
                            ->target_block();
                    fact = target == continue_target ||
                                   target == loop_entry ?
                               continue_bit :
                               invalid_bit;
                } else if (terminator->isa<ReturnInst>() ||
                           terminator->isa<UnreachableInst>() ||
                           terminator->isa<RasterDiscardInst>()) {
                    fact = invalid_bit;
                } else if (terminator->isa<LoopInst>() ||
                           terminator->isa<SimpleLoopInst>()) {
                    auto *control_flow_merge =
                        terminator->control_flow_merge();
                    if (control_flow_merge == nullptr ||
                        control_flow_merge->merge_block() ==
                            nullptr) {
                        fact = invalid_bit;
                    }
                }
            }
            if (fact != 0u) {
                _facts[id] = fact;
                _terminal[id] = uint8_t{1u};
                _work.emplace_back(id);
            }
        }
        for (auto cursor = size_t{0u};
             cursor < _work.size(); ++cursor) {
            auto node = _work[cursor];
            auto fact = _facts[node];
            for (auto edge = _reverse_edge_offsets[node];
                 edge < _reverse_edge_offsets[node + 1u];
                 ++edge) {
                ++_edge_visit_count;
                auto predecessor =
                    _reverse_edge_sources[edge];
                if (_fact_epochs[predecessor] !=
                    _visit_epoch) {
                    continue;
                }
                if (_terminal[predecessor] != 0u) { continue; }
                auto combined = static_cast<uint8_t>(
                    _facts[predecessor] | fact);
                if (combined != _facts[predecessor]) {
                    _facts[predecessor] = combined;
                    _work.emplace_back(predecessor);
                }
            }
        }
    }

    [[nodiscard]] LoopBoundaryTargetKind classify(
        BasicBlock *target) const noexcept {
        auto iter = _block_ids.find(target);
        if (iter == _block_ids.end() ||
            _fact_epochs[iter->second] != _visit_epoch) {
            return LoopBoundaryTargetKind::NONE;
        }
        auto fact = _facts[iter->second];
        if ((fact & invalid_bit) != 0u) {
            return LoopBoundaryTargetKind::NONE;
        }
        switch (fact & (break_bit | continue_bit)) {
            case break_bit:
                return LoopBoundaryTargetKind::BREAK;
            case continue_bit:
                return LoopBoundaryTargetKind::CONTINUE;
            case break_bit | continue_bit:
                return LoopBoundaryTargetKind::MIXED;
            default:
                return LoopBoundaryTargetKind::NONE;
        }
    }

    [[nodiscard]] size_t region_size() const noexcept {
        return _region.size();
    }

    [[nodiscard]] BasicBlock *region_block(
        size_t index) const noexcept {
        return _blocks[_region[index]];
    }

    [[nodiscard]] size_t active_block_count() const noexcept {
        return _active.size();
    }

    [[nodiscard]] size_t edge_visit_count() const noexcept {
        return _edge_visit_count;
    }
};

LoopBoundaryPathDataflow::LoopBoundaryPathDataflow(
    FunctionDefinition *definition) noexcept
    : _impl{luisa::make_unique<Impl>(definition)} {}

LoopBoundaryPathDataflow::~LoopBoundaryPathDataflow() noexcept = default;

LoopBoundaryPathDataflow::LoopBoundaryPathDataflow(
    LoopBoundaryPathDataflow &&) noexcept = default;

LoopBoundaryPathDataflow &
LoopBoundaryPathDataflow::operator=(
    LoopBoundaryPathDataflow &&) noexcept = default;

void LoopBoundaryPathDataflow::evaluate(
    BasicBlock *body,
    BasicBlock *continue_target,
    BasicBlock *loop_entry,
    BasicBlock *merge) noexcept {
    _impl->evaluate(
        body, continue_target, loop_entry, merge);
}

LoopBoundaryTargetKind LoopBoundaryPathDataflow::classify(
    BasicBlock *target) const noexcept {
    return _impl->classify(target);
}

size_t LoopBoundaryPathDataflow::region_size() const noexcept {
    return _impl->region_size();
}

BasicBlock *LoopBoundaryPathDataflow::region_block(
    size_t index) const noexcept {
    return _impl->region_block(index);
}

size_t LoopBoundaryPathDataflow::active_block_count() const noexcept {
    return _impl->active_block_count();
}

size_t LoopBoundaryPathDataflow::edge_visit_count() const noexcept {
    return _impl->edge_visit_count();
}

LoopContinueBatchAnalysis::LoopContinueBatchAnalysis(
    FunctionDefinition *definition,
    const DomTree &dominance) noexcept
    : _impl{luisa::make_unique<Impl>(
          definition, dominance)} {}

LoopContinueBatchAnalysis::~LoopContinueBatchAnalysis() noexcept =
    default;

LoopContinueBatchAnalysis::LoopContinueBatchAnalysis(
    LoopContinueBatchAnalysis &&) noexcept = default;

LoopContinueBatchAnalysis &
LoopContinueBatchAnalysis::operator=(
    LoopContinueBatchAnalysis &&) noexcept = default;

void LoopContinueBatchAnalysis::plan(
    size_t site_index,
    BasicBlock *loop_entry,
    BasicBlock *body,
    BasicBlock *continue_target,
    BasicBlock *merge) noexcept {
    _impl->plan(site_index, loop_entry, body,
                continue_target, merge);
}

size_t LoopContinueBatchAnalysis::rewrite_count()
    const noexcept {
    return _impl->rewrite_count();
}

const LoopContinueRewrite &LoopContinueBatchAnalysis::rewrite(
    size_t index) const noexcept {
    return _impl->rewrite(index);
}

const LoopContinueBatchStats &LoopContinueBatchAnalysis::stats()
    const noexcept {
    return _impl->stats();
}

}// namespace luisa::compute::xir
