#include "restructure_cfg_selection_merge.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/switch.h>

#include <cstdint>
#include <limits>
#include <utility>

namespace luisa::compute::xir::detail {
namespace {

constexpr auto invalid_index = std::numeric_limits<size_t>::max();

[[nodiscard]] BasicBlock *trivial_branch_target(
    BasicBlock *block) noexcept {
    if (block == nullptr || !block->is_terminated()) {
        return nullptr;
    }
    auto iter = block->instructions().begin();
    if (iter == block->instructions().end() ||
        *iter != block->terminator() ||
        !block->terminator()->isa<BranchInst>()) {
        return nullptr;
    }
    return static_cast<BranchInst *>(
               block->terminator())
        ->target_block();
}

[[nodiscard]] BasicBlock *structured_statement_merge(
    BasicBlock *block) noexcept {
    if (block == nullptr || !block->is_terminated()) {
        return nullptr;
    }
    auto *terminator = block->terminator();
    if (!terminator->isa<IfInst>() &&
        !terminator->isa<SwitchInst>() &&
        !terminator->isa<LoopInst>() &&
        !terminator->isa<SimpleLoopInst>()) {
        return nullptr;
    }
    auto *merge = terminator->control_flow_merge();
    return merge == nullptr ? nullptr : merge->merge_block();
}

struct StructuredLoop {
    BasicBlock *header{nullptr};
    luisa::vector<BasicBlock *> exits;
};

[[nodiscard]] luisa::vector<StructuredLoop>
collect_structured_loops(FunctionDefinition *definition) noexcept {
    luisa::vector<StructuredLoop> loops;
    if (definition == nullptr) { return loops; }
    definition->traverse_basic_blocks(
        [&](BasicBlock *block) noexcept {
            if (!block->is_terminated()) { return; }
            auto *terminator = block->terminator();
            StructuredLoop loop{.header = block};
            if (terminator->isa<LoopInst>()) {
                auto *inst = static_cast<LoopInst *>(terminator);
                if (auto *prepare = inst->prepare_block();
                    prepare != nullptr) {
                    loop.exits.emplace_back(prepare);
                }
                if (auto *update = inst->update_block();
                    update != nullptr) {
                    loop.exits.emplace_back(update);
                }
                if (auto *merge = inst->merge_block();
                    merge != nullptr) {
                    loop.exits.emplace_back(merge);
                }
            } else if (terminator->isa<SimpleLoopInst>()) {
                auto *inst =
                    static_cast<SimpleLoopInst *>(terminator);
                if (auto *body = inst->body_block();
                    body != nullptr) {
                    loop.exits.emplace_back(body);
                }
                if (auto *merge = inst->merge_block();
                    merge != nullptr) {
                    loop.exits.emplace_back(merge);
                }
            } else {
                return;
            }
            loops.emplace_back(std::move(loop));
        });
    return loops;
}

}// namespace

BasicBlock *canonical_trivial_branch_chain_target(
    BasicBlock *target) noexcept {
    if (target == nullptr) { return nullptr; }

    // Floyd first decides whether the functional branch graph terminates or
    // cycles. A second walk returns exactly the same canonical node as a
    // visited set: the terminal node, or the first cycle entry from `target`.
    auto *slow = target;
    auto *fast = target;
    for (;;) {
        slow = trivial_branch_target(slow);
        fast = trivial_branch_target(fast);
        if (slow == nullptr || fast == nullptr) { break; }
        fast = trivial_branch_target(fast);
        if (fast == nullptr || slow == fast) { break; }
    }
    if (slow != nullptr && slow == fast) {
        auto *entry = target;
        while (entry != slow) {
            entry = trivial_branch_target(entry);
            slow = trivial_branch_target(slow);
        }
        return entry;
    }
    auto *terminal = target;
    while (auto *next = trivial_branch_target(terminal)) {
        terminal = next;
    }
    return terminal;
}

class SelectionMergeBatchAnalysis::Impl {
private:
    struct LoopContext {
        size_t parent{invalid_index};
        luisa::vector<BasicBlock *> exits;
    };

    struct MergeScore {
        BasicBlock *block{nullptr};
        size_t block_id{invalid_index};
        size_t support{0u};
        size_t max_distance{invalid_index};
        size_t total_distance{invalid_index};
    };

    FunctionDefinition *_definition;
    const DomTree &_dominance;
    luisa::vector<BasicBlock *> _blocks;
    luisa::unordered_map<BasicBlock *, size_t> _block_ids;
    luisa::vector<LoopContext> _loop_contexts;
    luisa::vector<size_t> _block_loop_contexts;

    luisa::vector<uint32_t> _boundary_epochs;
    luisa::vector<uint32_t> _aggregate_epochs;
    luisa::vector<uint32_t> _visit_epochs;
    luisa::vector<size_t> _distances;
    luisa::vector<size_t> _support;
    luisa::vector<size_t> _minimum_distances;
    luisa::vector<size_t> _maximum_distances;
    luisa::vector<size_t> _total_distances;
    uint32_t _query_epoch{0u};
    uint32_t _visit_epoch{0u};
    luisa::vector<size_t> _queue;
    luisa::vector<size_t> _entry_visits;
    luisa::vector<size_t> _query_aggregates;
    SelectionMergeBatchStats _stats;

private:
    void _append_block(BasicBlock *block) noexcept {
        if (block == nullptr || _block_ids.contains(block)) {
            return;
        }
        auto id = _blocks.size();
        _blocks.emplace_back(block);
        _block_ids.emplace(block, id);
        _block_loop_contexts.emplace_back(invalid_index);
        _boundary_epochs.emplace_back(0u);
        _aggregate_epochs.emplace_back(0u);
        _visit_epochs.emplace_back(0u);
        _distances.emplace_back(0u);
        _support.emplace_back(0u);
        _minimum_distances.emplace_back(invalid_index);
        _maximum_distances.emplace_back(0u);
        _total_distances.emplace_back(0u);
    }

    void _begin_query() noexcept {
        ++_query_epoch;
        if (_query_epoch == 0u) {
            std::fill(
                _boundary_epochs.begin(),
                _boundary_epochs.end(), 0u);
            std::fill(
                _aggregate_epochs.begin(),
                _aggregate_epochs.end(), 0u);
            _query_epoch = 1u;
        }
        _query_aggregates.clear();
        ++_stats.query_count;
    }

    void _begin_entry() noexcept {
        ++_visit_epoch;
        if (_visit_epoch == 0u) {
            std::fill(
                _visit_epochs.begin(),
                _visit_epochs.end(), 0u);
            _visit_epoch = 1u;
        }
        _queue.clear();
        _entry_visits.clear();
    }

    [[nodiscard]] bool _is_boundary(
        BasicBlock *block) const noexcept {
        if (auto iter = _block_ids.find(block);
            iter != _block_ids.end()) {
            return _boundary_epochs[iter->second] ==
                   _query_epoch;
        }
        return false;
    }

    void _mark_boundaries(BasicBlock *header) noexcept {
        auto header_iter = _block_ids.find(header);
        if (header_iter == _block_ids.end()) { return; }
        auto context =
            _block_loop_contexts[header_iter->second];
        while (context != invalid_index) {
            for (auto *exit : _loop_contexts[context].exits) {
                if (auto iter = _block_ids.find(exit);
                    iter != _block_ids.end()) {
                    _boundary_epochs[iter->second] =
                        _query_epoch;
                }
            }
            context = _loop_contexts[context].parent;
        }
    }

    [[nodiscard]] bool _contains(
        BasicBlock *block,
        const luisa::unordered_map<BasicBlock *, BasicBlock *> *
            anchors) const noexcept {
        return _dominance.contains(block) ||
               (anchors != nullptr && anchors->contains(block));
    }

    [[nodiscard]] bool _header_dominates(
        BasicBlock *header,
        BasicBlock *block,
        const luisa::unordered_map<BasicBlock *, BasicBlock *> *
            anchors) const noexcept {
        if (_dominance.contains(block)) {
            return _dominance.dominates(header, block);
        }
        if (anchors != nullptr) {
            if (auto iter = anchors->find(block);
                iter != anchors->end()) {
                return _dominance.dominates(
                    header, iter->second);
            }
        }
        return false;
    }

    void _aggregate_entry() noexcept {
        for (auto id : _entry_visits) {
            auto distance = _distances[id];
            if (_aggregate_epochs[id] != _query_epoch) {
                _aggregate_epochs[id] = _query_epoch;
                _query_aggregates.emplace_back(id);
                _support[id] = 0u;
                _minimum_distances[id] = invalid_index;
                _maximum_distances[id] = 0u;
                _total_distances[id] = 0u;
            }
            ++_support[id];
            _minimum_distances[id] = std::min(
                _minimum_distances[id], distance);
            _maximum_distances[id] = std::max(
                _maximum_distances[id], distance);
            _total_distances[id] += distance;
        }
    }

    [[nodiscard]] bool _has_aggregate(size_t id) const noexcept {
        return _aggregate_epochs[id] == _query_epoch;
    }

    static void _consider(
        MergeScore &score,
        BasicBlock *candidate,
        size_t candidate_id,
        size_t support,
        size_t max_distance,
        size_t total_distance) noexcept {
        if (support > score.support ||
            (support == score.support &&
             max_distance < score.max_distance) ||
            (support == score.support &&
             max_distance == score.max_distance &&
             total_distance < score.total_distance) ||
            (support == score.support &&
             max_distance == score.max_distance &&
             total_distance == score.total_distance &&
             candidate_id < score.block_id)) {
            score = {
                candidate, candidate_id, support,
                max_distance, total_distance};
        }
    }

public:
    Impl(FunctionDefinition *definition,
         const DomTree &dominance) noexcept
        : _definition{definition}, _dominance{dominance} {
        if (_definition == nullptr) { return; }
        for (auto *block : _definition->basic_blocks()) {
            _append_block(block);
        }

        auto loops = collect_structured_loops(_definition);
        luisa::unordered_map<BasicBlock *, size_t>
            loop_indices;
        loop_indices.reserve(loops.size());
        for (auto i = size_t{0u}; i < loops.size(); ++i) {
            loop_indices.emplace(loops[i].header, i);
        }

        struct DomWalkFrame {
            const DomTreeNode *node{nullptr};
            size_t loop_context{invalid_index};
            size_t next_child{0u};
            bool entered{false};
        };
        luisa::vector<DomWalkFrame> walk;
        if (_dominance.root() != nullptr) {
            walk.emplace_back(
                DomWalkFrame{.node = _dominance.root()});
        }
        while (!walk.empty()) {
            auto &frame = walk.back();
            if (!frame.entered) {
                frame.entered = true;
                auto *block = frame.node->block();
                if (auto iter = loop_indices.find(block);
                    iter != loop_indices.end()) {
                    auto context = _loop_contexts.size();
                    _loop_contexts.emplace_back(LoopContext{
                        .parent = frame.loop_context,
                        .exits = loops[iter->second].exits});
                    frame.loop_context = context;
                }
                if (auto iter = _block_ids.find(block);
                    iter != _block_ids.end()) {
                    _block_loop_contexts[iter->second] =
                        frame.loop_context;
                }
            }
            auto children = frame.node->children();
            if (frame.next_child < children.size()) {
                auto *child = children[frame.next_child++];
                walk.emplace_back(DomWalkFrame{
                    .node = child,
                    .loop_context = frame.loop_context});
            } else {
                walk.pop_back();
            }
        }
        _stats.loop_context_count = _loop_contexts.size();
    }

    void register_overlay_block(BasicBlock *block) noexcept {
        _append_block(block);
    }

    [[nodiscard]] BasicBlock *infer(
        BasicBlock *header,
        luisa::span<BasicBlock *const> entries,
        const luisa::unordered_map<BasicBlock *, BasicBlock *> *
            anchors) noexcept {
        _begin_query();
        if (_definition == nullptr || header == nullptr ||
            entries.empty() || !_dominance.contains(header)) {
            return nullptr;
        }
        _mark_boundaries(header);

        for (auto *entry : entries) {
            _begin_entry();
            if (entry == nullptr || entry == header ||
                _is_boundary(entry) ||
                !_contains(entry, anchors) ||
                !_header_dominates(header, entry, anchors)) {
                continue;
            }
            auto entry_iter = _block_ids.find(entry);
            if (entry_iter == _block_ids.end()) { continue; }
            auto entry_id = entry_iter->second;
            _visit_epochs[entry_id] = _visit_epoch;
            _distances[entry_id] = 0u;
            _queue.emplace_back(entry_id);
            _entry_visits.emplace_back(entry_id);
            ++_stats.block_visit_count;

            for (auto cursor = size_t{0u};
                 cursor < _queue.size(); ++cursor) {
                auto block_id = _queue[cursor];
                auto *block = _blocks[block_id];
                if (!block->is_terminated()) { continue; }
                auto next_distance =
                    _distances[block_id] + 1u;
                block->traverse_successors(
                    false, [&](BasicBlock *successor) noexcept {
                        ++_stats.edge_visit_count;
                        if (successor == nullptr ||
                            successor == header ||
                            _is_boundary(successor) ||
                            !_contains(successor, anchors)) {
                            return;
                        }
                        auto iter = _block_ids.find(successor);
                        LUISA_DEBUG_ASSERT(
                            iter != _block_ids.end(),
                            "Selection-merge batch encountered an "
                            "unregistered CFG block.");
                        if (iter == _block_ids.end()) { return; }
                        auto successor_id = iter->second;
                        if (_visit_epochs[successor_id] !=
                            _visit_epoch) {
                            _visit_epochs[successor_id] =
                                _visit_epoch;
                            _distances[successor_id] =
                                next_distance;
                            _entry_visits.emplace_back(
                                successor_id);
                            ++_stats.block_visit_count;
                            // Match the historical query exactly: the first
                            // non-dominated successor remains a merge
                            // candidate, but traversal does not continue
                            // through it.
                            if (_header_dominates(
                                    header, successor,
                                    anchors)) {
                                _queue.emplace_back(successor_id);
                            }
                        } else if (
                            next_distance <
                            _distances[successor_id]) {
                            _distances[successor_id] =
                                next_distance;
                        }
                    });
            }
            _aggregate_entry();
        }

        MergeScore best;
        MergeScore boundary_proxy_best;
        auto required_support =
            std::min<size_t>(2u, entries.size());
        // `_query_aggregates` is exactly the support of `_has_aggregate` for
        // this epoch. Enumerating it is therefore equivalent to scanning all
        // block IDs and rejecting the complement. The explicit dense-ID
        // tie-break below preserves the historical first-in-block-order
        // result even though query discovery order is different.
        for (auto id : _query_aggregates) {
            ++_stats.aggregate_scan_count;
            auto *candidate = _blocks[id];
            if (candidate == nullptr || candidate == header ||
                _is_boundary(candidate) ||
                !_has_aggregate(id) ||
                _support[id] < required_support) {
                continue;
            }
            auto *canonical =
                canonical_trivial_branch_chain_target(candidate);
            auto &score = _is_boundary(canonical) ?
                              boundary_proxy_best :
                              best;
            _consider(
                score, candidate, id, _support[id],
                _maximum_distances[id],
                _total_distances[id]);
        }
        if (best.block != nullptr) { return best.block; }
        if (boundary_proxy_best.block != nullptr) {
            return boundary_proxy_best.block;
        }

        // A reachable block dominates `header` iff its DomTree node is on
        // `header`'s ancestor chain. Walking that chain enumerates exactly the
        // same candidate-header set as the former all-block dominance filter.
        // Dense block IDs retain its original tie order.
        auto fallback_block_id = invalid_index;
        auto *ancestor = _dominance.node_or_null(header);
        ancestor = ancestor == nullptr ? nullptr : ancestor->parent();
        while (ancestor != nullptr) {
            ++_stats.dominator_ancestor_visit_count;
            auto *candidate_header = ancestor->block();
            auto header_iter = _block_ids.find(candidate_header);
            LUISA_DEBUG_ASSERT(
                header_iter != _block_ids.end(),
                "Dominator-tree block must be value-numbered.");
            if (header_iter == _block_ids.end()) {
                ancestor = ancestor->parent();
                continue;
            }
            auto header_id = header_iter->second;
            if (!candidate_header->is_terminated()) {
                ancestor = ancestor->parent();
                continue;
            }
            auto *terminator = candidate_header->terminator();
            if (!terminator->isa<IfInst>() &&
                !terminator->isa<SwitchInst>()) {
                ancestor = ancestor->parent();
                continue;
            }
            auto *candidate =
                structured_statement_merge(candidate_header);
            auto candidate_iter = _block_ids.find(candidate);
            if (candidate == nullptr || candidate == header ||
                _is_boundary(candidate) ||
                _is_boundary(
                    canonical_trivial_branch_chain_target(
                        candidate)) ||
                candidate_iter == _block_ids.end() ||
                !_has_aggregate(candidate_iter->second)) {
                ancestor = ancestor->parent();
                continue;
            }
            auto min_distance =
                _minimum_distances[candidate_iter->second];
            if (min_distance < best.max_distance ||
                (min_distance == best.max_distance &&
                 header_id < fallback_block_id)) {
                best.block = candidate;
                best.max_distance = min_distance;
                fallback_block_id = header_id;
            }
            ancestor = ancestor->parent();
        }
        if (best.block != nullptr) { return best.block; }

        // One-normal-arm fallback immediately before a recovered construct.
        fallback_block_id = invalid_index;
        for (auto id : _query_aggregates) {
            ++_stats.aggregate_scan_count;
            auto *candidate = _blocks[id];
            if (candidate == nullptr || candidate == header ||
                _is_boundary(candidate) ||
                _is_boundary(
                    canonical_trivial_branch_chain_target(
                        candidate)) ||
                !candidate->is_terminated() ||
                structured_statement_merge(candidate) == nullptr ||
                !_has_aggregate(id)) {
                continue;
            }
            auto min_distance = _minimum_distances[id];
            if (min_distance < best.max_distance ||
                (min_distance == best.max_distance &&
                 id < fallback_block_id)) {
                best.block = candidate;
                best.max_distance = min_distance;
                fallback_block_id = id;
            }
        }
        return best.block;
    }

    [[nodiscard]] const SelectionMergeBatchStats &stats()
        const noexcept {
        return _stats;
    }
};

SelectionMergeBatchAnalysis::SelectionMergeBatchAnalysis(
    FunctionDefinition *definition,
    const DomTree &dominance) noexcept
    : _impl{luisa::make_unique<Impl>(
          definition, dominance)} {}

SelectionMergeBatchAnalysis::~SelectionMergeBatchAnalysis() noexcept = default;

SelectionMergeBatchAnalysis::SelectionMergeBatchAnalysis(
    SelectionMergeBatchAnalysis &&) noexcept = default;

SelectionMergeBatchAnalysis &
SelectionMergeBatchAnalysis::operator=(
    SelectionMergeBatchAnalysis &&) noexcept = default;

void SelectionMergeBatchAnalysis::register_overlay_block(
    BasicBlock *block) noexcept {
    _impl->register_overlay_block(block);
}

BasicBlock *SelectionMergeBatchAnalysis::infer(
    BasicBlock *header,
    luisa::span<BasicBlock *const> entries,
    const luisa::unordered_map<BasicBlock *, BasicBlock *> *
        dominance_anchors) noexcept {
    return _impl->infer(
        header, entries, dominance_anchors);
}

const SelectionMergeBatchStats &
SelectionMergeBatchAnalysis::stats() const noexcept {
    return _impl->stats();
}

}// namespace luisa::compute::xir::detail
