#include "restructure_cfg_post_dom.h"

#include <cstdint>
#include <limits>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/raster_discard.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/unreachable.h>

namespace luisa::compute::xir::detail {

namespace {

constexpr auto post_dom_invalid_index = std::numeric_limits<size_t>::max();

struct NumberedEdge {
    size_t source;
    size_t target;
};

[[nodiscard]] bool has_explicit_sink_terminator(
    BasicBlock *block) noexcept {
    auto *terminator = block->terminator();
    return terminator->isa<ReturnInst>() ||
           terminator->isa<UnreachableInst>() ||
           terminator->isa<RasterDiscardInst>();
}

}// namespace

bool is_restructure_cfg_sink(BasicBlock *block) noexcept {
    if (!block->is_terminated()) { return true; }
    if (has_explicit_sink_terminator(block)) {
        return true;
    }
    auto has_successor = false;
    block->traverse_successors(
        false, [&](BasicBlock *) noexcept {
            has_successor = true;
        });
    return !has_successor;
}

BasicBlock *RestructurePostDomInfo::immediate_postdom(
    BasicBlock *block) const noexcept {
    if (auto iter = nodes.find(block); iter != nodes.end()) {
        return iter->second.parent;
    }
    return nullptr;
}

BasicBlock *RestructurePostDomInfo::nearest_common_postdom(
    luisa::span<BasicBlock *const> blocks,
    size_t *ancestor_step_count) const noexcept {
    if (ancestor_step_count != nullptr) {
        *ancestor_step_count = 0u;
    }
    if (blocks.empty()) { return nullptr; }
    auto find_node = [&](BasicBlock *block) noexcept
        -> const Node * {
        if (auto iter = nodes.find(block); iter != nodes.end()) {
            return &iter->second;
        }
        return nullptr;
    };
    auto *common = blocks.front();
    auto *common_node = find_node(common);
    if (common_node == nullptr) { return nullptr; }
    for (auto index = size_t{1u}; index < blocks.size(); ++index) {
        auto *other = blocks[index];
        auto *other_node = find_node(other);
        if (other_node == nullptr) { return nullptr; }
        while (common_node->depth > other_node->depth) {
            common = common_node->parent;
            common_node = find_node(common);
            if (common_node == nullptr) { return nullptr; }
            if (ancestor_step_count != nullptr) {
                ++*ancestor_step_count;
            }
        }
        while (other_node->depth > common_node->depth) {
            other = other_node->parent;
            other_node = find_node(other);
            if (other_node == nullptr) { return nullptr; }
            if (ancestor_step_count != nullptr) {
                ++*ancestor_step_count;
            }
        }
        while (common != other) {
            common = common_node->parent;
            other = other_node->parent;
            common_node = find_node(common);
            other_node = find_node(other);
            if (common_node == nullptr || other_node == nullptr) {
                return nullptr;
            }
            if (ancestor_step_count != nullptr) {
                *ancestor_step_count += 2u;
            }
        }
    }
    return common == virtual_exit ? nullptr : common;
}

bool RestructurePostDomInfo::insert_transparent_merge(
    BasicBlock *block,
    BasicBlock *successor,
    TransparentMergeUpdateStats *stats) noexcept {
    TransparentMergeUpdateStats local_stats;
    if (block == nullptr || successor == nullptr ||
        nodes.contains(block)) {
        return false;
    }
    auto successor_iter = nodes.find(successor);
    if (successor_iter == nodes.end() ||
        !block->is_terminated() ||
        !block->terminator()->isa<BranchInst>() ||
        static_cast<BranchInst *>(block->terminator())
                ->target_block() != successor) {
        return false;
    }

    // Every executable incoming edge is a redirected old `u -> successor`
    // edge. Such a predecessor was already entry- and sink-reachable. If no
    // old active predecessor exists, `block` is referenced only as a declared
    // structured merge; ControlFlowMerge::merge_block is not an executable
    // CFG operand, so the block remains outside the fresh CHK domain.
    auto has_active_predecessor = false;
    block->traverse_predecessors(
        false, [&](BasicBlock *predecessor) noexcept {
            has_active_predecessor |= nodes.contains(predecessor);
        });
    if (!has_active_predecessor) {
        if (stats != nullptr) { *stats = local_stats; }
        return true;
    }

    if (++update_epoch == 0u) {
        for (auto &[_, node] : nodes) {
            node.update_epoch = 0u;
        }
        update_epoch = 1u;
    }
    const auto epoch = update_epoch;

    // Before the subdivision, every candidate below `successor` has it as a
    // postdominator. Start from that complete old subtree (top) and remove a
    // block as soon as one active successor is outside the candidate set.
    // This descending greatest-fixed-point solve matches the CHK treatment of
    // cycles: a cyclic region remains covered exactly when every active exit
    // from the region passes through the new merge.
    luisa::vector<BasicBlock *> candidates;
    luisa::vector<BasicBlock *> stack{
        successor_iter->second.children.begin(),
        successor_iter->second.children.end()};
    while (!stack.empty()) {
        auto *candidate = stack.back();
        stack.pop_back();
        auto iter = nodes.find(candidate);
        LUISA_ASSERT(
            iter != nodes.end(),
            "Postdom child is absent from its owning tree.");
        if (iter->second.update_epoch == epoch) { continue; }
        iter->second.update_epoch = epoch;
        candidates.emplace_back(candidate);
        for (auto *child : iter->second.children) {
            stack.emplace_back(child);
        }
    }
    local_stats.candidate_block_count = candidates.size();

    auto has_outside_active_successor =
        [&](BasicBlock *candidate) noexcept {
            ++local_stats.block_evaluation_count;
            auto outside = false;
            candidate->traverse_successors(
                false, [&](BasicBlock *next) noexcept {
                    ++local_stats.edge_visit_count;
                    if (next == block) { return; }
                    if (auto iter = nodes.find(next);
                        iter != nodes.end() &&
                        iter->second.update_epoch != epoch) {
                        outside = true;
                    }
                });
            return outside;
        };

    luisa::vector<BasicBlock *> removed;
    for (auto *candidate : candidates) {
        auto iter = nodes.find(candidate);
        if (iter->second.update_epoch == epoch &&
            has_outside_active_successor(candidate)) {
            iter->second.update_epoch = 0u;
            removed.emplace_back(candidate);
        }
    }
    for (auto cursor = size_t{0u};
         cursor < removed.size(); ++cursor) {
        removed[cursor]->traverse_predecessors(
            false, [&](BasicBlock *predecessor) noexcept {
                auto iter = nodes.find(predecessor);
                if (iter == nodes.end() ||
                    iter->second.update_epoch != epoch ||
                    !has_outside_active_successor(predecessor)) {
                    return;
                }
                iter->second.update_epoch = 0u;
                removed.emplace_back(predecessor);
            });
    }

    // Coverage is upward-closed in the old postdom tree: if the new merge
    // postdominates x, it also postdominates every old strict postdominator of
    // x below `successor`. Validate that property before changing the tree.
    for (auto *candidate : candidates) {
        auto iter = nodes.find(candidate);
        if (iter->second.update_epoch != epoch) { continue; }
        ++local_stats.covered_block_count;
        auto *parent = iter->second.parent;
        if (parent == successor) { continue; }
        auto parent_iter = nodes.find(parent);
        if (parent_iter == nodes.end() ||
            parent_iter->second.update_epoch != epoch) {
            return false;
        }
    }

    // Insert the new tree node immediately below `successor`, moving exactly
    // the covered old children and their intact subtrees underneath it.
    auto successor_children = std::move(
        successor_iter->second.children);
    luisa::vector<BasicBlock *> retained_children;
    luisa::vector<BasicBlock *> reparented_roots;
    retained_children.reserve(successor_children.size() + 1u);
    for (auto *child : successor_children) {
        auto child_iter = nodes.find(child);
        if (child_iter->second.update_epoch == epoch) {
            reparented_roots.emplace_back(child);
        } else {
            retained_children.emplace_back(child);
        }
    }
    local_stats.reparented_root_count =
        reparented_roots.size();
    const auto successor_depth = successor_iter->second.depth;
    nodes.emplace(
        block,
        Node{.parent = successor,
             .depth = successor_depth + 1u,
             .children = reparented_roots});
    successor_iter = nodes.find(successor);
    retained_children.emplace_back(block);
    successor_iter->second.children =
        std::move(retained_children);
    for (auto *root : reparented_roots) {
        nodes.find(root)->second.parent = block;
        stack.emplace_back(root);
    }
    while (!stack.empty()) {
        auto *descendant = stack.back();
        stack.pop_back();
        auto iter = nodes.find(descendant);
        ++iter->second.depth;
        for (auto *child : iter->second.children) {
            stack.emplace_back(child);
        }
    }
    if (stats != nullptr) { *stats = local_stats; }
    return true;
}

bool RestructurePostDomInfo::structurally_equals(
    const RestructurePostDomInfo &other) const noexcept {
    if (virtual_exit != other.virtual_exit ||
        nodes.size() != other.nodes.size()) {
        return false;
    }
    for (auto &[block, node] : nodes) {
        auto iter = other.nodes.find(block);
        if (iter == other.nodes.end() ||
            iter->second.parent != node.parent ||
            iter->second.depth != node.depth) {
            return false;
        }
    }
    return true;
}

RestructurePostDomInfo compute_restructure_post_dom(
    FunctionDefinition *definition,
    RestructurePostDomStats *stats) noexcept {
    RestructurePostDomStats local_stats;

    // Value-number the owned CFG once. Pointer hashing is confined to this
    // boundary; DFS and the immediate-dominator fixed point use only IDs.
    luisa::vector<BasicBlock *> blocks;
    definition->traverse_basic_blocks(
        [&](BasicBlock *block) noexcept {
            blocks.emplace_back(block);
        });
    const auto block_count = blocks.size();
    local_stats.numbered_block_count = block_count;
    luisa::unordered_map<BasicBlock *, size_t> block_indices;
    block_indices.reserve(block_count);
    for (auto index = size_t{0u};
         index < block_count; ++index) {
        block_indices.emplace(blocks[index], index);
    }

    // Number each local edge once, retaining definition/successor order. The
    // two CSR views are the original successor graph and its reverse.
    luisa::vector<NumberedEdge> edges;
    luisa::vector<uint8_t> sinks(block_count, 0u);
    for (auto source = size_t{0u};
         source < block_count; ++source) {
        auto *block = blocks[source];
        if (!block->is_terminated()) {
            sinks[source] = 1u;
            continue;
        }
        auto has_successor = false;
        block->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                has_successor = true;
                if (auto iter = block_indices.find(successor);
                    iter != block_indices.end()) {
                    edges.emplace_back(NumberedEdge{
                        source, iter->second});
                }
            });
        sinks[source] = static_cast<uint8_t>(
            has_explicit_sink_terminator(block) ||
            !has_successor);
    }
    local_stats.numbered_edge_count = edges.size();

    luisa::vector<size_t> successor_offsets(
        block_count + 1u, 0u);
    luisa::vector<size_t> predecessor_offsets(
        block_count + 1u, 0u);
    for (auto edge : edges) {
        ++successor_offsets[edge.source + 1u];
        ++predecessor_offsets[edge.target + 1u];
    }
    for (auto index = size_t{0u};
         index < block_count; ++index) {
        successor_offsets[index + 1u] +=
            successor_offsets[index];
        predecessor_offsets[index + 1u] +=
            predecessor_offsets[index];
    }
    luisa::vector<size_t> successor_indices(
        edges.size(), post_dom_invalid_index);
    luisa::vector<size_t> predecessor_indices(
        edges.size(), post_dom_invalid_index);
    auto successor_cursors = successor_offsets;
    auto predecessor_cursors = predecessor_offsets;
    for (auto edge : edges) {
        successor_indices[successor_cursors[edge.source]++] = edge.target;
        predecessor_indices[predecessor_cursors[edge.target]++] = edge.source;
    }

    // In the reversed CFG, a synthetic root has one outgoing edge to every
    // sink. Iterative DFS yields reverse postorder without host recursion.
    const auto virtual_exit_index = block_count;
    luisa::vector<size_t> sink_indices;
    for (auto index = size_t{0u};
         index < block_count; ++index) {
        if (sinks[index] != 0u) {
            sink_indices.emplace_back(index);
        }
    }
    struct DFSFrame {
        size_t node;
        size_t next_predecessor;
    };
    luisa::vector<uint8_t> visited(
        block_count + 1u, 0u);
    luisa::vector<DFSFrame> stack;
    luisa::vector<size_t> postorder;
    visited[virtual_exit_index] = 1u;
    stack.emplace_back(DFSFrame{
        virtual_exit_index, 0u});
    while (!stack.empty()) {
        auto &frame = stack.back();
        const auto predecessor_count =
            frame.node == virtual_exit_index ?
                sink_indices.size() :
                predecessor_offsets[frame.node + 1u] -
                    predecessor_offsets[frame.node];
        if (frame.next_predecessor < predecessor_count) {
            const auto predecessor =
                frame.node == virtual_exit_index ?
                    sink_indices[frame.next_predecessor++] :
                    predecessor_indices[predecessor_offsets[frame.node] +
                                        frame.next_predecessor++];
            if (visited[predecessor] == 0u) {
                visited[predecessor] = 1u;
                stack.emplace_back(DFSFrame{
                    predecessor, 0u});
            }
        } else {
            postorder.emplace_back(frame.node);
            stack.pop_back();
        }
    }
    luisa::vector<size_t> reverse_postorder{
        postorder.rbegin(), postorder.rend()};
    LUISA_DEBUG_ASSERT(
        !reverse_postorder.empty() &&
            reverse_postorder.front() == virtual_exit_index,
        "Post-dominator reverse CFG has no virtual root.");
    local_stats.active_block_count =
        reverse_postorder.size() - 1u;

    // Cooper-Harvey-Kennedy on the reversed CFG. Solver indices are reverse-
    // postorder IDs, so every resolved idom moves toward a smaller ID.
    luisa::vector<size_t> solver_indices(
        block_count + 1u, post_dom_invalid_index);
    for (auto index = size_t{0u};
         index < reverse_postorder.size(); ++index) {
        solver_indices[reverse_postorder[index]] = index;
    }
    luisa::vector<size_t> immediate_dominators(
        reverse_postorder.size(), post_dom_invalid_index);
    immediate_dominators.front() = 0u;
    const auto intersect = [&](size_t lhs,
                               size_t rhs) noexcept {
        while (lhs != rhs) {
            while (lhs > rhs) {
                lhs = immediate_dominators[lhs];
                ++local_stats.intersect_step_count;
            }
            while (rhs > lhs) {
                rhs = immediate_dominators[rhs];
                ++local_stats.intersect_step_count;
            }
        }
        return lhs;
    };
    for (;;) {
        ++local_stats.fixed_point_iteration_count;
        auto changed = false;
        for (auto solver_index = size_t{1u};
             solver_index < reverse_postorder.size();
             ++solver_index) {
            ++local_stats.fixed_point_block_visit_count;
            const auto block_index =
                reverse_postorder[solver_index];
            auto new_idom = post_dom_invalid_index;
            if (sinks[block_index] != 0u) {
                ++local_stats.fixed_point_edge_visit_count;
                new_idom = 0u;
            } else {
                for (auto edge_index =
                         successor_offsets[block_index];
                     edge_index <
                     successor_offsets[block_index + 1u];
                     ++edge_index) {
                    ++local_stats.fixed_point_edge_visit_count;
                    const auto successor_solver_index =
                        solver_indices[successor_indices[edge_index]];
                    if (successor_solver_index == post_dom_invalid_index ||
                        immediate_dominators[successor_solver_index] ==
                            post_dom_invalid_index) {
                        continue;
                    }
                    new_idom = new_idom == post_dom_invalid_index ?
                                   successor_solver_index :
                                   intersect(
                                       successor_solver_index,
                                       new_idom);
                }
            }
            if (new_idom != post_dom_invalid_index &&
                immediate_dominators[solver_index] !=
                    new_idom) {
                immediate_dominators[solver_index] =
                    new_idom;
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    for (auto idom : immediate_dominators) {
        LUISA_ASSERT(
            idom != post_dom_invalid_index,
            "Sink-reachable restructure CFG block has no "
            "immediate post-dominator.");
    }

    // Convert back to the historical pointer API only after the dense solve.
    static int virtual_exit_sentinel = 0;
    auto *virtual_exit = reinterpret_cast<BasicBlock *>(
        &virtual_exit_sentinel);
    RestructurePostDomInfo result;
    result.virtual_exit = virtual_exit;
    result.nodes.reserve(reverse_postorder.size());
    result.nodes.emplace(
        virtual_exit,
        RestructurePostDomInfo::Node{
            .parent = virtual_exit,
            .depth = 0u});
    luisa::vector<size_t> depths(
        reverse_postorder.size(), 0u);
    for (auto solver_index = size_t{1u};
         solver_index < reverse_postorder.size();
         ++solver_index) {
        const auto block_index =
            reverse_postorder[solver_index];
        const auto parent_solver_index =
            immediate_dominators[solver_index];
        auto *parent = parent_solver_index == 0u ?
                           virtual_exit :
                           blocks[reverse_postorder[parent_solver_index]];
        LUISA_DEBUG_ASSERT(
            parent_solver_index < solver_index,
            "Immediate post-dominator must precede its child in "
            "reverse-CFG RPO.");
        depths[solver_index] =
            depths[parent_solver_index] + 1u;
        result.nodes.emplace(
            blocks[block_index],
            RestructurePostDomInfo::Node{
                .parent = parent,
                .depth = depths[solver_index]});
    }
    for (auto &[block, node] : result.nodes) {
        if (block == virtual_exit) { continue; }
        auto parent_iter = result.nodes.find(node.parent);
        LUISA_ASSERT(
            parent_iter != result.nodes.end(),
            "Immediate post-dominator is absent from its tree.");
        parent_iter->second.children.emplace_back(block);
    }
    if (stats != nullptr) { *stats = local_stats; }
    return result;
}

}// namespace luisa::compute::xir::detail
