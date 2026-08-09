#include <algorithm>

#include <luisa/core/logging.h>

#include "verifier_dom_tree.h"

namespace luisa::compute::xir::detail {

size_t VerifierSparseDomTree::_intersect(
    size_t lhs, size_t rhs) const noexcept {
    // IDs are reverse-postorder indices, so walking an idom always moves
    // toward a smaller index (except for the root, which dominates itself).
    while (lhs != rhs) {
        while (lhs > rhs) { lhs = _immediate_dominators[lhs]; }
        while (rhs > lhs) { rhs = _immediate_dominators[rhs]; }
    }
    return lhs;
}

VerifierSparseDomTree::VerifierSparseDomTree(
    const BasicBlock *root,
    const VerifierBlockAdjacency &successors,
    const VerifierBlockAdjacency &predecessors,
    const VerifierBlockSet &reachable) noexcept {
    if (root == nullptr || !reachable.contains(root)) { return; }

    // Compute reverse postorder iteratively. Generated shaders can have deep
    // control flow, so host recursion is deliberately avoided here and below.
    struct CFGFrame {
        const BasicBlock *block;
        VerifierBlockSet::const_iterator next;
        VerifierBlockSet::const_iterator end;
    };
    const VerifierBlockSet empty;
    const auto outgoing = [&](const BasicBlock *block) noexcept
        -> const VerifierBlockSet & {
        if (auto iter = successors.find(block);
            iter != successors.end()) {
            return iter->second;
        }
        return empty;
    };
    luisa::vector<const BasicBlock *> postorder;
    postorder.reserve(reachable.size());
    VerifierBlockSet visited;
    visited.reserve(reachable.size());
    auto &&root_edges = outgoing(root);
    luisa::vector<CFGFrame> stack;
    stack.emplace_back(CFGFrame{
        root, root_edges.begin(), root_edges.end()});
    visited.emplace(root);
    while (!stack.empty()) {
        auto &frame = stack.back();
        if (frame.next != frame.end) {
            auto *successor = *frame.next++;
            if (!reachable.contains(successor) ||
                !visited.emplace(successor).second) {
                continue;
            }
            auto &&successor_edges = outgoing(successor);
            stack.emplace_back(CFGFrame{
                successor,
                successor_edges.begin(),
                successor_edges.end()});
        } else {
            postorder.emplace_back(frame.block);
            stack.pop_back();
        }
    }
    luisa::vector<const BasicBlock *> reverse_postorder{
        postorder.rbegin(), postorder.rend()};
    _block_indices.reserve(reverse_postorder.size());
    for (auto index = size_t{0u};
         index < reverse_postorder.size(); ++index) {
        _block_indices.emplace(reverse_postorder[index], index);
    }

    // Number pointer-valued predecessor edges once and encode them in CSR.
    // No pointer hashing remains in the fixed-point loop.
    luisa::vector<size_t> predecessor_offsets(
        reverse_postorder.size() + 1u, 0u);
    for (auto block_index = size_t{0u};
         block_index < reverse_postorder.size(); ++block_index) {
        auto count = size_t{0u};
        if (auto iter = predecessors.find(
                reverse_postorder[block_index]);
            iter != predecessors.end()) {
            for (auto *predecessor : iter->second) {
                count += _block_indices.contains(predecessor);
            }
        }
        predecessor_offsets[block_index + 1u] =
            predecessor_offsets[block_index] + count;
    }
    luisa::vector<size_t> predecessor_indices(
        predecessor_offsets.back(), invalid_index);
    for (auto block_index = size_t{0u};
         block_index < reverse_postorder.size(); ++block_index) {
        auto output_index = predecessor_offsets[block_index];
        if (auto iter = predecessors.find(
                reverse_postorder[block_index]);
            iter != predecessors.end()) {
            for (auto *predecessor : iter->second) {
                if (auto predecessor_iter =
                        _block_indices.find(predecessor);
                    predecessor_iter != _block_indices.end()) {
                    predecessor_indices[output_index++] =
                        predecessor_iter->second;
                }
            }
        }
        LUISA_DEBUG_ASSERT(
            output_index == predecessor_offsets[block_index + 1u],
            "Verifier predecessor CSR size mismatch.");
    }
    _cfg_edge_count = predecessor_indices.size();

    // Cooper-Harvey-Kennedy over numeric IDs. The state is one idom per
    // block, not a set of every dominating block.
    _immediate_dominators.assign(
        reverse_postorder.size(), invalid_index);
    _immediate_dominators.front() = 0u;
    for (;;) {
        ++_fixed_point_iteration_count;
        auto changed = false;
        for (auto block_index = size_t{1u};
             block_index < reverse_postorder.size(); ++block_index) {
            auto new_idom = invalid_index;
            for (auto edge_index = predecessor_offsets[block_index];
                 edge_index < predecessor_offsets[block_index + 1u];
                 ++edge_index) {
                auto predecessor_index =
                    predecessor_indices[edge_index];
                if (_immediate_dominators[predecessor_index] ==
                    invalid_index) {
                    continue;
                }
                new_idom = new_idom == invalid_index ?
                               predecessor_index :
                               _intersect(predecessor_index, new_idom);
            }
            if (_immediate_dominators[block_index] != new_idom) {
                _immediate_dominators[block_index] = new_idom;
                changed = true;
            }
        }
        if (!changed) { break; }
    }

    // Reachability plus RPO guarantees a resolved idom for every non-root
    // block. This assertion checks our implementation, not user IR validity.
    for (auto idom : _immediate_dominators) {
        LUISA_ASSERT(
            idom != invalid_index,
            "Reachable verifier CFG block has no immediate dominator.");
    }

    // Encode the sparse tree as first-child/next-sibling arrays. This avoids
    // one heap-backed vector per block while retaining exactly V - 1 edges.
    luisa::vector<size_t> first_children(
        reverse_postorder.size(), invalid_index);
    luisa::vector<size_t> next_siblings(
        reverse_postorder.size(), invalid_index);
    for (auto node = size_t{1u};
         node < _immediate_dominators.size(); ++node) {
        auto parent = _immediate_dominators[node];
        next_siblings[node] = first_children[parent];
        first_children[parent] = node;
    }

    _depths.assign(_immediate_dominators.size(), 0u);
    _preorder_indices.assign(
        _immediate_dominators.size(), invalid_index);
    _subtree_end_indices.assign(
        _immediate_dominators.size(), invalid_index);
    struct TreeFrame {
        size_t node;
        size_t next_child;
    };
    auto next_preorder_index = size_t{0u};
    _depths[0u] = 1u;
    _preorder_indices[0u] = next_preorder_index++;
    luisa::vector<TreeFrame> tree_stack;
    tree_stack.emplace_back(TreeFrame{0u, first_children[0u]});
    while (!tree_stack.empty()) {
        auto &frame = tree_stack.back();
        if (frame.next_child != invalid_index) {
            auto child = frame.next_child;
            frame.next_child = next_siblings[child];
            _depths[child] = _depths[frame.node] + 1u;
            _preorder_indices[child] = next_preorder_index++;
            tree_stack.emplace_back(TreeFrame{
                child, first_children[child]});
        } else {
            _subtree_end_indices[frame.node] = next_preorder_index;
            tree_stack.pop_back();
        }
    }
    LUISA_ASSERT(
        next_preorder_index == _immediate_dominators.size(),
        "Verifier dominator tree is disconnected.");
}

bool VerifierSparseDomTree::dominates(
    const BasicBlock *dominator,
    const BasicBlock *block) const noexcept {
    auto dominator_iter = _block_indices.find(dominator);
    auto block_iter = _block_indices.find(block);
    if (dominator_iter == _block_indices.end() ||
        block_iter == _block_indices.end()) {
        return false;
    }
    auto dominator_index = dominator_iter->second;
    auto block_index = block_iter->second;
    return _preorder_indices[dominator_index] <=
               _preorder_indices[block_index] &&
           _preorder_indices[block_index] <
               _subtree_end_indices[dominator_index];
}

size_t VerifierSparseDomTree::depth(
    const BasicBlock *block) const noexcept {
    if (auto iter = _block_indices.find(block);
        iter != _block_indices.end()) {
        return _depths[iter->second];
    }
    return 0u;
}

}// namespace luisa::compute::xir::detail
