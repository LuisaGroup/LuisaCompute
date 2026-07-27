#include <luisa/core/logging.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::compute::xir {

inline DomTreeNode::DomTreeNode(BasicBlock *block) noexcept
    : _block{block}, _parent{nullptr} {}

inline void DomTreeNode::add_child(DomTreeNode *child) noexcept {
    LUISA_DEBUG_ASSERT(child != nullptr && child->_parent == nullptr && child != this, "Invalid child.");
    LUISA_DEBUG_ASSERT(std::find(_children.begin(), _children.end(), child) == _children.end(), "Child already exists.");
    child->_parent = this;
    _children.emplace_back(child);
}

inline void DomTreeNode::add_frontier(DomTreeNode *frontier) noexcept {
    LUISA_DEBUG_ASSERT(frontier != nullptr, "Invalid frontier.");
    LUISA_DEBUG_ASSERT(std::find(_frontiers.begin(), _frontiers.end(), frontier) == _frontiers.end(), "Frontier already exists.");
    _frontiers.emplace_back(frontier);
}

DomTree::DomTree() noexcept : _root{nullptr} {}

inline DomTreeNode *DomTree::add_or_get_node(BasicBlock *block) noexcept {
    auto iter = _nodes.try_emplace(block).first;
    if (iter->second == nullptr) {
        iter->second = luisa::make_unique<DomTreeNode>(block);
    }
    return iter->second.get();
}

inline void DomTree::set_root(DomTreeNode *root) noexcept {
    LUISA_DEBUG_ASSERT(_root == nullptr, "Root already exists.");
    LUISA_DEBUG_ASSERT(root != nullptr, "Invalid root.");
    _root = root;
}

inline void DomTree::compute_dominance_frontiers() noexcept {
    luisa::fixed_vector<BasicBlock *, 16u> preds;
    luisa::unordered_map<DomTreeNode *, luisa::unordered_set<DomTreeNode *>> frontiers;
    for (auto &&[b, node] : _nodes) {
        preds.clear();
        b->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
            if (_nodes.contains(pred)) {// only consider reachable blocks
                preds.emplace_back(pred);
            }
        });
        if (preds.size() >= 2) {
            for (auto pred : preds) {
                auto runner_node = _nodes[pred].get();
                auto stop_node = node->parent();
                while (runner_node != stop_node) {
                    if (frontiers[runner_node].emplace(node.get()).second) {
                        runner_node->add_frontier(node.get());
                    }
                    runner_node = const_cast<DomTreeNode *>(runner_node->parent());
                }
            }
        }
    }
}

auto DomTree::node(BasicBlock *block) const noexcept -> const DomTreeNode * {
    auto iter = _nodes.find(block);
    LUISA_ASSERT(iter != _nodes.end(), "Block not found in the dom tree.");
    return iter->second.get();
}

auto DomTree::node_or_null(BasicBlock *block) const noexcept -> const DomTreeNode * {
    auto iter = _nodes.find(block);
    return iter == _nodes.cend() ? nullptr : iter->second.get();
}

bool DomTree::contains(BasicBlock *block) const noexcept {
    return _nodes.contains(block);
}

bool DomTree::dominates(BasicBlock *src, BasicBlock *dst) const noexcept {
    if (!contains(src) || !contains(dst)) { return false; }
    if (src == dst) { return true; }
    auto src_node = node(src);
    auto dst_node = node(dst);
    if (src_node == _root) { return true; }
    while (dst_node != _root) {
        if (dst_node == src_node) { return true; }
        dst_node = dst_node->parent();
    }
    return false;
}

bool DomTree::strictly_dominates(BasicBlock *src, BasicBlock *dst) const noexcept {
    return src != dst && dominates(src, dst);
}

auto DomTree::immediate_dominator(BasicBlock *block) const noexcept -> BasicBlock * {
    auto node = this->node_or_null(block);
    if (node == nullptr || node == _root) { return nullptr; }
    return node->parent()->block();
}

// Reference: A Simple, Fast Dominance Algorithm [Cooper et al. 2001]
DomTree compute_dom_tree(Function *function) noexcept {
    auto definition =
        function == nullptr ? nullptr : function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return {};
    }
    // compute reverse postorder
    luisa::vector<BasicBlock *> reverse_postorder;
    definition->traverse_basic_blocks(
        BasicBlockTraversalOrder::POST_ORDER,
        [&](BasicBlock *block) noexcept {
            reverse_postorder.emplace_back(block);
        });
    auto root_block = definition->body_block();
    LUISA_ASSERT(!reverse_postorder.empty() && reverse_postorder.back() == root_block,
                 "Invalid reverse postorder.");
    reverse_postorder.pop_back();// remove the root since we don't want to visit it during the traversal
    std::reverse(reverse_postorder.begin(), reverse_postorder.end());

    // Assign dense block IDs and compute postorder indices.
    size_t n = reverse_postorder.size() + 1;// +1 for root_block
    luisa::unordered_map<BasicBlock *, size_t> block_id;
    luisa::vector<size_t> postorder_index_vec(n, SIZE_MAX);
    // The postorder index is the position in the original postorder (before reversal).
    // Root was last in postorder (index = original_size - 1).
    // Other blocks have indices 0..original_size-2.
    size_t postorder_size = reverse_postorder.size() + 1;// original postorder size including root
    for (size_t i = 0; i < reverse_postorder.size(); i++) {
        auto *bb = reverse_postorder[i];
        block_id[bb] = i;
        // After reversal, reverse_postorder[i] was at postorder index (postorder_size - 2 - i).
        // Let's just recompute: reverse_postorder is in reverse postorder now,
        // so block at position i has postorder index = postorder_size - 1 - i - 1 (since root was last).
        // Simpler: the block at reverse_postorder[i] was originally at index (original_size - 2 - i) in postorder.
        postorder_index_vec[i] = postorder_size - 2 - i;
    }
    block_id[root_block] = reverse_postorder.size();// root gets the last slot
    postorder_index_vec[reverse_postorder.size()] = postorder_size - 1;// root was last in postorder

    // Dense dominator array.
    luisa::vector<BasicBlock *> doms_vec(n, nullptr);
    doms_vec[block_id[root_block]] = root_block;

    // Helper to get postorder index.
    auto get_postorder_idx = [&](BasicBlock *b) noexcept -> size_t {
        if (b == nullptr) { return SIZE_MAX; }
        auto it = block_id.find(b);
        if (it == block_id.end()) { return SIZE_MAX; }
        return postorder_index_vec[it->second];
    };
    // Helper to get dom.
    auto get_dom = [&](BasicBlock *b) noexcept -> BasicBlock * {
        if (b == nullptr) { return nullptr; }
        auto it = block_id.find(b);
        if (it == block_id.end()) { return nullptr; }
        return doms_vec[it->second];
    };

    auto intersect = [&](BasicBlock *b1, BasicBlock *b2) noexcept {
        LUISA_DEBUG_ASSERT(b1 != nullptr && b2 != nullptr, "Invalid block.");
        auto finger1 = b1;
        auto finger2 = b2;
        while (finger1 != finger2) {
            auto i1 = get_postorder_idx(finger1);
            auto i2 = get_postorder_idx(finger2);
            while (i1 < i2) {
                finger1 = get_dom(finger1);
                LUISA_DEBUG_ASSERT(finger1 != nullptr, "Invalid dom tree.");
                i1 = get_postorder_idx(finger1);
            }
            while (i2 < i1) {
                finger2 = get_dom(finger2);
                LUISA_DEBUG_ASSERT(finger2 != nullptr, "Invalid dom tree.");
                i2 = get_postorder_idx(finger2);
            }
        }
        return finger1;
    };

    // fixed-point iteration
    for (;;) {
        auto changed = false;
        for (auto block : reverse_postorder) {
            auto new_idom = static_cast<BasicBlock *>(nullptr);
            block->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                auto dom_of_pred = get_dom(pred);
                if (dom_of_pred != nullptr) {
                    if (new_idom == nullptr) {
                        new_idom = pred;
                    } else {
                        new_idom = intersect(pred, new_idom);
                    }
                }
            });
            auto &dom = doms_vec[block_id[block]];
            if (dom != new_idom) {
                dom = new_idom;
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    // create the dom tree
    DomTree tree;
    for (auto block : reverse_postorder) {
        auto parent_node = tree.add_or_get_node(doms_vec[block_id[block]]);
        auto block_node = tree.add_or_get_node(block);
        parent_node->add_child(block_node);
    }
    tree.set_root(tree.add_or_get_node(root_block));
    tree.compute_dominance_frontiers();
    return tree;
}

}// namespace luisa::compute::xir
