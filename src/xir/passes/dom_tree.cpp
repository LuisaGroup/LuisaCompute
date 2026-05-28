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
                auto runner = pred;
                while (runner != node->parent()->block()) {
                    auto runner_node = _nodes[runner].get();
                    if (frontiers[runner_node].emplace(node.get()).second) {
                        runner_node->add_frontier(node.get());
                    }
                    runner = runner_node->parent()->block();
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
    auto node = this->node(block);
    if (node == _root) { return nullptr; }
    return node->parent()->block();
}

// Reference: A Simple, Fast Dominance Algorithm [Cooper et al. 2001]
DomTree compute_dom_tree(Function *function) noexcept {
    auto definition = function->definition();
    LUISA_ASSERT(definition != nullptr, "Function has no definition.");
    // compute reverse postorder
    luisa::unordered_map<BasicBlock *, size_t> postorder_index;
    luisa::vector<BasicBlock *> reverse_postorder;
    definition->traverse_basic_blocks(
        BasicBlockTraversalOrder::POST_ORDER,
        [&](BasicBlock *block) noexcept {
            auto index = reverse_postorder.size();
            postorder_index.emplace(block, index);
            reverse_postorder.emplace_back(block);
        });
    auto root_block = definition->body_block();
    LUISA_ASSERT(!reverse_postorder.empty() && reverse_postorder.back() == root_block,
                 "Invalid reverse postorder.");
    reverse_postorder.pop_back();// remove the root since we don't want to visit it during the traversal
    std::reverse(reverse_postorder.begin(), reverse_postorder.end());
    // dominance information
    luisa::unordered_map<BasicBlock *, BasicBlock *> doms;
    auto intersect = [&](BasicBlock *b1, BasicBlock *b2) noexcept {
        auto checked = [](BasicBlock *b) noexcept {
            LUISA_DEBUG_ASSERT(b != nullptr, "Invalid block.");
            return b;
        };
        auto finger1 = b1;
        auto finger2 = b2;
        while (checked(finger1) != checked(finger2)) {
            while (postorder_index[checked(finger1)] < postorder_index[checked(finger2)]) {
                finger1 = doms[finger1];
            }
            while (postorder_index[checked(finger2)] < postorder_index[checked(finger1)]) {
                finger2 = doms[finger2];
            }
        }
        return finger1;
    };
    // initialize
    for (auto block : reverse_postorder) { doms[block] = nullptr; }
    doms[root_block] = root_block;
    for (;;) {
        auto changed = false;
        for (auto block : reverse_postorder) {
            auto new_idom = static_cast<BasicBlock *>(nullptr);
            block->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                if (auto iter = doms.find(pred); iter != doms.end() && iter->second != nullptr) {
                    if (new_idom == nullptr) {
                        new_idom = pred;
                    } else {
                        new_idom = intersect(pred, new_idom);
                    }
                }
            });
            if (auto &dom = doms[block]; dom != new_idom) {
                dom = new_idom;
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    // create the dom tree
    DomTree tree;
    for (auto block : reverse_postorder) {
        auto parent_node = tree.add_or_get_node(doms[block]);
        auto block_node = tree.add_or_get_node(block);
        parent_node->add_child(block_node);
    }
    tree.set_root(tree.add_or_get_node(root_block));
    tree.compute_dominance_frontiers();
    return tree;
}

}// namespace luisa::compute::xir
