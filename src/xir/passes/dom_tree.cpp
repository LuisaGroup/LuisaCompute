#include <luisa/core/logging.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::compute::xir {

inline DomTreeNode::DomTreeNode(BasicBlock *block) noexcept
    : _block{block},
      _parent{nullptr},
      _preorder_index{SIZE_MAX},
      _subtree_end_index{SIZE_MAX} {}

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

bool DomTreeNode::dominates(const DomTreeNode *other) const noexcept {
    if (other == nullptr) { return false; }
    LUISA_DEBUG_ASSERT(
        _preorder_index != SIZE_MAX &&
            _subtree_end_index != SIZE_MAX &&
            other->_preorder_index != SIZE_MAX &&
            other->_subtree_end_index != SIZE_MAX,
        "Dominator tree ancestry intervals have not been computed.");
    // Dominance is ancestry in the dominator tree. The DFS subtree interval
    // makes the relation a constant-time comparison once callers have
    // resolved their block handles to tree nodes.
    return _preorder_index <= other->_preorder_index &&
           other->_preorder_index < _subtree_end_index;
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

inline void DomTree::compute_ancestry_intervals() noexcept {
    LUISA_DEBUG_ASSERT(_root != nullptr, "Root not found.");
    struct StackFrame {
        DomTreeNode *node;
        size_t next_child_index;
    };
    auto next_preorder_index = size_t{0u};
    auto root = const_cast<DomTreeNode *>(_root);
    root->_preorder_index = next_preorder_index++;
    luisa::vector<StackFrame> stack;
    stack.emplace_back(root, 0u);
    while (!stack.empty()) {
        auto &frame = stack.back();
        if (frame.next_child_index < frame.node->_children.size()) {
            auto child = const_cast<DomTreeNode *>(
                frame.node->_children[frame.next_child_index++]);
            LUISA_DEBUG_ASSERT(
                child->_preorder_index == SIZE_MAX &&
                    child->_subtree_end_index == SIZE_MAX,
                "Dominator tree node visited more than once.");
            child->_preorder_index = next_preorder_index++;
            stack.emplace_back(child, 0u);
        } else {
            frame.node->_subtree_end_index = next_preorder_index;
            stack.pop_back();
        }
    }
    LUISA_DEBUG_ASSERT(
        next_preorder_index == _nodes.size(),
        "Dominator tree contains nodes unreachable from its root.");
}

inline void DomTree::compute_dominance_frontiers() noexcept {
    // Frontier construction is a pure derivative of the current tree. Clear
    // the node-local vectors so explicit late materialization is idempotent
    // without adding ABI-visible state to DomTree.
    for (auto &&[block, node] : _nodes) {
        static_cast<void>(block);
        node->_frontiers.clear();
    }
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
    auto src_node = node_or_null(src);
    if (src_node == nullptr) { return false; }
    // Reflexivity needs only one lookup, while a general query resolves each
    // block once. The former contains()+node() sequence resolved each block
    // twice and made a constant-time tree query perform four hash probes.
    if (src == dst) { return true; }
    return src_node->dominates(node_or_null(dst));
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
    return compute_dom_tree(function, {});
}

DomTree compute_dom_tree(
    Function *function,
    DomTreeBuildOptions options) noexcept {
    return compute_dom_tree(function, options, nullptr);
}

DomTree compute_dom_tree(
    Function *function,
    DomTreeBuildOptions options,
    DomTreeBuildStats *stats) noexcept {
    DomTreeBuildStats local_stats;
    auto definition =
        function == nullptr ? nullptr : function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        if (stats != nullptr) {
            *stats = local_stats;
        }
        return {};
    }

    // Preserve the historical reachable domain and traversal order, then
    // value-number it once in reverse postorder. Pointer hashing is confined
    // to this graph-construction boundary.
    luisa::vector<BasicBlock *> reverse_postorder;
    definition->traverse_basic_blocks(
        BasicBlockTraversalOrder::POST_ORDER,
        [&](BasicBlock *block) noexcept {
            reverse_postorder.emplace_back(block);
        });
    auto root_block = definition->body_block();
    LUISA_ASSERT(!reverse_postorder.empty() && reverse_postorder.back() == root_block,
                 "Invalid reverse postorder.");
    std::reverse(reverse_postorder.begin(), reverse_postorder.end());
    LUISA_ASSERT(reverse_postorder.front() == root_block,
                 "Dominator reverse postorder does not begin at the root.");
    const auto block_count = reverse_postorder.size();
    local_stats.numbered_block_count = block_count;
    luisa::unordered_map<BasicBlock *, size_t> block_ids;
    block_ids.reserve(block_count);
    for (auto id = size_t{0u}; id < block_count; ++id) {
        block_ids.emplace(reverse_postorder[id], id);
    }

    // Store only real predecessor edges between reachable blocks. The CSR
    // retains predecessor traversal order, so tie behavior remains identical
    // while the fixed point performs no pointer lookup or allocation.
    luisa::vector<size_t> predecessor_offsets(
        block_count + 1u, 0u);
    luisa::vector<size_t> predecessor_ids;
    for (auto block_id = size_t{0u};
         block_id < block_count; ++block_id) {
        reverse_postorder[block_id]->traverse_predecessors(
            false, [&](BasicBlock *predecessor) noexcept {
                if (auto iter = block_ids.find(predecessor);
                    iter != block_ids.end()) {
                    predecessor_ids.emplace_back(iter->second);
                }
            });
        predecessor_offsets[block_id + 1u] =
            predecessor_ids.size();
    }
    local_stats.numbered_edge_count = predecessor_ids.size();

    // Cooper-Harvey-Kennedy on dense RPO IDs. Every resolved immediate
    // dominator moves toward a smaller ID, so intersection is parent climbing
    // over one sparse relation rather than repeated hash-map queries.
    constexpr auto invalid_id = SIZE_MAX;
    luisa::vector<size_t> immediate_dominators(
        block_count, invalid_id);
    immediate_dominators.front() = 0u;
    const auto intersect = [&](size_t lhs,
                               size_t rhs) noexcept {
        while (lhs != rhs) {
            while (lhs > rhs) {
                lhs = immediate_dominators[lhs];
                LUISA_DEBUG_ASSERT(
                    lhs != invalid_id,
                    "Invalid dominator parent chain.");
                ++local_stats.intersect_step_count;
            }
            while (rhs > lhs) {
                rhs = immediate_dominators[rhs];
                LUISA_DEBUG_ASSERT(
                    rhs != invalid_id,
                    "Invalid dominator parent chain.");
                ++local_stats.intersect_step_count;
            }
        }
        return lhs;
    };
    for (;;) {
        ++local_stats.fixed_point_iteration_count;
        auto changed = false;
        for (auto block_id = size_t{1u};
             block_id < block_count; ++block_id) {
            ++local_stats.fixed_point_block_visit_count;
            auto new_immediate_dominator = invalid_id;
            for (auto edge_index = predecessor_offsets[block_id];
                 edge_index < predecessor_offsets[block_id + 1u];
                 ++edge_index) {
                ++local_stats.fixed_point_edge_visit_count;
                auto predecessor_id = predecessor_ids[edge_index];
                if (immediate_dominators[predecessor_id] ==
                    invalid_id) {
                    continue;
                }
                new_immediate_dominator =
                    new_immediate_dominator == invalid_id ?
                        predecessor_id :
                        intersect(predecessor_id,
                                  new_immediate_dominator);
            }
            if (new_immediate_dominator != invalid_id &&
                immediate_dominators[block_id] !=
                    new_immediate_dominator) {
                immediate_dominators[block_id] =
                    new_immediate_dominator;
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    for (auto id : immediate_dominators) {
        LUISA_ASSERT(id != invalid_id,
                     "Reachable block has no immediate dominator.");
    }

    // Convert dense parents to the historical pointer-based public tree only
    // after the solve has converged.
    DomTree tree;
    for (auto block_id = size_t{1u};
         block_id < block_count; ++block_id) {
        auto parent_node = tree.add_or_get_node(
            reverse_postorder[immediate_dominators[block_id]]);
        auto block_node = tree.add_or_get_node(
            reverse_postorder[block_id]);
        parent_node->add_child(block_node);
    }
    tree.set_root(tree.add_or_get_node(root_block));
    tree.compute_ancestry_intervals();
    if (options.compute_dominance_frontiers) {
        tree.compute_dominance_frontiers();
    }
    if (stats != nullptr) {
        *stats = local_stats;
    }
    return tree;
}

}// namespace luisa::compute::xir
