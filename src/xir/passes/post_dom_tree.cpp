#include <luisa/core/logging.h>
#include <luisa/xir/passes/post_dom_tree.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/instructions/raster_discard.h>

namespace luisa::compute::xir {

inline PostDomTreeNode::PostDomTreeNode(BasicBlock *block) noexcept
    : _block{block}, _parent{nullptr} {}

inline void PostDomTreeNode::add_child(PostDomTreeNode *child) noexcept {
    LUISA_DEBUG_ASSERT(child != nullptr && child->_parent == nullptr && child != this, "Invalid child.");
    LUISA_DEBUG_ASSERT(std::find(_children.begin(), _children.end(), child) == _children.end(), "Child already exists.");
    child->_parent = this;
    _children.emplace_back(child);
}

inline void PostDomTreeNode::add_frontier(PostDomTreeNode *frontier) noexcept {
    LUISA_DEBUG_ASSERT(frontier != nullptr, "Invalid frontier.");
    LUISA_DEBUG_ASSERT(std::find(_frontiers.begin(), _frontiers.end(), frontier) == _frontiers.end(), "Frontier already exists.");
    _frontiers.emplace_back(frontier);
}

inline PostDomTree::PostDomTree() noexcept : _virtual_root{nullptr}, _root{nullptr} {}

inline PostDomTreeNode *PostDomTree::add_or_get_node(BasicBlock *block) noexcept {
    if (block == nullptr) {
        if (_virtual_root == nullptr) {
            _virtual_root = luisa::make_unique<PostDomTreeNode>(nullptr);
        }
        return _virtual_root.get();
    }
    auto iter = _nodes.try_emplace(block).first;
    if (iter->second == nullptr) {
        iter->second = luisa::make_unique<PostDomTreeNode>(block);
    }
    return iter->second.get();
}

inline void PostDomTree::set_root(PostDomTreeNode *root) noexcept {
    LUISA_DEBUG_ASSERT(_root == nullptr, "Root already exists.");
    LUISA_DEBUG_ASSERT(root != nullptr, "Invalid root.");
    _root = root;
}

inline void PostDomTree::compute_post_dominance_frontiers() noexcept {
    luisa::fixed_vector<BasicBlock *, 16u> succs;
    luisa::unordered_map<PostDomTreeNode *, luisa::unordered_set<PostDomTreeNode *>> frontiers;
    for (auto &&[b, node] : _nodes) {
        succs.clear();
        b->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (_nodes.contains(succ)) {
                succs.emplace_back(succ);
            }
        });
        if (succs.size() >= 2) {
            for (auto succ : succs) {
                auto runner = succ;
                auto ipostdom_block = node->parent()->block();
                while (runner != ipostdom_block) {
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

auto PostDomTree::node(BasicBlock *block) const noexcept -> const PostDomTreeNode * {
    if (block == nullptr) {
        LUISA_ASSERT(_virtual_root != nullptr, "Virtual exit not found.");
        return _virtual_root.get();
    }
    auto iter = _nodes.find(block);
    LUISA_ASSERT(iter != _nodes.end(), "Block not found in the post-dom tree.");
    return iter->second.get();
}

auto PostDomTree::node_or_null(BasicBlock *block) const noexcept -> const PostDomTreeNode * {
    if (block == nullptr) {
        return _virtual_root.get();
    }
    auto iter = _nodes.find(block);
    return iter == _nodes.cend() ? nullptr : iter->second.get();
}

bool PostDomTree::contains(BasicBlock *block) const noexcept {
    if (block == nullptr) { return _virtual_root != nullptr; }
    return _nodes.contains(block);
}

bool PostDomTree::post_dominates(BasicBlock *a, BasicBlock *b) const noexcept {
    if (a == b) { return true; }
    if (a == nullptr) { return true; }
    auto a_node = node(a);
    auto b_node = node(b);
    while (b_node != _root) {
        if (b_node == a_node) { return true; }
        b_node = b_node->parent();
    }
    return false;
}

bool PostDomTree::strictly_post_dominates(BasicBlock *a, BasicBlock *b) const noexcept {
    return a != b && post_dominates(a, b);
}

auto PostDomTree::immediate_post_dominator(BasicBlock *block) const noexcept -> BasicBlock * {
    auto n = this->node_or_null(block);
    if (n == nullptr || n->parent() == _root) { return nullptr; }
    return n->parent()->block();
}

static const auto kUnknownDom = reinterpret_cast<BasicBlock *>(uintptr_t(-1));

PostDomTree compute_post_dom_tree(Function *function) noexcept {
    auto definition = function->definition();
    LUISA_ASSERT(definition != nullptr, "Function has no definition.");
    // collect all blocks and identify sink blocks
    luisa::vector<BasicBlock *> all_blocks;
    luisa::unordered_set<BasicBlock *> sinks;
    definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        all_blocks.emplace_back(block);
        if (auto term = block->terminator()) {
            if (term->isa<ReturnInst>() || term->isa<UnreachableInst>() || term->isa<RasterDiscardInst>()) {
                sinks.emplace(block);
            }
        }
    });
    // compute postorder on reversed CFG (follow original predecessors)
    luisa::unordered_set<BasicBlock *> visited;
    luisa::vector<BasicBlock *> postorder;
    luisa::vector<BasicBlock *> preds;
    for (auto sink : sinks) {
        if (visited.contains(sink)) { continue; }
        struct Frame {
            BasicBlock *block;
            size_t next_pred_idx;
        };
        luisa::vector<Frame> stack;
        stack.push_back({sink, 0});
        visited.emplace(sink);
        while (!stack.empty()) {
            auto &frame = stack.back();
            auto block = frame.block;
            preds.clear();
            block->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                preds.emplace_back(pred);
            });
            bool pushed = false;
            while (frame.next_pred_idx < preds.size()) {
                auto pred = preds[frame.next_pred_idx++];
                if (!visited.contains(pred)) {
                    visited.emplace(pred);
                    stack.push_back({pred, 0});
                    pushed = true;
                    break;
                }
            }
            if (!pushed) {
                postorder.emplace_back(block);
                stack.pop_back();
            }
        }
    }

    if (postorder.empty()) {
        PostDomTree tree;
        tree.set_root(tree.add_or_get_node(nullptr));
        return tree;
    }
    // build reverse postorder and postorder index map
    luisa::unordered_map<BasicBlock *, size_t> postorder_index;
    auto reverse_postorder = std::move(postorder);
    std::reverse(reverse_postorder.begin(), reverse_postorder.end());
    for (size_t i = 0; i < reverse_postorder.size(); ++i) {
        postorder_index.emplace(reverse_postorder[i], reverse_postorder.size() - 1 - i);
    }
    // dominance algorithm on reversed CFG
    luisa::unordered_map<BasicBlock *, BasicBlock *> doms;
    for (auto block : reverse_postorder) { doms[block] = kUnknownDom; }
    auto get_index = [&](BasicBlock *b) noexcept -> size_t {
        if (b == nullptr) { return SIZE_MAX; }
        LUISA_DEBUG_ASSERT(postorder_index.contains(b), "Block not in postorder.");
        return postorder_index[b];
    };
    auto get_dom = [&](BasicBlock *b) noexcept -> BasicBlock * {
        if (b == nullptr) { return nullptr; }
        auto it = doms.find(b);
        LUISA_DEBUG_ASSERT(it != doms.end() && it->second != kUnknownDom, "Dom not computed.");
        return it->second;
    };
    auto intersect = [&](BasicBlock *b1, BasicBlock *b2) noexcept -> BasicBlock * {
        auto finger1 = b1;
        auto finger2 = b2;
        while (finger1 != finger2) {
            while (get_index(finger1) < get_index(finger2)) {
                finger1 = get_dom(finger1);
            }
            while (get_index(finger2) < get_index(finger1)) {
                finger2 = get_dom(finger2);
            }
        }
        return finger1;
    };
    for (;;) {
        auto changed = false;
        for (auto block : reverse_postorder) {
            auto new_idom = static_cast<BasicBlock *>(nullptr);
            bool first = true;
            if (sinks.contains(block)) {
                new_idom = nullptr;
                first = false;
            }
            block->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (auto iter = doms.find(succ); iter != doms.end() && iter->second != kUnknownDom) {
                    if (first) {
                        new_idom = succ;
                        first = false;
                    } else {
                        new_idom = intersect(succ, new_idom);
                    }
                }
            });
            if (!first) {
                if (auto &dom = doms[block]; dom != new_idom) {
                    dom = new_idom;
                    changed = true;
                }
            }
        }
        if (!changed) { break; }
    }
    // build the post-dom tree
    PostDomTree tree;
    for (auto block : reverse_postorder) {
        LUISA_DEBUG_ASSERT(doms[block] != kUnknownDom, "Block has unknown post-dom.");
        auto parent_node = tree.add_or_get_node(doms[block]);
        auto block_node = tree.add_or_get_node(block);
        parent_node->add_child(block_node);
    }
    tree.set_root(tree.add_or_get_node(nullptr));
    tree.compute_post_dominance_frontiers();
    return tree;
}

}// namespace luisa::compute::xir
