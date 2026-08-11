#include "coro_semantic_graph.h"

#include <algorithm>
#include <limits>

#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>

#include "coro_semantic_cfg.h"

namespace luisa::compute::xir::detail {

namespace {

constexpr auto invalid_block_id = std::numeric_limits<size_t>::max();

struct DfsFrame {
    size_t block_id;
    size_t next_successor;
};

}// namespace

CoroSemanticGraph::CoroSemanticGraph(
    FunctionDefinition *definition) noexcept {
    if (definition == nullptr || definition->body_block() == nullptr) {
        return;
    }

    luisa::vector<BasicBlock *> owned_blocks;
    for (auto *block : definition->basic_blocks()) {
        owned_blocks.emplace_back(block);
    }
    if (owned_blocks.empty()) { return; }
    luisa::unordered_map<BasicBlock *, size_t> owned_ids;
    owned_ids.reserve(owned_blocks.size());
    for (size_t i = 0u; i < owned_blocks.size(); ++i) {
        owned_ids.emplace(owned_blocks[i], i);
    }
    auto root_iter = owned_ids.find(definition->body_block());
    if (root_iter == owned_ids.end()) { return; }

    CoroTransferGraph coro_transfers{definition};
    if (!coro_transfers.has_unique_complete_token_pairs()) { return; }

    luisa::vector<luisa::vector<size_t>> owned_successors(
        owned_blocks.size());
    auto add_edge = [&](size_t from, size_t to) noexcept {
        auto &successors = owned_successors[from];
        if (std::find(successors.begin(), successors.end(), to) ==
            successors.end()) {
            successors.emplace_back(to);
        }
    };
    auto owned_edges_valid = true;
    for (size_t block_id = 0u;
         block_id < owned_blocks.size(); ++block_id) {
        owned_blocks[block_id]->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                if (auto iter = owned_ids.find(successor);
                    iter != owned_ids.end()) {
                    add_edge(block_id, iter->second);
                } else {
                    owned_edges_valid = false;
                }
            });
    }
    for (size_t block_id = 0u;
         block_id < owned_blocks.size(); ++block_id) {
        coro_transfers.traverse_successors(
            owned_blocks[block_id], [&](BasicBlock *successor) noexcept {
                if (auto iter = owned_ids.find(successor);
                    iter != owned_ids.end()) {
                    add_edge(block_id, iter->second);
                } else {
                    owned_edges_valid = false;
                }
            });
    }
    if (!owned_edges_valid) { return; }

    // Number the augmented graph in reverse postorder. Pointer hashing and
    // token lookup end here; the dominance fixed point uses only dense IDs and
    // sparse predecessor lists.
    luisa::vector<uint8_t> visited(owned_blocks.size(), 0u);
    luisa::vector<size_t> postorder;
    luisa::vector<DfsFrame> stack;
    auto root_owned_id = root_iter->second;
    visited[root_owned_id] = 1u;
    stack.emplace_back(DfsFrame{root_owned_id, 0u});
    while (!stack.empty()) {
        auto &frame = stack.back();
        auto &successors = owned_successors[frame.block_id];
        if (frame.next_successor < successors.size()) {
            auto successor = successors[frame.next_successor++];
            if (visited[successor] == 0u) {
                visited[successor] = 1u;
                stack.emplace_back(DfsFrame{successor, 0u});
            }
        } else {
            postorder.emplace_back(frame.block_id);
            stack.pop_back();
        }
    }
    if (postorder.empty()) { return; }
    std::reverse(postorder.begin(), postorder.end());

    _blocks.reserve(postorder.size());
    _block_ids.reserve(postorder.size());
    luisa::vector<size_t> owned_to_rpo(
        owned_blocks.size(), invalid_block_id);
    for (size_t rpo_id = 0u; rpo_id < postorder.size(); ++rpo_id) {
        auto owned_id = postorder[rpo_id];
        auto *block = owned_blocks[owned_id];
        _blocks.emplace_back(block);
        _block_ids.emplace(block, rpo_id);
        owned_to_rpo[owned_id] = rpo_id;
    }

    _predecessors.resize(_blocks.size());
    _successors.resize(_blocks.size());
    for (size_t from_owned = 0u;
         from_owned < owned_blocks.size(); ++from_owned) {
        auto from = owned_to_rpo[from_owned];
        if (from == invalid_block_id) { continue; }
        for (auto to_owned : owned_successors[from_owned]) {
            auto to = owned_to_rpo[to_owned];
            if (to != invalid_block_id) {
                _successors[from].emplace_back(to);
                _predecessors[to].emplace_back(from);
                ++_edge_count;
            }
        }
    }

    _immediate_dominators.assign(_blocks.size(), invalid_block_id);
    _immediate_dominators.front() = 0u;
    auto intersect = [&](size_t lhs, size_t rhs) noexcept {
        while (lhs != rhs) {
            while (lhs > rhs) {
                lhs = _immediate_dominators[lhs];
                if (lhs == invalid_block_id) { return invalid_block_id; }
            }
            while (rhs > lhs) {
                rhs = _immediate_dominators[rhs];
                if (rhs == invalid_block_id) { return invalid_block_id; }
            }
        }
        return lhs;
    };
    for (;;) {
        auto changed = false;
        for (size_t block_id = 1u;
             block_id < _blocks.size(); ++block_id) {
            auto new_idom = invalid_block_id;
            for (auto predecessor : _predecessors[block_id]) {
                if (_immediate_dominators[predecessor] ==
                    invalid_block_id) {
                    continue;
                }
                new_idom = new_idom == invalid_block_id ?
                               predecessor :
                               intersect(predecessor, new_idom);
            }
            if (new_idom == invalid_block_id) { continue; }
            if (_immediate_dominators[block_id] != new_idom) {
                _immediate_dominators[block_id] = new_idom;
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    if (std::find(_immediate_dominators.begin(),
                  _immediate_dominators.end(),
                  invalid_block_id) !=
        _immediate_dominators.end()) {
        return;
    }

    luisa::vector<luisa::vector<size_t>> children(_blocks.size());
    for (size_t block_id = 1u;
         block_id < _blocks.size(); ++block_id) {
        children[_immediate_dominators[block_id]].emplace_back(block_id);
    }
    _preorder_indices.assign(_blocks.size(), invalid_block_id);
    _subtree_end_indices.assign(_blocks.size(), invalid_block_id);
    auto next_preorder = size_t{0u};
    _preorder_indices.front() = next_preorder++;
    stack.clear();
    stack.emplace_back(DfsFrame{0u, 0u});
    while (!stack.empty()) {
        auto &frame = stack.back();
        auto &block_children = children[frame.block_id];
        if (frame.next_successor < block_children.size()) {
            auto child = block_children[frame.next_successor++];
            _preorder_indices[child] = next_preorder++;
            stack.emplace_back(DfsFrame{child, 0u});
        } else {
            _subtree_end_indices[frame.block_id] = next_preorder;
            stack.pop_back();
        }
    }
    if (next_preorder != _blocks.size()) { return; }
    _valid = true;
}

bool CoroSemanticGraph::dominates(
    BasicBlock *definition, BasicBlock *use) const noexcept {
    if (!_valid || definition == nullptr || use == nullptr) {
        return false;
    }
    auto def_iter = _block_ids.find(definition);
    auto use_iter = _block_ids.find(use);
    if (def_iter == _block_ids.end() || use_iter == _block_ids.end()) {
        return false;
    }
    auto def = def_iter->second;
    auto observed = use_iter->second;
    return _preorder_indices[def] <= _preorder_indices[observed] &&
           _preorder_indices[observed] < _subtree_end_indices[def];
}

}// namespace luisa::compute::xir::detail
