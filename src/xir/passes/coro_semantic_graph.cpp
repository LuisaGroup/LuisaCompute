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
    luisa::vector<std::pair<size_t, size_t>> owned_suspend_edges;
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
                    owned_suspend_edges.emplace_back(
                        block_id, iter->second);
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
    _suspend_edges.reserve(owned_suspend_edges.size());
    for (auto [from_owned, to_owned] : owned_suspend_edges) {
        auto from = owned_to_rpo[from_owned];
        auto to = owned_to_rpo[to_owned];
        if (from != invalid_block_id && to != invalid_block_id) {
            _suspend_edges.emplace_back(from, to);
        }
    }
    std::sort(_suspend_edges.begin(), _suspend_edges.end());
    _suspend_edges.erase(
        std::unique(_suspend_edges.begin(), _suspend_edges.end()),
        _suspend_edges.end());
    _can_reach_suspend.reserve(_suspend_edges.size());
    _reachable_from_resume.reserve(_suspend_edges.size());
    for (auto [suspend, resume] : _suspend_edges) {
        auto reverse_reachable = luisa::vector<uint8_t>(
            _blocks.size(), 0u);
        auto forward_reachable = luisa::vector<uint8_t>(
            _blocks.size(), 0u);
        luisa::vector<size_t> worklist{suspend};
        reverse_reachable[suspend] = 1u;
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            for (auto predecessor : _predecessors[worklist[cursor]]) {
                if (reverse_reachable[predecessor] == 0u) {
                    reverse_reachable[predecessor] = 1u;
                    worklist.emplace_back(predecessor);
                }
            }
        }
        worklist.clear();
        worklist.emplace_back(resume);
        forward_reachable[resume] = 1u;
        for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
            for (auto successor : _successors[worklist[cursor]]) {
                if (forward_reachable[successor] == 0u) {
                    forward_reachable[successor] = 1u;
                    worklist.emplace_back(successor);
                }
            }
        }
        _can_reach_suspend.emplace_back(
            std::move(reverse_reachable));
        _reachable_from_resume.emplace_back(
            std::move(forward_reachable));
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

bool CoroSemanticGraph::_dominates_id(
    size_t definition_id, size_t use_id) const noexcept {
    return _preorder_indices[definition_id] <= _preorder_indices[use_id] &&
           _preorder_indices[use_id] < _subtree_end_indices[definition_id];
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
    return _dominates_id(def_iter->second, use_iter->second);
}

BasicBlock *CoroSemanticGraph::nearest_common_dominator(
    luisa::span<BasicBlock *const> blocks) const noexcept {
    if (!_valid || blocks.empty()) { return nullptr; }
    auto first = _block_ids.find(blocks.front());
    if (first == _block_ids.end()) { return nullptr; }
    auto common = first->second;
    auto intersect = [&](size_t lhs, size_t rhs) noexcept {
        while (lhs != rhs) {
            while (lhs > rhs) {
                lhs = _immediate_dominators[lhs];
            }
            while (rhs > lhs) {
                rhs = _immediate_dominators[rhs];
            }
        }
        return lhs;
    };
    for (size_t i = 1u; i < blocks.size(); ++i) {
        auto iter = _block_ids.find(blocks[i]);
        if (iter == _block_ids.end()) { return nullptr; }
        common = intersect(common, iter->second);
    }
    return _blocks[common];
}

bool CoroSemanticGraph::may_cross_suspend_between(
    BasicBlock *definition, BasicBlock *use) const noexcept {
    if (!_valid || definition == nullptr || use == nullptr) {
        return false;
    }
    auto definition_iter = _block_ids.find(definition);
    auto use_iter = _block_ids.find(use);
    if (definition_iter == _block_ids.end() ||
        use_iter == _block_ids.end()) {
        return false;
    }
    for (size_t i = 0u; i < _suspend_edges.size(); ++i) {
        if (_can_reach_suspend[i][definition_iter->second] != 0u &&
            _reachable_from_resume[i][use_iter->second] != 0u) {
            return true;
        }
    }
    return false;
}

bool CoroSemanticGraph::crosses_suspend_without_reentering(
    BasicBlock *definition, BasicBlock *use) const noexcept {
    if (!_valid || definition == nullptr || use == nullptr ||
        _suspend_edges.empty()) {
        return false;
    }
    auto definition_id = block_id(definition);
    auto use_id = block_id(use);
    if (definition_id >= _blocks.size() ||
        use_id >= _blocks.size() || definition_id == use_id) {
        return false;
    }
    // Necessary condition over suspend->resume edges:
    // - suspend edge i must lie between definition and use;
    // - its resume target must not re-enter definition itself;
    // - when definition dominates use, any path resume ->* use avoiding
    //   definition requires definition to strictly dominate resume.
    auto def_dom_use = _dominates_id(definition_id, use_id);
    auto possible = false;
    for (size_t i = 0u; i < _suspend_edges.size(); ++i) {
        auto resume = _suspend_edges[i].second;
        if (resume == definition_id) { continue; }
        if (def_dom_use && !_dominates_id(definition_id, resume)) { continue; }
        if (_can_reach_suspend[i][definition_id] != 0u &&
            _reachable_from_resume[i][use_id] != 0u) {
            possible = true;
            break;
        }
    }
    if (!possible) { return false; }
    if (auto cached = _crossed_from_def_cache.find(definition_id);
        cached != _crossed_from_def_cache.end()) {
        return (cached->second[use_id / 64u] &
                (uint64_t{1u} << (use_id % 64u))) != 0u;
    }

    // State is (block, crossed_suspend). The definition block is excluded
    // after the initial store occurrence because re-entering it executes that
    // same static store before any dominated load and starts a new version.
    constexpr size_t max_cached_sources = 256u;
    const auto cache_result = _crossed_from_def_cache.size() < max_cached_sources;
    if (_bfs_visit_epochs.size() != _blocks.size() * 2u) {
        _bfs_visit_epochs.assign(_blocks.size() * 2u, 0u);
        _bfs_epoch = 1u;
    } else if (++_bfs_epoch == 0u) {
        std::fill(_bfs_visit_epochs.begin(), _bfs_visit_epochs.end(), 0u);
        _bfs_epoch = 1u;
    }
    const auto epoch = _bfs_epoch;
    _bfs_worklist.clear();
    auto enqueue = [&](size_t block, bool crossed) noexcept {
        if (block == definition_id) { return; }
        auto state = block * 2u + static_cast<size_t>(crossed);
        if (_bfs_visit_epochs[state] != epoch) {
            _bfs_visit_epochs[state] = epoch;
            _bfs_worklist.emplace_back(state);
        }
    };
    for (auto successor : _successors[definition_id]) {
        enqueue(
            successor,
            is_suspend_edge(definition_id, successor));
    }
    luisa::vector<uint64_t> crossed_bits;
    if (cache_result) {
        crossed_bits.assign((_blocks.size() + 63u) / 64u, 0u);
    }
    for (size_t cursor = 0u; cursor < _bfs_worklist.size(); ++cursor) {
        auto state = _bfs_worklist[cursor];
        auto block = state / 2u;
        auto crossed = (state & 1u) != 0u;
        if (crossed) {
            if (cache_result) {
                crossed_bits[block / 64u] |= uint64_t{1u} << (block % 64u);
            } else if (block == use_id) {
                return true;
            }
        }
        for (auto successor : _successors[block]) {
            enqueue(
                successor,
                crossed || is_suspend_edge(block, successor));
        }
    }
    if (!cache_result) { return false; }
    auto result = (crossed_bits[use_id / 64u] &
                   (uint64_t{1u} << (use_id % 64u))) != 0u;
    _crossed_from_def_cache.emplace(definition_id, std::move(crossed_bits));
    return result;
}

bool CoroSemanticGraph::is_suspend_edge(
    size_t predecessor, size_t successor) const noexcept {
    return std::binary_search(
        _suspend_edges.begin(), _suspend_edges.end(),
        std::pair{predecessor, successor});
}

void CoroSemanticGraph::_ensure_dom_lift() const noexcept {
    if (_dom_lift_levels != 0u) { return; }
    const auto n = _blocks.size();
    if (n <= 1u) {
        _dom_lift_levels = 1u;
        return;
    }
    uint32_t levels = 1u;
    while ((size_t{1u} << (levels - 1u)) < n) {
        ++levels;
    }
    _dom_lift_levels = levels;
    _dom_lift.resize(static_cast<size_t>(levels) * n);
    _dom_lift[0u] = 0u;
    for (size_t i = 1u; i < n; ++i) {
        _dom_lift[i] = static_cast<uint32_t>(_immediate_dominators[i]);
    }
    for (uint32_t k = 1u; k < levels; ++k) {
        const auto prev_base = static_cast<size_t>(k - 1u) * n;
        const auto cur_base = static_cast<size_t>(k) * n;
        for (size_t i = 0u; i < n; ++i) {
            _dom_lift[cur_base + i] = _dom_lift[prev_base + _dom_lift[prev_base + i]];
        }
    }
}

size_t CoroSemanticGraph::shallowest_dominator_avoiding_preorders(
    size_t block_id,
    luisa::span<const size_t> sorted_landmark_preorders) const noexcept {
    const auto n = _blocks.size();
    if (block_id == 0u || block_id >= n) { return block_id; }
    auto has_landmark = [&](size_t x) noexcept {
        auto pre = _preorder_indices[x];
        auto end = _subtree_end_indices[x];
        auto it = std::lower_bound(
            sorted_landmark_preorders.begin(),
            sorted_landmark_preorders.end(),
            pre);
        return it != sorted_landmark_preorders.end() && *it < end;
    };
    if (has_landmark(block_id)) { return block_id; }
    _ensure_dom_lift();
    size_t cur = block_id;
    for (uint32_t k = _dom_lift_levels; k > 0u; --k) {
        auto anc = static_cast<size_t>(_dom_lift[static_cast<size_t>(k - 1u) * n + cur]);
        if (!has_landmark(anc)) {
            cur = anc;
        }
    }
    return cur;
}

}// namespace luisa::compute::xir::detail
