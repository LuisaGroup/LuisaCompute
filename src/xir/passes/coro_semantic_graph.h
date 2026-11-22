#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class BasicBlock;
class FunctionDefinition;

namespace detail {

// Dense, sparse-edge view of the executable coroutine CFG. CoroSuspend is a
// terminator in the ordinary CFG, but execution continues later at the unique
// CoroResume carrying the same token. This graph adds exactly those semantic
// edges and computes dominance over the resulting state machine.
class CoroSemanticGraph {

private:
    bool _valid{false};
    size_t _edge_count{0u};
    luisa::vector<BasicBlock *> _blocks;
    luisa::unordered_map<BasicBlock *, size_t> _block_ids;
    luisa::vector<luisa::vector<size_t>> _predecessors;
    luisa::vector<luisa::vector<size_t>> _successors;
    luisa::vector<std::pair<size_t, size_t>> _suspend_edges;
    luisa::vector<luisa::vector<uint8_t>> _can_reach_suspend;
    luisa::vector<luisa::vector<uint8_t>> _reachable_from_resume;
    luisa::vector<size_t> _immediate_dominators;
    luisa::vector<size_t> _preorder_indices;
    luisa::vector<size_t> _subtree_end_indices;
    mutable luisa::vector<uint32_t> _bfs_visit_epochs;
    mutable uint32_t _bfs_epoch{0u};
    mutable luisa::vector<size_t> _bfs_worklist;
    mutable luisa::unordered_map<size_t, luisa::vector<uint64_t>>
        _crossed_from_def_cache;
    mutable luisa::vector<uint32_t> _dom_lift;
    mutable uint32_t _dom_lift_levels{0u};

    void _ensure_dom_lift() const noexcept;

    [[nodiscard]] bool _dominates_id(
        size_t definition_id, size_t use_id) const noexcept;

public:
    explicit CoroSemanticGraph(
        FunctionDefinition *definition) noexcept;

    [[nodiscard]] bool valid() const noexcept { return _valid; }
    [[nodiscard]] size_t block_count() const noexcept {
        return _blocks.size();
    }
    [[nodiscard]] size_t edge_count() const noexcept {
        return _edge_count;
    }
    [[nodiscard]] BasicBlock *block(size_t id) const noexcept {
        return id < _blocks.size() ? _blocks[id] : nullptr;
    }
    [[nodiscard]] size_t block_id(BasicBlock *block) const noexcept {
        if (auto iter = _block_ids.find(block);
            iter != _block_ids.end()) {
            return iter->second;
        }
        return _blocks.size();
    }
    [[nodiscard]] luisa::span<const size_t>
    predecessors(size_t id) const noexcept {
        return id < _predecessors.size() ?
                   luisa::span<const size_t>{_predecessors[id]} :
                   luisa::span<const size_t>{};
    }
    [[nodiscard]] luisa::span<const size_t>
    successors(size_t id) const noexcept {
        return id < _successors.size() ?
                   luisa::span<const size_t>{_successors[id]} :
                   luisa::span<const size_t>{};
    }
    [[nodiscard]] bool contains(BasicBlock *block) const noexcept {
        return _block_ids.contains(block);
    }
    [[nodiscard]] bool dominates(
        BasicBlock *definition,
        BasicBlock *use) const noexcept;
    [[nodiscard]] bool dominates_id(
        size_t definition_id, size_t use_id) const noexcept {
        return _dominates_id(definition_id, use_id);
    }
    [[nodiscard]] size_t preorder_index(size_t block_id) const noexcept {
        return _preorder_indices[block_id];
    }
    [[nodiscard]] size_t subtree_end_index(size_t block_id) const noexcept {
        return _subtree_end_indices[block_id];
    }
    [[nodiscard]] luisa::span<const std::pair<size_t, size_t>>
    suspend_edges() const noexcept {
        return _suspend_edges;
    }
    [[nodiscard]] size_t shallowest_dominator_avoiding_preorders(
        size_t block_id,
        luisa::span<const size_t> sorted_landmark_preorders) const noexcept;
    // Returns the deepest block that dominates every input in the augmented
    // coroutine CFG. Suspend(token) -> resume(token) edges participate in the
    // same dominance relation, so the result is suitable for state whose
    // lifetime is local to one continuation iteration.
    [[nodiscard]] BasicBlock *nearest_common_dominator(
        luisa::span<BasicBlock *const> blocks) const noexcept;
    // Returns true when one semantic suspend->resume edge lies on some block
    // path from definition to use. This is a cheap necessary condition used
    // before the exact reaching-state fixed point.
    [[nodiscard]] bool may_cross_suspend_between(
        BasicBlock *definition,
        BasicBlock *use) const noexcept;
    // Exact single-static-store query. Re-entering the definition block
    // executes the store again and resets the cross-suspend state, so such
    // paths are barriers rather than evidence that the stored dynamic value
    // must survive a suspension.
    [[nodiscard]] bool crosses_suspend_without_reentering(
        BasicBlock *definition,
        BasicBlock *use) const noexcept;
    [[nodiscard]] bool is_suspend_edge(
        size_t predecessor, size_t successor) const noexcept;
};

}// namespace detail
}// namespace luisa::compute::xir
