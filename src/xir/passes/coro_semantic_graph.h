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
