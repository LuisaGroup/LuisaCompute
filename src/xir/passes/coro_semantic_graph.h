#pragma once

#include <cstddef>

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
    [[nodiscard]] bool contains(BasicBlock *block) const noexcept {
        return _block_ids.contains(block);
    }
    [[nodiscard]] bool dominates(
        BasicBlock *definition,
        BasicBlock *use) const noexcept;
};

}// namespace detail
}// namespace luisa::compute::xir
