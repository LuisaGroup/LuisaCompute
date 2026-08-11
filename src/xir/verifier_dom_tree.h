#pragma once

#include <luisa/core/stl/vector.h>

#include "verifier_containers.h"

namespace luisa::compute::xir {

class BasicBlock;

namespace detail {

using VerifierBlockSet =
    VerifierPointerSet<const BasicBlock *>;
using VerifierBlockAdjacency =
    VerifierPointerMap<const BasicBlock *, VerifierBlockSet>;

// Sparse immediate-dominator tree over a verifier-sanitized CFG.
//
// The verifier cannot use the ordinary DomTree builder until it has proved
// that every CFG operand belongs to the function under verification. This
// class therefore consumes the verifier's locally-owned reachable graph. It
// assigns dense numeric IDs to blocks, but never materializes the dense
// dominance relation: construction uses a sparse predecessor CSR and the
// resulting tree stores one immediate-dominator parent per reachable block.
class VerifierSparseDomTree {

private:
    static constexpr auto invalid_index = ~size_t{0u};
    VerifierPointerMap<const BasicBlock *, size_t> _block_indices;
    luisa::vector<size_t> _immediate_dominators;
    luisa::vector<size_t> _depths;
    luisa::vector<size_t> _preorder_indices;
    luisa::vector<size_t> _subtree_end_indices;
    size_t _cfg_edge_count{0u};
    size_t _fixed_point_iteration_count{0u};

private:
    [[nodiscard]] size_t _intersect(size_t lhs,
                                    size_t rhs) const noexcept;

public:
    VerifierSparseDomTree(
        const BasicBlock *root,
        const VerifierBlockAdjacency &successors,
        const VerifierBlockAdjacency &predecessors,
        const VerifierBlockSet &reachable) noexcept;

    [[nodiscard]] size_t size() const noexcept {
        return _immediate_dominators.size();
    }
    [[nodiscard]] size_t tree_edge_count() const noexcept {
        return size() == 0u ? 0u : size() - 1u;
    }
    [[nodiscard]] size_t cfg_edge_count() const noexcept {
        return _cfg_edge_count;
    }
    [[nodiscard]] size_t fixed_point_iteration_count() const noexcept {
        return _fixed_point_iteration_count;
    }
    [[nodiscard]] bool dominates(
        const BasicBlock *dominator,
        const BasicBlock *block) const noexcept;
    [[nodiscard]] size_t depth(
        const BasicBlock *block) const noexcept;
};

}// namespace detail

}// namespace luisa::compute::xir
