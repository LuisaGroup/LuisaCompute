#pragma once

#include <cstddef>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::compute::xir {

class BasicBlock;
class FunctionDefinition;

namespace detail {

struct SelectionMergeBatchStats {
    size_t loop_context_count{0u};
    size_t query_count{0u};
    size_t block_visit_count{0u};
    size_t edge_visit_count{0u};
};

// Reusable dense workspace for selection-merge queries on one immutable CFG
// quotient. The caller may insert transparent merge subdivisions between
// queries and register their immutable dominance anchors.
class SelectionMergeBatchAnalysis final {
private:
    class Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    SelectionMergeBatchAnalysis(
        FunctionDefinition *definition,
        const DomTree &dominance) noexcept;
    ~SelectionMergeBatchAnalysis() noexcept;
    SelectionMergeBatchAnalysis(
        SelectionMergeBatchAnalysis &&) noexcept;
    SelectionMergeBatchAnalysis &operator=(
        SelectionMergeBatchAnalysis &&) noexcept;
    SelectionMergeBatchAnalysis(
        const SelectionMergeBatchAnalysis &) = delete;
    SelectionMergeBatchAnalysis &operator=(
        const SelectionMergeBatchAnalysis &) = delete;

    void register_overlay_block(BasicBlock *block) noexcept;

    [[nodiscard]] BasicBlock *infer(
        BasicBlock *header,
        luisa::span<BasicBlock *const> entries,
        const luisa::unordered_map<BasicBlock *, BasicBlock *> *
            dominance_anchors = nullptr) noexcept;

    [[nodiscard]] const SelectionMergeBatchStats &stats() const noexcept;
};

// Return the terminal block of an acyclic chain of empty branches, or the
// first cycle-entry block of a cyclic chain. Uses O(1) scratch storage.
[[nodiscard]] BasicBlock *canonical_trivial_branch_chain_target(
    BasicBlock *target) noexcept;

}// namespace detail
}// namespace luisa::compute::xir
