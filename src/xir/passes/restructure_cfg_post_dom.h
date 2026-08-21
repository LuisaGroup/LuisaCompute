#pragma once

#include <cstddef>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class BasicBlock;
class FunctionDefinition;

namespace detail {

struct RestructurePostDomInfo {
    struct Node {
        BasicBlock *parent{nullptr};
        size_t depth{0u};
        luisa::vector<BasicBlock *> children;
        size_t update_epoch{0u};
    };

    luisa::unordered_map<BasicBlock *, Node> nodes;
    BasicBlock *virtual_exit{nullptr};
    size_t update_epoch{0u};

    [[nodiscard]] BasicBlock *immediate_postdom(
        BasicBlock *block) const noexcept;
    // Nearest common ancestor in the sparse immediate-postdominator tree.
    // The synthetic virtual exit is reported as nullptr, matching the
    // historical set-intersection query.
    [[nodiscard]] BasicBlock *nearest_common_postdom(
        luisa::span<BasicBlock *const> blocks,
        size_t *ancestor_step_count = nullptr) const noexcept;

    struct TransparentMergeUpdateStats {
        size_t candidate_block_count{0u};
        size_t block_evaluation_count{0u};
        size_t edge_visit_count{0u};
        size_t covered_block_count{0u};
        size_t reparented_root_count{0u};
    };

    // Update the exact tree after replacing a non-empty subset of executable
    // edges `u -> successor` by `u -> block -> successor`. The caller
    // guarantees that all changed old edges originate in the old strict
    // postdom subtree of `successor`; old nodes and all other executable edges
    // are unchanged. Returns false when the mutation is outside this
    // transparent-funnel model and the caller must rebuild the tree.
    [[nodiscard]] bool insert_transparent_merge(
        BasicBlock *block,
        BasicBlock *successor,
        TransparentMergeUpdateStats *stats = nullptr) noexcept;

    [[nodiscard]] bool structurally_equals(
        const RestructurePostDomInfo &other) const noexcept;
};

struct RestructurePostDomStats {
    size_t numbered_block_count{0u};
    size_t numbered_edge_count{0u};
    size_t active_block_count{0u};
    size_t fixed_point_iteration_count{0u};
    size_t fixed_point_block_visit_count{0u};
    size_t fixed_point_edge_visit_count{0u};
    size_t intersect_step_count{0u};
};

[[nodiscard]] bool is_restructure_cfg_sink(
    BasicBlock *block) noexcept;

[[nodiscard]] RestructurePostDomInfo
compute_restructure_post_dom(
    FunctionDefinition *definition,
    RestructurePostDomStats *stats = nullptr) noexcept;

}// namespace detail

}// namespace luisa::compute::xir
