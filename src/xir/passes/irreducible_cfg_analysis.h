#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class BasicBlock;
class FunctionDefinition;

namespace detail {

struct CFGStronglyConnectedComponent {
    luisa::vector<size_t> nodes;
    luisa::vector<size_t> entry_nodes;
    bool cyclic{false};

    [[nodiscard]] bool irreducible() const noexcept {
        return cyclic && entry_nodes.size() > 1u;
    }
};

struct CFGIrreducibleRegion {
    luisa::vector<size_t> nodes;
    luisa::vector<size_t> entry_nodes;
};

struct CFGStronglyConnectedComponents {
    static constexpr auto invalid_component =
        std::numeric_limits<size_t>::max();

    luisa::vector<BasicBlock *> blocks;
    luisa::unordered_map<BasicBlock *, size_t> block_indices;
    luisa::vector<luisa::vector<size_t>> successors;
    luisa::vector<luisa::vector<size_t>> predecessors;
    luisa::vector<size_t> component_ids;
    luisa::vector<CFGStronglyConnectedComponent> components;
    // Outermost multi-entry cyclic regions in the recursive SCC decomposition.
    // Maximal SCCs alone are insufficient: after removing the unique header of
    // a natural outer loop, its remaining subgraph may expose an irreducible
    // inner cycle.
    luisa::vector<CFGIrreducibleRegion> irreducible_regions;

    [[nodiscard]] size_t irreducible_region_count() const noexcept;
};

// Iterative Kosaraju analysis over executable CFG edges, followed by recursive
// SCC decomposition through each unique loop header. Entry nodes are targets
// of edges from outside the current region, plus the function body when it
// belongs to the region. Distinct incoming edges to one header therefore still
// describe one reducible loop entry, while nested multi-entry cycles are not
// hidden by a surrounding single-entry SCC.
[[nodiscard]] CFGStronglyConnectedComponents
analyze_cfg_strongly_connected_components(
    FunctionDefinition *definition) noexcept;

}// namespace detail
}// namespace luisa::compute::xir
