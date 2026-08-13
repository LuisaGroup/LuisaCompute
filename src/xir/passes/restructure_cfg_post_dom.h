#pragma once

#include <cstddef>

#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class BasicBlock;
class FunctionDefinition;

namespace detail {

struct RestructurePostDomInfo {
    luisa::unordered_map<BasicBlock *, BasicBlock *> ipostdom;
    BasicBlock *virtual_exit{nullptr};
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
