#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>

namespace luisa::compute::xir {

class Function;
class DomTree;

struct ConvergenceRegion {
    BasicBlock *entry{nullptr};
    BasicBlock *convergence_merge{nullptr};
    luisa::unordered_set<BasicBlock *> blocks;
    ConvergenceRegion *parent{nullptr};
    luisa::vector<luisa::unique_ptr<ConvergenceRegion>> children;
};

struct ConvergenceRegionInfo {
    luisa::unique_ptr<ConvergenceRegion> top_level;
    [[nodiscard]] LUISA_XIR_API const ConvergenceRegion *find_region(BasicBlock *bb) const noexcept;
};

[[nodiscard]] LUISA_XIR_API ConvergenceRegionInfo compute_convergence_regions(
    Function *function, const DomTree &dom) noexcept;

}// namespace luisa::compute::xir
