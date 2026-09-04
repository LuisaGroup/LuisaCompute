#pragma once

#include <luisa/ast/usage.h>
#include <luisa/tile/ir.h>
#include <luisa/xir/module.h>

namespace luisa::compute::tile::bridge::xir {

struct LowerOptions {
    uint32_t block_size{64u};
    // Bound compile-time expansion; never truncate a Tile or its operations.
    uint32_t max_expanded_values{262144u};
    // Empty preserves declaration order, useful as a fixed baseline.
    luisa::vector<uint32_t> root_axis_order;
};

struct NativeFunction {
    luisa::unique_ptr<compute::xir::Module> module;
    compute::xir::KernelFunction *function{nullptr};
    uint32_t dispatch_size{0u};
    luisa::vector<Usage> argument_usages;
    luisa::vector<size_t> argument_sizes_bytes;
    luisa::string error;
    [[nodiscard]] bool ok() const noexcept { return module != nullptr && function != nullptr && error.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// In-memory, verified SSA/CFG bridge, with no AST or TVM intermediate.
// One root parallel domain maps to independent Runtime workers. Static Tile
// elements remain distinct SSA values inside each worker; the SIMD backend
// packs workers, not the logical Tile's memory dimensions. Serial/reduction
// and pipeline recurrences preserve lexicographic order. This first CPU
// realization does not implement cooperative bindings or manual Memory.
[[nodiscard]] LUISA_TILE_XIR_BRIDGE_API NativeFunction lower(
    const Function &function, const LowerOptions &options = {}) noexcept;

}// namespace luisa::compute::tile::bridge::xir
