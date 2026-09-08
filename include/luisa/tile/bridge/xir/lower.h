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
    // Total compiler-owned snapshot storage per logical worker (not packet).
    // A hard bound, not a peak-liveness or target stack-size estimate.
    uint32_t max_local_bytes{262144u};
    // Larger Tile traversals use runtime loops. Zero keeps the fully expanded
    // realization as an explicit diagnostic baseline; it does not remove budgets.
    uint32_t max_unrolled_tile_elements{64u};
    // Bounded pure unordered reductions may partition contributions among
    // these independent accumulators. One preserves the sequential baseline.
    uint32_t reduction_partitions{4u};
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
// elements use SSA or bounded traversal of compiler-owned snapshots; pure
// single-use elementwise values may be deferred to their consumer. The SIMD
// backend packs workers, not the logical Tile's memory dimensions. Closed
// unordered reductions may use independent partials. Other recurrences
// preserve lexicographic order; explicit right folds
// visit the reversed logical sequence without changing update operands. This CPU
// realization does not implement cooperative bindings or manual Memory.
[[nodiscard]] LUISA_TILE_XIR_BRIDGE_API NativeFunction lower(
    const Function &function, const LowerOptions &options = {}) noexcept;

}// namespace luisa::compute::tile::bridge::xir
