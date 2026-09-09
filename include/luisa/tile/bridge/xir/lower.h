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
    // One keeps complete independent programs per physical lane. A power-of-
    // two packet width distributes a common local axis across the whole packet.
    // The caller must compile with this exact packet width; lower() validates
    // the admitted pointwise/closed-unordered-reduction program contract.
    uint32_t local_lanes{1u};
    // Fuse a load with its first pointwise unordered reduction, retaining a
    // snapshot for later consumers. Never moves reads across writes/stages.
    // Opt-in: fewer private reads can still increase masked-memory/CFG cost.
    bool enable_load_reduction_fusion{false};
    // Version closed common-domain pointwise regions: stream shared SSA DAGs
    // when resource intervals are disjoint, otherwise retain eager snapshots.
    // Opt-in while native profitability and resource costs are evaluated.
    bool enable_pointwise_fusion{false};
    // Move a materialized pure expression into its first reduction traversal.
    // Compute each point once, retaining its snapshot for later consumers.
    // Independent opt-in; preserves the chosen reduction tree and math policy.
    bool enable_expression_reduction_fusion{false};
};

struct NativeFunction {
    luisa::unique_ptr<compute::xir::Module> module;
    compute::xir::KernelFunction *function{nullptr};
    uint32_t dispatch_size{0u};
    luisa::vector<Usage> argument_usages;
    luisa::vector<size_t> argument_sizes_bytes;
    // Zero permits any packet width. Otherwise the consumer must preserve
    // this width: dispatch coordinates and collectives form one ABI contract.
    uint32_t required_packet_width{0u};
    // Static realization counts, not dynamic memory transactions.
    uint32_t fused_reduction_loads{0u};
    uint32_t elided_load_snapshots{0u};
    uint32_t fused_pointwise_regions{0u};
    uint32_t fused_pointwise_loads{0u};
    uint32_t fused_pointwise_stores{0u};
    uint32_t pointwise_alias_checks{0u};
    uint32_t fused_reduction_expressions{0u};
    uint32_t elided_expression_snapshots{0u};
    luisa::string error;
    [[nodiscard]] bool ok() const noexcept { return module != nullptr && function != nullptr && error.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// In-memory, verified SSA/CFG bridge, with no AST or TVM intermediate.
// One root parallel domain maps to independent logical programs. Static Tile
// elements use SSA or bounded traversal of compiler-owned snapshots; pure
// single-use elementwise values may be deferred to their consumer. The SIMD
// backend can pack whole programs or distribute a common local axis across
// a packet. Closed unordered reductions may use partials and packet shuffles.
// Other recurrences
// preserve lexicographic order; explicit right folds
// visit the reversed logical sequence without changing update operands. This CPU
// realization does not implement cooperative bindings or manual Memory.
[[nodiscard]] LUISA_TILE_XIR_BRIDGE_API NativeFunction lower(
    const Function &function, const LowerOptions &options = {}) noexcept;

}// namespace luisa::compute::tile::bridge::xir
