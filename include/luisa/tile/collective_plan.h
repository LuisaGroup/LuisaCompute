#pragma once

#include <luisa/tile/ir.h>

namespace luisa::compute::tile {

enum class CollectiveKind : uint8_t { SUM,
                                      MINIMUM,
                                      MAXIMUM,
                                      INCLUSIVE_SUM };

struct CollectiveWork {
    uint64_t operation_id{0u};
    CollectiveKind kind{CollectiveKind::SUM};
    ScalarType element{ScalarType::INVALID};
    uint64_t contribution_extent{0u};
    uint64_t independent_elements{0u};
    uint64_t input_elements{0u};
};

// Logical IR facts only: none of these are physical registers, spills, DRAM
// transactions, resident warps, instruction counts or calibrated time units.
// The first admitted family is a single static parallel region with a straight
// line of materialized Tile values, elementwise operations, view loads/stores
// and closed FP32 unordered sums/min/max/prefix sums in Tile maps. All effects
// and constraints are inspected; unsupported structure returns a diagnostic.
struct CollectiveWorkAnalysis {
    luisa::string error;
    uint64_t programs{0u};
    uint64_t elementwise_elements_per_program{0u};
    uint64_t global_read_bytes_per_program{0u};
    uint64_t global_write_bytes_per_program{0u};
    uint64_t materialized_tile_total_bytes{0u};
    // Conservative overlap of explicit direct-body Tile SSA values at each
    // operation boundary. Scalar/mapped temporaries inside regions are NOT
    // represented here, and compiler allocation can coalesce or eliminate SSA.
    uint64_t materialized_tile_peak_bytes{0u};
    uint64_t largest_materialized_tile_elements{0u};
    luisa::vector<CollectiveWork> collectives;
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !collectives.empty(); }
};

// Read-only, target-independent features for backend-owned candidate search.
// Numerical permissions/IR legality are never inferred from a cost score.
[[nodiscard]] LUISA_TILE_API CollectiveWorkAnalysis analyze_collective_work(const tile::Function &function) noexcept;

}// namespace luisa::compute::tile
