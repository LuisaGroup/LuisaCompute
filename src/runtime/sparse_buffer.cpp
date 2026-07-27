#include <luisa/runtime/sparse_buffer.h>
#include <luisa/core/logging.h>

#include "sparse_buffer_tile_plan.h"

namespace luisa::compute::detail {

LUISA_RUNTIME_API void check_sparse_buffer_tile_region(
    size_t size_bytes,
    size_t tile_size,
    uint32_t start_tile,
    uint32_t tile_count) noexcept {
    auto plan = plan_sparse_buffer_tile_region({.size_bytes = size_bytes,
                                                .tile_size_bytes = tile_size,
                                                .start_tile = start_tile,
                                                .tile_count = tile_count});
    LUISA_ASSERT(
        static_cast<bool>(plan),
        "Invalid sparse buffer tile region [{}, {}) in tile grid {}: {}.",
        start_tile, plan.end_tile, plan.tile_grid_size,
        sparse_buffer_tile_region_status_name(plan.status));
}

}// namespace luisa::compute::detail
