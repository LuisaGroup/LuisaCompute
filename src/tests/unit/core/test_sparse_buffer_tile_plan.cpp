// Test for runtime sparse buffer tile-region and heap validation.
// This test covers:
// - overflow-safe ceil division for partial final tiles
// - nonzero counts, checked tile ends, and out-of-range regions
// - sparse heap validity and device provenance

#include "ut/ut.hpp"

#include <luisa/runtime/sparse_buffer.h>

#include "sparse_buffer_tile_plan.h"
#include "sparse_heap_provenance.h"

#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;
namespace runtime_detail = luisa::compute::detail;

void instantiate_sparse_buffer_tile_updates(
    SparseBuffer<float> &buffer,
    const SparseBufferHeap &heap) {
    [[maybe_unused]] auto map = buffer.map_tile(0u, 1u, heap);
    [[maybe_unused]] auto unmap = buffer.unmap_tile(0u, 1u);
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "sparse_buffer_tile_plan_accepts_partial_edge_tile"_test = [] {
        constexpr auto plan = runtime_detail::plan_sparse_buffer_tile_region({.size_bytes = 257u,
                                                                              .tile_size_bytes = 128u,
                                                                              .start_tile = 2u,
                                                                              .tile_count = 1u});
        static_assert(plan);
        static_assert(plan.tile_grid_size == 3u);
        static_assert(plan.end_tile == 3u);
        expect(static_cast<bool>(plan));

        constexpr auto max_size =
            std::numeric_limits<size_t>::max();
        constexpr auto large = runtime_detail::plan_sparse_buffer_tile_region({.size_bytes = max_size,
                                                                               .tile_size_bytes = 2u,
                                                                               .start_tile = 0u,
                                                                               .tile_count = 1u});
        static_assert(large);
        static_assert(
            large.tile_grid_size == max_size / 2u + 1u);
        expect(static_cast<bool>(large))
            << "ceil division must not overflow size_bytes + tile_size - 1";
    };

    "sparse_buffer_tile_plan_rejects_zero_count_bounds_and_overflow"_test = [] {
        using Status = runtime_detail::SparseBufferTileRegionStatus;
        constexpr auto zero_count =
            runtime_detail::plan_sparse_buffer_tile_region({.size_bytes = 256u,
                                                            .tile_size_bytes = 128u,
                                                            .start_tile = 0u,
                                                            .tile_count = 0u});
        expect(zero_count.status == Status::ZERO_TILE_COUNT)
            << "map and unmap must both validate tile_count";

        constexpr auto out_of_range =
            runtime_detail::plan_sparse_buffer_tile_region({.size_bytes = 257u,
                                                            .tile_size_bytes = 128u,
                                                            .start_tile = 2u,
                                                            .tile_count = 2u});
        expect(out_of_range.status == Status::OUT_OF_RANGE);

        constexpr auto max_u32 =
            std::numeric_limits<uint32_t>::max();
        constexpr auto overflow =
            runtime_detail::plan_sparse_buffer_tile_region({.size_bytes = 256u,
                                                            .tile_size_bytes = 1u,
                                                            .start_tile = max_u32,
                                                            .tile_count = 2u});
        expect(overflow.status == Status::ARITHMETIC_OVERFLOW);
    };

    "sparse_heap_provenance_requires_valid_same_device_heap"_test = [] {
        using Status = runtime_detail::SparseHeapProvenanceStatus;
        int32_t resource_device{};
        int32_t other_device{};
        expect(runtime_detail::validate_sparse_heap_provenance(
                   true, &resource_device, &resource_device) ==
               Status::SUCCESS);
        expect(runtime_detail::validate_sparse_heap_provenance(
                   false, &resource_device, &resource_device) ==
               Status::INVALID_HEAP);
        expect(runtime_detail::validate_sparse_heap_provenance(
                   true, &resource_device, &other_device) ==
               Status::DEVICE_MISMATCH);
        expect(runtime_detail::validate_sparse_heap_provenance(
                   true, nullptr, nullptr) ==
               Status::DEVICE_MISMATCH);
    };
}
