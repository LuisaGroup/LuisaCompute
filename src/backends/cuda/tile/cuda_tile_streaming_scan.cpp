// Apply CUDA realization/ABI constraints to the shared closed-prefix proof.
#include "cuda_tile_streaming_scan.h"
#include <luisa/tile/collective_prefix.h>
#include <bit>

namespace luisa::compute::cuda::native_tile {

StreamingScanPlan match_streaming_scan(const tile::Function &function, const Artifact &original,
                                       uint32_t chunk) noexcept {
    StreamingScanPlan plan;
    auto fail = [&](luisa::string_view why) noexcept {
        plan.error.assign(why.data(), why.size());
        return std::move(plan);
    };
    if (!original.ok() || original.scan_chunk_extent != 0u || original.independent_axis_extent != 0u ||
        (chunk != 1024u && chunk != 2048u)) { return fail("requires successful untransformed native artifact and chunk 1024/2048"); }
    auto proof = tile::analyze_closed_prefix(function);
    if (!proof.ok()) { return fail(proof.error); }
    if (function.body().block(0u)->argument_count() != original.arguments.size()) {
        return fail("requires original direct-buffer ABI");
    }
    plan.storage = proof.storage;
    plan.rows = proof.logical_independent_extent;
    plan.columns = proof.logical_contribution_extent;
    auto row_extent = proof.original.independent_extent_per_program;
    plan.padded_columns = proof.collective.contribution_extent;
    if (plan.rows > 0x7fffffffu || plan.columns > 65536u ||
        !std::has_single_bit(row_extent) || row_extent > 8u ||
        plan.padded_columns != std::bit_ceil(plan.columns) || plan.padded_columns <= chunk) {
        return fail("requires power-of-two BR<=8, padded prefix width<=65536, and at least two chunks");
    }
    plan.rows_per_program = static_cast<uint32_t>(row_extent);
    if (original.grid != std::array<uint32_t, 3u>{static_cast<uint32_t>(proof.original.programs), 1u, 1u} ||
        original.block != std::array<uint32_t, 3u>{1u, 1u, 1u}) {
        return fail("original launch does not match proved program geometry");
    }
    plan.minimum_bytes = proof.disjoint.input.byte_count;
    plan.input_slot = proof.disjoint.input.argument_index;
    plan.output_slot = proof.disjoint.output.argument_index;
    if (plan.input_slot >= original.arguments.size() || plan.output_slot >= original.arguments.size()) { return fail("root slot outside original ABI"); }
    auto &ia = original.arguments[plan.input_slot];
    auto &oa = original.arguments[plan.output_slot];
    if (!ia.read || ia.written || oa.read || !oa.written || ia.minimum_size_bytes != plan.minimum_bytes ||
        oa.minimum_size_bytes != plan.minimum_bytes || ia.element != plan.storage || oa.element != plan.storage) {
        return fail("original resource metadata disagrees with the closed load/store chain");
    }
    for (auto slot = size_t{0u}; slot < original.arguments.size(); slot++) {
        if (slot != plan.input_slot && slot != plan.output_slot && (original.arguments[slot].read || original.arguments[slot].written)) {
            return fail("another live resource would invalidate closed streaming effects");
        }
    }
    plan.chunk_extent = chunk;
    return plan;
}

}// namespace luisa::compute::cuda::native_tile
