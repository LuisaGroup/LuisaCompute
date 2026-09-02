#pragma once

#include <tvm/tirx/stmt.h>

namespace luisa::compute::tile::bridge::tirx::detail {

// The structural bridge preserves logical domains and hard scope constraints.
// The target mapper consumes these only after resolving a legal realization.
inline constexpr auto logical_parallel_annotation = "luisa.tile.logical_parallel";
inline constexpr auto execution_scope_annotation = "luisa.tile.execution_scope";
// Positive rank of a perfect, rectangular serial element-loop nest. Keep its
// axes intact until the selected target actually needs worker partitioning.
inline constexpr auto independent_elements_annotation = "luisa.tile.independent_elements";
// Hard resource constraints survive structural export until target binding.
inline constexpr auto memory_resource_annotation = "luisa.tile.memory_resource";

// Give every logical vector lane its own compiler-local storage before TIRx
// vectorization. TIRx currently does not privatize AllocBuffer itself.
[[nodiscard]] tvm::tirx::Stmt privatize_vector_storage(const tvm::tirx::For &loop);

// Realize one logical group per Metal threadgroup. Independent element
// domains and child workers share group-owned compiler temporaries.
[[nodiscard]] tvm::tirx::Stmt map_metal_cooperative_group(
    const tvm::tirx::For &loop, uint32_t max_threads, uint64_t shared_memory_limit);

}// namespace luisa::compute::tile::bridge::tirx::detail
