#pragma once

#include <tvm/tirx/stmt.h>

namespace luisa::compute::tile::bridge::tirx::detail {

// The structural bridge preserves logical domains and hard scope constraints.
// The target mapper consumes these only after resolving a legal realization.
inline constexpr auto logical_parallel_annotation = "luisa.tile.logical_parallel";
inline constexpr auto execution_scope_annotation = "luisa.tile.execution_scope";

// Give every logical vector lane its own compiler-local storage before TIRx
// vectorization. TIRx currently does not privatize AllocBuffer itself.
[[nodiscard]] tvm::tirx::Stmt privatize_vector_storage(const tvm::tirx::For &loop);

}// namespace luisa::compute::tile::bridge::tirx::detail
