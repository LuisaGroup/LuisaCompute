#pragma once

#include <luisa/tile/collective_partition.h>
#include "cuda_tile_codegen.h"

namespace luisa::compute::cuda::native_tile {

// The shared planner proves the closed IR dataflow and row partition. This
// backend imposes its own static layout and Tile shape limits. Rejection leaves
// the successful original source and launch geometry unchanged.
void append_program_partition(Artifact &original, const tile::Function &function,
                              uint32_t target_rows) noexcept;

}// namespace luisa::compute::cuda::native_tile
