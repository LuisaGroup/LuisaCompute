#pragma once

// Optional benchmark/test composition using the existing Tile DSL. One row
// is processed per root program; no lane-dependent register gather is needed.
// Inputs exclude NaNs; ties preserve original indices and stored value bits.
#include <luisa/tile/dsl.h>
#include <bit>
#include <cstdint>
#include <limits>

namespace luisa::test::tile_selection {
using namespace compute;

template<typename T>
[[nodiscard]] tile::Kernel repeated_extrema_topk(int64_t rows, int64_t columns, int64_t count, int64_t padded) {
    using namespace tile;
    // The caller validates positive dimensions, count <= columns, columns <=
    // INT32_MAX, a power-of-two padded >= columns, and existing Tile-volume
    // limits before capture. Input/output views must not overlap: the benchmark
    // and unit allocate distinct buffers and verify the input stays unchanged.
    // The benchmark rejects NaNs before uploading.
    return tile_kernel("stable_repeated_extrema_topk", [=](TensorView<const T, 2> input,
                                                         TensorView<T, 2> output_values,
                                                         TensorView<int64_t, 2> output_indices) {
               auto local_row = axis("row", 1), column = axis("column", padded), one = axis("output", 1);
               auto domain = shape(local_row, column), output_domain = shape(local_row, one);
               for (auto &nest : parallel(shape(rows))) {
                   auto row = nest.index();
                   auto values = cast<float>(input.tile(coord(row, 0), domain).load());
                   auto indices = broadcast_to(cast<int32_t>(iota(column)), domain);
                   auto valid = indices < columns;
                   // Values stay invariant. Only the last selected total-order
                   // key crosses the serial backedge, as two singleton Tiles.
                   auto cutoff_value = full<float>(shape(local_row), std::numeric_limits<float>::infinity());
                   auto cutoff_index = full<int32_t>(shape(local_row), int32_t{-1});
                   for (auto &step : nest.serial(shape(count))) {
                       // Strictly after the previous winner in descending-value,
                       // ascending-index order. +Inf/-1 initially includes +Inf;
                       // equality treats +/-0 as ties without changing their bits.
                       auto eligible = valid && ((values < cutoff_value) ||
                                                 ((values == cutoff_value) && (indices > cutoff_index)));
                       auto peak = reduce(ite(eligible, values, -std::numeric_limits<float>::infinity()), column, maximum);
                       auto winner = reduce(ite(eligible && (values == peak), indices, std::numeric_limits<int32_t>::max()), column, minimum);
                       // winner is a singleton row Tile. Extract it outside a
                       // Tile map, giving one uniform scalar coordinate for the
                       // whole program, then select the original value bits.
                       auto selected_index = cast<int64_t>(winner.at(coord(0)));
                       // An excluded NaN input must not turn the sentinel into an
                       // out-of-bounds point load. Keep the original sentinel
                       // output index so correctness validation fails.
                       auto safe_index = ite(selected_index < columns, selected_index, int64_t{0});
                       // Load the one original storage element. A dynamic Tile
                       // extraction may materialize the whole register row.
                       auto selected_value = input.tile(coord(row, safe_index), output_domain).load();
                       output_values(coord(row, step.index()), output_domain).store(selected_value);
                       output_indices(coord(row, step.index()), output_domain).store(full<int64_t>(output_domain, selected_index));
                       cutoff_value = peak;
                       cutoff_index = winner;
                   }
               }
           }).capture(tensor_shape(rows, columns), tensor_shape(rows, count), tensor_shape(rows, count));
}

}// namespace luisa::test::tile_selection
