#pragma once

// Local benchmark/test composition. One independent row per program; inputs
// exclude NaNs and must not overlap either output. Values retain storage bits.
#include <luisa/tile/dsl.h>
#include <limits>

namespace luisa::test::tile_selection {
using namespace compute;

template<typename T>
[[nodiscard]] tile::Kernel stable_argmax(int64_t rows, int64_t columns, int64_t padded) {
    using namespace tile;
    // The caller checks positive sizes, padded >= columns and bounded volume.
    return tile_kernel("stable_first_index_argmax", [=](TensorView<const T, 2> input,
                                                       TensorView<T, 2> output_values,
                                                       TensorView<int64_t, 2> output_indices) {
               auto row_axis = axis("row", 1), column = axis("column", padded), one = axis("out", 1);
               auto domain = shape(row_axis, column), output_domain = shape(row_axis, one);
               for (auto &nest : parallel(shape(rows))) {
                   auto row = nest.index();
                   auto values = cast<float>(input.tile(coord(row, 0), domain).load());
                   auto positions = broadcast_to(iota(column), domain);
                   auto valid = positions < columns;
                   auto peak = reduce(ite(valid, values, -std::numeric_limits<float>::infinity()), column, maximum);
                   auto winner = reduce(ite(valid && (values == peak), positions, std::numeric_limits<int64_t>::max()), column, minimum);
                   auto selected = cast<int64_t>(winner.at(coord(0)));
                   // A forbidden NaN-only row cannot turn the sentinel into an
                   // invalid access; preserve the bad index so validation fails.
                   auto safe_index = ite((selected >= 0) && (selected < columns), selected, int64_t{0});
                   auto original = input.tile(coord(row, safe_index), output_domain).load();
                   output_values(coord(row, 0), output_domain).store(original);
                   output_indices(coord(row, 0), output_domain).store(full<int64_t>(output_domain, selected));
               }
           }).capture(tensor_shape(rows, columns), tensor_shape(rows, 1), tensor_shape(rows, 1));
}

}// namespace luisa::test::tile_selection
