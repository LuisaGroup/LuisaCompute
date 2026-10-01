#pragma once

// Local benchmark/test composition: genuine INT64 row IDs, uniform per program.
// Callers reject IDs outside [0, vocabulary) before dispatch. Buffers are disjoint.
#include <luisa/tile/dsl.h>
#include <luisa/core/stl/memory.h>
#include <algorithm>

namespace luisa::test::tile_embedding {
using namespace compute;

[[nodiscard]] inline bool valid_row_indices(span<const int64_t> ids, int64_t vocabulary) noexcept {
    return vocabulary > 0 && !ids.empty() &&
           std::all_of(ids.begin(), ids.end(), [vocabulary](int64_t id) noexcept { return id >= 0 && id < vocabulary; });
}

template<typename T>
[[nodiscard]] tile::Kernel embedding_rows(int64_t vocabulary, int64_t width,
                                         int64_t tokens, int64_t feature_tile) {
    using namespace tile;
    return tile_kernel("embedding_uniform_int64_rows", [=](TensorView<const T, 2> table,
                                                          TensorView<const int64_t, 1> ids,
                                                          TensorView<T, 2> output) {
               auto token_axis = axis("token", tokens);
               auto block_axis = axis("feature_block", (width + feature_tile - 1) / feature_tile);
               auto row = axis("row", 1), feature = axis("feature", feature_tile), one_id = axis("one_id", 1);
               for (auto &nest : parallel(shape(token_axis, block_axis))) {
                   auto token = nest.index(token_axis), start = nest.index(block_axis) * feature_tile;
                   // Scalar ElementRef loads have no native Tile domain. This
                   // singleton load and uniform extract use the existing ABI.
                   auto selected = ids.tile(coord(token), shape(one_id)).load().at(coord(0));
                   auto values = table.tile(coord(selected, start), shape(row, feature)).load();
                   output(coord(token, start), shape(row, feature)).store(values);
               }
           }).capture(tensor_shape(vocabulary, width), tensor_shape(tokens), tensor_shape(tokens, width));
}

}// namespace luisa::test::tile_embedding
