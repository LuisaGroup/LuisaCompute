#pragma once

// Portable benchmark composition. NaNs are excluded; FP32/FP16/BF16
// values use exact FP32 comparison/storage and retain their original bits.
#include <luisa/tile/algorithms.h>
#include <bit>
#include <cstdint>
#include <limits>

namespace luisa::test::tile_sort_pipeline {
using namespace compute::tile;

struct Stage {
    bool initial;
    bool final;
    int64_t width;
};
struct Plan {
    luisa::vector<Stage> stages;
    luisa::string error;
    int64_t rows{}, columns{}, padded{}, chunk{};
};

// The eventual caller must check this result before calling any factory.
// Scratch is two disjoint FP32+INT32 planes per ping-pong slot, [R,P].
[[nodiscard]] inline Plan plan(int64_t rows, int64_t columns, int64_t chunk) {
    Plan p;
    p.rows = rows;
    p.columns = columns;
    p.chunk = chunk;
    if (rows <= 0 || columns <= 0 || columns > 16384 || chunk <= 0 ||
        !std::has_single_bit(static_cast<uint64_t>(chunk))) {
        p.error = "positive R/N and power-of-two C required; N<=16384";
        return p;
    }
    p.padded = static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(columns)));
    if (chunk > p.padded || rows > std::numeric_limits<int32_t>::max() ||
        static_cast<uint64_t>(rows) > (1ull << 28u) / static_cast<uint64_t>(p.padded)) {
        p.error = "C must not exceed P; R fits launch range; padded work <=2^28";
        return p;
    }
    p.stages.emplace_back(Stage{true, chunk == p.padded, chunk});
    for (auto span = chunk * 2; span <= p.padded; span *= 2) {
        p.stages.emplace_back(Stage{false, span == p.padded, span});
    }
    return p;
}

struct Pair {
    Tile<float> values;
    Tile<int32_t> indices;
};

[[nodiscard]] inline Pair choose_pair(const Pair &self, const Pair &other, const Tile<bool> &first, bool descending) {
    auto other_before = (descending ? other.values > self.values : other.values < self.values) ||
                        ((other.values == self.values) && (other.indices < self.indices));
    // Non-NaN values and globally unique original indices make the two
    // possible strict pair orders complementary, including +/-0 and +/-Inf.
    auto take_other = first == other_before;
    return {ite(take_other, other.values, self.values), ite(take_other, other.indices, self.indices)};
}

[[nodiscard]] inline Pair exchange_axis(const Pair &self, Axis axis, int64_t stride,
                                       const Tile<bool> &first, bool descending) {
    auto coordinates = [&](const Nest &nest) {
        luisa::vector<Scalar<int64_t>> result;
        for (auto &&a : self.values.space().axes()) {
            auto i = nest.index(a.dimension);
            if (a.dimension == axis.dimension()) {
                // Exactly the existing native XOR-permutation recognizer.
                auto group = 2 * stride;
                result.emplace_back(i / group * group + (i + stride) % group);
            } else {
                result.emplace_back(i);
            }
        }
        return result;
    };
    Pair other{reindex(self.values, self.values.space(), coordinates),
               reindex(self.indices, self.indices.space(), coordinates)};
    return choose_pair(self, other, first, descending);
}

// V/I are float/int32_t for scratch, T/int64_t if the initial stage is final.
// output_columns is P for scratch or logical N for final output.
template<typename T, typename V = float, typename I = int32_t>
[[nodiscard]] Kernel initialize(const Plan &p, bool descending = true) {
    auto output_columns = p.chunk == p.padded ? p.columns : p.padded;
    return tile_kernel("chunked_bitonic_initialize", [=](TensorView<const T, 2> input,
                                                          TensorView<V, 2> values_out,
                                                          TensorView<I, 2> indices_out) {
               auto r = axis("r", 1), lane = axis("lane", p.chunk);
               auto gr = axis("gr", p.rows), gc = axis("gc", p.padded / p.chunk);
               auto domain = shape(r, lane);
               for (auto &nest : parallel(shape(gr, gc))) {
                   auto row = nest.index(gr), origin = nest.index(gc) * p.chunk;
                   auto position = broadcast_to(iota(lane) + origin, domain);
                   auto loaded = cast<float>(input.tile(coord(row, origin), domain).load());
                   auto fill = descending ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
                   Pair current{ite(position < p.columns, loaded, fill),
                                broadcast_to(cast<int32_t>(position), domain)};
                   for (auto span = int64_t{2}; span <= p.chunk; span *= 2) {
                       for (auto stride = span / 2; stride != 0; stride /= 2) {
                           auto first = broadcast_to((position / stride % 2 == 0) == (position / span % 2 == 0), domain);
                           current = exchange_axis(current, lane, stride, first, descending);
                       }
                   }
                   values_out(coord(row, origin), domain).store(cast<V>(current.values));
                   indices_out(coord(row, origin), domain).store(cast<I>(current.indices));
               }
           }).capture(tensor_shape(p.rows, p.columns), tensor_shape(p.rows, output_columns), tensor_shape(p.rows, output_columns));
}

// One complete 2L-bitonic merge per program. Earlier levels have produced
// alternating forward/reverse runs. The final global run is forward.
// V/I = float/int32_t except the final stage, which uses T/int64_t.
template<typename V = float, typename I = int32_t>
[[nodiscard]] Kernel merge_whole(const Plan &p, const Stage &stage, bool descending = true) {
    auto output_columns = stage.final ? p.columns : p.padded;
    return tile_kernel("chunked_bitonic_merge_whole", [=](TensorView<const float, 2> values_in,
                                                           TensorView<const int32_t, 2> indices_in,
                                                           TensorView<V, 2> values_out,
                                                           TensorView<I, 2> indices_out) {
               auto r = axis("r", 1), lane = axis("lane", stage.width);
               auto gr = axis("gr", p.rows), gc = axis("gc", p.padded / stage.width);
               auto domain = shape(r, lane);
               for (auto &nest : parallel(shape(gr, gc))) {
                   auto row = nest.index(gr), origin = nest.index(gc) * stage.width;
                   auto position = broadcast_to(iota(lane) + origin, domain);
                   Pair current{values_in.tile(coord(row, origin), domain).load(),
                                indices_in.tile(coord(row, origin), domain).load()};
                   for (auto stride = stage.width / 2; stride != 0; stride /= 2) {
                       auto first = broadcast_to((position / stride % 2 == 0) == (position / stage.width % 2 == 0), domain);
                       current = exchange_axis(current, lane, stride, first, descending);
                   }
                   values_out(coord(row, origin), domain).store(cast<V>(current.values));
                   indices_out(coord(row, origin), domain).store(cast<I>(current.indices));
               }
           }).capture(tensor_shape(p.rows, p.padded), tensor_shape(p.rows, p.padded),
                      tensor_shape(p.rows, output_columns), tensor_shape(p.rows, output_columns));
}

}// namespace luisa::test::tile_sort_pipeline
