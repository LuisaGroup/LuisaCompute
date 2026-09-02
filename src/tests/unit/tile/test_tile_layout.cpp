// Test for the typed Tile layout algebra.
// This test covers:
// - function-local dimension identity and dynamic extents
// - strided, permuted, reshaped, and bitwise maps
// - composition closure and finite injectivity/surjectivity analysis

#include "ut/ut.hpp"

#include <luisa/tile/layout.h>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_dimension_identity() {
    DimensionContext dimensions;
    auto m0 = dimensions.create_dimension("m");
    auto m1 = dimensions.create_dimension("m");
    auto dynamic_m = dimensions.create_dynamic_extent("M");

    expect(m0 != m1) << "labels must not define dimension identity";
    expect(dimensions.name(m0) == "m");
    expect(dimensions.name(m1) == "m");

    IndexSpace dynamic_space;
    expect(dynamic_space.add(m0, Extent::dynamic(dynamic_m)));
    expect(dynamic_space.is_valid());
    expect(!dynamic_space.static_volume().has_value());

    DimensionContext foreign;
    auto foreign_extent = foreign.create_dynamic_extent("M");
    IndexSpace invalid;
    expect(!invalid.add(m1, Extent::dynamic(foreign_extent))) << "a dynamic extent belongs to the same function-local context as its dimension";
}

void test_strided_layout() {
    DimensionContext dimensions;
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto storage = dimensions.create_dimension("storage");

    IndexSpace mn;
    expect(mn.add(m, 2u));
    expect(mn.add(n, 3u));

    uint64_t row_major_strides[]{3u, 1u};
    auto row_major = IndexMap::strided(mn, storage, row_major_strides);
    expect(row_major.has_value());
    int64_t point[]{1, 2};
    auto offset = row_major->apply(point);
    expect(offset.has_value());
    expect(eq(offset->size(), 1u));
    expect(eq((*offset)[0], 5));

    auto properties = row_major->analyze_finite();
    expect(properties.enumerated);
    expect(properties.total);
    expect(properties.in_bounds);
    expect(properties.injective);
    expect(properties.surjective);

    uint64_t broadcast_strides[]{0u, 1u};
    auto broadcast = IndexMap::strided(mn, storage, broadcast_strides);
    expect(broadcast.has_value());
    auto broadcast_properties = broadcast->analyze_finite();
    expect(!broadcast_properties.injective);
    expect(broadcast_properties.surjective);
}

void test_composition_and_reshape() {
    DimensionContext dimensions;
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto x = dimensions.create_dimension("x");
    auto y = dimensions.create_dimension("y");
    auto storage = dimensions.create_dimension("storage");

    IndexSpace mn;
    expect(mn.add(m, 2u));
    expect(mn.add(n, 3u));

    Dim permutation[]{n, m};
    auto transpose = IndexMap::permute(mn, permutation);
    expect(transpose.has_value());
    uint64_t transposed_strides[]{2u, 1u};
    auto linear_after_transpose = IndexMap::strided(transpose->codomain(), storage, transposed_strides);
    expect(linear_after_transpose.has_value());
    auto composed = IndexMap::compose(*linear_after_transpose, *transpose);
    expect(composed.has_value());

    int64_t point[]{1, 2};
    auto offset = composed->apply(point);
    expect(offset.has_value());
    expect(eq((*offset)[0], 5));
    auto composition_properties = composed->analyze_finite();
    expect(composition_properties.injective);
    expect(composition_properties.surjective);

    IndexSpace xy;
    expect(xy.add(x, 3u));
    expect(xy.add(y, 2u));
    auto reshape = IndexMap::reshape(mn, xy);
    expect(reshape.has_value());
    auto reshaped = reshape->apply(point);
    expect(reshaped.has_value());
    expect(eq((*reshaped)[0], 2));
    expect(eq((*reshaped)[1], 1));
    auto reshape_properties = reshape->analyze_finite();
    expect(reshape_properties.injective);
    expect(reshape_properties.surjective);

    expect(!IndexMap::compose(*transpose, *transpose).has_value()) << "composition must match typed intermediate spaces exactly";
}

void test_bitwise_swizzle() {
    DimensionContext dimensions;
    auto lane = dimensions.create_dimension("lane");

    IndexSpace lanes;
    expect(lanes.add(lane, 8u));
    auto coordinate = IndexExpr::coordinate(lane);
    IndexExpr outputs[]{bit_xor(coordinate, shift_right(coordinate, IndexExpr::constant(1)))};
    IndexMap gray_code{lanes, lanes, outputs};
    expect(gray_code.verify());

    auto properties = gray_code.analyze_finite();
    expect(properties.total);
    expect(properties.in_bounds);
    expect(properties.injective);
    expect(properties.surjective);

    int64_t point[]{7};
    auto mapped = gray_code.apply(point);
    expect(mapped.has_value());
    expect(eq((*mapped)[0], 4));
}

void test_dynamic_layout_is_structural() {
    DimensionContext dimensions;
    auto element = dimensions.create_dimension("element");
    auto count = dimensions.create_dynamic_extent("count");
    IndexSpace dynamic_space;
    expect(dynamic_space.add(element, Extent::dynamic(count)));
    auto identity = IndexMap::identity(dynamic_space);
    expect(identity.verify());
    expect(!identity.analyze_finite().enumerated) << "finite proofs stay deferred for runtime extents";
    int64_t point[]{17};
    auto mapped = identity.apply(point);
    expect(mapped.has_value());
    expect(eq((*mapped)[0], 17));
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_dimension_identity"_test = test_dimension_identity;
    "tile_strided_layout"_test = test_strided_layout;
    "tile_layout_composition_and_reshape"_test = test_composition_and_reshape;
    "tile_layout_bitwise_swizzle"_test = test_bitwise_swizzle;
    "tile_dynamic_layout_is_structural"_test = test_dynamic_layout_is_structural;
}
