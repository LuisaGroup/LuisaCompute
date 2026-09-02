// Test for the typed Tile layout algebra.
// This test covers:
// - function-local dimension identity and dynamic extents
// - strided, permuted, reshaped, and bitwise maps
// - composition closure and finite injectivity/surjectivity analysis
// - conservative affine proofs checked against exhaustive semantic oracles
// - structural GF(2) normalization, rank, bounds, and ragged-domain proofs
// - exact set-valued correspondences for replicated placement

#include "ut/ut.hpp"

#include <utility>
#include <array>
#include <bit>
#include <limits>

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

void test_index_expression_inspection() {
    DimensionContext dimensions;
    auto axis = dimensions.create_dimension("index");
    auto coordinate = IndexExpr::coordinate(axis);
    auto constant = IndexExpr::constant(3);
    auto expression = coordinate + constant;
    expect(IndexExpr{}.kind() == IndexExprKind::INVALID);
    expect(!IndexExpr{}.constant_value());
    expect(!IndexExpr{}.dimension());
    expect(!IndexExpr{}.lhs());
    expect(!IndexExpr{}.rhs());
    expect(expression.kind() == IndexExprKind::ADD);
    expect(expression.lhs().kind() == IndexExprKind::COORDINATE);
    expect(expression.lhs().dimension() == axis);
    expect(expression.rhs().kind() == IndexExprKind::CONSTANT);
    expect(eq(expression.rhs().constant_value().value_or(-1), 3));
    expect(!expression.constant_value());
    expect(!constant.dimension());
    expect(!coordinate.lhs());

    IndexSpace space;
    expect(space.add(axis, 16u));
    IndexExpr outputs[]{shift_right(shift_left(coordinate, IndexExpr::constant(60)), IndexExpr::constant(60))};
    IndexMap high_bit{space, space, outputs};
    auto properties = high_bit.analyze_finite();
    expect(properties.total && properties.in_bounds && properties.injective && properties.surjective);
    for (auto i = int64_t{0}; i < 16; i++) {
        int64_t point[]{i};
        auto value = high_bit.apply(point);
        expect(value.has_value());
        if (value) { expect(eq(value->front(), i)); }
    }
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

void test_large_affine_proofs() {
    DimensionContext dimensions;
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto storage = dimensions.create_dimension("storage");
    IndexSpace space;
    expect(space.add(m, 1048577u));
    expect(space.add(n, 7u));
    for (auto strides : {std::array<uint64_t, 2u>{11u, 1u}, std::array<uint64_t, 2u>{1u, 1048580u}}) {
        auto map = IndexMap::strided(space, storage, strides);
        expect(map.has_value());
        if (!map) { continue; }
        auto proof = map->prove(0u);
        expect(proof.is_storage_safe());
        expect(!proof.enumerated);
        expect(proof.surjective == ProofStatus::DISPROVEN);
        expect(!map->analyze_finite().enumerated);
    }
    auto identity = IndexMap::identity(space).prove(0u);
    expect(identity.is_storage_safe());
    expect(identity.surjective == ProofStatus::PROVEN);
    Dim order[]{n, m};
    auto transpose = IndexMap::permute(space, order);
    expect(transpose.has_value());
    if (!transpose) { return; }
    uint64_t strides[]{1048580u, 1u};
    auto storage_map = IndexMap::strided(transpose->codomain(), storage, strides);
    expect(storage_map.has_value());
    if (!storage_map) { return; }
    auto composed = IndexMap::compose(*storage_map, *transpose);
    expect(composed.has_value());
    if (composed) { expect(composed->prove(0u).is_storage_safe()); }
    auto i = IndexExpr::coordinate(m);
    auto j = IndexExpr::coordinate(n);
    IndexSpace reversed_storage;
    expect(reversed_storage.add(storage, 1048576u * 11u + 7u));
    IndexExpr reversed[]{(IndexExpr::constant(1048576) - i) * IndexExpr::constant(11) + j};
    expect(IndexMap{space, reversed_storage, reversed}.prove(0u).is_storage_safe());

    // Unit dimensions may carry zero strides without introducing aliases.
    IndexSpace unit;
    expect(unit.add(m, 1u));
    expect(unit.add(n, 1048577u));
    uint64_t unit_strides[]{0u, 1u};
    auto unit_map = IndexMap::strided(unit, storage, unit_strides);
    expect(unit_map.has_value());
    if (unit_map) { expect(unit_map->prove(0u).is_storage_safe()); }
}

void test_joint_affine_rank_proofs() {
    DimensionContext dimensions;
    auto group = dimensions.create_dimension("group");
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto u = dimensions.create_dimension("u");
    auto v = dimensions.create_dimension("v");
    IndexSpace domain;
    expect(domain.add(m, 1048577u));
    expect(domain.add(n, 1048577u));
    IndexSpace codomain;
    expect(codomain.add(u, 3145729u));
    expect(codomain.add(v, 3145729u));
    auto i = IndexExpr::coordinate(m);
    auto j = IndexExpr::coordinate(n);
    IndexExpr outputs[]{i * IndexExpr::constant(2) + j, i + j * IndexExpr::constant(2)};
    auto proof = IndexMap{domain, codomain, outputs}.prove(0u);
    expect(proof.is_storage_safe()) << "joint full rank proves injectivity even when no individual row separates axes";
    expect(proof.surjective == ProofStatus::DISPROVEN);
    expect(!proof.enumerated);

    // First recover a mixed-radix coordinate, then use matrix rank to recover
    // the remaining two. Neither condition alone proves this 3 -> 2 map.
    IndexSpace mixed_domain;
    expect(mixed_domain.add(group, 3u));
    expect(mixed_domain.add(m, 1025u));
    expect(mixed_domain.add(n, 1025u));
    IndexSpace mixed_codomain;
    expect(mixed_codomain.add(u, 10241u));
    expect(mixed_codomain.add(v, 3073u));
    IndexExpr mixed[]{IndexExpr::coordinate(group) * IndexExpr::constant(4096) + i + j,
                      i + j * IndexExpr::constant(2)};
    expect(IndexMap{mixed_domain, mixed_codomain, mixed}.prove(0u).is_storage_safe());

    // Rank failure modulo one prime is UNKNOWN, never evidence of singularity
    // over the integers. The independent finite fallback can still prove it.
    IndexSpace small;
    expect(small.add(m, 2u));
    expect(small.add(n, 2u));
    IndexSpace large;
    expect(large.add(u, 4294967295u));
    expect(large.add(v, 4294967295u));
    auto prime = IndexExpr::constant(2147483647);
    IndexExpr scaled[]{(i + j) * prime, (i - j) * prime + prime};
    IndexMap scaled_map{small, large, scaled};
    auto unknown = scaled_map.prove(0u);
    expect(unknown.total == ProofStatus::PROVEN);
    expect(unknown.injective == ProofStatus::UNKNOWN);
    expect(!unknown.is_storage_safe() && !unknown.is_storage_invalid());
    auto finite = scaled_map.prove(4u);
    expect(finite.enumerated && finite.is_storage_safe());
}

void test_affine_proof_counterexamples() {
    DimensionContext dimensions;
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto s = dimensions.create_dimension("storage");
    auto i = IndexExpr::coordinate(m);
    auto j = IndexExpr::coordinate(n);
    IndexSpace large;
    expect(large.add(m, 1048577u));
    expect(large.add(n, 7u));
    uint64_t broadcast_strides[]{0u, 1u};
    auto broadcast = IndexMap::strided(large, s, broadcast_strides);
    expect(broadcast.has_value());
    if (broadcast) {
        auto proof = broadcast->prove(0u);
        expect(proof.total == ProofStatus::PROVEN);
        expect(proof.in_bounds == ProofStatus::PROVEN);
        expect(proof.injective == ProofStatus::DISPROVEN);
        expect(proof.is_storage_invalid());
    }
    IndexSpace small;
    expect(small.add(m, 3u));
    expect(small.add(n, 2u));
    for (auto reversed : {false, true}) {
        IndexSpace storage;
        expect(storage.add(s, 9u));
        auto x = i * IndexExpr::constant(2);
        auto y = j * IndexExpr::constant(4);
        IndexExpr outputs[]{reversed ? x - y + IndexExpr::constant(4) : x + y};
        auto proof = IndexMap{small, storage, outputs}.prove(0u);
        expect(proof.total == ProofStatus::PROVEN);
        expect(proof.injective == ProofStatus::DISPROVEN) << "a GCD-derived pair is a concrete collision witness";
        expect(!proof.enumerated);
    }
    IndexSpace storage;
    expect(storage.add(s, 8u));
    IndexExpr nonseparated[]{i * IndexExpr::constant(2) + j * IndexExpr::constant(3)};
    IndexMap injective{small, storage, nonseparated};
    expect(injective.prove(0u).injective == ProofStatus::UNKNOWN);
    expect(injective.prove(5u).injective == ProofStatus::UNKNOWN);
    expect(injective.prove(6u).is_storage_safe());
    expect(injective.prove(6u).enumerated);

    IndexSpace line;
    expect(line.add(m, 3u));
    auto overflow = i * IndexExpr::constant(std::numeric_limits<int64_t>::max());
    IndexExpr cancelled[]{overflow - overflow};
    expect(IndexMap{line, storage, cancelled}.prove(0u).total == ProofStatus::DISPROVEN);
    IndexExpr outside[]{i + IndexExpr::constant(7)};
    expect(IndexMap{line, storage, outside}.prove(0u).is_storage_invalid());

    IndexSpace interior;
    expect(interior.add(m, 4u));
    auto product = i * (IndexExpr::constant(3) - i) * IndexExpr::constant(std::numeric_limits<int64_t>::max());
    IndexExpr invalid_interior[]{product - product};
    IndexMap partial{interior, storage, invalid_interior};
    expect(partial.prove(0u).total == ProofStatus::UNKNOWN) << "valid endpoints do not prove the interior safe";
    expect(partial.prove(4u).total == ProofStatus::DISPROVEN);
}

void test_layout_proof_cardinality_edges() {
    DimensionContext dimensions;
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto zero = dimensions.create_dimension("zero");
    auto count = dimensions.create_dynamic_extent("count");
    IndexSpace huge;
    expect(huge.add(m, uint64_t{1u} << 40u));
    expect(huge.add(n, uint64_t{1u} << 40u));
    expect(!huge.static_volume());
    auto proof = IndexMap::identity(huge).prove(0u);
    expect(proof.is_storage_safe());
    expect(proof.surjective == ProofStatus::PROVEN);
    expect(!proof.enumerated);
    auto empty = huge;
    expect(empty.add(zero, 0u));
    expect(empty.static_volume() == luisa::optional<uint64_t>{0u});
    IndexSpace storage;
    expect(storage.add(zero, 0u));
    IndexExpr unreachable[]{floor_div(IndexExpr::coordinate(m), IndexExpr::constant(0))};
    auto vacuous = IndexMap{empty, storage, unreachable}.prove(0u);
    expect(vacuous.is_storage_safe());
    expect(vacuous.surjective == ProofStatus::PROVEN);
    IndexSpace dynamic_empty;
    expect(dynamic_empty.add(m, Extent::dynamic(count)));
    expect(dynamic_empty.add(zero, 0u));
    expect(dynamic_empty.static_volume() == luisa::optional<uint64_t>{0u});
    expect(IndexMap{dynamic_empty, storage, unreachable}.prove(0u).is_storage_safe());

    IndexSpace boundary;
    expect(boundary.add(m, uint64_t{1u} << 63u));
    expect(IndexMap::identity(boundary).prove(0u).is_storage_safe());
    IndexSpace unrepresentable;
    expect(unrepresentable.add(m, (uint64_t{1u} << 63u) + 1u));
    expect(!IndexMap::identity(unrepresentable).prove(0u).is_storage_safe());
    IndexSpace dynamic;
    expect(dynamic.add(m, Extent::dynamic(count)));
    expect(IndexMap::identity(dynamic).prove(0u).total == ProofStatus::UNKNOWN);
}

void test_affine_proof_finite_oracle() {
    DimensionContext dimensions;
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto u = dimensions.create_dimension("u");
    auto v = dimensions.create_dimension("v");
    auto i = IndexExpr::coordinate(m);
    auto j = IndexExpr::coordinate(n);
    auto c = [](int64_t value) noexcept { return IndexExpr::constant(value); };
    auto agrees = [](ProofStatus proof, bool oracle) noexcept {
        return proof == ProofStatus::UNKNOWN || (proof == ProofStatus::PROVEN) == oracle;
    };
    auto cases = 0u;
    auto proven = 0u;
    auto disproven = 0u;
    for (auto extents : {std::array{2u, 3u}, std::array{3u, 2u}, std::array{1u, 4u}, std::array{4u, 1u}, std::array{3u, 3u}}) {
        IndexSpace domain;
        expect(domain.add(m, extents[0]));
        expect(domain.add(n, extents[1]));
        IndexSpace codomain;
        expect(codomain.add(u, 7u));
        expect(codomain.add(v, 11u));
        for (auto a = -2; a <= 2; a++) {
            for (auto b = -2; b <= 2; b++) {
                for (auto d = -2; d <= 2; d++) {
                    for (auto e = -2; e <= 2; e++) {
                        for (auto offset : {-1, 0, 4}) {
                            IndexExpr outputs[]{i * c(a) + j * c(b) + c(offset),
                                                i * c(d) + j * c(e) + c(4 - offset)};
                            IndexMap map{domain, codomain, outputs};
                            auto proof = map.prove(0u);
                            auto oracle = map.analyze_finite(16u);
                            auto sound = oracle.enumerated && !proof.enumerated &&
                                         agrees(proof.total, oracle.total) && agrees(proof.in_bounds, oracle.in_bounds) &&
                                         agrees(proof.injective, oracle.injective) && agrees(proof.surjective, oracle.surjective);
                            if (!sound) {
                                expect(sound) << "affine proof disagrees with exhaustive oracle: "
                                              << extents[0] << "," << extents[1] << " coefficients="
                                              << a << "," << b << "," << d << "," << e << " offset=" << offset;
                                return;
                            }
                            cases++;
                            proven += proof.is_storage_safe();
                            disproven += proof.is_storage_invalid();
                        }
                    }
                }
            }
        }
    }
    expect(eq(cases, 9375u));
    expect(proven != 0u && disproven != 0u);
}

void test_bit_linear_proofs() {
    DimensionContext dimensions;
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto s = dimensions.create_dimension("storage");
    auto i = IndexExpr::coordinate(m);
    auto j = IndexExpr::coordinate(n);
    auto c = [](int64_t value) noexcept { return IndexExpr::constant(value); };
    for (auto extent : std::array<uint64_t, 6u>{1u, 31u, 32u, 2097151u, 2097152u, uint64_t{1u} << 63u}) {
        IndexSpace domain;
        expect(domain.add(m, extent));
        IndexSpace storage;
        expect(storage.add(s, std::bit_ceil(extent)));
        IndexExpr outputs[]{bit_xor(i, shift_right(i, c(1)))};
        auto proof = IndexMap{domain, storage, outputs}.prove(0u);
        expect(proof.is_storage_safe());
        expect(!proof.enumerated);
        expect(proof.surjective == (std::has_single_bit(extent) ? ProofStatus::PROVEN : ProofStatus::DISPROVEN));
    }
    IndexSpace domain;
    expect(domain.add(m, 1024u));
    expect(domain.add(n, 2048u));
    IndexSpace physical;
    expect(physical.add(s, 2097152u));
    IndexExpr swizzled[]{i * c(2048) + bit_xor(j, bit_and(i, c(2047)))};
    auto swizzle = IndexMap{domain, physical, swizzled};
    auto proof = swizzle.prove(0u);
    expect(proof.is_storage_safe() && proof.surjective == ProofStatus::PROVEN && !proof.enumerated);
    auto address = IndexExpr::coordinate(s);
    IndexExpr unswizzled[]{floor_div(address, c(2048)), bit_xor(modulo(address, c(2048)), floor_div(address, c(2048)))};
    auto inverse = IndexMap{physical, domain, unswizzled};
    expect(inverse.prove(0u).is_storage_safe());
    auto roundtrip = IndexMap::compose(inverse, swizzle);
    expect(roundtrip.has_value());
    if (roundtrip) {
        expect(roundtrip->prove(0u).is_storage_safe());
        for (auto point : {std::array<int64_t, 2u>{0, 0}, {1023, 2047}, {17, 53}}) {
            auto mapped = roundtrip->apply(point);
            expect(mapped.has_value());
            if (mapped) { expect(*mapped == luisa::vector<int64_t>{point.begin(), point.end()}); }
        }
    }
    // Matrix rank and cardinality must not silently stop at 64 total bits.
    IndexSpace huge;
    expect(huge.add(m, uint64_t{1u} << 40u));
    expect(huge.add(n, uint64_t{1u} << 40u));
    IndexExpr coupled[]{bit_xor(i, j), j};
    auto wide = IndexMap{huge, huge, coupled}.prove(0u);
    expect(wide.is_storage_safe() && wide.surjective == ProofStatus::PROVEN && !wide.enumerated);
    IndexExpr lost_bit[]{bit_xor(i, j), bit_and(j, c((int64_t{1} << 39u) - 1))};
    auto deficient = IndexMap{huge, huge, lost_bit}.prove(0u);
    expect(deficient.total == ProofStatus::PROVEN);
    expect(deficient.injective == ProofStatus::DISPROVEN);
    expect(deficient.surjective == ProofStatus::DISPROVEN);
}

void test_bit_linear_bounds_and_checked_arithmetic() {
    DimensionContext dimensions;
    auto m = dimensions.create_dimension("m");
    auto s = dimensions.create_dimension("storage");
    auto i = IndexExpr::coordinate(m);
    auto c = [](int64_t value) noexcept { return IndexExpr::constant(value); };
    IndexSpace domain;
    expect(domain.add(m, 2097152u));
    auto high = shift_left(i, c(43));
    IndexExpr restored[]{shift_right(high, c(43))};
    expect(IndexMap{domain, domain, restored}.prove(0u).is_storage_safe());
    IndexExpr overflow[]{shift_right(i * c(int64_t{1} << 43u), c(43))};
    expect(IndexMap{domain, domain, overflow}.prove(0u).total == ProofStatus::DISPROVEN);
    IndexExpr complemented[]{bit_and(bit_xor(i, c(-1)), c(2097151))};
    expect(IndexMap{domain, domain, complemented}.prove(0u).is_storage_safe());
    IndexSpace shifted_storage;
    expect(shifted_storage.add(s, 4194304u));
    IndexExpr shifted[]{bit_xor(i, shift_right(i, c(1))) + c(2097152)};
    auto translated = IndexMap{domain, shifted_storage, shifted}.prove(0u);
    expect(translated.is_storage_safe());
    expect(translated.surjective == ProofStatus::DISPROVEN);

    IndexSpace four;
    expect(four.add(m, 4u));
    IndexSpace seven;
    expect(seven.add(s, 7u));
    // Image {0,3,5,6}: OR-ing all possible bits overestimates the maximum as 7.
    auto correlated = bit_xor(bit_and(i, c(1)) * c(3), shift_right(i, c(1)) * c(5));
    IndexExpr correlated_output[]{correlated};
    auto correlated_proof = IndexMap{four, seven, correlated_output}.prove(0u);
    expect(correlated_proof.is_storage_safe() && !correlated_proof.enumerated);
    IndexSpace six;
    expect(six.add(s, 6u));
    expect(IndexMap{four, six, correlated_output}.prove(0u).is_storage_invalid());
    IndexSpace three;
    expect(three.add(m, 3u));
    // The excluded fourth point alone is out of range. Envelope failure is
    // UNKNOWN for this ragged domain, never a disproof of the actual map.
    auto ragged = IndexMap{three, six, correlated_output};
    expect(ragged.prove(0u).total == ProofStatus::UNKNOWN);
    expect(ragged.prove(3u).is_storage_safe());

    IndexSpace two;
    expect(two.add(m, 2u));
    auto sign = bit_and(i, c(1)) * c(std::numeric_limits<int64_t>::min());
    IndexExpr sign_restore[]{shift_right(sign, c(63))};
    expect(IndexMap{two, two, sign_restore}.prove(0u).is_storage_safe());
    IndexExpr invalid_shift[]{bit_and(shift_left(i, c(64)), c(0))};
    expect(IndexMap{two, two, invalid_shift}.prove(0u).is_storage_invalid());
    IndexExpr invalid_division[]{bit_and(floor_div(i, c(0)), c(0))};
    expect(IndexMap{two, two, invalid_division}.prove(0u).is_storage_invalid());

    IndexSpace eight;
    expect(eight.add(m, 8u));
    auto nonlinear = bit_xor(i, shift_left(bit_and(i, shift_right(i, c(1))), c(1)));
    IndexExpr nonlinear_output[]{nonlinear};
    IndexMap nonlinear_map{eight, eight, nonlinear_output};
    // It looks like identity at 0 and all basis vectors, but f(1)=f(3)=1.
    // Sampling a basis without structurally proving linearity is unsound.
    for (auto point : {0, 1, 2, 4}) {
        int64_t coordinate[]{point};
        auto mapped = nonlinear_map.apply(coordinate);
        expect(mapped.has_value() && mapped->front() == point);
    }
    expect(!nonlinear_map.prove(0u).is_storage_safe());
    expect(nonlinear_map.prove(8u).injective == ProofStatus::DISPROVEN);
}

void test_bit_linear_proof_finite_oracle() {
    DimensionContext dimensions;
    auto m = dimensions.create_dimension("m");
    auto n = dimensions.create_dimension("n");
    auto u = dimensions.create_dimension("u");
    auto v = dimensions.create_dimension("v");
    auto i = IndexExpr::coordinate(m);
    auto j = IndexExpr::coordinate(n);
    auto c = [](int64_t value) noexcept { return IndexExpr::constant(value); };
    auto agrees = [](ProofStatus proof, bool oracle) noexcept {
        return proof == ProofStatus::UNKNOWN || (proof == ProofStatus::PROVEN) == oracle;
    };
    auto cases = 0u;
    for (auto extents : {std::array{2u, 4u}, std::array{2u, 3u}, std::array{1u, 4u}}) {
        IndexSpace domain;
        expect(domain.add(m, extents[0]));
        expect(domain.add(n, extents[1]));
        for (auto codomain_kind : {0, 1, 2}) {
            IndexSpace codomain;
            expect(codomain.add(u, codomain_kind == 0 ? 8u : codomain_kind == 1 ? 2u :
                                                                                  6u));
            if (codomain_kind == 1) { expect(codomain.add(v, 4u)); }
            // Every 3x3 GF(2) matrix and XOR offset, not random samples.
            for (auto matrix = 0u; matrix < 512u; matrix++) {
                for (auto offset = 0u; offset < 8u; offset++) {
                    auto packed = bit_xor(c(offset), bit_xor(i * c(matrix & 7u),
                                                             bit_xor(bit_and(j, c(1)) * c((matrix >> 3u) & 7u),
                                                                     shift_right(j, c(1)) * c((matrix >> 6u) & 7u))));
                    luisa::vector<IndexExpr> outputs;
                    if (codomain_kind == 1) {
                        outputs = {bit_and(packed, c(1)), shift_right(packed, c(1))};
                    } else {
                        outputs = {packed};
                    }
                    IndexMap map{domain, codomain, outputs};
                    auto proof = map.prove(0u);
                    auto oracle = map.analyze_finite(8u);
                    auto sound = oracle.enumerated && !proof.enumerated &&
                                 agrees(proof.total, oracle.total) && agrees(proof.in_bounds, oracle.in_bounds) &&
                                 agrees(proof.injective, oracle.injective) && agrees(proof.surjective, oracle.surjective);
                    if (extents[1] == 4u && codomain_kind != 2) {
                        sound &= proof.total != ProofStatus::UNKNOWN && proof.injective != ProofStatus::UNKNOWN && proof.surjective != ProofStatus::UNKNOWN;
                    }
                    if (!sound) {
                        expect(sound) << "GF(2) proof disagrees with exhaustive oracle: " << extents[0] << "," << extents[1]
                                      << " codomain=" << codomain_kind << " matrix=" << matrix << " offset=" << offset;
                        return;
                    }
                    cases++;
                }
            }
        }
    }
    expect(eq(cases, 36864u));
}

void test_layout_correspondence() {
    DimensionContext dimensions;
    auto logical_dimension = dimensions.create_dimension("logical");
    auto replica_dimension = dimensions.create_dimension("replica");
    auto physical_dimension = dimensions.create_dimension("physical");
    IndexSpace logical;
    IndexSpace fiber;
    IndexSpace coincident_physical;
    IndexSpace replicated_physical;
    expect(logical.add(logical_dimension, 4u));
    expect(fiber.add(logical_dimension, 4u));
    expect(fiber.add(replica_dimension, 3u));
    expect(coincident_physical.add(physical_dimension, 4u));
    expect(replicated_physical.add(physical_dimension, 12u));

    IndexExpr left_outputs[]{IndexExpr::coordinate(logical_dimension)};
    IndexExpr coincident_outputs[]{IndexExpr::coordinate(logical_dimension)};
    IndexExpr replicated_outputs[]{
        IndexExpr::coordinate(logical_dimension) * IndexExpr::constant(3) +
        IndexExpr::coordinate(replica_dimension)};
    IndexMap left{fiber, logical, left_outputs};
    LayoutCorrespondence coincident{
        left,
        IndexMap{fiber, coincident_physical, coincident_outputs}};
    LayoutCorrespondence replicated{
        std::move(left),
        IndexMap{fiber, replicated_physical, replicated_outputs}};

    auto coincident_properties = coincident.analyze_finite();
    expect(coincident_properties.covers_logical_space);
    expect(eq(coincident_properties.fiber_points, 12u));
    expect(eq(coincident_properties.minimum_replication, 1u));
    expect(eq(coincident_properties.maximum_replication, 1u));

    auto replicated_properties = replicated.analyze_finite();
    expect(replicated_properties.covers_logical_space);
    expect(eq(replicated_properties.minimum_replication, 3u));
    expect(eq(replicated_properties.maximum_replication, 3u));
    int64_t logical_point[]{2};
    auto placements = replicated.placements(logical_point);
    expect(placements.has_value());
    expect(eq(placements->size(), 3u));
    expect((*placements)[0] == luisa::vector<int64_t>{6});
    expect((*placements)[1] == luisa::vector<int64_t>{7});
    expect((*placements)[2] == luisa::vector<int64_t>{8});
    expect(replicated.converse().verify());
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_dimension_identity"_test = test_dimension_identity;
    "tile_strided_layout"_test = test_strided_layout;
    "tile_layout_composition_and_reshape"_test = test_composition_and_reshape;
    "tile_layout_bitwise_swizzle"_test = test_bitwise_swizzle;
    "tile_layout_index_expression_inspection"_test = test_index_expression_inspection;
    "tile_dynamic_layout_is_structural"_test = test_dynamic_layout_is_structural;
    "tile_large_affine_proofs"_test = test_large_affine_proofs;
    "tile_joint_affine_rank_proofs"_test = test_joint_affine_rank_proofs;
    "tile_affine_proof_counterexamples"_test = test_affine_proof_counterexamples;
    "tile_layout_proof_cardinality_edges"_test = test_layout_proof_cardinality_edges;
    "tile_affine_proof_finite_oracle"_test = test_affine_proof_finite_oracle;
    "tile_bit_linear_proofs"_test = test_bit_linear_proofs;
    "tile_bit_linear_bounds_and_checked_arithmetic"_test = test_bit_linear_bounds_and_checked_arithmetic;
    "tile_bit_linear_proof_finite_oracle"_test = test_bit_linear_proof_finite_oracle;
    "tile_layout_correspondence"_test = test_layout_correspondence;
}
