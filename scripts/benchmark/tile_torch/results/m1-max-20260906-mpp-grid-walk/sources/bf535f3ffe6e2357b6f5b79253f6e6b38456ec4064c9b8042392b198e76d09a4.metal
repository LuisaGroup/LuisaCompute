#define INPUT_ELEMENT float
#define ACCUMULATOR_ELEMENT float
#define TILE_M 32
#define TILE_N 32
#define SIMDGROUPS 1
#define COOPERATIVE_OUTPUT 1
#define RELAXED_PRECISION 0
#define STATIC_REDUCTION 0
#define REDUCTION_K 33
#define ROWS_M 1025
#define COLUMNS_N 129
#define INLINE_TENSORS 1
#define COHORT_ROWS 4
#define COHORT_COLUMNS 1
#define WALK_ROWS 1
#define WALK_COLUMNS 1
#define GRID_ROWS 9
#define GRID_COLUMNS 5

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

template<typename A, typename B, typename C>
void multiply_tile(thread A &tile_a, thread B &tile_b, thread C &tile_c) {
#if STATIC_REDUCTION
    constexpr auto descriptor = matmul2d_descriptor(TILE_M, TILE_N, REDUCTION_K,
                                                    false, false, RELAXED_PRECISION);
#else
    constexpr auto descriptor = matmul2d_descriptor(TILE_M, TILE_N, dynamic_length_v<int>,
                                                    false, false, RELAXED_PRECISION);
#endif
    matmul2d<descriptor, execution_simdgroups<SIMDGROUPS>> operation;
#if COOPERATIVE_OUTPUT
    auto result = operation.get_destination_cooperative_tensor<A, B,
                                                               ACCUMULATOR_ELEMENT>();
    operation.run(tile_a, tile_b, result);
    result.store(tile_c);
#else
    operation.run(tile_a, tile_b, tile_c);
#endif
}

kernel void mpp_gemm(uint2 physical_group [[threadgroup_position_in_grid]],
                     uint subgroup [[simdgroup_index_in_threadgroup]],
#if INLINE_TENSORS
                     device INPUT_ELEMENT *a_data [[buffer(0)]],
                     device INPUT_ELEMENT *b_data [[buffer(1)]],
                     device ACCUMULATOR_ELEMENT *c_data [[buffer(2)]]) {
    tensor<device INPUT_ELEMENT, dextents<int, 2>, tensor_inline> a(
        a_data, dextents<int, 2>(REDUCTION_K, ROWS_M), array<int, 2>{1, REDUCTION_K});
    tensor<device INPUT_ELEMENT, dextents<int, 2>, tensor_inline> b(
        b_data, dextents<int, 2>(COLUMNS_N, REDUCTION_K), array<int, 2>{1, COLUMNS_N});
    tensor<device ACCUMULATOR_ELEMENT, dextents<int, 2>, tensor_inline> c(
        c_data, dextents<int, 2>(COLUMNS_N, ROWS_M), array<int, 2>{1, COLUMNS_N});
#else
                     tensor<device INPUT_ELEMENT, dextents<int, 2>> a,
                     tensor<device INPUT_ELEMENT, dextents<int, 2>> b,
                     tensor<device ACCUMULATOR_ELEMENT, dextents<int, 2>> c) {
#endif
    uint2 group = physical_group;
#if WALK_ROWS > 0
    // Permute only independent programs. A partial final row stripe uses its
    // actual height, so this is a bijection without padding or duplicate work.
    const uint stripe_programs = uint(WALK_ROWS) * uint(GRID_COLUMNS);
    const uint stripe = physical_group.x / stripe_programs;
    const uint first_row = stripe * WALK_ROWS;
    const uint height = min(uint(WALK_ROWS), uint(GRID_ROWS) - first_row);
    const uint local_program = physical_group.x % stripe_programs;
    const uint rectangle_programs = height * uint(WALK_COLUMNS);
    const uint first_column = (local_program / rectangle_programs) * uint(WALK_COLUMNS);
    const uint width = min(uint(WALK_COLUMNS), uint(GRID_COLUMNS) - first_column);
    const uint rectangle_program = local_program % rectangle_programs;
    group = uint2(first_column + rectangle_program % width,
                  first_row + rectangle_program / width);
#endif
    // A multi-SIMD-group operation uses the whole threadgroup. Alternatively,
    // independent single-SIMD-group operations form a spatial cohort. Memory
    // views are composed with that execution map, not the other way around.
    const int local = SIMDGROUPS == 1 ? static_cast<int>(subgroup) : 0;
    const int origin_x = (static_cast<int>(group.x) * COHORT_COLUMNS + local % COHORT_COLUMNS) * TILE_N;
    const int origin_y = (static_cast<int>(group.y) * COHORT_ROWS + local / COHORT_COLUMNS) * TILE_M;
    if (origin_x >= COLUMNS_N || origin_y >= ROWS_M) { return; }
    // A static slice promises an in-bounds tile. Only interior groups may
    // make that promise; dynamic slices retain the original tensor bounds.
    if (origin_x <= COLUMNS_N - TILE_N && origin_y <= ROWS_M - TILE_M) {
#if STATIC_REDUCTION
        auto tile_a = a.slice<REDUCTION_K, TILE_M>(0, origin_y);
        auto tile_b = b.slice<TILE_N, REDUCTION_K>(origin_x, 0);
#else
        auto tile_a = a.slice<dynamic_extent, TILE_M>(0, origin_y);
        auto tile_b = b.slice<TILE_N, dynamic_extent>(origin_x, 0);
#endif
        auto tile_c = c.slice<TILE_N, TILE_M>(origin_x, origin_y);
        multiply_tile(tile_a, tile_b, tile_c);
    } else {
        auto tile_a = a.slice(0, origin_y);
        auto tile_b = b.slice(origin_x, 0);
        auto tile_c = c.slice(origin_x, origin_y);
        multiply_tile(tile_a, tile_b, tile_c);
    }
}
