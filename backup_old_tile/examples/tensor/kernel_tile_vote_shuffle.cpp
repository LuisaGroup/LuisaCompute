// =============================================================================
// kernel_tile_vote_shuffle.cpp — ANY_OF / ALL_OF / SHUFFLE
// =============================================================================
// Vote and shuffle operations

#include "kernels.h"

Tensor<tile_f32, 1> tile_vote_shuffle_kernel() {
    constexpr tile_i32 N = 2;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> W = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto v = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        LuisaTensor.fill(v(N), 1.0f);
        LuisaTensor.any_of(v(N));
        LuisaTensor.all_of(v(N));
        LuisaTensor.shfl_xor(v(N), 1);
        LuisaTensor.shfl_up(v(N), 1);
        LuisaTensor.shfl_down(v(N), 1);
        LuisaTensor.copy(v(N), W(0));
    }
    return W;
}
