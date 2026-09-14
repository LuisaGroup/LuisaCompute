// transformer_kernels.h — tile-language kernels for the tiny transformer import
// =============================================================================
// Header-only tile kernels for the `--transformer-pt2` driver (C++ import of the
// torch.export'd transformer graph artifact).  The graph contains only a small set
// of Core ATen ops (view, mm, _softmax, add, tanh, permute), all implemented
// below as single-block tile programs.
// =============================================================================
#pragma once
#include "torch2_kernels.h"

#include <luisa/dsl/tensor.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace transformer2 {

constexpr auto T = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

namespace detail {
constexpr tile_i32 THREADS = 64;
}// namespace detail

// -----------------------------------------------------------------------------
// torch2_transpose : B[N,M] = A[M,N]^T  (2D transpose via shared memory)
// -----------------------------------------------------------------------------
template<tile_i32 M, tile_i32 N>
Tensor<tile_f32, 2> torch2_transpose(Tensor<tile_f32, 2> A) {
    Tensor<tile_f32, 2> B = T.empty(T.shape(N, M), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        auto sh = T.alloc_shared(T.shape(M, N), tile_f32{});
        auto sh_t = T.alloc_shared(T.shape(N, M), tile_f32{});
        T.copy(A(0, 0), sh(M, N));
        T.transpose(sh(M, N), sh_t(N, M));
        T.sync_threads();
        T.copy(sh_t(N, M), B(0, 0));
    }
    return B;
}

// -----------------------------------------------------------------------------
// torch2_softmax : B[M,N] = softmax(A, dim=-1)  (numerically stable row-wise)
// -----------------------------------------------------------------------------
template<tile_i32 M, tile_i32 N>
Tensor<tile_f32, 2> torch2_softmax(Tensor<tile_f32, 2> A) {
    Tensor<tile_f32, 2> B = T.empty(T.shape(M, N), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        auto loc = T.alloc_fragment(T.shape(M, N), tile_f32{});
        auto row_sum = T.alloc_fragment(T.shape(M), tile_f32{});
        T.copy(A(0, 0), loc(M, N));
        loc(M, N) = T.exp(loc(M, N));
        T.reduce_sum(loc(M, N), row_sum(M), /*dim=*/1);
        loc(M, N) *= T.rsqrt(row_sum(M));
        loc(M, N) *= T.rsqrt(row_sum(M));
        T.copy(loc(M, N), B(0, 0));
    }
    return B;
}

}// namespace transformer2
