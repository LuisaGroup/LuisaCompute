// torch2_kernels.h — Luisa tile-language kernels for torch2 graph import
// =============================================================================
// Header-only tile kernels for the `--rnn-pt2` driver (C++ import of the
// torch.export'd RNN graph artifact, see torch2_import.cpp).  The executor
// flattens every tensor to rank-2 and instantiates these kernels through a
// dispatch switch over the parsed shapes (batch B in {1,2,4,8,16,32,64};
// K/N/H/C/T fixed by the graph).
//
// Supported ops (Core ATen IR from torch2_export.py):
//   aten.full / aten.zeros -> torch2_fill
//   aten.select            -> torch2_select (per-row anchored-copy gather)
//   aten.mm                -> torch2_mm (single T.gemm accumulator, no fused
//                             bias GEMM — the current tile_to_kernel lowering
//                             mis-executes multi-GEMM-accumulator kernels on
//                             some backends)
//   aten.add               -> torch2_add (same-shape) / torch2_add_bias
//                             ([B,N] + [1,N]/[N] row-broadcast)
//   aten.tanh              -> torch2_tanh
//
// Every kernel is a single-block tile program (one T.Kernel, 1D dispatch)
// traced with tile::jit(...).compile(), lowered with tile_to_kernel, compiled
// on the device, and dispatched on Luisa buffers — the same flow as
// examples/tensor/kernel_*.cpp / rnn_kernels.h / mlp_kernels.h.
// =============================================================================
#pragma once
#include <luisa/dsl/tensor.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace torch2 {
// constexpr handle for the tile DSL (T.* spelling, like TileLang's `T`)
constexpr auto T = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

namespace detail {
constexpr tile_i32 THREADS = 64;
}// namespace detail

// -----------------------------------------------------------------------------
// torch2_mm : Z[B,N] = A[B,K] @ W[K,N]   (single T.gemm accumulator)
// -----------------------------------------------------------------------------
template<tile_i32 B, tile_i32 K, tile_i32 N>
Tensor<tile_f32, 2> torch2_mm(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> W) {
    Tensor<tile_f32, 2> Z = T.empty(T.shape(B, N), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        auto A_sh = T.alloc_shared(T.shape(B, K), tile_f32{});
        auto W_sh = T.alloc_shared(T.shape(K, N), tile_f32{});
        auto acc = T.alloc_fragment(T.shape(B, N), tile_f32{});
        T.copy(A(0, 0), A_sh(B, K));
        T.copy(W(0, 0), W_sh(K, N));
        T.clear(acc);
        T.gemm(A_sh(B, K), W_sh(K, N), acc(B, N));
        T.copy(acc(B, N), Z(0, 0));
    }
    return Z;
}

// -----------------------------------------------------------------------------
// torch2_add : C[B,N] = A[B,N] + B[B,N]  (same-shape elementwise add)
// -----------------------------------------------------------------------------
template<tile_i32 B, tile_i32 N>
Tensor<tile_f32, 2> torch2_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> Bmat) {
    Tensor<tile_f32, 2> C = T.empty(T.shape(B, N), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        C(T.range(0, B), T.range(0, N)) =
            A(T.range(0, B), T.range(0, N)) + Bmat(T.range(0, B), T.range(0, N));
    }
    return C;
}

// -----------------------------------------------------------------------------
// torch2_add_bias : C[B,N] = A[B,N] + Bias[1,N]  (row-broadcast add)
// The current tile_to_kernel lowering ignores slice offsets for SHARED tiles
// (sub-tile shared slices all index offset 0 — the mlp_kernels.h caveat), so
// shared-based row replication does not work.  Global slice reads/writes DO
// honour their offsets (_global_index reconstructs the base from t->offset()),
// so the broadcast is implemented as a host-unrolled per-row binary over
// [1,N] global row slices: C(b, :) = A(b, :) + Bias(0, :).  Bias may be a
// rank-1 [N] buffer (viewed as [1,N]) or a [1,N] buffer.
// -----------------------------------------------------------------------------
template<tile_i32 B, tile_i32 N>
Tensor<tile_f32, 2> torch2_add_bias(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> Bias) {
    Tensor<tile_f32, 2> C = T.empty(T.shape(B, N), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        for (int b = 0; b < B; ++b) {// compile-time unrolled
            C(T.range(b, b + 1), T.range(0, N)) =
                A(T.range(b, b + 1), T.range(0, N)) + Bias(T.range(0, 1), T.range(0, N));
        }
    }
    return C;
}

// -----------------------------------------------------------------------------
// torch2_tanh : C[B,N] = tanh(A[B,N])  (elementwise unary)
// -----------------------------------------------------------------------------
template<tile_i32 B, tile_i32 N>
Tensor<tile_f32, 2> torch2_tanh(Tensor<tile_f32, 2> A) {
    Tensor<tile_f32, 2> C = T.empty(T.shape(B, N), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        C(T.range(0, B), T.range(0, N)) =
            T.tanh(A(T.range(0, B), T.range(0, N)));
    }
    return C;
}

// -----------------------------------------------------------------------------
// torch2_select : Out[B,I] = X[b*T_+t, :]  (aten.select on dim 1 of [B,T_,I])
// X is the rank-3 input flattened to [B*T_, I].  Tile slice overloads are
// rank-2 only, so the gather is a per-row anchored copy (host-unrolled over
// constexpr B): row is shared [1,I]; X(b*T_+t, 0) is a global anchored read and
// Out(b, 0) a global anchored write.
// -----------------------------------------------------------------------------
template<tile_i32 B, tile_i32 T_, tile_i32 I, tile_i32 t>
Tensor<tile_f32, 2> torch2_select(Tensor<tile_f32, 2> X) {
    Tensor<tile_f32, 2> Out = T.empty(T.shape(B, I), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        auto row = T.alloc_shared(T.shape(1, I), tile_f32{});
        for (int b = 0; b < B; ++b) {// compile-time unrolled
            T.copy(X(b * T_ + t, 0), row(1, I));
            T.copy(row(1, I), Out(b, 0));
        }
    }
    return Out;
}

// -----------------------------------------------------------------------------
// torch2_fill : C[N] = value  (aten.full / aten.zeros, scalar fill only)
// The fill value is a plain float parameter baked into the tile IR at trace
// time (the DSL records host-side literals only); the driver traces it through
// a capturing lambda so any scalar value works.
// -----------------------------------------------------------------------------
template<tile_i32 N>
Tensor<tile_f32, 1> torch2_fill(float value) {
    Tensor<tile_f32, 1> C = T.empty(T.shape(N), tile_f32{});
    for (auto bx : T.Kernel(1, detail::THREADS)) {
        auto F = T.alloc_fragment(T.shape(N), tile_f32{});
        T.fill(F(N), value);
        T.copy(F(N), C(0));
    }
    return C;
}

}// namespace torch2
