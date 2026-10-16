/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <type_traits>

// Standalone diagnostic: only 1024^3 row-major GEMM, 128 threads and aligned
// non-overlapping storage. Host checks the actual final pointers before launch.
// Adapted from NVIDIA's pinned SM80 CuTe tutorial; its F16 accumulator is NOT used.
template<class Element, class MmaOperation>
__device__ __forceinline__ void cute_gemm_1024(const Element *a, const Element *b, Element *output) {
    using namespace cute;
    auto problem = make_shape(Int<1024>{}, Int<1024>{}, Int<1024>{});
    auto cta = make_shape(_64{}, _64{}, _32{});
    // CuTe represents B as (N,K): input byte storage remains original B[K,N].
    auto stride_a = make_stride(Int<1024>{}, _1{});
    auto stride_b = make_stride(_1{}, Int<1024>{});
    auto stride_c = make_stride(Int<1024>{}, _1{});
    auto sa_layout = tile_to_shape(composition(Swizzle<2,3,3>{},
        Layout<Shape<_8,_32>, Stride<_32,_1>>{}), make_shape(_64{},_32{},_2{}));
    auto sb_layout = tile_to_shape(composition(Swizzle<3,3,3>{},
        Layout<Shape<_64,_8>, Stride<_1,_64>>{}), make_shape(_64{},_32{},_2{}));
    auto copy_a = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
        Layout<Shape<_32,_4>, Stride<_4,_1>>{}, Layout<Shape<_1,_8>>{});
    auto copy_b = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
        Layout<Shape<_8,_16>, Stride<_1,_8>>{}, Layout<Shape<_8,_1>>{});
    auto mma = make_tiled_mma(MmaOperation{}, Layout<Shape<_2,_2>>{}, Tile<_32,_32,_16>{});
    Copy_Atom<SM75_U32x4_LDSM_N, Element> s2r_atom_a;
    Copy_Atom<SM75_U16x8_LDSM_T, Element> s2r_atom_b;
    CUTE_STATIC_ASSERT_V(size(mma) == Int<128>{});
    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
    static_assert(sizeof(Element) == 2);

    struct SharedStorage {
        ArrayEngine<Element, cosize_v<decltype(sa_layout)>> a;
        ArrayEngine<Element, cosize_v<decltype(sb_layout)>> b;
    };
    static_assert(sizeof(SharedStorage) == 16384);
    __shared__ __align__(16) unsigned char shared_bytes[sizeof(SharedStorage)];
    auto &storage = *reinterpret_cast<SharedStorage *>(shared_bytes);
    Tensor sa = make_tensor(make_smem_ptr(storage.a.begin()), sa_layout);
    Tensor sb = make_tensor(make_smem_ptr(storage.b.begin()), sb_layout);
    Tensor ma = make_tensor(make_gmem_ptr(a), select<0,2>(problem), stride_a);
    Tensor mb = make_tensor(make_gmem_ptr(b), select<1,2>(problem), stride_b);
    Tensor mc = make_tensor(make_gmem_ptr(output), select<0,1>(problem), stride_c);
    auto coordinate = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor ga = local_tile(ma, cta, coordinate, Step<_1,X,_1>{});
    Tensor gb = local_tile(mb, cta, coordinate, Step<X,_1,_1>{});
    Tensor gc = local_tile(mc, cta, coordinate, Step<_1,_1,X>{});
    auto copy_thread_a = copy_a.get_slice(threadIdx.x);
    auto copy_thread_b = copy_b.get_slice(threadIdx.x);
    Tensor taga = copy_thread_a.partition_S(ga);
    Tensor tasa = copy_thread_a.partition_D(sa);
    Tensor tbgb = copy_thread_b.partition_S(gb);
    Tensor tbsb = copy_thread_b.partition_D(sb);
    CUTE_STATIC_ASSERT_V(size<1>(taga) == size<1>(tasa));
    CUTE_STATIC_ASSERT_V(size<2>(taga) == size<2>(tasa));
    CUTE_STATIC_ASSERT_V(size<1>(tbgb) == size<1>(tbsb));
    CUTE_STATIC_ASSERT_V(size<2>(tbgb) == size<2>(tbsb));
    auto pipe_count = size<3>(tasa);
    CUTE_STATIC_ASSERT_V(pipe_count == Int<2>{});
    int tiles_left = size<3>(taga);
    int next_tile = 0;
    CUTE_UNROLL
    for (int pipe = 0; pipe < pipe_count - 1; ++pipe) {
        copy(copy_a, taga(_,_,_,next_tile), tasa(_,_,_,pipe));
        copy(copy_b, tbgb(_,_,_,next_tile), tbsb(_,_,_,pipe));
        cp_async_fence();
        --tiles_left;
        if (tiles_left > 0) { ++next_tile; }
    }

    auto thread_mma = mma.get_slice(threadIdx.x);
    Tensor tcgc = thread_mma.partition_C(gc);
    Tensor tcra = thread_mma.partition_fragment_A(sa(_,_,0));
    Tensor tcrb = thread_mma.partition_fragment_B(sb(_,_,0));
    Tensor tcrc = thread_mma.make_fragment_C(tcgc);
    static_assert(std::is_same_v<typename decltype(tcrc)::value_type, float>);
    CUTE_STATIC_ASSERT_V((shape(tcrc) == take<0,3>(shape(tcgc))));
    clear(tcrc);
    auto s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    auto s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    auto s2r_thread_a = s2r_copy_a.get_slice(threadIdx.x);
    auto s2r_thread_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor txsa = s2r_thread_a.partition_S(sa);
    Tensor txsb = s2r_thread_b.partition_S(sb);
    Tensor txra = s2r_thread_a.retile_D(tcra);
    Tensor txrb = s2r_thread_b.retile_D(tcrb);
    int pipe_read = 0;
    int pipe_write = pipe_count - 1;
    Tensor txsa_pipe = txsa(_,_,_,pipe_read);
    Tensor txsb_pipe = txsb(_,_,_,pipe_read);
    auto k_blocks = size<2>(tcra);
    CUTE_STATIC_ASSERT_V(k_blocks == Int<2>{});
    CUTE_STATIC_ASSERT_V(k_blocks == size<2>(txra));
    CUTE_STATIC_ASSERT_V(k_blocks == size<2>(txrb));
    cp_async_wait<0>();
    __syncthreads();
    copy(s2r_atom_a, txsa_pipe(_,_,Int<0>{}), txra(_,_,Int<0>{}));
    copy(s2r_atom_b, txsb_pipe(_,_,Int<0>{}), txrb(_,_,Int<0>{}));

    // Same two-level pipeline ordering as the pinned NVIDIA tutorial.
    CUTE_NO_UNROLL
    while (tiles_left > -(pipe_count - 1)) {
        CUTE_UNROLL
        for (int block = 0; block < k_blocks; ++block) {
            if (block == k_blocks - 1) {
                txsa_pipe = txsa(_,_,_,pipe_read);
                txsb_pipe = txsb(_,_,_,pipe_read);
                cp_async_wait<0>();
                __syncthreads();
            }
            auto next_block = (block + Int<1>{}) % k_blocks;
            copy(s2r_atom_a, txsa_pipe(_,_,next_block), txra(_,_,next_block));
            copy(s2r_atom_b, txsb_pipe(_,_,next_block), txrb(_,_,next_block));
            if (block == 0) {
                copy(copy_a, taga(_,_,_,next_tile), tasa(_,_,_,pipe_write));
                copy(copy_b, tbgb(_,_,_,next_tile), tbsb(_,_,_,pipe_write));
                cp_async_fence();
                --tiles_left;
                if (tiles_left > 0) { ++next_tile; }
                pipe_write = pipe_read;
                pipe_read = pipe_read == pipe_count - 1 ? 0 : pipe_read + 1;
            }
            gemm(mma, tcra(_,_,block), tcrb(_,_,block), tcrc);
        }
    }
    // No read of the previous output and no reduced-precision partial sums.
    cutlass::NumericConverter<Element, float, cutlass::FloatRoundStyle::round_to_nearest> convert;
    CUTE_UNROLL
    for (int i = 0; i < size(tcrc); ++i) { tcgc(i) = convert(tcrc(i)); }
}

extern "C" __global__ __launch_bounds__(128)
void luisa_cute_fp16(const cutlass::half_t *a, const cutlass::half_t *b,
                     const cutlass::half_t *unused, cutlass::half_t *output) {
    (void)unused;
    cute_gemm_1024<cutlass::half_t, cute::SM80_16x8x16_F32F16F16F32_TN>(a, b, output);
}

extern "C" __global__ __launch_bounds__(128)
void luisa_cute_bf16(const cutlass::bfloat16_t *a, const cutlass::bfloat16_t *b,
                     const cutlass::bfloat16_t *unused, cutlass::bfloat16_t *output) {
    (void)unused;
    cute_gemm_1024<cutlass::bfloat16_t, cute::SM80_16x8x16_F32BF16BF16F32_TN>(a, b, output);
}
