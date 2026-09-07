// Multi-Head Latent Attention (MLA) Example -- DSL kernel factories.
//
// Each factory builds and returns a fully-defined kernel. The DSL lambda
// body is evaluated eagerly at construction time, so any host state captured
// by the kernel (e.g. the RoPE callable) only needs to outlive the factory
// call itself.

#pragma once

#include <luisa/dsl/syntax.h>

namespace mla {

// -- Kernel type aliases ---------------------------------------------------
using FloatBuffer = luisa::compute::Buffer<float>;
using ByteBuf = luisa::compute::ByteBuffer;

// Project queries: Q[b,h,i,d] = Wq[h,d,:] @ H[b,i,:] (shared-memory tiled).
// The unified signature always includes the ByteBuf for cooperative-vector
// access; the fallback path accepts but ignores it.
using ProjectQKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, ByteBuf)>;
// Fused project KV: cKV + Krope from a single H read per token.
// ByteBuf params for Wdkv and Wkr (H_byte_buf was unused in the coop version
// and has been dropped from the unified signature).
using ProjectKVKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer,
                                                       ByteBuf, ByteBuf)>;
// Online attention with shared-memory broadcast of cKV/Krope tiles.
// The unified signature always includes all 4 ByteBufs for cooperative-vector
// access; the fallback path accepts but ignores them.
using OnlineAttentionKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer,
                                                             ByteBuf, ByteBuf, ByteBuf, ByteBuf)>;
// Classic MHA kernel (broadcast K/V tiles via shared memory).
using MhaOnlineAttentionKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer)>;

// -- vLLM-style paged attention kernels ------------------------------------
// Reshape: scatter the dense K/V tensors into the paged KV pool through the
// block table. One thread per dense element. All page geometry is passed as
// runtime scalar args: tokens_per_page, pages_per_seq, elems_per_page.
// Params: dense K, dense V, paged K pool, paged V pool, block table,
//         tokens_per_page, pages_per_seq, elems_per_page.
using ReshapeKVToPagedKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer,
                                                             FloatBuffer, FloatBuffer,
                                                             luisa::compute::Buffer<luisa::uint>,
                                                             luisa::uint, luisa::uint, luisa::uint)>;
// Paged attention: block-table-indirected online softmax, one thread per
// (b, h, i) query row, page -> sub-tile loop structure. The shared-memory
// sub-tile size and the page geometry are baked in at construction
// (host-known; the kernel is compiled per discovered geometry), so the
// factory takes tokens_per_page / elems_per_page as host values while
// pages_per_seq (the block-table stride) stays a runtime kernel argument.
// Params: Q, paged K pool, paged V pool, O, block table, pages_per_seq.
using PagedAttentionKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer,
                                                           FloatBuffer, FloatBuffer,
                                                           luisa::compute::Buffer<luisa::uint>,
                                                           luisa::uint)>;

// -- Kernel factories ------------------------------------------------------
// Template factories: Cooperative=true selects the cooperative-vector inner
// loop; Cooperative=false selects the scalar fallback. Both instantiate the
// same unified kernel signature (the cooperative superset).
template <bool Cooperative>
[[nodiscard]] ProjectQKernel create_project_q_kernel();

template <bool Cooperative>
[[nodiscard]] ProjectKVKernel create_project_kv_kernel();

template <bool Cooperative>
[[nodiscard]] OnlineAttentionKernel create_online_attention_kernel();

[[nodiscard]] MhaOnlineAttentionKernel create_mha_online_attention_kernel();

// Paged factories (host geometry discovered at runtime from the sparse-buffer
// tile size). create_paged_attention_kernel bakes the shared-memory sub-tile
// (largest divisor of tokens_per_page <= paged_sub_tile_max) into the kernel.
[[nodiscard]] ReshapeKVToPagedKernel create_reshape_kv_to_paged_kernel();
[[nodiscard]] PagedAttentionKernel create_paged_attention_kernel(luisa::uint tokens_per_page_host,
                                                                 luisa::uint elems_per_page_host);

}// namespace mla
