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
using ProjectQKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer)>;
using ProjectQCoopKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, ByteBuf)>;
// Fused project KV: cKV + Krope from a single H read per token.
using ProjectKVKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer)>;
using ProjectKVCoopKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, ByteBuf, ByteBuf, ByteBuf)>;
// Online attention with shared-memory broadcast of cKV/Krope tiles.
using OnlineAttentionKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer)>;
using OnlineAttentionCoopKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer, ByteBuf, ByteBuf, ByteBuf, ByteBuf)>;
// Classic MHA kernel (broadcast K/V tiles via shared memory).
using MhaOnlineAttentionKernel = luisa::compute::Kernel1D<void(FloatBuffer, FloatBuffer, FloatBuffer, FloatBuffer)>;

// -- Kernel factories ------------------------------------------------------
[[nodiscard]] ProjectQKernel create_project_q_kernel();
[[nodiscard]] ProjectQCoopKernel create_project_q_coop_kernel();
[[nodiscard]] ProjectKVKernel create_project_kv_kernel();
[[nodiscard]] ProjectKVCoopKernel create_project_kv_coop_kernel();
[[nodiscard]] OnlineAttentionKernel create_online_attention_kernel();
[[nodiscard]] OnlineAttentionCoopKernel create_online_attention_coop_kernel();
[[nodiscard]] MhaOnlineAttentionKernel create_mha_online_attention_kernel();

}// namespace mla
