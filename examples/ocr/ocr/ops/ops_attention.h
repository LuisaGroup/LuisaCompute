#pragma once
// OCR engine -- attention operators (Phase 2). Device-free DSL factories.
// FP32 activations, FP32 softmax accumulation.
//
// Explicit AttentionMaskDescriptor (NOT bool-only):
//   BIDIRECTIONAL   : every query attends every key (vision).
//   CAUSAL          : query attends keys <= its absolute position. The causal
//                     diagonal is aligned to the END of the key sequence, so
//                     a short query block over long KV (prefill tail) works.
//   WINDOWED        : causal sliding window of `window` keys.
//   PREFIX_CAUSAL   : keys [0, prefix_len) fully visible; the rest causal.
//   CUSTOM_BLOCK_MASK: a caller-supplied [q_blocks * k_blocks] uint8 mask at
//                     `block_size` granularity. DeepSeek semantics are
//                     expressible: visual blocks bidirectional among
//                     themselves, query blocks see all visual blocks + prior
//                     query blocks (built host-side as data, not hardcoded).
// Optional additive relative bias [q_heads, q_len, k_len] and tanh softcap.
//
// Two prefill paths, A/B tested separately:
//   reference : scores -> FP32 softmax -> PV, one thread per (query, head),
//               two passes over keys (max/sum then weighted V). Naive, exact.
//   online    : FlashAttention-style single pass with running FP32 max/sum and
//               a running FP32 output accumulator, one thread per (query,head).
// Both support GQA/MHA (q_heads a multiple of kv_heads), packed ragged layouts.
//
// Separate contiguous-KV decode API: one query per batch row over a contiguous
// [batch, kv_capacity, kv_heads, head_dim] KV cache with per-batch kv_len and
// absolute query position. No paged cache in Phase 2.
#include <cstdint>

#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

enum class AttentionMaskKind : uint32_t {
    bidirectional = 0,
    causal = 1,
    windowed = 2,
    prefix_causal = 3,
    custom_block_mask = 4,
};

// Named distinctly from the host recipe's attention_mask_kind_name (the two
// AttentionMaskKind enums are separate types; an identical function name would
// collide at link time under MSVC mangling).
[[nodiscard]] luisa::string_view ops_attention_mask_kind_name(AttentionMaskKind kind) noexcept;

// Host-side mask descriptor. The device kernels receive the scalar fields as
// uniform args and the block mask / relative bias as buffers (dummy 1-element
// views when unused). `block_mask` is [ceil(q_len/block_size)*ceil(k_len/block_size)].
struct AttentionMaskDescriptor {
    AttentionMaskKind kind{AttentionMaskKind::bidirectional};
    uint32_t window{0};
    uint32_t prefix_len{0};
    uint32_t block_size{0};
    luisa::vector<uint8_t> block_mask;
    luisa::vector<float> rel_bias;// [q_heads*q_len*k_len], empty = none

    [[nodiscard]] Status validate(uint32_t q_len, uint32_t k_len, uint32_t q_heads) const;
};

// Reference prefill attention. q [q_len, q_heads, head_dim],
// k/v [k_len, kv_heads, head_dim], out [q_len, q_heads, head_dim].
// block_mask is a uint8 buffer (bind dummy when unused); rel_bias float buffer.
using AttentionPrefillKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q*/,
    compute::Buffer<float> /*k*/,
    compute::Buffer<float> /*v*/,
    compute::Buffer<float> /*out*/,
    compute::Buffer<uint32_t> /*block_mask (uint8 packed 4/word)*/,
    compute::Buffer<float> /*rel_bias*/,
    uint32_t /*q_len*/, uint32_t /*k_len*/, uint32_t /*q_heads*/, uint32_t /*kv_heads*/,
    uint32_t /*head_dim*/, float /*scale*/, float /*softcap*/,
    uint32_t /*mask_kind*/, uint32_t /*window*/, uint32_t /*prefix_len*/,
    uint32_t /*block_size*/, uint32_t /*has_rel_bias*/)>;

[[nodiscard]] AttentionPrefillKernel make_attention_prefill_reference_kernel() noexcept;
[[nodiscard]] AttentionPrefillKernel make_attention_prefill_online_kernel() noexcept;

// Contiguous-KV decode attention. q [batch, q_heads, head_dim],
// k/v [batch, kv_capacity, kv_heads, head_dim], kv_len[batch] (uint),
// positions[batch] (int32 absolute query position). out [batch, q_heads, head_dim].
//
// Fail-safe device contract (S2/S3): kv_len is CLAMPED to kv_capacity
// in-kernel (an invalid length can never read OOB); CUSTOM_BLOCK_MASK is
// honored via block_mask (uint8 packed 4/word, block grid over
// ceil(kv_capacity/block_size) key blocks, query block = position/block_size);
// block_size==0 with that kind masks EVERYTHING (zero output), never
// bidirectional. Unknown mask kinds also mask everything. `mask_bytes` is
// the EXACT byte count of the bound block_mask buffer; any mask index at or
// beyond it is disallowed WITHOUT reading (a large position can never read
// OOB even when host validation was skipped).
using AttentionDecodeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q*/,
    compute::Buffer<float> /*k*/,
    compute::Buffer<float> /*v*/,
    compute::Buffer<float> /*out*/,
    compute::Buffer<uint32_t> /*kv_len*/,
    compute::Buffer<int32_t> /*positions*/,
    compute::Buffer<uint32_t> /*block_mask (dummy when unused)*/,
    uint32_t /*batch*/, uint32_t /*kv_capacity*/, uint32_t /*q_heads*/, uint32_t /*kv_heads*/,
    uint32_t /*head_dim*/, float /*scale*/, float /*softcap*/,
    uint32_t /*mask_kind*/, uint32_t /*window*/, uint32_t /*prefix_len*/,
    uint32_t /*block_size*/, uint32_t /*mask_bytes*/)>;
[[nodiscard]] AttentionDecodeKernel make_attention_decode_kernel() noexcept;

// Host-side decode validation (S2): rejects kv_len[b] > kv_capacity and
// malformed custom-block masks BEFORE launch. The kernel additionally clamps
// defensively so a skipped validation can never read OOB.
[[nodiscard]] Status validate_attention_decode_inputs(luisa::span<const uint32_t> kv_len,
                                                      uint32_t kv_capacity,
                                                      const AttentionMaskDescriptor &mask);

// Position-aware overload: additionally rejects a custom block mask that does
// not cover the query block of the maximum NONNEGATIVE position (negative
// positions are fail-safe masked in-kernel and need no mask coverage).
[[nodiscard]] Status validate_attention_decode_inputs(luisa::span<const uint32_t> kv_len,
                                                      luisa::span<const int32_t> positions,
                                                      uint32_t kv_capacity,
                                                      const AttentionMaskDescriptor &mask);

// Dispatch: one thread per (query, head) for prefill, (batch, head) for decode.
[[nodiscard]] Result<LaunchPlan1D> plan_attention_prefill_dispatch(uint64_t q_len, uint64_t q_heads);
[[nodiscard]] Result<LaunchPlan1D> plan_attention_decode_dispatch(uint64_t batch, uint64_t q_heads);

[[nodiscard]] luisa::string attention_variant_key(luisa::string_view variant);

// Pack a uint8 block mask into uint32 words (4 bytes/word, low byte first) for
// the device buffer. Host helper; the kernel reads byte (idx%4) of word idx/4.
[[nodiscard]] luisa::vector<uint32_t> pack_block_mask_words(luisa::span<const uint8_t> mask);

}// namespace luisa::ocr