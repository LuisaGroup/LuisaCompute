#pragma once
// OCR engine -- decode / embedding / sampling operators (Phase 2 + Phase 3).
// Device-free DSL factories; FP32 activations.
//
// Real device kernels for the decode building blocks:
//   embedding_lookup : out[t] = table[ids[t]] (tied-embedding compatible: the
//                      LM head reuses the same table via ops_gemm with B=table
//                      transposed; no separate copy is required).
//   mm_scatter       : multimodal embedding replacement -- replace rows
//                      [start,start+count) of [seq,hidden] embeddings with
//                      vision features [count,hidden].
//   argmax           : greedy argmax per row (deterministic, lowest index wins
//                      ties).
//   softmax          : temperature-scaled FP32 softmax per row (sampling
//                      building block; the deterministic host sampler below
//                      consumes the same logits).
//   kv_append        : (Phase 3) write `count` token rows of K/V into a
//                      contiguous per-layer KV cache at token offset `start`.
//   residual_add / final norm live in ops_norm; LM head is ops_gemm.
//
// Phase 3 also adds HOST-side deterministic sampling building blocks
// (DecodeSamplingConfig + validate + sample_decode_token): explicit seed,
// temperature/top-k/top-p validation, HF-style repetition penalty and
// no-repeat-ngram processors. They are pure host functions over a logits row
// so the exact same code path serves the CPU oracle and the device executor;
// determinism is by construction (fixed-seed SplitMix64, stable ordering).
#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

// Embedding lookup: out[t,d] = table[ids[t], d]. table [vocab, hidden].
using EmbeddingLookupKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*table*/,
    compute::Buffer<int32_t> /*ids*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*num_ids*/, uint32_t /*hidden*/)>;
[[nodiscard]] EmbeddingLookupKernel make_embedding_lookup_kernel() noexcept;

// Multimodal scatter: copy embeddings, replacing rows [start,start+count).
using MmScatterKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*embeddings [seq,hidden]*/,
    compute::Buffer<float> /*features [count,hidden]*/,
    compute::Buffer<float> /*out [seq,hidden]*/,
    uint32_t /*seq*/, uint32_t /*hidden*/, uint32_t /*start*/, uint32_t /*count*/)>;
[[nodiscard]] MmScatterKernel make_mm_scatter_kernel() noexcept;

// Greedy argmax per row: out[row] = argmax_c logits[row,c] (lowest index ties).
using ArgmaxKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*logits [rows,cols]*/,
    compute::Buffer<int32_t> /*out [rows]*/,
    uint32_t /*rows*/, uint32_t /*cols*/)>;
[[nodiscard]] ArgmaxKernel make_argmax_kernel() noexcept;

// Temperature softmax per row (FP32, max-subtracted).
using SoftmaxKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*logits [rows,cols]*/,
    compute::Buffer<float> /*out [rows,cols]*/,
    uint32_t /*rows*/, uint32_t /*cols*/, float /*temperature*/)>;
[[nodiscard]] SoftmaxKernel make_softmax_kernel() noexcept;

// Host validators (S5): recoverable invalid inputs return invalid_argument
// BEFORE launch. The kernels additionally behave defensively (softmax with
// temperature <= 0 or non-finite writes 0; conv with invalid groups writes 0)
// so a skipped validation can never produce NaN-driven garbage or OOB.
[[nodiscard]] Status validate_softmax_inputs(uint64_t rows, uint64_t cols, float temperature);

// ---------------------------------------------------------------------------
// Phase 3: contiguous-KV append kernel.
// ---------------------------------------------------------------------------

// KV append: copy `count` token rows from packed [count, kv_heads*head_dim]
// K/V sources into a per-layer contiguous cache at token offset `start`.
// The destination row stride is `capacity * kv_heads * head_dim` elements
// (the cache's per-layer token capacity); the source is tightly packed.
// One thread per (token, element). `count` may be the whole prompt (prefill)
// or 1 (decode step). The kernel never writes beyond start+count rows.
using KvAppendKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*src_k [count, kv_dim]*/,
    compute::Buffer<float> /*src_v [count, kv_dim]*/,
    compute::Buffer<float> /*dst_k layer view [capacity, kv_dim]*/,
    compute::Buffer<float> /*dst_v layer view*/,
    uint32_t /*count*/, uint32_t /*kv_dim*/, uint32_t /*start*/)>;
[[nodiscard]] KvAppendKernel make_kv_append_kernel() noexcept;
[[nodiscard]] Result<LaunchPlan1D> plan_kv_append_dispatch(uint64_t count, uint64_t kv_dim);
// Host-side bounds check: start+count must fit the token capacity.
[[nodiscard]] Status validate_kv_append(uint64_t start, uint64_t count, uint64_t capacity);

// ---------------------------------------------------------------------------
// Phase 3: host deterministic sampling building blocks.
// ---------------------------------------------------------------------------

// Decoding configuration for one request. `greedy` bypasses all sampling.
// Non-greedy requires temperature > 0; top_k/top_p are validated. The
// repetition/no-repeat-ngram processors match the declared synthetic/
// reference config semantics (HF generate: penalty divides positive logits
// and multiplies negative logits of previously-seen tokens; no-repeat-ngram
// bans any token that would complete a repeated n-gram).
struct DecodeSamplingConfig {
    bool greedy{true};
    float temperature{1.0f};
    uint32_t top_k{0};   // 0 = disabled
    float top_p{1.0f};   // 1 = disabled
    uint64_t seed{0};    // explicit; required for reproducibility
    float repetition_penalty{1.0f};// 1 = disabled; must be > 0
    uint32_t no_repeat_ngram_size{0};// 0 = disabled

    [[nodiscard]] Status validate(uint32_t vocab_size) const;
};

// Validate the sampling config against the vocabulary. Greedy ignores
// temperature/top-k/top-p but still rejects malformed repetition settings.
[[nodiscard]] Status validate_decode_sampling(const DecodeSamplingConfig &config, uint32_t vocab_size);

// Deterministic per-request sampler state. Construct with the validated
// config + vocabulary; feed each generated token via observe() so the
// repetition/no-repeat-ngram processors see the full history (prompt tokens
// are seeded through the constructor's `prompt` span).
class DeterministicSampler {
public:
    DeterministicSampler() noexcept = default;
    DeterministicSampler(DecodeSamplingConfig config, uint32_t vocab_size,
                         luisa::span<const uint32_t> prompt);

    // Pick the next token from a logits row (size == vocab_size). Greedy:
    // argmax with lowest-index tie-break, identical to the device kernel.
    // Sampling: repetition penalty -> no-repeat-ngram ban -> temperature ->
    // top-k -> top-p -> categorical draw from a fixed-seed SplitMix64 stream.
    // The RNG advances exactly once per SUCCESSFUL non-greedy sample; a
    // failure (invalid config, wrong size, or no finite selectable logit) does
    // NOT advance it, so a replayed request with the same seed/logits
    // reproduces the sequence. -inf is an allowed masked/banned entry; NaN and
    // +inf are invalid. If no finite selectable logit remains (all NaN/+inf/
    // banned), sample() returns invalid_argument -- never a silent token 0 or
    // a banned token.
    [[nodiscard]] Result<uint32_t> sample(luisa::span<const float> logits);

    // Record a token that was chosen (by this sampler or by greedy argmax)
    // so the processors' history stays complete. sample() calls this itself.
    void observe(uint32_t token);

    [[nodiscard]] const DecodeSamplingConfig &config() const noexcept { return _config; }

private:
    DecodeSamplingConfig _config;
    uint32_t _vocab_size{0};
    uint64_t _rng_state{0};
    luisa::vector<uint32_t> _history;// prompt + generated tokens
};

// SplitMix64 (fixed, portable, deterministic). Exposed for tests/oracles.
[[nodiscard]] uint64_t decode_splitmix64(uint64_t &state) noexcept;

// Dispatch helpers.
[[nodiscard]] Result<LaunchPlan1D> plan_embedding_dispatch(uint64_t num_ids, uint64_t hidden);
[[nodiscard]] Result<LaunchPlan1D> plan_mm_scatter_dispatch(uint64_t seq, uint64_t hidden);
[[nodiscard]] Result<LaunchPlan1D> plan_argmax_dispatch(uint64_t rows);
[[nodiscard]] Result<LaunchPlan1D> plan_softmax_dispatch(uint64_t rows);

[[nodiscard]] luisa::string decode_variant_key(luisa::string_view op);

}// namespace luisa::ocr