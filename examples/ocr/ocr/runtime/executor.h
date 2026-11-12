#pragma once
// OCR engine -- command-recording executor contract (Phase 3).
//
// Ownership rules (plan §7.1), frozen here for every executor:
//   * Weights are IMMUTABLE and SHARED. An executor borrows a WeightStore
//     (installed once by the owner) and never mutates it; several requests
//     may read the same weights concurrently because the buffers are never
//     written after install.
//   * Workspaces are PER REQUEST / PER WORKER. An executor owns (or borrows)
//     a DeviceWorkspace that is reset between requests; no activation region
//     is retained across requests except what the caller explicitly keeps.
//   * Shaders are CACHED for the executor's lifetime (TypedShaderCache);
//     each device-free factory compiles once per variant.
//   * All host transfers use luisa::span; the referenced host memory and
//     device buffers stay alive until the owning stream completes.
//   * No exceptions, no RTTI: every recoverable failure is a Status/Result.
//
// The interface is deliberately small: prefill and single-step decode are
// SEPARATE command-recording entry points. They append to a caller-owned
// CommandList and never synchronize internally; the caller decides when to
// commit/drain (host decode scheduling in Phase 3). This keeps command lists
// bounded to one prefill chunk or one decode step.
#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/runtime/decode_cache.h"
#include "ocr/runtime/weights.h"
#include "ocr/runtime/workspace.h"
#include "ocr/status.h"

namespace luisa::ocr {

// One decode request's inputs. Token ids are host data uploaded by the
// executor through luisa::span copies; multimodal features (when present)
// are already-produced device rows [count, hidden] that REPLACE the prompt
// rows [mm_start, mm_start+count) of the token embeddings.
struct DecodeRequest {
    luisa::vector<uint32_t> prompt_tokens;
    // Optional multimodal replacement (empty = text-only).
    compute::BufferView<float> mm_features;// [mm_count, hidden] device rows
    uint32_t mm_start{0};
    uint32_t mm_count{0};

    [[nodiscard]] bool has_multimodal() const noexcept { return mm_count != 0u; }
    [[nodiscard]] Status validate() const;
};

// Result of a completed decode step / prefill, host-side. Logits are the
// LAST prompt row (prefill) or the single decode row; the executor downloads
// them through a caller-provided span so no hidden host copy is retained.
struct DecodeStepOutput {
    uint32_t position{0};   // absolute position of the row that produced logits
    uint32_t kv_len{0};     // committed KV length AFTER this step
    int32_t token{-1};      // argmax token (greedy); -1 until sampled
};

// Abstract command-recording executor. Concrete implementations (dense VLM
// here; CNN/layout/GDN/MoE in later phases) implement prefill + decode_step
// against their own component graphs. The interface never exposes a paged
// cache or a fused prefill+decode loop.
class Executor {
public:
    virtual ~Executor() noexcept = default;

    // Record the full prompt prefill into `list`: embedding lookup (+ optional
    // multimodal replacement), every decoder block, KV append for the whole
    // prompt, final norm + LM head over the LAST row. On success the cache
    // holds `prompt_len` committed tokens and `logits_out` (size vocab_size)
    // receives the last-row logits when the caller drains the stream.
    // `logits_out` must stay alive until completion.
    [[nodiscard]] virtual Status record_prefill(compute::CommandList &list,
                                                const DecodeRequest &request,
                                                ContiguousKvCache &cache,
                                                luisa::span<float> logits_out) = 0;

    // Record ONE decode step for `input_token` at absolute position ==
    // cache.length(0) before the step. Appends the new K/V row, runs the
    // blocks over the single query, final norm + LM head, and records the
    // advanced device metadata (kv_len/position/step/state). `logits_out`
    // (size vocab_size) receives the row logits after the caller drains.
    [[nodiscard]] virtual Status record_decode_step(compute::CommandList &list,
                                                    uint32_t input_token,
                                                    ContiguousKvCache &cache,
                                                    luisa::span<float> logits_out) = 0;

    // Resident weight bytes this executor reads (diagnostics/budget).
    [[nodiscard]] virtual uint64_t weight_bytes() const noexcept = 0;
    // Current workspace capacity (per-request arena).
    [[nodiscard]] virtual uint64_t workspace_bytes() const noexcept = 0;
};

}// namespace luisa::ocr