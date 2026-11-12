#pragma once
// OCR engine -- decode KV cache + device-visible decode metadata (Phase 3).
//
// DISTINCT cache interfaces (plan §7.2) -- they never collapse into one:
//   1. ContiguousKvCache: the correctness baseline for batch 1. One
//      [layers, capacity, kv_heads, head_dim] K buffer and one V buffer;
//      append writes the next `count` token rows, read exposes exactly the
//      written prefix. Layout matches the Phase 2 decode-attention kernel's
//      [batch=1, kv_capacity, kv_heads, head_dim] expectation per layer.
//   2. PagedKvCache: a separate OPTIONAL interface for continuous batching /
//      fragmentation. Phase 3 declares it and fails closed (unsupported);
//      the contiguous cache is NOT implemented on top of it.
//   3. DecodeState: device-visible position/step/state metadata for decode
//      (and later graph replay). The device buffers are the source of truth
//      during a decode step; the host mirrors them only at synchronization
//      points. The Phase 2 decode-attention kernel reads kv_len/positions
//      DIRECTLY from these buffers -- no per-step host re-upload.
//
// Capacities are derived from the ACTUAL checkpoint geometry (layers, KV
// heads, head dim, bytes/element, max sequence, batch) via checked 64-bit
// math and preflighted against an explicit byte budget BEFORE allocation.
// A completed sequence releases its capacity IMMEDIATELY (release()), with
// deferred buffer destruction handled by the runtime; there is no pool churn
// and no synchronous reallocation per step.
//
// FP32 storage only in Phase 3 (the compute baseline); dtype_bytes is part of
// the capacity math so later packed storage plugs in without API changes.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/status.h"

namespace luisa::ocr {

// Forward declaration: the cache preflights against a MemoryBudget without
// pulling api.h/model_recipe.h (whose host-only AttentionMaskKind would clash
// with the ops enum in device TUs). memory_budget.h is included by the .cpp.
class MemoryBudget;

// Geometry of one contiguous KV cache. All fields concrete; checked.
struct KvCacheConfig {
    uint32_t num_layers{0};
    uint32_t num_kv_heads{0};
    uint32_t head_dim{0};
    uint32_t max_seq_len{0};// token capacity per sequence (prompt + output)
    uint32_t batch{1};      // Phase 3 contiguous baseline is batch 1
    uint32_t dtype_bytes{4};// FP32 storage baseline

    [[nodiscard]] Status validate() const;
    // Checked total bytes: layers * batch * tokens * kv_heads * head_dim * 2
    // * dtype_bytes (plan §7.2), via the shared memory_budget helper.
    [[nodiscard]] Result<uint64_t> total_bytes() const;
    // Checked per-layer token-row element count (tokens * kv_heads * head_dim).
    [[nodiscard]] Result<uint64_t> layer_elements() const;
};

// Device-visible decode metadata. Buffers are sized for `batch` entries:
//   kv_len    [batch] uint32 : committed KV token count per sequence
//   positions [batch] int32  : absolute position of the NEXT decode query
//   step      [batch] uint32 : number of decode steps executed
//   state     [batch] uint32 : DecodeSeqState
// The host writes them through record_* helpers (luisa::span transfers) at
// sequence boundaries; kernels read them in-place. `finished` sequences are
// never attended (the executor stops scheduling them).
enum class DecodeSeqState : uint32_t {
    empty = 0,    // no sequence bound
    prefill = 1,  // prompt committed, decode not started
    decoding = 2, // decode steps in progress
    finished = 3, // EOS / stop / budget reached; capacity released
};

class DecodeState {
public:
    DecodeState() noexcept = default;

    // Allocate + zero the metadata buffers. `batch` must be nonzero. The zero
    // initialization is SYNCHRONOUS: create() uploads zeros through `stream`
    // and drains it before returning, so an unbound sequence can never be
    // attended (kv_len=0, state=empty) even before the first record().
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream, uint32_t batch);

    // Record a full metadata snapshot (host -> device). Sizes must equal batch.
    // The snapshot is COPIED into heap-owned staging retained by a completion
    // callback, so the caller's spans need only live through this call (safe
    // for multiple records into one CommandList and for stack-local spans).
    [[nodiscard]] Status record(compute::CommandList &list,
                                luisa::span<const uint32_t> kv_len,
                                luisa::span<const int32_t> positions,
                                luisa::span<const uint32_t> step,
                                luisa::span<const uint32_t> state) const;

    // Convenience single-sequence (batch 1) update of kv_len + position +
    // state after a prefill/append. `step` is preserved.
    [[nodiscard]] Status record_sequence(compute::CommandList &list, uint32_t index,
                                         uint32_t kv_len, int32_t position,
                                         DecodeSeqState state) const;

    [[nodiscard]] uint32_t batch() const noexcept { return _batch; }
    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(_kv_len); }
    [[nodiscard]] const compute::Buffer<uint32_t> &kv_len() const noexcept { return _kv_len; }
    [[nodiscard]] const compute::Buffer<int32_t> &positions() const noexcept { return _positions; }
    [[nodiscard]] const compute::Buffer<uint32_t> &step() const noexcept { return _step; }
    [[nodiscard]] const compute::Buffer<uint32_t> &state() const noexcept { return _state; }
    [[nodiscard]] uint64_t device_bytes() const noexcept;

    // Download a snapshot (caller synchronizes the stream afterwards).
    [[nodiscard]] Status download(compute::Stream &stream,
                                  luisa::vector<uint32_t> &kv_len,
                                  luisa::vector<int32_t> &positions,
                                  luisa::vector<uint32_t> &step,
                                  luisa::vector<uint32_t> &state) const;

private:
    compute::Buffer<uint32_t> _kv_len;
    compute::Buffer<int32_t> _positions;
    compute::Buffer<uint32_t> _step;
    compute::Buffer<uint32_t> _state;
    uint32_t _batch{0};
};

// Abstract contiguous-cache interface. The concrete FP32 implementation below
// is the only Phase 3 backend; the interface keeps paged/packed variants from
// collapsing into it.
class KvCache {
public:
    virtual ~KvCache() noexcept = default;
    [[nodiscard]] virtual const KvCacheConfig &config() const noexcept = 0;
    // Committed token count of sequence `index`.
    [[nodiscard]] virtual uint32_t length(uint32_t index) const noexcept = 0;
    // Release a completed sequence's HOST capacity immediately so a new
    // sequence can bind. This does NOT touch device metadata; the caller
    // records the device-side finished state via record_finished() (the
    // controller does this at the completion boundary). Device buffers are
    // dropped when the cache is destroyed/reset.
    virtual void release(uint32_t index) noexcept = 0;
    [[nodiscard]] virtual uint64_t device_bytes() const noexcept = 0;
};

// Contiguous FP32 KV cache for batch 1 correctness. Owns K and V device
// buffers plus the device-visible DecodeState. Append/read are command-list
// operations: append() records the KV write for the rows the caller just
// produced (the executor binds the cache views directly into attention, so
// no extra copy is needed on the hot path); read_* expose typed views.
class ContiguousKvCache final : public KvCache {
public:
    ContiguousKvCache() noexcept = default;

    // Preflight + allocate. `budget` (optional) is checked BEFORE allocation
    // and updated with the actual resident bytes on success. Fails clearly on
    // overflow / budget refusal / allocation failure. `stream` drains the
    // synchronous DecodeState zero-initialization. The budget preflight and
    // the recorded requirement BOTH include the DecodeState metadata bytes.
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const KvCacheConfig &config, MemoryBudget *budget = nullptr);

    [[nodiscard]] const KvCacheConfig &config() const noexcept override { return _config; }
    [[nodiscard]] uint32_t length(uint32_t index) const noexcept override;
    void release(uint32_t index) noexcept override;
    [[nodiscard]] uint64_t device_bytes() const noexcept override;

    // Bind a new sequence at `index` with `prompt_len` already-written prompt
    // tokens (the executor writes them via the layer views during prefill).
    // Fails when the sequence is occupied or prompt_len exceeds capacity.
    [[nodiscard]] Status bind_sequence(uint32_t index, uint32_t prompt_len);

    // Typed per-layer views over the FULL capacity (attention reads only the
    // committed prefix via DecodeState::kv_len). Shape per layer:
    // [batch, max_seq_len, kv_heads, head_dim] flattened.
    [[nodiscard]] compute::BufferView<float> k_layer(uint32_t layer) const noexcept;
    [[nodiscard]] compute::BufferView<float> v_layer(uint32_t layer) const noexcept;
    // Per-layer view of just token row `token` (one [kv_heads*head_dim] row
    // per batch entry; batch 1 -> contiguous row). Used by the append kernel.
    [[nodiscard]] compute::BufferView<float> k_row(uint32_t layer, uint32_t token) const noexcept;
    [[nodiscard]] compute::BufferView<float> v_row(uint32_t layer, uint32_t token) const noexcept;

    [[nodiscard]] DecodeState &state() noexcept { return _state; }
    [[nodiscard]] const DecodeState &state() const noexcept { return _state; }

    // Host-side committed length bookkeeping (mirrors the device metadata at
    // synchronization points). append_tokens() advances it after the caller
    // proves the KV write completed; it never exceeds capacity.
    [[nodiscard]] Status append_tokens(uint32_t index, uint32_t count);

    // Download the committed KV prefix of one layer (test/oracle access).
    [[nodiscard]] Status download_layer(compute::Stream &stream, uint32_t layer, uint32_t index,
                                        luisa::vector<float> &k_out, luisa::vector<float> &v_out) const;

    // Reset ALL sequences (host bookkeeping + device metadata zero). The zero
    // snapshot is heap-owned staging retained through command completion, so
    // this is safe to commit and drain later (no dangling stack spans). Legal
    // only when no in-flight command references the cache (caller drained).
    [[nodiscard]] Status reset(compute::CommandList &list);

    // Record a sequence's completion into the device metadata (state=finished,
    // kv_len=0, position=0) so a released slot can never be attended on reuse.
    // This is the device-side counterpart of release(); the controller calls
    // it at the completion boundary. Heap-owned staging (completion-retained).
    [[nodiscard]] Status record_finished(compute::CommandList &list, uint32_t index);

    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(_k); }

private:
    KvCacheConfig _config;
    compute::Buffer<float> _k;
    compute::Buffer<float> _v;
    DecodeState _state;
    luisa::vector<uint32_t> _lengths;// host mirror, sized batch
    uint64_t _layer_stride{0};     // elements per layer (batch*tokens*kv_heads*head_dim)
};

// OPTIONAL paged KV interface (plan §7.2 item 2). Declared so later phases
// implement it WITHOUT changing the contiguous path; Phase 3 fails closed.
class PagedKvCache final : public KvCache {
public:
    PagedKvCache() noexcept = default;
    // Always unsupported in Phase 3: no page tables, no allocation.
    [[nodiscard]] Status create(compute::Device &device, const KvCacheConfig &config,
                                uint32_t page_size, MemoryBudget *budget = nullptr);
    [[nodiscard]] const KvCacheConfig &config() const noexcept override { return _config; }
    [[nodiscard]] uint32_t length(uint32_t index) const noexcept override { return 0u; }
    void release(uint32_t /*index*/) noexcept override {}
    [[nodiscard]] uint64_t device_bytes() const noexcept override { return 0u; }

private:
    KvCacheConfig _config;
};

}// namespace luisa::ocr