#pragma once
// bge-m3 embedding engine -- graph executor (embedding_model_plan.md §3.4,
// milestone M3; dispatch/memory plan §4/§5).
//
// DX/VK GRID-CAP ENVELOPE (parent fix, test-plan wave T3/T5/T8 evidence): a
// single grid dimension is capped at 65535 groups on dx and fails silently
// (ERROR_DEVICE_LOST) on vk. The Scheduler guarantees <= 16383 padded rows
// (B*S) per job (scheduler.cpp kMaxJobRows); at H=1024 that bounds every
// 1D-grid kernel: residual_add groups = 4*rows <= 65532, embed/layernorm/
// heads groups = rows, gemm tiles_m = rows/16, naive-attention y =
// B*heads*ceil(S/16) <= ~rows, flash y = B*heads. The one kernel that scales
// past the cap at legal shapes is softmax_row (groups = B*heads*S = 16*rows),
// so record_encode() chunks its dispatch over row ranges with buffer views.
// Any NEW kernel/call site must keep groups <= 65535 per dimension for every
// job inside this envelope (or get its own chunking), and must NOT derive
// bounds from buffer.size() on dx (it lowers to 0xFFFFFFFF without debug
// info -- kernels_attention.cpp header).
//
// The executor records ONE forward pass (staging upload -> embed_lookup ->
// 24 x {qkv GEMM, attention, out GEMM + residual + LN1, ffn.up GEMM(GELU-erf),
// ffn.down GEMM + residual + LN2} -> slice_cls -> l2_normalize -> heads ->
// readback) into a caller-provided CommandList. All kernels are compiled once
// at Executor construction and cached as Shader objects; every shape is a
// runtime kernel argument, so one compilation serves all (B, S).
//
// OWNERSHIP MODEL (documented choice, plan §3.4 "the executor takes buffers/
// workspace from the caller OR owns a per-instance pool"):
//   - Weights live in a shared read-only WeightStore (created once per device
//     from a loaded Package; uploaded on an init stream by the Scheduler behind
//     the weights_ready fence, §3.5). Multiple Executors (one per worker) hold
//     a const reference to the SAME WeightStore.
//   - Activations/workspaces/staging are a PER-INSTANCE grow-only pool inside
//     the Executor (never shared between workers -> no per-encode sync on them).
//   - Host-side in/out pointers (ids, mask, readback spans) belong to the
//     calling Job and MUST stay alive until the recorded CommandList completes
//     (the Scheduler guarantees this via worker-busy accounting / callbacks).
//
// TRACE HOOK (contract for B3/B4, plan §9.1 D4). Operative semantics:
//   When a callback is set, record_encode() additionally appends a
//   device->device copy of each traced activation into a dedicated trace-slot
//   buffer (25 slots: layer 0 = embedding output, layers 1..24 = post-LN2
//   outputs -- matching src/tests/data/embedding/golden_layers.bin indexing).
//   After the encode has COMPLETED (sync path: the caller invokes
//   fire_trace_callbacks() once the worker event fence is signaled), the
//   executor invokes the callback for each traced layer with a BufferView of
//   the immutable slot buffer plus (rows, cols). The callback runs on the
//   caller's host thread -- NOT inside a device callback -- so the consumer may
//   issue its own `stream << activations.copy_to(host_span) << synchronize()`
//   to move bytes into caller-owned host storage. Zero overhead when unset
//   (no slot buffers are ever allocated). Tracing is intended for synchronous
//   single-session debugging; concurrent encodes on the same worker share slot
//   buffers and are not safe to trace.
//
// ATTENTION CALL SITES (B2 contract, kernels_attention.h):
//   S <= 1024 -> naive trio (#4 scores -> #5 softmax in place -> #6 pv) over an
//   explicit [B*heads, S, S] logits workspace (§5: <= 64 MiB at B=1,S=1024).
//   S >  1024 -> fused flash kernel (#M4), lazily compiled on first use.
//   Both factories currently resolve to B2's loud stub; the flash call site is
//   lazy so gates that never exceed S=1024 are unaffected until B2 lands.
//   DISPATCH CONTRACT: kernels_attention.h does not declare B2's launch
//   helpers, so executor.cpp re-declares them verbatim against the
//   definitions in kernels_attention.cpp and calls them:
//       attention_scores_dispatch(B, S, heads)
//       softmax_row_dispatch(B, S, heads)     (DX 65535-block clamp idiom)
//       attention_pv_dispatch(B, S, heads, head_dim)
//       flash_attention_dispatch(B, S, heads)
//       flash_attention_scratch_floats(B, S, heads)
//   The flash `scratch` argument is a DUMMY (1 float; all online-softmax state
//   is register/shared-resident -- kernels_attention.cpp header contract).

#include <cstdint>
#include <utility>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include "kernels/kernels_attention.h"
#include "kernels/kernels_coop.h"
#include "kernels/kernels_embed.h"
#include "kernels/kernels_embed_vec.h"
#include "kernels/kernels_gemm.h"
#include "kernels/kernels_heads.h"
#include "kernels/kernels_heads_vec.h"
#include "kernels/kernels_norm.h"
#include "kernels/kernels_norm_vec.h"
#include "package.h"
#include "tensor_path.h"

namespace luisa::embed {

// Compiled-shader storage type for a Kernel1D/2D factory result.
template<typename K>
using ShaderFor = decltype(std::declval<compute::Device &>().compile(std::declval<K>(), std::declval<compute::ShaderOption &>()));

// bge-m3 constants verified by the packer (pack_model.cpp header; plan §0/§5).
inline constexpr uint32_t kEmbedPosOffset = 2u;   // XLM-R +2
inline constexpr float kEmbedLayerNormEps = 1e-5f;// config layer_norm_eps
inline constexpr float kAttentionNegLarge = -3.4028234663852886e38f;
// Naive-trio cap (§4). Wave-1 review experiment (O-B evidence: tuned flash
// beats the trio from S≈256) tried lowering this to 256 so S≤512 encodes use
// the tuned flash kernel; it BROKE e2e LEG5 batch-vs-solo BIT-identity: the
// flash online-softmax rescale order depends on the K-tile count, so a solo
// encode at S=300 and a bucket-512 batch job (S=512) round differently, while
// the naive trio's masked-pad reductions stay exact-zero and bit-stable at
// any S. Reverted (§7.4: gates are not softened). Revisit only together with
// a solo-encodes-at-bucket-padding design change.
inline constexpr uint32_t kNaiveAttentionMaxSeq = 1024u;

// Activation-pool buckets (plan §3.4). A batch job pads its sequence length to
// the smallest bucket covering the longest member; a solo job runs at its exact
// length (kernels accept arbitrary ragged S).
[[nodiscard]] uint32_t embed_bucket_for(uint32_t seq) noexcept;

// Layer trace hook (contract, see header comment).
using LayerTraceCallback = luisa::function<void(uint32_t layer, luisa::compute::BufferView<float> activations, uint32_t rows, uint32_t cols)>;

// Shared, immutable after upload, device-side copy of every packed tensor the
// forward pass touches. Created once per device from a loaded Package.
//
// WEIGHT STORAGE FORMAT (optimization plan §3.1 G7 / §4 Q1-Q2, O-G; INT8
// quantization phase 3): three package generations reach this class (BGM3 v1
// fp32, v2 bf16/fp16, v3 int8 -- package.h), and each maps to ONE explicit
// device layout. Nothing is inferred from "the precision is not fp32": a half
// package is uploaded packed-half and an int8 package is uploaded packed-int8,
// and a request that does not match the package is refused (see the toggle
// table below). The class-level `pkg_half` predicate in executor.cpp means
// bf16/fp16 ONLY; int8 is the separate `pkg_int8` path.
//
//   v2 half package, packed auto mode (unchanged since wave 2): the four GEMM
//     weight matrices per layer (qkv, attn.out, ffn.up, ffn.down) are uploaded
//     VERBATIM as packed halves -- ceil(n/2) device floats, two LE 16-bit
//     elements per word (the kernels_gemm.h packed-B contract) -- and
//     converted bf16/fp16 -> fp32 at the kernel B-tile load.
//   v3 int8 package, packed auto mode (phase 3): the same four matrices are
//     uploaded as 1 + ceil(n/4) device float words in the v3 layout
//     [word 0 = the per-tensor positive scale | four signed int8 codes per
//     word, little-endian] (kernels_gemm.h INT8 contract). The upload is the
//     scale word written into zero-initialized owned scratch followed by an
//     exact n-byte memcpy of the blob, then one luisa::span upload of all
//     words, so the 1..3 unused tail bytes of the last word stay zero.
// The kernel decodes `code * scale` at the B-tile fill.
//   v1 fp32 package + bf16/fp16 (phase 1b): the same four matrices end up in
//     the SAME packed-half layout, but produced on the DEVICE at construction
//     time instead of copied from the blob -- each one is uploaded fp32 into a
//     transient staging buffer and packed by make_pack_half_kernel()
//     (kernels_gemm.h; bit-identical to the packer's blob), so the steady state
//     is exactly the v2 packed auto mode above. The staging copies are
//     ONE-SHOT: the Scheduler takes ownership (take_conversion_sources()) and
//     destroys them as soon as the pack dispatches retire, because parking them
//     in this store would keep ~1.1 GiB of fp32 GEMM weights resident for the
//     whole session. Every other tensor stays plain fp32 (host upload, no
//     conversion), and v1 + int8 remains REFUSED -- the device never
//     re-quantizes weights. See the toggle table below.
//   Every other tensor (embeddings, biases, LN gamma/beta, head vectors) is
//     up-converted/dequantized HOST-side once at upload into the fp32 device
//     buffers its (untouched, fp32) kernels expect -- in packed int8 mode that
//   is q * scale through embed/quantization.h for emb.word/emb.pos, every
//   bias, LN gamma/beta, head.colbert.* and the head.sparse.b scalar (which
//   never becomes a device buffer). Their per-encode traffic is O(rows) or
//   O(1), so shrinking their device footprint buys no steady-state bandwidth,
//   while the GEMM matrices are the >1 GB of weight traffic G7 targets.
//   BIASES, LN AND HEAD WEIGHTS STAY FP32 in every packed mode.
//   Measured justification: embed_opt_log wave-2 rows (B=1 encode latency,
//   fp32 vs packed-bf16 in the same build, LUISA_EMBED_WEIGHTS toggle A/B).
//
// ACCUMULATION IS FP32 IN EVERY MODE. bf16/fp16/int8 are STORAGE formats only:
// the element is converted to fp32 before the multiply, the FMA chain, the
// accumulators, the bias/residual operand and every activation stay fp32, and
// there is NO integer/DP4A-style or half tensor-core arithmetic anywhere in
// this engine (kernels_gemm.h header contract).
//
// Toggle: env LUISA_EMBED_WEIGHTS=fp32|bf16|fp16|int8 (case-insensitive; read
// once per WeightStore, logged; unknown values are a loud error). Every
// request is validated against the package format and mismatches fail closed:
//   - package fp32 (v1) + unset/fp32       -> the historical path, byte-identical.
//   - package fp32 (v1) + bf16/fp16       -> the phase-1b device conversion
//     above: the four GEMM matrices are packed ONCE on the GPU (one-shot fp32
//     staging), then read by the packed-half kernels; every other tensor stays
//     fp32. Bit-identical to packing a v2 package of the same format.
//   - package fp32 (v1) + int8            -> REFUSED (LUISA_ERROR): an fp32 blob
//     is not a packed blob, and fp32 -> int8 needs a per-tensor calibrated scale
//     that only the packer computes; the device never re-quantizes weights.
//     Pack a v3 --precision int8 package instead.
//   - package half (v2) + unset           -> auto: packed-half upload +
//     convert-on-read.
//   - package half (v2) + fp32            -> A/B arm: convert everything on
//     upload, fp32 device buffers + fp32 kernels (measures the pure
//     steady-state effect).
//   - package half (v2) + the other half format or int8 -> REFUSED
//     (LUISA_ERROR): the stored bit patterns are not interchangeable.
//   - package int8 (v3) + unset/int8      -> auto: the v3 packed GEMM upload
//     above, host-dequantized fp32 side tensors, and Executors that compile
//     GemmWeightFormat::Int8 GEMM kernels.
//   - package int8 (v3) + fp32            -> A/B arm: ALL tensors dequantized
//     host-side once at upload (q * scale), fp32 device buffers + the original
//     fp32 kernels (the un-quantized reference arm for quality comparisons).
//   - package int8 (v3) + bf16/fp16       -> REFUSED (LUISA_ERROR): signed int8
//     codes plus a scale are not 16-bit elements; pack a v2 package instead.
class WeightStore {
public:
    WeightStore(compute::Device &device, const Package &package);

    // Appends the host->device upload commands (one copy_from per tensor, via
    // luisa::span -- SAFE-mode compatible) to `cl`. The Package must outlive
    // execution of these commands (Session keeps it alive). In the phase-1b
    // v1 fp32 -> bf16/fp16 mode it ALSO appends one pack dispatch per converted
    // GEMM matrix (fp32 staging -> packed destination), so those dispatches read
    // the staging buffers: the list has to be fully retired before the staging
    // is dropped (what the Scheduler does), and this call must precede
    // take_conversion_sources() because the pack bookkeeping points into it.
    void record_upload(compute::CommandList &cl) const;

    // The transient fp32 staging buffers of the v1 fp32 -> packed-half
    // conversion, handed OFF to the caller (normally the Scheduler) exactly
    // once. They have to stay alive until the pack dispatches recorded by
    // record_upload() retire -- and no longer: a bge-m3 package stages ~1.1 GiB
    // of fp32 GEMM weights, so whoever takes them is expected to synchronise the
    // init stream and then drop them, instead of parking that cost in this store
    // for the whole session. Empty for every other package/toggle combination
    // (nothing converted => nothing to free, and no reason to synchronize).
    struct ConversionStaging {
        luisa::vector<compute::Buffer<float>> sources;// the transient fp32 copies
        size_t bytes{};                              // their total size (log/diagnostics)
        [[nodiscard]] bool empty() const noexcept { return sources.empty(); }
    };
    // One-shot: moves the staging buffers out and empties this store's view of
    // them. Must be called AFTER record_upload() (whose pack items borrow from
    // the staging) and BEFORE the fp32 copies are freed, i.e. after the
    // conversion retires.
    [[nodiscard]] ConversionStaging take_conversion_sources() noexcept;

    struct Layer {
        compute::Buffer<float> qkv, qkv_bias, attn_out, attn_out_bias;
        compute::Buffer<float> ln1_gamma, ln1_beta;
        compute::Buffer<float> ffn_up, ffn_up_bias, ffn_down, ffn_down_bias;
        compute::Buffer<float> ln2_gamma, ln2_beta;
    };

    [[nodiscard]] const compute::Buffer<float> &word_emb() const noexcept { return _word; }
    [[nodiscard]] const compute::Buffer<float> &pos_emb() const noexcept { return _pos; }
    [[nodiscard]] const compute::Buffer<float> &emb_ln_gamma() const noexcept { return _emb_ln_gamma; }
    [[nodiscard]] const compute::Buffer<float> &emb_ln_beta() const noexcept { return _emb_ln_beta; }
    [[nodiscard]] const Layer &layer(uint32_t i) const noexcept { return _layers[i]; }
    [[nodiscard]] const compute::Buffer<float> &sparse_w() const noexcept { return _sparse_w; }
    [[nodiscard]] float sparse_b() const noexcept { return _sparse_b; }
    [[nodiscard]] const compute::Buffer<float> &colbert_w() const noexcept { return _colbert_w; }
    [[nodiscard]] const compute::Buffer<float> &colbert_b() const noexcept { return _colbert_b; }
    [[nodiscard]] uint32_t num_layers() const noexcept { return static_cast<uint32_t>(_layers.size()); }
    [[nodiscard]] uint32_t hidden_size() const noexcept { return _hidden; }
    [[nodiscard]] uint32_t num_heads() const noexcept { return _heads; }
    [[nodiscard]] uint32_t head_dim() const noexcept { return _head_dim; }
    [[nodiscard]] uint32_t intermediate_size() const noexcept { return _inter; }
    // B-operand format the Executors on this store must compile their GEMM
    // kernels with (Fp32 for an fp32 package or for the fp32 A/B override of a
    // half/int8 package; Bf16/Fp16 for a half package in packed auto mode OR for
    // a v1 fp32 package converted on the device; Int8 for an int8 package in
    // packed auto mode). See the class comment.
    [[nodiscard]] GemmWeightFormat gemm_weight_format() const noexcept { return _gemm_format; }
    // Package element format as loaded (for reporting; the effective device
    // layout follows gemm_weight_format()).
    [[nodiscard]] PackagePrecision package_precision() const noexcept { return _pkg_precision; }

private:
    compute::Device &_device;
    uint32_t _hidden{}, _heads{}, _head_dim{}, _inter{};
    GemmWeightFormat _gemm_format{GemmWeightFormat::Fp32};
    PackagePrecision _pkg_precision{PackagePrecision::fp32};
    compute::Buffer<float> _word, _pos, _emb_ln_gamma, _emb_ln_beta;
    luisa::vector<Layer> _layers;
    compute::Buffer<float> _sparse_w;
    float _sparse_b{};
    compute::Buffer<float> _colbert_w, _colbert_b;
// host->device upload bookkeeping (pointers into the members above or into
      // _owned_scratch, stable because nothing is mutated after construction).
      // `src` is fp32 package data, a packed-half blob (raw memcpy -- the v2 blob
    // already IS the packed buffer), a packed-int8 scratch image ([scale word |
    // four code bytes per word], built in _owned_scratch), or fp32 host storage
    // built by the half->fp32 / int8->fp32 upload conversion; `count` is always
    // the number of DEVICE FLOATS to write (half a blob's element count when
    // packed half, 1 + ceil(n/4) when packed int8). Every src is held in
    // storage owned by this WeightStore (package image, or _owned_scratch for
    // the packed/copy-converted tensors -- the packed copies avoid reading the
    // blob's 2-byte (half) or 1-byte (int8) tail as floats).
    struct UploadItem {
        compute::Buffer<float> *buffer;
        const float *src;
        size_t count;
    };
    luisa::vector<UploadItem> _uploads;
    // half -> fp32 / int8 -> fp32 upload-conversion (and packed-word) results,
    // alive until the upload commands retire (same lifetime contract as the
    // Package; record_upload's host spans borrow from here). Populated for a
    // half or int8 package's non-GEMM tensors, for its packed GEMM word images,
    // and for the fp32 A/B override of a half/int8 package.
    luisa::vector<luisa::vector<float>> _owned_scratch{};

    // --- v1 fp32 -> packed-half device conversion (phase 1b) -------------------
    // Populated ONLY when the toggle asks for bf16/fp16 on an fp32 (v1) package;
    // empty on every other path, so the packed/int8/fp32 modes are untouched.
    // `_pack_half` is the fp32 -> packed-word converter, compiled in the ctor
    // only when there is something to convert, and owned here because
    // record_upload() (const, possibly long after construction) records its
    // dispatches. `_packs` mirrors `_uploads`: for every converted GEMM matrix it
    // names the transient fp32 staging buffer (an element of
    // `_conversion_sources`) and the persistent packed destination (the Layer
    // member an Executor binds as the packed B operand, exactly as in the v2
    // packed arm). Both vectors are RESERVED before the first Buffer pointer is
    // taken, so the element addresses stay valid until the one-shot handoff.
    struct PackItem {
        compute::Buffer<float> *src;   // fp32 staging, `count` elements
        compute::Buffer<float> *dst;   // packed words, ceil(count / 2)
        size_t count;                  // logical fp32 elements
    };
    luisa::vector<PackItem> _packs{};
    // The transient fp32 copies themselves -- the only storage here that is
    // meant to die early (see take_conversion_sources()). Never destroyed from a
    // backend callback and never kept for the session.
    luisa::vector<compute::Buffer<float>> _conversion_sources{};
    size_t _conversion_bytes{};// total fp32 staging bytes (for the release log)
    luisa::optional<ShaderFor<PackHalfKernel>> _pack_half;
};

// One recorded forward pass. All spans are host pointers owned by the caller;
// sizes follow the layout documented in executor.h.
struct EncodeJob {
    uint32_t batch{};               // B
    uint32_t seq{};                 // padded sequence length (all rows)
    luisa::span<const uint32_t> ids;// [B*S] token ids (pad = 1)
    luisa::span<const float> mask;  // [B*S] 1.0 real / 0.0 pad
    bool want_sparse{};
    bool want_colbert{};
    luisa::span<float> dense_out;// [B*H] L2-normalized CLS rows (required)
    // [B*S] lexical weights (when want_sparse). Under EMBED_P1_SPARSE_COMPACT
    // (opt plan §3.4 P1) the GPU->host wire carries only the real-token values
    // (mode 1, window) or the compact (row, weight) pair stream + count
    // (mode 2, pairs) and is re-expanded into this span at readback/completion;
    // the positions every consumer reads ([b*S + 1, b*S + N-1) per row, the
    // scheduler's compaction window) hold bit-identical legacy values, and the
    // padded tail [b*S + N, b*S + S) is zero-filled (legacy left the pad rows'
    // relu outputs there -- unread, contract-wise "no weight"). The caller's
    // span must stay alive until the recorded CommandList completes, as before.
    luisa::span<float> sparse_out; // [B*S] lexical weights (when want_sparse)
    luisa::span<float> colbert_out;// [B*S*H] projected token vectors (when want_colbert)
};

// Per-worker graph executor. NOT thread-safe: one owner (scheduler worker) at
// a time. Kernels are compiled once and cached; encode() cost is O(1) host
// command recording (~150-290 dispatches, plan §5).
class Executor {
public:
    // `owner_stream` is the single stream this executor's CommandLists are
    // committed on (grow-only buffer reallocations flush it to retire
    // validation-layer usage bookkeeping; see executor.cpp).
    Executor(compute::Device &device, const WeightStore &weights,
             compute::Stream &owner_stream) noexcept;

    void set_layer_trace(LayerTraceCallback cb) noexcept { _trace_cb = std::move(cb); }

    // Records the full forward pass (and trace-slot copies when tracing is on)
    // into `cl` and appends the readback copies into the job's host spans.
    void record_encode(compute::CommandList &cl, const EncodeJob &job);

    // Sync path only: after the worker event fence completed, invoke the
    // trace callback for every traced layer of the most recent recorded job.
    void fire_trace_callbacks() noexcept;

    [[nodiscard]] const WeightStore &weights() const noexcept { return _weights; }
    [[nodiscard]] uint32_t hidden_size() const noexcept { return _weights.hidden_size(); }

private:
    void _ensure_pool(uint32_t rows, uint32_t batch, uint32_t seq);
    void _ensure_trace_slots(uint32_t floats_per_slot);

    compute::Device &_device;
    compute::Stream &_owner_stream;
    const WeightStore &_weights;

    // Compiled kernel cache (flash is lazy -- stub until B2/M4).
    ShaderFor<EmbedLookupKernel> _embed;
    ShaderFor<GemmKernel> _gemm_bias, _gemm_gelu;
    ShaderFor<ResidualAddKernel> _residual;
    ShaderFor<LayerNormKernel> _layernorm;
    // Opt plan C3: ONE dispatch that does residual_add + layernorm over a cached
    // shared-memory residual row, numerically equivalent to the _residual +
    // _layernorm pair above within a focused 1e-5 ABSOLUTE tolerance and with the
    // same reduction order (kernels_norm.h "fused residual_add + layernorm"; not
    // bit-identical -- dx re-emits ~1 ULP). Replaces both the residual1+LN1 and
    // the residual2+LN2 pair on the Legacy path.
    //
    // Mode selection is settled in the constructor (H is fixed per WeightStore):
    // exactly ONE of {_residual/_layernorm, _residual_layernorm} is compiled, so
    // the default path never pays for a shader it cannot dispatch and the
    // shader it does dispatch is always present. _fuse_residual_ln is the single
    // source of truth for both the compile and the record-time branch -- they
    // must never disagree. Off when EMBED_FUSE_RESIDUAL_LN=0 (kill switch:
    // restores the legacy pair's dispatch sequence unchanged) or when the row
    // length exceeds kResidualNormMaxHidden (shared-cache contract).
    ShaderFor<ResidualLayerNormKernel> _residual_layernorm;
    bool _fuse_residual_ln{};
    ShaderFor<L2NormalizeKernel> _l2norm;
    ShaderFor<SliceClsKernel> _slice_cls;
    ShaderFor<DensePoolKernel> _dense_pool;// fused slice_cls+l2 (plan §3.3 N1(b))
    ShaderFor<SparseHeadKernel> _sparse_head;
    // P1 sparse-head readback compaction (opt plan §3.4; EMBED_P1_SPARSE_COMPACT):
    // "pairs" mode (mode 2) swaps the sparse_head dispatch for these two;
    // "window" mode (mode 1, default) keeps sparse_head untouched and only
    // narrows the readback copies (see record_encode).
    ShaderFor<SparseHeadCompactKernel> _sparse_head_compact;
    ShaderFor<SparseCountResetKernel> _sparse_count_reset;
    ShaderFor<ColbertProjKernel> _colbert_proj;
    ShaderFor<AttentionScoresKernel> _attn_scores;
    ShaderFor<SoftmaxRowKernel> _attn_softmax;
    ShaderFor<AttentionPvKernel> _attn_pv;
    luisa::optional<ShaderFor<FlashAttentionKernel>> _flash;

    // TENSOR-PATH OPTIMIZATION SHADERS (tensor_optimize.md 2.1/5.4; opt-in).
    // These are compiled ONLY when the resolved path is Vector or Coop (never
    // on Legacy -- the kill switch stays byte-identical and cheap to start).
    // Default-constructed (empty) when unused; assigned in the ctor body.
    TensorPath _tensor_path{TensorPath::Legacy};

    // vec4 (float4) bandwidth/compute twins of the eager kernels above; bound
    // with BufferView<float>::as<float4>() zero-copy views (H % 4 == 0).
    ShaderFor<EmbedLookupVec4Kernel> _embed_vec4;
    ShaderFor<ResidualAddVec4Kernel> _residual_vec4;
    ShaderFor<LayerNormVec4Kernel> _layernorm_vec4;
    ShaderFor<L2NormalizeVec4Kernel> _l2norm_vec4;
    ShaderFor<DensePoolVec4Kernel> _dense_pool_vec4;
    ShaderFor<SparseHeadVec4Kernel> _sparse_head_vec4;
    ShaderFor<ColbertProjVec4Kernel> _colbert_proj_vec4;

    // coop matvec projection (only compiled when the resolved path is Coop,
    // which already guarantees fp32 weights). Bias / BiasGeluErf epilogues
    // mirror _gemm_bias / _gemm_gelu; the config also feeds the host guards.
    ShaderFor<CoopGemmKernel> _coop_gemm_bias, _coop_gemm_gelu;
    CoopGemmConfig _coop_cfg{};

    // Grow-only per-worker pool.
    uint32_t _pool_rows{};// capacity in token rows (B*S)
    uint32_t _logits_floats{};
    compute::Buffer<uint32_t> _ids_dev;              // staging [rows] token ids
    compute::Buffer<float> _mask_dev;                // staging [rows] attention mask
    compute::Buffer<float> _h0, _h1;                 // hidden ping [rows*H]
    compute::Buffer<float> _qkv;                     // [rows*3H]
    compute::Buffer<float> _att, _tmp1, _tmp2;       // [rows*H]
    compute::Buffer<float> _ffn;                     // [rows*I]
    compute::Buffer<float> _logits;                  // [B*heads*S*S] (naive)
    compute::Buffer<float> _flash_scratch;           // [B*heads*S*4] (flash)
    compute::Buffer<float> _cls, _dense_dev;         // [H], [B*H]
    compute::Buffer<float> _sparse_dev, _colbert_dev;// [rows], [rows*H]
    // P1 "pairs" mode staging (opt plan §3.4; allocated with the pool, used
    // only when EMBED_P1_SPARSE_COMPACT=2): compact (row, weight) append
    // stream + its 1-uint global count. Emissions are mask-gated, so they are
    // bounded by the real-token count <= rows -- the [rows] capacity can never
    // overflow.
    compute::Buffer<uint32_t> _sparse_pair_ids;// [rows]
    compute::Buffer<float> _sparse_pair_w;     // [rows]
    compute::Buffer<uint32_t> _sparse_count;   // [1]

    // Layer trace (all zero-cost while unset).
    LayerTraceCallback _trace_cb;
    static constexpr uint32_t kTraceStates = 25u;// 0..24, golden_layers.bin
    compute::Buffer<float> _trace_slots[kTraceStates];
    uint32_t _trace_floats{}; // slot capacity (elements)
    uint32_t _traced_states{};// states recorded by last encode
    uint32_t _trace_rows{}, _trace_cols{};
};

}// namespace luisa::embed
