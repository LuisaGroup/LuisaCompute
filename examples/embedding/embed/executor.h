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
#include "kernels/kernels_embed.h"
#include "kernels/kernels_gemm.h"
#include "kernels/kernels_heads.h"
#include "kernels/kernels_norm.h"
#include "package.h"

namespace luisa::embed {

// Compiled-shader storage type for a Kernel1D/2D factory result.
template<typename K>
using ShaderFor = decltype(std::declval<compute::Device &>().compile(std::declval<K>(), std::declval<compute::ShaderOption &>()));

// bge-m3 constants verified by the packer (pack_model.cpp header; plan §0/§5).
inline constexpr uint32_t kEmbedPosOffset = 2u;             // XLM-R +2
inline constexpr float kEmbedLayerNormEps = 1e-5f;          // config layer_norm_eps
inline constexpr float kAttentionNegLarge = -3.4028234663852886e38f;
inline constexpr uint32_t kNaiveAttentionMaxSeq = 1024u;    // §4: naive path cap

// Activation-pool buckets (plan §3.4). A batch job pads its sequence length to
// the smallest bucket covering the longest member; a solo job runs at its exact
// length (kernels accept arbitrary ragged S).
[[nodiscard]] uint32_t embed_bucket_for(uint32_t seq) noexcept;

// Layer trace hook (contract, see header comment).
using LayerTraceCallback = luisa::function<void(uint32_t layer, luisa::compute::BufferView<float> activations, uint32_t rows, uint32_t cols)>;

// Shared, immutable after upload, device-side copy of every packed tensor the
// forward pass touches. Created once per device from a loaded Package.
class WeightStore {
public:
    WeightStore(compute::Device &device, const Package &package);

    // Appends the host->device upload commands (one copy_from per tensor, via
    // luisa::span -- SAFE-mode compatible) to `cl`. The Package must outlive
    // execution of these commands (Session keeps it alive).
    void record_upload(compute::CommandList &cl) const;

    struct Layer {
        compute::Buffer<float> qkv, qkv_bias, attn_out, attn_out_bias;
        compute::Buffer<float> ln1_gamma, ln1_beta;
        compute::Buffer<float> ffn_up, ffn_up_bias, ffn_down, ffn_down_bias;
        compute::Buffer<float> ln2_gamma, ln2_beta;
    };

    [[nodiscard]] const compute::Buffer<float> &word_emb() const noexcept { return m_word; }
    [[nodiscard]] const compute::Buffer<float> &pos_emb() const noexcept { return m_pos; }
    [[nodiscard]] const compute::Buffer<float> &emb_ln_gamma() const noexcept { return m_emb_ln_gamma; }
    [[nodiscard]] const compute::Buffer<float> &emb_ln_beta() const noexcept { return m_emb_ln_beta; }
    [[nodiscard]] const Layer &layer(uint32_t i) const noexcept { return m_layers[i]; }
    [[nodiscard]] const compute::Buffer<float> &sparse_w() const noexcept { return m_sparse_w; }
    [[nodiscard]] float sparse_b() const noexcept { return m_sparse_b; }
    [[nodiscard]] const compute::Buffer<float> &colbert_w() const noexcept { return m_colbert_w; }
    [[nodiscard]] const compute::Buffer<float> &colbert_b() const noexcept { return m_colbert_b; }
    [[nodiscard]] uint32_t num_layers() const noexcept { return static_cast<uint32_t>(m_layers.size()); }
    [[nodiscard]] uint32_t hidden_size() const noexcept { return m_hidden; }
    [[nodiscard]] uint32_t num_heads() const noexcept { return m_heads; }
    [[nodiscard]] uint32_t head_dim() const noexcept { return m_head_dim; }
    [[nodiscard]] uint32_t intermediate_size() const noexcept { return m_inter; }

private:
    compute::Device &m_device;
    uint32_t m_hidden{}, m_heads{}, m_head_dim{}, m_inter{};
    compute::Buffer<float> m_word, m_pos, m_emb_ln_gamma, m_emb_ln_beta;
    luisa::vector<Layer> m_layers;
    compute::Buffer<float> m_sparse_w;
    float m_sparse_b{};
    compute::Buffer<float> m_colbert_w, m_colbert_b;
    // host->device upload bookkeeping (pointers into the members above, stable
    // because nothing is mutated after construction).
    struct UploadItem {
        compute::Buffer<float> *buffer;
        const float *src;
        size_t count;
    };
    luisa::vector<UploadItem> m_uploads;
};

// One recorded forward pass. All spans are host pointers owned by the caller;
// sizes follow the layout documented in executor.h.
struct EncodeJob {
    uint32_t batch{};                    // B
    uint32_t seq{};                      // padded sequence length (all rows)
    luisa::span<const uint32_t> ids;     // [B*S] token ids (pad = 1)
    luisa::span<const float> mask;       // [B*S] 1.0 real / 0.0 pad
    bool want_sparse{};
    bool want_colbert{};
    luisa::span<float> dense_out;        // [B*H] L2-normalized CLS rows (required)
    luisa::span<float> sparse_out;       // [B*S] lexical weights (when want_sparse)
    luisa::span<float> colbert_out;      // [B*S*H] projected token vectors (when want_colbert)
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

    void set_layer_trace(LayerTraceCallback cb) noexcept { m_trace_cb = std::move(cb); }

    // Records the full forward pass (and trace-slot copies when tracing is on)
    // into `cl` and appends the readback copies into the job's host spans.
    void record_encode(compute::CommandList &cl, const EncodeJob &job);

    // Sync path only: after the worker event fence completed, invoke the
    // trace callback for every traced layer of the most recent recorded job.
    void fire_trace_callbacks() noexcept;

    [[nodiscard]] const WeightStore &weights() const noexcept { return m_weights; }
    [[nodiscard]] uint32_t hidden_size() const noexcept { return m_weights.hidden_size(); }

private:
    void m_ensure_pool(uint32_t rows, uint32_t batch, uint32_t seq);
    void m_ensure_trace_slots(uint32_t floats_per_slot);

    compute::Device &m_device;
    compute::Stream &m_owner_stream;
    const WeightStore &m_weights;

    // Compiled kernel cache (flash is lazy -- stub until B2/M4).
    ShaderFor<EmbedLookupKernel> m_embed;
    ShaderFor<GemmKernel> m_gemm_bias, m_gemm_gelu;
    ShaderFor<ResidualAddKernel> m_residual;
    ShaderFor<LayerNormKernel> m_layernorm;
    ShaderFor<L2NormalizeKernel> m_l2norm;
    ShaderFor<SliceClsKernel> m_slice_cls;
    ShaderFor<SparseHeadKernel> m_sparse_head;
    ShaderFor<ColbertProjKernel> m_colbert_proj;
    ShaderFor<AttentionScoresKernel> m_attn_scores;
    ShaderFor<SoftmaxRowKernel> m_attn_softmax;
    ShaderFor<AttentionPvKernel> m_attn_pv;
    luisa::optional<ShaderFor<FlashAttentionKernel>> m_flash;

    // Grow-only per-worker pool.
    uint32_t m_pool_rows{};   // capacity in token rows (B*S)
    uint32_t m_logits_floats{};
    compute::Buffer<uint32_t> m_ids_dev;                  // staging [rows] token ids
    compute::Buffer<float> m_mask_dev;                    // staging [rows] attention mask
    compute::Buffer<float> m_h0, m_h1;                    // hidden ping [rows*H]
    compute::Buffer<float> m_qkv;                         // [rows*3H]
    compute::Buffer<float> m_att, m_tmp1, m_tmp2;         // [rows*H]
    compute::Buffer<float> m_ffn;                         // [rows*I]
    compute::Buffer<float> m_logits;                      // [B*heads*S*S] (naive)
    compute::Buffer<float> m_flash_scratch;               // [B*heads*S*4] (flash)
    compute::Buffer<float> m_cls, m_dense_dev;            // [H], [B*H]
    compute::Buffer<float> m_sparse_dev, m_colbert_dev;   // [rows], [rows*H]

    // Layer trace (all zero-cost while unset).
    LayerTraceCallback m_trace_cb;
    static constexpr uint32_t kTraceStates = 25u;         // 0..24, golden_layers.bin
    compute::Buffer<float> m_trace_slots[kTraceStates];
    uint32_t m_trace_floats{};                            // slot capacity (elements)
    uint32_t m_traced_states{};                           // states recorded by last encode
    uint32_t m_trace_rows{}, m_trace_cols{};
};

}// namespace luisa::embed
