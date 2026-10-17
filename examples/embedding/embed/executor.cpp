// bge-m3 embedding engine -- graph executor implementation (plan §3.4/§4/§5).
// See executor.h for the ownership model, trace-hook semantics, and the
// B2 dispatch-contract notes.
//
// ONNX-op mapping of the recorded sequence (plan §4 table):
//   embed_lookup  -> Gather x2 + Add + embedding LayerNormalization (#1)
//   qkv GEMM      -> MatMul + Add(bias) with fused [1024,3072] weight (#3 Bias)
//   attention     -> #4 MatMul(QK^T)+Mul(scale)+mask Add, #5 Softmax, #6
//                    MatMul(PV)+Transpose -- or the M4 fused flash kernel
//   attn.out GEMM -> MatMul + Add (#3 Bias) ; residual Add (#7) ; LN1 (#2)
//   ffn.up GEMM   -> MatMul + Add + Erf-chain (BiasGeluErf epilogue, #3)
//   ffn.down GEMM -> MatMul + Add (#3 Bias) ; residual Add (#7) ; LN2 (#2)
//   slice_cls (#8) + l2_normalize (#9) -> dense pooling path
//   sparse_head / colbert_proj (#10)   -> optional retrieval heads
#include "executor.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

#include "kernels/kernels_attention.h"
#include "kernels/kernels_embed.h"
#include "kernels/kernels_gemm.h"
#include "kernels/kernels_heads.h"
#include "kernels/kernels_norm.h"

namespace luisa::embed {

using compute::Buffer;
using compute::BufferView;
using compute::CommandList;
using compute::Device;
using compute::Shader;
using compute::ShaderOption;

uint32_t embed_bucket_for(uint32_t seq) noexcept {
    static constexpr uint32_t kBuckets[] = {128u, 512u, 2048u, 8192u};// plan §3.4
    for(const auto b : kBuckets) {
        if(seq <= b) {
            return b;
        }
    }
    return kBuckets[3u];
}

namespace {

const Tensor &m_require(const Package &p, const luisa::string &name) {
    const auto *t = p.find(name);
    if(t == nullptr) [[unlikely]] {
        LUISA_ERROR("embedding WeightStore: package is missing tensor \"{}\".", name);
    }
    return *t;
}

// Host exact upcasts for the v2 half blobs (mirrors of the device-side load
// helpers in kernels_gemm.cpp; both are bit-exact for every half value).
[[nodiscard]] float m_bf16_up(uint16_t h) noexcept {
    return std::bit_cast<float>(static_cast<uint32_t>(h) << 16u);
}
[[nodiscard]] float m_fp16_up(uint16_t h) noexcept {
    const uint32_t mag = h & 0x7fffu;
    const bool neg = (h >> 15u) != 0u;
    float pos;
    if(mag >= 0x7c00u) {
        pos = std::bit_cast<float>(0x7f800000u | ((mag & 0x03ffu) << 13u));
    } else if(mag >= 0x0400u) {
        pos = std::bit_cast<float>((mag << 13u) + 0x38000000u);
    } else {
        pos = static_cast<float>(mag) * 5.9604644775390625e-8f;// 2^-24
    }
    return neg ? -pos : pos;
}
[[nodiscard]] float m_half_up(uint16_t h, PackagePrecision p) noexcept {
    return p == PackagePrecision::bf16 ? m_bf16_up(h) : m_fp16_up(h);
}

// The GEMM weight matrices of one layer (the tensors uploaded packed-half in
// the convert-on-read mode; every other tensor goes through the host
// upconverter -- see the WeightStore class comment in executor.h).
constexpr bool m_is_gemm_weight(luisa::string_view name) noexcept {
    // "layer{i}.qkv" / ".attn.out" / ".ffn.up" / ".ffn.down" -- NOT the
    // ".qkv.bias"/".ffn.up.bias" siblings, and not head.* (their kernels are
    // not the GEMM family).
    static constexpr luisa::string_view suffixes[] = {".qkv", ".attn.out", ".ffn.up", ".ffn.down"};
    for(const auto s : suffixes) {
        if(name.size() > s.size() + 6u && name.starts_with("layer") &&
           name.ends_with(s)) {// "layer" + "0".."23" + "." + suffix: the '.' is
            // implied by the suffix start, length guard rejects bare "layerN.qk"
            return true;
        }
    }
    return false;
}

// Plan §3.3 N1(b): fused dense-pool kernel (CLS-slice + L2-normalize in one
// launch, bit-identical to the slice_cls + l2_normalize pair -- scratch harness:
// 0 differing floats over B in {1,16}, H=1024 on dx+vk+fallback). It folds the
// per-batch-row 2B dispatches into 1 and drops the m_cls intermediate round-trip.
// Default ON; EMBED_N1_FUSE_POOL=0 restores the legacy slice_cls + l2_normalize
// loop (kill switch for perf A/B and the toggle-regression gate). Pure dispatch
// fusion: no math/eps/normalization change.
[[nodiscard]] bool embed_fuse_pool_enabled() noexcept {
    static const bool on = [] {
        const auto *raw = std::getenv("EMBED_N1_FUSE_POOL");
        return raw == nullptr || luisa::string_view{raw} != luisa::string_view{"0"};
    }();
    return on;
}

// Plan §3.4 P1: sparse-head readback compaction. The sparse wire stays the
// caller's [B*S] float span (EncodeJob); what changes is how many bytes the
// recorded DtoH copies move into it. Toggle EMBED_P1_SPARSE_COMPACT:
//   "1"/unset -> WINDOW (default): the legacy sparse_head dispatch is kept
//     bit-identical, but the readback copies only each row's real-token
//     prefix [0, N_b) instead of the padded S floats (the scheduler's
//     compaction window, m_finalize, never reads the padded tail; it is
//     zero-filled here instead of carrying the pad rows' unread relu outputs).
//     Bytes moved = 4 B * sum(N_b): never worse than legacy, strictly better
//     on padded batches. No kernel change, no callbacks, fully deterministic.
//   "2" -> PAIRS (the literal §4.3 queue-append design): a compact kernel
//     computes the same y bit-identically, thresholds it, and appends
//     (row, weight) pairs through block-local shared atomics -> ONE global
//     atomic per emitting block; readback = 1-uint count + the pair prefix
//     bounded by sum(N_b) (host-known from the mask); the completion callback
//     expands the order-free pairs back into the caller's span (order-free
//     scatter by unique row index -> the array, and every consumer-visible
//     value, is run-invariant). This model's sparse head is value-dense in
//     practice (golden_sparse.json: 12/13, 9/9, 15/15... of real tokens have
//     w > 0 -> pairs cost 8 B/real-token vs the window's 4 B), so PAIRS is
//     the opt-in audit arm, not the default.
//   "0" -> LEGACY: the pre-P1 full [B*S] float copy.
enum class EmbedP1Mode { legacy, window, pairs };
[[nodiscard]] EmbedP1Mode embed_p1_mode() noexcept {
    static const EmbedP1Mode mode = [] {
        const auto *raw = std::getenv("EMBED_P1_SPARSE_COMPACT");
        const luisa::string_view v{raw != nullptr ? raw : ""};
        if(v == "0") {
            return EmbedP1Mode::legacy;
        }
        if(v == "2") {
            return EmbedP1Mode::pairs;
        }
        return EmbedP1Mode::window;
    }();
    return mode;
}

// Host pass over the uploaded mask: total real tokens and the per-row
// prefix-contiguity precondition the WINDOW mode needs (the scheduler always
// pads trailing rows, mask = 1..1 0..0; anything else falls back to the
// legacy whole-array copy, loudly once).
struct SparseRowWindows {
    bool prefix_ok{true};
    uint32_t real_total{};
    luisa::vector<uint32_t> row_real;// [B]
};

[[nodiscard]] SparseRowWindows m_sparse_row_windows(luisa::span<const float> mask, uint32_t B, uint32_t S) {
    SparseRowWindows w;
    w.row_real.resize(B);
    for(uint32_t b = 0; b < B; b++) {
        const auto *row = mask.data() + static_cast<size_t>(b) * S;
        uint32_t n = 0u;
        while(n < S && row[n] != 0.0f) {
            n++;
        }
        w.real_total += n;
        w.row_real[b] = n;
        for(uint32_t k = n; k < S; k++) {
            if(row[k] != 0.0f) {
                w.prefix_ok = false;
                break;
            }
        }
        if(!w.prefix_ok) [[unlikely]] {
            break;
        }
    }
    return w;
}

// PAIRS-mode host staging: one allocation per encode, shared_ptr-captured by
// the completion callback that expands the pair stream back into the caller's
// span (lc_optimize §8.8.4: callbacks own what they touch).
struct SparsePairStaging {
    luisa::vector<uint32_t> count;// [1]
    luisa::vector<uint32_t> ids;  // [cap]
    luisa::vector<float> weights; // [cap]
};

}// namespace

// B2's launch helpers are DEFINED in kernels_attention.cpp but not declared in
// the parent-owned kernels_attention.h; re-declared verbatim here (namespace
// luisa::embed, EXTERNAL linkage) so the executor dispatch matches the kernel geometry exactly
// (see executor.h DISPATCH CONTRACT).
uint2 attention_scores_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept;
uint2 attention_pv_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim) noexcept;
uint32_t softmax_row_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept;
uint2 flash_attention_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept;
uint32_t flash_attention_scratch_floats(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept;


WeightStore::WeightStore(Device &device, const Package &package)
    : m_device{device}, m_pkg_precision{package.precision()} {

    const auto L = package.num_layers();
    m_hidden = package.hidden_size();
    m_heads = package.num_heads();
    m_head_dim = m_hidden / m_heads;
    m_inter = package.intermediate_size();
    if(m_hidden == 0u || m_heads == 0u || m_hidden % m_heads != 0u || L == 0u) [[unlikely]] {
        LUISA_ERROR("embedding WeightStore: invalid package hyper-parameters (layers={}, H={}, heads={}).",
                    L, m_hidden, m_heads);
    }

    // --- weight-format toggle (opt plan §3.1 G7 / §4 Q1-Q2) ------------------
    // auto-detect from the package; LUISA_EMBED_WEIGHTS may force the fp32
    // A/B arm or re-interpret -- mismatches are refused loudly.
    const bool pkg_half = m_pkg_precision != PackagePrecision::fp32;
    bool packed_half = pkg_half;// upload v2 blobs verbatim + convert-on-read
    if(const auto *env = std::getenv("LUISA_EMBED_WEIGHTS"); env != nullptr && env[0] != '\0') {
        luisa::string req;
        req.reserve(6u);
        for(const char *c = env; *c != '\0'; c++) {
            req.push_back((*c >= 'A' && *c <= 'Z') ? static_cast<char>(*c + 32) : *c);
        }
        PackagePrecision want;
        if(req == "fp32") {
            want = PackagePrecision::fp32;
        } else if(req == "bf16") {
            want = PackagePrecision::bf16;
        } else if(req == "fp16") {
            want = PackagePrecision::fp16;
        } else [[unlikely]] {
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" is not fp32|bf16|fp16 (fail-closed).", env);
        }
        if(!pkg_half && want != PackagePrecision::fp32) [[unlikely]] {
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" asks for the {} convert-on-read "
                        "kernel path, but the package is fp32 (v1) -- its blobs are not packed-half buffers; "
                        "pack a v2 --precision {} package to use it.",
                        env, req, req);
        }
        if(pkg_half && want != PackagePrecision::fp32 && want != m_pkg_precision) [[unlikely]] {
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" disagrees with the package "
                        "format '{}' (bf16 and fp16 bit patterns must not be reinterpreted across).",
                        env, package_precision_name(m_pkg_precision));
        }
        packed_half = pkg_half && want != PackagePrecision::fp32;
        LUISA_INFO("embedding WeightStore: package precision '{}', LUISA_EMBED_WEIGHTS='{}' -> device mode '{}'.",
                   package_precision_name(m_pkg_precision), env,
                   pkg_half ? (packed_half ? "packed-half convert-on-read" : "fp32 convert-on-upload (A/B)")
                            : "fp32");
    }
    if(pkg_half && packed_half) {
        LUISA_INFO("embedding WeightStore: half-precision package '{}' -> GEMM weights uploaded packed "
                   "({} B/elt), converted at the kernel B-tile load; fp32 accumulation unchanged.",
                   package_precision_name(m_pkg_precision), 2);
    } else if(pkg_half) {
        LUISA_INFO("embedding WeightStore: half-precision package '{}' in fp32 A/B mode -- every tensor "
                   "up-converted at upload, fp32 kernel path.",
                   package_precision_name(m_pkg_precision));
    }
    m_gemm_format = (pkg_half && packed_half)
                        ? (m_pkg_precision == PackagePrecision::bf16 ? GemmWeightFormat::Bf16 : GemmWeightFormat::Fp16)
                        : GemmWeightFormat::Fp32;

    // Role-aware tensor acquisition. allow_packed tensors (the four GEMM weight
    // matrices per layer) keep the packed-half device layout in the
    // convert-on-read mode; everything else -- and every tensor in the fp32
    // A/B mode -- is up-converted host-side at construction time. The
    // conversion results live in m_owned_scratch for the store's lifetime, so
    // the upload spans borrow from storage owned by this WeightStore (the same
    // "outlives the recorded commands" contract as the Package image).
    const auto m_take = [&](const luisa::string &name, Buffer<float> &dst, bool allow_packed) {
        const auto &t = m_require(package, name);
        const auto n = static_cast<size_t>(t.element_count());
        if(m_pkg_precision == PackagePrecision::fp32) {
            dst = m_device.create_buffer<float>(n);
            m_uploads.push_back(UploadItem{&dst, t.data, n});
            return;
        }
        if(allow_packed && packed_half) {
            const auto words = (n + 1u) / 2u;// two LE halves per device float
            dst = m_device.create_buffer<float>(words);
            m_owned_scratch.emplace_back(words, 0.0f);
            auto &scratch = m_owned_scratch.back();
            std::memcpy(scratch.data(), t.half_data, n * 2u); // exact blob bytes;
            // zero-init covers the 2-byte tail of an odd n (no blob overrun).
            m_uploads.push_back(UploadItem{&dst, scratch.data(), words});
            return;
        }
        dst = m_device.create_buffer<float>(n);
        m_owned_scratch.emplace_back(n);
        auto &scratch = m_owned_scratch.back();
        for(size_t i = 0u; i < n; i++) {
            scratch[i] = m_half_up(t.half_data[i], m_pkg_precision);
        }
        m_uploads.push_back(UploadItem{&dst, scratch.data(), n});
    };

    m_take("emb.word", m_word, false);
    m_take("emb.pos", m_pos, false);
    m_take("emb.ln.gamma", m_emb_ln_gamma, false);
    m_take("emb.ln.beta", m_emb_ln_beta, false);

    m_layers.reserve(L);
    for(uint32_t i = 0; i < L; i++) {
        m_layers.emplace_back();
        auto &lw = m_layers.back();// fill IN PLACE: UploadItem stores pointers to
        // these member buffers, so they must never move (m_layers is reserved
        // above and append-only afterwards).
        m_take(luisa::format("layer{}.qkv", i), lw.qkv, true);
        m_take(luisa::format("layer{}.qkv.bias", i), lw.qkv_bias, false);
        m_take(luisa::format("layer{}.attn.out", i), lw.attn_out, true);
        m_take(luisa::format("layer{}.attn.out.bias", i), lw.attn_out_bias, false);
        m_take(luisa::format("layer{}.ln1.gamma", i), lw.ln1_gamma, false);
        m_take(luisa::format("layer{}.ln1.beta", i), lw.ln1_beta, false);
        m_take(luisa::format("layer{}.ffn.up", i), lw.ffn_up, true);
        m_take(luisa::format("layer{}.ffn.up.bias", i), lw.ffn_up_bias, false);
        m_take(luisa::format("layer{}.ffn.down", i), lw.ffn_down, true);
        m_take(luisa::format("layer{}.ffn.down.bias", i), lw.ffn_down_bias, false);
        m_take(luisa::format("layer{}.ln2.gamma", i), lw.ln2_gamma, false);
        m_take(luisa::format("layer{}.ln2.beta", i), lw.ln2_beta, false);
    }

    m_take("head.sparse.w", m_sparse_w, false);// [H,1] == plain [H] dot vector
    {
        const auto &sb = m_require(package, "head.sparse.b");
        m_sparse_b = m_pkg_precision == PackagePrecision::fp32 ? sb.data[0] : m_half_up(sb.half_data[0], m_pkg_precision);
    }
    m_take("head.colbert.w", m_colbert_w, false);
    m_take("head.colbert.b", m_colbert_b, false);
}

void WeightStore::record_upload(CommandList &cl) const {
    for(const auto &u : m_uploads) {
        cl << u.buffer->view().copy_from(luisa::span{u.src, u.count});
    }
}

Executor::Executor(Device &device, const WeightStore &weights, compute::Stream &owner_stream) noexcept
    : m_device{device}, m_owner_stream{owner_stream}, m_weights{weights} {

    const auto compile = [&](auto &&kernel, const char *name) {
        ShaderOption opt;
        opt.name = name;
        return m_device.compile(std::forward<decltype(kernel)>(kernel), opt);
    };

    // GEMM kernels are compiled in the B-operand format the WeightStore
    // uploads (opt plan §3.1 G7): Fp32 -> the historical bit-identical path;
    // Bf16/Fp16 -> the convert-on-read packed-B load (fp32 accumulation).
    const auto gemm_wf = m_weights.gemm_weight_format();
    const luisa::string wf_sfx = gemm_wf == GemmWeightFormat::Fp32
                                    ? luisa::string{}
                                    : luisa::format("_{}", gemm_weight_format_name(gemm_wf));
    const auto gname = [&wf_sfx](luisa::string_view base) {
        return luisa::string{base} + wf_sfx;
    };

    m_embed = compile(make_embed_lookup_kernel(), "emb_lookup");
    m_gemm_bias = compile(make_gemm_kernel(GemmEpilogue::Bias, gemm_wf), gname("gemm_bias").c_str());
    m_gemm_gelu = compile(make_gemm_kernel(GemmEpilogue::BiasGeluErf, gemm_wf), gname("gemm_bias_gelu_erf").c_str());
    m_residual = compile(make_residual_add_kernel(), "residual_add");
    m_layernorm = compile(make_layernorm_kernel(), "layernorm");
    m_l2norm = compile(make_l2_normalize_kernel(), "l2_normalize");
    m_slice_cls = compile(make_slice_cls_kernel(), "slice_cls");
    m_dense_pool = compile(make_dense_pool_kernel(), "dense_pool");
    m_sparse_head = compile(make_sparse_head_kernel(), "sparse_head");
    m_sparse_head_compact = compile(make_sparse_head_compact_kernel(), "sparse_head_compact");
    m_sparse_count_reset = compile(make_sparse_count_reset_kernel(), "sparse_count_reset");
    m_colbert_proj = compile(make_colbert_proj_kernel(), "colbert_proj");
    m_attn_scores = compile(make_attention_scores_kernel(), "attn_scores");
    m_attn_softmax = compile(make_softmax_row_kernel(), "softmax_row");
    m_attn_pv = compile(make_attention_pv_kernel(), "attn_pv");
    // m_flash stays unset: lazily compiled on the first S > 1024 encode (M4).
}

void Executor::m_ensure_pool(uint32_t rows, uint32_t batch, uint32_t seq) {

    const uint32_t H = m_weights.hidden_size();
    const uint32_t I = m_weights.intermediate_size();

    if(rows > m_pool_rows) {
        // Grow-only reallocation (rare, amortized): flush the owning worker
        // stream before the replaced buffers are destroyed, so the vk
        // validation layer retires its resource-usage bookkeeping
        // ("destroyed while stream is still using it" abort under
        // LUISA_ENABLE_VALIDATION). The scheduler only dispatches to an IDLE
        // worker, so this sync is a zero-wait bookkeeping flush, never a
        // completion stall; the non-growth fast path never synchronizes.
        m_pool_rows = rows;
        const auto rf = static_cast<size_t>(rows);
        m_ids_dev = m_device.create_buffer<uint32_t>(rf);
        m_mask_dev = m_device.create_buffer<float>(rf);
        m_h0 = m_device.create_buffer<float>(rf * H);
        m_h1 = m_device.create_buffer<float>(rf * H);
        m_qkv = m_device.create_buffer<float>(rf * 3u * H);
        m_att = m_device.create_buffer<float>(rf * H);
        m_tmp1 = m_device.create_buffer<float>(rf * H);
        m_tmp2 = m_device.create_buffer<float>(rf * H);
        m_ffn = m_device.create_buffer<float>(rf * I);
        m_cls = m_device.create_buffer<float>(H);
        m_dense_dev = m_device.create_buffer<float>(rf * H);// B*H <= B*seq*H (seq >= 1)
        m_sparse_dev = m_device.create_buffer<float>(rf);
        m_colbert_dev = m_device.create_buffer<float>(rf * H);
        // P1 "pairs" mode staging (EMBED_P1_SPARSE_COMPACT=2); allocated with
        // the pool so a grow never sees a torn buffer. The counter is a fixed
        // [1] buffer: (re)created only while empty (a grow rebinds the others
        // in place).
        m_sparse_pair_ids = m_device.create_buffer<uint32_t>(rf);
        m_sparse_pair_w = m_device.create_buffer<float>(rf);
        if(!m_sparse_count) {
            m_sparse_count = m_device.create_buffer<uint32_t>(1u);
        }
    }

    if(seq <= kNaiveAttentionMaxSeq) {
        // logits workspace [B*heads, S, S] fp32 (plan §5: <= 64 MiB at B=1,S=1024)
        const auto need = static_cast<size_t>(batch) * m_weights.num_heads() * seq * seq;
        if(need > m_logits_floats) {
            m_owner_stream.synchronize();// retire-usage flush (see above)
            m_logits_floats = static_cast<uint32_t>(need);
            m_logits = m_device.create_buffer<float>(need);
        }
    } else {
        // flash scratch: DUMMY per the B2 contract (1 float, never touched)
        if(!m_flash_scratch) {
            const auto need = static_cast<size_t>(flash_attention_scratch_floats(batch, seq, m_weights.num_heads()));
            m_flash_scratch = m_device.create_buffer<float>(need);
        }
    }
}

void Executor::m_ensure_trace_slots(uint32_t floats_per_slot) {
    if(floats_per_slot <= m_trace_floats) {
        return;
    }
    m_owner_stream.synchronize();// same retire-usage note as m_ensure_pool
    m_trace_floats = floats_per_slot;
    m_owner_stream.synchronize();// retire-usage flush before slot replacement
    for(uint32_t k = 0; k < kTraceStates; k++) {
        m_trace_slots[k] = m_device.create_buffer<float>(floats_per_slot);
        m_trace_slots[k].set_name(luisa::format("embed_trace_slot_{}", k));
    }
}

void Executor::record_encode(CommandList &cl, const EncodeJob &job) {

    const auto B = job.batch;
    const auto S = job.seq;
    const auto H = m_weights.hidden_size();
    const auto heads = m_weights.num_heads();
    const auto hd = m_weights.head_dim();
    const auto I = m_weights.intermediate_size();
    const auto L = m_weights.num_layers();
    const uint32_t rows = B * S;
    const auto rf = static_cast<size_t>(rows);

    if(S == 0u || B == 0u || S + kEmbedPosOffset > 8194u) [[unlikely]] {
        LUISA_ERROR("executor: invalid job shape B={} S={} (position table ceiling is 8194).", B, S);
    }
    if(job.ids.size() != rf || job.mask.size() != rf || job.dense_out.size() != static_cast<size_t>(B) * H) [[unlikely]] {
        LUISA_ERROR("executor: job staging sizes do not match B*S (ids={}, mask={}, dense={}).",
                    job.ids.size(), job.mask.size(), job.dense_out.size());
    }

    m_ensure_pool(rows, B, S);

    const bool tracing = static_cast<bool>(m_trace_cb);
    if(tracing) {
        m_ensure_trace_slots(static_cast<uint32_t>(rf * H));
    }
    m_traced_states = 0u;
    m_trace_rows = rows;
    m_trace_cols = H;

    const auto trace = [&](uint32_t layer, BufferView<float> src) {
        if(tracing) {
            const auto n = rf * H;
            cl << m_trace_slots[layer].view(0, n).copy_from(src.subview(0, n));
            m_traced_states = layer + 1u;
        }
    };

    //--- staging upload -------------------------------------------------------
    cl << m_ids_dev.view(0, rf).copy_from(job.ids)
       << m_mask_dev.view(0, rf).copy_from(job.mask);

    //--- embedding block (kernel #1; emb.word already carries the folded
    // token-type row 0; XLM-R position offset +2; padded rows keep the uniform
    // offset and are masked/dropped downstream) ------------------------------
    // ADAPTATION (documented deviation, B1): the shipped embed_lookup kernel
    // reads pos_emb at row (s) and IGNORES its pos_offset argument (proved by
    // direct-call diff vs. the host reference in test_embed_executor's BGEM3_DIAG
    // path; Phase-A only ever passed 0). The executor compensates by binding a
    // view of emb.pos shifted down by kEmbedPosOffset rows, and passes
    // pos_offset = 0. This is correct both pre- and post-fix of kernel #1: a
    // fixed kernel adding a 0 offset is a no-op. The +2 shift reproduces
    // XLM-R's position convention (first real token = row 2; pack_model.cpp
    // header) so traced layer 0 matches golden_layers.bin to 4.8e-7.
    const auto pos_rows = m_weights.pos_emb().size();// full table capacity
    cl << m_embed(m_weights.word_emb().view(),
                  m_weights.pos_emb().view(static_cast<size_t>(kEmbedPosOffset) * H,
                                           (pos_rows / H - kEmbedPosOffset) * H),
                  m_ids_dev.view(0, rf),
                  m_weights.emb_ln_gamma().view(), m_weights.emb_ln_beta().view(),
                  m_h0.view(0, rf * H),
                  S, H, kEmbedLayerNormEps, 0u)
             .dispatch(static_cast<uint>(rf * kEmbedBlock));
    trace(0u, m_h0.view(0, rf * H));

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const auto scores_disp = attention_scores_dispatch(B, S, heads);
    const auto pv_disp = attention_pv_dispatch(B, S, heads, hd);

    //--- 24 encoder layers ----------------------------------------------------
    for(uint32_t i = 0; i < L; i++) {
        const auto &lw = m_weights.layer(i);

        // qkv projection: [rows,H] @ [H,3H] + bias -> qkv
        cl << m_gemm_bias(m_h0.view(0, rf * H), lw.qkv.view(),
                          m_qkv.view(0, rf * 3u * H), lw.qkv_bias.view(),
                          rows, 3u * H, H)
                 .dispatch(gemm_dispatch_size(rows, 3u * H));

        // attention (naive trio for S <= 1024, fused flash above)
        if(S <= kNaiveAttentionMaxSeq) {
            const auto lf = static_cast<size_t>(B) * heads * S * S;
            cl << m_attn_scores(m_qkv.view(0, rf * 3u * H), m_mask_dev.view(0, rf),
                                m_logits.view(0, lf),
                                S, heads, hd, scale, kAttentionNegLarge)
                     .dispatch(scores_disp);
            const uint32_t sm_rows = B * heads * S;
            const uint32_t sm_chunk = std::max(4u, (65535u / heads) & ~3u);// mult of 4
            for(uint32_t r0 = 0u; r0 < sm_rows; r0 += sm_chunk) {
                const uint32_t n = std::min(sm_chunk, sm_rows - r0);
                cl << m_attn_softmax(m_logits.view(static_cast<size_t>(r0) * S,
                                                   static_cast<size_t>(n) * S),
                                     S)
                         .dispatch(n * kAttentionBlock);
            }
            cl << m_attn_pv(m_qkv.view(0, rf * 3u * H), m_logits.view(0, lf),
                            m_att.view(0, rf * H), S, heads, hd)
                     .dispatch(pv_disp);
        } else {
            if(!m_flash.has_value()) [[unlikely]] {
                ShaderOption opt;
                opt.name = "flash_attention";
                m_flash.emplace(m_device.compile(make_flash_attention_kernel(), opt));
            }
            const auto sf = static_cast<size_t>(flash_attention_scratch_floats(B, S, heads));
            cl << (*m_flash)(m_qkv.view(0, rf * 3u * H), m_mask_dev.view(0, rf),
                             m_att.view(0, rf * H), m_flash_scratch.view(0, sf),
                             B, S, heads, hd, scale, kAttentionNegLarge)
                     .dispatch(flash_attention_dispatch(B, S, heads));
        }

        // out projection + residual + LN1
        cl << m_gemm_bias(m_att.view(0, rf * H), lw.attn_out.view(),
                          m_tmp1.view(0, rf * H), lw.attn_out_bias.view(),
                          rows, H, H)
                 .dispatch(gemm_dispatch_size(rows, H))
           << m_residual(m_h0.view(0, rf * H), m_tmp1.view(0, rf * H),
                         m_tmp2.view(0, rf * H), rf * H)
                 .dispatch(static_cast<uint>(rf * H))
           << m_layernorm(m_tmp2.view(0, rf * H), m_h1.view(0, rf * H),
                          lw.ln1_gamma.view(), lw.ln1_beta.view(), H, kEmbedLayerNormEps)
                 .dispatch(static_cast<uint>(rf * kNormBlock));

        // FFN (erf-GELU fused in the up-projection epilogue) + residual + LN2.
        // The result returns to m_h0 in place: m_h0's layer input is fully
        // consumed by the first residual_add before LN2 writes back.
        cl << m_gemm_gelu(m_h1.view(0, rf * H), lw.ffn_up.view(),
                          m_ffn.view(0, rf * I), lw.ffn_up_bias.view(),
                          rows, I, H)
                 .dispatch(gemm_dispatch_size(rows, I))
           << m_gemm_bias(m_ffn.view(0, rf * I), lw.ffn_down.view(),
                          m_tmp1.view(0, rf * H), lw.ffn_down_bias.view(),
                          rows, H, I)
                 .dispatch(gemm_dispatch_size(rows, H))
           << m_residual(m_h1.view(0, rf * H), m_tmp1.view(0, rf * H),
                         m_tmp2.view(0, rf * H), rf * H)
                 .dispatch(static_cast<uint>(rf * H))
           << m_layernorm(m_tmp2.view(0, rf * H), m_h0.view(0, rf * H),
                          lw.ln2_gamma.view(), lw.ln2_beta.view(), H, kEmbedLayerNormEps)
                 .dispatch(static_cast<uint>(rf * kNormBlock));

        trace(i + 1u, m_h0.view(0, rf * H));
    }

    //--- dense pooling: CLS row + L2 normalize (kernels #8, #9) ---------------
    // N1(b): default fused path -- one dense_pool dispatch reads every batch's CLS
    // row straight out of the [B*S, H] final hidden state (m_h0) and writes the
    // L2-normalized [B, H] dense result into m_dense_dev, folding the old per-row
    // slice_cls + l2_normalize pair (2B dispatches + an m_cls round-trip) into one.
    // Bit-identical to the legacy loop below (scratch harness, 0/16384 differing
    // floats on dx+vk+fallback). EMBED_N1_FUSE_POOL=0 restores the legacy pair.
    if(embed_fuse_pool_enabled()) [[likely]] {
        cl << m_dense_pool(m_h0.view(0, rf * H),
                          m_dense_dev.view(0, static_cast<size_t>(B) * H),
                          S, H)
                 .dispatch(static_cast<uint>(B * kNormBlock));
    } else {
        for(uint32_t b = 0; b < B; b++) {
            cl << m_slice_cls(m_h0.view(static_cast<size_t>(b) * S * H, H),
                              m_cls.view(0, H), H)
                     .dispatch(H)
               << m_l2norm(m_cls.view(0, H), m_dense_dev.view(static_cast<size_t>(b) * H, H), H)
                     .dispatch(kNormBlock);
        }
    }

    //--- optional retrieval heads (kernel #10) --------------------------------
    // P1 (opt plan §3.4): sparse readback compaction mode; see embed_p1_mode()
    // and the readback section below. LEGACY and WINDOW share the untouched
    // legacy sparse_head dispatch; PAIRS replaces it with the compact-append
    // kernel (bit-identical y, same tree order -- see kernels_heads.cpp).
    const auto p1_mode = embed_p1_mode();
    // Host mask scan (also used by the readback below): O(B*S) floats, ~us.
    const auto sparse_windows = job.want_sparse
                                    ? m_sparse_row_windows(job.mask, B, S)
                                    : SparseRowWindows{};
    if(job.want_sparse) {
        if(job.sparse_out.size() != rf) [[unlikely]] {
            LUISA_ERROR("executor: want_sparse but sparse_out has {} floats (need {}).",
                        job.sparse_out.size(), rf);
        }
        if(p1_mode == EmbedP1Mode::pairs && sparse_windows.real_total > 0u) {
            cl << m_sparse_count_reset(m_sparse_count.view()).dispatch(32u)
               << m_sparse_head_compact(m_h0.view(0, rf * H), m_weights.sparse_w().view(),
                                        m_weights.sparse_b(), m_mask_dev.view(0, rf),
                                        m_sparse_pair_ids.view(0, rf), m_sparse_pair_w.view(0, rf),
                                        m_sparse_count.view(0, 1), H)
                       .dispatch(static_cast<uint>(rf * kHeadsBlock));
        } else {
            cl << m_sparse_head(m_h0.view(0, rf * H), m_weights.sparse_w().view(),
                                m_weights.sparse_b(), m_sparse_dev.view(0, rf), H)
                     .dispatch(static_cast<uint>(rf * kHeadsBlock));
        }
    }
    if(job.want_colbert) {
        if(job.colbert_out.size() != rf * H) [[unlikely]] {
            LUISA_ERROR("executor: want_colbert but colbert_out has {} floats (need {}).",
                        job.colbert_out.size(), rf * H);
        }
        cl << m_colbert_proj(m_h0.view(0, rf * H), m_weights.colbert_w().view(),
                             m_weights.colbert_b().view(), m_colbert_dev.view(0, rf * H),
                             H, H, 1u)
                 .dispatch(static_cast<uint>(rf * kHeadsBlock));
    }

    //--- readback --------------------------------------------------------------
    cl << m_dense_dev.view(0, static_cast<size_t>(B) * H).copy_to(job.dense_out);
    if(job.want_sparse) {
        // P1 modes (see embed_p1_mode): WINDOW copies each row's real-token
        // prefix out of the legacy [rows] array and zero-fills the (unread)
        // padded tail at record time; PAIRS copies the 1-uint count + the
        // count-bounded pair prefix into executor-owned host staging and
        // re-expands into the caller's span from the list's completion
        // callback (fires after the readback memcpys of this list drained --
        // lc_optimize §8.8; the scheduler's own finalize callback is appended
        // after record_encode returns, i.e. runs after this expansion in the
        // completion-thread order). Any unexpected mask pattern (non-suffix
        // padding -- the scheduler never produces it) or an empty real set
        // falls back to the exact legacy whole-array copy.
        auto *out = job.sparse_out.data();
        if(p1_mode == EmbedP1Mode::pairs && sparse_windows.real_total > 0u) {
            const auto cap = sparse_windows.real_total;
            auto staging = luisa::make_shared<SparsePairStaging>();
            staging->count.resize(1u);
            staging->ids.resize(cap);
            staging->weights.resize(cap);
            const auto dst = job.sparse_out;
            cl << m_sparse_count.view(0, 1).copy_to(luisa::span<uint32_t>{staging->count})
               << m_sparse_pair_ids.view(0, cap).copy_to(luisa::span<uint32_t>{staging->ids})
               << m_sparse_pair_w.view(0, cap).copy_to(luisa::span<float>{staging->weights});
            cl.add_callback([staging, dst]() noexcept {
                std::memset(dst.data(), 0, dst.size() * sizeof(float));
                const uint32_t n = staging->count[0];
                const auto lim = std::min<size_t>(n, staging->ids.size());
                for(size_t i = 0; i < lim; i++) {
                    const auto t = staging->ids[i];
                    if(t < dst.size()) [[likely]] {
                        dst[t] = staging->weights[i];
                    }
                }
            });
        } else if(p1_mode == EmbedP1Mode::window && sparse_windows.prefix_ok) {
            for(uint32_t b = 0; b < B; b++) {
                const auto off = static_cast<size_t>(b) * S;
                const auto n = sparse_windows.row_real[b];
                if(n > 0u) {
                    cl << m_sparse_dev.view(off, n)
                              .copy_to(luisa::span<float>{out + off, n});
                }
                if(n < S) {
                    std::memset(out + off + n, 0, static_cast<size_t>(S - n) * sizeof(float));
                }
            }
        } else {
            if(p1_mode == EmbedP1Mode::window) [[unlikely]] {
                static std::atomic_flag warned{};
                if(!warned.test_and_set()) {
                    LUISA_WARNING("embedding P1 window: non-suffix padding mask at B={} S={}; "
                                  "using the legacy whole-array sparse readback for this encode.",
                                  B, S);
                }
            }
            cl << m_sparse_dev.view(0, rf).copy_to(job.sparse_out);
        }
    }
    if(job.want_colbert) {
        cl << m_colbert_dev.view(0, rf * H).copy_to(job.colbert_out);
    }
}

void Executor::fire_trace_callbacks() noexcept {
    if(!m_trace_cb) {
        return;
    }
    const auto n = static_cast<size_t>(m_trace_rows) * m_trace_cols;
    for(uint32_t k = 0; k < m_traced_states; k++) {
        m_trace_cb(k, m_trace_slots[k].view(0, n), m_trace_rows, m_trace_cols);
    }
    m_traced_states = 0u;
}

}// namespace luisa::embed
