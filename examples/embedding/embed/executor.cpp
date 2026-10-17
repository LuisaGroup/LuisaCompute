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
    for (const auto b : kBuckets) {
        if (seq <= b) {
            return b;
        }
    }
    return kBuckets[3u];
}

namespace {

const Tensor &require_tensor(const Package &p, const luisa::string &name) {
    const auto *t = p.find(name);
    if (t == nullptr) [[unlikely]] {
        LUISA_ERROR("embedding WeightStore: package is missing tensor \"{}\".", name);
    }
    return *t;
}

// Host exact upcasts for the v2 half blobs (mirrors of the device-side load
// helpers in kernels_gemm.cpp; both are bit-exact for every half value).
[[nodiscard]] float bf16_up(uint16_t h) noexcept {
    return std::bit_cast<float>(static_cast<uint32_t>(h) << 16u);
}
[[nodiscard]] float fp16_up(uint16_t h) noexcept {
    const uint32_t mag = h & 0x7fffu;
    const bool neg = (h >> 15u) != 0u;
    float pos;
    if (mag >= 0x7c00u) {
        pos = std::bit_cast<float>(0x7f800000u | ((mag & 0x03ffu) << 13u));
    } else if (mag >= 0x0400u) {
        pos = std::bit_cast<float>((mag << 13u) + 0x38000000u);
    } else {
        pos = static_cast<float>(mag) * 5.9604644775390625e-8f;// 2^-24
    }
    return neg ? -pos : pos;
}
[[nodiscard]] float half_up(uint16_t h, PackagePrecision p) noexcept {
    return p == PackagePrecision::bf16 ? bf16_up(h) : fp16_up(h);
}

// The GEMM weight matrices of one layer (the tensors uploaded packed-half in
// the convert-on-read mode; every other tensor goes through the host
// upconverter -- see the WeightStore class comment in executor.h).
constexpr bool is_gemm_weight(luisa::string_view name) noexcept {
    // "layer{i}.qkv" / ".attn.out" / ".ffn.up" / ".ffn.down" -- NOT the
    // ".qkv.bias"/".ffn.up.bias" siblings, and not head.* (their kernels are
    // not the GEMM family).
    static constexpr luisa::string_view suffixes[] = {".qkv", ".attn.out", ".ffn.up", ".ffn.down"};
    for (const auto s : suffixes) {
        if (name.size() > s.size() + 6u && name.starts_with("layer") &&
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
// per-batch-row 2B dispatches into 1 and drops the _cls intermediate round-trip.
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
//     compaction window, _finalize, never reads the padded tail; it is
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
enum class EmbedP1Mode { legacy,
                         window,
                         pairs };
[[nodiscard]] EmbedP1Mode embed_p1_mode() noexcept {
    static const EmbedP1Mode mode = [] {
        const auto *raw = std::getenv("EMBED_P1_SPARSE_COMPACT");
        const luisa::string_view v{raw != nullptr ? raw : ""};
        if (v == "0") {
            return EmbedP1Mode::legacy;
        }
        if (v == "2") {
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

[[nodiscard]] SparseRowWindows sparse_row_windows(luisa::span<const float> mask, uint32_t B, uint32_t S) {
    SparseRowWindows w;
    w.row_real.resize(B);
    for (uint32_t b = 0; b < B; b++) {
        const auto *row = mask.data() + static_cast<size_t>(b) * S;
        uint32_t n = 0u;
        while (n < S && row[n] != 0.0f) {
            n++;
        }
        w.real_total += n;
        w.row_real[b] = n;
        for (uint32_t k = n; k < S; k++) {
            if (row[k] != 0.0f) {
                w.prefix_ok = false;
                break;
            }
        }
        if (!w.prefix_ok) [[unlikely]] {
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
    : _device{device}, _pkg_precision{package.precision()} {

    const auto L = package.num_layers();
    _hidden = package.hidden_size();
    _heads = package.num_heads();
    _head_dim = _hidden / _heads;
    _inter = package.intermediate_size();
    if (_hidden == 0u || _heads == 0u || _hidden % _heads != 0u || L == 0u) [[unlikely]] {
        LUISA_ERROR("embedding WeightStore: invalid package hyper-parameters (layers={}, H={}, heads={}).",
                    L, _hidden, _heads);
    }

    // --- weight-format toggle (opt plan §3.1 G7 / §4 Q1-Q2) ------------------
    // auto-detect from the package; LUISA_EMBED_WEIGHTS may force the fp32
    // A/B arm or re-interpret -- mismatches are refused loudly.
    const bool pkg_half = _pkg_precision != PackagePrecision::fp32;
    bool packed_half = pkg_half;// upload v2 blobs verbatim + convert-on-read
    if (const auto *env = std::getenv("LUISA_EMBED_WEIGHTS"); env != nullptr && env[0] != '\0') {
        luisa::string req;
        req.reserve(6u);
        for (const char *c = env; *c != '\0'; c++) {
            req.push_back((*c >= 'A' && *c <= 'Z') ? static_cast<char>(*c + 32) : *c);
        }
        PackagePrecision want;
        if (req == "fp32") {
            want = PackagePrecision::fp32;
        } else if (req == "bf16") {
            want = PackagePrecision::bf16;
        } else if (req == "fp16") {
            want = PackagePrecision::fp16;
        } else [[unlikely]] {
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" is not fp32|bf16|fp16 (fail-closed).", env);
        }
        if (!pkg_half && want != PackagePrecision::fp32) [[unlikely]] {
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" asks for the {} convert-on-read "
                        "kernel path, but the package is fp32 (v1) -- its blobs are not packed-half buffers; "
                        "pack a v2 --precision {} package to use it.",
                        env, req, req);
        }
        if (pkg_half && want != PackagePrecision::fp32 && want != _pkg_precision) [[unlikely]] {
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" disagrees with the package "
                        "format '{}' (bf16 and fp16 bit patterns must not be reinterpreted across).",
                        env, package_precision_name(_pkg_precision));
        }
        packed_half = pkg_half && want != PackagePrecision::fp32;
        LUISA_INFO("embedding WeightStore: package precision '{}', LUISA_EMBED_WEIGHTS='{}' -> device mode '{}'.",
                   package_precision_name(_pkg_precision), env,
                   pkg_half ? (packed_half ? "packed-half convert-on-read" : "fp32 convert-on-upload (A/B)") : "fp32");
    }
    if (pkg_half && packed_half) {
        LUISA_INFO("embedding WeightStore: half-precision package '{}' -> GEMM weights uploaded packed "
                   "({} B/elt), converted at the kernel B-tile load; fp32 accumulation unchanged.",
                   package_precision_name(_pkg_precision), 2);
    } else if (pkg_half) {
        LUISA_INFO("embedding WeightStore: half-precision package '{}' in fp32 A/B mode -- every tensor "
                   "up-converted at upload, fp32 kernel path.",
                   package_precision_name(_pkg_precision));
    }
    _gemm_format = (pkg_half && packed_half) ? (_pkg_precision == PackagePrecision::bf16 ? GemmWeightFormat::Bf16 : GemmWeightFormat::Fp16) : GemmWeightFormat::Fp32;

    // Role-aware tensor acquisition. allow_packed tensors (the four GEMM weight
    // matrices per layer) keep the packed-half device layout in the
    // convert-on-read mode; everything else -- and every tensor in the fp32
    // A/B mode -- is up-converted host-side at construction time. The
    // conversion results live in _owned_scratch for the store's lifetime, so
    // the upload spans borrow from storage owned by this WeightStore (the same
    // "outlives the recorded commands" contract as the Package image).
    const auto load_weight = [&](const luisa::string &name, Buffer<float> &dst, bool allow_packed) {
        const auto &t = require_tensor(package, name);
        const auto n = static_cast<size_t>(t.element_count());
        if (_pkg_precision == PackagePrecision::fp32) {
            dst = _device.create_buffer<float>(n);
            _uploads.push_back(UploadItem{&dst, t.data, n});
            return;
        }
        if (allow_packed && packed_half) {
            const auto words = (n + 1u) / 2u;// two LE halves per device float
            dst = _device.create_buffer<float>(words);
            _owned_scratch.emplace_back(words, 0.0f);
            auto &scratch = _owned_scratch.back();
            std::memcpy(scratch.data(), t.half_data, n * 2u);// exact blob bytes;
            // zero-init covers the 2-byte tail of an odd n (no blob overrun).
            _uploads.push_back(UploadItem{&dst, scratch.data(), words});
            return;
        }
        dst = _device.create_buffer<float>(n);
        _owned_scratch.emplace_back(n);
        auto &scratch = _owned_scratch.back();
        for (size_t i = 0u; i < n; i++) {
            scratch[i] = half_up(t.half_data[i], _pkg_precision);
        }
        _uploads.push_back(UploadItem{&dst, scratch.data(), n});
    };

    load_weight("emb.word", _word, false);
    load_weight("emb.pos", _pos, false);
    load_weight("emb.ln.gamma", _emb_ln_gamma, false);
    load_weight("emb.ln.beta", _emb_ln_beta, false);

    _layers.reserve(L);
    for (uint32_t i = 0; i < L; i++) {
        _layers.emplace_back();
        auto &lw = _layers.back();// fill IN PLACE: UploadItem stores pointers to
        // these member buffers, so they must never move (_layers is reserved
        // above and append-only afterwards).
        load_weight(luisa::format("layer{}.qkv", i), lw.qkv, true);
        load_weight(luisa::format("layer{}.qkv.bias", i), lw.qkv_bias, false);
        load_weight(luisa::format("layer{}.attn.out", i), lw.attn_out, true);
        load_weight(luisa::format("layer{}.attn.out.bias", i), lw.attn_out_bias, false);
        load_weight(luisa::format("layer{}.ln1.gamma", i), lw.ln1_gamma, false);
        load_weight(luisa::format("layer{}.ln1.beta", i), lw.ln1_beta, false);
        load_weight(luisa::format("layer{}.ffn.up", i), lw.ffn_up, true);
        load_weight(luisa::format("layer{}.ffn.up.bias", i), lw.ffn_up_bias, false);
        load_weight(luisa::format("layer{}.ffn.down", i), lw.ffn_down, true);
        load_weight(luisa::format("layer{}.ffn.down.bias", i), lw.ffn_down_bias, false);
        load_weight(luisa::format("layer{}.ln2.gamma", i), lw.ln2_gamma, false);
        load_weight(luisa::format("layer{}.ln2.beta", i), lw.ln2_beta, false);
    }

    load_weight("head.sparse.w", _sparse_w, false);// [H,1] == plain [H] dot vector
    {
        const auto &sb = require_tensor(package, "head.sparse.b");
        _sparse_b = _pkg_precision == PackagePrecision::fp32 ? sb.data[0] : half_up(sb.half_data[0], _pkg_precision);
    }
    load_weight("head.colbert.w", _colbert_w, false);
    load_weight("head.colbert.b", _colbert_b, false);
}

void WeightStore::record_upload(CommandList &cl) const {
    for (const auto &u : _uploads) {
        cl << u.buffer->view().copy_from(luisa::span{u.src, u.count});
    }
}

Executor::Executor(Device &device, const WeightStore &weights, compute::Stream &owner_stream) noexcept
    : _device{device}, _owner_stream{owner_stream}, _weights{weights} {

    const auto compile = [&](auto &&kernel, const char *name) {
        ShaderOption opt;
        opt.name = name;
        return _device.compile(std::forward<decltype(kernel)>(kernel), opt);
    };

    // GEMM kernels are compiled in the B-operand format the WeightStore
    // uploads (opt plan §3.1 G7): Fp32 -> the historical bit-identical path;
    // Bf16/Fp16 -> the convert-on-read packed-B load (fp32 accumulation).
    const auto gemm_wf = _weights.gemm_weight_format();
    const luisa::string wf_sfx = gemm_wf == GemmWeightFormat::Fp32 ? luisa::string{} : luisa::format("_{}", gemm_weight_format_name(gemm_wf));
    const auto gname = [&wf_sfx](luisa::string_view base) {
        return luisa::string{base} + wf_sfx;
    };

    _embed = compile(make_embed_lookup_kernel(), "emb_lookup");
    _gemm_bias = compile(make_gemm_kernel(GemmEpilogue::Bias, gemm_wf), gname("gemm_bias").c_str());
    _gemm_gelu = compile(make_gemm_kernel(GemmEpilogue::BiasGeluErf, gemm_wf), gname("gemm_bias_gelu_erf").c_str());
    _residual = compile(make_residual_add_kernel(), "residual_add");
    _layernorm = compile(make_layernorm_kernel(), "layernorm");
    _l2norm = compile(make_l2_normalize_kernel(), "l2_normalize");
    _slice_cls = compile(make_slice_cls_kernel(), "slice_cls");
    _dense_pool = compile(make_dense_pool_kernel(), "dense_pool");
    _sparse_head = compile(make_sparse_head_kernel(), "sparse_head");
    _sparse_head_compact = compile(make_sparse_head_compact_kernel(), "sparse_head_compact");
    _sparse_count_reset = compile(make_sparse_count_reset_kernel(), "sparse_count_reset");
    _colbert_proj = compile(make_colbert_proj_kernel(), "colbert_proj");
    _attn_scores = compile(make_attention_scores_kernel(), "attn_scores");
    _attn_softmax = compile(make_softmax_row_kernel(), "softmax_row");
    _attn_pv = compile(make_attention_pv_kernel(), "attn_pv");
    // _flash stays unset: lazily compiled on the first S > 1024 encode (M4).
}

void Executor::_ensure_pool(uint32_t rows, uint32_t batch, uint32_t seq) {

    const uint32_t H = _weights.hidden_size();
    const uint32_t I = _weights.intermediate_size();

    if (rows > _pool_rows) {
        // Grow-only reallocation (rare, amortized): flush the owning worker
        // stream before the replaced buffers are destroyed, so the vk
        // validation layer retires its resource-usage bookkeeping
        // ("destroyed while stream is still using it" abort under
        // LUISA_ENABLE_VALIDATION). The scheduler only dispatches to an IDLE
        // worker, so this sync is a zero-wait bookkeeping flush, never a
        // completion stall; the non-growth fast path never synchronizes.
        _pool_rows = rows;
        const auto rf = static_cast<size_t>(rows);
        _ids_dev = _device.create_buffer<uint32_t>(rf);
        _mask_dev = _device.create_buffer<float>(rf);
        _h0 = _device.create_buffer<float>(rf * H);
        _h1 = _device.create_buffer<float>(rf * H);
        _qkv = _device.create_buffer<float>(rf * 3u * H);
        _att = _device.create_buffer<float>(rf * H);
        _tmp1 = _device.create_buffer<float>(rf * H);
        _tmp2 = _device.create_buffer<float>(rf * H);
        _ffn = _device.create_buffer<float>(rf * I);
        _cls = _device.create_buffer<float>(H);
        _dense_dev = _device.create_buffer<float>(rf * H);// B*H <= B*seq*H (seq >= 1)
        _sparse_dev = _device.create_buffer<float>(rf);
        _colbert_dev = _device.create_buffer<float>(rf * H);
        // P1 "pairs" mode staging (EMBED_P1_SPARSE_COMPACT=2); allocated with
        // the pool so a grow never sees a torn buffer. The counter is a fixed
        // [1] buffer: (re)created only while empty (a grow rebinds the others
        // in place).
        _sparse_pair_ids = _device.create_buffer<uint32_t>(rf);
        _sparse_pair_w = _device.create_buffer<float>(rf);
        if (!_sparse_count) {
            _sparse_count = _device.create_buffer<uint32_t>(1u);
        }
    }

    if (seq <= kNaiveAttentionMaxSeq) {
        // logits workspace [B*heads, S, S] fp32 (plan §5: <= 64 MiB at B=1,S=1024)
        const auto need = static_cast<size_t>(batch) * _weights.num_heads() * seq * seq;
        if (need > _logits_floats) {
            _owner_stream.synchronize();// retire-usage flush (see above)
            _logits_floats = static_cast<uint32_t>(need);
            _logits = _device.create_buffer<float>(need);
        }
    } else {
        // flash scratch: DUMMY per the B2 contract (1 float, never touched)
        if (!_flash_scratch) {
            const auto need = static_cast<size_t>(flash_attention_scratch_floats(batch, seq, _weights.num_heads()));
            _flash_scratch = _device.create_buffer<float>(need);
        }
    }
}

void Executor::_ensure_trace_slots(uint32_t floats_per_slot) {
    if (floats_per_slot <= _trace_floats) {
        return;
    }
    _owner_stream.synchronize();// same retire-usage note as _ensure_pool
    _trace_floats = floats_per_slot;
    _owner_stream.synchronize();// retire-usage flush before slot replacement
    for (uint32_t k = 0; k < kTraceStates; k++) {
        _trace_slots[k] = _device.create_buffer<float>(floats_per_slot);
        _trace_slots[k].set_name(luisa::format("embed_trace_slot_{}", k));
    }
}

void Executor::record_encode(CommandList &cl, const EncodeJob &job) {

    const auto B = job.batch;
    const auto S = job.seq;
    const auto H = _weights.hidden_size();
    const auto heads = _weights.num_heads();
    const auto hd = _weights.head_dim();
    const auto I = _weights.intermediate_size();
    const auto L = _weights.num_layers();
    const uint32_t rows = B * S;
    const auto rf = static_cast<size_t>(rows);

    if (S == 0u || B == 0u || S + kEmbedPosOffset > 8194u) [[unlikely]] {
        LUISA_ERROR("executor: invalid job shape B={} S={} (position table ceiling is 8194).", B, S);
    }
    if (job.ids.size() != rf || job.mask.size() != rf || job.dense_out.size() != static_cast<size_t>(B) * H) [[unlikely]] {
        LUISA_ERROR("executor: job staging sizes do not match B*S (ids={}, mask={}, dense={}).",
                    job.ids.size(), job.mask.size(), job.dense_out.size());
    }

    _ensure_pool(rows, B, S);

    const bool tracing = static_cast<bool>(_trace_cb);
    if (tracing) {
        _ensure_trace_slots(static_cast<uint32_t>(rf * H));
    }
    _traced_states = 0u;
    _trace_rows = rows;
    _trace_cols = H;

    const auto trace = [&](uint32_t layer, BufferView<float> src) {
        if (tracing) {
            const auto n = rf * H;
            cl << _trace_slots[layer].view(0, n).copy_from(src.subview(0, n));
            _traced_states = layer + 1u;
        }
    };

    //--- staging upload -------------------------------------------------------
    cl << _ids_dev.view(0, rf).copy_from(job.ids)
       << _mask_dev.view(0, rf).copy_from(job.mask);

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
    const auto pos_rows = _weights.pos_emb().size();// full table capacity
    cl << _embed(_weights.word_emb().view(),
                 _weights.pos_emb().view(static_cast<size_t>(kEmbedPosOffset) * H,
                                         (pos_rows / H - kEmbedPosOffset) * H),
                 _ids_dev.view(0, rf),
                 _weights.emb_ln_gamma().view(), _weights.emb_ln_beta().view(),
                 _h0.view(0, rf * H),
                 S, H, kEmbedLayerNormEps, 0u)
              .dispatch(static_cast<uint>(rf * kEmbedBlock));
    trace(0u, _h0.view(0, rf * H));

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const auto scores_disp = attention_scores_dispatch(B, S, heads);
    const auto pv_disp = attention_pv_dispatch(B, S, heads, hd);

    //--- 24 encoder layers ----------------------------------------------------
    for (uint32_t i = 0; i < L; i++) {
        const auto &lw = _weights.layer(i);

        // qkv projection: [rows,H] @ [H,3H] + bias -> qkv
        cl << _gemm_bias(_h0.view(0, rf * H), lw.qkv.view(),
                         _qkv.view(0, rf * 3u * H), lw.qkv_bias.view(),
                         rows, 3u * H, H)
                  .dispatch(gemm_dispatch_size(rows, 3u * H));

        // attention (naive trio for S <= 1024, fused flash above)
        if (S <= kNaiveAttentionMaxSeq) {
            const auto lf = static_cast<size_t>(B) * heads * S * S;
            cl << _attn_scores(_qkv.view(0, rf * 3u * H), _mask_dev.view(0, rf),
                               _logits.view(0, lf),
                               S, heads, hd, scale, kAttentionNegLarge)
                      .dispatch(scores_disp);
            const uint32_t sm_rows = B * heads * S;
            const uint32_t sm_chunk = std::max(4u, (65535u / heads) & ~3u);// mult of 4
            for (uint32_t r0 = 0u; r0 < sm_rows; r0 += sm_chunk) {
                const uint32_t n = std::min(sm_chunk, sm_rows - r0);
                cl << _attn_softmax(_logits.view(static_cast<size_t>(r0) * S,
                                                 static_cast<size_t>(n) * S),
                                    S)
                          .dispatch(n * kAttentionBlock);
            }
            cl << _attn_pv(_qkv.view(0, rf * 3u * H), _logits.view(0, lf),
                           _att.view(0, rf * H), S, heads, hd)
                      .dispatch(pv_disp);
        } else {
            if (!_flash.has_value()) [[unlikely]] {
                ShaderOption opt;
                opt.name = "flash_attention";
                _flash.emplace(_device.compile(make_flash_attention_kernel(), opt));
            }
            const auto sf = static_cast<size_t>(flash_attention_scratch_floats(B, S, heads));
            cl << (*_flash)(_qkv.view(0, rf * 3u * H), _mask_dev.view(0, rf),
                            _att.view(0, rf * H), _flash_scratch.view(0, sf),
                            B, S, heads, hd, scale, kAttentionNegLarge)
                      .dispatch(flash_attention_dispatch(B, S, heads));
        }

        // out projection + residual + LN1
        cl << _gemm_bias(_att.view(0, rf * H), lw.attn_out.view(),
                         _tmp1.view(0, rf * H), lw.attn_out_bias.view(),
                         rows, H, H)
                  .dispatch(gemm_dispatch_size(rows, H))
           << _residual(_h0.view(0, rf * H), _tmp1.view(0, rf * H),
                        _tmp2.view(0, rf * H), rf * H)
                  .dispatch(static_cast<uint>(rf * H))
           << _layernorm(_tmp2.view(0, rf * H), _h1.view(0, rf * H),
                         lw.ln1_gamma.view(), lw.ln1_beta.view(), H, kEmbedLayerNormEps)
                  .dispatch(static_cast<uint>(rf * kNormBlock));

        // FFN (erf-GELU fused in the up-projection epilogue) + residual + LN2.
        // The result returns to _h0 in place: _h0's layer input is fully
        // consumed by the first residual_add before LN2 writes back.
        cl << _gemm_gelu(_h1.view(0, rf * H), lw.ffn_up.view(),
                         _ffn.view(0, rf * I), lw.ffn_up_bias.view(),
                         rows, I, H)
                  .dispatch(gemm_dispatch_size(rows, I))
           << _gemm_bias(_ffn.view(0, rf * I), lw.ffn_down.view(),
                         _tmp1.view(0, rf * H), lw.ffn_down_bias.view(),
                         rows, H, I)
                  .dispatch(gemm_dispatch_size(rows, H))
           << _residual(_h1.view(0, rf * H), _tmp1.view(0, rf * H),
                        _tmp2.view(0, rf * H), rf * H)
                  .dispatch(static_cast<uint>(rf * H))
           << _layernorm(_tmp2.view(0, rf * H), _h0.view(0, rf * H),
                         lw.ln2_gamma.view(), lw.ln2_beta.view(), H, kEmbedLayerNormEps)
                  .dispatch(static_cast<uint>(rf * kNormBlock));

        trace(i + 1u, _h0.view(0, rf * H));
    }

    //--- dense pooling: CLS row + L2 normalize (kernels #8, #9) ---------------
    // N1(b): default fused path -- one dense_pool dispatch reads every batch's CLS
    // row straight out of the [B*S, H] final hidden state (_h0) and writes the
    // L2-normalized [B, H] dense result into _dense_dev, folding the old per-row
    // slice_cls + l2_normalize pair (2B dispatches + an _cls round-trip) into one.
    // Bit-identical to the legacy loop below (scratch harness, 0/16384 differing
    // floats on dx+vk+fallback). EMBED_N1_FUSE_POOL=0 restores the legacy pair.
    if (embed_fuse_pool_enabled()) [[likely]] {
        cl << _dense_pool(_h0.view(0, rf * H),
                          _dense_dev.view(0, static_cast<size_t>(B) * H),
                          S, H)
                  .dispatch(static_cast<uint>(B * kNormBlock));
    } else {
        for (uint32_t b = 0; b < B; b++) {
            cl << _slice_cls(_h0.view(static_cast<size_t>(b) * S * H, H),
                             _cls.view(0, H), H)
                      .dispatch(H)
               << _l2norm(_cls.view(0, H), _dense_dev.view(static_cast<size_t>(b) * H, H), H)
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
    const auto sparse_windows = job.want_sparse ? sparse_row_windows(job.mask, B, S) : SparseRowWindows{};
    if (job.want_sparse) {
        if (job.sparse_out.size() != rf) [[unlikely]] {
            LUISA_ERROR("executor: want_sparse but sparse_out has {} floats (need {}).",
                        job.sparse_out.size(), rf);
        }
        if (p1_mode == EmbedP1Mode::pairs && sparse_windows.real_total > 0u) {
            cl << _sparse_count_reset(_sparse_count.view()).dispatch(32u)
               << _sparse_head_compact(_h0.view(0, rf * H), _weights.sparse_w().view(),
                                       _weights.sparse_b(), _mask_dev.view(0, rf),
                                       _sparse_pair_ids.view(0, rf), _sparse_pair_w.view(0, rf),
                                       _sparse_count.view(0, 1), H)
                      .dispatch(static_cast<uint>(rf * kHeadsBlock));
        } else {
            cl << _sparse_head(_h0.view(0, rf * H), _weights.sparse_w().view(),
                               _weights.sparse_b(), _sparse_dev.view(0, rf), H)
                      .dispatch(static_cast<uint>(rf * kHeadsBlock));
        }
    }
    if (job.want_colbert) {
        if (job.colbert_out.size() != rf * H) [[unlikely]] {
            LUISA_ERROR("executor: want_colbert but colbert_out has {} floats (need {}).",
                        job.colbert_out.size(), rf * H);
        }
        cl << _colbert_proj(_h0.view(0, rf * H), _weights.colbert_w().view(),
                            _weights.colbert_b().view(), _colbert_dev.view(0, rf * H),
                            H, H, 1u)
                  .dispatch(static_cast<uint>(rf * kHeadsBlock));
    }

    //--- readback --------------------------------------------------------------
    cl << _dense_dev.view(0, static_cast<size_t>(B) * H).copy_to(job.dense_out);
    if (job.want_sparse) {
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
        if (p1_mode == EmbedP1Mode::pairs && sparse_windows.real_total > 0u) {
            const auto cap = sparse_windows.real_total;
            auto staging = luisa::make_shared<SparsePairStaging>();
            staging->count.resize(1u);
            staging->ids.resize(cap);
            staging->weights.resize(cap);
            const auto dst = job.sparse_out;
            cl << _sparse_count.view(0, 1).copy_to(luisa::span<uint32_t>{staging->count})
               << _sparse_pair_ids.view(0, cap).copy_to(luisa::span<uint32_t>{staging->ids})
               << _sparse_pair_w.view(0, cap).copy_to(luisa::span<float>{staging->weights});
            cl.add_callback([staging, dst]() noexcept {
                std::memset(dst.data(), 0, dst.size() * sizeof(float));
                const uint32_t n = staging->count[0];
                const auto lim = std::min<size_t>(n, staging->ids.size());
                for (size_t i = 0; i < lim; i++) {
                    const auto t = staging->ids[i];
                    if (t < dst.size()) [[likely]] {
                        dst[t] = staging->weights[i];
                    }
                }
            });
        } else if (p1_mode == EmbedP1Mode::window && sparse_windows.prefix_ok) {
            for (uint32_t b = 0; b < B; b++) {
                const auto off = static_cast<size_t>(b) * S;
                const auto n = sparse_windows.row_real[b];
                if (n > 0u) {
                    cl << _sparse_dev.view(off, n)
                              .copy_to(luisa::span<float>{out + off, n});
                }
                if (n < S) {
                    std::memset(out + off + n, 0, static_cast<size_t>(S - n) * sizeof(float));
                }
            }
        } else {
            if (p1_mode == EmbedP1Mode::window) [[unlikely]] {
                static std::atomic_flag warned{};
                if (!warned.test_and_set()) {
                    LUISA_WARNING("embedding P1 window: non-suffix padding mask at B={} S={}; "
                                  "using the legacy whole-array sparse readback for this encode.",
                                  B, S);
                }
            }
            cl << _sparse_dev.view(0, rf).copy_to(job.sparse_out);
        }
    }
    if (job.want_colbert) {
        cl << _colbert_dev.view(0, rf * H).copy_to(job.colbert_out);
    }
}

void Executor::fire_trace_callbacks() noexcept {
    if (!_trace_cb) {
        return;
    }
    const auto n = static_cast<size_t>(_trace_rows) * _trace_cols;
    for (uint32_t k = 0; k < _traced_states; k++) {
        _trace_cb(k, _trace_slots[k].view(0, n), _trace_rows, _trace_cols);
    }
    _traced_states = 0u;
}

}// namespace luisa::embed
