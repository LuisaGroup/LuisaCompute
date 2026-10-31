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
#include "kernels/kernels_coop.h"
#include "kernels/kernels_embed.h"
#include "kernels/kernels_embed_vec.h"
#include "kernels/kernels_gemm.h"
#include "kernels/kernels_heads.h"
#include "kernels/kernels_heads_vec.h"
#include "kernels/kernels_norm.h"
#include "kernels/kernels_norm_vec.h"
#include "quantization.h"
#include "tensor_path.h"

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

// Fail-closed access to a v3 tensor's quantized payload. The loader already
// validated the table entry (package.cpp: finite, > 0), but the upload path
// re-checks so a consumer can never dequantize with an invalid scale or read a
// null blob (package.h "exactly one typed view is non-null" contract).
void require_int8_payload(const Tensor &t, luisa::string_view name) {
    if (!int8_scale_is_valid(t.int8_scale)) [[unlikely]] {
        LUISA_ERROR("embedding WeightStore: int8 tensor '{}' carries the invalid per-tensor scale {} "
                    "(must be finite and > 0; fail-closed).",
                    name, t.int8_scale);
    }
    if (t.int8_data == nullptr) [[unlikely]] {
        LUISA_ERROR("embedding WeightStore: int8 tensor '{}' has no int8 blob (loader contract violation).",
                    name);
    }
}

// The GEMM weight matrices of one layer (the tensors uploaded packed in the
// convert/dequantize-on-read mode: packed half for a v2 package, packed int8
// words for a v3 package; every other tensor goes through the host
// upconverter/dequantizer -- see the WeightStore class comment in executor.h).
constexpr bool is_gemm_weight(luisa::string_view name) noexcept {
    // "layer{i}.qkv" / ".attn.out" / ".ffn.up" / ".ffn.down" -- NOT the
    // ".qkv.bias"/".ffn.up.bias" siblings, and not head.* (their kernels are
    // not the GEMM family).
    constexpr luisa::string_view suffixes[] = {".qkv", ".attn.out", ".ffn.up", ".ffn.down"};
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

// Opt plan C3: fused residual_add + layernorm. Default ON; EMBED_FUSE_RESIDUAL_LN=0
// restores the legacy `residual_add` -> `layernorm` pair's dispatch sequence
// unchanged (kill switch for perf A/B and the toggle-regression gate). Pure
// dispatch fusion on the value side: the fused kernel caches the fp32 residual
// row in shared memory and reuses make_layernorm_kernel()'s exact
// reduction/expression order, so the output is NUMERICALLY EQUIVALENT to the
// pair within a focused 1e-5 ABSOLUTE tolerance and never a numerical change --
// but NOT bit-identical (dx re-emits ~1 ULP against the two-kernel pair; see
// kernels_norm.h).
[[nodiscard]] bool embed_fuse_residual_ln_enabled() noexcept {
    static const bool on = [] {
        const auto *raw = std::getenv("EMBED_FUSE_RESIDUAL_LN");
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

// Tensor-path (tensor_optimize.md 2.1/5.4): the coop matvec variant, selected
// by an exact-name env override LUISA_EMBED_COOP_CONFIG -- mirroring the
// whole-variant arm of LUISA_EMBED_GEMM_CONFIG (kernels_gemm.cpp:777-787: an
// unknown value warns once and falls back to the shipped default, never a
// hard error at startup). Resolved once per process (thread-safe magic-static,
// same idiom as embed_p1_mode()).
[[nodiscard]] const CoopGemmConfig *embed_coop_variant() noexcept {
    static const CoopGemmConfig *cfg = [] {
        if (const auto *name = std::getenv("LUISA_EMBED_COOP_CONFIG"); name != nullptr && name[0] != '\0') {
            if (const auto *c = coop_gemm_variant_by_name(name); c != nullptr) {
                LUISA_INFO("embed coop: variant '{}' selected by LUISA_EMBED_COOP_CONFIG "
                           "(kslab {}, ntile {}, tokens_per_block {}).",
                           c->name, c->kslab, c->ntile, c->tokens_per_block);
                return c;
            }
            LUISA_WARNING("embed coop: LUISA_EMBED_COOP_CONFIG=\"{}\" is not a known variant; "
                          "falling back to the shipped default '{}'.",
                          name, coop_gemm_variant(0u)->name);
        }
        const auto *c = coop_gemm_variant(0u);
        LUISA_INFO("embed coop: default variant '{}' (kslab {}, ntile {}, tokens_per_block {}).",
                   c->name, c->kslab, c->ntile, c->tokens_per_block);
        return c;
    }();
    return cfg;
}

// vec4 / coop host-binding helpers (tensor_optimize.md 1.1 / 5.2: zero-copy
// reinterpretation of Buffer<float> as float4 / raw bytes -- no device copy).

// float4 view of a Buffer<float> or BufferView<float> starting at element 0.
// Offset 0 keeps every element-0 address at the buffer base (16 B aligned),
// which is what the vec4 kernels' host contract requires.
[[nodiscard]] auto v4_at0(const compute::Buffer<float> &b) noexcept {
    return b.view().template as<float4>();
}
[[nodiscard]] auto v4_at0(compute::BufferView<float> v) noexcept {
    return v.as<float4>();
}
// float4 view of `b`'s [off, off+n) with n % 4 == 0 -- off == 0 OR 16 B
// aligned -- via subview-then-as (byte-preserving; the vec4 view inherits the
// subview byte offset). The element-offset-alignment rule of
// kernels_norm_vec.h:10-16.
[[nodiscard]] auto v4_win(const compute::Buffer<float> &b, size_t off, size_t n) noexcept {
    return b.view(off, n).template as<float4>();
}
// Raw byte view for the coop kernel args (coop_api.md 4: ByteBufferView over a
// typed Buffer<float> is a zero-copy alias).
[[nodiscard]] auto byte_at0(const compute::Buffer<float> &b) noexcept {
    return compute::ByteBufferView{b};
}
[[nodiscard]] auto byte_at0(compute::BufferView<float> v) noexcept {
    return compute::ByteBufferView{v};
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

    // --- weight-format toggle (opt plan §3.1 G7 / §4 Q1-Q2; INT8 quantization
    // phase 3) ----------------------------------------------------------------
    // auto-detect from the package; LUISA_EMBED_WEIGHTS may force the fp32
    // A/B arm or re-interpret -- mismatches are refused loudly. `pkg_half`
    // means bf16/fp16 ONLY: a v3 int8 package is a SEPARATE format with its own
    // upload layout (see the WeightStore class comment in executor.h).
    const bool pkg_half = _pkg_precision == PackagePrecision::bf16 || _pkg_precision == PackagePrecision::fp16;
    const bool pkg_int8 = _pkg_precision == PackagePrecision::int8;
    bool packed_half = pkg_half;// upload v2 blobs verbatim + convert-on-read
    bool packed_int8 = pkg_int8;// upload v3 [scale|codes] words + dequant-on-read
    // Phase 1b: a v1 fp32 package + bf16/fp16 has no packed blob to upload, so
    // its four per-layer GEMM matrices are CONVERTED ON THE DEVICE once at
    // construction (fp32 staging -> make_pack_half_kernel -> packed words). That
    // is a third mode of its own -- neither `packed_half` (nothing packed is
    // uploaded) nor the fp32 path (the GEMM B operand ends up packed) -- and it
    // leaves every non-GEMM tensor fp32.
    bool fp32_pack_half = false;
    GemmWeightFormat fp32_pack_wf = GemmWeightFormat::Fp32;
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
        } else if (req == "int8") {
            want = PackagePrecision::int8;
        } else [[unlikely]] {
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" is not fp32|bf16|fp16|int8 (fail-closed).", env);
        }
        if (!pkg_half && !pkg_int8 && want == PackagePrecision::int8) [[unlikely]] {
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" asks for the int8 (v3) device path, "
                        "but the package is fp32 (v1) -- fp32 -> int8 needs a per-tensor calibrated scale that "
                        "only the packer computes, so the device never re-quantizes weights (fail-closed); "
                        "pack a v3 --precision int8 package to use it.",
                        env);
        }
        if (!pkg_half && !pkg_int8) {
            // v1 fp32 + bf16/fp16 (phase 1b): supported, but only through the
            // one-shot device conversion below -- the fp32 blob is never
            // reinterpreted as packed words.
            fp32_pack_half = want != PackagePrecision::fp32;
            fp32_pack_wf = want == PackagePrecision::bf16 ? GemmWeightFormat::Bf16 : GemmWeightFormat::Fp16;
        }
        if (pkg_half && want != PackagePrecision::fp32 && want != _pkg_precision) [[unlikely]] {
            if (want == PackagePrecision::int8) {
                LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" asks for the quantized (v3 int8) "
                            "device path, but the package is half-precision '{}' (v2) -- its blobs are "
                            "16-bit elements, not signed int8 codes with a per-tensor scale; pack a v3 "
                            "--precision int8 package to use it.",
                            env, package_precision_name(_pkg_precision));
            }
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" disagrees with the package "
                        "format '{}' (bf16 and fp16 bit patterns must not be reinterpreted across).",
                        env, package_precision_name(_pkg_precision));
        }
        if (pkg_int8 && (want == PackagePrecision::bf16 || want == PackagePrecision::fp16)) [[unlikely]] {
            LUISA_ERROR("embedding WeightStore: LUISA_EMBED_WEIGHTS=\"{}\" asks for the {} device path, but the "
                        "package is int8 (v3) -- its blobs are signed int8 codes plus a per-tensor scale, "
                        "which must never be reinterpreted as 16-bit elements; pack a v2 --precision {} "
                        "package to use it.",
                        env, req, req);
        }
        packed_half = pkg_half && want != PackagePrecision::fp32;
        packed_int8 = pkg_int8 && want != PackagePrecision::fp32;
        LUISA_INFO("embedding WeightStore: package precision '{}', LUISA_EMBED_WEIGHTS='{}' -> device mode '{}'.",
                   package_precision_name(_pkg_precision), env,
                   pkg_int8 ? (packed_int8 ? "packed-int8 dequantize-on-read" : "fp32 dequantize-on-upload (A/B)")
                   : pkg_half ? (packed_half ? "packed-half convert-on-read" : "fp32 convert-on-upload (A/B)")
                   : fp32_pack_half ? "device fp32 -> packed-half conversion (v1 package, one-shot staging)"
                                    : "fp32");
    }
    if (pkg_int8 && packed_int8) {
        LUISA_INFO("embedding WeightStore: int8 package '{}' -> the four per-layer GEMM matrices uploaded as "
                   "v3 [scale | four signed codes per word] blobs ({} B/elt), dequantized as q * scale at "
                   "the kernel B-tile load; every other tensor dequantized host-side once. fp32 "
                   "accumulation unchanged (no int8 arithmetic on the device).",
                   package_precision_name(_pkg_precision), 1);
    } else if (pkg_int8) {
        LUISA_INFO("embedding WeightStore: int8 package '{}' in fp32 A/B mode -- every tensor dequantized "
                   "host-side once at upload (q * scale), fp32 kernel path.",
                   package_precision_name(_pkg_precision));
    } else if (pkg_half && packed_half) {
        LUISA_INFO("embedding WeightStore: half-precision package '{}' -> GEMM weights uploaded packed "
                   "({} B/elt), converted at the kernel B-tile load; fp32 accumulation unchanged.",
                   package_precision_name(_pkg_precision), 2);
    } else if (pkg_half) {
        LUISA_INFO("embedding WeightStore: half-precision package '{}' in fp32 A/B mode -- every tensor "
                   "up-converted at upload, fp32 kernel path.",
                   package_precision_name(_pkg_precision));
    } else if (fp32_pack_half) {
        LUISA_INFO("embedding WeightStore: fp32 (v1) package -> the four per-layer GEMM matrices are packed "
                   "ONCE on the device (fp32 staging -> make_pack_half_kernel -> the exact v2 packed-word "
                   "layout, bit-identical to the packer) and then read by the packed-half GEMM path. Every "
                   "other tensor stays fp32, the fp32 staging copies are released as soon as the conversion "
                   "retires (they are not kept for the session), and fp32 accumulation is unchanged.");
    }
    if (packed_int8) {
        _gemm_format = GemmWeightFormat::Int8;
    } else if (pkg_half && packed_half) {
        _gemm_format = _pkg_precision == PackagePrecision::bf16 ? GemmWeightFormat::Bf16 : GemmWeightFormat::Fp16;
    } else if (fp32_pack_half) {
        _gemm_format = fp32_pack_wf;
    } else {
        _gemm_format = GemmWeightFormat::Fp32;
    }
    if (fp32_pack_half) {
        LUISA_INFO("embedding WeightStore: GEMM B operand stored as '{}' (device-packed from the v1 fp32 "
                   "package).",
                   gemm_weight_format_name(_gemm_format));
    }

    // Phase 1b: the ONLY mode that needs an fp32 -> packed-half converter is the
    // v1 fp32 package converted on the device -- a v2 package already carries
    // packed words, a v3 package is int8, and the fp32 arms read plain fp32 B.
    // The shader is therefore compiled here and only here; the store owns it for
    // its lifetime because the pack dispatches are recorded from it
    // (record_upload is const and may run long after construction).
    if (fp32_pack_half) {
        const luisa::string pack_name = luisa::format("pack_half_{}", gemm_weight_format_name(_gemm_format));
        ShaderOption pack_opt;
        pack_opt.name = pack_name.c_str();
        _pack_half = _device.compile(make_pack_half_kernel(_gemm_format), pack_opt);
    }

    // Reserve the conversion bookkeeping BEFORE the first Buffer pointer is
    // taken: UploadItem/PackItem store raw pointers to elements of these vectors
    // (`_conversion_sources`) and to the Layer members, so a later reallocation
    // would leave the recorded uploads/pack dispatches pointing at freed
    // storage. Four converted GEMM matrices per layer, nothing on every other
    // path.
    if (fp32_pack_half) {
        _packs.reserve(static_cast<size_t>(L) * 4u);
        _conversion_sources.reserve(static_cast<size_t>(L) * 4u);
    }

    // Role-aware tensor acquisition. allow_packed tensors (the four GEMM weight
    // matrices per layer) keep the packed device layout in the packed modes
    // (packed half for a v2 package -- or for a v1 fp32 package converted on the
    // device, phase 1b --, packed int8 words for a v3 package); everything else
    // -- and every tensor in the fp32 A/B mode -- is up-converted (half) or
    // dequantized (int8, q * scale via quantization.h) host-side at construction
    // time. The conversion results live in _owned_scratch for the store's
    // lifetime, so the upload spans borrow from storage owned by this
    // WeightStore (the same "outlives the recorded commands" contract as the
    // Package image).
    const auto load_weight = [&](const luisa::string &name, Buffer<float> &dst, bool allow_packed) {
        const auto &t = require_tensor(package, name);
        const auto n = static_cast<size_t>(t.element_count());
        if (_pkg_precision == PackagePrecision::fp32) {
            if (allow_packed && fp32_pack_half) {
                // Phase 1b: v1 fp32 source -> device-packed half B operand.
                // `dst` plays exactly the role it has in the v2 packed arm (the
                // packed word buffer an Executor binds), so the steady state and
                // the compiled GEMM kernels are the already-reviewed packed-half
                // ones; only the PRODUCTION of the words differs (a device pack
                // kernel here, a raw blob memcpy there). The fp32 copy in
                // `_conversion_sources` is transient: it is the pack kernel's
                // source and is handed to the Scheduler, which frees it as soon
                // as the conversion retires.
                const auto words = (n + 1u) / 2u;
                dst = _device.create_buffer<float>(words);
                dst.set_name(luisa::format("embed_packed_{}", name));
                // The staging element address must stay stable for the pointers
                // below, so the reservation above is a hard requirement, not an
                // optimization: fail loudly instead of recording a dangling
                // upload/pack pair.
                if (_conversion_sources.size() == _conversion_sources.capacity()) [[unlikely]] {
                    LUISA_ERROR("embedding WeightStore: conversion staging reservation exhausted ({} buffers); "
                                "the fp32 -> packed-half staging must be reserved before any Buffer pointer "
                                "is taken.",
                                _conversion_sources.size());
                }
                _conversion_sources.emplace_back(_device.create_buffer<float>(n));
                auto &staging = _conversion_sources.back();// reserved above -> stable
                staging.set_name(luisa::format("embed_pack_src_{}", name));
                _uploads.push_back(UploadItem{&staging, t.data, n});
                _packs.push_back(PackItem{&staging, &dst, n});
                _conversion_bytes += n * sizeof(float);
                return;
            }
            dst = _device.create_buffer<float>(n);
            _uploads.push_back(UploadItem{&dst, t.data, n});
            return;
        }
        if (allow_packed && packed_int8) {
            // v3 device layout (kernels_gemm.h INT8 contract): word 0 = the
            // positive per-tensor scale, words 1.. = the logical row-major
            // [K,N] codes, four signed int8 per word, little-endian. One raw
            // memcpy of the blob behind the scale word; the zero-initialized
            // scratch keeps the 1..3 tail bytes of the last word zero (the
            // kernel never reads them, but the buffer is fully initialized).
            require_int8_payload(t, name);
            const auto words = 1u + (n + 3u) / 4u;
            dst = _device.create_buffer<float>(words);
            _owned_scratch.emplace_back(words, 0.0f);
            auto &scratch = _owned_scratch.back();
            scratch[0] = t.int8_scale;
            std::memcpy(scratch.data() + 1, t.int8_data, n);
            _uploads.push_back(UploadItem{&dst, scratch.data(), words});
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
        if (_pkg_precision == PackagePrecision::int8) {
            // v3 host dequantization (fp32 A/B arm, and every non-GEMM tensor
            // in the packed arm): q * scale, once, through quantization.h --
            // which also re-validates the stored scale (fail-closed).
            require_int8_payload(t, name);
            int8_dequantize_tensor(luisa::span<const int8_t>{t.int8_data, n},
                                   luisa::span<float>{scratch.data(), n},
                                   t.int8_scale);
        } else {
            for (size_t i = 0u; i < n; i++) {
                scratch[i] = half_up(t.half_data[i], _pkg_precision);
            }
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
        // Scalar bias: never a device buffer, but it follows the same rule as
        // the head weight above (host-side dequantize/upcast; int8 via
        // quantization.h int8_dequantize == q * scale).
        const auto &sb = require_tensor(package, "head.sparse.b");
        if (_pkg_precision == PackagePrecision::fp32) {
            _sparse_b = sb.data[0];
        } else if (_pkg_precision == PackagePrecision::int8) {
            require_int8_payload(sb, "head.sparse.b");
            _sparse_b = int8_dequantize(sb.int8_data[0], sb.int8_scale);
        } else {
            _sparse_b = half_up(sb.half_data[0], _pkg_precision);
        }
    }
    load_weight("head.colbert.w", _colbert_w, false);
    load_weight("head.colbert.b", _colbert_b, false);
}

void WeightStore::record_upload(CommandList &cl) const {
    for (const auto &u : _uploads) {
        cl << u.buffer->view().copy_from(luisa::span{u.src, u.count});
    }
    if (_packs.empty()) {
        return;// no conversion on this path: nothing but the uploads above
    }
    // Phase 1b: pack the fp32 staging copies just queued into the persistent
    // packed-half destinations, ON THE DEVICE, in the same command list. One
    // dispatch per converted GEMM matrix, so the init stream's weights_ready
    // fence also covers the conversion -- no worker can observe a half-written
    // B operand, and no host-side packing (nor a second full-size fp32 set)
    // is ever needed. Cost and commands: zero on every other path.
    if (!_pack_half.has_value()) [[unlikely]] {
        LUISA_ERROR("embedding WeightStore: {} recorded pack dispatch(es) but owns no fp32 -> packed-half "
                    "shader (internal contract violation).",
                    _packs.size());
    }
    const auto &pack = *_pack_half;
    for (const auto &p : _packs) {
        const auto count = static_cast<uint32_t>(p.count);
        // dispatch() counts TOTAL threads: pack_half_dispatch_size rounds the
        // ceil(count / 2) word-threads up to whole blocks (kernels_gemm.h).
        cl << pack(p.src->view(), p.dst->view(), count)
                  .dispatch(pack_half_dispatch_size(count));
    }
}

WeightStore::ConversionStaging WeightStore::take_conversion_sources() noexcept {
    ConversionStaging out;
    out.bytes = _conversion_bytes;
    out.sources = std::move(_conversion_sources);
    // Moved-from: keep the member empty and reusable, and report 0 bytes from
    // here on (the handoff is one-shot; a second take returns nothing).
    _conversion_sources.clear();
    _conversion_bytes = 0u;
    // `_packs` still holds the staging addresses it was recorded with; that is
    // fine by contract -- record_upload() has already emitted the dispatches, and
    // the handoff is documented as happening after it.
    return out;
}

Executor::Executor(Device &device, const WeightStore &weights, compute::Stream &owner_stream) noexcept
    : _device{device}, _owner_stream{owner_stream}, _weights{weights} {

    const auto compile = [&](auto &&kernel, const char *name) {
        ShaderOption opt;
        opt.name = name;
        return _device.compile(std::forward<decltype(kernel)>(kernel), opt);
    };

    // GEMM kernels are compiled in the B-operand format the WeightStore
    // uploads (opt plan §3.1 G7; INT8 quantization phase 3): Fp32 -> the
    // historical bit-identical path; Bf16/Fp16 -> the convert-on-read packed-B
    // load; Int8 -> the v3 [scale | codes] dequantize-on-read load. Both packed
    // arms accumulate in fp32, so the only difference between the compiled
    // shaders is the B-tile decode step, and the shader name carries the
    // format suffix ("gemm_bias_int8") for profiling.
    const auto gemm_wf = _weights.gemm_weight_format();
    const luisa::string wf_sfx = gemm_wf == GemmWeightFormat::Fp32 ? luisa::string{} : luisa::format("_{}", gemm_weight_format_name(gemm_wf));
    const auto gname = [&wf_sfx](luisa::string_view base) {
        return luisa::string{base} + wf_sfx;
    };

    _embed = compile(make_embed_lookup_kernel(), "emb_lookup");
    _gemm_bias = compile(make_gemm_kernel(GemmEpilogue::Bias, gemm_wf), gname("gemm_bias").c_str());
    _gemm_gelu = compile(make_gemm_kernel(GemmEpilogue::BiasGeluErf, gemm_wf), gname("gemm_bias_gelu_erf").c_str());

    //--- opt plan C3: residual + layernorm --------------------------------
    // The two (residual_add, layernorm) pairs per encoder layer collapse to one
    // fused dispatch each. Whether the fusion is legal is a property of the
    // WEIGHTS (the row length H is fixed per WeightStore), so the decision is
    // made here, once: exactly one of the two shader sets is compiled. A
    // recorder that branched the other way would either dispatch a shader it
    // never compiled or pay a compile it cannot use, so record_encode branches
    // on this very flag.
    _fuse_residual_ln = embed_fuse_residual_ln_enabled() &&
                        residual_layernorm_supported(_weights.hidden_size());
    if (_fuse_residual_ln) {
        _residual_layernorm = compile(make_residual_layernorm_kernel(), "residual_layernorm");
    } else {
        _residual = compile(make_residual_add_kernel(), "residual_add");
        _layernorm = compile(make_layernorm_kernel(), "layernorm");
    }
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

    //--- tensor-path selection (tensor_optimize.md 2.1/5.4) -------------------
    // Legacy (the default; LUISA_EMBED_TENSOR_PATH unset -> Legacy) compiles NONE
    // of the vector/coop shaders below. Only a Vector/Coop
    // request pays the extra compiles. resolve_tensor_path() never returns Coop
    // when cooperative support is absent (it errors on an explicit coop
    // request, falls back to Vector on auto), so coop here already implies the
    // device passed the C0 probe.
    const auto req = tensor_path_requested();
    luisa::string path_reason;
    _tensor_path = resolve_tensor_path(_device, req);
    path_reason = tensor_path_name(_tensor_path);

    if (_tensor_path != TensorPath::Legacy) {
        // vec4 bandwidth/compute twins (DX/VK/CUDA eligible). Each is bound at
        // record time through a zero-copy BufferView<float>::as<float4>() view
        // behind a per-site H % 4 == 0 guard, so a ragged shape stays on the
        // legacy scalar kernel (tensor_optimize.md 4.1: no vec4 tail split).
        _embed_vec4 = compile(make_embed_lookup_vec4_kernel(), "embed_lookup_vec4");
        _residual_vec4 = compile(make_residual_add_vec4_kernel(), "residual_add_vec4");
        _layernorm_vec4 = compile(make_layernorm_vec4_kernel(), "layernorm_vec4");
        _l2norm_vec4 = compile(make_l2_normalize_vec4_kernel(), "l2_normalize_vec4");
        _dense_pool_vec4 = compile(make_dense_pool_vec4_kernel(), "dense_pool_vec4");
        _sparse_head_vec4 = compile(make_sparse_head_vec4_kernel(), "sparse_head_vec4");
        _colbert_proj_vec4 = compile(make_colbert_proj_vec4_kernel(), "colbert_proj_vec4");

        // coop is a projection replacement only; its matvec reads plain fp32
        // weights and cannot reinterpret ANY packed B operand -- packed half
        // (bf16/fp16) or packed int8 ([scale | codes]) alike
        // (tensor_optimize.md 1.3: no BF16 coop; 5.5: non-fp32 formats gated on
        // fp32 correctness). The demotion is therefore written against
        // "not Fp32", not against a specific packed format, and it happens
        // BEFORE any coop shader is compiled (the coop factory would assert
        // wf == Fp32 otherwise).
        if (_tensor_path == TensorPath::Coop && gemm_wf != GemmWeightFormat::Fp32) {
            LUISA_WARNING("embedding tensor path: requested/resolved 'coop' but the WeightStore "
                          "holds '{}' (non-fp32) GEMM weights; the coop matvec requires fp32 "
                          "weights (plan 1.3/5.5), so demoting to 'vector'. Set "
                          "LUISA_EMBED_WEIGHTS=fp32 to keep coop on this package.",
                          gemm_weight_format_name(gemm_wf));
            _tensor_path = TensorPath::Vector;
            path_reason = luisa::format("coop demoted to vector: '{}' weights cannot enter the coop path",
                                        gemm_weight_format_name(gemm_wf));
        }

        if (_tensor_path == TensorPath::Coop) {
            // fp32 weights are guaranteed here (demoted above otherwise).
            // Resolve the variant (env LUISA_EMBED_COOP_CONFIG exact name,
            // else shipped default coop_gemm_variant(0), mirroring the gemm
            // config idiom), then compile the two epilogue shaders used by the
            // projection sites. The shader names bake variant + tile + epilogue
            // for profiling (plan 5.4), e.g. "embed_coop_gemm_bias_k64_n16_t1".
            _coop_cfg = *embed_coop_variant();
            const luisa::string_view tile_tag = [&] {
                const luisa::string_view n{_coop_cfg.name};
                constexpr luisa::string_view pfx{"coop_"};
                return n.starts_with(pfx) ? n.substr(pfx.size()) : n;
            }();
            const luisa::string coop_bias_name = luisa::format("embed_coop_gemm_bias_{}", tile_tag);
            const luisa::string coop_gelu_name = luisa::format("embed_coop_gemm_bias_gelu_erf_{}", tile_tag);
            _coop_gemm_bias = compile(make_coop_gemm_kernel(GemmEpilogue::Bias, _coop_cfg, coop_bias_name.c_str(), gemm_wf),
                                      coop_bias_name.c_str());
            _coop_gemm_gelu = compile(make_coop_gemm_kernel(GemmEpilogue::BiasGeluErf, _coop_cfg, coop_gelu_name.c_str(), gemm_wf),
                                      coop_gelu_name.c_str());
        }
    }

    // Log once per Executor (plan 2.1 logging requirement: backend, requested
    // path, actual path, reason; the summary line carries the GEMM variant
    // source + weight format).
    log_tensor_path_selection(_device.backend_name(), req, _tensor_path, path_reason);
    {
        // gemm-variant source: coop => the coop matvec variant name; else =>
        // the resolved register-GEMM config string. Named locals keep the
        // .c_str() alive to the log call.
        const luisa::string gemm_src = _tensor_path == TensorPath::Coop
                                           ? luisa::string{luisa::string_view{_coop_cfg.name}}
                                           : gemm_resolved_config_source();
        LUISA_INFO("embedding executor tensor path: resolved='{}', gemm_variant='{}', weight_format='{}'.",
                   tensor_path_name(_tensor_path), gemm_src, gemm_weight_format_name(gemm_wf));
    }
    {
        // opt plan C3 decision (see the compile block above). Reported once so a
        // perf A/B cannot silently compare two different modes: 'fused' is the
        // default, the legacy pair is either the kill switch or the shared-cache
        // contract's H bound.
        if (_fuse_residual_ln) {
            LUISA_INFO("embedding executor residual+layernorm: fused 'residual_layernorm' "
                       "(H={}, 1 dispatch per pair; kill switch EMBED_FUSE_RESIDUAL_LN=0).",
                       _weights.hidden_size());
        } else {
            // Named local keeps the .c_str() alive across the log call.
            const luisa::string reason = !embed_fuse_residual_ln_enabled()
                                             ? luisa::string{"EMBED_FUSE_RESIDUAL_LN=0"}
                                             : luisa::format("H={} > kResidualNormMaxHidden={}",
                                                             _weights.hidden_size(), kResidualNormMaxHidden);
            LUISA_INFO("embedding executor residual+layernorm: legacy pair "
                       "'residual_add'+'layernorm' (reason: {}).",
                       reason.c_str());
        }
    }
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

    // TENSOR-PATH flags (tensor_optimize.md 2.1/5.4). Legacy => both false and
    // EVERY site below falls through to the original, byte-identical dispatch
    // (the kill switch). vec_ok gates the float4 bandwidth/compute kernels
    // (Vector and Coop both use them for the non-projection sites); coop_ok
    // gates only the coop matvec projection replacement. Each site ALSO adds
    // its own divisibility guard (H % 4, N % ntile, K % kslab) so a ragged
    // shape keeps the legacy expression even on a new path.
    const bool vec_ok = (_tensor_path == TensorPath::Vector || _tensor_path == TensorPath::Coop);
    const bool coop_ok = (_tensor_path == TensorPath::Coop);
    // Residual+LN sub-block predicate: residual_add_vec4 needs (rf*H) % 4 == 0
    // and layernorm_vec4 needs H % 4 == 0. The two are COUPLED -- vec4 LN is
    // in-place, so the post-LN1 hidden lives in _tmp2 on the vec4 path (vs _h1
    // under legacy) and every buffer slot in the residual1/LN1/ffn-up-input/
    // residual2/LN2 chain follows from that. The sub-block therefore switches as
    // ONE unit: full vec4 routing when both guards hold, otherwise the exact
    // legacy expressions (bit-identical kill switch, and never a half-vec
    // route that reads a slot the other half never wrote).
    const bool rn_vec = vec_ok && H % 4u == 0u && (rf * H) % 4u == 0u;

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
    if (vec_ok && H % 4u == 0u) {
        // embed_lookup_vec4: same fused gather+add+LN at float4 width, with the
        // identical XLM-R +2 adaptation -- bind a shifted pos view (offset
        // 2*H floats = 16B aligned for H%4==0) and pass pos_offset = 0. Arg
        // order differs from legacy: (word, ids, pos, gamma, beta, out, ...).
        const auto pos_shift_elems = static_cast<size_t>(kEmbedPosOffset) * H;
        cl << _embed_vec4(v4_at0(_weights.word_emb()),
                          _ids_dev.view(0, rf),
                          v4_win(_weights.pos_emb(), pos_shift_elems,
                                 (pos_rows / H - kEmbedPosOffset) * H),
                          v4_at0(_weights.emb_ln_gamma()), v4_at0(_weights.emb_ln_beta()),
                          v4_win(_h0, 0u, rf * H),
                          S, H, kEmbedLayerNormEps, 0u)
                  .dispatch(static_cast<uint>(rf * kEmbedVecBlock));
    } else {
        cl << _embed(_weights.word_emb().view(),
                     _weights.pos_emb().view(static_cast<size_t>(kEmbedPosOffset) * H,
                                             (pos_rows / H - kEmbedPosOffset) * H),
                     _ids_dev.view(0, rf),
                     _weights.emb_ln_gamma().view(), _weights.emb_ln_beta().view(),
                     _h0.view(0, rf * H),
                     S, H, kEmbedLayerNormEps, 0u)
                  .dispatch(static_cast<uint>(rf * kEmbedBlock));
    }
    trace(0u, _h0.view(0, rf * H));

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const auto scores_disp = attention_scores_dispatch(B, S, heads);
    const auto pv_disp = attention_pv_dispatch(B, S, heads, hd);

    // coop matvec projection recorder (tensor_optimize.md 5.3/5.4). Records a
    // coop dispatch into `cl` and returns true ONLY when the resolved path is
    // coop AND the shape is coop-clean (N % ntile == 0, K % kslab == 0); fp32
    // weights are already guaranteed by the ctor (coop was demoted to vector
    // otherwise). Ragged or non-coop shapes return false so the caller records
    // the unchanged legacy _gemm_* dispatch. `A`/`C`/`bias` are bound as
    // zero-copy ByteBufferViews (coop_api.md 4).
    const auto record_coop_proj = [&](const ShaderFor<CoopGemmKernel> &shader,
                                      BufferView<float> A, const Buffer<float> &W,
                                      BufferView<float> C, const Buffer<float> &bias,
                                      uint32_t M, uint32_t N, uint32_t K) noexcept -> bool {
        if (!coop_ok || N % _coop_cfg.ntile != 0u || K % _coop_cfg.kslab != 0u) [[likely]] {
            return false;
        }
        cl << shader(byte_at0(A), byte_at0(W), byte_at0(bias), byte_at0(C), M, N, K)
                  .dispatch(coop_gemm_dispatch_size_config(_coop_cfg, M).x);
        return true;
    };

    //--- 24 encoder layers ----------------------------------------------------
    for (uint32_t i = 0; i < L; i++) {
        const auto &lw = _weights.layer(i);

        // qkv projection: [rows,H] @ [H,3H] + bias -> qkv
        if (!record_coop_proj(_coop_gemm_bias, _h0.view(0, rf * H), lw.qkv,
                              _qkv.view(0, rf * 3u * H), lw.qkv_bias, rows, 3u * H, H)) {
            cl << _gemm_bias(_h0.view(0, rf * H), lw.qkv.view(),
                             _qkv.view(0, rf * 3u * H), lw.qkv_bias.view(),
                             rows, 3u * H, H)
                      .dispatch(gemm_dispatch_size(rows, 3u * H));
        }

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
        if (!record_coop_proj(_coop_gemm_bias, _att.view(0, rf * H), lw.attn_out,
                              _tmp1.view(0, rf * H), lw.attn_out_bias, rows, H, H)) {
            cl << _gemm_bias(_att.view(0, rf * H), lw.attn_out.view(),
                             _tmp1.view(0, rf * H), lw.attn_out_bias.view(),
                             rows, H, H)
                      .dispatch(gemm_dispatch_size(rows, H));
        }
        // residual (_h0 + _tmp1 -> _tmp2) + LN1 (_tmp2 -> post-LN1). Under
        // legacy LN1 writes _h1; the in-place layernorm_vec4 normalizes _tmp2,
        // so the vec4 sub-block keeps the post-LN1 hidden in _tmp2 and leaves
        // _h1 unused (rn_vec drives the whole chain together, see its
        // definition above). The layer-entry running hidden stays in _h0: both
        // residual forms take a=_h0, and LN's result never lands there until
        // LN2 writes the next layer's running value.
        //
        // Opt plan C3: on the Legacy path (and on the ragged vector shapes that
        // stay scalar) the pair is ONE fused dispatch that caches the residual
        // row in shared memory -- see kernels_norm.h. It writes the SAME slot as
        // the legacy pair (`_h1` here, `_h0` for LN2 below), so buffer routing
        // and every downstream read are unchanged; `_tmp2` simply goes unused on
        // that path. Numerically equivalent to the pair within the focused 1e-5
        // ABSOLUTE tolerance (test_embed_kernel_layernorm) and with the same
        // reduction order -- NOT bit-identical (dx re-emits ~1 ULP on the fused
        // form), so this is a rounding-level substitution, not an exact one.
        // EMBED_FUSE_RESIDUAL_LN=0 (or H > kResidualNormMaxHidden) takes the
        // else branch and restores the legacy pair's dispatch sequence
        // unchanged; the choice is fixed in the ctor and both branches only ever
        // reference a shader that was compiled for it.
        if (rn_vec) {
            cl << _residual_vec4(v4_win(_h0, 0u, rf * H), v4_win(_tmp1, 0u, rf * H),
                                 v4_win(_tmp2, 0u, rf * H),
                                 static_cast<uint32_t>(rf * H / 4u))
                      .dispatch(static_cast<uint>(rf * H / 4u))
               << _layernorm_vec4(v4_win(_tmp2, 0u, rf * H),
                                  v4_at0(lw.ln1_gamma), v4_at0(lw.ln1_beta),
                                  H, kEmbedLayerNormEps)
                      .dispatch(static_cast<uint>(rf * kNormVecBlock));
        } else if (_fuse_residual_ln) [[likely]] {
            // a=_h0, b=_tmp1, out=_h1: no read-after-write overlap (and even an
            // aliased out would be safe -- the residual row is cached before the
            // first write). dispatch() = TOTAL THREADS, one block per row.
            cl << _residual_layernorm(_h0.view(0, rf * H), _tmp1.view(0, rf * H),
                                      _h1.view(0, rf * H),
                                      lw.ln1_gamma.view(), lw.ln1_beta.view(),
                                      H, kEmbedLayerNormEps)
                      .dispatch(static_cast<uint>(rf * kNormBlock));
        } else {
            cl << _residual(_h0.view(0, rf * H), _tmp1.view(0, rf * H),
                            _tmp2.view(0, rf * H), rf * H)
                      .dispatch(static_cast<uint>(rf * H))
               << _layernorm(_tmp2.view(0, rf * H), _h1.view(0, rf * H),
                             lw.ln1_gamma.view(), lw.ln1_beta.view(), H, kEmbedLayerNormEps)
                      .dispatch(static_cast<uint>(rf * kNormBlock));
        }

        // FFN (erf-GELU fused in the up-projection epilogue) + residual + LN2.
        // Legacy: _h1 (post-LN1) -> ffn.up -> ffn.down -> _tmp1; residual2
        // (_h1 + _tmp1 -> _tmp2); LN2 (_tmp2 -> _h0). The result returns to _h0
        // in place: _h0's layer input is fully consumed by residual1 before LN2
        // writes back.
        //
        // vec4 routing: layernorm_vec4 has NO separate y operand (it normalizes
        // x in place), so the rn_vec path keeps the post-LN1 hidden in _tmp2
        // (LN1's in-place slot) instead of _h1 -- _h1 is entirely unused there.
        if (!record_coop_proj(_coop_gemm_gelu, rn_vec ? _tmp2.view(0, rf * H) : _h1.view(0, rf * H), lw.ffn_up,
                              _ffn.view(0, rf * I), lw.ffn_up_bias, rows, I, H)) {
            cl << _gemm_gelu(rn_vec ? _tmp2.view(0, rf * H) : _h1.view(0, rf * H),
                             lw.ffn_up.view(),
                             _ffn.view(0, rf * I), lw.ffn_up_bias.view(),
                             rows, I, H)
                      .dispatch(gemm_dispatch_size(rows, I));
        }
        if (!record_coop_proj(_coop_gemm_bias, _ffn.view(0, rf * I), lw.ffn_down,
                              _tmp1.view(0, rf * H), lw.ffn_down_bias, rows, H, I)) {
            cl << _gemm_bias(_ffn.view(0, rf * I), lw.ffn_down.view(),
                             _tmp1.view(0, rf * H), lw.ffn_down_bias.view(),
                             rows, H, I)
                      .dispatch(gemm_dispatch_size(rows, H));
        }
        // residual2 + LN2 (vec path: _tmp2 (post-LN1) + _tmp1 -> _h0, LN2
        // normalizes _h0 in place; legacy: _h1 + _tmp1 -> _tmp2, LN2 -> _h0).
        // Fused (C3): a=_h1 (post-LN1), b=_tmp1 (ffn.down), out=_h0 -- the same
        // slots the legacy pair uses, so _h0 carries the next layer's running
        // hidden and the trace slot below is unchanged (_tmp2, the legacy
        // residual1/LN2 slot, simply goes unused on this path).
        if (rn_vec) {
            cl << _residual_vec4(v4_win(_tmp2, 0u, rf * H), v4_win(_tmp1, 0u, rf * H),
                                 v4_win(_h0, 0u, rf * H),
                                 static_cast<uint32_t>(rf * H / 4u))
                      .dispatch(static_cast<uint>(rf * H / 4u))
               << _layernorm_vec4(v4_win(_h0, 0u, rf * H),
                                  v4_at0(lw.ln2_gamma), v4_at0(lw.ln2_beta),
                                  H, kEmbedLayerNormEps)
                      .dispatch(static_cast<uint>(rf * kNormVecBlock));
        } else if (_fuse_residual_ln) [[likely]] {
            cl << _residual_layernorm(_h1.view(0, rf * H), _tmp1.view(0, rf * H),
                                      _h0.view(0, rf * H),
                                      lw.ln2_gamma.view(), lw.ln2_beta.view(),
                                      H, kEmbedLayerNormEps)
                      .dispatch(static_cast<uint>(rf * kNormBlock));
        } else {
            cl << _residual(_h1.view(0, rf * H), _tmp1.view(0, rf * H),
                            _tmp2.view(0, rf * H), rf * H)
                      .dispatch(static_cast<uint>(rf * H))
               << _layernorm(_tmp2.view(0, rf * H), _h0.view(0, rf * H),
                             lw.ln2_gamma.view(), lw.ln2_beta.view(), H, kEmbedLayerNormEps)
                      .dispatch(static_cast<uint>(rf * kNormBlock));
        }

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
        if (vec_ok && H % 4u == 0u) {
            // fused CLS-slice + L2-normalize at float4 width; hidden [B*S*H]
            // and out [B*H] views start at element 0 (16 B aligned for H%4==0).
            cl << _dense_pool_vec4(v4_win(_h0, 0u, rf * H),
                                   v4_win(_dense_dev, 0u, static_cast<size_t>(B) * H),
                                   S, H)
                      .dispatch(static_cast<uint>(B * kNormVecBlock));
        } else {
            cl << _dense_pool(_h0.view(0, rf * H),
                              _dense_dev.view(0, static_cast<size_t>(B) * H),
                              S, H)
                      .dispatch(static_cast<uint>(B * kNormBlock));
        }
    } else {
        // legacy unfused pair (EMBED_N1_FUSE_POOL=0 kill switch): slice_cls has
        // no vec4 twin, so it stays scalar; l2_normalize uses its vec4 variant
        // whenever the guards allow (both kernels touch the same [b*H, b*H+H)
        // windows, whose byte offsets are multiples of 16 when H % 4 == 0).
        const bool l2_vec = vec_ok && H % 4u == 0u;
        for (uint32_t b = 0; b < B; b++) {
            cl << _slice_cls(_h0.view(static_cast<size_t>(b) * S * H, H),
                             _cls.view(0, H), H)
                      .dispatch(H);
            if (l2_vec) {
                cl << _l2norm_vec4(v4_win(_cls, 0u, H),
                                   v4_win(_dense_dev, static_cast<size_t>(b) * H, H),
                                   H)
                          .dispatch(kNormVecBlock);
            } else {
                cl << _l2norm(_cls.view(0, H), _dense_dev.view(static_cast<size_t>(b) * H, H), H)
                          .dispatch(kNormBlock);
            }
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
            // PAIRS stays on the legacy compact kernel for the vec4 v1 path:
            // sparse_head_compact has NO vec4 twin yet (the vec4 set covers
            // sparse_head only), and mixing vec4 y with the compact-append
            // kernel would change neither the values nor the pairing -- but
            // there is simply nothing to swap in, so this mode keeps the exact
            // legacy dispatch. (Documented v1 limitation.)
            cl << _sparse_count_reset(_sparse_count.view()).dispatch(32u)
               << _sparse_head_compact(_h0.view(0, rf * H), _weights.sparse_w().view(),
                                       _weights.sparse_b(), _mask_dev.view(0, rf),
                                       _sparse_pair_ids.view(0, rf), _sparse_pair_w.view(0, rf),
                                       _sparse_count.view(0, 1), H)
                      .dispatch(static_cast<uint>(rf * kHeadsBlock));
        } else if (vec_ok && H % 4u == 0u) {
            // WINDOW / LEGACY readback modes (they share this dispatch): the
            // float4 dot variant -- X [rows,H] and W_s [H] as float4 views,
            // b_s stays the host scalar, out [rows] stays plain float.
            cl << _sparse_head_vec4(v4_win(_h0, 0u, rf * H), v4_at0(_weights.sparse_w()),
                                    _weights.sparse_b(), _sparse_dev.view(0, rf), H)
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
        // colbert_proj_vec4 host contract: H % 4 == 0 AND N % 4 == 0 (here the
        // call site projects [rows,H] -> [rows,H], so N == H and one guard
        // covers both); X, W_c [H,N] row-major, b_c [N], out [rows,N] all at
        // element offset 0 (16 B aligned).
        const uint32_t colbert_n = H;
        if (vec_ok && H % 4u == 0u && colbert_n % 4u == 0u) {
            cl << _colbert_proj_vec4(v4_win(_h0, 0u, rf * H),
                                     v4_at0(_weights.colbert_w()),
                                     v4_at0(_weights.colbert_b()),
                                     v4_win(_colbert_dev, 0u, rf * colbert_n),
                                     H, colbert_n, 1u)
                      .dispatch(static_cast<uint>(rf * kHeadsVecBlock));
        } else {
            cl << _colbert_proj(_h0.view(0, rf * H), _weights.colbert_w().view(),
                                _weights.colbert_b().view(), _colbert_dev.view(0, rf * H),
                                H, H, 1u)
                      .dispatch(static_cast<uint>(rf * kHeadsBlock));
        }
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
