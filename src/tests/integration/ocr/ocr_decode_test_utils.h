#pragma once
// Shared synthetic dense-decoder fixture + INDEPENDENT CPU oracle for the
// Phase 3 decode tests. Synthetic only: known deterministic weights, a small
// declared geometry, no model download, no real-model parity claim.
//
// The oracle is built ONLY from the Phase 2 host references (cpu_reference.h),
// composed into a full decoder forward pass. It shares NO code with the device
// executor, so a test compares two independent implementations.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ut/ut.hpp"
#include "ocr_test_utils.h"

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/ops/cpu_reference.h"
#include "ocr/package.h"
#include "ocr/runtime/weights.h"
#include "ocr/tokenizer.h"

namespace luisa::ocr::test {

// ---------------------------------------------------------------------------
// Synthetic geometry (declared, not a real checkpoint).
// ---------------------------------------------------------------------------
inline constexpr uint32_t kSynVocab = 32;
inline constexpr uint32_t kSynHidden = 16;
inline constexpr uint32_t kSynLayers = 2;
inline constexpr uint32_t kSynQHeads = 4;
inline constexpr uint32_t kSynKvHeads = 2;// GQA group = 2
inline constexpr uint32_t kSynHeadDim = 4;
inline constexpr uint32_t kSynInter = 32;
inline constexpr uint32_t kSynMaxPos = 64;
inline constexpr uint32_t kSynEos = 31;
inline constexpr float kSynEps = 1e-5f;
inline constexpr float kSynTheta = 10000.0f;

[[nodiscard]] inline DenseDecoderConfig synthetic_config(uint32_t vocab = kSynVocab,
                                                         uint32_t eos = kSynEos,
                                                         RopePairLayout rope = RopePairLayout::adjacent) {
    DenseDecoderConfig c;
    c.vocab_size = vocab;
    c.hidden_size = kSynHidden;
    c.num_layers = kSynLayers;
    c.num_attention_heads = kSynQHeads;
    c.num_kv_heads = kSynKvHeads;
    c.head_dim = kSynHeadDim;
    c.intermediate_size = kSynInter;
    c.rms_norm_eps = kSynEps;
    c.rope_theta = kSynTheta;
    c.rope_pair_layout = rope;
    c.max_position_embeddings = kSynMaxPos;
    c.eos_token_ids = {eos};
    return c;
}

// ---------------------------------------------------------------------------
// Host weight set (row-major; linear weights stored PACKED [K, N]).
// ---------------------------------------------------------------------------
struct SyntheticWeights {
    luisa::vector<float> embed;// [vocab, hidden]
    luisa::vector<float> final_norm;// [hidden]
    luisa::vector<float> lm_head;// [vocab, hidden] (== embed when tied)
    bool tied{true};             // when false, lm_head is a standalone tensor
    uint32_t vocab{0};           // 0 => kSynVocab (set by the generator)
    struct Layer {
        luisa::vector<float> input_norm, post_norm;// [hidden]
        luisa::vector<float> q_w, k_w, v_w, o_w;// [K, N]
        luisa::vector<float> q_b, k_b, v_b;// [N]
        luisa::vector<float> gate_w, up_w, down_w;// [K, N]
    };
    luisa::vector<Layer> layers;
};

// ---------------------------------------------------------------------------
// Scripted deterministic decoder (for stop/EOS/max-token controller tests).
//
// Mechanism: a DOMINANT one-hot embedding + identity LM head makes greedy
// decoding an exact token state-machine. embed[t] = A * e_{next[t]} and
// lm_head[v] = A * e_v, final_norm = 1. The residual stream carries A*e_{next}
// from the embedding; the (shrunk) transformer blocks add only bounded
// corrections, and RMSNorm preserves direction, so logits[v] ~ A^2 *
// [v == next[input]] dominates and argmax == next[input]. The blocks still run
// for real (KV/RoPE/attention/MLP); the CPU oracle computes the true logits.
//
// REQUIRES vocab <= kSynHidden (the one-hot index must fit a hidden row). The
// stop tests use a compact synthetic tokenizer with a small vocabulary.
inline void make_scripted_decoder(SyntheticWeights &w, luisa::span<const uint32_t> next) {
    const uint32_t vocab = w.vocab == 0 ? kSynVocab : w.vocab;
    LUISA_ASSERT(next.size() == vocab, "scripted next map must cover the vocabulary");
    LUISA_ASSERT(vocab <= kSynHidden, "scripted decoder needs vocab <= hidden (one-hot rows)");
    w.tied = false;
    constexpr float A = 1.0e3f;
    w.embed.assign(static_cast<size_t>(vocab) * kSynHidden, 0.0f);
    w.lm_head.assign(static_cast<size_t>(vocab) * kSynHidden, 0.0f);
    for (uint32_t t = 0; t < vocab; t++) {
        w.embed[static_cast<size_t>(t) * kSynHidden + next[t]] = A;
        w.lm_head[static_cast<size_t>(t) * kSynHidden + t] = A;
    }
    w.final_norm.assign(kSynHidden, 1.0f);
    for (auto &L : w.layers) {
        const auto shrink = [](luisa::vector<float> &v) {
            for (auto &x : v) { x *= 1.0e-3f; }
        };
        shrink(L.q_w);
        shrink(L.k_w);
        shrink(L.v_w);
        shrink(L.o_w);
        shrink(L.gate_w);
        shrink(L.up_w);
        shrink(L.down_w);
        shrink(L.q_b);
        shrink(L.k_b);
        shrink(L.v_b);
    }
}

// ---------------------------------------------------------------------------
// Compact synthetic Unigram tokenizer (small vocab <= kSynHidden) for the
// scripted stop tests. A VALID Unigram tokenizer.json parsed by the real
// Phase 1 loader: raw UTF-8 pieces (so multi-byte characters can split across
// tokens) plus an EOS special token. Generated in-memory; no fixture file.
//
// Pieces (id -> text):
//   0: "he"   1: "llo"  2: " "   3: "wo"   4: "rld"
//   5: "你"   6: "好"   7: "世"   8: "界"   9: "</s>" (EOS, special)
// Unigram decode concatenates pieces verbatim (no metaspace when the
// normalizer is null), so the detokenizer output is the exact text.
// ---------------------------------------------------------------------------
inline constexpr uint32_t kCompactVocab = 10;
inline constexpr uint32_t kCompactEos = 9;

[[nodiscard]] inline luisa::string json_escape(luisa::string_view s) {
    luisa::string esc;
    for (char c : s) {
        if (c == '"' || c == '\\') { esc += '\\'; esc += c; }
        else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
            esc += buf;
        } else { esc += c; }
    }
    return esc;
}

[[nodiscard]] inline luisa::string compact_tokenizer_json() {
    const luisa::vector<luisa::string> pieces = {
        "he", "llo", " ", "wo", "rld",
        "\xE4\xBD\xA0", "\xE5\xA5\xBD", "\xE4\xB8\x96", "\xE7\x95\x8C", "</s>",
    };
    luisa::string vocab = "[";
    for (size_t i = 0; i < pieces.size(); i++) {
        vocab += "[\"" + json_escape(pieces[i]) + "\",-1.0]";
        if (i + 1 < pieces.size()) { vocab += ","; }
    }
    vocab += "]";
    luisa::string json = "{";
    json += "\"version\":\"1.0\",";
    json += "\"added_tokens\":[{\"id\":9,\"content\":\"</s>\",\"special\":true,"
            "\"lstrip\":false,\"rstrip\":false,\"single_word\":false}],";
    json += "\"normalizer\":null,";
    json += "\"pre_tokenizer\":null,";
    json += "\"post_processor\":null,";
    json += "\"model\":{\"type\":\"Unigram\",\"unk_id\":0,\"byte_fallback\":false,"
            "\"add_bos_token\":false,\"add_eos_token\":false,";
    json += "\"vocab\":" + vocab;
    json += "}}";
    return json;
}

[[nodiscard]] inline luisa::unique_ptr<Tokenizer, TokenizerDeleter> load_compact_tokenizer() {
    auto json = compact_tokenizer_json();
    auto r = Tokenizer::parse(json);
    if (!r.ok()) {
        LUISA_WARNING("compact tokenizer parse failed: {}", r.status.message);
        return nullptr;
    }
    return std::move(*r.value);
}

// Deterministic weight generation (LCG, no <random>). Values are bounded so
// the forward pass stays numerically well-conditioned and greedy tokens are
// stable across backends.
[[nodiscard]] inline SyntheticWeights generate_synthetic_weights(uint32_t seed = 0x5EEDu,
                                                                 uint32_t vocab = kSynVocab) {
    SyntheticWeights w;
    w.vocab = vocab;
    Lcg lcg;
    lcg.state = seed;
    const auto fill = [&lcg](luisa::vector<float> &v, size_t n, float lo, float hi) {
        v.resize(n);
        for (auto &x : v) { x = lcg.uniform(lo, hi); }
    };
    fill(w.embed, static_cast<size_t>(vocab) * kSynHidden, -0.6f, 0.6f);
    w.lm_head = w.embed;// tied head
    fill(w.final_norm, kSynHidden, 0.7f, 1.3f);
    w.layers.resize(kSynLayers);
    const uint32_t qd = kSynQHeads * kSynHeadDim;
    const uint32_t kvd = kSynKvHeads * kSynHeadDim;
    for (auto &L : w.layers) {
        fill(L.input_norm, kSynHidden, 0.7f, 1.3f);
        fill(L.post_norm, kSynHidden, 0.7f, 1.3f);
        fill(L.q_w, static_cast<size_t>(kSynHidden) * qd, -0.35f, 0.35f);
        fill(L.k_w, static_cast<size_t>(kSynHidden) * kvd, -0.35f, 0.35f);
        fill(L.v_w, static_cast<size_t>(kSynHidden) * kvd, -0.35f, 0.35f);
        fill(L.o_w, static_cast<size_t>(qd) * kSynHidden, -0.35f, 0.35f);
        fill(L.q_b, qd, -0.1f, 0.1f);
        fill(L.k_b, kvd, -0.1f, 0.1f);
        fill(L.v_b, kvd, -0.1f, 0.1f);
        fill(L.gate_w, static_cast<size_t>(kSynHidden) * kSynInter, -0.35f, 0.35f);
        fill(L.up_w, static_cast<size_t>(kSynHidden) * kSynInter, -0.35f, 0.35f);
        fill(L.down_w, static_cast<size_t>(kSynInter) * kSynHidden, -0.35f, 0.35f);
    }
    return w;
}

// ---------------------------------------------------------------------------
// Package writing + weight upload (exercises the real Phase 1/2 path).
// ---------------------------------------------------------------------------
[[nodiscard]] inline luisa::vector<std::byte> f32_bytes(luisa::span<const float> v) {
    luisa::vector<std::byte> out(v.size() * 4);
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

[[nodiscard]] inline PackageManifest synthetic_manifest() {
    PackageManifest m;
    m.model = "glm_ocr";// synthetic test artifact; NOT a real model claim
    m.recipe_version = "1";
    m.checkpoint_revision = "synthetic-rev";
    m.config_fingerprint = "synthetic-fp";
    m.license = "test";
    m.producer = "ocr_decode_test";
    m.producer_version = "1";
    m.preprocessing_version = "1";
    m.tokenizer_version = "1";
    m.output_parser_version = "1";
    m.component = "decoder";
    return m;
}

// Write the synthetic decoder tensors to a .lcocr package. Linear weights use
// layout tag `kn` ([K,N]); the LM head is omitted (tied to the embedding).
[[nodiscard]] inline Status write_synthetic_package(const luisa::filesystem::path &path,
                                                    const SyntheticWeights &w) {
    auto writer = PackageWriter::create(path);
    if (!writer.ok()) { return writer.status; }
    auto &wr = *writer.value;
    const uint32_t vocab = w.vocab == 0 ? kSynVocab : w.vocab;
    const DenseDecoderTensorNames names;
    const auto add = [&wr](const luisa::string &name, luisa::vector<uint64_t> shape,
                           TensorLayout layout, luisa::span<const float> data) -> Status {
        PackageWriter::TensorSpec spec;
        spec.name = name;
        spec.encoding = TensorEncoding::fp32;
        spec.layout = layout;
        spec.shape = std::move(shape);
        spec.payload = f32_bytes(data);
        return wr->add_tensor(spec);
    };
    if (auto s = add(names.embed_tokens(), {vocab, kSynHidden}, TensorLayout::row_major, w.embed); !s.ok()) { return s; }
    if (auto s = add(names.final_norm(), {kSynHidden}, TensorLayout::row_major, w.final_norm); !s.ok()) { return s; }
    if (!w.tied) {
        // Standalone LM head, packed [hidden, vocab] (kn). The host stores it
        // [vocab, hidden]; transpose for the package.
        luisa::vector<float> head_t(static_cast<size_t>(kSynHidden) * vocab);
        for (uint32_t v = 0; v < vocab; v++) {
            for (uint32_t h = 0; h < kSynHidden; h++) {
                head_t[static_cast<size_t>(h) * vocab + v] = w.lm_head[static_cast<size_t>(v) * kSynHidden + h];
            }
        }
        if (auto s = add(names.lm_head(), {kSynHidden, vocab}, TensorLayout::kn, head_t); !s.ok()) { return s; }
    }
    const uint64_t qd = kSynQHeads * kSynHeadDim, kvd = kSynKvHeads * kSynHeadDim;
    for (uint32_t l = 0; l < kSynLayers; l++) {
        const auto &L = w.layers[l];
        if (auto s = add(names.input_layernorm(l), {kSynHidden}, TensorLayout::row_major, L.input_norm); !s.ok()) { return s; }
        if (auto s = add(names.post_attention_layernorm(l), {kSynHidden}, TensorLayout::row_major, L.post_norm); !s.ok()) { return s; }
        if (auto s = add(names.q_proj(l), {kSynHidden, qd}, TensorLayout::kn, L.q_w); !s.ok()) { return s; }
        if (auto s = add(names.k_proj(l), {kSynHidden, kvd}, TensorLayout::kn, L.k_w); !s.ok()) { return s; }
        if (auto s = add(names.v_proj(l), {kSynHidden, kvd}, TensorLayout::kn, L.v_w); !s.ok()) { return s; }
        if (auto s = add(names.o_proj(l), {qd, kSynHidden}, TensorLayout::kn, L.o_w); !s.ok()) { return s; }
        if (auto s = add(names.q_bias(l), {qd}, TensorLayout::row_major, L.q_b); !s.ok()) { return s; }
        if (auto s = add(names.k_bias(l), {kvd}, TensorLayout::row_major, L.k_b); !s.ok()) { return s; }
        if (auto s = add(names.v_bias(l), {kvd}, TensorLayout::row_major, L.v_b); !s.ok()) { return s; }
        if (auto s = add(names.gate_proj(l), {kSynHidden, kSynInter}, TensorLayout::kn, L.gate_w); !s.ok()) { return s; }
        if (auto s = add(names.up_proj(l), {kSynHidden, kSynInter}, TensorLayout::kn, L.up_w); !s.ok()) { return s; }
        if (auto s = add(names.down_proj(l), {kSynInter, kSynHidden}, TensorLayout::kn, L.down_w); !s.ok()) { return s; }
    }
    return wr->finish(synthetic_manifest());
}

// ---------------------------------------------------------------------------
// Negative-test package writer (M1): writes the synthetic decoder with ONE
// tensor deliberately corrupted (wrong shape / wrong layout / undersized), so
// the executor's create() must refuse it BEFORE any dispatch. The corruption
// is applied to the on-disk descriptor; the payload may be truncated/oversized
// accordingly. `corrupt_name` selects the tensor; `mode` selects the fault.
// ---------------------------------------------------------------------------
enum class CorruptMode {
    undersize_shape,   // shrink the last dim by 1 (element count too small)
    wrong_layout,      // flip kn <-> row_major (layout tag mismatch)
    transpose_shape,   // swap the two dims of a 2-D tensor (shape mismatch)
};

[[nodiscard]] inline Status write_corrupt_synthetic_package(const luisa::filesystem::path &path,
                                                            const SyntheticWeights &w,
                                                            const luisa::string &corrupt_name,
                                                            CorruptMode mode,
                                                            uint32_t config_vocab = 0) {
    auto writer = PackageWriter::create(path);
    if (!writer.ok()) { return writer.status; }
    auto &wr = *writer.value;
    // The WRITTEN embed vocab is w.vocab; config_vocab (when nonzero) is the
    // vocab the executor config will declare, used only to size the standalone
    // lm_head so a non-square [hidden, config_vocab] head can be transposed.
    const uint32_t vocab = w.vocab == 0 ? kSynVocab : w.vocab;
    const uint32_t head_vocab = config_vocab == 0 ? vocab : config_vocab;
    const DenseDecoderTensorNames names;
    const auto add = [&](const luisa::string &name, luisa::vector<uint64_t> shape,
                         TensorLayout layout, luisa::span<const float> data) -> Status {
        PackageWriter::TensorSpec spec;
        spec.name = name;
        spec.encoding = TensorEncoding::fp32;
        spec.layout = layout;
        spec.shape = shape;
        luisa::vector<float> payload(data.begin(), data.end());
        if (name == corrupt_name) {
            switch (mode) {
                case CorruptMode::undersize_shape:
                    if (!spec.shape.empty() && spec.shape.back() > 1u) { spec.shape.back() -= 1u; }
                    break;
                case CorruptMode::wrong_layout:
                    spec.layout = (layout == TensorLayout::kn) ? TensorLayout::row_major : TensorLayout::kn;
                    break;
                case CorruptMode::transpose_shape:
                    if (spec.shape.size() == 2u) { std::swap(spec.shape[0], spec.shape[1]); }
                    break;
            }
        }
        // Resize the payload to the (possibly corrupted) shape product so the
        // package writer accepts it; the executor must then reject the
        // mismatched shape/layout/element-count at create().
        uint64_t want = 1u;
        for (auto d : spec.shape) { want *= d; }
        payload.resize(static_cast<size_t>(want), 0.0f);
        spec.payload = f32_bytes(payload);
        return wr->add_tensor(spec);
    };
    if (auto s = add(names.embed_tokens(), {vocab, kSynHidden}, TensorLayout::row_major, w.embed); !s.ok()) { return s; }
    if (auto s = add(names.final_norm(), {kSynHidden}, TensorLayout::row_major, w.final_norm); !s.ok()) { return s; }
    if (!w.tied) {
        // Standalone head [hidden, head_vocab]; the host lm_head is stored
        // [head_vocab, hidden] (resize/pad from w.lm_head which is [vocab,hidden]).
        luisa::vector<float> head_host(static_cast<size_t>(head_vocab) * kSynHidden, 0.0f);
        const uint32_t copy_vocab = head_vocab < vocab ? head_vocab : vocab;
        for (uint32_t v = 0; v < copy_vocab; v++) {
            for (uint32_t h = 0; h < kSynHidden; h++) {
                head_host[static_cast<size_t>(v) * kSynHidden + h] = w.lm_head[static_cast<size_t>(v) * kSynHidden + h];
            }
        }
        luisa::vector<float> head_t(static_cast<size_t>(kSynHidden) * head_vocab);
        for (uint32_t v = 0; v < head_vocab; v++) {
            for (uint32_t h = 0; h < kSynHidden; h++) {
                head_t[static_cast<size_t>(h) * head_vocab + v] = head_host[static_cast<size_t>(v) * kSynHidden + h];
            }
        }
        if (auto s = add(names.lm_head(), {kSynHidden, head_vocab}, TensorLayout::kn, head_t); !s.ok()) { return s; }
    }
    const uint64_t qd = kSynQHeads * kSynHeadDim, kvd = kSynKvHeads * kSynHeadDim;
    for (uint32_t l = 0; l < kSynLayers; l++) {
        const auto &L = w.layers[l];
        if (auto s = add(names.input_layernorm(l), {kSynHidden}, TensorLayout::row_major, L.input_norm); !s.ok()) { return s; }
        if (auto s = add(names.post_attention_layernorm(l), {kSynHidden}, TensorLayout::row_major, L.post_norm); !s.ok()) { return s; }
        if (auto s = add(names.q_proj(l), {kSynHidden, qd}, TensorLayout::kn, L.q_w); !s.ok()) { return s; }
        if (auto s = add(names.k_proj(l), {kSynHidden, kvd}, TensorLayout::kn, L.k_w); !s.ok()) { return s; }
        if (auto s = add(names.v_proj(l), {kSynHidden, kvd}, TensorLayout::kn, L.v_w); !s.ok()) { return s; }
        if (auto s = add(names.o_proj(l), {qd, kSynHidden}, TensorLayout::kn, L.o_w); !s.ok()) { return s; }
        if (auto s = add(names.q_bias(l), {qd}, TensorLayout::row_major, L.q_b); !s.ok()) { return s; }
        if (auto s = add(names.k_bias(l), {kvd}, TensorLayout::row_major, L.k_b); !s.ok()) { return s; }
        if (auto s = add(names.v_bias(l), {kvd}, TensorLayout::row_major, L.v_b); !s.ok()) { return s; }
        if (auto s = add(names.gate_proj(l), {kSynHidden, kSynInter}, TensorLayout::kn, L.gate_w); !s.ok()) { return s; }
        if (auto s = add(names.up_proj(l), {kSynHidden, kSynInter}, TensorLayout::kn, L.up_w); !s.ok()) { return s; }
        if (auto s = add(names.down_proj(l), {kSynInter, kSynHidden}, TensorLayout::kn, L.down_w); !s.ok()) { return s; }
    }
    return wr->finish(synthetic_manifest());
}

// Upload the synthetic decoder component into a WeightStore (sync).
[[nodiscard]] inline Status upload_synthetic_weights(compute::Device &device, compute::Stream &stream,
                                                     const luisa::filesystem::path &package,
                                                     WeightStore &store) {
    auto reader = PackageReader::open(package);
    if (!reader.ok()) { return reader.status; }
    return store.upload_component_sync(device, stream, **reader.value, "decoder");
}

// ---------------------------------------------------------------------------
// Independent CPU oracle (composed from Phase 2 host references).
// ---------------------------------------------------------------------------
struct OracleLayerTrace {
    luisa::vector<float> input_norm;// [rows, hidden]
    luisa::vector<float> q, k, v;// pre-RoPE [rows, qd/kvd/kvd]
    luisa::vector<float> q_rope, k_rope;// [rows, qd/kvd]
    luisa::vector<float> attn;// [rows, qd]
    luisa::vector<float> after_o;// [rows, hidden] residual after o-proj
    luisa::vector<float> post_norm;// [rows, hidden]
    luisa::vector<float> gate, up, mlp;// [rows, inter]
    luisa::vector<float> out;// [rows, hidden] residual after MLP
};

struct OracleResult {
    luisa::vector<float> embeddings;// [rows, hidden] after mm replacement
    luisa::vector<OracleLayerTrace> layers;
    luisa::vector<float> final_norm;// [rows, hidden]
    luisa::vector<float> logits;// [rows, vocab]
    // Committed KV after the pass: per layer [total_tokens, kvd] (K is RoPE'd).
    luisa::vector<luisa::vector<float>> k_cache, v_cache;
};

// One decoder block over `rows` query tokens at absolute position pos0,
// reading/writing the contiguous KV prefix. `k_cache`/`v_cache` hold the FULL
// committed prefix [0, pos0+rows) on entry; the new rows are appended.
[[nodiscard]] inline Status oracle_block(const SyntheticWeights &w, uint32_t layer,
                                         luisa::span<const float> x, uint32_t rows, uint32_t pos0,
                                         luisa::vector<float> &k_cache, luisa::vector<float> &v_cache,
                                         OracleLayerTrace &trace, luisa::vector<float> &x_out,
                                         RopePairLayout rope = RopePairLayout::adjacent) {
    const auto &L = w.layers[layer];
    const uint32_t hidden = kSynHidden, qd = kSynQHeads * kSynHeadDim, kvd = kSynKvHeads * kSynHeadDim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kSynHeadDim));
    auto s = ref_rms_norm(x, L.input_norm, rows, hidden, kSynEps, trace.input_norm);
    if (!s.ok()) { return s; }
    s = ref_gemm(trace.input_norm, L.q_w, L.q_b, rows, qd, hidden, RefGemmEpilogue::bias, trace.q);
    if (!s.ok()) { return s; }
    s = ref_gemm(trace.input_norm, L.k_w, L.k_b, rows, kvd, hidden, RefGemmEpilogue::bias, trace.k);
    if (!s.ok()) { return s; }
    s = ref_gemm(trace.input_norm, L.v_w, L.v_b, rows, kvd, hidden, RefGemmEpilogue::bias, trace.v);
    if (!s.ok()) { return s; }
    // Positions for the new rows.
    luisa::vector<int32_t> positions(rows);
    for (uint32_t i = 0; i < rows; i++) { positions[i] = static_cast<int32_t>(pos0 + i); }
    // Double-precision RoPE oracle (small positions => tight tolerance). The
    // pair layout matches the executor's selected convention: adjacent
    // (2d,2d+1) or rotate-half (d, d+head_dim/2).
    if (rope == RopePairLayout::rotate_half) {
        s = ref_rope_rotate_half(trace.q, positions, rows, kSynQHeads, kSynHeadDim, kSynTheta, trace.q_rope);
        if (!s.ok()) { return s; }
        s = ref_rope_rotate_half(trace.k, positions, rows, kSynKvHeads, kSynHeadDim, kSynTheta, trace.k_rope);
        if (!s.ok()) { return s; }
    } else {
        s = ref_rope_1d(trace.q, positions, rows, kSynQHeads, kSynHeadDim, kSynTheta, trace.q_rope);
        if (!s.ok()) { return s; }
        s = ref_rope_1d(trace.k, positions, rows, kSynKvHeads, kSynHeadDim, kSynTheta, trace.k_rope);
        if (!s.ok()) { return s; }
    }
    // Append the new K (RoPE'd) / V rows to the cache prefix.
    s = ref_kv_append(trace.k_rope, trace.v, k_cache, v_cache, rows, kvd, pos0, pos0 + rows);
    if (!s.ok()) { return s; }
    // Causal attention over the full prefix [0, pos0+rows).
    RefAttentionMask mask;
    mask.kind = RefMaskKind::causal;
    const uint32_t k_len = pos0 + rows;
    s = ref_attention(trace.q_rope, k_cache, v_cache, rows, k_len, kSynQHeads, kSynKvHeads,
                      kSynHeadDim, scale, 0.0f, mask, trace.attn);
    if (!s.ok()) { return s; }
    // o-proj + residual.
    s = ref_gemm(trace.attn, L.o_w, x, rows, hidden, qd, RefGemmEpilogue::residual, trace.after_o);
    if (!s.ok()) { return s; }
    s = ref_rms_norm(trace.after_o, L.post_norm, rows, hidden, kSynEps, trace.post_norm);
    if (!s.ok()) { return s; }
    s = ref_gemm(trace.post_norm, L.gate_w, {}, rows, kSynInter, hidden, RefGemmEpilogue::none, trace.gate);
    if (!s.ok()) { return s; }
    s = ref_gemm(trace.post_norm, L.up_w, {}, rows, kSynInter, hidden, RefGemmEpilogue::none, trace.up);
    if (!s.ok()) { return s; }
    s = ref_gated_act(trace.gate, trace.up, 0u, trace.mlp);
    if (!s.ok()) { return s; }
    s = ref_gemm(trace.mlp, L.down_w, trace.after_o, rows, hidden, kSynInter, RefGemmEpilogue::residual, trace.out);
    if (!s.ok()) { return s; }
    x_out = trace.out;
    return Status::success();
}

// Full forward over `tokens` (prompt prefill, or a single decode token with a
// pre-existing KV prefix). `mm_features` replaces rows [mm_start, mm_start+count)
// of the token embeddings when non-empty. `kv_prefix` is the committed token
// count already present in each per-layer cache (0 for a fresh prefill).
// `k_caches`/`v_caches` are sized [layers]; each is resized to (kv_prefix+rows)
// rows and the new rows are appended in place. On a fresh prefill pass empty
// vectors; on a decode step pass the caches returned by the previous call.
[[nodiscard]] inline Status oracle_forward(const SyntheticWeights &w,
                                           luisa::span<const uint32_t> tokens,
                                           luisa::span<const float> mm_features,
                                           uint32_t mm_start, uint32_t mm_count,
                                           uint32_t kv_prefix,
                                           luisa::vector<luisa::vector<float>> &k_caches,
                                           luisa::vector<luisa::vector<float>> &v_caches,
                                           OracleResult &result,
                                           RopePairLayout rope = RopePairLayout::adjacent) {
    const uint32_t rows = static_cast<uint32_t>(tokens.size());
    const uint32_t hidden = kSynHidden, kvd = kSynKvHeads * kSynHeadDim;
    const uint32_t vocab = w.vocab == 0 ? kSynVocab : w.vocab;
    luisa::vector<int32_t> ids(tokens.begin(), tokens.end());
    auto s = ref_embedding_lookup(w.embed, ids, vocab, hidden, result.embeddings);
    if (!s.ok()) { return s; }
    if (!mm_features.empty()) {
        luisa::vector<float> replaced;
        s = ref_mm_scatter(result.embeddings, mm_features, rows, hidden, mm_start, replaced);
        if (!s.ok()) { return s; }
        result.embeddings = std::move(replaced);
    }
    if (k_caches.empty()) {
        k_caches.resize(kSynLayers);
        v_caches.resize(kSynLayers);
    }
    const uint32_t total = kv_prefix + rows;
    for (uint32_t l = 0; l < kSynLayers; l++) {
        k_caches[l].resize(static_cast<size_t>(total) * kvd);
        v_caches[l].resize(static_cast<size_t>(total) * kvd);
    }
    result.layers.resize(kSynLayers);
    luisa::vector<float> x = result.embeddings;
    for (uint32_t l = 0; l < kSynLayers; l++) {
        luisa::vector<float> x_out;
        s = oracle_block(w, l, x, rows, kv_prefix, k_caches[l], v_caches[l],
                         result.layers[l], x_out, rope);
        if (!s.ok()) { return s; }
        x = std::move(x_out);
    }
    s = ref_rms_norm(x, w.final_norm, rows, hidden, kSynEps, result.final_norm);
    if (!s.ok()) { return s; }
    // LM head: logits[r,v] = sum_h final_norm[r,h] * head[v,h]. The device
    // uses B = head^T [hidden, vocab]; build that and call ref_gemm. The head
    // is the embedding when tied, else the standalone lm_head.
    const luisa::vector<float> &head = w.tied ? w.embed : w.lm_head;
    luisa::vector<float> lm_head_t(static_cast<size_t>(hidden) * vocab);
    for (uint32_t v = 0; v < vocab; v++) {
        for (uint32_t h = 0; h < hidden; h++) {
            lm_head_t[static_cast<size_t>(h) * vocab + v] = head[static_cast<size_t>(v) * hidden + h];
        }
    }
    s = ref_gemm(result.final_norm, lm_head_t, {}, rows, vocab, hidden, RefGemmEpilogue::none, result.logits);
    if (!s.ok()) { return s; }
    result.k_cache = k_caches;
    result.v_cache = v_caches;
    return Status::success();
}

// Greedy argmax over a logits row (lowest-index tie-break).
[[nodiscard]] inline uint32_t oracle_argmax(luisa::span<const float> logits) {
    uint32_t best = 0u;
    float best_v = logits[0];
    for (uint32_t i = 1; i < logits.size(); i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Tokenizer fixture (the frozen Phase 1 BPE tokenizer; vocab 264). The decode
// tests that need real detokenization use a config whose vocab matches it.
// ---------------------------------------------------------------------------
[[nodiscard]] inline luisa::filesystem::path bpe_tokenizer_path() {
    return luisa::filesystem::path{LUISA_OCR_TEST_DATA_DIR} / "tokenizer_bpe.json";
}

// Load the frozen BPE tokenizer (vocab 264, eos </s> = 263).
[[nodiscard]] inline luisa::unique_ptr<Tokenizer, TokenizerDeleter> load_bpe_tokenizer() {
    auto r = Tokenizer::load(bpe_tokenizer_path());
    if (!r.ok()) {
        LUISA_WARNING("failed to load BPE tokenizer fixture: {}", r.status.message);
        return nullptr;
    }
    return std::move(*r.value);
}

// Encode text to token ids with the BPE tokenizer (no special tokens).
[[nodiscard]] inline luisa::vector<uint32_t> bpe_encode(const Tokenizer &tok, luisa::string_view text) {
    auto r = tok.encode(text, /*add_special_tokens=*/false);
    if (!r.ok()) {
        LUISA_WARNING("BPE encode failed: {}", r.status.message);
        return {};
    }
    return std::move(*r.value);
}

// Temp dir helper (RAII cleanup).
struct TempDir {
    luisa::filesystem::path path;
    TempDir() {
        path = luisa::filesystem::temp_directory_path() /
               ("lcocr_decode_test_" + luisa::format("{}", reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

}// namespace luisa::ocr::test