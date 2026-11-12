#pragma once
// Shared synthetic Qwen3-VL fixture + INDEPENDENT CPU oracle for the Phase 4
// vertical-slice tests. Synthetic only: known deterministic weights, the
// declared small geometry (models/qwen3_vl.h qwen3_vl_synthetic_config), no
// model download, no real-model parity claim.
//
// The oracle is built ONLY from the Phase 2 host references (cpu_reference.h)
// composed into the full Qwen3-VL forward pass (vision + merger + DeepStack +
// Qwen3 decoder). It shares NO code with the device executor, so a test
// compares two independent implementations.
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

#include "ocr/models/qwen3_vl.h"
#include "ocr/ops/cpu_reference.h"
#include "ocr/package.h"
#include "ocr/runtime/qwen3_vl_executor.h"
#include "ocr/runtime/weights.h"

// Host preprocessing bridge. This header is included ONLY by host-oracle
// test TUs that also include the device executor; the bridge header pulls
// preprocess.h -> api.h -> model_recipe.h (host AttentionMaskKind). To keep
// the device TU clean, the bridge function is forward-declared here over the
// shared plain data types and the test links the bridge TU.
namespace luisa::ocr {
[[nodiscard]] Result<Qwen3VlPreprocessedImage> qwen3_vl_preprocess_image_host(
    const Qwen3VlImageInput &image, luisa::span<const float> mean, luisa::span<const float> std_dev);
}

namespace luisa::ocr::test {

// ---------------------------------------------------------------------------
// Synthetic geometry aliases (the declared small contract).
// ---------------------------------------------------------------------------
inline constexpr uint32_t kQ3Vocab = 32;
inline constexpr uint32_t kQ3Hidden = 32;
inline constexpr uint32_t kQ3Layers = 3;
inline constexpr uint32_t kQ3QHeads = 4;
inline constexpr uint32_t kQ3KvHeads = 2;
inline constexpr uint32_t kQ3HeadDim = 8;
inline constexpr uint32_t kQ3Inter = 64;
inline constexpr uint32_t kQ3MaxPos = 128;
inline constexpr uint32_t kQ3Eos = 31;
inline constexpr uint32_t kQ3VHidden = 16;
inline constexpr uint32_t kQ3VHeads = 2;
inline constexpr uint32_t kQ3VHeadDim = 8;
inline constexpr uint32_t kQ3VLayers = 2;
inline constexpr uint32_t kQ3VInter = 32;
inline constexpr uint32_t kQ3Patch = 2;
inline constexpr uint32_t kQ3Merge = 2;
inline constexpr uint32_t kQ3MergerInter = 32;
inline constexpr float kQ3Eps = 1e-5f;
inline constexpr float kQ3Theta = 10000.0f;

// ---------------------------------------------------------------------------
// Host weight set (row-major; linear weights stored PACKED [K, N]).
// ---------------------------------------------------------------------------
struct Qwen3SyntheticWeights {
    // Vision.
    luisa::vector<float> patch_embed_w;// [patch_dim, vhidden]
    luisa::vector<float> patch_embed_b;// [vhidden]
    struct VisionBlock {
        luisa::vector<float> norm1_w, norm1_b, norm2_w, norm2_b;// [vhidden]
        luisa::vector<float> q_w, k_w, v_w, o_w;// [vhidden, vhidden]
        luisa::vector<float> q_b, k_b, v_b, o_b;// [vhidden]
        luisa::vector<float> q_norm, k_norm;// [vhead_dim]
        luisa::vector<float> fc1_w;// [vhidden, vinter]
        luisa::vector<float> fc1_b;// [vinter]
        luisa::vector<float> fc2_w;// [vinter, vhidden]
        luisa::vector<float> fc2_b;// [vhidden]
    };
    luisa::vector<VisionBlock> blocks;
    struct Merger {
        luisa::vector<float> ln_w, ln_b;// [merged_dim]
        luisa::vector<float> fc1_w;// [merged_dim, merger_inter]
        luisa::vector<float> fc1_b;// [merger_inter]
        luisa::vector<float> fc2_w;// [merger_inter, hidden]
        luisa::vector<float> fc2_b;// [hidden]
    };
    Merger merger;
    luisa::vector<Merger> ds_mergers;
    // Text decoder (Qwen3: QK-Norm, no QKV bias).
    luisa::vector<float> embed;// [vocab, hidden]
    luisa::vector<float> final_norm;// [hidden]
    luisa::vector<float> lm_head;// [vocab, hidden] (== embed when tied)
    bool tied{true};
    struct TextLayer {
        luisa::vector<float> input_norm, post_norm;// [hidden]
        luisa::vector<float> q_w, k_w, v_w, o_w;// [K, N]
        luisa::vector<float> q_norm, k_norm;// [head_dim]
        luisa::vector<float> gate_w, up_w, down_w;// [K, N]
    };
    luisa::vector<TextLayer> layers;
};

// Deterministic weight generation (LCG). Bounded values keep the forward pass
// well-conditioned and greedy tokens stable across backends.
[[nodiscard]] inline Qwen3SyntheticWeights generate_qwen3_synthetic_weights(uint32_t seed = 0x5EEDu) {
    Qwen3SyntheticWeights w;
    Lcg lcg;
    lcg.state = seed;
    const auto fill = [&lcg](luisa::vector<float> &v, size_t n, float lo, float hi) {
        v.resize(n);
        for (auto &x : v) { x = lcg.uniform(lo, hi); }
    };
    const uint32_t patch_dim = 3u * kQ3Patch * kQ3Patch;
    const uint32_t merged_dim = kQ3VHidden * kQ3Merge * kQ3Merge;
    fill(w.patch_embed_w, static_cast<size_t>(patch_dim) * kQ3VHidden, -0.3f, 0.3f);
    fill(w.patch_embed_b, kQ3VHidden, -0.1f, 0.1f);
    w.blocks.resize(kQ3VLayers);
    for (auto &B : w.blocks) {
        fill(B.norm1_w, kQ3VHidden, 0.7f, 1.3f);
        fill(B.norm1_b, kQ3VHidden, -0.1f, 0.1f);
        fill(B.norm2_w, kQ3VHidden, 0.7f, 1.3f);
        fill(B.norm2_b, kQ3VHidden, -0.1f, 0.1f);
        fill(B.q_w, static_cast<size_t>(kQ3VHidden) * kQ3VHidden, -0.3f, 0.3f);
        fill(B.k_w, static_cast<size_t>(kQ3VHidden) * kQ3VHidden, -0.3f, 0.3f);
        fill(B.v_w, static_cast<size_t>(kQ3VHidden) * kQ3VHidden, -0.3f, 0.3f);
        fill(B.o_w, static_cast<size_t>(kQ3VHidden) * kQ3VHidden, -0.3f, 0.3f);
        fill(B.q_b, kQ3VHidden, -0.1f, 0.1f);
        fill(B.k_b, kQ3VHidden, -0.1f, 0.1f);
        fill(B.v_b, kQ3VHidden, -0.1f, 0.1f);
        fill(B.o_b, kQ3VHidden, -0.1f, 0.1f);
        fill(B.q_norm, kQ3VHeadDim, 0.7f, 1.3f);
        fill(B.k_norm, kQ3VHeadDim, 0.7f, 1.3f);
        fill(B.fc1_w, static_cast<size_t>(kQ3VHidden) * kQ3VInter, -0.3f, 0.3f);
        fill(B.fc1_b, kQ3VInter, -0.1f, 0.1f);
        fill(B.fc2_w, static_cast<size_t>(kQ3VInter) * kQ3VHidden, -0.3f, 0.3f);
        fill(B.fc2_b, kQ3VHidden, -0.1f, 0.1f);
    }
    const auto fill_merger = [&](Qwen3SyntheticWeights::Merger &M) {
        fill(M.ln_w, merged_dim, 0.7f, 1.3f);
        fill(M.ln_b, merged_dim, -0.1f, 0.1f);
        fill(M.fc1_w, static_cast<size_t>(merged_dim) * kQ3MergerInter, -0.3f, 0.3f);
        fill(M.fc1_b, kQ3MergerInter, -0.1f, 0.1f);
        fill(M.fc2_w, static_cast<size_t>(kQ3MergerInter) * kQ3Hidden, -0.3f, 0.3f);
        fill(M.fc2_b, kQ3Hidden, -0.1f, 0.1f);
    };
    fill_merger(w.merger);
    w.ds_mergers.resize(2);
    for (auto &M : w.ds_mergers) { fill_merger(M); }
    fill(w.embed, static_cast<size_t>(kQ3Vocab) * kQ3Hidden, -0.6f, 0.6f);
    w.lm_head = w.embed;
    fill(w.final_norm, kQ3Hidden, 0.7f, 1.3f);
    w.layers.resize(kQ3Layers);
    const uint32_t qd = kQ3QHeads * kQ3HeadDim, kvd = kQ3KvHeads * kQ3HeadDim;
    for (auto &L : w.layers) {
        fill(L.input_norm, kQ3Hidden, 0.7f, 1.3f);
        fill(L.post_norm, kQ3Hidden, 0.7f, 1.3f);
        fill(L.q_w, static_cast<size_t>(kQ3Hidden) * qd, -0.3f, 0.3f);
        fill(L.k_w, static_cast<size_t>(kQ3Hidden) * kvd, -0.3f, 0.3f);
        fill(L.v_w, static_cast<size_t>(kQ3Hidden) * kvd, -0.3f, 0.3f);
        fill(L.o_w, static_cast<size_t>(qd) * kQ3Hidden, -0.3f, 0.3f);
        fill(L.q_norm, kQ3HeadDim, 0.7f, 1.3f);
        fill(L.k_norm, kQ3HeadDim, 0.7f, 1.3f);
        fill(L.gate_w, static_cast<size_t>(kQ3Hidden) * kQ3Inter, -0.3f, 0.3f);
        fill(L.up_w, static_cast<size_t>(kQ3Hidden) * kQ3Inter, -0.3f, 0.3f);
        fill(L.down_w, static_cast<size_t>(kQ3Inter) * kQ3Hidden, -0.3f, 0.3f);
    }
    return w;
}

// ---------------------------------------------------------------------------
// Scripted deterministic decoder (for deterministic-token / stop tests).
// Same mechanism as the Phase 3 scripted decoder: a dominant one-hot
// embedding + identity LM head makes greedy decoding an exact token state
// machine; the (shrunk) transformer blocks add only bounded corrections.
// REQUIRES vocab <= kQ3Hidden.
// ---------------------------------------------------------------------------
inline void make_qwen3_scripted_decoder(Qwen3SyntheticWeights &w, luisa::span<const uint32_t> next) {
    LUISA_ASSERT(next.size() == kQ3Vocab, "scripted next map must cover the vocabulary");
    LUISA_ASSERT(kQ3Vocab <= kQ3Hidden, "scripted decoder needs vocab <= hidden");
    w.tied = false;
    constexpr float A = 1.0e3f;
    w.embed.assign(static_cast<size_t>(kQ3Vocab) * kQ3Hidden, 0.0f);
    w.lm_head.assign(static_cast<size_t>(kQ3Vocab) * kQ3Hidden, 0.0f);
    for (uint32_t t = 0; t < kQ3Vocab; t++) {
        w.embed[static_cast<size_t>(t) * kQ3Hidden + next[t]] = A;
        w.lm_head[static_cast<size_t>(t) * kQ3Hidden + t] = A;
    }
    w.final_norm.assign(kQ3Hidden, 1.0f);
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
    }
    // Shrink the vision path so the image features cannot dominate the
    // scripted text embedding (the scripted tests use a blank image).
    const auto shrinkv = [](luisa::vector<float> &v) {
        for (auto &x : v) { x *= 1.0e-3f; }
    };
    shrinkv(w.patch_embed_w);
    shrinkv(w.patch_embed_b);
    for (auto &B : w.blocks) {
        shrinkv(B.q_w); shrinkv(B.k_w); shrinkv(B.v_w); shrinkv(B.o_w);
        shrinkv(B.q_b); shrinkv(B.k_b); shrinkv(B.v_b); shrinkv(B.o_b);
        shrinkv(B.fc1_w); shrinkv(B.fc1_b); shrinkv(B.fc2_w); shrinkv(B.fc2_b);
    }
    const auto shrinkm = [&](Qwen3SyntheticWeights::Merger &M) {
        shrinkv(M.fc1_w); shrinkv(M.fc1_b); shrinkv(M.fc2_w); shrinkv(M.fc2_b);
    };
    shrinkm(w.merger);
    for (auto &M : w.ds_mergers) { shrinkm(M); }
}

// ---------------------------------------------------------------------------
// Package writing + weight upload (exercises the real Phase 1/2 path).
// ---------------------------------------------------------------------------
[[nodiscard]] inline luisa::vector<std::byte> q3_f32_bytes(luisa::span<const float> v) {
    luisa::vector<std::byte> out(v.size() * 4);
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

[[nodiscard]] inline PackageManifest qwen3_synthetic_manifest(luisa::string_view component) {
    PackageManifest m;
    m.model = "qwen3_vl_2b_instruct";// synthetic test artifact; NOT a real model claim
    m.recipe_version = "1";
    m.checkpoint_revision = "synthetic-rev";
    m.config_fingerprint = "synthetic-fp";
    m.license = "test";
    m.producer = "ocr_qwen3_test";
    m.producer_version = "1";
    m.preprocessing_version = "1";
    m.tokenizer_version = "1";
    m.output_parser_version = "1";
    m.component = luisa::string{component};
    return m;
}

// Write the synthetic Qwen3-VL tensors to TWO .lcocr packages (vision and
// decoder components) under the given tensor-name map. Linear weights use
// layout tag `kn` ([K,N]); the LM head is omitted when tied.
[[nodiscard]] inline Status write_qwen3_synthetic_packages(const luisa::filesystem::path &vision_path,
                                                           const luisa::filesystem::path &decoder_path,
                                                           const Qwen3SyntheticWeights &w,
                                                           const Qwen3VlTensorNames &names) {
    const uint32_t patch_dim = 3u * kQ3Patch * kQ3Patch;
    const uint32_t merged_dim = kQ3VHidden * kQ3Merge * kQ3Merge;
    const uint32_t qd = kQ3QHeads * kQ3HeadDim, kvd = kQ3KvHeads * kQ3HeadDim;
    // Vision package.
    {
        auto writer = PackageWriter::create(vision_path);
        if (!writer.ok()) { return writer.status; }
        auto &wr = *writer.value;
        const auto add = [&wr](const luisa::string &name, luisa::vector<uint64_t> shape,
                               TensorLayout layout, luisa::span<const float> data) -> Status {
            PackageWriter::TensorSpec spec;
            spec.name = name;
            spec.encoding = TensorEncoding::fp32;
            spec.layout = layout;
            spec.shape = std::move(shape);
            spec.payload = q3_f32_bytes(data);
            return wr->add_tensor(spec);
        };
        if (auto s = add(names.patch_embed_weight(), {patch_dim, kQ3VHidden}, TensorLayout::kn, w.patch_embed_w); !s.ok()) { return s; }
        if (auto s = add(names.patch_embed_bias(), {kQ3VHidden}, TensorLayout::row_major, w.patch_embed_b); !s.ok()) { return s; }
        for (uint32_t l = 0; l < w.blocks.size(); l++) {
            const auto &B = w.blocks[l];
            if (auto s = add(names.block_norm1_weight(l), {kQ3VHidden}, TensorLayout::row_major, B.norm1_w); !s.ok()) { return s; }
            if (auto s = add(names.block_norm1_bias(l), {kQ3VHidden}, TensorLayout::row_major, B.norm1_b); !s.ok()) { return s; }
            if (auto s = add(names.block_norm2_weight(l), {kQ3VHidden}, TensorLayout::row_major, B.norm2_w); !s.ok()) { return s; }
            if (auto s = add(names.block_norm2_bias(l), {kQ3VHidden}, TensorLayout::row_major, B.norm2_b); !s.ok()) { return s; }
            if (auto s = add(names.block_q_proj(l), {kQ3VHidden, kQ3VHidden}, TensorLayout::kn, B.q_w); !s.ok()) { return s; }
            if (auto s = add(names.block_k_proj(l), {kQ3VHidden, kQ3VHidden}, TensorLayout::kn, B.k_w); !s.ok()) { return s; }
            if (auto s = add(names.block_v_proj(l), {kQ3VHidden, kQ3VHidden}, TensorLayout::kn, B.v_w); !s.ok()) { return s; }
            if (auto s = add(names.block_o_proj(l), {kQ3VHidden, kQ3VHidden}, TensorLayout::kn, B.o_w); !s.ok()) { return s; }
            if (auto s = add(names.block_q_bias(l), {kQ3VHidden}, TensorLayout::row_major, B.q_b); !s.ok()) { return s; }
            if (auto s = add(names.block_k_bias(l), {kQ3VHidden}, TensorLayout::row_major, B.k_b); !s.ok()) { return s; }
            if (auto s = add(names.block_v_bias(l), {kQ3VHidden}, TensorLayout::row_major, B.v_b); !s.ok()) { return s; }
            if (auto s = add(names.block_o_bias(l), {kQ3VHidden}, TensorLayout::row_major, B.o_b); !s.ok()) { return s; }
            if (auto s = add(names.block_q_norm(l), {kQ3VHeadDim}, TensorLayout::row_major, B.q_norm); !s.ok()) { return s; }
            if (auto s = add(names.block_k_norm(l), {kQ3VHeadDim}, TensorLayout::row_major, B.k_norm); !s.ok()) { return s; }
            if (auto s = add(names.block_fc1(l), {kQ3VHidden, kQ3VInter}, TensorLayout::kn, B.fc1_w); !s.ok()) { return s; }
            if (auto s = add(names.block_fc1_bias(l), {kQ3VInter}, TensorLayout::row_major, B.fc1_b); !s.ok()) { return s; }
            if (auto s = add(names.block_fc2(l), {kQ3VInter, kQ3VHidden}, TensorLayout::kn, B.fc2_w); !s.ok()) { return s; }
            if (auto s = add(names.block_fc2_bias(l), {kQ3VHidden}, TensorLayout::row_major, B.fc2_b); !s.ok()) { return s; }
        }
        const auto add_merger = [&](const luisa::string &prefix, const Qwen3SyntheticWeights::Merger &M) -> Status {
            if (auto s = add(Qwen3VlTensorNames::merger_ln_weight(prefix), {merged_dim}, TensorLayout::row_major, M.ln_w); !s.ok()) { return s; }
            if (auto s = add(Qwen3VlTensorNames::merger_ln_bias(prefix), {merged_dim}, TensorLayout::row_major, M.ln_b); !s.ok()) { return s; }
            if (auto s = add(Qwen3VlTensorNames::merger_fc1(prefix), {merged_dim, kQ3MergerInter}, TensorLayout::kn, M.fc1_w); !s.ok()) { return s; }
            if (auto s = add(Qwen3VlTensorNames::merger_fc1_bias(prefix), {kQ3MergerInter}, TensorLayout::row_major, M.fc1_b); !s.ok()) { return s; }
            if (auto s = add(Qwen3VlTensorNames::merger_fc2(prefix), {kQ3MergerInter, kQ3Hidden}, TensorLayout::kn, M.fc2_w); !s.ok()) { return s; }
            if (auto s = add(Qwen3VlTensorNames::merger_fc2_bias(prefix), {kQ3Hidden}, TensorLayout::row_major, M.fc2_b); !s.ok()) { return s; }
            return Status::success();
        };
        if (auto s = add_merger(names.merger_prefix(), w.merger); !s.ok()) { return s; }
        for (uint32_t i = 0; i < w.ds_mergers.size(); i++) {
            if (auto s = add_merger(names.deepstack_prefix(i), w.ds_mergers[i]); !s.ok()) { return s; }
        }
        if (auto s = wr->finish(qwen3_synthetic_manifest("vision")); !s.ok()) { return s; }
    }
    // Decoder package.
    {
        auto writer = PackageWriter::create(decoder_path);
        if (!writer.ok()) { return writer.status; }
        auto &wr = *writer.value;
        const auto add = [&wr](const luisa::string &name, luisa::vector<uint64_t> shape,
                               TensorLayout layout, luisa::span<const float> data) -> Status {
            PackageWriter::TensorSpec spec;
            spec.name = name;
            spec.encoding = TensorEncoding::fp32;
            spec.layout = layout;
            spec.shape = std::move(shape);
            spec.payload = q3_f32_bytes(data);
            return wr->add_tensor(spec);
        };
        const auto &tn = names.text;
        if (auto s = add(tn.embed_tokens(), {kQ3Vocab, kQ3Hidden}, TensorLayout::row_major, w.embed); !s.ok()) { return s; }
        if (auto s = add(tn.final_norm(), {kQ3Hidden}, TensorLayout::row_major, w.final_norm); !s.ok()) { return s; }
        if (!w.tied) {
            luisa::vector<float> head_t(static_cast<size_t>(kQ3Hidden) * kQ3Vocab);
            for (uint32_t v = 0; v < kQ3Vocab; v++) {
                for (uint32_t h = 0; h < kQ3Hidden; h++) {
                    head_t[static_cast<size_t>(h) * kQ3Vocab + v] = w.lm_head[static_cast<size_t>(v) * kQ3Hidden + h];
                }
            }
            if (auto s = add(tn.lm_head(), {kQ3Hidden, kQ3Vocab}, TensorLayout::kn, head_t); !s.ok()) { return s; }
        }
        for (uint32_t l = 0; l < w.layers.size(); l++) {
            const auto &L = w.layers[l];
            if (auto s = add(tn.input_layernorm(l), {kQ3Hidden}, TensorLayout::row_major, L.input_norm); !s.ok()) { return s; }
            if (auto s = add(tn.post_attention_layernorm(l), {kQ3Hidden}, TensorLayout::row_major, L.post_norm); !s.ok()) { return s; }
            if (auto s = add(tn.q_proj(l), {kQ3Hidden, qd}, TensorLayout::kn, L.q_w); !s.ok()) { return s; }
            if (auto s = add(tn.k_proj(l), {kQ3Hidden, kvd}, TensorLayout::kn, L.k_w); !s.ok()) { return s; }
            if (auto s = add(tn.v_proj(l), {kQ3Hidden, kvd}, TensorLayout::kn, L.v_w); !s.ok()) { return s; }
            if (auto s = add(tn.o_proj(l), {qd, kQ3Hidden}, TensorLayout::kn, L.o_w); !s.ok()) { return s; }
            if (auto s = add(tn.q_norm(l), {kQ3HeadDim}, TensorLayout::row_major, L.q_norm); !s.ok()) { return s; }
            if (auto s = add(tn.k_norm(l), {kQ3HeadDim}, TensorLayout::row_major, L.k_norm); !s.ok()) { return s; }
            if (auto s = add(tn.gate_proj(l), {kQ3Hidden, kQ3Inter}, TensorLayout::kn, L.gate_w); !s.ok()) { return s; }
            if (auto s = add(tn.up_proj(l), {kQ3Hidden, kQ3Inter}, TensorLayout::kn, L.up_w); !s.ok()) { return s; }
            if (auto s = add(tn.down_proj(l), {kQ3Inter, kQ3Hidden}, TensorLayout::kn, L.down_w); !s.ok()) { return s; }
        }
        if (auto s = wr->finish(qwen3_synthetic_manifest("decoder")); !s.ok()) { return s; }
    }
    return Status::success();
}

// Upload both components into a WeightStore (sync).
[[nodiscard]] inline Status upload_qwen3_synthetic_weights(compute::Device &device, compute::Stream &stream,
                                                           const luisa::filesystem::path &vision_path,
                                                           const luisa::filesystem::path &decoder_path,
                                                           WeightStore &store) {
    {
        auto reader = PackageReader::open(vision_path);
        if (!reader.ok()) { return reader.status; }
        if (auto s = store.upload_component_sync(device, stream, **reader.value, "vision"); !s.ok()) { return s; }
    }
    {
        auto reader = PackageReader::open(decoder_path);
        if (!reader.ok()) { return reader.status; }
        if (auto s = store.upload_component_sync(device, stream, **reader.value, "decoder"); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Independent CPU oracle (composed from Phase 2 host references).
// ---------------------------------------------------------------------------

// Vision oracle for ONE image. Returns the patch embeddings, the residual
// stream after each vision block, the main merger features, and the DeepStack
// merger features (in deepstack order).
struct Qwen3VisionOracle {
    luisa::vector<float> patches;// [patch_count, patch_dim]
    luisa::vector<float> patch_embed;// [patch_count, vhidden]
    luisa::vector<luisa::vector<float>> block_out;// per block [patch_count, vhidden]
    luisa::vector<float> features;// [merged_count, hidden]
    luisa::vector<luisa::vector<float>> ds_features;// per injection [merged_count, hidden]
};

// Spatial merge oracle: [T, H] -> [T/(m*m), H*m*m], row-major m x m blocks.
[[nodiscard]] inline luisa::vector<float> oracle_spatial_merge(luisa::span<const float> x,
                                                               uint32_t T, uint32_t H,
                                                               uint32_t tokens_w, uint32_t m) {
    const uint32_t mm = m * m;
    const uint32_t out_tokens = T / mm;
    const uint32_t out_hidden = H * mm;
    const uint32_t blocks_w = tokens_w / m;
    luisa::vector<float> out(static_cast<size_t>(out_tokens) * out_hidden);
    for (uint32_t ot = 0; ot < out_tokens; ot++) {
        const uint32_t bx = ot % blocks_w;
        const uint32_t by = ot / blocks_w;
        for (uint32_t sub = 0; sub < mm; sub++) {
            const uint32_t sy = sub / m, sx = sub % m;
            const uint32_t src_token = (by * m + sy) * tokens_w + (bx * m + sx);
            for (uint32_t d = 0; d < H; d++) {
                out[static_cast<size_t>(ot) * out_hidden + sub * H + d] =
                    x[static_cast<size_t>(src_token) * H + d];
            }
        }
    }
    return out;
}

// One merger oracle: spatial merge + LayerNorm + fc1(GELU-erf) + fc2.
[[nodiscard]] inline Status oracle_merger(const Qwen3SyntheticWeights::Merger &M,
                                          luisa::span<const float> x, uint32_t patch_rows,
                                          uint32_t grid_w, luisa::vector<float> &out) {
    const uint32_t merged_rows = patch_rows / (kQ3Merge * kQ3Merge);
    const uint32_t merged_dim = kQ3VHidden * kQ3Merge * kQ3Merge;
    auto merged = oracle_spatial_merge(x, patch_rows, kQ3VHidden, grid_w, kQ3Merge);
    luisa::vector<float> norm;
    auto s = ref_layer_norm(merged, M.ln_w, M.ln_b, merged_rows, merged_dim, kQ3Eps, norm);
    if (!s.ok()) { return s; }
    luisa::vector<float> fc1;
    s = ref_gemm(norm, M.fc1_w, M.fc1_b, merged_rows, kQ3MergerInter, merged_dim, RefGemmEpilogue::bias_gelu_erf, fc1);
    if (!s.ok()) { return s; }
    return ref_gemm(fc1, M.fc2_w, M.fc2_b, merged_rows, kQ3Hidden, kQ3MergerInter, RefGemmEpilogue::bias, out);
}

[[nodiscard]] inline Status oracle_vision(const Qwen3SyntheticWeights &w,
                                          luisa::span<const float> pixels_chw,
                                          const Qwen3VlPackedImage &packed,
                                          Qwen3VisionOracle &out) {
    const uint32_t patch_dim = 3u * kQ3Patch * kQ3Patch;
    const uint32_t rows = static_cast<uint32_t>(packed.patch_count);
    // Patchify: CHW -> [rows, patch_dim] in (c, py, px) order.
    out.patches.resize(static_cast<size_t>(rows) * patch_dim);
    const uint32_t H = packed.resized_height, W = packed.resized_width;
    for (uint32_t ph = 0; ph < packed.grid_h; ph++) {
        for (uint32_t pw = 0; pw < packed.grid_w; pw++) {
            const uint32_t t = ph * packed.grid_w + pw;
            for (uint32_t c = 0; c < 3u; c++) {
                for (uint32_t py = 0; py < kQ3Patch; py++) {
                    for (uint32_t px = 0; px < kQ3Patch; px++) {
                        const uint32_t y = ph * kQ3Patch + py;
                        const uint32_t x = pw * kQ3Patch + px;
                        const uint32_t d = (c * kQ3Patch + py) * kQ3Patch + px;
                        out.patches[static_cast<size_t>(t) * patch_dim + d] =
                            pixels_chw[(static_cast<size_t>(c) * H + y) * W + x];
                    }
                }
            }
        }
    }
    // Patch embed.
    auto s = ref_gemm(out.patches, w.patch_embed_w, w.patch_embed_b, rows, kQ3VHidden, patch_dim,
                      RefGemmEpilogue::bias, out.patch_embed);
    if (!s.ok()) { return s; }
    // Vision positions (height, width) + axis map.
    auto positions = qwen3_vl_vision_positions(packed);
    auto axis = qwen3_vl_vision_axis_of_pair(qwen3_vl_synthetic_config().vision);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kQ3VHeadDim));
    RefAttentionMask mask;
    mask.kind = RefMaskKind::bidirectional;
    luisa::vector<float> x = out.patch_embed;
    // Use the ACTUAL weight block count (a truncated subset is valid for the
    // isolated first-block test); the deepstack selection is bounded by it.
    const uint32_t num_vblocks = static_cast<uint32_t>(w.blocks.size());
    out.block_out.resize(num_vblocks);
    for (uint32_t l = 0; l < num_vblocks; l++) {
        const auto &B = w.blocks[l];
        luisa::vector<float> norm, q, k, v, qn, kn, qr, kr, attn, after_o, norm2, fc1, fc2;
        s = ref_layer_norm(x, B.norm1_w, B.norm1_b, rows, kQ3VHidden, kQ3Eps, norm);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm, B.q_w, B.q_b, rows, kQ3VHidden, kQ3VHidden, RefGemmEpilogue::bias, q);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm, B.k_w, B.k_b, rows, kQ3VHidden, kQ3VHidden, RefGemmEpilogue::bias, k);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm, B.v_w, B.v_b, rows, kQ3VHidden, kQ3VHidden, RefGemmEpilogue::bias, v);
        if (!s.ok()) { return s; }
        // QK-Norm: per-head RMSNorm over head_dim.
        s = ref_rms_norm(q, B.q_norm, rows * kQ3VHeads, kQ3VHeadDim, kQ3Eps, qn);
        if (!s.ok()) { return s; }
        s = ref_rms_norm(k, B.k_norm, rows * kQ3VHeads, kQ3VHeadDim, kQ3Eps, kn);
        if (!s.ok()) { return s; }
        // 2D axial RoPE (adjacent pairs).
        s = ref_rope_axial(qn, positions, axis, rows, kQ3VHeads, kQ3VHeadDim, 2u, kQ3Theta, qr);
        if (!s.ok()) { return s; }
        s = ref_rope_axial(kn, positions, axis, rows, kQ3VHeads, kQ3VHeadDim, 2u, kQ3Theta, kr);
        if (!s.ok()) { return s; }
        s = ref_attention(qr, kr, v, rows, rows, kQ3VHeads, kQ3VHeads, kQ3VHeadDim, scale, 0.0f, mask, attn);
        if (!s.ok()) { return s; }
        // o-proj + bias + residual.
        luisa::vector<float> o_proj;
        s = ref_gemm(attn, B.o_w, B.o_b, rows, kQ3VHidden, kQ3VHidden, RefGemmEpilogue::bias, o_proj);
        if (!s.ok()) { return s; }
        s = ref_residual_add(x, o_proj, after_o);
        if (!s.ok()) { return s; }
        s = ref_layer_norm(after_o, B.norm2_w, B.norm2_b, rows, kQ3VHidden, kQ3Eps, norm2);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm2, B.fc1_w, B.fc1_b, rows, kQ3VInter, kQ3VHidden, RefGemmEpilogue::bias_gelu_erf, fc1);
        if (!s.ok()) { return s; }
        luisa::vector<float> fc2_proj;
        s = ref_gemm(fc1, B.fc2_w, B.fc2_b, rows, kQ3VHidden, kQ3VInter, RefGemmEpilogue::bias, fc2_proj);
        if (!s.ok()) { return s; }
        s = ref_residual_add(after_o, fc2_proj, x);
        if (!s.ok()) { return s; }
        out.block_out[l] = x;
        // DeepStack mergers consume the selected block outputs (bounded by the
        // actual ds_merger count so a truncated subset never reads OOB).
        const auto cfg = qwen3_vl_synthetic_config();
        for (uint32_t di = 0; di < cfg.deepstack.count() && di < w.ds_mergers.size(); di++) {
            if (cfg.deepstack.vision_layers[di] == l) {
                if (out.ds_features.size() <= di) { out.ds_features.resize(cfg.deepstack.count()); }
                s = oracle_merger(w.ds_mergers[di], x, rows, packed.grid_w, out.ds_features[di]);
                if (!s.ok()) { return s; }
            }
        }
    }
    return oracle_merger(w.merger, x, rows, packed.grid_w, out.features);
}

// Text decoder oracle (Qwen3: QK-Norm, no QKV bias, interleaved MRoPE,
// DeepStack injection). `position_ids` is [rows*3]; `deepstack` are the
// projected image features [mm_count, hidden] per injection; the injection
// range is [mm_start, mm_start+mm_count).
struct Qwen3TextOracle {
    luisa::vector<float> embeddings;// [rows, hidden] after mm replacement
    luisa::vector<luisa::vector<float>> layer_out;// per layer [rows, hidden]
    luisa::vector<luisa::vector<float>> ds_injected;// per injection [rows, hidden] (after inject)
    luisa::vector<float> final_norm;
    luisa::vector<float> logits;// [rows, vocab]
    luisa::vector<luisa::vector<float>> k_cache, v_cache;// per layer [total, kvd]
};

[[nodiscard]] inline Status oracle_qwen3_block(const Qwen3SyntheticWeights &w, uint32_t layer,
                                               luisa::span<const float> x, uint32_t rows, uint32_t pos0,
                                               luisa::span<const int32_t> position_ids,
                                               luisa::span<const uint32_t> axis_of_pair,
                                               luisa::vector<float> &k_cache, luisa::vector<float> &v_cache,
                                               luisa::vector<float> &x_out) {
    const auto &L = w.layers[layer];
    const uint32_t qd = kQ3QHeads * kQ3HeadDim, kvd = kQ3KvHeads * kQ3HeadDim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kQ3HeadDim));
    luisa::vector<float> norm, q, k, v, qn, kn, qr, kr, attn, after_o, norm2, gate, up, mlp;
    auto s = ref_rms_norm(x, L.input_norm, rows, kQ3Hidden, kQ3Eps, norm);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, L.q_w, {}, rows, qd, kQ3Hidden, RefGemmEpilogue::none, q);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, L.k_w, {}, rows, kvd, kQ3Hidden, RefGemmEpilogue::none, k);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, L.v_w, {}, rows, kvd, kQ3Hidden, RefGemmEpilogue::none, v);
    if (!s.ok()) { return s; }
    s = ref_rms_norm(q, L.q_norm, rows * kQ3QHeads, kQ3HeadDim, kQ3Eps, qn);
    if (!s.ok()) { return s; }
    s = ref_rms_norm(k, L.k_norm, rows * kQ3KvHeads, kQ3HeadDim, kQ3Eps, kn);
    if (!s.ok()) { return s; }
    // Interleaved MRoPE: positions for the NEW rows only (position_ids always
    // covers exactly the `rows` new tokens, indexed from 0 -- for prefill that
    // is the whole prompt at pos0=0; for a decode step it is the single new
    // token, independent of the absolute KV write position pos0).
    luisa::vector<int32_t> pos(position_ids.begin(),
                               position_ids.begin() + static_cast<size_t>(rows) * 3u);
    s = ref_rope_interleaved(qn, pos, axis_of_pair, rows, kQ3QHeads, kQ3HeadDim, 3u, kQ3Theta, qr);
    if (!s.ok()) { return s; }
    s = ref_rope_interleaved(kn, pos, axis_of_pair, rows, kQ3KvHeads, kQ3HeadDim, 3u, kQ3Theta, kr);
    if (!s.ok()) { return s; }
    s = ref_kv_append(kr, v, k_cache, v_cache, rows, kvd, pos0, pos0 + rows);
    if (!s.ok()) { return s; }
    RefAttentionMask mask;
    mask.kind = RefMaskKind::causal;
    const uint32_t k_len = pos0 + rows;
    s = ref_attention(qr, k_cache, v_cache, rows, k_len, kQ3QHeads, kQ3KvHeads, kQ3HeadDim, scale, 0.0f, mask, attn);
    if (!s.ok()) { return s; }
    s = ref_gemm(attn, L.o_w, x, rows, kQ3Hidden, qd, RefGemmEpilogue::residual, after_o);
    if (!s.ok()) { return s; }
    s = ref_rms_norm(after_o, L.post_norm, rows, kQ3Hidden, kQ3Eps, norm2);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm2, L.gate_w, {}, rows, kQ3Inter, kQ3Hidden, RefGemmEpilogue::none, gate);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm2, L.up_w, {}, rows, kQ3Inter, kQ3Hidden, RefGemmEpilogue::none, up);
    if (!s.ok()) { return s; }
    s = ref_gated_act(gate, up, 0u, mlp);
    if (!s.ok()) { return s; }
    return ref_gemm(mlp, L.down_w, after_o, rows, kQ3Hidden, kQ3Inter, RefGemmEpilogue::residual, x_out);
}

[[nodiscard]] inline Status oracle_qwen3_forward(const Qwen3SyntheticWeights &w,
                                                 luisa::span<const uint32_t> tokens,
                                                 luisa::span<const int32_t> position_ids,
                                                 luisa::span<const float> mm_features,
                                                 uint32_t mm_start, uint32_t mm_count,
                                                 luisa::span<const luisa::vector<float>> deepstack,
                                                 uint32_t kv_prefix,
                                                 luisa::vector<luisa::vector<float>> &k_caches,
                                                 luisa::vector<luisa::vector<float>> &v_caches,
                                                 Qwen3TextOracle &result) {
    const uint32_t rows = static_cast<uint32_t>(tokens.size());
    const uint32_t kvd = kQ3KvHeads * kQ3HeadDim;
    const auto cfg = qwen3_vl_synthetic_config();
    auto axis = cfg.mrope.axis_of_pair();
    luisa::vector<int32_t> ids(tokens.begin(), tokens.end());
    auto s = ref_embedding_lookup(w.embed, ids, kQ3Vocab, kQ3Hidden, result.embeddings);
    if (!s.ok()) { return s; }
    if (!mm_features.empty()) {
        luisa::vector<float> replaced;
        s = ref_mm_scatter(result.embeddings, mm_features, rows, kQ3Hidden, mm_start, replaced);
        if (!s.ok()) { return s; }
        result.embeddings = std::move(replaced);
    }
    // Use the ACTUAL weight layer count (a truncated subset is valid for the
    // isolated first-layer test).
    const uint32_t num_layers = static_cast<uint32_t>(w.layers.size());
    if (k_caches.empty()) {
        k_caches.resize(num_layers);
        v_caches.resize(num_layers);
    }
    const uint32_t total = kv_prefix + rows;
    for (uint32_t l = 0; l < num_layers; l++) {
        k_caches[l].resize(static_cast<size_t>(total) * kvd);
        v_caches[l].resize(static_cast<size_t>(total) * kvd);
    }
    result.layer_out.resize(num_layers);
    result.ds_injected.resize(deepstack.size());
    luisa::vector<float> x = result.embeddings;
    for (uint32_t l = 0; l < num_layers; l++) {
        luisa::vector<float> x_out;
        s = oracle_qwen3_block(w, l, x, rows, kv_prefix, position_ids, axis, k_caches[l], v_caches[l], x_out);
        if (!s.ok()) { return s; }
        x = std::move(x_out);
        result.layer_out[l] = x;
        // DeepStack injection AFTER the selected layer (image rows only).
        for (uint32_t di = 0; di < deepstack.size(); di++) {
            if (cfg.deepstack.decoder_layers[di] != l) { continue; }
            luisa::vector<float> injected;
            s = ref_deepstack_inject(x, deepstack[di], rows, kQ3Hidden, mm_start, injected);
            if (!s.ok()) { return s; }
            x = std::move(injected);
            result.ds_injected[di] = x;
        }
    }
    s = ref_rms_norm(x, w.final_norm, rows, kQ3Hidden, kQ3Eps, result.final_norm);
    if (!s.ok()) { return s; }
    const luisa::vector<float> &head = w.tied ? w.embed : w.lm_head;
    luisa::vector<float> lm_head_t(static_cast<size_t>(kQ3Hidden) * kQ3Vocab);
    for (uint32_t v = 0; v < kQ3Vocab; v++) {
        for (uint32_t h = 0; h < kQ3Hidden; h++) {
            lm_head_t[static_cast<size_t>(h) * kQ3Vocab + v] = head[static_cast<size_t>(v) * kQ3Hidden + h];
        }
    }
    s = ref_gemm(result.final_norm, lm_head_t, {}, rows, kQ3Vocab, kQ3Hidden, RefGemmEpilogue::none, result.logits);
    if (!s.ok()) { return s; }
    result.k_cache = k_caches;
    result.v_cache = v_caches;
    return Status::success();
}

// Greedy argmax over a logits row (lowest-index tie-break).
[[nodiscard]] inline uint32_t oracle_argmax(luisa::span<const float> logits) {
    uint32_t best = 0u;
    float best_v = logits[0];
    for (size_t i = 1; i < logits.size(); i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

// Temp dir helper (RAII cleanup).
struct Qwen3TempDir {
    luisa::filesystem::path path;
    Qwen3TempDir() {
        path = luisa::filesystem::temp_directory_path() /
               ("lcocr_qwen3_test_" + luisa::format("{}", reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~Qwen3TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

}// namespace luisa::ocr::test
