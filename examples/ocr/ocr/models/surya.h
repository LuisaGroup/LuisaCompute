#pragma once
// OCR engine -- Surya graph contract (Phase 9). Host-safe / device-safe header.
//
// Surya is a HYBRID GDN OCR system with THREE distinct stages:
//   1. an EfficientViT-Segformer-style LINE DETECTOR (CNN backbone + DB-style
//      probability head) producing line polygons in SOURCE coordinates;
//   2. a ViT vision tower + spatial patch-merger connector applied per line crop;
//   3. a HYBRID decoder whose layer schedule EXPLICITLY mixes 18 GDN layers and
//      6 full GQA attention layers across 24 layers (full at every 4th layer).
//      It is NEVER a KV-only transformer.
//
// The full 24-layer / 18-GDN / 6-full schedule is a RECIPE-DRIVEN DECLARED
// CONTRACT: a small synthetic slice may scale the hidden / channel dimensions
// (and run fewer channels), but the full schedule is validated explicitly
// (surya_declared_decoder_config + surya_validate_schedule + signature gate).
// The executing graph reuses runtime/hybrid_gdn_executor.* (HybridVlmGraph) for
// vision + hybrid decode, and the shared CNN conv/activation/FPN/sigmoid ops +
// output/db_postprocess for the detector.
//
// CONFIDENCE (plan §10.2): the ONLY confidence carried is the DETECTOR mean
// probability inside the unclipped box (output::DbBox::score), labeled as a
// detector score. Generative line text carries NO pseudo-confidence. Polygons /
// blocks / HTML are emitted only where the declared reference defines them;
// Surya's synthetic contract emits line polygons + plain/spotting/markdown text
// (no fabricated table HTML).
//
// IDENTITY (plan §6.1): the outer `model_type` string is NEVER identity. A
// recipe match requires the EXACT tensor/config signature
// (surya_validate_signature). Real artifacts fail closed through
// surya_validate_real_artifact().
//
// This header pulls NO api.h/model_recipe.h (host AttentionMaskKind clash) and
// NO ops_attention.h / DSL headers, so both host and device TUs may include it.
//
// HONESTY (plan §1/§6): NO official Surya checkpoint has been pinned, inspected,
// or run. Every dimension / tensor name here is a DECLARED SYNTHETIC CONTRACT
// exercised by the synthetic vertical slice. The official gates are in
// MODELS.md. No RTX 4060 performance claim is made.
#include <cstdint>
#include <utility>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/output/db_postprocess.h"
#include "ocr/package.h"
#include "ocr/runtime/hybrid_gdn_config.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"
#include "ocr/tokenizer.h"// borrowed const Tokenizer* (host-only, no DSL/api.h)

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Detector contract (EfficientViT-Segformer-style; declared synthetic).
// ---------------------------------------------------------------------------
// One detector conv stage (NCHW). act: 0=identity, 1=relu, 2=relu6, 3=hard_swish
// (the shared CnnActKernel tags). The synthetic detector uses hard_swish
// (EfficientViT's activation) on the backbone and identity on the DB head.
struct SuryaConvStage {
    uint32_t in_c{0};
    uint32_t out_c{0};
    uint32_t k{0};
    uint32_t stride{0};
    uint32_t pad{0};
    uint32_t act{0};
    [[nodiscard]] Status validate() const;
};

struct SuryaDetectorConfig {
    uint32_t input_height{0};
    uint32_t input_width{0};
    // stem -> stage1 -> stage2 -> db_head (1x1 -> 1 channel logit -> sigmoid).
    SuryaConvStage stem;
    SuryaConvStage stage1;
    SuryaConvStage stage2;
    SuryaConvStage db_head;
    luisa::vector<float> mean{0.5f, 0.5f, 0.5f};
    luisa::vector<float> std_dev{0.5f, 0.5f, 0.5f};
    [[nodiscard]] Status validate() const;
};

// Detector tensor-name contract (component "detector"). Conv weights are stored
// [out_c, in_c, k, k] (row_major); biases [out_c].
struct SuryaDetectorTensorNames {
    luisa::string prefix{"detector."};
    [[nodiscard]] Status validate() const;
    [[nodiscard]] luisa::string stem_weight() const;
    [[nodiscard]] luisa::string stem_bias() const;
    [[nodiscard]] luisa::string stage1_weight() const;
    [[nodiscard]] luisa::string stage1_bias() const;
    [[nodiscard]] luisa::string stage2_weight() const;
    [[nodiscard]] luisa::string stage2_bias() const;
    [[nodiscard]] luisa::string db_weight() const;
    [[nodiscard]] luisa::string db_bias() const;
};

// ---------------------------------------------------------------------------
// Full Surya config
// ---------------------------------------------------------------------------
struct SuryaConfig {
    SuryaDetectorConfig detector;
    SuryaDetectorTensorNames detector_names;
    // Vision tower + connector + packing (P7 conventions). vision.text must
    // match the hybrid decoder hidden/vocab and use rope1d.
    P7SliceConfig vision;
    P7TensorNames vision_names;
    // Hybrid GDN decoder (explicit 18-GDN / 6-full schedule).
    HybridGdnExecutorConfig decoder;
    HybridGdnTensorNames decoder_names;
    output::DbPostprocessOptions db;
    uint32_t max_lines_per_page{16};
    uint32_t max_new_tokens{8};
    uint32_t max_seq_len{64};
    MemoryLimits memory;
    // BORROWED tokenizer used to detokenize every recognized line (and to apply
    // stop strings). LIFETIME CONTRACT: the caller owns it and it MUST outlive
    // BOTH SuryaExecutor::create() and EVERY recognize() call; the executor
    // stores only this non-owning pointer and never copies or frees it. It is
    // read-only (const) and may be shared across executors/threads only if the
    // caller guarantees the tokenizer itself is not mutated concurrently.
    //
    // REQUIRED for text output: Surya is a text recognizer, so a null tokenizer
    // is refused by create() (a text-capable run must never silently emit empty
    // content). Its vocabulary MUST cover decoder.decoder.vocab_size.
    const Tokenizer *tokenizer{nullptr};

    [[nodiscard]] Status validate() const;
};

// The DECLARED full Surya hybrid decoder contract: 24 layers, 18 GDN + 6 full
// (full at every 4th layer, 0-based indices 3,7,11,15,19,23). The hidden/head/
// intermediate values are the declared contract geometry (NOT a verified
// checkpoint). Used to validate the 24/18/6 schedule explicitly even when the
// executing slice is scaled.
[[nodiscard]] HybridGdnExecutorConfig surya_declared_decoder_config();

// Validate a hybrid layer schedule against the declared 24/18/6 contract.
[[nodiscard]] Status surya_validate_schedule(const luisa::vector<HybridLayerKind> &kinds);

// The SMALL SYNTHETIC Surya slice used by the vertical-slice tests. NOT a real
// checkpoint. Detector: a scaled 4-stage CNN (fewer channels) at a small input
// size. Vision: 2-layer ViT + spatial-merge merger. Decoder: a SCALED hybrid
// geometry (6 layers = 4 GDN + 2 full) with the SAME conventions as the full
// 24/18/6 contract. The full schedule is checked separately.
[[nodiscard]] SuryaConfig surya_synthetic_config();

// Strict Surya config parsing (synthetic contract schema). Fails closed on MoE
// markers and unknown structural fields; the layer schedule is REQUIRED.
[[nodiscard]] Result<SuryaConfig> parse_surya_config(luisa::string_view json);

// Full expected tensor inventory (detector + vision + connector + hybrid decoder).
[[nodiscard]] luisa::vector<ExpectedTensor> surya_expected_tensors(const SuryaConfig &config);
[[nodiscard]] Status surya_validate_tensors(luisa::span<const PackageTensor> detector_tensors,
                                            luisa::span<const PackageTensor> vision_tensors,
                                            luisa::span<const PackageTensor> decoder_tensors,
                                            const SuryaConfig &config);

// Exact tensor/config signature gate (outer model_type alone is never identity).
struct SuryaSignature {
    luisa::string checkpoint_revision;
    luisa::string config_fingerprint;
    luisa::vector<luisa::string> tensor_names;
};
[[nodiscard]] Status surya_validate_signature(const SuryaConfig &config, const SuryaSignature &signature);

// Real-artifact gate: ALWAYS fail-closed in Phase 9.
[[nodiscard]] Status surya_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                  luisa::string_view config_fingerprint);

// ---------------------------------------------------------------------------
// Memory preflight (host; no allocation).
// ---------------------------------------------------------------------------
struct SuryaMemoryReport {
    uint64_t decoder_weight_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t cache_bytes{0};
    uint64_t total_device_bytes{0};
    uint64_t available_device_bytes{0};
    bool fits{false};
};
[[nodiscard]] Result<SuryaMemoryReport> surya_preflight_decoder(
    const SuryaConfig &config, uint32_t max_seq_len, MemoryLimits limits);

// ---------------------------------------------------------------------------
// Plain results (device-free; the host compose maps them to LayoutDocument).
// ---------------------------------------------------------------------------
// One recognized line. Polygon/bounds are in ORIGINAL source coordinates.
// `confidence` is the DETECTOR mean probability (DbBox::score), NOT a generative
// probability; has_confidence is true only because the detector score is an
// explicitly defined reference quantity (plan §10.2).
struct SuryaLine {
    luisa::string text;
    luisa::vector<uint32_t> tokens;
    uint32_t page_index{0};
    uint32_t reading_order{0};
    output::DbRect bounds;
    luisa::vector<output::DbPoint> polygon;
    bool has_confidence{false};
    float confidence{0.0f};// detector mean probability
    bool truncated{false};
    bool hit_eos{false};
};

struct SuryaPageResult {
    uint32_t page_index{0};
    uint32_t source_width{0};
    uint32_t source_height{0};
    uint32_t det_map_width{0};
    uint32_t det_map_height{0};
    luisa::vector<float> det_probabilities;
    luisa::vector<output::DbBox> boxes;
    luisa::vector<SuryaLine> lines;// in reading order
    luisa::string text;
};

struct SuryaRunResult {
    luisa::vector<SuryaPageResult> pages;
    uint64_t weight_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t cache_bytes{0};
};

// One source page input: decoded HWC uint8 RGB pixels + source size + page index.
struct SuryaPageInput {
    luisa::vector<uint8_t> pixels;
    uint32_t source_width{0};
    uint32_t source_height{0};
    uint32_t page_index{0};
};

// The Surya device executor (detector -> line crops -> vision -> hybrid decode)
// lives in runtime/surya_executor.* (device-safe), mirroring the CNN architecture
// where the model contract is host-safe (this header) and the executor is a
// separate device TU. This keeps api.h/model_recipe.h out of every device TU.

// ---------------------------------------------------------------------------
// Host composition (host-only; defined in the .cpp). Surya supports plain +
// spotting + markdown; extract fails closed. Line confidence is the DETECTOR
// score (labeled); generative text carries no pseudo-confidence. No fabricated
// table HTML (the synthetic reference defines line polygons + text only).
// ---------------------------------------------------------------------------
[[nodiscard]] Result<LayoutDocument> surya_compose(const SuryaRunResult &run, LayoutOutputMode mode);

}// namespace luisa::ocr
