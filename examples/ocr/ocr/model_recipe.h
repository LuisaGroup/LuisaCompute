#pragma once
// OCR engine -- model identity and recipe contracts (Phase 0).
//
// A ModelRecipe describes everything a later phase needs to load and run one
// exact checkpoint: identity/revision/fingerprint, preprocessing semantics,
// component graph contracts, tensor mappings, prompt/stop/parser contracts,
// capability matrix, context limits, license, and support state. Phase 0
// freezes these host-side contracts; NO recipe here is validated against an
// official artifact yet, so every registered recipe is `planned` and every
// graph-contract field is a contract to verify, not a verified fact.
//
// Host-only header: no DSL/runtime device includes.
#include <cstdint>

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/pp_ocr_size.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Model identity
// ---------------------------------------------------------------------------

// PpOcrSize (the PP-OCRv6 size variant enum) lives in models/pp_ocr_size.h so
// device TUs can use it without pulling this header's host-only AttentionMaskKind.
// It is re-exported here for existing callers.

// Semantic model identity, independent of arbitrary Hugging Face `model_type`
// strings. Covers all 15 requirement targets (PP-OCRv6 counts as one family
// with three size variants => 17 selectable targets).
enum class ModelId : uint32_t {
    unknown = 0,
    pp_ocrv6,// size selects tiny/small/medium
    paddle_ocr_vl_1_6,
    ovis_ocr2,
    tele_ocr,
    glm_ocr,
    mineru_2_5_pro_2605_1_2b,
    hunyuan_ocr_1_5,
    surya_2_5,
    open_doc_0_1b,
    fire_red_ocr,
    qwen3_vl_2b_instruct,
    deepseek_ocr_2,
    dots_mocr,
    got_ocr2_0,
    olm_ocr_2,
};

[[nodiscard]] luisa::string_view model_id_name(ModelId id) noexcept;
// Exact inverse of model_id_name for known names; unknown names map to
// ModelId::unknown (fail-closed callers reject it).
[[nodiscard]] ModelId model_id_from_name(luisa::string_view name) noexcept;

// Execution-path family. These distinctions are load-bearing: hybrid GDN
// models are not KV-only transformers, DeepSeek is not a generic causal mask,
// and Qwen3-VL must not collapse into Qwen2.5-VL.
enum class ArchitectureFamily : uint32_t {
    unknown = 0,
    cnn,        // detector + CTC recognizer (PP-OCRv6)
    dense_vlm,  // ViT/connector + dense autoregressive decoder
    layout_vlm, // layout front end + region-routed VLM
    hybrid_gdn, // Gated DeltaNet + periodic full-attention layers
    custom_vlm, // model-specific vision/position conventions (Hunyuan)
    moe_vlm,    // routed-expert decoder with custom visual/query masks
};

[[nodiscard]] luisa::string_view architecture_family_name(ArchitectureFamily family) noexcept;

// ---------------------------------------------------------------------------
// Capability / mode contracts
// ---------------------------------------------------------------------------

enum class RecognizeMode : uint32_t {
    plain = 0,// canonical plain text
    spotting, // text + source coordinates (requires validated grounding)
    markdown, // structured document Markdown
    extract,  // schema/prompt-directed JSON extraction
};

[[nodiscard]] luisa::string_view recognize_mode_name(RecognizeMode mode) noexcept;
[[nodiscard]] luisa::optional<RecognizeMode> recognize_mode_from_name(luisa::string_view name) noexcept;

// Numerical precisions used for weight storage, activations/accumulation and
// decode-cache storage. `fp32` is the portable correctness baseline.
// `automatic` is a policy sentinel, NOT a concrete storage format: it resolves
// to the recipe's validated default, else FP32 (see resolve_precision). Result
// metadata must only ever carry concrete values, never `automatic`.
enum class Precision : uint32_t {
    fp32 = 0,
    fp16,
    bf16,
    int8,
    int4,// grouped symmetric/asymmetric weights, dequantized at read
    automatic,// policy sentinel: recipe validated default, else FP32
};

[[nodiscard]] luisa::string_view precision_name(Precision precision) noexcept;
[[nodiscard]] luisa::optional<Precision> precision_from_name(luisa::string_view name) noexcept;

// Support state of a recipe. Phase 0: every recipe is `planned`. A model may
// only be marked `validated` after its exact checkpoint revision, package
// version, backends, precisions, input kinds and output modes pass load-time
// validation, layered golden tests, deterministic end-to-end tests and
// resource-limit tests (plan §1).
enum class SupportState : uint32_t {
    planned = 0,   // contract written down; nothing implemented/verified yet
    load_only,     // package loads and validates; inference not qualified
    experimental,  // runs only under the explicit unsafe override; unvalidated
    validated,     // qualified for its exact declared support matrix
};

[[nodiscard]] luisa::string_view support_state_name(SupportState state) noexcept;

// What a recipe declares it intends to support (intent, not a support claim).
// Capability negotiation validates requests against these declarations even
// while the model is unimplemented, so later phases inherit the same rules.
struct ModelCapabilities {
    luisa::vector<RecognizeMode> modes;
    // BCP-47-ish language tags ("zh", "en", "multi", ...). Empty = unspecified.
    // Matching is EXACT: "multi" is a literal tag a recipe may declare, never a
    // wildcard that accepts arbitrary hints. The list may only expand when the
    // corresponding language is actually validated.
    luisa::vector<luisa::string> languages;
    // DECLARED INTENT that this model can emit source coordinates. It is not a
    // support claim: actual validated grounding is gated separately by the
    // recipe's SupportState (and the qualification gates in MODELS.md). While a
    // recipe is `planned`, this flag only drives capability negotiation.
    bool has_source_coordinates{false};
    // Conventional OCR (CTC) or explicitly validated spotting/layout scores.
    // Generative VLMs must keep this false; results then report
    // has_confidence == false instead of a pseudo-probability.
    bool has_calibrated_confidence{false};
    // Region/color prompt guidance (GOT-style) declared by the recipe.
    bool supports_region_guidance{false};
    bool supports_color_guidance{false};
    // extract requires a schema/prompt when the recipe says so.
    bool extract_requires_schema{true};
    // Multi-page composition (cross-page reading order/merging).
    bool supports_multi_page{false};
};

// ---------------------------------------------------------------------------
// Graph contracts (to verify against pinned official artifacts)
// ---------------------------------------------------------------------------

enum class AttentionMaskKind : uint32_t {
    unspecified = 0,
    bidirectional,
    causal,
    windowed,
    prefix_causal,
    custom_block_mask,// e.g. DeepSeek visual tokens + causal-flow queries
};

[[nodiscard]] luisa::string_view attention_mask_kind_name(AttentionMaskKind kind) noexcept;

enum class PositionalEncodingKind : uint32_t {
    unspecified = 0,
    none,
    rope_1d,
    rope_2d,
    mrope,
    interleaved_mrope,
    xd_rope,
    learned_absolute,
    sinusoidal_absolute,
    relative_bias,
};

[[nodiscard]] luisa::string_view positional_encoding_kind_name(PositionalEncodingKind kind) noexcept;

enum class CacheLayoutKind : uint32_t {
    unspecified = 0,
    none,          // CNN pipelines
    contiguous_kv, // correctness baseline
    paged_kv,      // optional later optimization
    gdn_state,     // recurrent + conv ring state (+ periodic full-attn KV)
};

[[nodiscard]] luisa::string_view cache_layout_kind_name(CacheLayoutKind kind) noexcept;

// Preprocessing semantics frozen per recipe: everything that must match the
// official processor bit-for-bit before model work starts.
struct PreprocessingContract {
    luisa::string version;// recipe preprocessing version tag
    luisa::string resize_semantics;// e.g. "aspect_bilinear_antialias", "native_resolution"
    luisa::string color_order;// "rgb" | "bgr"
    luisa::vector<float> mean;// per-channel normalization (empty = none)
    luisa::vector<float> std_dev;
    uint32_t patch_size{0};// 0 = not patch-based
    luisa::string tile_semantics;// "" = single view; else named tiling contract
    // Overview/crop pixel ceilings the official pipeline expects (0 = unset).
    uint64_t reference_max_overview_pixels{0};
    uint64_t reference_max_crop_pixels{0};
};

// One graph component (layout/detector/vision/connector/decoder/head).
// Dimensions are read from the checkpoint config at load time in later
// phases and checked against tensors; Phase 0 keeps them unset (0).
struct ComponentContract {
    luisa::string role;// "layout", "detector", "vision", "connector", "decoder", "head"
    luisa::string kind;// contract name, e.g. "lcnetv4", "navit", "qwen3_dense"
    uint32_t num_layers{0};
    uint32_t num_attention_heads{0};
    uint32_t num_kv_heads{0};
    uint32_t hidden_size{0};
    uint32_t intermediate_size{0};
    uint32_t num_experts{0};// MoE routed experts (0 = dense)
    uint32_t num_shared_experts{0};
    uint32_t top_k_experts{0};
    uint32_t gdn_state_size{0};// hybrid GDN per-head state (0 = not GDN)
    luisa::string normalization;// "layernorm" | "rmsnorm" | ...
    AttentionMaskKind attention_mask{AttentionMaskKind::unspecified};
    PositionalEncodingKind positional_encoding{PositionalEncodingKind::unspecified};
    luisa::string notes;// free-form contract notes to verify
};

// Expected tensor mapping entry: name -> shape/layout/transform contract.
struct TensorMapping {
    luisa::string name;// canonical tensor name in the .lcocr package
    bool required{true};
    luisa::vector<uint64_t> expected_shape;// 0 entries = dynamic dimension
    luisa::string layout;// e.g. "kn" for packed linear weights [K,N]
    luisa::string transform;// transpose/fold/merge rule id ("" = verbatim)
    bool quantization_eligible{false};
};

// Prompt/generation contract: validated against official apply_chat_template
// in later phases; ids stay unset (0/empty) until pinned.
struct PromptContract {
    luisa::string template_id;
    luisa::vector<uint32_t> special_token_ids;
    luisa::vector<uint32_t> stop_token_ids;
    luisa::vector<luisa::string> stop_strings;
    // Greedy is the engine default; recipe overrides must be explicit.
    bool greedy_default{true};
    luisa::string output_parser_id;// "ctc", "markdown", "json_extract", ...
};

// ---------------------------------------------------------------------------
// Recipe
// ---------------------------------------------------------------------------

// Identity of one exact artifact. Phase 0 registers planned recipes with
// EMPTY revision/fingerprint strings: no official checkpoint has been pinned
// or inspected yet, and inventing hashes/revisions is forbidden.
struct RecipeIdentity {
    luisa::string recipe_version;// engine recipe-format version, e.g. "0"
    luisa::string checkpoint_revision;// pinned upstream revision ("" = unpinned)
    luisa::string config_fingerprint;// exact expected config hash ("" = unset)
    luisa::string package_format_version;// .lcocr format version this recipe targets
};

struct ModelRecipe {
    ModelId id{ModelId::unknown};
    PpOcrSize pp_size{PpOcrSize::none};// only for pp_ocrv6
    ArchitectureFamily family{ArchitectureFamily::unknown};
    RecipeIdentity identity;
    SupportState support{SupportState::planned};

    PreprocessingContract preprocessing;
    luisa::vector<ComponentContract> components;
    CacheLayoutKind cache_layout{CacheLayoutKind::unspecified};
    luisa::vector<TensorMapping> tensor_mappings;
    PromptContract prompt;
    ModelCapabilities capabilities;

    // Official context/token ceilings (0 = unknown/unset until pinned).
    uint32_t max_context_tokens{0};
    uint32_t max_output_tokens{0};
    luisa::string license;// SPDX identifier or note; "" = unreviewed
    luisa::string notes;// contract notes / caveats to verify

    // Stable selector string: model name plus ":size" for PP-OCRv6 variants.
    [[nodiscard]] luisa::string target_name() const;
    // True when identity is pinned (revision AND fingerprint non-empty).
    [[nodiscard]] bool identity_pinned() const noexcept {
        return !identity.checkpoint_revision.empty() && !identity.config_fingerprint.empty();
    }
};

// Structural validation of a recipe declaration (host-side, no artifacts).
// Checks identity/family consistency, PP-OCRv6 size rules, mode declarations,
// and contract completeness flags. Does NOT contact any registry or model.
[[nodiscard]] Status validate_recipe(const ModelRecipe &recipe);

// Capability negotiation for one request against a recipe's declared
// capabilities. Rejects: unknown modes, spotting without declared source
// coordinates, extract without required schema/prompt, languages not exactly
// declared ("multi" is a literal tag, not a wildcard), region/color guidance
// the recipe does not declare. `language` may be empty (unspecified). This runs
// before any device work, in every phase. Declared capability is intent; actual
// validated support is gated separately by SupportState.
[[nodiscard]] Status validate_request_capabilities(
    const ModelRecipe &recipe, RecognizeMode mode,
    luisa::string_view language, bool has_schema_or_prompt,
    bool requests_region_guidance, bool requests_color_guidance);

// Resolve a precision policy value to a concrete storage format. `automatic`
// (and any unrecognized sentinel) resolves to `recipe_default`; a concrete
// value passes through unchanged. `recipe_default` itself must be concrete;
// callers pass the recipe's validated default or FP32. Result metadata must
// only ever contain the resolved concrete value.
[[nodiscard]] Precision resolve_precision(Precision requested, Precision recipe_default) noexcept;

}// namespace luisa::ocr
