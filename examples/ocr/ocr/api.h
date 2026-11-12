#pragma once
// OCR engine -- stable public inference API (Phase 0 contract freeze).
//
// This header defines the request/result surface that later phases implement
// without breaking Phase 0 tests. Phase 0 ships argument validation and
// fail-closed session creation only: no model is validated, so recognize()
// never produces OCR output and Session::create() always refuses after
// validating its arguments. No mock output exists.
//
// Host-only header: no DSL/runtime device includes. Device binding is added
// by later phases through additional overloads, not by changing these types.
#include <cstdint>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/model_recipe.h"
#include "ocr/model_registry.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

// Explicit numerical policy. Each component defaults to `automatic`, which
// resolves to the recipe's validated default, else FP32 (the portable
// correctness baseline). A concrete value is an explicit opt-in that must be
// validated per recipe before it may be selected; `automatic` never silently
// changes model semantics. Resolution uses resolve_precision(); result
// metadata only ever carries the resolved concrete values.
struct PrecisionPolicy {
    Precision weights{Precision::automatic};
    Precision activations{Precision::automatic};
    Precision cache{Precision::automatic};
};

// Decode-cache policy. `automatic` resolves to contiguous KV (the correctness
// baseline); paged KV is a later, independently validated optimization.
enum class CachePolicy : uint32_t {
    automatic = 0,
    contiguous,
    paged,
};

// Host offload is never implicit: it must be explicitly requested and is
// reported in metadata/diagnostics when used.
enum class OffloadPolicy : uint32_t {
    disabled = 0,
    explicit_host,
};

// CUDA-graph capture policy (DSL-only command lists; later phase).
enum class GraphPolicy : uint32_t {
    disabled = 0,
    automatic,
    enabled,
};

// Optimized-kernel fast paths (fusions/native shaders; later phases). Every
// fast path needs a validated A/B arm; `automatic` may only pick paths that
// passed parity for the exact recipe/precision/shape.
enum class FastPathPolicy : uint32_t {
    disabled = 0,
    automatic,
    enabled,
};

// Sampling: greedy is the default. Non-greedy sampling requires an explicit
// temperature > 0 and is only available where the recipe validated it.
struct SamplingOptions {
    bool greedy{true};
    float temperature{0.0f};// ignored when greedy
    float top_p{1.0f};
    uint32_t top_k{0};// 0 = disabled
    uint64_t seed{0};// deterministic seed for non-greedy sampling
};

// Resource limits (MemoryLimits) are defined in ocr/runtime/memory_budget.h
// and re-exported here so the public API and the device-side budget share one
// type without a circular include.

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

struct RecognizeOptions {
    RecognizeMode mode{RecognizeMode::plain};

    // Separate pixel bounds for the overview/layout pass and for region
    // crops. 0 = engine default. `max_pixels` is a compatibility ceiling:
    // when non-zero it additionally caps BOTH effective bounds (min wins).
    uint64_t max_overview_pixels{0};
    uint64_t max_crop_pixels{0};
    uint64_t max_pixels{0};// compatibility bound

    // Batching limits (0 is invalid; defaults are conservative engine values,
    // not model claims).
    uint32_t max_patches_per_vision_batch{4096};
    uint32_t max_regions_per_batch{4};
    uint32_t max_decode_batch{1};

    // Output token budgets (0 is invalid).
    uint32_t max_output_tokens_per_region{2048};
    uint32_t max_output_tokens_per_page{8192};

    // Prompt/schema for extract and prompt-directed modes.
    luisa::string prompt;
    luisa::string schema;// JSON schema text for extract

    // Region/color guidance for models that declare support (e.g. GOT).
    luisa::string region_guidance;
    luisa::string color_guidance;

    // Optional BCP-47-ish language hint ("" = unspecified).
    luisa::string language;

    SamplingOptions sampling;
    PrecisionPolicy precision;
    CachePolicy cache_policy{CachePolicy::automatic};
    OffloadPolicy offload_policy{OffloadPolicy::disabled};
    GraphPolicy graph_policy{GraphPolicy::disabled};
    FastPathPolicy fast_path_policy{FastPathPolicy::disabled};

    MemoryLimits memory;
    // Wall-time budget in milliseconds; 0 = no explicit limit. Must be finite
    // and non-negative when set.
    double max_wall_time_ms{0.0};
    // Cancellation: polled at safe boundaries; returning true stops the
    // request with StatusCode::cancelled. Empty = never cancelled.
    luisa::function<bool()> cancel_callback;
};

// One input page. PDF pages are rasterized outside the inference process in
// later phases; Phase 0 only validates the description.
struct PageInput {
    luisa::filesystem::path path;
    uint32_t source_page_index{0};
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

// Source coordinates are in the ORIGINAL page pixel space; later phases must
// preserve them through overview resize, layout crop, perspective warp, tile
// composition and cross-page merging.
struct SourcePoint {
    float x{0.0f};
    float y{0.0f};
};

struct SourceBox {
    float x0{0.0f};
    float y0{0.0f};
    float x1{0.0f};
    float y1{0.0f};

    [[nodiscard]] bool empty() const noexcept { return x1 <= x0 || y1 <= y0; }
};

struct SourceRegion {
    uint32_t page_index{0};
    SourceBox box;
    // Optional polygon (empty = use box). Points are in page pixel space.
    luisa::vector<SourcePoint> polygon;
};

enum class BlockType : uint32_t {
    text = 0,
    title,
    table,
    formula,
    figure,
    header,
    footer,
    list,
    other,
};

[[nodiscard]] luisa::string_view block_type_name(BlockType type) noexcept;

// One recognized line. Confidence is present only for conventional OCR or an
// explicitly validated spotting/layout confidence source; generative VLMs
// report has_confidence == false (never a pseudo-probability).
struct OcrLine {
    luisa::string text;
    SourceRegion source;
    uint32_t reading_order{0};
    bool has_confidence{false};
    float confidence{0.0f};
};

struct OcrBlock {
    BlockType type{BlockType::text};
    luisa::string content;// canonical content per type/mode (Markdown/LaTeX/JSON)
    SourceRegion source;
    uint32_t reading_order{0};
    bool has_confidence{false};
    float confidence{0.0f};
};

// Provenance of an explicit router/fallback decision. Phase 0 performs NO
// routing and NO implicit fallback: these fields stay at their defaults
// (used=false). A later, explicitly configured router fills them so a fallback
// model is always visible in metadata, never silent.
struct RouterMetadata {
    bool used{false};
    luisa::string router_id;// configured router identity ("" = none)
    ModelId initial_model{ModelId::unknown};
    ModelId final_model{ModelId::unknown};
    luisa::string reason;// why a switch/fallback happened ("" = none)
};

// Provenance of a result. Phase 0 never produces a result, but the schema is
// frozen so later phases cannot silently drop identity/precision information.
// The precision fields must contain RESOLVED CONCRETE values only (never
// Precision::automatic); the *_policy fields record what was actually applied.
struct ModelMetadata {
    ModelId model_id{ModelId::unknown};
    PpOcrSize pp_size{PpOcrSize::none};
    luisa::string target_name;
    luisa::string checkpoint_revision;
    luisa::string package_hash;
    luisa::string recipe_version;
    luisa::string backend;
    Precision weight_precision{Precision::fp32};
    Precision activation_precision{Precision::fp32};
    Precision cache_precision{Precision::fp32};
    // Actual policy state applied to this request (resolved, not requested).
    CachePolicy cache_policy{CachePolicy::automatic};
    OffloadPolicy offload_policy{OffloadPolicy::disabled};
    GraphPolicy graph_policy{GraphPolicy::disabled};
    FastPathPolicy fast_path_policy{FastPathPolicy::disabled};
    // Effective caps actually applied to this request.
    uint64_t max_overview_pixels{0};
    uint64_t max_crop_pixels{0};
    uint32_t max_output_tokens_per_region{0};
    uint32_t max_output_tokens_per_page{0};
    luisa::vector<luisa::string> enabled_optimizations;
    SupportState support_state{SupportState::planned};
    bool validated{false};
    // Explicit router/fallback provenance (defaults: no routing, no fallback).
    RouterMetadata router;
};

struct PerformanceCounters {
    double load_ms{0.0};
    double preprocess_ms{0.0};
    double vision_ms{0.0};
    double decode_ms{0.0};
    double postprocess_ms{0.0};
    double total_ms{0.0};
    uint64_t pages_processed{0};
    uint64_t regions_processed{0};
    uint64_t patches_processed{0};
    uint64_t output_tokens{0};
    uint64_t peak_device_bytes{0};
    uint64_t peak_host_bytes{0};
    uint64_t resident_weight_bytes{0};
};

struct RecognizeResult {
    // Canonical text: plain text, or serialized Markdown/JSON per mode.
    luisa::string text;
    luisa::vector<OcrLine> lines;
    luisa::vector<OcrBlock> blocks;
    ModelMetadata metadata;
    luisa::vector<Diagnostic> diagnostics;
    PerformanceCounters counters;
};

// ---------------------------------------------------------------------------
// Validation (host-side, no device/model needed)
// ---------------------------------------------------------------------------

// Structural validation of request options independent of any model. Checks
// batching/token budgets, sampling consistency, time budget finiteness, and
// policy enum validity. Mode/language/schema capability checks against a
// concrete recipe live in validate_request_capabilities().
[[nodiscard]] Status validate_recognize_options(const RecognizeOptions &options);

// Effective pixel bounds after applying the compatibility ceiling.
struct EffectivePixelBounds {
    uint64_t overview{0};
    uint64_t crop{0};
};
[[nodiscard]] EffectivePixelBounds effective_pixel_bounds(const RecognizeOptions &options) noexcept;

}// namespace luisa::ocr
