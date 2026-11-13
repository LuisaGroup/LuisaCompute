#pragma once
// OCR engine -- layout/region scheduling infrastructure (Phase 6).
//
// DEVICE-SAFE header: plain geometry/plan types plus the token-budgeted region
// scheduler and the detector-output cache. It deliberately includes NO api.h /
// model_recipe.h (whose host-only AttentionMaskKind clashes with the ops enum
// in device TUs) and NO device/DSL headers, so both host and device TUs may
// include it. The public RecognizeResult is assembled by a host-only bridge
// from the plain LayoutDocument below.
//
// Scheduling contract (plan §7 / Phase 6):
//   * ONE overview/layout pass per page produces the region tasks.
//   * Crops are sorted by approximate patch/token count (descending) and packed
//     into vision/prefill batches under max_patches_per_vision_batch and
//     max_regions_per_batch.
//   * Overview/crop pixel budgets and per-region/per-page output-token budgets
//     are SEPARATE limits, preflighted before any work. The per-region output
//     cap is enforced by the recognizer; the per-page cap is enforced by the
//     orchestrator DURING accumulation (a visible output_truncated diagnostic).
//   * Regions are decoded INDEPENDENTLY (no prefix sharing across unrelated
//     crops); results are restored to the original page/reading order.
//   * Detector outputs can be cached for recognition retry under an explicit
//     content/geometry key (LayoutCache); a cache hit never re-runs layout.
//
// EXECUTION (Phase 11): plan_vision_batches is a real approximate-cost packing
// plan. P7 recognizers record ragged multi-crop vision graphs, split again by
// exact packed-patch capacity when needed; Paddle/OpenDoc still loop per crop.
// Every recognizer uses a fresh KV cache and independent batch-1 text decode.
// plan_decode_batches / max_decode_batch remain PLANNING-ONLY: no batched decode
// executor exists. Batch-vs-1 tests assert ordered equivalence of actual outputs.
//
// Determinism: plan_vision_batches / plan_decode_batches are pure, stable sorts
// with explicit tie-breaking, so region batching and batch-1 produce equivalent
// ordered outputs under the declared deterministic policy.
#include <cstdint>

#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/output/db_postprocess.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Plain block/line types (mirror the public api.h schema WITHOUT pulling it).
// The host bridge maps these to OcrBlock/OcrLine/SourceRegion.
// ---------------------------------------------------------------------------

// Mirrors api.h BlockType. Kept in the same order so the bridge is a static_cast.
enum class LayoutBlockType : uint32_t {
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

[[nodiscard]] luisa::string_view layout_block_type_name(LayoutBlockType type) noexcept;

// Output mode for layout composition. Mirrors api.h RecognizeMode values.
enum class LayoutOutputMode : uint32_t {
    plain = 0,
    spotting,
    markdown,
    extract,
};

// ---------------------------------------------------------------------------
// Region tasks + scheduling limits
// ---------------------------------------------------------------------------

// One region to recognize, in ORIGINAL source-page coordinates. `reading_order`
// is the layout reading-order index WITHIN its page (assigned by the layout
// pass). `region_id` is a DOCUMENT-GLOBAL identifier: the orchestrator assigns
// it monotonically across pages in input order, so it is unique across the whole
// request (a page's regions get a contiguous block). Within one page it is dense
// 0-based only for the FIRST page. `approx_patches` and `approx_tokens` are the
// scheduler's cost estimates (patch count for vision, prompt+output token
// estimate for decode), computed host-side from the crop geometry and the model
// packing contract.
struct RegionTask {
    uint32_t region_id{0};
    uint32_t page_index{0};
    uint32_t reading_order{0};
    LayoutBlockType block_type{LayoutBlockType::text};
    uint32_t approx_patches{0};
    uint32_t approx_tokens{0};
    output::DbRect source_box;
    luisa::vector<output::DbPoint> polygon;
};

// Separate overview/crop/patch/output budgets plus cancellation/time/memory.
// 0 pixel/token limits mean "engine default / unset" except the batch caps,
// which must be nonzero. All limits are preflighted before any device work.
struct SchedulerLimits {
    uint32_t max_patches_per_vision_batch{4096};
    uint32_t max_regions_per_batch{4};
    // PLANNING-ONLY (Phase 6): plan_decode_batches chunks under this cap, but no
    // batched decode executor exists yet; actual decode execution is batch-1.
    uint32_t max_decode_batch{1};
    // Phase 11 opt-in single-sequence CUDA graph replay, forwarded through the
    // orchestrator to each generative region recognizer. Default off.
    bool enable_cuda_graph{false};
    uint64_t max_overview_pixels{0};
    uint64_t max_crop_pixels{0};
    // Per-region output-token cap. The orchestrator forwards this to the
    // recognizer's RegionRecognizeOptions::max_output_tokens (single source of
    // truth); a generative recognizer stops at the cap and marks `truncated`.
    uint32_t max_output_tokens_per_region{2048};
    // Per-page output-token cap, enforced DURING accumulation: once the page's
    // accumulated output tokens reach it, later regions are not recognized and a
    // visible `output_truncated` diagnostic is emitted.
    uint32_t max_output_tokens_per_page{8192};
    double max_wall_time_ms{0.0};
    luisa::function<bool()> cancel_callback;
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

// One packed vision/prefill batch: indices into the task array, plus the total
// approximate patch count (<= max_patches_per_vision_batch).
struct RegionBatch {
    luisa::vector<uint32_t> region_indices;
    uint64_t total_patches{0};
};

// The token-budgeted region scheduler. Pure planning: no device, no allocation.
class RegionScheduler {
public:
    explicit RegionScheduler(SchedulerLimits limits) noexcept : _limits(limits) {}

    [[nodiscard]] const SchedulerLimits &limits() const noexcept { return _limits; }

    // Sort `tasks` by approximate patch count DESCENDING (stable tie-break on
    // region_id) and pack into vision/prefill batches under the patch and region
    // caps. A single task whose approx_patches exceeds the per-batch patch cap
    // is rejected (out_of_memory) rather than silently split -- the caller must
    // shrink the crop or raise the cap deliberately. Empty input -> empty plan.
    [[nodiscard]] Result<luisa::vector<RegionBatch>> plan_vision_batches(
        luisa::span<const RegionTask> tasks) const;

    // Pack region indices into decode batches under max_decode_batch, preserving
    // the given order (the caller passes the vision-batch order). Empty -> empty.
    [[nodiscard]] Result<luisa::vector<RegionBatch>> plan_decode_batches(
        luisa::span<const uint32_t> region_indices) const;

    // Restore results to the original page/reading order. `results` are indexed
    // by region_id (results[region_id]); the output is sorted by (page_index,
    // reading_order, region_id). Returns the sorted region_id sequence.
    [[nodiscard]] static Result<luisa::vector<uint32_t>> restore_order(
        luisa::span<const RegionTask> tasks);

    // Preflight the per-page output-token budget against the sum of per-region
    // caps implied by `region_count`. Fails (out_of_memory) when the page budget
    // cannot hold even one token per region, so the caller can shrink the batch.
    [[nodiscard]] Status preflight_page_output_budget(uint32_t region_count) const;

private:
    SchedulerLimits _limits;
};

// ---------------------------------------------------------------------------
// Detector-output cache (recognition retry)
// ---------------------------------------------------------------------------

// Explicit content/geometry key for cached detector output. `content_hash`
// identifies the page pixels; `geometry_hash` identifies the layout geometry
// (source size + detector input/transform). Both are caller-computed stable
// hashes; the cache never recomputes them from device state.
struct LayoutCacheKey {
    uint64_t content_hash{0};
    uint64_t geometry_hash{0};

    [[nodiscard]] bool operator==(const LayoutCacheKey &other) const noexcept {
        return content_hash == other.content_hash && geometry_hash == other.geometry_hash;
    }
};

// Cached layout result for one page: the detected regions in reading order.
// Stored by value so a retry never aliases a released workspace.
struct CachedLayout {
    luisa::vector<RegionTask> regions;
};

// Bounded detector-output cache keyed by content/geometry. A hit returns the
// cached regions WITHOUT re-running layout; a miss returns nullptr. The cache
// is FIFO-bounded (max_entries) so a long document cannot grow it unboundedly.
class LayoutCache {
public:
    explicit LayoutCache(uint32_t max_entries = 64u) noexcept : _max_entries(max_entries == 0u ? 1u : max_entries) {}

    // Store (or overwrite) the layout for `key`. Evicts the oldest entry when
    // full. Returns the number of entries after insertion.
    uint32_t store(LayoutCacheKey key, CachedLayout layout);
    // Find a cached layout; nullptr on miss. The returned pointer is valid
    // until the next store/eviction.
    [[nodiscard]] const CachedLayout *find(LayoutCacheKey key) const noexcept;
    [[nodiscard]] uint32_t size() const noexcept { return static_cast<uint32_t>(_entries.size()); }
    void clear() noexcept { _entries.clear(); }

private:
    struct Entry {
        LayoutCacheKey key;
        CachedLayout layout;
    };
    uint32_t _max_entries;
    luisa::vector<Entry> _entries;// insertion order (FIFO eviction)
};

// ---------------------------------------------------------------------------
// Plain run / composition results (device-safe; host bridge maps to api.h)
// ---------------------------------------------------------------------------

// One region's RAW recognizer output, before type-specific composition. For a
// generative region recognizer this is the detokenized text plus the generated
// token ids; for a CTC region recognizer it is the decoded text (tokens empty).
// `truncated`/`hit_eos` carry the decode-stop provenance. NO confidence is
// produced for generated content; `has_confidence` is set only for an
// explicitly validated detector/CTC score and labeled internal/test.
struct RawRegionOutput {
    uint32_t region_id{0};
    luisa::string text;
    luisa::vector<uint32_t> tokens;
    bool truncated{false};
    bool hit_eos{false};
    // Detector/CTC score (internal/test only). Generative content keeps this
    // false; the public bridge never emits a pseudo-probability.
    bool has_confidence{false};
    float confidence{0.0f};
    // Approximate native patch count actually consumed (diagnostics/counters).
    uint32_t native_patches{0};
    uint32_t output_tokens{0};
};

// One region after composition: type-specific canonical content plus the
// source geometry and reading order. `content` is Markdown/LaTeX/JSON/plain per
// type and mode; for a table it is the validated HTML/OTSL source.
struct LayoutRegionResult {
    uint32_t region_id{0};
    uint32_t page_index{0};
    uint32_t reading_order{0};
    LayoutBlockType block_type{LayoutBlockType::text};
    luisa::string content;
    output::DbRect source_box;
    luisa::vector<output::DbPoint> polygon;
    bool has_confidence{false};
    float confidence{0.0f};
    // Heading level for title blocks (1..6); 0 otherwise.
    uint32_t heading_level{0};
};

struct LayoutPageResult {
    uint32_t page_index{0};
    uint32_t source_width{0};
    uint32_t source_height{0};
    luisa::vector<LayoutRegionResult> regions;// in reading order
};

// Sort page indices by source page_index ASCENDING (stable: equal page_index
// keeps input order). The orchestrator calls this before composition so a
// shuffled input page span still yields document-ordered pages. Returns the
// sorted permutation of [0, pages.size()). Pure host planning.
[[nodiscard]] luisa::vector<uint32_t> layout_sort_pages_by_index(
    luisa::span<const LayoutPageResult> pages);

// The full plain run result: per-page composed regions plus counters and
// diagnostics. The host bridge maps this to RecognizeResult.
struct LayoutRunResult {
    luisa::vector<LayoutPageResult> pages;
    luisa::vector<Diagnostic> diagnostics;
    uint64_t regions_processed{0};
    uint64_t patches_processed{0};
    uint64_t output_tokens{0};
    uint64_t layout_cache_hits{0};
    // Resident weight bytes (layout detector + region recognizer).
    uint64_t weight_bytes{0};
    // Fixed per-request device workspace bytes (layout + recognizer arenas).
    uint64_t workspace_bytes{0};
    // PEAK per-region KV-cache bytes actually allocated during the run (0 for a
    // CTC-only recognizer, which has no KV cache). The orchestrator tracks the
    // high-water mark reported by the recognizer.
    uint64_t kv_cache_peak_bytes{0};
    // Peak resident device bytes = weight_bytes + workspace_bytes +
    // kv_cache_peak_bytes. This is the honest peak the host bridge reports as
    // peak_device_bytes (it previously reported workspace alone).
    uint64_t peak_device_bytes{0};
};

// One composed line/block for the public result. Plain mirror of OcrLine /
// OcrBlock; the host bridge converts geometry to SourceRegion.
struct LayoutLine {
    luisa::string text;
    uint32_t page_index{0};
    uint32_t reading_order{0};
    output::DbRect source_box;
    luisa::vector<output::DbPoint> polygon;
    bool has_confidence{false};
    float confidence{0.0f};
};

struct LayoutBlock {
    LayoutBlockType type{LayoutBlockType::text};
    luisa::string content;
    uint32_t page_index{0};
    uint32_t reading_order{0};
    output::DbRect source_box;
    luisa::vector<output::DbPoint> polygon;
    bool has_confidence{false};
    float confidence{0.0f};
};

// The composed document: canonical text (plain/Markdown/JSON per mode) plus
// ordered lines/blocks. Produced by the host-only model compose functions.
struct LayoutDocument {
    luisa::string text;
    luisa::vector<LayoutLine> lines;
    luisa::vector<LayoutBlock> blocks;
    luisa::vector<Diagnostic> diagnostics;
};

// Stable FNV-1a 64-bit hashes for cache keys (content/geometry). Deterministic
// and portable; NOT a cryptographic digest (the cache is an optimization, not
// a security boundary). Exposed so tests and callers compute identical keys.
[[nodiscard]] uint64_t layout_hash_bytes(luisa::span<const uint8_t> bytes) noexcept;
[[nodiscard]] uint64_t layout_hash_geometry(uint32_t source_width, uint32_t source_height,
                                            uint32_t det_width, uint32_t det_height,
                                            uint32_t page_index) noexcept;

}// namespace luisa::ocr