#pragma once
// OCR engine -- host-safe benchmark utility (options parsing, canonical
// workload identity, sample statistics, baseline comparison, JSON/CSV
// reporting). This is the measurement/reporting side of the benchmark
// harness: it never dispatches work itself, so a device runner constructs
// Rows and a Report and serializes them here.
//
// This header is HOST-ONLY: it must not include DSL/runtime device headers,
// ocr/api.h, or ocr/model_recipe.h. Device TUs include it alongside the ops
// headers, so nothing here may clash with device-side names (e.g.
// AttentionMaskKind in ocr/ops/ops_attention.h).
//
// Ordinary failures (bad arguments, unreadable baselines, non-finite samples)
// are reported through Status/Result values, never exceptions. Optional
// fields use luisa::optional; NaN/Inf is never a sentinel and is never
// emitted to JSON.
#include <cstdint>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Stable enum names + strict parsers (full-string, lowercase).
// ---------------------------------------------------------------------------

// Benchmark suites. `all` runs every suite in a fixed order.
enum class BenchmarkSuite : uint32_t {
    ops = 0,// operator microbenchmarks
    decode,  // autoregressive decode loops
    cnn,     // convolutional backbones
    layout,  // layout detection/compose
    gdn,     // gated delta net blocks
    moe,     // mixture-of-experts blocks
    model,   // whole-model end-to-end
    all,     // every suite above
};

// Input artifact class: synthetic (generated, always allowed) or real (pinned
// package+manifest; execution stays fail-closed until real artifacts are
// validated -- see check_real_execution).
enum class ArtifactClass : uint32_t {
    synthetic = 0,
    real,
};

// Cache state the run is required to measure under.
enum class BenchmarkCacheMode : uint32_t {
    miss = 0,// cold cache, every iteration reloads/refetches
    hit,     // warm cache, steady-state reuse
    disabled,// caching turned off entirely
};

// Per-row outcome. `ok` rows carry samples/statistics; `failed` rows carry a
// Status reason; `skipped` rows were not run (filtered out); `blocked` rows
// name an external blocker; `refused` rows are expected fail-closed outcomes;
// `invalid` rows produced data that cannot be summarized.
enum class RowStatus : uint32_t {
    ok = 0,
    failed,
    skipped,
    blocked,
    refused,
    invalid,
};

// Timed metrics. All are milliseconds; every one is optional per sample.
// record_ms: HOST command-list construction only (never a GPU-duration claim).
// submit_sync_ms: queue-idle commit through completed Stream::synchronize.
// request_ms: full public/executor operation including inherent sync/readback.
enum class BenchmarkMetric : uint32_t {
    record_ms = 0,
    submit_sync_ms,
    request_ms,
};

// Baseline comparison verdict after threshold + noise floor.
enum class ComparisonOutcome : uint32_t {
    not_significant = 0,
    regression, // current median slower than baseline beyond the threshold
    improvement,// current median faster than baseline beyond the threshold
};

[[nodiscard]] luisa::string_view benchmark_suite_name(BenchmarkSuite suite) noexcept;
[[nodiscard]] luisa::string_view artifact_class_name(ArtifactClass artifact_class) noexcept;
[[nodiscard]] luisa::string_view benchmark_cache_mode_name(BenchmarkCacheMode mode) noexcept;
[[nodiscard]] luisa::string_view row_status_name(RowStatus status) noexcept;
[[nodiscard]] luisa::string_view benchmark_metric_name(BenchmarkMetric metric) noexcept;
[[nodiscard]] luisa::string_view comparison_outcome_name(ComparisonOutcome outcome) noexcept;

[[nodiscard]] bool parse_benchmark_suite(luisa::string_view text, BenchmarkSuite &out) noexcept;
[[nodiscard]] bool parse_artifact_class(luisa::string_view text, ArtifactClass &out) noexcept;
[[nodiscard]] bool parse_benchmark_cache_mode(luisa::string_view text, BenchmarkCacheMode &out) noexcept;
[[nodiscard]] bool parse_on_off(luisa::string_view text, bool &out) noexcept;

// ---------------------------------------------------------------------------
// Benchmark options (strict UTF-8 argv parser).
// ---------------------------------------------------------------------------

// Fully parsed benchmark command line. backend (argv[1]) and suite (argv[2])
// are explicit positional arguments; `help` short-circuits everything else
// (no backend/suite required).
struct BenchmarkOptions {
    bool help{false};
    luisa::string backend;                 // argv[1]: backend plugin name
    BenchmarkSuite suite{BenchmarkSuite::ops};// argv[2]
    // Selectors (empty = no filtering).
    luisa::string target;
    luisa::string case_name; // --case
    luisa::string variant;
    luisa::string workload;
    ArtifactClass artifact_class{ArtifactClass::synthetic};
    uint32_t warmup{3};
    uint32_t iterations{20};
    uint32_t dispatch_batch{1};
    bool graph{false};
    bool fusion{false};
    bool cooperative_norm{true};
    BenchmarkCacheMode cache{BenchmarkCacheMode::miss};
    bool instrument{false};
    luisa::filesystem::path json_path;    // --json (empty = stdout)
    luisa::filesystem::path csv_path;     // --csv
    luisa::filesystem::path baseline_path;// --baseline
    // Provenance overrides (empty = capture from the environment).
    luisa::string source_revision;
    luisa::string source_tree_state;
    luisa::string gpu_name;
    luisa::string driver_version;
    // Real-artifact pinned inputs (required for --artifact-class real).
    luisa::filesystem::path package_path;
    luisa::filesystem::path manifest_path;
    // Repeatable --provenance key=value records (CUDA runtime, clocks, corpus...).
    luisa::vector<luisa::string> provenance;
    bool pretty{true};// --no-pretty disables
};

// Parse UTF-8 argument bytes (argv[0] is the program name and is skipped).
// Strict: unknown arguments, duplicate options, missing values, malformed
// numbers (std::from_chars, full-string), and out-of-range counts are all
// invalid_argument with the offending option named in `detail`. --help/-h
// anywhere short-circuits to success with `help` set. Path arguments are
// decoded with path_from_utf8 and rejected when not valid UTF-8.
[[nodiscard]] Result<BenchmarkOptions> parse_benchmark_options(int argc, const char *const *argv);

// Usage text (also used by --help).
[[nodiscard]] luisa::string benchmark_usage_text();

// Explicit fail-closed gate for REAL-artifact execution. Synthetic runs pass.
// Real runs validate their pinned inputs first (invalid_argument when
// --package/--manifest are missing) and then refuse with `unvalidated`:
// no real-artifact benchmark execution is validated yet, mirroring the
// Session fail-closed contract (see examples/ocr/QUALIFICATION.md).
[[nodiscard]] Status check_real_execution(const BenchmarkOptions &options);

// ---------------------------------------------------------------------------
// Workload identity.
// ---------------------------------------------------------------------------

struct KeyValue {
    luisa::string key;
    luisa::string value;
};

// Immutable description of one benchmark workload. `shape` and `options` hold
// every shape-/code-affecting option as key-value pairs; they are canonicalized
// (sorted by key) for identity, so input order does not matter. The counts are
// the ACTUAL work the run performed (pages/regions/patches/prompt/output/
// dispatches) plus byte accounting (flops, bytes moved, weight bytes).
struct WorkloadDescriptor {
    luisa::string id;      // stable workload id (unique per suite/target/case)
    BenchmarkSuite suite{BenchmarkSuite::ops};
    luisa::string target;  // model/op target name
    luisa::string case_name;
    luisa::string variant; // code-variant label (e.g. kernel flavor)
    ArtifactClass artifact_class{ArtifactClass::synthetic};
    uint64_t seed{0};
    luisa::vector<KeyValue> shape;  // shape-affecting options (canonicalized)
    luisa::vector<KeyValue> options;// code-affecting options (canonicalized)
    uint64_t pages{0};
    uint64_t regions{0};
    uint64_t patches{0};
    uint64_t prompt_tokens{0};
    uint64_t output_tokens{0};
    uint64_t dispatches{0};
    uint64_t flops{0};
    uint64_t bytes{0};
    uint64_t weight_bytes{0};
    // Explicit accounting convention for flops/bytes (dense upper bound,
    // actual allowed pairs, weight+activation traffic, etc.).
    luisa::string byte_accounting;
};

// Canonical deterministic identity string: every identity/shape/code-affecting
// field in a fixed order with length-prefixed values; shape/options pairs are
// sorted by (key, value) first, so permutations of the same pairs collide.
[[nodiscard]] luisa::string canonical_workload_key(const WorkloadDescriptor &workload);

// 64-bit FNV-1a over the canonical key. NON-CRYPTOGRAPHIC: identity/dedup
// only, never a security or integrity boundary.
[[nodiscard]] uint64_t workload_fingerprint64(const WorkloadDescriptor &workload);

// ---------------------------------------------------------------------------
// Environment / provenance.
// ---------------------------------------------------------------------------

struct EnvironmentMetadata {
    luisa::string source_revision;
    luisa::string source_tree_state;
    luisa::string build_type;   // "debug" | "release" | ...
    luisa::string compiler;     // e.g. "msvc 19.4" / "clang 18"
    luisa::string os;           // OS name/version
    luisa::string backend;      // backend plugin name
    luisa::string cpu;
    luisa::string gpu;
    luisa::string driver_version;
    luisa::string luisa_version;
    uint32_t warp_size{0};
    uint64_t shared_memory_bytes{0};// per-block shared memory
    uint64_t total_device_bytes{0};
    BenchmarkCacheMode cache{BenchmarkCacheMode::miss};
    luisa::vector<KeyValue> provenance;// extra pinned provenance (package hash, ...)
};

// ---------------------------------------------------------------------------
// Samples / statistics.
// ---------------------------------------------------------------------------

// Optional stage durations within one full request. These are measured only
// at existing executor/synchronization boundaries; stage sums need not equal
// request_ms because recording, allocation, composition, and instrumentation
// overhead are reported separately rather than forced into a stage.
struct StageDurations {
    luisa::optional<double> load_ms;
    luisa::optional<double> preprocess_ms;
    luisa::optional<double> vision_ms;
    luisa::optional<double> prefill_ms;
    luisa::optional<double> decode_ms;
    luisa::optional<double> postprocess_ms;
};

// One timed iteration. Every metric is optional: a row only carries the
// metrics the runner recorded (use luisa::nullopt, never NaN, for "absent").
struct Sample {
    luisa::optional<double> record_ms;
    luisa::optional<double> submit_sync_ms;
    luisa::optional<double> request_ms;
    StageDurations stage;
};

// Summary of one metric's samples. p95 uses the nearest-rank method:
// sorted[ceil(0.95 * n) - 1] (the smallest sample at or above the 95th
// percentile). `stddev` is the SAMPLE standard deviation (n-1 denominator;
// 0 for n == 1). `mad` is the median absolute deviation from the median.
// [ci95_low, ci95_high] is the 1.96-sigma confidence interval of the mean.
struct Statistics {
    uint32_t n{0};
    double min{0.0};
    double median{0.0};
    double mean{0.0};
    double p95{0.0};
    double max{0.0};
    double stddev{0.0};
    double mad{0.0};
    double ci95_low{0.0};
    double ci95_high{0.0};
};

// Summarize finite samples. Rejects (invalid_argument) an empty vector or any
// non-finite value; never fabricates a summary.
[[nodiscard]] Result<Statistics> summarize_samples(const luisa::vector<double> &samples);

// ---------------------------------------------------------------------------
// Baseline comparison.
// ---------------------------------------------------------------------------

struct StageStatistics {
    luisa::optional<Statistics> load_ms;
    luisa::optional<Statistics> preprocess_ms;
    luisa::optional<Statistics> vision_ms;
    luisa::optional<Statistics> prefill_ms;
    luisa::optional<Statistics> decode_ms;
    luisa::optional<Statistics> postprocess_ms;
};

struct Comparison {
    BenchmarkMetric metric{BenchmarkMetric::record_ms};
    double baseline_median_ms{0.0};
    double current_median_ms{0.0};
    double median_delta_percent{0.0};// (current - baseline) / baseline * 100
    double threshold_percent{0.0};   // effective threshold applied
    ComparisonOutcome outcome{ComparisonOutcome::not_significant};
};

// Regression thresholds on the median percent delta: 5% for operator-facing
// metrics (record_ms, submit_sync_ms), 3% for the end-to-end request metric.
[[nodiscard]] double default_regression_threshold_percent(BenchmarkMetric metric) noexcept;

// Deltas within +/- this percent are never significant (measurement noise).
[[nodiscard]] double default_noise_floor_percent() noexcept;

// Compare medians. The effective threshold is max(metric threshold, noise
// floor). A non-positive baseline median cannot yield a percent delta; the
// comparison is then reported as not_significant with delta 0 (treat
// zero-time baselines as suspect).
[[nodiscard]] Comparison compare_median_to_baseline(BenchmarkMetric metric,
                                                    const Statistics &current,
                                                    const Statistics &baseline,
                                                    double noise_floor_percent);

// A promotion (regression call or improvement claim) additionally requires at
// least three INDEPENDENT process runs with interleaved/reference-randomized
// A/B order: one or two runs never support promotion, regardless of delta.
[[nodiscard]] bool comparison_supports_promotion(const Comparison &comparison,
                                                 uint32_t independent_runs) noexcept;

// ---------------------------------------------------------------------------
// Rows / report.
// ---------------------------------------------------------------------------

struct Row {
    WorkloadDescriptor workload;
    RowStatus status{RowStatus::ok};
    Status reason;// ok unless the row failed/was invalidated
    bool correct{false};
    bool fallback_used{false};
    bool truncated{false};
    luisa::vector<Sample> samples;// raw samples (warmup excluded), in order
    luisa::optional<Statistics> record_ms;
    luisa::optional<Statistics> submit_sync_ms;
    luisa::optional<Statistics> request_ms;
    StageStatistics stage;
    luisa::vector<KeyValue> counters;
    luisa::vector<Comparison> comparisons;
};

// Compute the available metric summaries for an ok row and invalidate bad
// rows: an empty workload id or any non-finite sample value flips the row to
// `invalid` with the reason recorded (statistics are cleared). Rows already
// in a non-ok state are left untouched. Returns ok unless the row itself
// cannot be processed.
[[nodiscard]] Status finalize_row(Row &row);

struct Report {
    luisa::string schema_version{"luisa.ocr.benchmark.v1"};
    EnvironmentMetadata environment;
    BenchmarkOptions options;// full run options = provenance
    luisa::vector<Row> rows;
};

// ---------------------------------------------------------------------------
// Serialization.
// ---------------------------------------------------------------------------

struct BenchmarkJsonOptions {
    bool pretty{true};
};

// Full deterministic report JSON: schema version, environment, complete
// provenance (options), and every row with its workload, status/reason,
// flags, RAW samples, all statistics, counters and comparisons. Returns
// invalid_argument rather than emitting NaN/Inf (never invalid JSON).
[[nodiscard]] Result<luisa::string> serialize_report(const Report &report,
                                                     const BenchmarkJsonOptions &options = {});

// Write the report to a file (stdout when the path is empty). A trailing
// newline is appended.
[[nodiscard]] Status write_report(const Report &report, const luisa::filesystem::path &path,
                                  const BenchmarkJsonOptions &options = {});

// Minimal CSV field quoting: fields containing ',', '"', '\n' or '\r' are
// quoted, inner quotes doubled. Exposed for tests.
[[nodiscard]] luisa::string csv_escape(luisa::string_view field);

// CSV append projection: one line per (row, metric-with-statistics). The
// header is written exactly once (only when the file is absent/empty), so
// repeated appends accumulate rows. Existing content is never rewritten.
[[nodiscard]] Status append_report_csv(const Report &report, const luisa::filesystem::path &path);

// ---------------------------------------------------------------------------
// Baseline input (a previously serialized report).
// ---------------------------------------------------------------------------

struct BaselineEntry {
    luisa::string workload_id;
    luisa::string variant;
    BenchmarkMetric metric{BenchmarkMetric::record_ms};
    double baseline_median_ms{0.0};
};

// Read a baseline report JSON (as produced by serialize_report) and extract
// the per-row/per-median medians needed for matching. io_error on read or
// parse failure; invalid_argument when the document has no rows array.
[[nodiscard]] Result<luisa::vector<BaselineEntry>> read_baseline_json(const luisa::filesystem::path &path);

// First entry matching workload id + variant + metric.
[[nodiscard]] luisa::optional<BaselineEntry> find_baseline_entry(
    const luisa::vector<BaselineEntry> &entries, const luisa::string &workload_id,
    const luisa::string &variant, BenchmarkMetric metric);

}// namespace luisa::ocr
