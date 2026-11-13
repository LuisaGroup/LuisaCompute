// OCR engine -- host-safe benchmark utility. See benchmark.h for the contract:
// strict options parsing, canonical workload identity, sample statistics,
// baseline comparison, deterministic JSON/CSV reporting. yyjson with correct
// copy-vs-borrow ownership; non-finite values are rejected, never emitted.
#include "ocr/benchmark.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <system_error>
#include <utility>

#include <yyjson.h>

namespace luisa::ocr {
namespace {

// ---------------------------------------------------------------------------
// Scalar parsing (strict, full-string).
// ---------------------------------------------------------------------------

[[nodiscard]] bool parse_u64(std::string_view text, uint64_t &out) noexcept {
    if (text.empty()) { return false; }
    uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) { return false; }
    out = value;
    return true;
}

[[nodiscard]] bool parse_u32(std::string_view text, uint32_t &out) noexcept {
    uint64_t value = 0;
    if (!parse_u64(text, value) || value > 0xffffffffull) { return false; }
    out = static_cast<uint32_t>(value);
    return true;
}

// ---------------------------------------------------------------------------
// Canonical workload identity.
// ---------------------------------------------------------------------------

void append_field(luisa::string &out, luisa::string_view name, luisa::string_view value) {
    out.append(name.data(), name.size());
    out.push_back('=');
    out.append(std::to_string(value.size()));
    out.push_back(':');
    out.append(value.data(), value.size());
    out.push_back('|');
}

void append_count(luisa::string &out, luisa::string_view name, uint64_t value) {
    append_field(out, name, std::to_string(value));
}

[[nodiscard]] luisa::vector<KeyValue> sorted_pairs(const luisa::vector<KeyValue> &pairs) {
    auto sorted = pairs;
    std::sort(sorted.begin(), sorted.end(), [](const KeyValue &a, const KeyValue &b) {
        if (a.key != b.key) { return a.key < b.key; }
        return a.value < b.value;
    });
    return sorted;
}

void append_pairs(luisa::string &out, luisa::string_view section, const luisa::vector<KeyValue> &pairs) {
    out.append(section.data(), section.size());
    out.push_back('|');
    for (const auto &pair : sorted_pairs(pairs)) {
        append_field(out, pair.key, pair.value);
    }
}

// ---------------------------------------------------------------------------
// Statistics helpers.
// ---------------------------------------------------------------------------

[[nodiscard]] double median_of_sorted(const luisa::vector<double> &sorted) noexcept {
    const auto n = sorted.size();
    if (n % 2u == 1u) { return sorted[n / 2u]; }
    // Even count: average the middle two samples.
    return (sorted[n / 2u - 1u] + sorted[n / 2u]) * 0.5;
}

[[nodiscard]] bool all_finite(const Statistics &s) noexcept {
    return std::isfinite(s.min) && std::isfinite(s.median) && std::isfinite(s.mean) &&
           std::isfinite(s.p95) && std::isfinite(s.max) && std::isfinite(s.stddev) &&
           std::isfinite(s.mad) && std::isfinite(s.ci95_low) && std::isfinite(s.ci95_high);
}

// ---------------------------------------------------------------------------
// yyjson RAII + copied-string helpers. Keys are string literals (static
// storage, safe to borrow); every dynamic string/key is COPIED into the
// document (strncpy/strcpy); the writer buffer is freed with the matching
// default allocator.
// ---------------------------------------------------------------------------

struct MutDoc {
    yyjson_mut_doc *doc{nullptr};
    MutDoc() noexcept : doc{yyjson_mut_doc_new(nullptr)} {}
    ~MutDoc() noexcept {
        if (doc != nullptr) { yyjson_mut_doc_free(doc); }
    }
    MutDoc(const MutDoc &) = delete;
    MutDoc &operator=(const MutDoc &) = delete;
};

struct Doc {
    yyjson_doc *doc{nullptr};
    ~Doc() noexcept {
        if (doc != nullptr) { yyjson_doc_free(doc); }
    }
    Doc() = default;
    explicit Doc(yyjson_doc *d) noexcept : doc{d} {}
    Doc(const Doc &) = delete;
    Doc &operator=(const Doc &) = delete;
};

void add_str(yyjson_mut_doc *doc, yyjson_mut_val *obj, const char *key, luisa::string_view value) {
    yyjson_mut_obj_add_strncpy(doc, obj, key, value.data(), value.size());
}

[[nodiscard]] bool add_finite_real(yyjson_mut_doc *doc, yyjson_mut_val *obj, const char *key, double value) {
    if (!std::isfinite(value)) { return false; }
    yyjson_mut_obj_add_real(doc, obj, key, value);
    return true;
}

void add_kv_map(yyjson_mut_doc *doc, yyjson_mut_val *parent, const char *key,
                const luisa::vector<KeyValue> &pairs) {
    auto *obj = yyjson_mut_obj(doc);
    for (const auto &pair : pairs) {
        // Dynamic keys must be copied too: yyjson borrows plain obj_add keys.
        auto *k = yyjson_mut_strncpy(doc, pair.key.data(), pair.key.size());
        auto *v = yyjson_mut_strncpy(doc, pair.value.data(), pair.value.size());
        yyjson_mut_obj_add(obj, k, v);
    }
    yyjson_mut_obj_add_val(doc, parent, key, obj);
}

[[nodiscard]] bool add_statistics(yyjson_mut_doc *doc, yyjson_mut_val *parent, const char *key,
                                  const Statistics &stats) {
    if (!all_finite(stats)) { return false; }
    auto *obj = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_uint(doc, obj, "n", stats.n);
    yyjson_mut_obj_add_real(doc, obj, "min", stats.min);
    yyjson_mut_obj_add_real(doc, obj, "median", stats.median);
    yyjson_mut_obj_add_real(doc, obj, "mean", stats.mean);
    yyjson_mut_obj_add_real(doc, obj, "p95", stats.p95);
    yyjson_mut_obj_add_real(doc, obj, "max", stats.max);
    yyjson_mut_obj_add_real(doc, obj, "stddev", stats.stddev);
    yyjson_mut_obj_add_real(doc, obj, "mad", stats.mad);
    yyjson_mut_obj_add_real(doc, obj, "ci95_low", stats.ci95_low);
    yyjson_mut_obj_add_real(doc, obj, "ci95_high", stats.ci95_high);
    yyjson_mut_obj_add_val(doc, parent, key, obj);
    return true;
}

[[nodiscard]] bool add_sample(yyjson_mut_doc *doc, yyjson_mut_val *arr, const Sample &sample) {
    auto *obj = yyjson_mut_obj(doc);
    if (sample.record_ms.has_value() &&
        !add_finite_real(doc, obj, "record_ms", *sample.record_ms)) { return false; }
    if (sample.submit_sync_ms.has_value() &&
        !add_finite_real(doc, obj, "submit_sync_ms", *sample.submit_sync_ms)) { return false; }
    if (sample.request_ms.has_value() &&
        !add_finite_real(doc, obj, "request_ms", *sample.request_ms)) { return false; }
    auto *stage = yyjson_mut_obj(doc);
    bool has_stage = false;
    const auto add_stage = [&](const char *key, const luisa::optional<double> &value) {
        if (!value.has_value()) { return true; }
        has_stage = true;
        return add_finite_real(doc, stage, key, *value);
    };
    if (!add_stage("load_ms", sample.stage.load_ms) ||
        !add_stage("preprocess_ms", sample.stage.preprocess_ms) ||
        !add_stage("vision_ms", sample.stage.vision_ms) ||
        !add_stage("prefill_ms", sample.stage.prefill_ms) ||
        !add_stage("decode_ms", sample.stage.decode_ms) ||
        !add_stage("postprocess_ms", sample.stage.postprocess_ms)) { return false; }
    if (has_stage) { yyjson_mut_obj_add_val(doc, obj, "stage", stage); }
    yyjson_mut_arr_append(arr, obj);
    return true;
}

[[nodiscard]] Result<luisa::string> write_doc(yyjson_mut_doc *doc, const BenchmarkJsonOptions &options) {
    using R = Result<luisa::string>;
    const auto flags = options.pretty ? YYJSON_WRITE_PRETTY_TWO_SPACES : YYJSON_WRITE_NOFLAG;
    size_t length = 0;
    char *json = yyjson_mut_write(doc, flags, &length);
    if (json == nullptr) {
        return R::failure(StatusCode::out_of_memory, "cannot serialize benchmark report", "json");
    }
    luisa::string out{json, length};
    std::free(json);// yyjson default allocator
    return R::success(std::move(out));
}

[[nodiscard]] Status write_text(const luisa::string &text, const luisa::filesystem::path &path) {
    if (path.empty()) {
        if (std::fputs(text.c_str(), stdout) < 0) {
            return Status::error(StatusCode::io_error, "failed to write benchmark report to stdout", "output");
        }
        return Status::success();
    }
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "wb");
    if (file == nullptr) {
        return Status::error(StatusCode::io_error,
                             luisa::string{"cannot open output file '"} + path_text + "'", "output");
    }
    const bool wrote = std::fwrite(text.data(), 1, text.size(), file) == text.size();
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !closed) {
        return Status::error(StatusCode::io_error,
                             luisa::string{"failed to write output file '"} + path_text + "'", "output");
    }
    return Status::success();
}

void add_environment(yyjson_mut_doc *doc, yyjson_mut_val *root, const EnvironmentMetadata &env) {
    auto *obj = yyjson_mut_obj(doc);
    add_str(doc, obj, "source_revision", env.source_revision);
    add_str(doc, obj, "source_tree_state", env.source_tree_state);
    add_str(doc, obj, "build_type", env.build_type);
    add_str(doc, obj, "compiler", env.compiler);
    add_str(doc, obj, "os", env.os);
    add_str(doc, obj, "backend", env.backend);
    add_str(doc, obj, "cpu", env.cpu);
    add_str(doc, obj, "gpu", env.gpu);
    add_str(doc, obj, "driver_version", env.driver_version);
    add_str(doc, obj, "luisa_version", env.luisa_version);
    yyjson_mut_obj_add_uint(doc, obj, "warp_size", env.warp_size);
    yyjson_mut_obj_add_uint(doc, obj, "shared_memory_bytes", env.shared_memory_bytes);
    yyjson_mut_obj_add_uint(doc, obj, "total_device_bytes", env.total_device_bytes);
    add_str(doc, obj, "cache", benchmark_cache_mode_name(env.cache));
    add_kv_map(doc, obj, "provenance", env.provenance);
    yyjson_mut_obj_add_val(doc, root, "environment", obj);
}

void add_options(yyjson_mut_doc *doc, yyjson_mut_val *root, const BenchmarkOptions &options) {
    auto *obj = yyjson_mut_obj(doc);
    add_str(doc, obj, "backend", options.backend);
    add_str(doc, obj, "suite", benchmark_suite_name(options.suite));
    add_str(doc, obj, "target", options.target);
    add_str(doc, obj, "case", options.case_name);
    add_str(doc, obj, "variant", options.variant);
    add_str(doc, obj, "workload", options.workload);
    add_str(doc, obj, "artifact_class", artifact_class_name(options.artifact_class));
    yyjson_mut_obj_add_uint(doc, obj, "warmup", options.warmup);
    yyjson_mut_obj_add_uint(doc, obj, "iterations", options.iterations);
    yyjson_mut_obj_add_uint(doc, obj, "dispatch_batch", options.dispatch_batch);
    yyjson_mut_obj_add_bool(doc, obj, "graph", options.graph);
    yyjson_mut_obj_add_bool(doc, obj, "fusion", options.fusion);
    yyjson_mut_obj_add_bool(doc, obj, "cooperative_norm", options.cooperative_norm);
    add_str(doc, obj, "cache", benchmark_cache_mode_name(options.cache));
    yyjson_mut_obj_add_bool(doc, obj, "instrument", options.instrument);
    add_str(doc, obj, "json", luisa::to_string(options.json_path));
    add_str(doc, obj, "csv", luisa::to_string(options.csv_path));
    add_str(doc, obj, "baseline", luisa::to_string(options.baseline_path));
    add_str(doc, obj, "source_revision", options.source_revision);
    add_str(doc, obj, "source_tree_state", options.source_tree_state);
    add_str(doc, obj, "gpu_name", options.gpu_name);
    add_str(doc, obj, "driver_version", options.driver_version);
    add_str(doc, obj, "package", luisa::to_string(options.package_path));
    add_str(doc, obj, "manifest", luisa::to_string(options.manifest_path));
    auto *provenance = yyjson_mut_arr(doc);
    for (const auto &entry : options.provenance) {
        yyjson_mut_arr_add_strncpy(doc, provenance, entry.data(), entry.size());
    }
    yyjson_mut_obj_add_val(doc, obj, "provenance", provenance);
    yyjson_mut_obj_add_bool(doc, obj, "pretty", options.pretty);
    yyjson_mut_obj_add_val(doc, root, "options", obj);
}

void add_workload(yyjson_mut_doc *doc, yyjson_mut_val *row_obj,
                  const WorkloadDescriptor &workload) {
    auto *obj = yyjson_mut_obj(doc);
    add_str(doc, obj, "id", workload.id);
    add_str(doc, obj, "suite", benchmark_suite_name(workload.suite));
    add_str(doc, obj, "target", workload.target);
    add_str(doc, obj, "case", workload.case_name);
    add_str(doc, obj, "variant", workload.variant);
    add_str(doc, obj, "artifact_class", artifact_class_name(workload.artifact_class));
    yyjson_mut_obj_add_uint(doc, obj, "seed", workload.seed);
    yyjson_mut_obj_add_uint(doc, obj, "fingerprint", workload_fingerprint64(workload));
    add_kv_map(doc, obj, "shape", workload.shape);
    add_kv_map(doc, obj, "options", workload.options);
    auto *counts = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_uint(doc, counts, "pages", workload.pages);
    yyjson_mut_obj_add_uint(doc, counts, "regions", workload.regions);
    yyjson_mut_obj_add_uint(doc, counts, "patches", workload.patches);
    yyjson_mut_obj_add_uint(doc, counts, "prompt_tokens", workload.prompt_tokens);
    yyjson_mut_obj_add_uint(doc, counts, "output_tokens", workload.output_tokens);
    yyjson_mut_obj_add_uint(doc, counts, "dispatches", workload.dispatches);
    yyjson_mut_obj_add_uint(doc, counts, "flops", workload.flops);
    yyjson_mut_obj_add_uint(doc, counts, "bytes", workload.bytes);
    yyjson_mut_obj_add_uint(doc, counts, "weight_bytes", workload.weight_bytes);
    add_str(doc, counts, "byte_accounting", workload.byte_accounting);
    yyjson_mut_obj_add_val(doc, obj, "counts", counts);
    yyjson_mut_obj_add_val(doc, row_obj, "workload", obj);
}

[[nodiscard]] bool add_row(yyjson_mut_doc *doc, yyjson_mut_val *rows, const Row &row) {
    auto *obj = yyjson_mut_obj(doc);
    add_workload(doc, obj, row.workload);
    add_str(doc, obj, "status", row_status_name(row.status));
    auto *reason = yyjson_mut_obj(doc);
    add_str(doc, reason, "code", status_code_name(row.reason.code));
    add_str(doc, reason, "message", row.reason.message);
    add_str(doc, reason, "detail", row.reason.detail);
    yyjson_mut_obj_add_val(doc, obj, "reason", reason);
    yyjson_mut_obj_add_bool(doc, obj, "correct", row.correct);
    yyjson_mut_obj_add_bool(doc, obj, "fallback_used", row.fallback_used);
    yyjson_mut_obj_add_bool(doc, obj, "truncated", row.truncated);

    auto *samples = yyjson_mut_arr(doc);
    for (const auto &sample : row.samples) {
        if (!add_sample(doc, samples, sample)) { return false; }
    }
    yyjson_mut_obj_add_val(doc, obj, "samples", samples);

    auto *stats = yyjson_mut_obj(doc);
    if (row.record_ms.has_value() &&
        !add_statistics(doc, stats, "record_ms", *row.record_ms)) { return false; }
    if (row.submit_sync_ms.has_value() &&
        !add_statistics(doc, stats, "submit_sync_ms", *row.submit_sync_ms)) { return false; }
    if (row.request_ms.has_value() &&
        !add_statistics(doc, stats, "request_ms", *row.request_ms)) { return false; }
    auto *stage = yyjson_mut_obj(doc);
    bool has_stage = false;
    const auto add_stage = [&](const char *key, const luisa::optional<Statistics> &value) {
        if (!value.has_value()) { return true; }
        has_stage = true;
        return add_statistics(doc, stage, key, *value);
    };
    if (!add_stage("load_ms", row.stage.load_ms) ||
        !add_stage("preprocess_ms", row.stage.preprocess_ms) ||
        !add_stage("vision_ms", row.stage.vision_ms) ||
        !add_stage("prefill_ms", row.stage.prefill_ms) ||
        !add_stage("decode_ms", row.stage.decode_ms) ||
        !add_stage("postprocess_ms", row.stage.postprocess_ms)) { return false; }
    if (has_stage) { yyjson_mut_obj_add_val(doc, stats, "stage", stage); }
    yyjson_mut_obj_add_val(doc, obj, "statistics", stats);

    add_kv_map(doc, obj, "counters", row.counters);

    auto *comparisons = yyjson_mut_arr(doc);
    for (const auto &comparison : row.comparisons) {
        if (!std::isfinite(comparison.baseline_median_ms) ||
            !std::isfinite(comparison.current_median_ms) ||
            !std::isfinite(comparison.median_delta_percent) ||
            !std::isfinite(comparison.threshold_percent)) { return false; }
        auto *c = yyjson_mut_obj(doc);
        add_str(doc, c, "metric", benchmark_metric_name(comparison.metric));
        yyjson_mut_obj_add_real(doc, c, "baseline_median_ms", comparison.baseline_median_ms);
        yyjson_mut_obj_add_real(doc, c, "current_median_ms", comparison.current_median_ms);
        yyjson_mut_obj_add_real(doc, c, "median_delta_percent", comparison.median_delta_percent);
        yyjson_mut_obj_add_real(doc, c, "threshold_percent", comparison.threshold_percent);
        add_str(doc, c, "outcome", comparison_outcome_name(comparison.outcome));
        yyjson_mut_arr_append(comparisons, c);
    }
    yyjson_mut_obj_add_val(doc, obj, "comparisons", comparisons);

    yyjson_mut_arr_append(rows, obj);
    return true;
}

[[nodiscard]] luisa::string statistics_csv_row(luisa::string_view metric_name, const Statistics &s) {
    luisa::string out;
    out.append(metric_name.data(), metric_name.size());
    out.push_back(',');
    out.append(std::to_string(s.n));
    out.push_back(',');
    out.append(std::to_string(s.min));
    out.push_back(',');
    out.append(std::to_string(s.median));
    out.push_back(',');
    out.append(std::to_string(s.mean));
    out.push_back(',');
    out.append(std::to_string(s.p95));
    out.push_back(',');
    out.append(std::to_string(s.max));
    out.push_back(',');
    out.append(std::to_string(s.stddev));
    out.push_back(',');
    out.append(std::to_string(s.mad));
    out.push_back(',');
    out.append(std::to_string(s.ci95_low));
    out.push_back(',');
    out.append(std::to_string(s.ci95_high));
    return out;
}

}// namespace

// ---------------------------------------------------------------------------
// Enum names + parsers.
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::string_view benchmark_suite_name(BenchmarkSuite suite) noexcept {
    switch (suite) {
        case BenchmarkSuite::ops: return "ops";
        case BenchmarkSuite::decode: return "decode";
        case BenchmarkSuite::cnn: return "cnn";
        case BenchmarkSuite::layout: return "layout";
        case BenchmarkSuite::gdn: return "gdn";
        case BenchmarkSuite::moe: return "moe";
        case BenchmarkSuite::model: return "model";
        case BenchmarkSuite::all: return "all";
        default: return "unknown";
    }
}

[[nodiscard]] luisa::string_view artifact_class_name(ArtifactClass artifact_class) noexcept {
    switch (artifact_class) {
        case ArtifactClass::synthetic: return "synthetic";
        case ArtifactClass::real: return "real";
        default: return "unknown";
    }
}

[[nodiscard]] luisa::string_view benchmark_cache_mode_name(BenchmarkCacheMode mode) noexcept {
    switch (mode) {
        case BenchmarkCacheMode::miss: return "miss";
        case BenchmarkCacheMode::hit: return "hit";
        case BenchmarkCacheMode::disabled: return "disabled";
        default: return "unknown";
    }
}

[[nodiscard]] luisa::string_view row_status_name(RowStatus status) noexcept {
    switch (status) {
        case RowStatus::ok: return "ok";
        case RowStatus::failed: return "failed";
        case RowStatus::skipped: return "skipped";
        case RowStatus::blocked: return "blocked";
        case RowStatus::refused: return "refused";
        case RowStatus::invalid: return "invalid";
        default: return "unknown";
    }
}

[[nodiscard]] luisa::string_view benchmark_metric_name(BenchmarkMetric metric) noexcept {
    switch (metric) {
        case BenchmarkMetric::record_ms: return "record_ms";
        case BenchmarkMetric::submit_sync_ms: return "submit_sync_ms";
        case BenchmarkMetric::request_ms: return "request_ms";
        default: return "unknown";
    }
}

[[nodiscard]] luisa::string_view comparison_outcome_name(ComparisonOutcome outcome) noexcept {
    switch (outcome) {
        case ComparisonOutcome::not_significant: return "not_significant";
        case ComparisonOutcome::regression: return "regression";
        case ComparisonOutcome::improvement: return "improvement";
        default: return "unknown";
    }
}

[[nodiscard]] bool parse_benchmark_suite(luisa::string_view text, BenchmarkSuite &out) noexcept {
    for (auto suite : {BenchmarkSuite::ops, BenchmarkSuite::decode, BenchmarkSuite::cnn,
                       BenchmarkSuite::layout, BenchmarkSuite::gdn, BenchmarkSuite::moe,
                       BenchmarkSuite::model, BenchmarkSuite::all}) {
        if (text == benchmark_suite_name(suite)) {
            out = suite;
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool parse_artifact_class(luisa::string_view text, ArtifactClass &out) noexcept {
    if (text == "synthetic") {
        out = ArtifactClass::synthetic;
        return true;
    }
    if (text == "real") {
        out = ArtifactClass::real;
        return true;
    }
    return false;
}

[[nodiscard]] bool parse_benchmark_cache_mode(luisa::string_view text, BenchmarkCacheMode &out) noexcept {
    for (auto mode : {BenchmarkCacheMode::miss, BenchmarkCacheMode::hit, BenchmarkCacheMode::disabled}) {
        if (text == benchmark_cache_mode_name(mode)) {
            out = mode;
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool parse_on_off(luisa::string_view text, bool &out) noexcept {
    if (text == "on") {
        out = true;
        return true;
    }
    if (text == "off") {
        out = false;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Options parser.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<BenchmarkOptions> parse_benchmark_options(int argc, const char *const *argv) {
    using R = Result<BenchmarkOptions>;
    BenchmarkOptions options;
    // --help short-circuits everything, even alongside other arguments.
    for (int i = 1; i < argc; i++) {
        const std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            options.help = true;
            return R::success(std::move(options));
        }
    }
    if (argc < 2 || std::string_view{argv[1]}.empty() ||
        std::string_view{argv[1]}.starts_with("--")) {
        return R::failure(StatusCode::invalid_argument,
                          "missing backend argument (argv[1]); see --help", "backend");
    }
    options.backend = luisa::string{std::string_view{argv[1]}};
    if (argc < 3) {
        return R::failure(StatusCode::invalid_argument,
                          "missing suite argument (argv[2]): ops|decode|cnn|layout|gdn|moe|model|all",
                          "suite");
    }
    if (!parse_benchmark_suite(std::string_view{argv[2]}, options.suite)) {
        return R::failure(StatusCode::invalid_argument,
                          "unknown suite; expected ops|decode|cnn|layout|gdn|moe|model|all", "suite");
    }

    // Duplicate-option detection.
    struct Seen {
        bool target{false}, case_name{false}, variant{false}, workload{false};
        bool artifact_class{false}, warmup{false}, iterations{false}, dispatch_batch{false};
        bool graph{false}, fusion{false}, cooperative_norm{false}, cache{false}, instrument{false};
        bool json{false}, csv{false}, baseline{false};
        bool source_revision{false}, source_tree_state{false}, gpu_name{false}, driver_version{false};
        bool package{false}, manifest{false}, no_pretty{false};
    } seen;
    const auto mark = [](bool &flag, const char *name) -> Status {
        if (flag) {
            const char *field = name;
            while (*field == '-') { field++; }
            return Status::error(StatusCode::invalid_argument,
                                 luisa::string{"duplicate option '"} + name + "'", field);
        }
        flag = true;
        return Status::success();
    };
    const auto need_value = [&](int &i, luisa::string_view &value) -> bool {
        if (i + 1 >= argc) { return false; }
        value = std::string_view{argv[++i]};
        return true;
    };
    const auto parse_path = [&](int &i, luisa::string_view &value, const char *name,
                                luisa::filesystem::path &out) -> Status {
        if (!need_value(i, value)) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::string{"'"} + name + "' requires a value", name);
        }
        if (!luisa::path_from_utf8(value, out)) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::string{"'"} + name + "' is not valid UTF-8", name);
        }
        return Status::success();
    };

    for (int i = 3; i < argc; i++) {
        const std::string_view arg{argv[i]};
        luisa::string_view value;
        if (!arg.starts_with("--")) {
            luisa::string message{"unexpected positional argument '"};
            message.append(arg.data(), arg.size());
            message.push_back('\'');
            return R::failure(StatusCode::invalid_argument, std::move(message), "argv");
        }
        if (arg == "--target") {
            if (auto s = mark(seen.target, "--target"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--target requires a value", "target"); }
            options.target = luisa::string{value};
        } else if (arg == "--case") {
            if (auto s = mark(seen.case_name, "--case"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--case requires a value", "case"); }
            options.case_name = luisa::string{value};
        } else if (arg == "--variant") {
            if (auto s = mark(seen.variant, "--variant"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--variant requires a value", "variant"); }
            options.variant = luisa::string{value};
        } else if (arg == "--workload") {
            if (auto s = mark(seen.workload, "--workload"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--workload requires a value", "workload"); }
            options.workload = luisa::string{value};
        } else if (arg == "--artifact-class") {
            if (auto s = mark(seen.artifact_class, "--artifact-class"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--artifact-class requires a value", "artifact-class"); }
            if (!parse_artifact_class(value, options.artifact_class)) {
                return R::failure(StatusCode::invalid_argument,
                                  "--artifact-class must be synthetic|real", "artifact-class");
            }
        } else if (arg == "--warmup" || arg == "--iterations" || arg == "--dispatch-batch") {
            const bool is_warmup = arg == "--warmup";
            const bool is_iterations = arg == "--iterations";
            bool &flag = is_warmup ? seen.warmup : (is_iterations ? seen.iterations : seen.dispatch_batch);
            const char *name = is_warmup ? "--warmup" : (is_iterations ? "--iterations" : "--dispatch-batch");
            if (auto s = mark(flag, name); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::string{"'"} + name + "' requires a value", name);
            }
            uint32_t parsed = 0;
            if (!parse_u32(value, parsed)) {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::string{"'"} + name + "' requires an unsigned 32-bit integer", name);
            }
            if (is_warmup) {
                options.warmup = parsed;
            } else if (is_iterations) {
                if (parsed < 1u) {
                    return R::failure(StatusCode::invalid_argument, "--iterations must be >= 1", "iterations");
                }
                options.iterations = parsed;
            } else {
                if (parsed < 1u) {
                    return R::failure(StatusCode::invalid_argument, "--dispatch-batch must be >= 1", "dispatch-batch");
                }
                options.dispatch_batch = parsed;
            }
        } else if (arg == "--graph" || arg == "--fusion" ||
                   arg == "--cooperative-norm" || arg == "--instrument") {
            bool *flag = &seen.instrument;
            bool *out = &options.instrument;
            const char *name = "--instrument";
            if (arg == "--graph") {
                flag = &seen.graph;
                out = &options.graph;
                name = "--graph";
            } else if (arg == "--fusion") {
                flag = &seen.fusion;
                out = &options.fusion;
                name = "--fusion";
            } else if (arg == "--cooperative-norm") {
                flag = &seen.cooperative_norm;
                out = &options.cooperative_norm;
                name = "--cooperative-norm";
            }
            if (auto s = mark(*flag, name); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::string{"'"} + name + "' requires a value", name);
            }
            if (!parse_on_off(value, *out)) {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::string{"'"} + name + "' must be on|off", name);
            }
        } else if (arg == "--cache") {
            if (auto s = mark(seen.cache, "--cache"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--cache requires a value", "cache"); }
            if (!parse_benchmark_cache_mode(value, options.cache)) {
                return R::failure(StatusCode::invalid_argument, "--cache must be miss|hit|disabled", "cache");
            }
        } else if (arg == "--json") {
            if (auto s = mark(seen.json, "--json"); !s.ok()) { return R::failure(std::move(s)); }
            if (auto s = parse_path(i, value, "--json", options.json_path); !s.ok()) { return R::failure(std::move(s)); }
        } else if (arg == "--csv") {
            if (auto s = mark(seen.csv, "--csv"); !s.ok()) { return R::failure(std::move(s)); }
            if (auto s = parse_path(i, value, "--csv", options.csv_path); !s.ok()) { return R::failure(std::move(s)); }
        } else if (arg == "--baseline") {
            if (auto s = mark(seen.baseline, "--baseline"); !s.ok()) { return R::failure(std::move(s)); }
            if (auto s = parse_path(i, value, "--baseline", options.baseline_path); !s.ok()) { return R::failure(std::move(s)); }
        } else if (arg == "--source-revision") {
            if (auto s = mark(seen.source_revision, "--source-revision"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--source-revision requires a value", "source-revision"); }
            options.source_revision = luisa::string{value};
        } else if (arg == "--source-tree-state") {
            if (auto s = mark(seen.source_tree_state, "--source-tree-state"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--source-tree-state requires a value", "source-tree-state"); }
            options.source_tree_state = luisa::string{value};
        } else if (arg == "--gpu-name") {
            if (auto s = mark(seen.gpu_name, "--gpu-name"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--gpu-name requires a value", "gpu-name"); }
            options.gpu_name = luisa::string{value};
        } else if (arg == "--driver-version") {
            if (auto s = mark(seen.driver_version, "--driver-version"); !s.ok()) { return R::failure(std::move(s)); }
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--driver-version requires a value", "driver-version"); }
            options.driver_version = luisa::string{value};
        } else if (arg == "--package") {
            if (auto s = mark(seen.package, "--package"); !s.ok()) { return R::failure(std::move(s)); }
            if (auto s = parse_path(i, value, "--package", options.package_path); !s.ok()) { return R::failure(std::move(s)); }
        } else if (arg == "--manifest") {
            if (auto s = mark(seen.manifest, "--manifest"); !s.ok()) { return R::failure(std::move(s)); }
            if (auto s = parse_path(i, value, "--manifest", options.manifest_path); !s.ok()) { return R::failure(std::move(s)); }
        } else if (arg == "--provenance") {
            if (!need_value(i, value)) { return R::failure(StatusCode::invalid_argument, "--provenance requires key=value", "provenance"); }
            const auto equals = value.find('=');
            if (equals == luisa::string_view::npos || equals == 0u || equals + 1u == value.size()) {
                return R::failure(StatusCode::invalid_argument, "--provenance requires nonempty key=value", "provenance");
            }
            options.provenance.emplace_back(value);
        } else if (arg == "--no-pretty") {
            if (auto s = mark(seen.no_pretty, "--no-pretty"); !s.ok()) { return R::failure(std::move(s)); }
            options.pretty = false;
        } else {
            luisa::string message{"unknown argument '"};
            message.append(arg.data(), arg.size());
            message.push_back('\'');
            return R::failure(StatusCode::invalid_argument, std::move(message), "argv");
        }
    }

    // Real-artifact runs need pinned inputs up front; execution itself stays
    // fail-closed (check_real_execution).
    if (options.artifact_class == ArtifactClass::real &&
        (options.package_path.empty() || options.manifest_path.empty())) {
        return R::failure(StatusCode::invalid_argument,
                          "--artifact-class real requires --package and --manifest",
                          "artifact-class");
    }
    return R::success(std::move(options));
}

[[nodiscard]] luisa::string benchmark_usage_text() {
    return luisa::string{
        "example_ocr_bench -- LuisaCompute OCR benchmark harness (host reporting\n"
        "utility; measurement execution lives in the device runner)\n"
        "\n"
        "usage: example_ocr_bench <backend> <suite> [options]\n"
        " example_ocr_bench --help\n"
        "\n"
        "positional:\n"
        "  backend backend plugin: cuda | vk | dx | fallback | ...\n"
        "  suite ops | decode | cnn | layout | gdn | moe | model | all\n"
        "\n"
        "offline:\n"
        "  --help show this usage text without requiring a backend/suite (exit 0)\n"
        "\n"
        "selectors (all optional; empty = no filter; duplicates are rejected):\n"
        "  --target <name> target filter (model/op target)\n"
        "  --case <name> case filter\n"
        "  --variant <name> code-variant filter (e.g. kernel flavor)\n"
        "  --workload <id> single workload id\n"
        "\n"
        "inputs:\n"
        "  --artifact-class <c> synthetic (default) | real\n"
        "  --package <path> pinned .lcocr package (required for real)\n"
        "  --manifest <path> package/component manifest (required for real)\n"
        " real-artifact execution additionally fails closed with `unvalidated`\n"
        " until real artifacts are validated (see QUALIFICATION.md)\n"
        "\n"
        "measurement:\n"
        "  --warmup <n> warmup iterations (default 3)\n"
        "  --iterations <n> measured iterations (default 20, min 1)\n"
        "  --dispatch-batch <n> dispatches per iteration (default 1, min 1)\n"
        "  --graph on|off graph capture (default off)\n"
                "  --fusion on|off kernel fusion (default off)\n"
                "  --cooperative-norm on|off hidden-norm fast path (default on)\n"
                "  --cache <m> miss (default) | hit | disabled\n"
        "  --instrument on|off extra instrumentation (default off)\n"
        "\n"
        "output:\n"
        "  --json <path> full report JSON (default: stdout)\n"
        "  --csv <path> append CSV projection (header written once)\n"
        "  --baseline <path> baseline report JSON for median comparison\n"
        "  --no-pretty minified JSON (default: two-space pretty)\n"
        "\n"
        "provenance overrides (default: captured from the environment):\n"
        "  --source-revision <rev> source revision\n"
        "  --source-tree-state <state> source tree state\n"
        "  --gpu-name <name> GPU name\n"
        "  --driver-version <v> driver version\n"
        "  --provenance <k=v> extra provenance (repeatable: CUDA runtime, clocks, corpus)\n"};
}

[[nodiscard]] Status check_real_execution(const BenchmarkOptions &options) {
    if (options.artifact_class != ArtifactClass::real) { return Status::success(); }
    // Argument validation first, mirroring the Session fail-closed order.
    if (options.package_path.empty() || options.manifest_path.empty()) {
        return Status::error(StatusCode::invalid_argument,
                             "real-artifact runs require --package and --manifest",
                             "artifact-class");
    }
    return Status::error(StatusCode::unvalidated,
                         "real-artifact benchmark execution is not validated yet",
                         "package");
}

// ---------------------------------------------------------------------------
// Workload identity.
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::string canonical_workload_key(const WorkloadDescriptor &workload) {
    luisa::string out;
    out.append("luisa.ocr.workload.v1|");
    append_field(out, "id", workload.id);
    append_field(out, "suite", benchmark_suite_name(workload.suite));
    append_field(out, "target", workload.target);
    append_field(out, "case", workload.case_name);
    append_field(out, "variant", workload.variant);
    append_field(out, "artifact", artifact_class_name(workload.artifact_class));
    append_count(out, "seed", workload.seed);
    append_pairs(out, "shape", workload.shape);
    append_pairs(out, "options", workload.options);
    append_count(out, "pages", workload.pages);
    append_count(out, "regions", workload.regions);
    append_count(out, "patches", workload.patches);
    append_count(out, "prompt_tokens", workload.prompt_tokens);
    append_count(out, "output_tokens", workload.output_tokens);
    append_count(out, "dispatches", workload.dispatches);
    append_count(out, "flops", workload.flops);
    append_count(out, "bytes", workload.bytes);
    append_count(out, "weight_bytes", workload.weight_bytes);
    append_field(out, "byte_accounting", workload.byte_accounting);
    return out;
}

[[nodiscard]] uint64_t workload_fingerprint64(const WorkloadDescriptor &workload) {
    const auto key = canonical_workload_key(workload);
    // FNV-1a 64-bit. NON-CRYPTOGRAPHIC: identity/dedup only.
    uint64_t hash = 14695981039346656037ull;
    for (const unsigned char c : key) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    return hash;
}

// ---------------------------------------------------------------------------
// Statistics.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<Statistics> summarize_samples(const luisa::vector<double> &samples) {
    using R = Result<Statistics>;
    if (samples.empty()) {
        return R::failure(StatusCode::invalid_argument, "no samples to summarize", "samples");
    }
    for (const auto sample : samples) {
        if (!std::isfinite(sample)) {
            return R::failure(StatusCode::invalid_argument,
                              "non-finite sample cannot be summarized", "samples");
        }
    }
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    Statistics stats;
    stats.n = static_cast<uint32_t>(sorted.size());
    stats.min = sorted.front();
    stats.max = sorted.back();
    stats.median = median_of_sorted(sorted);
    double sum = 0.0;
    for (const auto sample : sorted) { sum += sample; }
    stats.mean = sum / static_cast<double>(sorted.size());
    // p95, nearest-rank: sorted[ceil(0.95 * n) - 1].
    auto rank = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(sorted.size())));
    if (rank == 0u) { rank = 1u; }
    if (rank > sorted.size()) { rank = sorted.size(); }
    stats.p95 = sorted[rank - 1u];
    // Sample standard deviation (n-1; 0 for n == 1).
    if (sorted.size() > 1u) {
        double squared = 0.0;
        for (const auto sample : sorted) {
            const auto d = sample - stats.mean;
            squared += d * d;
        }
        stats.stddev = std::sqrt(squared / static_cast<double>(sorted.size() - 1u));
    }
    // Median absolute deviation from the median.
    luisa::vector<double> deviations;
    deviations.reserve(sorted.size());
    for (const auto sample : sorted) { deviations.push_back(std::fabs(sample - stats.median)); }
    std::sort(deviations.begin(), deviations.end());
    stats.mad = median_of_sorted(deviations);
    // 95% confidence interval of the mean (1.96 sigma).
    const auto half = 1.96 * stats.stddev / std::sqrt(static_cast<double>(sorted.size()));
    stats.ci95_low = stats.mean - half;
    stats.ci95_high = stats.mean + half;
    return R::success(std::move(stats));
}

// ---------------------------------------------------------------------------
// Baseline comparison.
// ---------------------------------------------------------------------------

[[nodiscard]] double default_regression_threshold_percent(BenchmarkMetric metric) noexcept {
    // Operator-facing metrics tolerate 5%; the end-to-end request metric is
    // user-visible and gets a tighter 3% promotion threshold.
    return metric == BenchmarkMetric::request_ms ? 3.0 : 5.0;
}

[[nodiscard]] double default_noise_floor_percent() noexcept {
    return 1.0;
}

[[nodiscard]] Comparison compare_median_to_baseline(BenchmarkMetric metric,
                                                    const Statistics &current,
                                                    const Statistics &baseline,
                                                    double noise_floor_percent) {
    Comparison comparison;
    comparison.metric = metric;
    comparison.baseline_median_ms = baseline.median;
    comparison.current_median_ms = current.median;
    comparison.threshold_percent = std::max(default_regression_threshold_percent(metric),
                                            noise_floor_percent);
    if (baseline.median <= 0.0) {
        // No usable percent baseline: report no delta and no significance.
        return comparison;
    }
    comparison.median_delta_percent =
        (current.median - baseline.median) / baseline.median * 100.0;
    if (std::fabs(comparison.median_delta_percent) <= comparison.threshold_percent) {
        comparison.outcome = ComparisonOutcome::not_significant;
    } else {
        comparison.outcome = comparison.median_delta_percent > 0.0 ?
                                 ComparisonOutcome::regression :
                                 ComparisonOutcome::improvement;
    }
    return comparison;
}

[[nodiscard]] bool comparison_supports_promotion(const Comparison &comparison,
                                                 uint32_t independent_runs) noexcept {
    // Accepted A/B decisions require three independent process runs.
    return comparison.outcome != ComparisonOutcome::not_significant && independent_runs >= 3u;
}

// ---------------------------------------------------------------------------
// Rows.
// ---------------------------------------------------------------------------

[[nodiscard]] Status finalize_row(Row &row) {
    if (row.status != RowStatus::ok) { return Status::success(); }
    const auto invalidate = [&row](luisa::string message, luisa::string detail) {
        row.status = RowStatus::invalid;
        row.reason = Status::error(StatusCode::invalid_argument, std::move(message), std::move(detail));
        row.record_ms.reset();
        row.submit_sync_ms.reset();
        row.request_ms.reset();
        row.stage = {};
    };
    if (row.workload.id.empty()) {
        invalidate("row workload id is empty", "workload.id");
        return Status::success();
    }
    for (const auto &sample : row.samples) {
        const auto bad = [](const luisa::optional<double> &v) {
            return v.has_value() && !std::isfinite(*v);
        };
        if (bad(sample.record_ms) || bad(sample.submit_sync_ms) || bad(sample.request_ms) ||
            bad(sample.stage.load_ms) || bad(sample.stage.preprocess_ms) ||
            bad(sample.stage.vision_ms) || bad(sample.stage.prefill_ms) ||
            bad(sample.stage.decode_ms) || bad(sample.stage.postprocess_ms)) {
            invalidate("non-finite sample value", "samples");
            return Status::success();
        }
    }
    luisa::vector<double> record, submit_sync, request;
    for (const auto &sample : row.samples) {
        if (sample.record_ms.has_value()) { record.push_back(*sample.record_ms); }
        if (sample.submit_sync_ms.has_value()) { submit_sync.push_back(*sample.submit_sync_ms); }
        if (sample.request_ms.has_value()) { request.push_back(*sample.request_ms); }
    }
    const auto summarize = [](const luisa::vector<double> &values) -> luisa::optional<Statistics> {
        if (values.empty()) { return luisa::nullopt; }
        auto result = summarize_samples(values);// finite: cannot fail here
        if (!result.ok()) { return luisa::nullopt; }
        return luisa::optional<Statistics>{std::move(*result.value)};
    };
    row.record_ms = summarize(record);
    row.submit_sync_ms = summarize(submit_sync);
    row.request_ms = summarize(request);
    const auto summarize_stage = [&](auto member) -> luisa::optional<Statistics> {
        luisa::vector<double> values;
        for (const auto &sample : row.samples) {
            if (const auto &value = sample.stage.*member; value.has_value()) {
                values.push_back(*value);
            }
        }
        return summarize(values);
    };
    row.stage.load_ms = summarize_stage(&StageDurations::load_ms);
    row.stage.preprocess_ms = summarize_stage(&StageDurations::preprocess_ms);
    row.stage.vision_ms = summarize_stage(&StageDurations::vision_ms);
    row.stage.prefill_ms = summarize_stage(&StageDurations::prefill_ms);
    row.stage.decode_ms = summarize_stage(&StageDurations::decode_ms);
    row.stage.postprocess_ms = summarize_stage(&StageDurations::postprocess_ms);
    return Status::success();
}

// ---------------------------------------------------------------------------
// Serialization.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::string> serialize_report(const Report &report,
                                                     const BenchmarkJsonOptions &options) {
    using R = Result<luisa::string>;
    MutDoc holder;
    if (holder.doc == nullptr) {
        return R::failure(StatusCode::out_of_memory, "cannot allocate JSON document", "json");
    }
    auto *doc = holder.doc;
    auto *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    add_str(doc, root, "schema_version", report.schema_version);
    add_environment(doc, root, report.environment);
    add_options(doc, root, report.options);
    auto *rows = yyjson_mut_arr(doc);
    for (const auto &row : report.rows) {
        if (!add_row(doc, rows, row)) {
            return R::failure(StatusCode::invalid_argument,
                              "non-finite value in benchmark row (sample/statistics/comparison)",
                              "rows");
        }
    }
    yyjson_mut_obj_add_val(doc, root, "rows", rows);
    return write_doc(doc, options);
}

[[nodiscard]] Status write_report(const Report &report, const luisa::filesystem::path &path,
                                  const BenchmarkJsonOptions &options) {
    auto json = serialize_report(report, options);
    if (!json.ok()) { return json.status; }
    auto out = *json.value;
    out.push_back('\n');
    return write_text(out, path);
}

[[nodiscard]] luisa::string csv_escape(luisa::string_view field) {
    const auto needs_quotes = field.find_first_of(",\"\n\r") != luisa::string_view::npos;
    if (!needs_quotes) { return luisa::string{field}; }
    luisa::string out;
    out.reserve(field.size() + 2u);
    out.push_back('"');
    for (const auto c : field) {
        if (c == '"') {
            out.push_back('"');
            out.push_back('"');
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

[[nodiscard]] Status append_report_csv(const Report &report, const luisa::filesystem::path &path) {
    if (path.empty()) {
        return Status::error(StatusCode::invalid_argument, "CSV path is empty", "csv");
    }
    // Header exactly once: only when the file is absent or empty.
    std::error_code ec;
    bool need_header = true;
    if (luisa::filesystem::exists(path, ec)) {
        need_header = luisa::filesystem::file_size(path, ec) == 0u;
    }
    luisa::string text;
    if (need_header) {
        text.append("workload_id,suite,target,case,variant,artifact_class,status,correct,"
                    "metric,n,min,median,mean,p95,max,stddev,mad,ci95_low,ci95_high\n");
    }
    const auto append_fields = [&text](const Row &row) {
        text.append(csv_escape(row.workload.id));
        text.push_back(',');
        text.append(benchmark_suite_name(row.workload.suite));
        text.push_back(',');
        text.append(csv_escape(row.workload.target));
        text.push_back(',');
        text.append(csv_escape(row.workload.case_name));
        text.push_back(',');
        text.append(csv_escape(row.workload.variant));
        text.push_back(',');
        text.append(artifact_class_name(row.workload.artifact_class));
        text.push_back(',');
        text.append(row_status_name(row.status));
        text.push_back(',');
        text.append(row.correct ? "true" : "false");
        text.push_back(',');
    };
    for (const auto &row : report.rows) {
        const auto append_metric = [&](BenchmarkMetric metric, const luisa::optional<Statistics> &stats) {
            if (!stats.has_value()) { return; }
            append_fields(row);
            text.append(statistics_csv_row(benchmark_metric_name(metric), *stats));
            text.push_back('\n');
        };
        // Never emit NaN/Inf: validate before any byte is written.
        if (row.record_ms.has_value() && !all_finite(*row.record_ms)) {
            return Status::error(StatusCode::invalid_argument, "non-finite record_ms statistics", "csv");
        }
        if (row.submit_sync_ms.has_value() && !all_finite(*row.submit_sync_ms)) {
            return Status::error(StatusCode::invalid_argument, "non-finite submit_sync_ms statistics", "csv");
        }
        if (row.request_ms.has_value() && !all_finite(*row.request_ms)) {
            return Status::error(StatusCode::invalid_argument, "non-finite request_ms statistics", "csv");
        }
        append_metric(BenchmarkMetric::record_ms, row.record_ms);
        append_metric(BenchmarkMetric::submit_sync_ms, row.submit_sync_ms);
        append_metric(BenchmarkMetric::request_ms, row.request_ms);
    }
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "ab");
    if (file == nullptr) {
        return Status::error(StatusCode::io_error,
                             luisa::string{"cannot open CSV file '"} + path_text + "'", "csv");
    }
    const bool wrote = text.empty() || std::fwrite(text.data(), 1, text.size(), file) == text.size();
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !closed) {
        return Status::error(StatusCode::io_error,
                             luisa::string{"failed to append CSV file '"} + path_text + "'", "csv");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Baseline input.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::vector<BaselineEntry>> read_baseline_json(const luisa::filesystem::path &path) {
    using R = Result<luisa::vector<BaselineEntry>>;
    if (path.empty()) {
        return R::failure(StatusCode::invalid_argument, "baseline path is empty", "baseline");
    }
    const auto path_text = luisa::to_string(path);
    yyjson_read_err err{};
    Doc holder{yyjson_read_file(path_text.c_str(), 0, nullptr, &err)};
    if (holder.doc == nullptr) {
        return R::failure(StatusCode::io_error,
                          luisa::string{"cannot read baseline JSON '"} + path_text + "'", "baseline");
    }
    auto *root = yyjson_doc_get_root(holder.doc);
    auto *rows = yyjson_obj_get(root, "rows");
    if (rows == nullptr || !yyjson_is_arr(rows)) {
        return R::failure(StatusCode::invalid_argument,
                          "baseline JSON has no \"rows\" array", "baseline");
    }
    luisa::vector<BaselineEntry> entries;
    size_t idx = 0, max = 0;
    yyjson_val *row = nullptr;
    yyjson_arr_foreach(rows, idx, max, row) {
        if (!yyjson_is_obj(row)) { continue; }
        auto *workload = yyjson_obj_get(row, "workload");
        auto *stats = yyjson_obj_get(row, "statistics");
        if (workload == nullptr || !yyjson_is_obj(workload) ||
            stats == nullptr || !yyjson_is_obj(stats)) { continue; }
        auto *id = yyjson_obj_get(workload, "id");
        if (id == nullptr || !yyjson_is_str(id)) { continue; }
        luisa::string workload_id{yyjson_get_str(id), yyjson_get_len(id)};
        luisa::string variant;
        if (auto *v = yyjson_obj_get(workload, "variant"); v != nullptr && yyjson_is_str(v)) {
            variant = luisa::string{yyjson_get_str(v), yyjson_get_len(v)};
        }
        for (const auto *metric_name : {"record_ms", "submit_sync_ms", "request_ms"}) {
            auto *m = yyjson_obj_get(stats, metric_name);
            if (m == nullptr || !yyjson_is_obj(m)) { continue; }
            auto *median = yyjson_obj_get(m, "median");
            if (median == nullptr || !yyjson_is_num(median)) { continue; }
            const auto median_ms = yyjson_get_num(median);
            if (!std::isfinite(median_ms)) { continue; }
            BenchmarkMetric metric = BenchmarkMetric::record_ms;
            if (std::string_view{metric_name} == "submit_sync_ms") {
                metric = BenchmarkMetric::submit_sync_ms;
            } else if (std::string_view{metric_name} == "request_ms") {
                metric = BenchmarkMetric::request_ms;
            }
            entries.emplace_back(BaselineEntry{workload_id, variant, metric, median_ms});
        }
    }
    return R::success(std::move(entries));
}

[[nodiscard]] luisa::optional<BaselineEntry> find_baseline_entry(
    const luisa::vector<BaselineEntry> &entries, const luisa::string &workload_id,
    const luisa::string &variant, BenchmarkMetric metric) {
    for (const auto &entry : entries) {
        if (entry.workload_id == workload_id && entry.variant == variant && entry.metric == metric) {
            return entry;
        }
    }
    return luisa::nullopt;
}

}// namespace luisa::ocr
