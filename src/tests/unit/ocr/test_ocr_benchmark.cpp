// Test for examples/ocr/ocr/benchmark.h/.cpp -- the host-safe benchmark
// utility: options parsing, canonical workload identity, sample statistics,
// baseline comparison, JSON/CSV reporting. Host-only: no device, no backend,
// no model, no exceptions.
//
// This test covers:
// - options: defaults, full parse, malformed argv (missing backend/suite,
//   unknown suite, unknown/duplicate options, missing values, bad numbers,
//   stray positionals), synthetic/real labels + pinned-input requirements,
//   --help short-circuit, real-execution fail-closed (unvalidated);
// - canonical identity: same content -> same key/fingerprint (pair order
//   insensitive), any code/shape difference -> different fingerprint;
// - statistics: even/odd median, nearest-rank p95, sample stddev (n-1), MAD,
//   1.96 mean CI, empty/nonfinite rejection;
// - finalize_row: per-metric summaries, invalidation of bad rows;
// - baseline comparison: percent delta, 5% operator / 3% request thresholds,
//   noise floor, independent-run promotion gate;
// - JSON: valid finite roundtrip (yyjson), raw sample preservation,
//   nonfinite rejection;
// - CSV: quoting, header written once across appends;
// - baseline JSON reading: roundtrip through a serialized report, malformed
//   input errors.
// Report/baseline files live in the system temp dir and are cleaned up.
#include "ut/ut.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include <luisa/core/stl/filesystem.h>
#include <yyjson.h>

#include "ocr/benchmark.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// A unique temp directory for report/baseline/CSV files (cleaned at scope exit).
struct TempDir {
    luisa::filesystem::path path;
    TempDir() {
        auto base = luisa::filesystem::temp_directory_path();
        path = base / ("lcocr_bench_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

[[nodiscard]] Result<BenchmarkOptions> parse(std::initializer_list<const char *> args) {
    luisa::vector<const char *> argv;
    argv.emplace_back("example_ocr_bench");
    for (auto a : args) { argv.emplace_back(a); }
    return parse_benchmark_options(static_cast<int>(argv.size()), argv.data());
}

[[nodiscard]] WorkloadDescriptor make_workload() {
    WorkloadDescriptor w;
    w.id = "decode/glm_ocr/prefill";
    w.suite = BenchmarkSuite::decode;
    w.target = "glm_ocr";
    w.case_name = "prefill";
    w.variant = "persistent";
    w.artifact_class = ArtifactClass::synthetic;
    w.seed = 42;
    w.shape.emplace_back(KeyValue{"tokens", "128"});
    w.shape.emplace_back(KeyValue{"hidden", "1024"});
    w.options.emplace_back(KeyValue{"graph", "off"});
    w.pages = 1;
    w.prompt_tokens = 128;
    w.output_tokens = 16;
    w.dispatches = 4;
    w.flops = 1000;
    w.bytes = 2000;
    w.weight_bytes = 1500;
    return w;
}

[[nodiscard]] Statistics stats_of(std::initializer_list<double> values) {
    luisa::vector<double> v;
    for (auto x : values) { v.push_back(x); }
    auto result = summarize_samples(v);
    expect(result.ok());
    return result.ok() ? *result.value : Statistics{};
}

[[nodiscard]] bool is_valid_json(const luisa::string &text) {
    yyjson_doc *doc = yyjson_read(text.data(), text.size(), 0);
    if (doc == nullptr) { return false; }
    yyjson_doc_free(doc);
    return true;
}

[[nodiscard]] luisa::string read_file(const luisa::filesystem::path &path) {
    std::ifstream in{path, std::ios::binary};
    std::ostringstream ss;
    ss << in.rdbuf();
    return luisa::string{ss.str()};
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

void test_option_defaults() {
    const auto parsed = parse({"vk", "ops"});
    expect(parsed.ok());
    if (!parsed.ok()) { return; }
    const auto &options = *parsed.value;
    expect(!options.help);
    expect(options.backend == "vk");
    expect(options.suite == BenchmarkSuite::ops);
    expect(options.target.empty());
    expect(options.case_name.empty());
    expect(options.variant.empty());
    expect(options.workload.empty());
    expect(options.artifact_class == ArtifactClass::synthetic);
    expect(options.warmup == 3u);
    expect(options.iterations == 20u);
    expect(options.dispatch_batch == 1u);
    expect(!options.graph);
    expect(!options.fusion);
    expect(options.cooperative_norm);
    expect(options.cache == BenchmarkCacheMode::miss);
    expect(!options.instrument);
    expect(options.json_path.empty());
    expect(options.csv_path.empty());
    expect(options.baseline_path.empty());
    expect(options.package_path.empty());
    expect(options.manifest_path.empty());
    expect(options.provenance.empty());
    expect(options.pretty);
    expect(artifact_class_name(options.artifact_class) == "synthetic");
}

void test_option_full_parse() {
    const auto parsed = parse({"cuda", "decode",
                               "--target", "glm_ocr",
                               "--case", "prefill",
                               "--variant", "v1",
                               "--workload", "w/1",
                               "--artifact-class", "synthetic",
                               "--warmup", "5",
                               "--iterations", "50",
                               "--dispatch-batch", "4",
                               "--graph", "on",
                               "--fusion", "on",
                               "--cooperative-norm", "off",
                               "--cache", "hit",
                               "--instrument", "on",
                               "--json", "out.json",
                               "--csv", "out.csv",
                               "--baseline", "base.json",
                               "--source-revision", "abc123",
                               "--source-tree-state", "clean",
                               "--gpu-name", "RTX 4060",
                               "--driver-version", "551.0",
                               "--provenance", "cuda_runtime=13.0",
                               "--provenance", "power_state=default",
                               "--no-pretty"});
    expect(parsed.ok());
    if (!parsed.ok()) { return; }
    const auto &options = *parsed.value;
    expect(options.backend == "cuda");
    expect(options.suite == BenchmarkSuite::decode);
    expect(options.target == "glm_ocr");
    expect(options.case_name == "prefill");
    expect(options.variant == "v1");
    expect(options.workload == "w/1");
    expect(options.warmup == 5u);
    expect(options.iterations == 50u);
    expect(options.dispatch_batch == 4u);
    expect(options.graph);
    expect(options.fusion);
    expect(!options.cooperative_norm);
    expect(options.cache == BenchmarkCacheMode::hit);
    expect(options.instrument);
    expect(options.json_path == luisa::filesystem::path{"out.json"});
    expect(options.csv_path == luisa::filesystem::path{"out.csv"});
    expect(options.baseline_path == luisa::filesystem::path{"base.json"});
    expect(options.source_revision == "abc123");
    expect(options.source_tree_state == "clean");
    expect(options.gpu_name == "RTX 4060");
    expect(options.driver_version == "551.0");
    expect(options.provenance.size() == 2u);
    expect(options.provenance[0] == "cuda_runtime=13.0");
    expect(options.provenance[1] == "power_state=default");
    expect(!options.pretty);
    // Cache/graph/fusion/instrument round-trip through their stable names.
    expect(benchmark_cache_mode_name(options.cache) == "hit");
}

void test_option_malformed() {
    expect(parse({}).status.code == StatusCode::invalid_argument);// no backend
    expect(parse({"vk"}).status.code == StatusCode::invalid_argument);// no suite
    expect(parse({"vk", "bogus"}).status.code == StatusCode::invalid_argument);
    expect(parse({"--json", "x"}).status.code == StatusCode::invalid_argument);// backend missing
    expect(parse({"vk", "ops", "extra"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--bogus"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--target"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--target", "a", "--target", "b"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--no-pretty", "--no-pretty"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--warmup", "x"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--warmup", "1.5"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--warmup", "-1"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--iterations", "0"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--iterations", "12x"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--dispatch-batch", "0"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--graph", "yes"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--fusion", "maybe"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--cooperative-norm", "maybe"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--cooperative-norm", "on", "--cooperative-norm", "off"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--cache", "warm"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--artifact-class", "bogus"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--provenance", "missing_equals"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--provenance", "=value"}).status.code == StatusCode::invalid_argument);
    expect(parse({"vk", "ops", "--provenance", "key="}).status.code == StatusCode::invalid_argument);
    // Duplicate detection reports the offending option in `detail`.
    const auto dup = parse({"vk", "ops", "--cache", "hit", "--cache", "miss"});
    expect(dup.status.code == StatusCode::invalid_argument);
    expect(dup.status.detail == "cache");
    // Warmup may be 0 (unlike iterations/dispatch-batch, which must be >= 1).
    expect(parse({"vk", "ops", "--warmup", "0"}).ok());
    // Every suite name parses.
    for (auto name : {"ops", "decode", "cnn", "layout", "gdn", "moe", "model", "all"}) {
        expect(parse({"vk", name}).ok()) << name;
    }
}

void test_option_help() {
    const auto parsed = parse({"--help"});
    expect(parsed.ok());
    expect(parsed.value->help);
    // --help short-circuits even alongside other (possibly invalid) arguments.
    const auto mixed = parse({"cuda", "ops", "--help", "--bogus"});
    expect(mixed.ok());
    expect(mixed.value->help);
    expect(!benchmark_usage_text().empty());
    expect(benchmark_usage_text().find("--artifact-class") != luisa::string::npos);
}

void test_option_real_inputs_and_fail_closed() {
    // Real runs require pinned --package and --manifest at parse time.
    expect(parse({"vk", "model", "--artifact-class", "real"}).status.code ==
           StatusCode::invalid_argument);
    expect(parse({"vk", "model", "--artifact-class", "real", "--package", "p.lcocr"}).status.code ==
           StatusCode::invalid_argument);
    const auto parsed = parse({"vk", "model", "--artifact-class", "real",
                               "--package", "p.lcocr", "--manifest", "m.json"});
    expect(parsed.ok());
    if (!parsed.ok()) { return; }
    expect(parsed.value->artifact_class == ArtifactClass::real);
    expect(artifact_class_name(parsed.value->artifact_class) == "real");
    expect(parsed.value->package_path == luisa::filesystem::path{"p.lcocr"});
    expect(parsed.value->manifest_path == luisa::filesystem::path{"m.json"});

    // The execution gate is separate and fail-closed: synthetic passes,
    // real validates inputs first and then refuses with `unvalidated`.
    const auto synthetic = parse({"vk", "ops"});
    expect(synthetic.ok());
    expect(check_real_execution(*synthetic.value).ok());
    expect(check_real_execution(*parsed.value).code == StatusCode::unvalidated);
    // Argument validation happens BEFORE the unvalidated refusal.
    BenchmarkOptions bad = *parsed.value;
    bad.manifest_path.clear();
    expect(check_real_execution(bad).code == StatusCode::invalid_argument);
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

void test_identity() {
    const auto a = make_workload();
    auto b = make_workload();
    expect(canonical_workload_key(a) == canonical_workload_key(b));
    expect(workload_fingerprint64(a) == workload_fingerprint64(b));
    // Pair order does not matter (canonicalization sorts).
    std::swap(b.shape[0], b.shape[1]);
    expect(canonical_workload_key(a) == canonical_workload_key(b));
    expect(workload_fingerprint64(a) == workload_fingerprint64(b));
    // Every code/shape-affecting difference changes the fingerprint.
    {
        auto w = make_workload();
        w.shape[1].value = "2048";
        expect(workload_fingerprint64(w) != workload_fingerprint64(a));
    }
    {
        auto w = make_workload();
        w.options[0].value = "on";
        expect(workload_fingerprint64(w) != workload_fingerprint64(a));
    }
    {
        auto w = make_workload();
        w.seed = 43;
        expect(workload_fingerprint64(w) != workload_fingerprint64(a));
    }
    {
        auto w = make_workload();
        w.variant = "naive";
        expect(workload_fingerprint64(w) != workload_fingerprint64(a));
    }
    {
        auto w = make_workload();
        w.artifact_class = ArtifactClass::real;
        expect(workload_fingerprint64(w) != workload_fingerprint64(a));
    }
    {
        auto w = make_workload();
        w.dispatches = 5;
        expect(workload_fingerprint64(w) != workload_fingerprint64(a));
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

void test_statistics_even_odd() {
    const auto even = stats_of({1.0, 2.0, 3.0, 4.0});
    expect(even.n == 4u);
    expect(std::fabs(even.median - 2.5) < 1e-12);// average of middle two
    expect(std::fabs(even.min - 1.0) < 1e-12);
    expect(std::fabs(even.max - 4.0) < 1e-12);
    expect(std::fabs(even.mean - 2.5) < 1e-12);
    const auto odd = stats_of({1.0, 2.0, 3.0});
    expect(odd.n == 3u);
    expect(std::fabs(odd.median - 2.0) < 1e-12);
}

void test_statistics_p95_nearest_rank() {
    // n = 10: ceil(0.95 * 10) = 10 -> sorted[9] = 10.
    luisa::vector<double> ten;
    for (int i = 1; i <= 10; i++) { ten.push_back(static_cast<double>(i)); }
    auto s = summarize_samples(ten);
    expect(s.ok());
    expect(std::fabs(s.value->p95 - 10.0) < 1e-12);
    // n = 20: ceil(0.95 * 20) = 19 -> sorted[18] = 19.
    luisa::vector<double> twenty;
    for (int i = 1; i <= 20; i++) { twenty.push_back(static_cast<double>(i)); }
    s = summarize_samples(twenty);
    expect(s.ok());
    expect(std::fabs(s.value->p95 - 19.0) < 1e-12);
    // n = 1: the single sample.
    s = summarize_samples({7.5});
    expect(s.ok());
    expect(std::fabs(s.value->p95 - 7.5) < 1e-12);
    expect(std::fabs(s.value->stddev) < 1e-12);
    expect(std::fabs(s.value->ci95_low - 7.5) < 1e-12);
    expect(std::fabs(s.value->ci95_high - 7.5) < 1e-12);
}

void test_statistics_stddev_mad_ci() {
    const auto s = stats_of({2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0});
    expect(s.n == 8u);
    expect(std::fabs(s.mean - 5.0) < 1e-12);
    // Sample standard deviation: sqrt(32 / 7).
    expect(std::fabs(s.stddev - std::sqrt(32.0 / 7.0)) < 1e-9);
    // Median is (4+5)/2 = 4.5. Absolute deviations from 4.5 are
    // {2.5,0.5,0.5,0.5,0.5,0.5,2.5,4.5} -> MAD = 0.5.
    expect(std::fabs(s.median - 4.5) < 1e-12);
    expect(std::fabs(s.mad - 0.5) < 1e-12);
    // 95% CI of the mean: mean +/- 1.96 * stddev / sqrt(n).
    const auto half = 1.96 * s.stddev / std::sqrt(8.0);
    expect(std::fabs((s.ci95_high - s.ci95_low) - 2.0 * half) < 1e-9);
    expect(std::fabs((s.ci95_low + s.ci95_high) * 0.5 - s.mean) < 1e-9);
    // Classic MAD check: {1,1,2,2,4,6,6} -> median 2, MAD 1.
    const auto m = stats_of({1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 6.0});
    expect(std::fabs(m.median - 2.0) < 1e-12);
    expect(std::fabs(m.mad - 1.0) < 1e-12);
}

void test_statistics_rejects_empty_and_nonfinite() {
    expect(!summarize_samples({}).ok());
    expect(summarize_samples({}).status.code == StatusCode::invalid_argument);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    expect(!summarize_samples({1.0, nan}).ok());
    expect(!summarize_samples({inf}).ok());
}

// ---------------------------------------------------------------------------
// finalize_row
// ---------------------------------------------------------------------------

void test_finalize_row() {
    Row row;
    row.workload = make_workload();
    row.correct = true;
    row.samples.emplace_back(Sample{1.0, 10.0, 100.0});
    row.samples.emplace_back(Sample{2.0, 20.0, luisa::nullopt});
    row.samples.emplace_back(Sample{3.0, luisa::nullopt, luisa::nullopt});
    expect(finalize_row(row).ok());
    expect(row.status == RowStatus::ok);
    expect(row.record_ms.has_value());
    expect(row.submit_sync_ms.has_value());
    expect(row.request_ms.has_value());
    expect(row.record_ms->n == 3u);
    expect(row.submit_sync_ms->n == 2u);
    expect(row.request_ms->n == 1u);
    expect(std::fabs(row.record_ms->median - 2.0) < 1e-12);
    // p95 of {1,2,3}: ceil(2.85) = 3 -> sorted[2] = 3.
    expect(std::fabs(row.record_ms->p95 - 3.0) < 1e-12);
}

void test_finalize_row_invalidates() {
    // A non-finite sample invalidates the row and clears statistics.
    Row row;
    row.workload = make_workload();
    row.samples.emplace_back(Sample{1.0, luisa::nullopt, luisa::nullopt});
    row.samples.emplace_back(Sample{std::numeric_limits<double>::quiet_NaN(),
                                    luisa::nullopt, luisa::nullopt});
    expect(finalize_row(row).ok());
    expect(row.status == RowStatus::invalid);
    expect(!row.reason.ok());
    expect(row.reason.code == StatusCode::invalid_argument);
    expect(!row.record_ms.has_value());
    Row bad_stage;
    bad_stage.workload = make_workload();
    bad_stage.samples.emplace_back(Sample{1.0, 2.0, 3.0});
    bad_stage.samples.back().stage.vision_ms = std::numeric_limits<double>::infinity();
    expect(finalize_row(bad_stage).ok());
    expect(bad_stage.status == RowStatus::invalid);
    expect(!bad_stage.stage.vision_ms.has_value());
    // An empty workload id invalidates the row.
    Row empty_id;
    empty_id.samples.emplace_back(Sample{1.0, luisa::nullopt, luisa::nullopt});
    expect(finalize_row(empty_id).ok());
    expect(empty_id.status == RowStatus::invalid);
    // A pre-failed row is left untouched.
    Row failed;
    failed.workload = make_workload();
    failed.status = RowStatus::failed;
    failed.reason = Status::error(StatusCode::timeout, "wall budget exceeded");
    failed.samples.emplace_back(Sample{1.0, luisa::nullopt, luisa::nullopt});
    expect(finalize_row(failed).ok());
    expect(failed.status == RowStatus::failed);
    expect(!failed.record_ms.has_value());
}

// ---------------------------------------------------------------------------
// Baseline comparison
// ---------------------------------------------------------------------------

void test_baseline_comparison() {
    const auto baseline = stats_of({100.0, 100.0, 100.0});
    // 10% slower on an operator metric -> regression past the 5% threshold.
    {
        const auto current = stats_of({110.0, 110.0, 110.0});
        const auto c = compare_median_to_baseline(BenchmarkMetric::record_ms, current,
                                                  baseline, default_noise_floor_percent());
        expect(c.outcome == ComparisonOutcome::regression);
        expect(std::fabs(c.median_delta_percent - 10.0) < 1e-9);
        expect(std::fabs(c.threshold_percent - 5.0) < 1e-12);
        // Promotion needs the delta AND at least two independent runs.
    expect(comparison_supports_promotion(c, 3u));
    expect(comparison_supports_promotion(c, 3u));
    expect(!comparison_supports_promotion(c, 2u));
    expect(!comparison_supports_promotion(c, 0u));
        expect(comparison_outcome_name(c.outcome) == "regression");
    }
    // 0.5% is inside the noise floor -> not significant on any metric.
    {
        const auto current = stats_of({100.5, 100.5, 100.5});
        const auto c = compare_median_to_baseline(BenchmarkMetric::record_ms, current,
                                                  baseline, default_noise_floor_percent());
        expect(c.outcome == ComparisonOutcome::not_significant);
        expect(!comparison_supports_promotion(c, 4u));
    }
    // The request metric promotes at 3%: 3.5% is a regression, 2.5% is not.
    {
        const auto slow = stats_of({103.5, 103.5, 103.5});
        const auto c = compare_median_to_baseline(BenchmarkMetric::request_ms, slow,
                                                  baseline, default_noise_floor_percent());
        expect(c.outcome == ComparisonOutcome::regression);
        expect(std::fabs(c.threshold_percent - 3.0) < 1e-12);
        const auto fast_enough = stats_of({102.5, 102.5, 102.5});
        const auto c2 = compare_median_to_baseline(BenchmarkMetric::request_ms, fast_enough,
                                                   baseline, default_noise_floor_percent());
        expect(c2.outcome == ComparisonOutcome::not_significant);
    }
    // 10% faster -> improvement; promotion still needs three runs.
    {
        const auto current = stats_of({90.0, 90.0, 90.0});
        const auto c = compare_median_to_baseline(BenchmarkMetric::submit_sync_ms, current,
                                                  baseline, default_noise_floor_percent());
        expect(c.outcome == ComparisonOutcome::improvement);
        expect(comparison_supports_promotion(c, 3u));
        expect(!comparison_supports_promotion(c, 2u));
        expect(!comparison_supports_promotion(c, 0u));
        expect(comparison_outcome_name(c.outcome) == "improvement");
    }
    // A zero baseline median cannot yield a percent delta: not significant.
    {
        const auto zero = stats_of({0.0, 0.0, 0.0});
        const auto current = stats_of({50.0, 50.0, 50.0});
        const auto c = compare_median_to_baseline(BenchmarkMetric::record_ms, current,
                                                  zero, default_noise_floor_percent());
        expect(c.outcome == ComparisonOutcome::not_significant);
        expect(std::fabs(c.median_delta_percent) < 1e-12);
    }
}

void test_baseline_read_and_match() {
    TempDir tmp;
    const auto path = tmp.path / "baseline.json";
    // Build a report, serialize it, and read it back as a baseline.
    Report report;
    report.environment.source_revision = "rev1";
    Row row;
    row.workload = make_workload();
    row.correct = true;
    row.samples.emplace_back(Sample{10.0, luisa::nullopt, 100.0});
    row.samples.emplace_back(Sample{20.0, luisa::nullopt, 200.0});
    row.samples.emplace_back(Sample{30.0, luisa::nullopt, 300.0});
    expect(finalize_row(row).ok());
    report.rows.emplace_back(row);
    expect(write_report(report, path).ok());

    const auto entries = read_baseline_json(path);
    expect(entries.ok());
    if (!entries.ok()) { return; }
    expect(entries.value->size() == 2u);// record_ms + request_ms
    const auto record = find_baseline_entry(*entries.value, row.workload.id,
                                            row.workload.variant, BenchmarkMetric::record_ms);
    expect(record.has_value());
    if (record.has_value()) {
        expect(std::fabs(record->baseline_median_ms - 20.0) < 1e-12);
    }
    const auto request = find_baseline_entry(*entries.value, row.workload.id,
                                             row.workload.variant, BenchmarkMetric::request_ms);
    expect(request.has_value());
    if (request.has_value()) {
        expect(std::fabs(request->baseline_median_ms - 200.0) < 1e-12);
    }
    expect(!find_baseline_entry(*entries.value, row.workload.id, row.workload.variant,
                                BenchmarkMetric::submit_sync_ms).has_value());
    expect(!find_baseline_entry(*entries.value, "other/id", row.workload.variant,
                                BenchmarkMetric::record_ms).has_value());
    expect(!find_baseline_entry(*entries.value, row.workload.id, "other",
                                BenchmarkMetric::record_ms).has_value());

    // Malformed input: io_error; valid JSON without rows: invalid_argument.
    const auto bad = tmp.path / "bad.json";
    {
        std::ofstream out{bad, std::ios::binary};
        out << "not json";
    }
    expect(read_baseline_json(bad).status.code == StatusCode::io_error);
    const auto no_rows = tmp.path / "no_rows.json";
    {
        std::ofstream out{no_rows, std::ios::binary};
        out << "{\"schema_version\":\"x\"}";
    }
    expect(read_baseline_json(no_rows).status.code == StatusCode::invalid_argument);
    expect(read_baseline_json(tmp.path / "missing.json").status.code == StatusCode::io_error);
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

[[nodiscard]] Report make_report() {
    Report report;
    report.environment.source_revision = "rev123";
    report.environment.source_tree_state = "clean";
    report.environment.build_type = "release";
    report.environment.compiler = "test-compiler";
    report.environment.os = "test-os";
    report.environment.backend = "vk";
    report.environment.cpu = "test-cpu";
    report.environment.gpu = "test-gpu";
    report.environment.driver_version = "1.0";
    report.environment.luisa_version = "test";
    report.environment.warp_size = 32;
    report.environment.shared_memory_bytes = 49152;
    report.environment.total_device_bytes = 8589934592ull;
    report.environment.cache = BenchmarkCacheMode::miss;
    report.environment.provenance.emplace_back(KeyValue{"package_hash", "abc"});
    report.options.backend = "vk";
    report.options.suite = BenchmarkSuite::decode;
    report.options.iterations = 3;

    Row row;
    row.workload = make_workload();
    row.correct = true;
    row.samples.emplace_back(Sample{1.0, 10.0, 100.0});
    row.samples.emplace_back(Sample{2.0, 20.0, 200.0});
    row.samples.emplace_back(Sample{3.0, 30.0, 300.0});
    for (size_t i = 0; i < row.samples.size(); i++) {
        row.samples[i].stage.vision_ms = 4.0 + static_cast<double>(i);
        row.samples[i].stage.decode_ms = 40.0 + static_cast<double>(i) * 10.0;
    }
    expect(finalize_row(row).ok());
    row.counters.emplace_back(KeyValue{"pages", "1"});
    // Baseline median 1ms vs current median 2ms: +100% -> regression.
    const auto baseline = stats_of({1.0, 1.0, 1.0});
    row.comparisons.emplace_back(compare_median_to_baseline(
        BenchmarkMetric::record_ms, *row.record_ms, baseline, default_noise_floor_percent()));
    report.rows.emplace_back(row);

    Row skipped;
    skipped.workload = make_workload();
    skipped.workload.id = "decode/glm_ocr/decode_step";
    skipped.status = RowStatus::skipped;
    skipped.reason = Status::error(StatusCode::unsupported, "suite filtered out");
    report.rows.emplace_back(skipped);
    return report;
}

void test_json_roundtrip() {
    const auto report = make_report();
    const auto json = serialize_report(report);
    expect(json.ok());
    if (!json.ok()) { return; }
    expect(is_valid_json(*json.value));
    // The document parses and preserves the raw rows/samples exactly.
    yyjson_doc *doc = yyjson_read(json.value->data(), json.value->size(), 0);
    expect(doc != nullptr);
    if (doc == nullptr) { return; }
    auto *root = yyjson_doc_get_root(doc);
    expect(yyjson_equals_str(yyjson_obj_get(root, "schema_version"), "luisa.ocr.benchmark.v1"));
    auto *env = yyjson_obj_get(root, "environment");
    expect(env != nullptr);
    expect(yyjson_equals_str(yyjson_obj_get(env, "source_revision"), "rev123"));
    expect(yyjson_get_uint(yyjson_obj_get(env, "warp_size")) == 32u);
    expect(yyjson_get_uint(yyjson_obj_get(env, "total_device_bytes")) == 8589934592ull);
    auto *options = yyjson_obj_get(root, "options");
    expect(options != nullptr);
    expect(yyjson_equals_str(yyjson_obj_get(options, "backend"), "vk"));
    expect(yyjson_equals_str(yyjson_obj_get(options, "suite"), "decode"));
    auto *rows = yyjson_obj_get(root, "rows");
    expect(rows != nullptr && yyjson_arr_size(rows) == 2u);
    if (rows == nullptr || yyjson_arr_size(rows) != 2u) {
        yyjson_doc_free(doc);
        return;
    }
    auto *row = yyjson_arr_get(rows, 0);
    expect(yyjson_equals_str(yyjson_obj_get(row, "status"), "ok"));
    // Raw samples: 3 entries with all three metrics preserved.
    auto *samples = yyjson_obj_get(row, "samples");
    expect(samples != nullptr && yyjson_arr_size(samples) == 3u);
    if (samples != nullptr && yyjson_arr_size(samples) == 3u) {
        auto *first = yyjson_arr_get(samples, 0);
        expect(yyjson_get_real(yyjson_obj_get(first, "record_ms")) == 1.0);
        expect(yyjson_get_real(yyjson_obj_get(first, "submit_sync_ms")) == 10.0);
        expect(yyjson_get_real(yyjson_obj_get(first, "request_ms")) == 100.0);
        auto *stage = yyjson_obj_get(first, "stage");
        expect(stage != nullptr);
        expect(yyjson_get_real(yyjson_obj_get(stage, "vision_ms")) == 4.0);
        expect(yyjson_get_real(yyjson_obj_get(stage, "decode_ms")) == 40.0);
        expect(yyjson_obj_get(stage, "preprocess_ms") == nullptr);
        auto *last = yyjson_arr_get(samples, 2);
        expect(yyjson_get_real(yyjson_obj_get(last, "request_ms")) == 300.0);
    }
    // Statistics: record_ms n=3, median 2; the comparison is serialized.
    auto *stats = yyjson_obj_get(row, "statistics");
    expect(stats != nullptr);
    if (stats != nullptr) {
        auto *record = yyjson_obj_get(stats, "record_ms");
        expect(record != nullptr);
        expect(yyjson_get_uint(yyjson_obj_get(record, "n")) == 3u);
        expect(yyjson_get_real(yyjson_obj_get(record, "median")) == 2.0);
        auto *stage = yyjson_obj_get(stats, "stage");
        expect(stage != nullptr);
        auto *vision = yyjson_obj_get(stage, "vision_ms");
        expect(vision != nullptr);
        expect(yyjson_get_uint(yyjson_obj_get(vision, "n")) == 3u);
        expect(yyjson_get_real(yyjson_obj_get(vision, "median")) == 5.0);
        auto *decode = yyjson_obj_get(stage, "decode_ms");
        expect(decode != nullptr);
        expect(yyjson_get_real(yyjson_obj_get(decode, "median")) == 50.0);
        expect(yyjson_obj_get(stage, "preprocess_ms") == nullptr);
    }
    auto *comparisons = yyjson_obj_get(row, "comparisons");
    expect(comparisons != nullptr && yyjson_arr_size(comparisons) == 1u);
    if (comparisons != nullptr && yyjson_arr_size(comparisons) == 1u) {
        auto *c = yyjson_arr_get(comparisons, 0);
        expect(yyjson_equals_str(yyjson_obj_get(c, "outcome"), "regression"));
    }
    auto *counters = yyjson_obj_get(row, "counters");
    expect(counters != nullptr);
    expect(yyjson_equals_str(yyjson_obj_get(counters, "pages"), "1"));
    // Workload identity: id + fingerprint are present.
    auto *workload = yyjson_obj_get(row, "workload");
    expect(workload != nullptr);
    expect(yyjson_equals_str(yyjson_obj_get(workload, "id"), "decode/glm_ocr/prefill"));
    expect(yyjson_obj_get(workload, "fingerprint") != nullptr);
    // The skipped row keeps its status and reason.
    auto *row2 = yyjson_arr_get(rows, 1);
    expect(yyjson_equals_str(yyjson_obj_get(row2, "status"), "skipped"));
    auto *reason = yyjson_obj_get(row2, "reason");
    expect(reason != nullptr);
    expect(yyjson_equals_str(yyjson_obj_get(reason, "code"), "unsupported"));
    yyjson_doc_free(doc);

    // write_report appends a trailing newline and round-trips through a file.
    TempDir tmp;
    const auto path = tmp.path / "report.json";
    expect(write_report(report, path).ok());
    const auto text = read_file(path);
    expect(!text.empty());
    expect(text.back() == '\n');
    expect(is_valid_json(text));
}

void test_json_rejects_nonfinite() {
    auto report = make_report();
    // A non-finite statistic must fail serialization (never invalid JSON).
    report.rows[0].record_ms->mean = std::numeric_limits<double>::quiet_NaN();
    expect(!serialize_report(report).ok());
    // A non-finite raw sample (row not finalized) must also fail.
    auto report2 = make_report();
    report2.rows[0].samples[1].request_ms = std::numeric_limits<double>::infinity();
    expect(!serialize_report(report2).ok());
    auto report_stage = make_report();
    report_stage.rows[0].samples[1].stage.decode_ms = std::numeric_limits<double>::quiet_NaN();
    expect(!serialize_report(report_stage).ok());
    // A non-finite comparison must fail.
    auto report3 = make_report();
    report3.rows[0].comparisons[0].median_delta_percent =
        std::numeric_limits<double>::quiet_NaN();
    expect(!serialize_report(report3).ok());
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

void test_csv_quoting() {
    expect(csv_escape("plain") == "plain");
    expect(csv_escape("with space") == "with space");
    expect(csv_escape("a,b") == "\"a,b\"");
    expect(csv_escape("a\"b") == "\"a\"\"b\"");
    expect(csv_escape("a\nb") == "\"a\nb\"");
    expect(csv_escape("a\r\nb") == "\"a\r\nb\"");
    expect(csv_escape("") == "");
}

void test_csv_append_header_once() {
    TempDir tmp;
    const auto path = tmp.path / "report.csv";
    const auto report = make_report();
    expect(append_report_csv(report, path).ok());
    expect(append_report_csv(report, path).ok());// append again
    const auto text = read_file(path);
    // Header exactly once: 1 header + 2 appends x 3 metric lines (the ok row
    // carries record_ms/submit_sync_ms/request_ms; the skipped row has none).
    const auto header_count = [](const luisa::string &t) {
        size_t count = 0;
        size_t pos = 0;
        while ((pos = t.find("workload_id,suite,target", pos)) != luisa::string::npos) {
            count++;
            pos += 1;
        }
        return count;
    };
    expect(header_count(text) == 1u);
    size_t lines = 0;
    for (const auto c : text) {
        if (c == '\n') { lines++; }
    }
    expect(lines == 1u + 6u) << "lines=" << lines;
    // Fields are quoted: the workload id contains no comma here, but the
    // full header + the metric name appear verbatim.
    expect(text.find("decode/glm_ocr/prefill") != luisa::string::npos);
    expect(text.find("record_ms") != luisa::string::npos);
    expect(text.find("request_ms") != luisa::string::npos);
    expect(text.find("skipped") == luisa::string::npos);// no stats -> no line
}

void test_csv_quotes_fields_with_commas() {
    TempDir tmp;
    const auto path = tmp.path / "quoted.csv";
    auto report = make_report();
    report.rows[0].workload.id = "id,with\"quote";
    expect(append_report_csv(report, path).ok());
    const auto text = read_file(path);
    expect(text.find("\"id,with\"\"quote\"") != luisa::string::npos);
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

void test_stable_names() {
    expect(benchmark_suite_name(BenchmarkSuite::all) == "all");
    expect(benchmark_suite_name(static_cast<BenchmarkSuite>(99u)) == "unknown");
    expect(row_status_name(RowStatus::ok) == "ok");
    expect(row_status_name(RowStatus::failed) == "failed");
    expect(row_status_name(RowStatus::skipped) == "skipped");
    expect(row_status_name(RowStatus::blocked) == "blocked");
    expect(row_status_name(RowStatus::refused) == "refused");
    expect(row_status_name(RowStatus::invalid) == "invalid");
    expect(row_status_name(static_cast<RowStatus>(99u)) == "unknown");
    expect(artifact_class_name(ArtifactClass::synthetic) == "synthetic");
    expect(artifact_class_name(ArtifactClass::real) == "real");
    expect(benchmark_metric_name(BenchmarkMetric::record_ms) == "record_ms");
    expect(benchmark_metric_name(BenchmarkMetric::submit_sync_ms) == "submit_sync_ms");
    expect(benchmark_metric_name(BenchmarkMetric::request_ms) == "request_ms");
    expect(benchmark_cache_mode_name(BenchmarkCacheMode::disabled) == "disabled");
    expect(comparison_outcome_name(ComparisonOutcome::not_significant) == "not_significant");
}

}// namespace

static auto test_ocr_benchmark_registration = [] {
    "option_defaults"_test = [] { test_option_defaults(); };
    "option_full_parse"_test = [] { test_option_full_parse(); };
    "option_malformed"_test = [] { test_option_malformed(); };
    "option_help"_test = [] { test_option_help(); };
    "option_real_inputs_and_fail_closed"_test = [] { test_option_real_inputs_and_fail_closed(); };
    "identity"_test = [] { test_identity(); };
    "statistics_even_odd"_test = [] { test_statistics_even_odd(); };
    "statistics_p95_nearest_rank"_test = [] { test_statistics_p95_nearest_rank(); };
    "statistics_stddev_mad_ci"_test = [] { test_statistics_stddev_mad_ci(); };
    "statistics_rejects_empty_and_nonfinite"_test = [] { test_statistics_rejects_empty_and_nonfinite(); };
    "finalize_row"_test = [] { test_finalize_row(); };
    "finalize_row_invalidates"_test = [] { test_finalize_row_invalidates(); };
    "baseline_comparison"_test = [] { test_baseline_comparison(); };
    "baseline_read_and_match"_test = [] { test_baseline_read_and_match(); };
    "json_roundtrip"_test = [] { test_json_roundtrip(); };
    "json_rejects_nonfinite"_test = [] { test_json_rejects_nonfinite(); };
    "csv_quoting"_test = [] { test_csv_quoting(); };
    "csv_append_header_once"_test = [] { test_csv_append_header_once(); };
    "csv_quotes_fields_with_commas"_test = [] { test_csv_quotes_fields_with_commas(); };
    "stable_names"_test = [] { test_stable_names(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
