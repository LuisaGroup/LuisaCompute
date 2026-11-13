// example_ocr_bench -- controlled-hardware, correctness-gated OCR benchmark.
// Synthetic rows measure the declared engine graphs only. Real artifacts are
// refused while every registry recipe remains planned/unvalidated. This binary
// is not a CTest performance assertion and never regenerates golden data.
#include <cstdio>
#include <cwchar>
#include <utility>

#include <luisa/core/clock.h>
#include <luisa/runtime/context.h>

#include "bench/benchmark_device.h"
#include "ocr/output/json_output.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace luisa::ocr;

namespace {

#ifdef _WIN32
[[nodiscard]] bool wide_arg_to_utf8(const wchar_t *wide, luisa::string &out) noexcept {
    out.clear();
    if (wide == nullptr) { return true; }
    const auto length = static_cast<int>(::wcslen(wide));
    if (length == 0) { return true; }
    const int needed = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide, length,
                                             nullptr, 0, nullptr, nullptr);
    if (needed <= 0) { return false; }
    out.resize(static_cast<size_t>(needed));
    const int written = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide, length,
                                              out.data(), needed, nullptr, nullptr);
    if (written != needed) {
        out.clear();
        return false;
    }
    return true;
}
#endif

[[nodiscard]] int write_status_error(const Status &status, int code) {
    const auto written = output::write_status(status, {});
    return written.ok() ? code : 3;
}

void attach_baseline_comparisons(Report &report, const luisa::vector<BaselineEntry> &entries) {
    for (auto &row : report.rows) {
        const auto attach = [&](BenchmarkMetric metric, const luisa::optional<Statistics> &statistics) {
            if (!statistics) { return; }
            const auto baseline = find_baseline_entry(entries, row.workload.id, row.workload.variant, metric);
            if (baseline) {
                row.comparisons.push_back(compare_median_to_baseline(
                    metric, *statistics, Statistics{.n = 1, .median = baseline->baseline_median_ms},
                    default_noise_floor_percent()));
            }
        };
        attach(BenchmarkMetric::record_ms, row.record_ms);
        attach(BenchmarkMetric::submit_sync_ms, row.submit_sync_ms);
        attach(BenchmarkMetric::request_ms, row.request_ms);
    }
}

[[nodiscard]] int run_cli(int argc, const char *const *argv) {
    auto parsed = parse_benchmark_options(argc, argv);
    if (!parsed.ok()) { return write_status_error(parsed.status, 1); }
    Report report;
    report.options = std::move(*parsed.value);
    if (report.options.help) {
        const auto usage = benchmark_usage_text();
        return std::fputs(usage.c_str(), stdout) < 0 ? 3 : 0;
    }

    if (report.options.artifact_class == ArtifactClass::real) {
        const auto refusal = check_real_execution(report.options);
        bench::capture_process_environment(report);
        if (!refusal.ok()) {
            Row row;
            row.workload.suite = report.options.suite;
            row.workload.target = report.options.target.empty() ? "real_package" : report.options.target;
            row.workload.case_name = "qualified_public_path";
            row.workload.variant = "unvalidated";
            row.workload.artifact_class = ArtifactClass::real;
            row.workload.id = "real.qualified_public_path";
            row.status = RowStatus::refused;
            row.reason = refusal;
            report.rows.push_back(std::move(row));
            const auto write = write_report(report, report.options.json_path,
                                            BenchmarkJsonOptions{report.options.pretty});
            return write.ok() ? 2 : 3;
        }
    }

    luisa::Clock device_clock;
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(report.options.backend);
    const auto device_create_ms = device_clock.toc();
    bench::capture_device_environment(device, report);
    report.environment.provenance.push_back({"device_creation_ms", luisa::format("{}", device_create_ms)});
    report.environment.provenance.push_back({"process_series", "fresh_process_device_creation_then_warm_request_series"});
    report.environment.provenance.push_back({"package_path", luisa::to_string(report.options.package_path)});
    report.environment.provenance.push_back({"manifest_path", luisa::to_string(report.options.manifest_path)});

    auto run_status = bench::run_benchmark(device, report);
    if (!report.options.baseline_path.empty()) {
        auto baseline = read_baseline_json(report.options.baseline_path);
        if (!baseline.ok()) {
            if (run_status.ok()) { run_status = baseline.status; }
        } else {
            attach_baseline_comparisons(report, *baseline.value);
        }
    }
    // Always preserve rows completed before a runner/selection/report failure.
    const auto write = write_report(report, report.options.json_path,
                                    BenchmarkJsonOptions{report.options.pretty});
    if (!write.ok()) { return write_status_error(write, 3); }
    if (!run_status.ok()) { return write_status_error(run_status, 2); }
    for (const auto &row : report.rows) {
        if (row.status == RowStatus::failed || row.status == RowStatus::invalid) {
            return 2;
        }
    }
    return 0;
}

}// namespace

#ifdef _WIN32
int wmain(int argc, wchar_t *argv[]) {
    luisa::vector<luisa::string> utf8_storage;
    utf8_storage.reserve(static_cast<size_t>(argc));
    luisa::vector<const char *> utf8_argv;
    utf8_argv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; i++) {
        luisa::string converted;
        if (!wide_arg_to_utf8(argv[i], converted)) {
            return write_status_error(Status::error(StatusCode::invalid_argument,
                                                    "command-line argument is not representable as UTF-8", "argv"), 1);
        }
        utf8_storage.emplace_back(std::move(converted));
        utf8_argv.emplace_back(utf8_storage.back().c_str());
    }
    return run_cli(argc, utf8_argv.data());
}
#else
int main(int argc, char *argv[]) {
    return run_cli(argc, argv);
}
#endif
