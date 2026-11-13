// OCR benchmark device environment and suite dispatcher.
#include "benchmark_device.h"
#include "benchmark_runners.h"

#include <charconv>

#include <luisa/core/platform.h>
#include <luisa/version.h>

namespace luisa::ocr::bench {
namespace {

#ifndef LUISA_OCR_BENCHMARK_SOURCE_REVISION
#define LUISA_OCR_BENCHMARK_SOURCE_REVISION unknown
#endif
#define LUISA_OCR_BENCHMARK_STRINGIFY_IMPL(value) #value
#define LUISA_OCR_BENCHMARK_STRINGIFY(value) LUISA_OCR_BENCHMARK_STRINGIFY_IMPL(value)

[[nodiscard]] luisa::string compiler_name() {
#if defined(__clang__)
    return luisa::format("clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(_MSC_VER)
    return luisa::format("msvc {}", _MSC_VER);
#elif defined(__GNUC__)
    return luisa::format("gcc {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    return "unknown";
#endif
}

[[nodiscard]] luisa::string os_name() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

void append_family(compute::Device &device, const BenchmarkOptions &options,
                   BenchmarkSuite suite, luisa::vector<Row> &rows) {
    switch (suite) {
        case BenchmarkSuite::ops: append_ops_rows(device, options, rows); break;
        case BenchmarkSuite::decode: append_decode_rows(device, options, rows); break;
        case BenchmarkSuite::cnn: append_cnn_rows(device, options, rows); break;
        case BenchmarkSuite::layout: append_layout_rows(device, options, rows); break;
        case BenchmarkSuite::gdn: append_gdn_rows(device, options, rows); break;
        case BenchmarkSuite::moe: append_moe_rows(device, options, rows); break;
        case BenchmarkSuite::model: append_model_rows(device, options, rows); break;
        default: break;
    }
}

[[nodiscard]] Status emit_rows(Report &report, luisa::vector<Row> &rows) {
    for (auto &row : rows) {
        report.rows.emplace_back(std::move(row));
        // Incremental evidence: a later failing case cannot erase earlier rows.
        // Stdout is emitted once at the end so it remains ONE valid JSON document.
        if (!report.options.json_path.empty()) {
            if (auto status = write_report(report, report.options.json_path,
                                           BenchmarkJsonOptions{report.options.pretty}); !status.ok()) { return status; }
        }
        if (!report.options.csv_path.empty()) {
            Report one = report;
            one.rows.clear();
            one.rows.push_back(report.rows.back());
            if (auto status = append_report_csv(one, report.options.csv_path); !status.ok()) { return status; }
        }
    }
    return Status::success();
}

}// namespace

void capture_process_environment(Report &report) {
    auto &environment = report.environment;
    environment.source_revision = report.options.source_revision.empty() ?
                                      LUISA_OCR_BENCHMARK_STRINGIFY(LUISA_OCR_BENCHMARK_SOURCE_REVISION) : report.options.source_revision;
    environment.source_tree_state = report.options.source_tree_state.empty() ?
                                        "unknown (pass --source-tree-state for a pinned result)" :
                                        report.options.source_tree_state;
#ifdef NDEBUG
    environment.build_type = "release";
#else
    environment.build_type = "debug";
#endif
    environment.compiler = compiler_name();
    environment.os = os_name();
    environment.cpu = luisa::cpu_name();
    environment.gpu = report.options.gpu_name.empty() ?
                          "unreported (pass --gpu-name; no universal GPU-name query assumed)" : report.options.gpu_name;
    environment.driver_version = report.options.driver_version.empty() ?
                                     "unreported (pass --driver-version)" : report.options.driver_version;
    environment.luisa_version = luisa::format("{}.{}.{}", LUISA_COMPUTE_VERSION_MAJOR,
                                              LUISA_COMPUTE_VERSION_MINOR, LUISA_COMPUTE_VERSION_PATCH);
    environment.cache = report.options.cache;
    for (const auto &entry : report.options.provenance) {
        const auto equals = entry.find('=');
        if (equals != luisa::string::npos && equals != 0u && equals + 1u < entry.size()) {
            environment.provenance.push_back({entry.substr(0, equals), entry.substr(equals + 1u)});
        }
    }
}

void capture_device_environment(compute::Device &device, Report &report) {
    capture_process_environment(report);
    auto &environment = report.environment;
    const auto active = device.backend_name();
    environment.backend = active.empty() ? report.options.backend : active;
    environment.warp_size = device.compute_warp_size();
    environment.shared_memory_bytes = device.compute_max_shared_memory_size();
    if (environment.backend == "cuda") {
        const auto total = device.query("total_memory");
        uint64_t bytes = 0;
        const auto parsed = std::from_chars(total.data(), total.data() + total.size(), bytes);
        if (parsed.ec == std::errc{} && parsed.ptr == total.data() + total.size()) {
            environment.total_device_bytes = bytes;
        }
    }
}

[[nodiscard]] Status run_benchmark(compute::Device &device, Report &report) {
    if (auto status = check_real_execution(report.options); !status.ok()) { return status; }
    capture_device_environment(device, report);
    constexpr BenchmarkSuite order[] = {
        BenchmarkSuite::ops, BenchmarkSuite::decode, BenchmarkSuite::cnn,
        BenchmarkSuite::layout, BenchmarkSuite::gdn, BenchmarkSuite::moe,
        BenchmarkSuite::model,
    };
    if (report.options.suite == BenchmarkSuite::all) {
        for (const auto suite : order) {
            luisa::vector<Row> rows;
            append_family(device, report.options, suite, rows);
            if (auto status = emit_rows(report, rows); !status.ok()) { return status; }
        }
    } else {
        luisa::vector<Row> rows;
        append_family(device, report.options, report.options.suite, rows);
        if (auto status = emit_rows(report, rows); !status.ok()) { return status; }
    }
    if (report.rows.empty()) {
        return Status::error(StatusCode::invalid_argument,
                             "selection matched no benchmark workload (check --target/--case/--variant/--workload)",
                             "selection");
    }
    return Status::success();
}

[[nodiscard]] Status run_benchmark_smoke(compute::Device &device, BenchmarkSuite suite, Report &report) {
    if (suite == BenchmarkSuite::all) { return Status::error(StatusCode::invalid_argument, "smoke requires one family", "suite"); }
    report.options = BenchmarkOptions{};
    report.options.backend = device.backend_name();
    report.options.suite = suite;
    report.options.warmup = 0;
    report.options.iterations = 1;
    report.options.workload = "smoke";
    report.options.pretty = false;
    switch (suite) {
        case BenchmarkSuite::ops: report.options.target = "gemm"; break;
        case BenchmarkSuite::decode:
            report.options.target = "dense_synthetic";
            report.options.case_name = "generate_greedy_4";
            break;
        case BenchmarkSuite::cnn: report.options.target = "pp_ocrv6:tiny"; break;
        case BenchmarkSuite::layout: report.options.target = "pp_doc_layout_v2"; break;
        case BenchmarkSuite::gdn: report.options.target = "ovis_ocr2"; break;
        case BenchmarkSuite::moe: report.options.target = "deepseek_ocr_2"; break;
        case BenchmarkSuite::model: report.options.target = "qwen3_vl_2b_instruct"; break;
        default: return Status::error(StatusCode::invalid_argument, "unknown smoke family", "suite");
    }
    if (auto status = run_benchmark(device, report); !status.ok()) { return status; }
    if (report.rows.size() != 1u) {
        return Status::error(StatusCode::internal_error,
                             luisa::format("smoke expected one row, got {}", report.rows.size()), "smoke");
    }
    const auto &row = report.rows.back();
    if (row.status != RowStatus::ok || !row.correct || !row.request_ms && !row.submit_sync_ms) {
        return row.reason.ok() ? Status::error(StatusCode::internal_error, "smoke row failed its correctness/timing gate", "smoke") : row.reason;
    }
    return Status::success();
}

}// namespace luisa::ocr::bench
