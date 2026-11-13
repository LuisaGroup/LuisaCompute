#pragma once
// Shared device-benchmark mechanics. No timing helper performs a correctness
// download inside a timed region and no host timer is labeled as GPU time.
#include <cmath>
#include <string_view>
#include <type_traits>

#include <luisa/core/clock.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/functional.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>

#include "ocr/benchmark.h"

namespace luisa::ocr::bench {

using RecordOperator = luisa::function<Status(compute::CommandList &, uint32_t)>;
using CorrectnessGate = luisa::function<Status()>;
using Request = luisa::function<Status()>;

[[nodiscard]] inline bool matches_filter(luisa::string_view filter, luisa::string_view value) noexcept {
    return filter.empty() || filter == "all" || filter == value;
}

[[nodiscard]] inline luisa::string workload_option(const WorkloadDescriptor &workload, luisa::string_view key) {
    for (const auto &item : workload.options) {
        if (item.key == key) { return item.value; }
    }
    return {};
}

[[nodiscard]] inline bool workload_selected(const BenchmarkOptions &options, const WorkloadDescriptor &workload) {
    if (!matches_filter(options.target, workload.target) ||
        !matches_filter(options.case_name, workload.case_name) ||
        !matches_filter(options.variant, workload.variant)) { return false; }
    if (options.workload.empty() || options.workload == "default" || options.workload == "smoke") {
        return workload_option(workload, "workload") == "smoke";
    }
    if (options.workload == "all") { return true; }
    if (options.workload == "tier1") {
        const auto tier = workload_option(workload, "tier");
        return tier == "1" || tier.empty();
    }
    return options.workload == workload_option(workload, "workload") || options.workload == workload.id;
}

[[nodiscard]] inline WorkloadDescriptor make_workload(const BenchmarkOptions &options, BenchmarkSuite suite,
                                                      luisa::string target, luisa::string case_name,
                                                      luisa::string variant, luisa::string series,
                                                      uint64_t seed = 0x5EEDu) {
    WorkloadDescriptor workload;
    workload.suite = suite;
    workload.target = std::move(target);
    workload.case_name = std::move(case_name);
    workload.variant = std::move(variant);
    workload.artifact_class = ArtifactClass::synthetic;
    workload.seed = seed;
    workload.id = luisa::string{benchmark_suite_name(suite)} + "." + workload.target + "." + workload.case_name;
    workload.options = {
        {"workload", std::move(series)},
        {"series", "warm"},
        {"graph", options.graph ? "on" : "off"},
        {"fusion", options.fusion ? "on" : "off"},
        {"cooperative_norm", options.cooperative_norm ? "on" : "off"},
        {"cache", luisa::string{benchmark_cache_mode_name(options.cache)}},
        {"instrument", options.instrument ? "on" : "off"},
        {"dispatch_batch", luisa::format("{}", options.dispatch_batch)},
        {"warmup", luisa::format("{}", options.warmup)},
        {"iterations", luisa::format("{}", options.iterations)},
    };
    return workload;
}

inline void append_norm_variant(const BenchmarkOptions &options, luisa::string &variant) {
    if (!options.cooperative_norm) {
        variant += variant == "reference" ? "_norm" : "+reference_norm";
    }
}

inline void add_counter(Row &row, luisa::string key, luisa::string value) {
    row.counters.push_back({std::move(key), std::move(value)});
}
inline void add_counter(Row &row, luisa::string key, uint64_t value) {
    add_counter(row, std::move(key), luisa::format("{}", value));
}
template<typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, uint64_t>)
inline void add_counter(Row &row, luisa::string key, T value) {
    add_counter(row, std::move(key), static_cast<uint64_t>(value));
}
inline void add_counter(Row &row, luisa::string key, double value) {
    add_counter(row, std::move(key), luisa::format("{}", value));
}

inline void fail_row(Row &row, Status status, RowStatus outcome = RowStatus::failed) {
    row.status = outcome;
    row.reason = std::move(status);
    row.correct = false;
    row.record_ms.reset();
    row.submit_sync_ms.reset();
    row.request_ms.reset();
    row.stage = {};
}

[[nodiscard]] inline Status check_close(luisa::span<const float> got, luisa::span<const float> ref,
                                        float atol, float rtol, luisa::string_view what) {
    if (got.size() != ref.size()) {
        return Status::error(StatusCode::internal_error,
                             luisa::format("{}: size mismatch (got {}, expected {})", what, got.size(), ref.size()),
                             "correctness");
    }
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); i++) {
        const float g = got[i];
        const float r = ref[i];
        if (std::isnan(r)) {
            if (!std::isnan(g)) {
                return Status::error(StatusCode::internal_error,
                                     luisa::format("{}: expected NaN at element {}", what, i), "correctness");
            }
            continue;
        }
        const float diff = std::abs(g - r);
        worst = std::max(worst, diff);
        if (!std::isfinite(diff) || diff > atol + rtol * std::abs(r)) {
            return Status::error(StatusCode::internal_error,
                                 luisa::format("{}: mismatch at element {} (got {}, expected {}, worst abs diff {})",
                                               what, i, g, r, worst),
                                 "correctness");
        }
    }
    return Status::success();
}

// Correctness gate -> warmups -> timed batches -> correctness gate. The gate
// itself performs any required dispatch/download and is NEVER in a timed sample.
inline void time_operator(compute::Stream &stream, const BenchmarkOptions &options, Row &row,
                          const RecordOperator &record, const CorrectnessGate &gate) {
    if (auto status = gate(); !status.ok()) {
        fail_row(row, std::move(status));
        return;
    }
    for (uint32_t i = 0; i < options.warmup; i++) {
        auto list = compute::CommandList::create(static_cast<size_t>(options.dispatch_batch));
        if (auto status = record(list, options.dispatch_batch); !status.ok()) {
            fail_row(row, std::move(status));
            return;
        }
        stream << list.commit() << compute::synchronize();
    }
    Clock clock;
    for (uint32_t i = 0; i < options.iterations; i++) {
        auto list = compute::CommandList::create(static_cast<size_t>(options.dispatch_batch));
        Sample sample;
        clock.tic();
        if (auto status = record(list, options.dispatch_batch); !status.ok()) {
            fail_row(row, std::move(status));
            return;
        }
        sample.record_ms = clock.toc();
        clock.tic();
        stream << list.commit() << compute::synchronize();
        sample.submit_sync_ms = clock.toc();
        if (!std::isfinite(*sample.record_ms) || !std::isfinite(*sample.submit_sync_ms)) {
            fail_row(row, Status::error(StatusCode::internal_error, "non-finite operator timing", "timer"), RowStatus::invalid);
            return;
        }
        row.samples.push_back(sample);
    }
    if (auto status = gate(); !status.ok()) {
        fail_row(row, std::move(status));
        return;
    }
    row.correct = true;
    if (row.fallback_used) {
        fail_row(row, Status::error(StatusCode::unsupported, "unexpected fallback invalidates this timing row", "fallback"));
        return;
    }
    if (auto status = finalize_row(row); !status.ok()) {
        fail_row(row, std::move(status), RowStatus::invalid);
        return;
    }
    if (row.record_ms && row.workload.dispatches != 0u) {
        add_counter(row, "record_us_per_dispatch",
                    row.record_ms->median * 1000.0 / static_cast<double>(row.workload.dispatches));
    }
    if (row.submit_sync_ms && row.workload.dispatches != 0u) {
        add_counter(row, "submit_sync_us_per_dispatch",
                    row.submit_sync_ms->median * 1000.0 / static_cast<double>(row.workload.dispatches));
        if (row.workload.flops != 0u) {
            add_counter(row, "gflops_per_second",
                        static_cast<double>(row.workload.flops) / (row.submit_sync_ms->median * 1.0e6));
        }
        if (row.workload.bytes != 0u) {
            add_counter(row, "effective_gb_per_second",
                        static_cast<double>(row.workload.bytes) / (row.submit_sync_ms->median * 1.0e6));
        }
    }
}

// Time full independent requests. The request callable owns all required state
// reset/allocation and stores its latest result for the correctness callable.
inline void time_requests(compute::Stream &stream, const BenchmarkOptions &options, Row &row,
                          const Request &request, const CorrectnessGate &gate,
                          StageDurations *latest_stage = nullptr) {
    const auto run_checked = [&]() -> Status {
        stream << compute::synchronize();
        if (latest_stage != nullptr) { *latest_stage = {}; }
        if (auto status = request(); !status.ok()) { return status; }
        stream << compute::synchronize();
        return gate();
    };
    if (auto status = run_checked(); !status.ok()) {
        fail_row(row, std::move(status));
        return;
    }
    for (uint32_t i = 0; i < options.warmup; i++) {
        stream << compute::synchronize();
        if (latest_stage != nullptr) { *latest_stage = {}; }
        if (auto status = request(); !status.ok()) {
            fail_row(row, std::move(status));
            return;
        }
        stream << compute::synchronize();
    }
    Clock clock;
    for (uint32_t i = 0; i < options.iterations; i++) {
        stream << compute::synchronize();
        if (latest_stage != nullptr) { *latest_stage = {}; }
        clock.tic();
        if (auto status = request(); !status.ok()) {
            fail_row(row, std::move(status));
            return;
        }
        stream << compute::synchronize();
        Sample sample;
        sample.request_ms = clock.toc();
        if (latest_stage != nullptr) { sample.stage = *latest_stage; }
        if (!std::isfinite(*sample.request_ms)) {
            fail_row(row, Status::error(StatusCode::internal_error, "non-finite request timing", "timer"), RowStatus::invalid);
            return;
        }
        row.samples.push_back(sample);
    }
    if (auto status = gate(); !status.ok()) {
        fail_row(row, std::move(status));
        return;
    }
    row.correct = true;
    if (row.fallback_used) {
        fail_row(row, Status::error(StatusCode::unsupported, "unexpected fallback invalidates this timing row", "fallback"));
        return;
    }
    if (auto status = finalize_row(row); !status.ok()) {
        fail_row(row, std::move(status), RowStatus::invalid);
        return;
    }
    if (row.request_ms && row.workload.output_tokens != 0u) {
        add_counter(row, "generated_tokens_per_second",
                    static_cast<double>(row.workload.output_tokens) * 1000.0 / row.request_ms->median);
    }
    if (row.request_ms && row.workload.prompt_tokens != 0u) {
        add_counter(row, "prompt_tokens_per_second",
                    static_cast<double>(row.workload.prompt_tokens) * 1000.0 / row.request_ms->median);
    }
    if (row.request_ms && row.workload.pages != 0u) {
        add_counter(row, "pages_per_second",
                    static_cast<double>(row.workload.pages) * 1000.0 / row.request_ms->median);
    }
    if (row.request_ms && row.workload.regions != 0u) {
        add_counter(row, "regions_per_second",
                    static_cast<double>(row.workload.regions) * 1000.0 / row.request_ms->median);
    }
}

}// namespace luisa::ocr::bench
