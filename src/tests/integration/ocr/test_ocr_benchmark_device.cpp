// Test for the OCR benchmark harness itself (synthetic correctness smoke only).
// This test covers:
// - one correctness-gated, one-sample smoke run for every harness family;
// - no wall-time thresholds, performance promotion, or golden regeneration.
#include "ut/ut.hpp"
#include "test_device.h"

#include "bench/benchmark_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_family(Device &device, BenchmarkSuite suite) {
    Report report;
    auto status = bench::run_benchmark_smoke(device, suite, report);
    expect(status.ok()) << luisa::string{benchmark_suite_name(suite)} + ": " + status.message;
    expect(report.rows.size() == 1u);
    if (report.rows.size() == 1u) {
        const auto &row = report.rows.back();
        expect(row.status == RowStatus::ok) << row.reason.message;
        expect(row.correct);
        expect(row.samples.size() == 1u);
        expect(row.workload.artifact_class == ArtifactClass::synthetic);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "benchmark_ops_smoke"_test = [&] { test_family(device, BenchmarkSuite::ops); };
    "benchmark_decode_smoke"_test = [&] { test_family(device, BenchmarkSuite::decode); };
    "benchmark_cnn_smoke"_test = [&] { test_family(device, BenchmarkSuite::cnn); };
    "benchmark_layout_smoke"_test = [&] { test_family(device, BenchmarkSuite::layout); };
    "benchmark_gdn_smoke"_test = [&] { test_family(device, BenchmarkSuite::gdn); };
    "benchmark_moe_smoke"_test = [&] { test_family(device, BenchmarkSuite::moe); };
    "benchmark_model_smoke"_test = [&] { test_family(device, BenchmarkSuite::model); };
}
