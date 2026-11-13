#pragma once
// OCR synthetic benchmark device entry points. Kept out of lc-example-ocr:
// implementation files privately reuse test-only fixture builders.
#include "ocr/benchmark.h"

#include <luisa/runtime/device.h>

namespace luisa::ocr::bench {

// Fill process fields (also used for pre-device real-artifact refusals), then
// backend/device fields without inventing a universal GPU-name/driver query.
// User-supplied provenance in report.options takes precedence.
void capture_process_environment(Report &report);
void capture_device_environment(compute::Device &device, Report &report);

// Run all selected cases. Case failures are preserved as rows and do not erase
// earlier evidence. The returned Status is reserved for runner-level failures
// (real-artifact refusal, invalid selection, or an empty selected catalog).
[[nodiscard]] Status run_benchmark(compute::Device &device, Report &report);

// One correctness-gated, one-sample smoke case per requested family. Used by
// the integration test; it is not a performance assertion or CTest timing gate.
[[nodiscard]] Status run_benchmark_smoke(compute::Device &device, BenchmarkSuite suite, Report &report);

}// namespace luisa::ocr::bench
