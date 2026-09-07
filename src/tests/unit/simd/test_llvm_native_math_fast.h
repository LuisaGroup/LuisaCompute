#pragma once

// Runs the shared precise/fast provider audit from the existing SIMD native
// math test executable, keeping only one expensive JIT module per run.
[[nodiscard]] bool test_llvm_native_math_fast();
