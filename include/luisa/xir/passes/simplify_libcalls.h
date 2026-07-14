#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

/// Statistics for the simplify-libcalls pass.
struct SimplifyLibCallsInfo {
    size_t simplified_count{0u};
};

/// Run the simplify-libcalls pass on a single function definition.
///
/// Recognizes math library patterns in ArithmeticInst and replaces them
/// with simplified or canonical forms:
///   - LERP(x, y, 0.0) → x
///   - LERP(x, y, 1.0) → y
///   - CLAMP(x, 0.0, 1.0) → SATURATE(x)
///   - STEP(0.0, x) → 1.0
///   - ABS(x) for unsigned x → x
///   - SELECT(cond, x, x) → x
[[nodiscard]] LUISA_XIR_API SimplifyLibCallsInfo simplify_libcalls_pass_run_on_function(FunctionDefinition *def) noexcept;

/// Run the simplify-libcalls pass on every function definition in a module.
[[nodiscard]] LUISA_XIR_API SimplifyLibCallsInfo simplify_libcalls_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
