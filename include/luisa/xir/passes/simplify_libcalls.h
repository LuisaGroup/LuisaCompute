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
///   - CLAMP(x, +0.0, 1.0) → SATURATE(x)
///   - ABS(x) for unsigned x → x
///   - SELECT(cond, x, x) → x
///
/// Floating-point rewrites are required to preserve strict semantics. In
/// particular, LERP endpoint rewrites are deliberately not performed because
/// target implementations may use a fused multiply-add formulation.
/// Rewrites that create a replacement instruction clone all metadata. Identity
/// rewrites are conservatively skipped when the removed instruction has
/// metadata because a shared operand is not a unique metadata owner.
[[nodiscard]] LUISA_XIR_API SimplifyLibCallsInfo simplify_libcalls_pass_run_on_function(FunctionDefinition *def) noexcept;

/// Run the simplify-libcalls pass on every function definition in a module.
[[nodiscard]] LUISA_XIR_API SimplifyLibCallsInfo simplify_libcalls_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
