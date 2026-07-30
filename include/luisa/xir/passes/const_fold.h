#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

// This pass folds constant expressions at compile time.
// Arithmetic operations with all-constant operands are evaluated
// and replaced with a Constant value.
// Target-dependent floating-point operations, including transcendental
// functions, SQRT/RSQRT, POW, LERP, and SMOOTHSTEP, are left intact when a host
// evaluation would not preserve strict backend semantics. Ordinary operations
// with subnormal inputs/results or NaN inputs/results are likewise retained:
// denormal modes and manufactured NaN payloads are target properties. ISINF
// and ISNAN classification remain foldable. Signed-zero cases are folded only
// when backend behavior is unambiguous. Host evaluation is performed under
// round-to-nearest and restores the caller's floating-point environment before
// returning.

struct ConstFoldInfo {
    size_t folded_inst_count{0u};
};

// Folded constants are module-uniqued. Annotated instructions are retained so
// per-instruction metadata is never attached to a shared constant. Null inputs
// are no-ops.

[[nodiscard]] LUISA_XIR_API ConstFoldInfo const_fold_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API ConstFoldInfo const_fold_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
