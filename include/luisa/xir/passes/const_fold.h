#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

// This pass folds constant expressions at compile time.
// Arithmetic operations with all-constant operands are evaluated
// and replaced with a Constant value.
// Target-dependent floating-point operations such as LERP and SMOOTHSTEP are
// left intact when a host evaluation would not preserve strict backend
// semantics. NaN and signed-zero cases are folded only when backend behavior is
// unambiguous.

struct ConstFoldInfo {
    size_t folded_inst_count{0u};
};

[[nodiscard]] LUISA_XIR_API ConstFoldInfo const_fold_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API ConstFoldInfo const_fold_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
