#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// This pass folds constant expressions at compile time.
// Arithmetic operations with all-constant operands are evaluated
// and replaced with a Constant value.

struct ConstFoldInfo {
    size_t folded_inst_count{0u};
};

[[nodiscard]] LUISA_XIR_API ConstFoldInfo const_fold_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API ConstFoldInfo const_fold_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir
