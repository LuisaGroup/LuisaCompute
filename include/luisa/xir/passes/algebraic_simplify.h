#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

// Algebraic simplification pass.
// Applies simple algebraic identities to reduce instruction count:
//   x*0 → 0,  x*1 → x,  x+0 → x,
//   x/1 → x,  0/C → 0 for a proven nonzero integer C,
//   x-x → 0,   x&0 → 0,
//   x|0 → x,  x^0 → x,  x<<0 → x,  x>>0 → x

struct AlgebraicSimplifyInfo {
    size_t simplified_inst_count{0u};
};

struct AlgebraicSimplifyOptions {
    bool enable_fast_math{false};
};

// Annotated arithmetic instructions are conservatively retained because the
// replacement may be a pooled/shared value without a unique metadata owner.
// Null inputs are no-ops.

[[nodiscard]] LUISA_XIR_API AlgebraicSimplifyInfo algebraic_simplify_pass_run_on_function(Function *function, AlgebraicSimplifyOptions options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API AlgebraicSimplifyInfo algebraic_simplify_pass_run_on_module(Module *module, AlgebraicSimplifyOptions options = {}, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
