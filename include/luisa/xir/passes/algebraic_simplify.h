#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// Algebraic simplification pass.
// Applies simple algebraic identities to reduce instruction count:
//   x*0 → 0,  x*1 → x,  x+0 → x,  0-x → -x,
//   x/1 → x,  0/x → 0,  x-x → 0,   x&0 → 0,
//   x|0 → x,  x^0 → x,  x<<0 → x,  x>>0 → x

struct AlgebraicSimplifyInfo {
    size_t simplified_inst_count{0u};
};

struct AlgebraicSimplifyOptions {
    bool enable_fast_math{false};
};

[[nodiscard]] LUISA_XIR_API AlgebraicSimplifyInfo algebraic_simplify_pass_run_on_function(Function *function, AlgebraicSimplifyOptions options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API AlgebraicSimplifyInfo algebraic_simplify_pass_run_on_module(Module *module, AlgebraicSimplifyOptions options = {}) noexcept;

}// namespace luisa::compute::xir
