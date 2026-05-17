#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// SROA (Scalar Replacement of Aggregates) pass.
// Decomposes aggregate (struct/array) allocas into individual
// scalar allocas to enable mem2reg promotion.
//
// Only handles allocas where all GEP accesses have constant indices.

struct SROAInfo {
    size_t decomposed_alloca_count{0u};
    size_t inserted_alloca_count{0u};
};

[[nodiscard]] LUISA_XIR_API SROAInfo sroa_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API SROAInfo sroa_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir
