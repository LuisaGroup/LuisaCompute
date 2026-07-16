#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// This pass is used to trace back cascaded GEP instructions to the base pointer,
// in the hope of simplifying other passes that may need to analyze the GEP chain.
//
// For example, if we have a GEP instruction like:
// x = gep(base, i0, i1, i2)
// y = gep(x, j0, j1)
// z = gep(y, k0, k1)
//
// The pass will transform the above instructions to:
// x = gep(base, i0, i1, i2)
// y = gep(base, i0, i1, i2, j0, j1)
// z = gep(base, i0, i1, i2, j0, j1, k0, k1)

class GEPInst;

struct TraceGEPInfo {
    size_t traced_gep_count{0u};
};

[[nodiscard]] LUISA_XIR_API TraceGEPInfo trace_gep_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API TraceGEPInfo trace_gep_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir
