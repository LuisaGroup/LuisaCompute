#pragma once

#include <luisa/core/dll_export.h>
#include <cstddef>
#include <luisa/ast/coro_call.h>

namespace luisa::compute::xir {
class Function;

struct CoroCallInfo {
    CoroCallGraph graph;
    size_t callable_count{0u};
    size_t call_site_count{0u};
};

// Lower the nonrecursive suspending call graph into shared CFG regions in
// root. Every callee is cloned exactly once, irrespective of call-site count.
// Calls store a static activation and return selector; returns dispatch to
// the matching call continuation. Only real suspend instructions yield.
// Run before coroutine distillation, on AST-derived (PHI-free) XIR. Ordinary
// callees are untouched. Reference/resource arguments use pointer-free alias
// selectors; dynamic access indices are captured at each call.
[[nodiscard]] LUISA_XIR_API CoroCallInfo coro_call_pass_run_on_function(Function *root);
// After source optimization, preserve dynamic definitions when a resume scope
// can re-enter a shared region. Unlike ordinary reg2mem, dominance is not a
// reason to keep an SSA use crossing this continuation ownership boundary.
LUISA_XIR_API void coro_call_demote_cross_block_values(Function *root);
// Structure a PHI-free, materialized continuation without duplicating regions.
// The local block selector is not a suspend token and never enters the frame.
LUISA_XIR_API void coro_call_structure_continuation(Function *function);
}// namespace luisa::compute::xir
