#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/passes/coro_graph_analysis.h>
#include <luisa/xir/passes/coroutine_split.h>

// Materializer: consumes CoroGraphInfo and emits one CallableFunction per
// scope. Each callable has signature (frame_ref, ...original_args) → void.
//
// Handles:
//   - Simple nodes: clone with InstructionCloneValueResolver
//   - ConditionStackReplay: emit constant assignments for replayed conditions
//   - MakeFirstFlag: emit alloca_local(bool) = true
//   - SkipIfFirstFlag: emit if(!flag) { ...body... }
//   - ClearFirstFlag: emit store(flag, false)
//   - Loop: clone loop structure recursively
//   - If/Switch: clone structure recursively
//   - Suspend: emit store(frame[0], next_token); return_void
//   - Terminate: emit store(frame[0], 0); return_void

namespace luisa::compute::xir {

class CallableFunction;
class Module;

}// namespace luisa::compute::xir

namespace luisa::compute::xir::coro {

struct CoroMaterializeResult {
    bool ok{};
    CoroutineSplitInfo split_info;// populated with frame type, slots, continuations
    luisa::vector<luisa::string> diagnostics;
};

// Run the full pipeline: analysis → scope split → materialize.
// Produces one CallableFunction per scope in the same module.
[[nodiscard]] LUISA_XIR_API CoroMaterializeResult
coro_materialize_run_on_function(Function *function) noexcept;

}// namespace luisa::compute::xir::coro
