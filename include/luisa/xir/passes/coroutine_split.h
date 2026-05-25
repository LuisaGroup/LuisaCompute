#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/passes/coroutine.h>

// Coroutine splitting pass for XIR (flat-coroutine subset).
//
// Given a function annotated with $suspend markers (CoroSuspendInst), this
// pass runs the existing `coroutine_analysis` and then materializes one
// standalone CallableFunction per continuation. Each continuation callable
// has signature
//
//     void cont_k(Frame &frame, <original kernel args...>)
//
// where `Frame` is a struct containing
//
//     uint  target_token;          // 0 = terminated, otherwise next continuation id
//     <T0>  field_0;               // one slot per frame_candidate alloca
//     <T1>  field_1;
//     ...
//
// SUPPORTED INPUT (flat coroutines):
//   - All CoroSuspendInst instances live at function scope (no enclosing
//     structured loop, switch, or if).
//   - Continuations form a linear DAG with no back-edges.
//   - Frame candidates are AllocaInst values that are stored directly.
//
// UNSUPPORTED INPUT (returns is_supported=false with a diagnostic):
//   - Suspends inside structured loops. The paper "GPU Coroutines for
//     Flexible Splitting and Scheduling of Rendering Tasks" (SIGGRAPH Asia
//     2024) addresses this with a control-flow reconstruction stage that
//     duplicates relevant CFG nodes and replays condition values up to the
//     suspension point on resumption (Sec. "Control flow reconstruction").
//     Implementing that strategy in XIR is the next milestone; until then,
//     coroutines with suspends inside `$for`/`$while`/`$switch` should keep
//     using `coroutine_lower` (single-dispatch state machine).
//
// The pass appends the generated callables to the same module as the source
// coroutine and returns a CoroutineSplitInfo describing what was produced.

namespace luisa::compute::xir {

class CallableFunction;
class FunctionDefinition;
class Module;

struct CoroutineSplitFrameSlot {
    AllocaInst *source_alloca{}; // original alloca in the coroutine being split
    size_t field_index{};        // index into the generated frame struct (>=1, slot 0 is target_token)
    const luisa::compute::Type *type{};
};

struct CoroutineSplitContinuation {
    size_t id{};                 // 0 = entry, else matches the suspend id that produced this continuation
    CallableFunction *callable{};
    luisa::vector<size_t> outgoing_suspends; // suspend ids reachable from this continuation
};

struct CoroutineSplitInfo {
    bool is_supported{};                                  // false for unsupported inputs (e.g. loops)
    bool changed{};                                       // true if any callables were generated
    const luisa::compute::Type *frame_type{};             // generated struct type for the frame
    luisa::vector<CoroutineSplitFrameSlot> frame_slots;   // ordered to match frame_type members
    luisa::vector<CoroutineSplitContinuation> continuations; // one entry + one per non-terminating suspend
    luisa::vector<luisa::string> diagnostics;
};

// Run the split pass on a single function. The function must already contain
// CoroSuspendInst markers. Returns a CoroutineSplitInfo with `is_supported`
// indicating whether the input was within the supported flat-coroutine subset.
[[nodiscard]] LUISA_XIR_API CoroutineSplitInfo
coroutine_split_run_on_function(Function *function) noexcept;

// Run the split pass on every coroutine function in `module`. Functions that
// have no suspend markers are skipped. Returns a vector of per-function
// results in the order they appeared in the module.
[[nodiscard]] LUISA_XIR_API luisa::vector<CoroutineSplitInfo>
coroutine_split_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir
