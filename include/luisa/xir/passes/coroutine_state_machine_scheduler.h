#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/passes/coroutine_split.h>

// State-machine scheduler emitter (XIR-only, no DSL bridging).
//
// Given the result of `coroutine_split_run_on_function` (a frame type plus
// N+1 continuation callables), this emitter materializes a KernelFunction
// inside the same module that drives the state machine described in
//
//   "GPU Coroutines for Flexible Splitting and Scheduling of Rendering
//    Tasks" (SIGGRAPH Asia 2024), Sec. 4.1 (state-machine scheduler).
//
// The generated kernel has the same argument list as the source coroutine
// and behaves as if the coroutine were executed start-to-finish on each
// thread:
//
//   kernel(args...) {
//       Frame frame;                    // local alloca of split.frame_type
//       frame.target_token = 0;
//       cont_entry(frame, args...);     // continuation 0
//       loop {
//           switch (frame.target_token) {
//               case 0: break;          // 0 = terminated
//               case 1: cont_1(frame, args...); break;
//               ...
//               default: unreachable;
//           }
//           if (frame.target_token == 0) break;
//       }
//   }
//
// Returns a CoroutineStateMachineSchedulerInfo whose `kernel` field points
// at the freshly created KernelFunction. The caller owns the kernel through
// the module.

namespace luisa::compute::xir {

class KernelFunction;
class Module;

struct CoroutineStateMachineSchedulerConfig {
    luisa::uint3 block_size = luisa::make_uint3(64u, 1u, 1u);
};

struct CoroutineStateMachineSchedulerInfo {
    bool ok{};                 // false on bad input (split not supported, etc.)
    KernelFunction *kernel{};  // generated scheduler kernel, or nullptr
    luisa::vector<luisa::string> diagnostics;
};

// Emit a state-machine scheduler kernel into `module` from a populated
// CoroutineSplitInfo. Requires `split.is_supported && split.changed` —
// otherwise returns ok=false with a diagnostic.
[[nodiscard]] LUISA_XIR_API CoroutineStateMachineSchedulerInfo
coroutine_state_machine_scheduler_emit(
    Module *module,
    const CoroutineSplitInfo &split,
    const CoroutineStateMachineSchedulerConfig &config = {}) noexcept;

}// namespace luisa::compute::xir
