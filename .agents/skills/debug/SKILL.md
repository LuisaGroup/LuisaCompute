---
name: debug
description: Debug crashes and test failures via stack-traces, host/device logging, and DSL buffer inspection.
---

# Debugging LuisaCompute

## 1. Interpreting Stack-Traces

When a crash or `LUISA_ERROR` is emitted, capture the full console output first.

**What to look for:**
- **Top frames** — the actual fault (null dereference, assertion, backend error).
- **LuisaCompute frames** — functions prefixed with `luisa::`, especially `luisa::compute::` and the DSL inline namespace `luisa::compute::dsl::`.
- **Backend frames** — plugins load as `luisa-backend-<name>`; the installed backends are `cuda`, `dx`, `fallback`, `hip`, `metal`, `metal4`, `simd`, `vk`. There is **no `cpu` backend** — the CPU paths are `fallback` (native C++/JIT) and `simd`.
- **Last log line** — often the preceding `LUISA_INFO`/`LUISA_VERBOSE` shows the dispatch or shader name that triggered the bug.

**What the trace itself looks like** (`src/core/platform.cpp`, `include/luisa/core/stl/format.h`):
- `LUISA_ERROR` → `luisa::log_error` appends the frames to the error message itself (` NN [0xADDR]: module :: symbol + offset`), then flushes and calls `std::abort()` — so expect a second dump from the abort handler.
- Windows installs `SetUnhandledExceptionFilter`, `std::set_terminate` and a `SIGABRT` handler (static `StackTracerInit`), which print `!!! Unhandled structured exception !!!`, `!!! std::terminate called (uncaught exception) !!!` or `!!! SIGABRT / std::abort() called !!!`, followed by `----- Stack Trace (N frames) -----` rows of `  [ii] symbol  at module+0xoff`.
- Frames come from `luisa::backtrace()` — dbghelp (`CaptureStackBackTrace` + `SymFromAddr`) on Windows, `::backtrace`/`backtrace_symbols` on Unix. On Windows this whole path is `#ifndef NDEBUG`: a Release build yields **no frames at all**, so reproduce under `-m debug` or `-m releasedbg` first.

**Action:**
1. Read the innermost frame (first after the crash header). This is the immediate cause.
2. Walk upward until you hit a recognizable LuisaCompute API call (e.g., `Device::compile`, `Buffer::copy_to`/`copy_from`, `Stream::synchronize`, or the `shader(args...).dispatch(...)` command built by `ShaderInvoke<N>::dispatch`). That is the *call-site*.
3. If the trace ends inside a driver/shared library, suspect (a) invalid resource usage (out-of-bounds buffer/image access), or (b) backend-specific limitation.

## 2. Plan Before Fixing

Once the stack-trace points to a file/line or API call, write a **debug plan** in this order:

1. **Hypothesis** — state what you believe caused the failure in one sentence.
2. **Verification** — describe the smallest code change or log addition that can confirm/disprove the hypothesis.
3. **Fix strategy** — if verified, what exactly will you change.
4. **Rollback marker** — note the original state so you can undo cleanly.

**If the fix fails:**
- Save the failed attempt with memory.
- Re-read the stack-trace and the saved steps. Do not repeat a failed hypothesis.
- Pick the next most likely cause and repeat from step 1.

## 3. When There Is No Stack-Trace

Silent failures (hang, wrong result, test timeout) provide no trace.

Distinguish hang types by CPU usage first:
- **High-CPU "hang"** (process burns a core indefinitely) is usually a spin on
  corrupted state, e.g. a `spin_mutex` inside a freed object — a UAF that
  looks like a hang. Treat it as a memory bug: audit recent lifetime changes
  and destructors that unlink cross-object references.
- **Zero-CPU hang** is a genuine wait (GPU fence, deadlocked mutex).

For multi-test binaries, isolate by running each test alone (Boost.UT takes
positional name patterns after the backend arg: `test.exe vk my_test`, i.e.
`xmake run <test_target> vk my_test`). When a
fix candidate emerges, A/B it with `git stash` — a failure that persists with
the fix stashed is pre-existing and out of scope; do not chase it.

Cheapest ways to make a silent failure talk:
- Run with `LUISA_ENABLE_VALIDATION=1`: `Context::create_device` wraps the backend device in the `luisa-validation-layer` plugin, so API misuse turns into a logged `LUISA_ERROR` (with trace) instead of driver-side corruption.
- Build with safe mode — `lc_safe_mode` (xmake) or `LUISA_COMPUTE_ENABLE_SAFE_MODE=ON` (CMake), which defines `LUISA_ENABLE_SAFE_MODE`. Resource constructors then `LUISA_ERROR` on an invalid handle at the creation call (e.g. `include/luisa/runtime/buffer.h`, `src/runtime/stream.cpp`) instead of failing later.
- Remember that a Release (`NDEBUG`) Windows binary cannot print frames at all (Section 1) — 'no stack trace' may just be the build mode.

**Find the entry point:**
- Read `CMakeLists.txt` or `xmake.lua` near the failing target to locate the executable source file and its `main()`.
- Identify the test harness (e.g., `test_device.h`, `boost::ut`) and how the device is created.

**Add host-side logging:**
```cpp
#include <luisa/core/logging.h>

// In host code (C++ runtime)
LUISA_VERBOSE("Entering {}::{}", __FILE__, __func__);
LUISA_INFO("Buffer size = {}", buf.size());
LUISA_VERBOSE_WITH_LOCATION("Dispatching kernel X");
```

**Set log level early** (before Context creation if possible):
```cpp
luisa::log_level_verbose();  // or log_level_info()
```

**Progressive narrowing:**
1. Log at the start of `main()` and at every major phase (context → device → stream → compile → dispatch).
2. If the failure happens during a kernel dispatch, move to device-side logging (Section 4).
3. If the failure is a wrong numerical result, move to buffer read-back (Section 5).

## 4. DSL / Device-Side Logging

Inside kernels, use `device_log` (`include/luisa/dsl/stmt.h`, lowered to `FunctionBuilder::print_`) to emit per-thread messages. They are collected by the stream and flushed to the host callback (`Stream::set_log_callback`); with no callback installed the backends print them through the host logger as `LUISA_INFO("[DEVICE] {}", ...)`.

**Basic usage:**
```cpp
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

Kernel2D k = [&]() noexcept {
    UInt2 coord = dispatch_id().xy();
    $if (coord.x == 1) {
        device_log("hello {} {}", coord, make_float3x3());
    };
};
```

**Custom log callback on the stream:**
```cpp
Stream stream = device.create_stream();
stream.set_log_callback([](luisa::string_view message) {
    LUISA_INFO("device: {}", message);
});
stream << shader().dispatch(128u, 128u) << synchronize();
```

**Structured severity prefixes** (for custom routing):
```cpp
// Severity prefixes + dispatch-id pattern: src/tests/unit/runtime/test_printer_custom_callback.cpp
// (it also emits a 'V' prefix and appends `$dispatch_id` to every message)
#define DEVICE_INFO(FMT, ...) \
    device_log(luisa::format("I" FMT) __VA_OPT__(, ) __VA_ARGS__)
#define DEVICE_WARNING(FMT, ...) \
    device_log(luisa::format("W" FMT) __VA_OPT__(, ) __VA_ARGS__)
#define DEVICE_ERROR(FMT, ...) \
    device_log(luisa::format("E" FMT) __VA_OPT__(, ) __VA_ARGS__)

stream.set_log_callback([](luisa::string_view msg) {
    if (!msg.empty()) {
        switch (msg.front()) {
            case 'I': luisa::log_info("{}", msg.substr(1)); break;
            case 'W': luisa::log_warning("{}", msg.substr(1)); break;
            case 'E': luisa::log_warning("device error: {}", msg.substr(1)); break;
            default:  luisa::log_verbose("{}", msg); break;
        }
    }
});
```

Never route a device message to `luisa::log_error`/`LUISA_ERROR` from the callback: it is `[[noreturn]]`, prints a host backtrace and calls `std::abort()` (`include/luisa/core/logging.h`), so one device-side 'E' line kills the host process mid-flush.

No build switch or environment variable is needed for device logging: `device_log` is lowered to `FunctionBuilder::print_`, each backend ships the format strings with the compiled shader (`printers`), and the host decodes the log buffer in `format_shader_print` (`src/backends/common/shader_print_formatter.h`). There is no `enable_logging`, `logging_buffer`, or `buffer_printf` API in this codebase.

**Important:** Device logs are asynchronous. Always `synchronize()` the stream before assuming all logs have arrived. If a kernel hangs, the callback may never fire for logs buffered inside the failing dispatch.

## 5. Using Buffer for DSL Debug

When you need to inspect many values or avoid per-thread log flooding, write results into a `Buffer` and read back on the host.

**Buffer-based inspection:**
```cpp
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

Buffer<float4> debug_buf = device.create_buffer<float4>(1024);

Kernel1D k = [](BufferVar<float4> out) noexcept {
    UInt idx = dispatch_id().x;
    Float4 v = make_float4(cast<float>(idx),
                           cast<float>(idx) * 2.0f,
                           cast<float>(idx) * 3.0f,
                           0.0f);
    out.write(idx, v);
};

auto shader = device.compile(k);
stream << shader(debug_buf).dispatch(1024)
       << synchronize();

// Read back
luisa::vector<float4> host(1024);
stream << debug_buf.copy_to(luisa::span{host}) << synchronize();
for (size_t i = 0; i < 8; ++i) {
    LUISA_INFO("host[{}] = {}", i, host[i]);
}
```

**Counter + slot pattern for conditional values:**
- Allocate a `Buffer<uint>` (e.g. `counter_buf`) plus a payload `Buffer<T>` of the size you want to capture.
- In the kernel, take a slot atomically and write through it: `auto slot = counter_buf.atomic(0u).fetch_add(1u); debug_buf.write(slot, payload);` — `BufferVar` exposes `read(i)` / `write(i, v)` / `atomic(i)`, not `operator[]` (`include/luisa/dsl/resource.h`, `include/luisa/dsl/atomic.h`; live example: `src/tests/unit/coro/test_coro_wavefront.cpp:1071`).
- Guard the slot (`if (slot < N)`) so a flooded kernel cannot write past the buffer.
- This captures the first N interesting threads without over-allocating.

## 6. Environment Variables for Backend Diagnosis

177 distinct `LUISA_*` names are read from the environment in `src/` outside `src/tests` (114 of them `LUISA_SIMD_*` codegen toggles), plus the system `MTL_*`/`METAL_*` variables; enumerate them with `grep -rn '"LUISA_' src/`. The table lists the diagnosis-relevant subset. Booleans are read through `luisa::compute::detail::env_flag` (`src/backends/common/env_flag.h:11-17`: `1`, `true`, `TRUE`, `on`, `ON`), while a few checks compare against exactly `"1"` (noted below). Set flags to `1` and you are always safe.

| Variable | Effect | Evidence |
|---|---|---|
| `LUISA_DUMP_SOURCE=1` | Dump generated shader sources/bytecode (must be exactly `1` on the DX/Vulkan/CUDA paths). | `src/backends/common/backend_print_code.h:8-10` |
| `LUISA_LOG_LEVEL` = `verbose` / `info` / `warning` / `error` | Startup log level (case-insensitive); same effect as `luisa::log_level_verbose()`. Invalid values warn and keep the default (`debug` in non-`NDEBUG`, else `info`). | `src/core/logging.cpp:52-78` |
| `LUISA_ENABLE_VALIDATION=1` | `Context::create_device` wraps the backend device in the `luisa-validation-layer` plugin (must be exactly `1`). | `src/runtime/context.cpp:155,372-391` |
| `LUISA_OPTIX_VALIDATION=1` | Full OptiX device-context validation on CUDA (large slowdown; warns when on). | `src/backends/cuda/cuda_device.cpp:324-329,1576-1579` |
| `LUISA_VULKAN_VALIDATION=1` | Force the Vulkan instance validation layers on (they are already on by default in `NDEBUG`-off builds). | `src/backends/vk/device.cpp:220-232` |
| `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` | Fail instead of silently falling back to HLSL→SPIR-V for a user compute kernel. | `src/backends/vk/device.cpp:71-74,3346` |
| `LUISA_DISABLE_COMMAND_REORDER=1` | Process-wide kill switch for DX/Vulkan command reordering → strict submission order (A/B a reorder bug). | `src/backends/common/command_reorder_switch.h:17-36` |
| `LUISA_DUMP_XIR=1` (+ `LUISA_DUMP_XIR_DIR`) | Dump XIR (`kernel.<hash>.xir`, `.opt.xir`, …); CUDA/XIR, Metal4, fallback. | `src/backends/fallback/fallback_shader.cpp:66-71`, `src/backends/cuda/cuda_device.cpp:79-100` |
| `LUISA_DUMP_LLVM_IR=1` / `LUISA_DUMP_ASM=1` | Dump LLVM IR (`kernel.<hash>.ll`, `.opt.ll`) / assembly (`.s`) — fallback JIT and the LLVM codegen paths. | `src/backends/fallback/fallback_shader.cpp:73-85,766,845,856` |
| `LUISA_DUMP_SPV=1` | Dump the pre-validation SPIR-V binary (hard-coded path `/tmp/opencode/kernel_<hash>.spv`). | `src/backends/common/spirv/spirv_codegen/entry.cpp:440-444` |
| `LUISA_XIR_TRACE_PASSES=1` | Per-pass tracing while planning the XIR→SPIR-V pipeline. | `src/backends/common/spirv/spirv_codegen/utils.cpp:218-219` |
| `LUISA_VULKAN_PROFILE_COMPILATION=1` | Per-phase compile timing on the native SPIR-V path. | `src/backends/common/spirv/spirv_codegen/entry.cpp:21-24`, `optimizer.cpp:586` |
| `LUISA_XIR_NORMALIZE_CFG=1` / `LUISA_XIR_RESTRUCTURE_CFG=1` / `LUISA_XIR_ELIMINATE_EARLY_RETURN=1` | Per-backend XIR CFG pass toggles (CUDA, HIP, fallback) — bisect a lowering bug. | `src/backends/cuda/cuda_device.cpp:54-56`, `src/backends/fallback/fallback_shader.cpp:87-121` |
| `LUISA_SINGLE_THREADING=1` | Fallback backend: run every thread of a dispatch serially on one host thread (deterministic device-side debugging). | `src/backends/fallback/fallback_command_queue.cpp:247-257` |
| `LUISA_CUDA_FALLBACK_RTX=1` / `LUISA_DX_FALLBACK_RTX=1` / `LUISA_VK_FALLBACK_RTX=1` | Force the software ray-tracing fallback instead of OptiX/DXR/native Vulkan RT. | `src/backends/cuda/cuda_device.cpp:398`, `src/backends/dx/DXApi/LCDevice.cpp:65`, `src/backends/vk/device.cpp:1894` |
| `LUISA_SIMD_WARP_WIDTH=…`, `LUISA_SIMD_WORKER_COUNT=…`, `LUISA_SIMD_REPORT_OPTIMIZATIONS=1`, `LUISA_SIMD_DUMP_ASSEMBLY_DIR=…` | SIMD (CPU) backend overrides/diagnostics; `LUISA_SIMD_*` also covers dozens of `DISABLE_*`/`FORCE_*` codegen toggles. | `src/backends/simd/runtime/simd_device.cpp:124,152`, `src/backends/simd/simd_compiler.cpp:898` |
| `LUISA_METAL_SHADER_INFO=1`, `LUISA_METAL_COMMAND_BUFFER_PROFILE=1`, `LUISA_DUMP_METAL_LIBRARY=1` | Metal/Metal4 compile-info and command-buffer staging profile; metallib dump. | `src/backends/metal/metal_shader.cpp:81,102`, `src/backends/metal/metal_compiler.cpp:239` |
| `MTL_DEBUG_LAYER=1`, `MTL_SHADER_VALIDATION=1`, `MTL_ENABLE_CAPTURE=1`, `METAL_CAPTURE_ENABLED=1` | System Metal env vars the backends forward/force when deciding to keep debug info. | `src/backends/metal4/metal_compiler.cpp:45-49,452-456` |
| `LUISA_HIP_PROFILE=1`, `LUISA_HIP_WAVE64=1`, `LUISA_DUMP_HIP_ISA=<dir>`, `LUISA_DUMP_PTX=1`, `LUISA_DUMP_AMDGPU=1` | HIP/CUDA codegen-path diagnostics. | `src/backends/hip/hip_stream.cpp:114`, `src/backends/hip/hip_device.cpp:623`, `src/backends/cuda/llvm_codegen/cuda_codegen_llvm.cpp:19` |
| remaining families | `LUISA_SPIRV_OPT_LEVEL`/`_PASSES`/`_PASS_FLAGS`/`_MAX_ITERATIONS`/`_SROA_LIMIT`/`_MAX_ID_BOUND`/`_PRESERVE_BINDINGS` (bisect spv-opt), `LUISA_XIR_DISABLE_OPTIMIZATION`, `LUISA_XIR_ENABLE_SCALARIZER`, `LUISA_VK_DUMP_FALLBACK_HLSL`, `LUISA_EXPERIMENTAL_XIR_CODEGEN`/`LUISA_EXPERIMENTAL_LLVM_CODEGEN` (CUDA codegen route), `LUISA_CORO_*` coroutine-pass dumps, `LUISA_GUI_RASTER*`, `LUISA_REMOTE_TOKEN`. | `src/backends/common/spirv/spirv_codegen/optimizer.cpp:72-586`, `.../utils.cpp:491-505`, `src/backends/vk/device.cpp:3157`, `src/backends/cuda/cuda_device.cpp:65-86`, `src/xir/passes/coro_alloca_scope.cpp:546` |

(`luisa::log_level_verbose()` etc. are the programmatic equivalents, `include/luisa/core/logging.h:136-142`. Build-time switches are *not* env vars: CMake `LUISA_COMPUTE_*` options and xmake `lc_*` options.)

Use `LUISA_DUMP_SOURCE=1` when you suspect a code-generation bug (wrong instruction, missing binding, incorrect type).

**Where to find the dumps:**
- **DirectX:** `hlsl_output_<name>.hlsl` in the current working directory.
- **Vulkan user compute (XIR→SPIR-V path):** `spv_code_<name>.spvasm` in the current working directory.
- **Vulkan user compute (LLVM→SPIR-V path):** `spv_code_llvm_<name>.spvasm`.
- **Vulkan internal HLSL consumers:** backend builtins/raster may dump `hlsl_output_<name>.hlsl`; ordinary `Device::compile(Function)` compute shaders must not.
- **CUDA:** `.cu` source in the runtime `.cache` directory (`write_shader_source`); PTX + `.metadata` also land in `.cache` (`write_shader_cache`) — only explicitly named/AOT bytecode goes to the data directory itself (`write_shader_bytecode`). The `.data` directory holds *internal* shaders, not user PTX.
- **Metal:** `.metal` source in the runtime `.cache` directory; Metal4 dumps `.metallib` there.
- **Fallback:** does **not** read `LUISA_DUMP_SOURCE`; set `LUISA_DUMP_XIR=1` / `LUISA_DUMP_LLVM_IR=1` / `LUISA_DUMP_ASM=1` to get `kernel.<hash>.xir`, `.opt.xir`, `.ll`, `.opt.ll`, `.s` in the current working directory.

The runtime directories are printed by `LUISA_INFO` at context creation; they default to the executable directory. When running under `xmake run`, dumps written directly to the current working directory will appear in the project root.

## 7. Decision Checklist

| Symptom | First Action | Next Action |
|---|---|---|
| Crash with stack-trace | Read innermost + first Luisa frame | Hypothesize → plan → fix |
| Crash with **no** frames | Rebuild `-m debug`/`-m releasedbg` (Windows backtrace is `#ifndef NDEBUG`) | Attach `scripts/debugger.py` (Section 8) |
| Silent wrong result | Add `LUISA_INFO` at host entry points | Use buffer read-back to inspect values |
| Kernel dispatch hangs | Check synchronize() and stream callback | Add minimal device_log at start of kernel |
| Hang burning CPU | Suspect spin on freed memory (UAF masquerading as hang) | Isolate per-test; audit destructor unlink paths |
| Unsure if failure is yours | Stash changes and re-run the same binary path | Persistent failure = pre-existing, out of scope |
| Backend compilation error | Set `LUISA_DUMP_SOURCE=1` (fallback: `LUISA_DUMP_XIR=1`) | Inspect generated `.spvasm` or `.hlsl` |
| Suspected API/resource misuse | Set `LUISA_ENABLE_VALIDATION=1` | Re-run and read validation messages |
| Test timeout | Read build file for target entry | Narrow phase with host logging |

## 8. Windows Crash Debugging with `scripts/debugger.py`

A lightweight Python debugger using Windows Debug API + DbgHelp.dll (pure `ctypes`, no third-party packages) to launch an x64 executable under the debugger, print a symbolic stack trace for every exception event, and stop at the second-chance one. Output: `[!] Caught exception:` header (code/address/`FirstChance`), then `=== C++ stack trace (with PDB symbols) ===` rows `#NN 0xADDR func` followed by ` file:line` when line info exists.

**Usage:**
```bash
python scripts/debugger.py <path_to_exe> [pdb_search_path] [-- <args>...]
```

- Arguments after `--` are forwarded to the target executable.
- Symbols come from `SRV*C:\Symbols*https://msdl.microsoft.com/download/symbols;<pdb_search_path>`, so the PDB must sit next to the EXE or in `pdb_search_path` (system DLLs are fetched from the Microsoft symbol server).
- Prints a trace for **first-chance** exceptions too (breakpoints, MSVC C++ EH `0xE06D7363`) and keeps going; only a second-chance exception ends the session.
- Works on **Windows x64** with **Python 3.x** (64-bit recommended).

**Example:**
```bash
# tests take the backend as argv[1] (Boost.UT name patterns follow it) — no gtest here
python scripts/debugger.py bin/debug/test_runtime.exe -- dx
```

## 9. Tracking Memory Growth with `scripts/mem_monitor.py`

A cross-platform (Windows/Linux) Python wrapper that launches a process, samples its memory at a fixed interval, writes a timeline log, and **kills the process tree** when private memory exceeds a threshold — essential when a runaway test could exhaust machine RAM before you can observe it.

- **Windows:** Private Bytes / Working Set via `GetProcessMemoryInfo`; kills the tree with `taskkill /F /T`.
- **Linux:** private memory = `Private_Clean + Private_Dirty` from `/proc/<pid>/smaps_rollup` (fallback `RssAnon + VmSwap`), working set = `VmRSS`; kills the child's whole process group with `SIGKILL` (the child is started in its own session).

**Usage:**
```bash
python scripts/mem_monitor.py [--kill-gb N] [--interval S] [--log FILE] -- <exe> [args...]
```

- `--kill-gb N` (default 20): hard kill threshold in GiB of private memory (Private Bytes on Windows, `smaps_rollup` private on Linux). Pick a value that leaves the machine usable (e.g. total free RAM minus headroom).
- `--interval S` (default 0.25): sampling interval in seconds.
- `--log FILE` (default `mem_monitor.log`): timeline output, one `time_s private_mb working_set_mb` row per sample.
- Arguments after `--` are forwarded to the target executable.
- On exit it prints a summary: exit code, whether it killed the process, peak private memory, and the first/last samples.

**Example (run a device test, kill if it exceeds 6 GiB):**
```bash
python scripts/mem_monitor.py --kill-gb 6 --interval 0.25 --log mem.log -- bin/debug/test_runtime.exe dx
```

**Workflow for a suspected leak / memory blow-up:**
1. Run the full repro under the monitor with a safe `--kill-gb`. A straight-line, ever-growing timeline (constant MiB/s) usually means an **unbounded accumulation loop** (e.g. appending commands to a `CommandList` that never terminates), not a classic leak; stepwise growth synchronized with test phases suggests per-phase leaks.
2. Isolate the phase: run individual tests (Boost.UT: positional test-name patterns after the backend arg, e.g. `test.exe <backend> my_test`) and compare peak memory and timeline shape.
3. A phase that terminates with flat memory is innocent; a phase whose memory grows without plateau owns the bug. Then read that phase's code for loops whose trip count can make zero progress (e.g. `n = min(remaining, k)` with `k == 0`).
4. After the fix, re-run under the monitor: peak private memory should be bounded (hundreds of MiB for small device tests) and the timeline flat.

**Note:** GPU/device memory is mostly invisible to host-side private memory; this tool tracks host-side growth. Pair with backend allocator logging if device-side exhaustion is suspected.

## Summary

- **Stack-traces** → innermost frame = cause; upward walk = call-site.
- **Always plan** before editing; record each failed attempt so it is not repeated.
- **No trace** → read `CMakeLists.txt`/`xmake.lua`, add `LUISA_INFO`/`LUISA_VERBOSE`, then `device_log`.
- **DSL values** → prefer `Buffer` write + host read-back for bulk inspection; use `device_log` for targeted per-thread messages.
- **Backend/codegen issues** → set `LUISA_DUMP_SOURCE=1` to inspect generated shaders and `LUISA_ENABLE_VALIDATION=1` to catch API/resource misuse.
- **Memory blow-up / suspected leak** → run under `scripts/mem_monitor.py` with `--kill-gb` (Windows/Linux); constant-rate growth = unbounded accumulation loop, stepwise growth = per-phase leak.
