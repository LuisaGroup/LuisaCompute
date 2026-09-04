---
name: debug
description: Debug crashes and test failures via stack-traces, host/device logging, and DSL buffer inspection.
---

# Debugging LuisaCompute

## 1. Interpreting Stack-Traces

When a crash or `LUISA_ERROR` is emitted, capture the full console output first.

**What to look for:**
- **Top frames** — the actual fault (null dereference, assertion, backend error).
- **LuisaCompute frames** — functions prefixed with `luisa::`, especially `luisa::compute::` or `luisa::dsl::`.
- **Backend frames** — `cuda`, `dx`, `metal`, `cpu` backend symbols tell you which path failed.
- **Last log line** — often the preceding `LUISA_INFO`/`LUISA_VERBOSE` shows the dispatch or shader name that triggered the bug.

**Action:**
1. Read the innermost frame (first after the crash header). This is the immediate cause.
2. Walk upward until you hit a recognizable LuisaCompute API call (e.g., `Device::compile`, `Stream::dispatch`, `Buffer::copy_from`). That is the *call-site*.
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
positional name patterns after the backend arg: `test.exe vk my_test`). When a
fix candidate emerges, A/B it with `git stash` — a failure that persists with
the fix stashed is pre-existing and out of scope; do not chase it.

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

Inside kernels, use `device_log` to emit per-thread messages. They are collected by the stream and flushed to the host callback or default logger.

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
// Example pattern from test_printer_custom_callback.cpp
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
            case 'E': luisa::log_error("{}", msg.substr(1)); break;
            default:  luisa::log_verbose("{}", msg); break;
        }
    }
});
```

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

**Reducer pattern for conditional values:**
- Allocate a `Buffer<uint>` counter at index 0.
- In the kernel, atomically increment the counter and write the debug payload into `debug_buf[counter]`.
- This captures the first N interesting threads without over-allocating.

## 6. Environment Variables for Backend Diagnosis

| Variable | Effect |
|---|---|
| `LUISA_DUMP_SOURCE=1` | Dumps generated shader sources/bytecode for the active backend. |
| `LUISA_LOG_LEVEL=verbose` | Equivalent to `log_level_verbose()` at startup. |
| `LUISA_ENABLE_VALIDATION=1` | Wraps the device in the validation layer (catches API misuse, out-of-bounds accesses, etc.). |
| `LUISA_OPTIX_VALIDATION=1` | Enables OptiX validation on the CUDA backend. |

Use `LUISA_DUMP_SOURCE=1` when you suspect a code-generation bug (wrong instruction, missing binding, incorrect type).

**Where to find the dumps:**
- **DirectX:** `hlsl_output_<name>.hlsl` in the current working directory.
- **Vulkan user compute (XIR→SPIR-V path):** `spv_code_<name>.spvasm` in the current working directory.
- **Vulkan user compute (LLVM→SPIR-V path):** `spv_code_llvm_<name>.spvasm`.
- **Vulkan internal HLSL consumers:** backend builtins/raster may dump `hlsl_output_<name>.hlsl`; ordinary `Device::compile(Function)` compute shaders must not.
- **CUDA:** `.cu` source in the runtime `.cache` directory; PTX/metadata in the runtime `.data` directory.
- **Metal:** `.metal` source in the runtime `.cache` directory.
- **Fallback:** the native C++ fallback backend also respects `LUISA_DUMP_SOURCE` and may dump intermediate sources.

The runtime directories are printed by `LUISA_INFO` at context creation; they default to the executable directory. When running under `xmake run`, dumps written directly to the current working directory will appear in the project root.

## 7. Decision Checklist

| Symptom | First Action | Next Action |
|---|---|---|
| Crash with stack-trace | Read innermost + first Luisa frame | Hypothesize → plan → fix |
| Silent wrong result | Add `LUISA_INFO` at host entry points | Use buffer read-back to inspect values |
| Kernel dispatch hangs | Check synchronize() and stream callback | Add minimal device_log at start of kernel |
| Hang burning CPU | Suspect spin on freed memory (UAF masquerading as hang) | Isolate per-test; audit destructor unlink paths |
| Unsure if failure is yours | Stash changes and re-run the same binary path | Persistent failure = pre-existing, out of scope |
| Backend compilation error | Set `LUISA_DUMP_SOURCE=1` | Inspect generated `.spvasm` or `.hlsl` |
| Suspected API/resource misuse | Set `LUISA_ENABLE_VALIDATION=1` | Re-run and read validation messages |
| Test timeout | Read build file for target entry | Narrow phase with host logging |

## 8. Windows Crash Debugging with `scripts/debugger.py`

A lightweight Python debugger using Windows Debug API + DbgHelp.dll to launch an x64 executable, catch second-chance exceptions, and print a symbolic stack trace from PDB symbols.

**Usage:**
```bash
python scripts/debugger.py <path_to_exe> [pdb_search_path] [-- <args>...]
```

- Arguments after `--` are forwarded to the target executable.
- The PDB must be next to the EXE or in `pdb_search_path`.
- Works on **Windows x64** with **Python 3.x** (64-bit recommended).

**Example:**
```bash
python scripts/debugger.py build/bin/test.exe -- --gtest_filter=MyTest
```

## 9. Tracking Memory Growth with `scripts/mem_monitor.py`

A Windows Python wrapper that launches a process, samples its memory (Private Bytes / Working Set via `GetProcessMemoryInfo`) at a fixed interval, writes a timeline log, and **kills the process tree** (`taskkill /F /T`) when Private Bytes exceed a threshold — essential when a runaway test could exhaust machine RAM before you can observe it.

**Usage:**
```bash
python scripts/mem_monitor.py [--kill-gb N] [--interval S] [--log FILE] -- <exe> [args...]
```

- `--kill-gb N` (default 20): hard kill threshold in GiB of Private Bytes. Pick a value that leaves the machine usable (e.g. total free RAM minus headroom).
- `--interval S` (default 0.25): sampling interval in seconds.
- `--log FILE` (default `mem_monitor.log`): timeline output, one `time_s private_mb working_set_mb` row per sample.
- Arguments after `--` are forwarded to the target executable.
- On exit it prints a summary: exit code, whether it killed the process, peak Private Bytes, and the first/last samples.

**Example (run a device test, kill if it exceeds 6 GiB):**
```bash
python scripts/mem_monitor.py --kill-gb 6 --interval 0.25 --log mem.log -- bin/debug/dgm-ao-kernels-test.exe dx
```

**Workflow for a suspected leak / memory blow-up:**
1. Run the full repro under the monitor with a safe `--kill-gb`. A straight-line, ever-growing timeline (constant MiB/s) usually means an **unbounded accumulation loop** (e.g. appending commands to a `CommandList` that never terminates), not a classic leak; stepwise growth synchronized with test phases suggests per-phase leaks.
2. Isolate the phase: run individual tests (Boost.UT: positional test-name patterns after the backend arg, e.g. `test.exe dx bench_uv_usage`) and compare peak memory and timeline shape.
3. A phase that terminates with flat memory is innocent; a phase whose memory grows without plateau owns the bug. Then read that phase's code for loops whose trip count can make zero progress (e.g. `n = min(remaining, k)` with `k == 0`).
4. After the fix, re-run under the monitor: peak Private Bytes should be bounded (hundreds of MiB for small device tests) and the timeline flat.

**Note:** GPU/device memory is mostly invisible to Private Bytes; this tool tracks host-side growth. Pair with backend allocator logging if device-side exhaustion is suspected.

## 10. Ray-Query / Accel Failures: Check Mesh-vs-Accel Lifetime First

If ray queries (Accel.intersect) return all-miss or backend-flaky hits (e.g. passes on dx, fails on vk; or results change depending on unrelated preceding dispatches), suspect a **dangling BLAS: the Mesh was destroyed while the Accel still references it**.

The lifetime contract:
- `Mesh` OWNS the BLAS. `~Mesh` -> `device->destroy_mesh` -> backends `delete` the BLAS object **immediately** (vk: `vkDestroyAccelerationStructureKHR` + immediate `vmaDestroyBuffer` of the BLAS storage).
- `Accel` (TLAS) references the BLAS only by **raw device address** baked into its instance buffer; it does NOT keep the Mesh/BLAS alive.
- Therefore the Mesh must outlive every use of the Accel. If a helper builds `{positions, triangles, mesh, accel}` and returns only the `Accel`, the BLAS is freed when the helper returns.

Why it "works" until it doesn't:
- The freed BLAS memory keeps valid content until a later allocation reuses it. dx tolerates this because its resource GC defers memory reuse past the GPU timeline; vk's VMA frees immediately, so the next large kernel-output buffer/arena allocation recycles the BLAS memory and subsequent traces read garbage -> deterministic-looking all-miss.
- Symptom fingerprint: early traces hit; traces after a sufficiently large compute dispatch (hundreds of writing threads, i.e. a big output buffer) all miss; a freshly-built accel traced afterwards works again; rebuilding the TLAS does NOT recover (the corruption is the freed BLAS, not the TLAS).

Diagnostic probes that isolated this (pattern-gated `world_accel_probe` in the AO test, since removed): trace literal axis-aligned rays (hit = freed memory still intact), full-scene downward-ray scan (dx shows floor hits everywhere, vk all-miss after a big dispatch), canary buffer scan (pristine: the "corruption" is memory *reuse*, not stray writes), fresh-accel retrace (works: per-accel lifetime bug, not global query-state breakage).

Fix: keep the `Mesh` alive alongside the `Accel` (e.g. return a struct holding both; meshes first so the accel is destroyed first).

## Summary

- **Stack-traces** → innermost frame = cause; upward walk = call-site.
- **Always plan** before editing; `StepMemory` saves failed attempts.
- **No trace** → read `CMakeLists.txt`/`xmake.lua`, add `LUISA_INFO`/`LUISA_VERBOSE`, then `device_log`.
- **DSL values** → prefer `Buffer` write + host read-back for bulk inspection; use `device_log` for targeted per-thread messages.
- **Backend/codegen issues** → set `LUISA_DUMP_SOURCE=1` to inspect generated shaders and `LUISA_ENABLE_VALIDATION=1` to catch API/resource misuse.
- **Memory blow-up / suspected leak** → run under `scripts/mem_monitor.py` with `--kill-gb`; constant-rate growth = unbounded accumulation loop, stepwise growth = per-phase leak.
- **Ray-query all-miss / dx-passes-vk-fails** → check the `Mesh` outlives the `Accel`; a helper returning only the `Accel` leaves a dangling BLAS whose freed memory vk reuses immediately (dx hides it via GC).
