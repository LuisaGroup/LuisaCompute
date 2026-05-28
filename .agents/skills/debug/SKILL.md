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
    device_log(luisa::format("I" FMT), __VA_ARGS__)

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

**Important:** Device logs are asynchronous. Always `synchronize()` the stream before assuming all logs have arrived.

## 5. Using Printer and Buffer for DSL Debug

When you need to inspect many values or avoid per-thread log flooding, write results into a `Buffer` and read back on the host.

**Buffer-based inspection:**
```cpp
Buffer<float4> debug_buf = device.create_buffer<float4>(1024);

Kernel1D k = [&debug_buf](BufferVar<float4> out) noexcept {
    UInt idx = dispatch_id().x;
    Float4 v = make_float4(idx, idx * 2.0f, idx * 3.0f, 0.0f);
    out.write(idx, v);
};

auto shader = device.compile(k);
stream << shader(debug_buf).dispatch(1024)
       << synchronize();

// Read back
luisa::vector<float4> host(1024);
stream << debug_buf.copy_to(host.data()) << synchronize();
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
| `LUISA_DUMP_SOURCE=1` | Dumps SPIR-V assembly to `bin/debug/spirv_output.spvasm` and HLSL to `bin/debug/hlsl_output.hlsl`. |
| `LUISA_LOG_LEVEL=verbose` | Equivalent to `log_level_verbose()` at startup. |

Use `LUISA_DUMP_SOURCE=1` when you suspect a code-generation bug (wrong instruction, missing binding, incorrect type).

## 7. Decision Checklist

| Symptom | First Action | Next Action |
|---|---|---|
| Crash with stack-trace | Read innermost + first Luisa frame | Hypothesize → plan → fix |
| Silent wrong result | Add `LUISA_INFO` at host entry points | Use buffer read-back to inspect values |
| Kernel dispatch hangs | Check `synchronize()` and stream callback | Add minimal `device_log` at start of kernel |
| Backend compilation error | Set `LUISA_DUMP_SOURCE=1` | Inspect generated `.spvasm` or `.hlsl` |
| Test timeout | Read build file for target entry | Narrow phase with host logging |

## Summary

- **Stack-traces** → innermost frame = cause; upward walk = call-site.
- **Always plan** before editing; `StepMemory` saves failed attempts.
- **No trace** → read `CMakeLists.txt`/`xmake.lua`, add `LUISA_INFO`/`LUISA_VERBOSE`, then `device_log`.
- **DSL values** → prefer `Buffer` write + host read-back for bulk inspection; use `device_log` for targeted per-thread messages.
