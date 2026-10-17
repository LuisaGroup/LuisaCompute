// Debug & diagnostics harness for the bge-m3 embedding engine (plan §9.1,
// items D1/D3/D4 -- the always-on, zero-cost-when-idle layer). Everything
// heavier than this layer is opt-in via env vars or temporary scaffolding
// (plan §9.2/§9.3); nothing here is benchmarked or committed with guards.
//
// D1  install_validation_preset(): documents/enforces the
//     LUISA_ENABLE_VALIDATION=1 dev preset. The runtime reads the variable
//     once inside Context::create_device (src/runtime/context.cpp compares it
//     against exactly "1"), so call this BEFORE creating the device; the
//     helper only warns when the preset is off -- it never mutates the
//     environment.
//
// D3  route_stream_logs(stream): installs stream.set_log_callback with the
//     severity-prefix pattern from plan §9.1 D3. Device lines that start with
//     'I'/'W'/'E' are routed to luisa::log_info / luisa::log_warning; 'E'
//     lines are ALSO routed to log_warning, never to log_error -- luisa::
//     log_error is [[noreturn]] + std::abort, so one device-side 'E' would
//     kill the host process mid-flush (plan §9.1 D3 warning). Routed line
//     counts are exposed through the routed_*_count() helpers so a device
//     test can assert host-side receipt after synchronize(). Note that
//     set_log_callback REPLACES the previous callback: route_stream_logs must
//     be the only installer per stream (the scheduler should call it on every
//     worker stream at creation -- see the integration note in the B4 report).
//
//     DEVICE_INFO/DEVICE_WARNING/DEVICE_ERROR are kernel-side macros that
//     prepend the severity letter to a device_log line (pattern proven in
//     src/tests/unit/runtime/test_printer_custom_callback.cpp). They are
//     text-only: a kernel TU using them must include <luisa/dsl/syntax.h>
//     (device_log) and <luisa/core/stl/format.h> (luisa::format) -- the DSL
//     headers are deliberately NOT pulled into this host-side header.
//
// D4  Activation dump ("which of the 24 layers diverged?" bisect harness).
//     The executor owns a layer-trace hook (Executor::set_layer_trace, B1
//     contract, embed/executor.h: callback (uint32_t layer, BufferView<float>
//     activations, rows, cols); layer 0 = embedding output, 1..24 = encoder
//     post-LN2 -- the golden_layers.bin indexing). The hook is fired on the
//     CALLER's host thread after the encode completed
//     (Executor::fire_trace_callbacks()); the BufferView addresses an
//     immutable trace-slot buffer, so reading it back on a private stream is
//     safe. install_layer_dump() registers (layer -> out_dir) entries (one
//     hook-set per install re-arms the dispatcher; installs accumulate) and,
//     for the NEXT encode, reads the traced layer back to
//     <out_dir>/layer<layer>.bin (raw fp32, row-major) plus
//     <out_dir>/layer<layer>.json (rows/cols metadata).
//     dump_requested_from_env() parses BGEM3_DUMP_LAYER=<n> so any tool gets
//     the harness from a single env var via maybe_install_layer_dump(). Out
//     dirs must be ASCII-representable in the native code page (files are
//     opened through luisa::to_string, never throwing).
//
//     The D4 executor-bound API is only declared while the executor header
//     exists (same-wave integration guard); the env parser and the D1/D3
//     layer are unconditional.
//
// Device logs and trace callbacks are asynchronous: synchronize() the stream
// (and let encode() return) before asserting on routed counts or dump files.
#pragma once

#include <cstdint>
#include <filesystem>

#include <luisa/runtime/stream.h>

// Same-wave integration guard: the executor (layer-trace hook owner) and the
// Session facade are owned by B1. When they land, the D4 APIs below become
// live automatically (the Executor-level API needs executor.h; the Session
// overload additionally needs session.h).
#if __has_include("embedding/embed/executor.h")
#define LUISA_EMBED_HAS_EXECUTOR 1
#endif
#if __has_include("embedding/embed/session.h")
#define LUISA_EMBED_HAS_SESSION 1
#endif

namespace luisa::embed {

// --- D1: validation-layer dev preset ---------------------------------------
// Warns unless LUISA_ENABLE_VALIDATION is set to exactly "1" (the only value
// Context::create_device honors, src/runtime/context.cpp). Call before device
// creation; see plan §9.1 D1 and the debug skill §3 for what the validation
// layer turns API misuse into.
void install_validation_preset() noexcept;

// --- D3: structured device-log routing -------------------------------------
// Installs the severity-prefix log callback on `stream` (replaces any
// previously installed callback). See the file header for the routing table.
void route_stream_logs(compute::Stream &stream) noexcept;
[[nodiscard]] uint64_t routed_info_count() noexcept;   // 'I' lines -> log_info
[[nodiscard]] uint64_t routed_warning_count() noexcept;// 'W' lines -> log_warning
[[nodiscard]] uint64_t routed_error_count() noexcept;  // 'E' lines -> log_warning (never log_error)
[[nodiscard]] uint64_t routed_other_count() noexcept;  // unprefixed lines -> log_verbose
[[nodiscard]] uint64_t routed_total_count() noexcept;  // every callback invocation
void reset_routed_counts() noexcept;                   // for tests

// Kernel-side severity macros: wrap device_log with the D3 prefix and append
// the dispatch id -- the proven pattern of
// src/tests/unit/runtime/test_printer_custom_callback.cpp. Note the
// "I{} [dispatch{{}}]" outer template: the DSL-side luisa::format is
// evaluated at record time, so the caller's FMT must be passed as the FIRST
// argument (a compile-time string) and never spliced into the template
// (a "{}" inside the template with no matching argument aborts fmt at record
// time). #ifndef-guarded so a kernel TU that already carries the reference
// copy of the pattern does not get a macro-redefinition warning.
#ifndef DEVICE_INFO
#define DEVICE_INFO(FMT, ...) \
    device_log(luisa::format("I{} [dispatch{{}}]", FMT) __VA_OPT__(, ) __VA_ARGS__, $dispatch_id)
#endif
#ifndef DEVICE_WARNING
#define DEVICE_WARNING(FMT, ...) \
    device_log(luisa::format("W{} [dispatch{{}}]", FMT) __VA_OPT__(, ) __VA_ARGS__, $dispatch_id)
#endif
#ifndef DEVICE_ERROR
#define DEVICE_ERROR(FMT, ...) \
    device_log(luisa::format("E{} [dispatch{{}}]", FMT) __VA_OPT__(, ) __VA_ARGS__, $dispatch_id)
#endif

// --- D4: activation dump harness --------------------------------------------
// Parses BGEM3_DUMP_LAYER. Returns true and fills `layer` when the variable
// holds a decimal value that fits uint32_t; false (layer untouched) when unset
// or unparsable (a malformed value is reported with LUISA_WARNING).
[[nodiscard]] bool dump_requested_from_env(uint32_t &layer) noexcept;

#if defined(LUISA_EMBED_HAS_EXECUTOR)
class Executor;// complete type via embedding/embed/executor.h (B1-owned)

// Registers `layer` for a dump into `out_dir` on the next encode and arms the
// executor trace hook. `device` is used to create the read-back stream and
// must outlive the session usage it dumps for (all installs of one process
// must use the same device). Idempotent per layer: a repeated install
// replaces the previous entry for that layer.
void install_layer_dump(Executor &executor, compute::Device &device,
                        uint32_t layer, const std::filesystem::path &out_dir) noexcept;

// One-liner for tools: if BGEM3_DUMP_LAYER is set, install_layer_dump() with
// the requested layer into `out_dir`. Returns true when a dump was armed.
[[nodiscard]] bool maybe_install_layer_dump(Executor &executor, compute::Device &device,
                                            const std::filesystem::path &out_dir) noexcept;

// Successfully written dump files (bin + json pair) since the last reset.
// A device test calls encode() (and, at Executor level,
// fire_trace_callbacks()), then expects this to equal the number of
// registered layers.
[[nodiscard]] uint64_t layer_dump_completed_count() noexcept;
void reset_layer_dump_completed_count() noexcept;
#endif// LUISA_EMBED_HAS_EXECUTOR

#if defined(LUISA_EMBED_HAS_SESSION) && defined(LUISA_EMBED_HAS_EXECUTOR)
class Session;// complete type via embedding/embed/session.h (B1-owned)

// Session-level convenience: forwards to Executor::set_layer_trace via
// Session::set_layer_trace (B1: broadcasts to every worker executor,
// synchronous firing). Same registry semantics as the Executor overload.
void install_layer_dump(Session &session, compute::Device &device,
                        uint32_t layer, const std::filesystem::path &out_dir) noexcept;
[[nodiscard]] bool maybe_install_layer_dump(Session &session, compute::Device &device,
                                            const std::filesystem::path &out_dir) noexcept;
#endif// Session gate

}// namespace luisa::embed
