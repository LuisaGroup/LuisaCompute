#pragma once

// Ordered batch shader compilation for the Vulkan backend (plan P5.1).
//
// What this extension is
// ----------------------
// `create_shader()` compiles one kernel on the calling thread: code generation,
// SPIR-V optimization, SPIR-V validation, artifact encoding and finally
// `vkCreateShaderModule` + `vkCreateComputePipelines`. On a cold start a
// consumer that compiles a few hundred kernels pays that whole cost serially
// (plan §4 F9: no CPU parallelism exists in the backend today).
//
// This extension splits the work of a *batch* of requests into two phases:
//
//  1. **prepare** (parallel, on bounded preparation lanes): everything that
//     produces host data - the XIR -> SPIR-V route decision, the shader-cache
//     lookup, code generation, the SPIR-V optimizer and validator, and the
//     encoding of the bytecode artifact. No Vulkan object is created here, so a
//     preparation lane never enters the driver. The lanes run on the marl fiber
//     scheduler captured on the Device constructor thread when one was bound;
//     otherwise the device owns std::thread workers. A captured scheduler is
//     external and must outlive the Device and every batch.
//  2. **publish** (serial, in request order, on the calling thread):
//     `vkCreateShaderModule`, `vkCreateComputePipelines` (through the device
//     pipeline cache's external-synchronization lease), the interned-layout
//     acquisition of the new `Shader`, and the shader-cache write of the artifact
//     the prepare phase produced.
//
// The sequence number of a request is its index in the batch, so publication
// order is the caller's order, independent of which worker finished first. The
// handles in `results` therefore always correspond one-to-one, in order, to the
// requests - a batch produces the same shaders, in the same order, as a serial
// loop of `Device::compile` calls on the same device.
//
// What is refused, and how
// ------------------------
// Nothing here throws and nothing is fatal. A request that cannot be prepared on
// a worker falls back to serial compilation *on the publishing thread* and is
// reported with `serial_fallback = true`:
//
//  * the compatibility HLSL-to-SPIR-V route (kernels with a `native_include`
//    block, printers, async copy, the software ray-tracing fallback, the BF16
//    fallback, `accel.size()` and a few other shapes - see
//    `detail::plan_user_compute_codegen_route`), because that route is compiled by
//    the process-global DXC compatibility compiler whose thread contract this
//    backend does not own;
//  * `ShaderOption::compile_only` requests (AOT emission is a file-producing
//    operation that keeps its historical serial shape);
//  * a request made on a worker thread of this pool (nested compilation), and any
//    request after `shutdown_pool()` - the pool never reopens.
//
// A serial fallback costs exactly what the same request costs today, so a batch
// that uses it can only be as slow as the serial loop, never slower.
//
// Threading contract
// ------------------
//  * `compile_shaders()` is synchronous. It may be called concurrently from
//    several threads or scheduler fibers; each call prepares on the shared pool
//    and publishes its own requests in its own order. A call made re-entrantly
//    from one of this pool's preparation lanes falls back to serial compilation
//    instead of waiting for the pool itself. A scheduler captured with zero
//    worker threads likewise uses serial preparation and never creates an
//    owning std::thread fallback.
//  * The caller must keep every request's AST alive for the duration of the call
//    (the same rule as `Device::compile`, which needs the DSL kernel object to
//    outlive the call). The requests are not consumed.
//  * `shutdown_pool()` is idempotent and may be called while a batch is in
//    flight: already-queued requests still complete, and every later request
//    reports `POOL_SHUTDOWN`. `~Device` calls it before it destroys the
//    pipeline cache, the interned layouts and the BinaryIO. Shutdown joins only
//    workers owned by the pool; an external fiber scheduler remains caller-owned.
//  * The extension holds no per-call state of its own: it is a view over the
//    device, like `VulkanReplayExt`.
//
// The header deliberately includes no Vulkan header and names no backend-private
// type: handles are `uint64_t`, statuses are plain enums.
#include <cstdint>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute {

// Outcome of one entry point. `SUCCESS` is 0, so `status != SUCCESS` is the
// failure test.
enum class VulkanShaderCompileStatus : uint32_t {
    SUCCESS = 0u,
    // The request list was empty; nothing was compiled.
    INVALID_ARGUMENT,
    // The pool was stopped (`shutdown_pool()` or device teardown), so the batch
    // was not scheduled. No shader was created.
    POOL_SHUTDOWN,
    // The batch itself could not be scheduled (e.g. worker creation failed) and
    // was compiled serially instead. Every `results[i]` still holds a real shader
    // when its own status is `SUCCESS`.
    FELL_BACK_TO_SERIAL,
    // The shader failed to compile (the backend reported the reason). Mirrors a
    // `compile_ok == false` answer of a serial compile.
    COMPILATION_FAILED,
};

[[nodiscard]] constexpr luisa::string_view
vulkan_shader_compile_status_name(VulkanShaderCompileStatus status) noexcept {
    switch (status) {
        case VulkanShaderCompileStatus::SUCCESS: return "success";
        case VulkanShaderCompileStatus::INVALID_ARGUMENT: return "invalid_argument";
        case VulkanShaderCompileStatus::POOL_SHUTDOWN: return "pool_shutdown";
        case VulkanShaderCompileStatus::FELL_BACK_TO_SERIAL: return "fell_back_to_serial";
        case VulkanShaderCompileStatus::COMPILATION_FAILED: return "compilation_failed";
    }
    return "unknown";
}

// One shader to compile. `kernel` is the AST handed to `Device::create_shader`;
// the caller keeps it (and the DSL kernel object that owns it) alive for the
// duration of `compile_shaders()`.
struct VulkanShaderCompileRequest {
    ShaderOption option;
    Function kernel;
};

// The answer for one request. `shader` is valid iff `status == SUCCESS`.
struct VulkanShaderCompileResult {
    VulkanShaderCompileStatus status{VulkanShaderCompileStatus::SUCCESS};
    ShaderCreationInfo shader;
    // True when this particular request was compiled on the publishing thread
    // instead of a pool worker (HLSL compatibility route, compile-only, nested
    // compilation or a worker-context setup failure). The shader is still valid.
    bool serial_fallback{false};
    // Filled for every non-SUCCESS status.
    luisa::string message;
};

struct VulkanShaderCompileCapabilities {
    // The two structural answers of this implementation.
    bool parallel_prepare{true};
    bool ordered_publication{true};
    // True when this build can prepare the compatibility HLSL-to-SPIR-V route on
    // a worker (per-worker DXC context). False means such a request always
    // reports `serial_fallback`.
    bool parallel_hlsl_compatibility_route{false};
    // Upper bound of `worker_count` a caller may ask for; the effective lane
    // count is clamped to it and to the available hardware/scheduler concurrency.
    uint32_t max_workers{32u};
};

// Deterministic counters of the pool, for tests and bug reports. They never feed
// a decision.
struct VulkanShaderCompileStats {
    uint32_t worker_count{0u};
    bool pool_created{false};
    bool pool_shutdown{false};
    uint64_t batches{0u};
    uint64_t requests{0u};
    // Requests whose prepare phase really ran on a worker.
    uint64_t parallel_prepares{0u};
    // Requests prepared on the publishing thread.
    uint64_t serial_prepares{0u};
    // Requests whose cached artifact was loaded instead of compiled.
    uint64_t cache_hits{0u};
    // Requests that reported `serial_fallback`.
    uint64_t serial_fallbacks{0u};
    // How often the ordered publisher had to wait for the next sequence number,
    // and the total host time it spent waiting (ns).
    uint64_t publish_order_waits{0u};
    uint64_t publish_order_wait_ns{0u};
};

// Ordered batch compilation of compute shaders (plan P5.1).
class VulkanShaderCompileExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "VulkanShaderCompileExt";
    [[nodiscard]] virtual VulkanShaderCompileCapabilities capabilities() const noexcept = 0;
    // Compile `requests` and publish the shaders in request order on the calling
    // thread. `results` is resized to `requests.size()`; `results[i]` describes
    // `requests[i]`. `worker_count == 0` means "use the device default"
    // (one fewer than hardware_concurrency for owned threads, or the captured
    // scheduler's worker count for fibers, clamped to `capabilities().max_workers`);
    // `worker_count == 1` runs the same two-phase machinery with a single lane,
    // which is the A/B control for the pool.
    virtual VulkanShaderCompileStatus compile_shaders(
        luisa::span<const VulkanShaderCompileRequest> requests,
        luisa::vector<VulkanShaderCompileResult> &results,
        uint32_t worker_count = 0u) noexcept = 0;
    // Stops and joins the worker pool. Idempotent. After this call every batch
    // reports `POOL_SHUTDOWN`; the pool is never restarted, so a device whose
    // pool was shut down keeps behaving exactly like a device that never had one.
    virtual void shutdown_pool() noexcept = 0;
    [[nodiscard]] virtual VulkanShaderCompileStats stats() const noexcept = 0;
protected:
    ~VulkanShaderCompileExt() = default;
};

}// namespace luisa::compute