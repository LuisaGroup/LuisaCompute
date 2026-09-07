#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {
class Device;
class Window;
}// namespace luisa::compute

namespace luisa::ref {

struct PathTracingTestResult {
    bool success{};
    luisa::string error;
    uint2 resolution{};
    uint64_t completed_spp{};
    /// Backward-compatible elapsed time. When collect_stage_timings is true,
    /// this is the synchronized pure render time; otherwise it retains the
    /// historical end-to-end render-loop timing.
    double elapsed_ms{};
    double scene_setup_cpu_ms{};
    double acceleration_build_ms{};
    double kernel_definition_ms{};
    double shader_compile_ms{};
    double initialization_ms{};
    double render_ms{};
    double readback_ms{};
    double total_ms{};
    luisa::vector<std::array<uint8_t, 4u>> pixels;
};

using PathTracingProgressCallback = luisa::function<void(
    uint64_t completed_spp, double elapsed_ms)>;
using PathTracingSnapshotCallback = luisa::function<void(
    uint2 resolution, uint64_t completed_spp, double elapsed_ms,
    const luisa::vector<std::array<uint8_t, 4u>> &pixels)>;

struct PathTracingTestOptions {
    bool offline{true};
    uint32_t spp{1u};
    std::optional<uint32_t> max_spp_per_dispatch;
    /// Insert synchronization boundaries around setup, rendering, and
    /// readback so benchmarks can report non-overlapping stage timings.
    /// Disabled by default to preserve the interactive renderer's pipelining.
    bool collect_stage_timings{false};
    /// Optional externally owned platform window. If omitted, interactive
    /// desktop runs create a GLFW-backed Window as before. iOS supplies a
    /// Window that wraps its UIKit-owned CAMetalLayer.
    compute::Window *window{nullptr};
    /// If non-zero, performs a one-time readback after this many accumulated
    /// samples and invokes snapshot_callback while interactive rendering keeps
    /// running.
    uint32_t snapshot_spp{};
    PathTracingProgressCallback progress_callback;
    PathTracingSnapshotCallback snapshot_callback;
};

[[nodiscard]] PathTracingTestResult run_path_tracing_test(
    compute::Device &device,
    const PathTracingTestOptions &options);

}// namespace luisa::ref
