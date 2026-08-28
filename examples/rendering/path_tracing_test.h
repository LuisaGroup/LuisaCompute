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
    double elapsed_ms{};
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
