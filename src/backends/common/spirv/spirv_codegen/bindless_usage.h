#pragma once

#include <luisa/xir/op.h>

namespace lc::spirv {

struct SpirvBindlessResourceUsage {
    // Buffer size queries consume only the per-array local metadata buffer.
    // Actual reads/writes additionally consume the global unbounded heap.
    bool buffer_heap{false};
    bool buffer_metadata{false};
    bool texture_2d{false};
    bool texture_3d{false};

    constexpr void merge(SpirvBindlessResourceUsage other) noexcept {
        buffer_heap |= other.buffer_heap;
        buffer_metadata |= other.buffer_metadata;
        texture_2d |= other.texture_2d;
        texture_3d |= other.texture_3d;
    }
};

// These classifiers describe resources required by exact XIR instructions.
// Reachable optimized XIR is the sole native emission source, so descriptor
// planning deliberately does not consult the broader AST builtin bitset.
[[nodiscard]] SpirvBindlessResourceUsage
spirv_bindless_resource_usage(
    luisa::compute::xir::ResourceQueryOp op) noexcept;

[[nodiscard]] SpirvBindlessResourceUsage
spirv_bindless_resource_usage(
    luisa::compute::xir::ResourceReadOp op) noexcept;

[[nodiscard]] SpirvBindlessResourceUsage
spirv_bindless_resource_usage(
    luisa::compute::xir::ResourceWriteOp op) noexcept;

}// namespace lc::spirv
