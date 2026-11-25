#pragma once

#include <luisa/xir/op.h>

namespace lc::spirv {

struct SpirvBindlessResourceUsage {
    // Buffer descriptors and buffer-view metadata are independent domains.
    // Every bindless buffer access - typed or untyped - takes its descriptor
    // index and byte bias from the per-array metadata record: the slot record of
    // the indices buffer is only 3 words for a MULTIPLE array, 4 for a
    // BUFFER_ONLY one and 1 for a texture-only one, so no single shader access
    // could read the typed bias from it. Buffer reads/writes therefore consume
    // metadata plus the global heap, and every size/address query consumes
    // metadata.
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
    luisa::compute::xir::ResourceQueryOp op,
    luisa::compute::xir::BindlessResourceAccess access = {}) noexcept;

[[nodiscard]] SpirvBindlessResourceUsage
spirv_bindless_resource_usage(
    luisa::compute::xir::ResourceReadOp op,
    luisa::compute::xir::BindlessResourceAccess access = {}) noexcept;

[[nodiscard]] SpirvBindlessResourceUsage
spirv_bindless_resource_usage(
    luisa::compute::xir::ResourceWriteOp op,
    luisa::compute::xir::BindlessResourceAccess access = {}) noexcept;

}// namespace lc::spirv
