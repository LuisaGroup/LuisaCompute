#pragma once

#include <cstdint>
#include <limits>

#include <hip/hip_runtime.h>

namespace luisa::compute::hip {

enum class HIPMotionMeshValidationError : uint8_t {
    NONE = 0u,
    NON_FINITE_POSITION = 1u,
    INVALID_TRIANGLE_INDEX = 2u,
};

struct alignas(16) HIPMotionMeshValidationStatus {
    static constexpr auto no_failure = std::numeric_limits<uint64_t>::max();
    static constexpr auto element_mask = (1ull << 48u) - 1ull;

    uint64_t failure_key{no_failure};
    uint64_t reserved{no_failure};

    [[nodiscard]] bool valid() const noexcept {
        return failure_key == no_failure;
    }
    [[nodiscard]] HIPMotionMeshValidationError error() const noexcept {
        return valid() ? HIPMotionMeshValidationError::NONE :
                         static_cast<HIPMotionMeshValidationError>(failure_key >> 56u);
    }
    [[nodiscard]] uint64_t element_index() const noexcept {
        return failure_key >> 8u & element_mask;
    }
    [[nodiscard]] uint32_t component() const noexcept {
        return static_cast<uint32_t>(failure_key & 0xffu);
    }
};

static_assert(sizeof(HIPMotionMeshValidationStatus) == 16u);
static_assert(alignof(HIPMotionMeshValidationStatus) == 16u);

class HIPMotionMeshBuiltin {

private:
    hipModule_t _module{};
    hipFunction_t _pack_motion_vertices{};
    hipFunction_t _build_motion_triangle_aabbs{};

public:
    HIPMotionMeshBuiltin() noexcept;
    ~HIPMotionMeshBuiltin() noexcept;
    HIPMotionMeshBuiltin(HIPMotionMeshBuiltin &&) = delete;
    HIPMotionMeshBuiltin(const HIPMotionMeshBuiltin &) = delete;
    HIPMotionMeshBuiltin &operator=(HIPMotionMeshBuiltin &&) = delete;
    HIPMotionMeshBuiltin &operator=(const HIPMotionMeshBuiltin &) = delete;

    void reset_validation_status(
        hipStream_t stream,
        hipDeviceptr_t validation_status) const noexcept;

    void pack_motion_vertices(
        hipStream_t stream,
        hipDeviceptr_t source_vertices,
        hipDeviceptr_t packed_vertices,
        uint64_t vertex_count,
        uint32_t source_vertex_stride,
        hipDeviceptr_t validation_status) const noexcept;

    void build_motion_triangle_aabbs(
        hipStream_t stream,
        hipDeviceptr_t source_triangles,
        hipDeviceptr_t packed_vertices,
        hipDeviceptr_t packed_triangles,
        hipDeviceptr_t aabbs,
        uint32_t triangle_count,
        uint32_t vertices_per_keyframe,
        uint32_t keyframe_count,
        hipDeviceptr_t validation_status) const noexcept;
};

}// namespace luisa::compute::hip
