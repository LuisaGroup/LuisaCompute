//
// Created by mike on 1/30/26.
//

#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rtx/mesh.h>

#include "hip_geometry.h"

namespace luisa::compute::hip {

class HIPCommandEncoder;

/// Device-visible deforming-triangle data consumed by hiprt_device_wrapper.hip.
/// Vertices for all keyframes are stacked contiguously as packed xyz float
/// triplets, while triangle indices are local to one keyframe. Keep this layout
/// in sync with the device wrapper.
struct alignas(16) HIPMotionTriangleDeviceData {
    uint64_t vertices;
    uint64_t triangles;
    uint32_t vertex_stride;
    uint32_t vertices_per_keyframe;
    uint32_t keyframe_count;
    uint32_t flags;
    float time_start;
    float time_end;
};

static_assert(sizeof(HIPMotionTriangleDeviceData) == 48u);
static_assert(alignof(HIPMotionTriangleDeviceData) == 16u);
static_assert(offsetof(HIPMotionTriangleDeviceData, vertices) == 0u);
static_assert(offsetof(HIPMotionTriangleDeviceData, triangles) == 8u);
static_assert(offsetof(HIPMotionTriangleDeviceData, vertex_stride) == 16u);
static_assert(offsetof(HIPMotionTriangleDeviceData, vertices_per_keyframe) == 20u);
static_assert(offsetof(HIPMotionTriangleDeviceData, keyframe_count) == 24u);
static_assert(offsetof(HIPMotionTriangleDeviceData, flags) == 28u);
static_assert(offsetof(HIPMotionTriangleDeviceData, time_start) == 32u);
static_assert(offsetof(HIPMotionTriangleDeviceData, time_end) == 36u);

class HIPMesh : public HIPGeometry {

private:
    AccelOption _option;
    hiprtContext _hiprt_ctx{nullptr};
    hiprtGeometry _geometry{nullptr};
    hiprtBuildFlags _build_flags{};
    hipDeviceptr_t _vertex_buffer{};
    size_t _vertex_buffer_size{};
    size_t _vertex_stride{};
    hipDeviceptr_t _triangle_buffer{};
    size_t _triangle_buffer_size{};
    hipDeviceptr_t _motion_vertex_buffer{};
    size_t _motion_vertex_buffer_size{};
    hipDeviceptr_t _motion_triangle_buffer{};
    size_t _motion_triangle_buffer_size{};
    hipDeviceptr_t _motion_aabb_buffer{};
    size_t _motion_aabb_buffer_size{};
    hipDeviceptr_t _motion_device_data{};
    mutable spin_mutex _mutex;

public:
    explicit HIPMesh(hiprtContext ctx, const AccelOption &option) noexcept;
    ~HIPMesh() noexcept;
    void build(HIPCommandEncoder &encoder, MeshBuildCommand *command) noexcept;
    [[nodiscard]] hiprtGeometry handle() const noexcept override {
        std::scoped_lock lock{_mutex};
        return _geometry;
    }
    [[nodiscard]] Kind kind() const noexcept override {
        return _option.motion.is_enabled() ? Kind::MOTION_TRIANGLE : Kind::TRIANGLE;
    }
    [[nodiscard]] uint64_t codegen_handle() const noexcept override {
        std::scoped_lock lock{_mutex};
        return _option.motion.is_enabled() ?
                   reinterpret_cast<uint64_t>(_motion_device_data) :
                   reinterpret_cast<uint64_t>(_geometry);
    }
    [[nodiscard]] auto option() const noexcept { return _option; }
};

}// namespace luisa::compute::hip
