#include <algorithm>
#include <luisa/core/logging.h>

#include "hip_check.h"
#include "hip_motion_mesh_builtin.h"
#include "hip_motion_mesh_builtin_embedded.h"

namespace luisa::compute::hip {

namespace {

constexpr auto motion_mesh_block_size = 256u;
constexpr auto motion_mesh_max_block_count = 65535u;

[[nodiscard]] uint32_t motion_mesh_block_count(uint64_t count) noexcept {
    auto blocks = (count + motion_mesh_block_size - 1u) /
                  motion_mesh_block_size;
    return static_cast<uint32_t>(
        std::min<uint64_t>(blocks, motion_mesh_max_block_count));
}

}// namespace

HIPMotionMeshBuiltin::HIPMotionMeshBuiltin() noexcept {
    LUISA_ASSERT(luisa_compute_hip_motion_mesh_builtin_size != 0u,
                 "Embedded HIP motion-mesh module is empty.");
    LUISA_CHECK_HIP(hipModuleLoadData(
        &_module, luisa_compute_hip_motion_mesh_builtin));
    LUISA_CHECK_HIP(hipModuleGetFunction(
        &_pack_motion_vertices, _module, "pack_motion_vertices"));
    LUISA_CHECK_HIP(hipModuleGetFunction(
        &_build_motion_triangle_aabbs, _module,
        "build_motion_triangle_aabbs"));
}

HIPMotionMeshBuiltin::~HIPMotionMeshBuiltin() noexcept {
    if (_module != nullptr) {
        LUISA_CHECK_HIP(hipModuleUnload(_module));
    }
}

void HIPMotionMeshBuiltin::reset_validation_status(
    hipStream_t stream,
    hipDeviceptr_t validation_status) const noexcept {
    LUISA_ASSERT(validation_status != nullptr,
                 "HIP motion-mesh validation status is null.");
    LUISA_CHECK_HIP(hipMemsetAsync(
        validation_status, 0xff,
        sizeof(HIPMotionMeshValidationStatus), stream));
}

void HIPMotionMeshBuiltin::pack_motion_vertices(
    hipStream_t stream,
    hipDeviceptr_t source_vertices,
    hipDeviceptr_t packed_vertices,
    uint64_t vertex_count,
    uint32_t source_vertex_stride,
    hipDeviceptr_t validation_status) const noexcept {
    if (vertex_count == 0u) { return; }
    LUISA_ASSERT(source_vertices != nullptr && packed_vertices != nullptr &&
                     validation_status != nullptr,
                 "Null buffer passed to HIP motion-vertex packing.");
    LUISA_ASSERT(source_vertex_stride >= sizeof(float) * 3u,
                 "Invalid HIP motion-mesh vertex stride {}.",
                 source_vertex_stride);
    LUISA_ASSERT(vertex_count <=
                     HIPMotionMeshValidationStatus::element_mask,
                 "HIP motion-mesh vertex count {} exceeds the validation ABI limit.",
                 vertex_count);
    constexpr auto position_size = sizeof(float) * 3u;
    LUISA_ASSERT(vertex_count == 0u ||
                     vertex_count - 1u <=
                         (std::numeric_limits<uint64_t>::max() - position_size) /
                             source_vertex_stride,
                 "HIP motion-mesh source address calculation overflows "
                 "(count = {}, stride = {}).",
                 vertex_count, source_vertex_stride);
    void *arguments[]{
        &source_vertices,
        &packed_vertices,
        &vertex_count,
        &source_vertex_stride,
        &validation_status};
    LUISA_CHECK_HIP(hipModuleLaunchKernel(
        _pack_motion_vertices,
        motion_mesh_block_count(vertex_count), 1u, 1u,
        motion_mesh_block_size, 1u, 1u,
        0u, stream, arguments, nullptr));
}

void HIPMotionMeshBuiltin::build_motion_triangle_aabbs(
    hipStream_t stream,
    hipDeviceptr_t source_triangles,
    hipDeviceptr_t packed_vertices,
    hipDeviceptr_t packed_triangles,
    hipDeviceptr_t aabbs,
    uint32_t triangle_count,
    uint32_t vertices_per_keyframe,
    uint32_t keyframe_count,
    hipDeviceptr_t validation_status) const noexcept {
    if (triangle_count == 0u) { return; }
    LUISA_ASSERT(source_triangles != nullptr && packed_vertices != nullptr &&
                     packed_triangles != nullptr && aabbs != nullptr &&
                     validation_status != nullptr,
                 "Null buffer passed to HIP motion-triangle preprocessing.");
    LUISA_ASSERT(vertices_per_keyframe != 0u && keyframe_count != 0u,
                 "HIP motion-triangle preprocessing requires nonzero vertex and keyframe counts.");
    void *arguments[]{
        &source_triangles,
        &packed_vertices,
        &packed_triangles,
        &aabbs,
        &triangle_count,
        &vertices_per_keyframe,
        &keyframe_count,
        &validation_status};
    LUISA_CHECK_HIP(hipModuleLaunchKernel(
        _build_motion_triangle_aabbs,
        motion_mesh_block_count(triangle_count), 1u, 1u,
        motion_mesh_block_size, 1u, 1u,
        0u, stream, arguments, nullptr));
}

}// namespace luisa::compute::hip
