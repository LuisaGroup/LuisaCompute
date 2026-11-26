//
// Created by mike on 1/30/26.
//

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

#include <luisa/core/logging.h>
#include <luisa/core/pool.h>
#include <luisa/runtime/rtx/aabb.h>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_command_encoder.h"
#include "hip_motion_mesh_builtin.h"
#include "hip_stage_buffer_pool.h"
#include "hip_stream.h"
#include "hip_device.h"
#include "hip_geometry_build_policy.h"
#include "hip_geometry_memory_policy.h"
#include "hip_mesh.h"

namespace luisa::compute::hip {

namespace {

struct HIPMeshSceneProfileStats {
    std::atomic<uint64_t> meshes_built{0u};
    std::atomic<uint64_t> compacted_count{0u};
    std::atomic<uint64_t> recreate_count{0u};
    std::atomic<uint64_t> batch_calls{0u};
    std::atomic<uint64_t> batch_built{0u};
    std::atomic<uint64_t> ploc_built{0u};
    std::atomic<uint64_t> sbvh_built{0u};
    std::atomic<uint64_t> lbvh_built{0u};
    std::atomic<uint64_t> policy_ns{0u};
    std::atomic<uint64_t> create_ns{0u};
    std::atomic<uint64_t> build_ns{0u};
    std::atomic<uint64_t> compact_ns{0u};
    std::atomic<uint64_t> total_ns{0u};
    std::atomic<uint64_t> bucket_le_64{0u};
    std::atomic<uint64_t> bucket_65_512{0u};
    std::atomic<uint64_t> bucket_513_4096{0u};
    std::atomic<uint64_t> bucket_gt_4096{0u};
};

[[nodiscard]] bool hip_scene_profile_enabled() noexcept {
    static const bool enabled = [] {
        using namespace std::string_view_literals;
        auto env = std::getenv("LUISA_HIP_PROFILE_SCENE_BUILD");
        return env != nullptr && std::string_view{env} == "1"sv;
    }();
    return enabled;
}

[[nodiscard]] uint32_t hip_batch_build_max_prims() noexcept {
    static const uint32_t limit = [] {
        auto env = std::getenv("LUISA_HIP_BATCH_BUILD_MAX_PRIMS");
        if (env == nullptr || env[0] == '\0') {
            return static_cast<uint32_t>(hiprtMaxBatchBuildMaxPrimCount);
        }
        char *end = nullptr;
        auto parsed = std::strtoull(env, &end, 10);
        if (end == env || *end != '\0') {
            return static_cast<uint32_t>(hiprtMaxBatchBuildMaxPrimCount);
        }
        return static_cast<uint32_t>(
            std::min<uint64_t>(parsed, hiprtMaxBatchBuildMaxPrimCount));
    }();
    return limit;
}

[[nodiscard]] uint64_t hip_sbvh_min_tris() noexcept {
    static const uint64_t threshold = [] {
        auto env = std::getenv("LUISA_HIP_SBVH_MIN_TRIS");
        if (env == nullptr || env[0] == '\0') {
            return std::numeric_limits<uint64_t>::max();
        }
        char *end = nullptr;
        auto parsed = std::strtoull(env, &end, 10);
        if (end == env || *end != '\0') {
            return std::numeric_limits<uint64_t>::max();
        }
        return static_cast<uint64_t>(parsed);
    }();
    return threshold;
}

[[nodiscard]] HIPMeshSceneProfileStats &hip_mesh_scene_profile_stats() noexcept {
    static HIPMeshSceneProfileStats stats;
    static const bool registered = [] {
        if (hip_scene_profile_enabled()) {
            std::atexit([] {
                auto &s = stats;
                auto n = s.meshes_built.load(std::memory_order_relaxed);
                if (n == 0u) { return; }
                auto to_ms = [](uint64_t ns) noexcept {
                    return static_cast<double>(ns) * 1e-6;
                };
                LUISA_INFO(
                    "[HIP_SCENE_PROFILE] meshes={} compacted={} recreated={} "
                    "total={:.2f}ms (policy={:.2f}ms, create={:.2f}ms, build={:.2f}ms, compact={:.2f}ms) "
                    "tris[<=64:{}, 65..512:{}, 513..4096:{}, >4096:{}]",
                    n,
                    s.compacted_count.load(std::memory_order_relaxed),
                    s.recreate_count.load(std::memory_order_relaxed),
                    to_ms(s.total_ns.load(std::memory_order_relaxed)),
                    to_ms(s.policy_ns.load(std::memory_order_relaxed)),
                    to_ms(s.create_ns.load(std::memory_order_relaxed)),
                    to_ms(s.build_ns.load(std::memory_order_relaxed)),
                    to_ms(s.compact_ns.load(std::memory_order_relaxed)),
                    s.bucket_le_64.load(std::memory_order_relaxed),
                    s.bucket_65_512.load(std::memory_order_relaxed),
                    s.bucket_513_4096.load(std::memory_order_relaxed),
                    s.bucket_gt_4096.load(std::memory_order_relaxed));
                LUISA_INFO(
                    "[HIP_SCENE_PROFILE] batches={} builder[batch:{}, ploc:{}, sbvh:{}, lbvh:{}]",
                    s.batch_calls.load(std::memory_order_relaxed),
                    s.batch_built.load(std::memory_order_relaxed),
                    s.ploc_built.load(std::memory_order_relaxed),
                    s.sbvh_built.load(std::memory_order_relaxed),
                    s.lbvh_built.load(std::memory_order_relaxed));
            });
        }
        return true;
    }();
    static_cast<void>(registered);
    return stats;
}

void record_mesh_profile_buckets(uint64_t triangle_count) noexcept {
    auto &stats = hip_mesh_scene_profile_stats();
    stats.meshes_built.fetch_add(1u, std::memory_order_relaxed);
    if (triangle_count <= 64u) {
        stats.bucket_le_64.fetch_add(1u, std::memory_order_relaxed);
    } else if (triangle_count <= 512u) {
        stats.bucket_65_512.fetch_add(1u, std::memory_order_relaxed);
    } else if (triangle_count <= 4096u) {
        stats.bucket_513_4096.fetch_add(1u, std::memory_order_relaxed);
    } else {
        stats.bucket_gt_4096.fetch_add(1u, std::memory_order_relaxed);
    }
}

void record_mesh_profile_builder(
    hiprtBuildFlags flags,
    bool used_batch_builder,
    uint64_t count = 1u) noexcept {
    auto &stats = hip_mesh_scene_profile_stats();
    if (used_batch_builder) {
        stats.batch_built.fetch_add(count, std::memory_order_relaxed);
    } else if ((flags & 3u) == hiprtBuildFlagBitPreferBalancedBuild) {
        stats.ploc_built.fetch_add(count, std::memory_order_relaxed);
    } else if ((flags & 3u) == hiprtBuildFlagBitPreferHighQualityBuild) {
        stats.sbvh_built.fetch_add(count, std::memory_order_relaxed);
    } else {
        stats.lbvh_built.fetch_add(count, std::memory_order_relaxed);
    }
}

struct ValidatedMeshBuildInput {
    hipDeviceptr_t new_vertex_buffer{};
    size_t vertex_size{};
    size_t vertex_stride{};
    hipDeviceptr_t new_triangle_buffer{};
    size_t triangle_size{};
    size_t vertex_count{};
    size_t triangle_count{};
    size_t motion_keyframe_count{};
    size_t vertices_per_keyframe{};
};

[[nodiscard]] ValidatedMeshBuildInput validate_mesh_build_command(
    const MeshBuildCommand *command,
    const AccelOption &option) noexcept {
    auto vertex_buffer = reinterpret_cast<const HIPBuffer *>(command->vertex_buffer());
    auto triangle_buffer = reinterpret_cast<const HIPBuffer *>(command->triangle_buffer());
    auto vertex_offset = command->vertex_buffer_offset();
    auto vertex_size = command->vertex_buffer_size();
    auto vertex_stride = command->vertex_stride();
    auto triangle_offset = command->triangle_buffer_offset();
    auto triangle_size = command->triangle_buffer_size();
    LUISA_ASSERT(vertex_offset <= vertex_buffer->size_bytes() &&
                     vertex_size <= vertex_buffer->size_bytes() - vertex_offset,
                 "Vertex buffer offset + size exceeds buffer size {}.", vertex_buffer->size_bytes());
    LUISA_ASSERT(triangle_offset <= triangle_buffer->size_bytes() &&
                     triangle_size <= triangle_buffer->size_bytes() - triangle_offset,
                 "Triangle buffer offset + size exceeds buffer size {}.", triangle_buffer->size_bytes());
    constexpr auto position_size = sizeof(float) * 3u;
    LUISA_ASSERT(vertex_stride >= position_size &&
                     vertex_stride <= std::numeric_limits<uint32_t>::max(),
                 "Invalid HIP mesh vertex stride {}.", vertex_stride);
    LUISA_ASSERT(vertex_size % vertex_stride == 0u,
                 "HIP mesh vertex buffer size {} is not divisible by stride {}.",
                 vertex_size, vertex_stride);
    LUISA_ASSERT(triangle_size % sizeof(Triangle) == 0u,
                 "HIP mesh triangle buffer size {} is not divisible by {}.",
                 triangle_size, sizeof(Triangle));
    if (!option.motion.is_enabled()) {
        LUISA_ASSERT(vertex_offset % alignof(float) == 0u &&
                         vertex_stride % alignof(float) == 0u,
                     "Static HIP mesh vertex offset {} and stride {} must be "
                     "{}-byte aligned for HIPRT.",
                     vertex_offset, vertex_stride, alignof(float));
        LUISA_ASSERT(triangle_offset % alignof(Triangle) == 0u,
                     "Static HIP mesh triangle offset {} must be {}-byte aligned "
                     "for HIPRT.",
                     triangle_offset, alignof(Triangle));
    }

    auto vertex_count = vertex_size / vertex_stride;
    auto triangle_count = triangle_size / sizeof(Triangle);
    auto motion_keyframe_count = option.motion.is_enabled() ?
                                     static_cast<size_t>(option.motion.keyframe_count) :
                                     1u;
    LUISA_ASSERT(vertex_count % motion_keyframe_count == 0u,
                 "HIP motion mesh vertex count {} is not divisible by its {} keyframes.",
                 vertex_count, motion_keyframe_count);
    auto vertices_per_keyframe = vertex_count / motion_keyframe_count;
    LUISA_ASSERT(vertices_per_keyframe >= 3u &&
                     vertices_per_keyframe <= std::numeric_limits<uint32_t>::max() &&
                     triangle_count > 0u &&
                     triangle_count <= std::numeric_limits<uint32_t>::max(),
                 "HIP meshes require at least three 32-bit-addressable vertices per "
                 "keyframe and a nonzero 32-bit triangle count.");
    return ValidatedMeshBuildInput{
        .new_vertex_buffer = static_cast<std::byte *>(vertex_buffer->handle()) + vertex_offset,
        .vertex_size = vertex_size,
        .vertex_stride = vertex_stride,
        .new_triangle_buffer = static_cast<std::byte *>(triangle_buffer->handle()) + triangle_offset,
        .triangle_size = triangle_size,
        .vertex_count = vertex_count,
        .triangle_count = triangle_count,
        .motion_keyframe_count = motion_keyframe_count,
        .vertices_per_keyframe = vertices_per_keyframe};
}

[[nodiscard]] hiprtGeometryBuildInput make_static_triangle_build_input(
    const ValidatedMeshBuildInput &validated) noexcept {
    hiprtTriangleMeshPrimitive mesh_prim{};
    mesh_prim.vertices = reinterpret_cast<hiprtDevicePtr>(validated.new_vertex_buffer);
    mesh_prim.vertexCount = static_cast<uint32_t>(validated.vertex_count);
    mesh_prim.vertexStride = static_cast<uint32_t>(validated.vertex_stride);
    mesh_prim.triangleIndices = reinterpret_cast<hiprtDevicePtr>(validated.new_triangle_buffer);
    mesh_prim.triangleCount = static_cast<uint32_t>(validated.triangle_count);
    mesh_prim.triangleStride = sizeof(Triangle);

    hiprtGeometryBuildInput build_input{};
    build_input.type = hiprtPrimitiveTypeTriangleMesh;
    build_input.geomType = 0u;
    build_input.primitive.triangleMesh = mesh_prim;
    return build_input;
}

[[nodiscard]] hiprtBuildFlags select_static_triangle_build_flags(
    hiprtContext ctx,
    const hiprtGeometryBuildInput &build_input,
    const AccelOption &option,
    size_t triangle_count,
    bool use_batch_builder,
    size_t &cached_free_memory_size) noexcept {
    if (option.hint == AccelOption::UsageHint::FAST_BUILD || use_batch_builder) {
        return make_hiprt_build_flags(option, hiprtBuildFlagBitPreferFastBuild);
    }
    if (triangle_count < hip_sbvh_min_tris()) {
        return make_hiprt_build_flags(option, hiprtBuildFlagBitPreferBalancedBuild);
    }
    const auto hq_flags = make_hiprt_build_flags(
        option, hiprtBuildFlagBitPreferHighQualityBuild);
    hiprtBuildOptions hq_options{};
    hq_options.buildFlags = hq_flags;
    size_t hq_temp_size = 0u;
    LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(
        ctx, build_input, hq_options, hq_temp_size));
    if (cached_free_memory_size == std::numeric_limits<size_t>::max()) {
        size_t total_memory_size = 0u;
        LUISA_CHECK_HIP(hipMemGetInfo(&cached_free_memory_size, &total_memory_size));
        static_cast<void>(total_memory_size);
    }
    if (hiprt_high_quality_scratch_exceeds_budget(hq_temp_size, cached_free_memory_size)) {
        return make_hiprt_build_flags(option, hiprtBuildFlagBitPreferBalancedBuild);
    }
    return hq_flags;
}

class HIPMotionMeshValidationCallback final : public HIPCallbackContext {

private:
    HIPStageBufferPool::View *_host_status;
    uint32_t _vertices_per_keyframe;

    [[nodiscard]] static auto &_pool() noexcept {
        static Pool<HIPMotionMeshValidationCallback, true> pool;
        return pool;
    }

public:
    HIPMotionMeshValidationCallback(
        HIPStageBufferPool::View *host_status,
        uint32_t vertices_per_keyframe) noexcept
        : _host_status{host_status},
          _vertices_per_keyframe{vertices_per_keyframe} {}

    [[nodiscard]] static auto create(
        HIPStageBufferPool::View *host_status,
        uint32_t vertices_per_keyframe) noexcept {
        return _pool().create(host_status, vertices_per_keyframe);
    }

    void recycle() noexcept override {
        HIPMotionMeshValidationStatus status{};
        std::memcpy(&status, _host_status->address(), sizeof(status));
        auto vertices_per_keyframe = _vertices_per_keyframe;
        _host_status->recycle();
        _pool().destroy(this);
        if (status.valid()) { return; }
        switch (status.error()) {
            case HIPMotionMeshValidationError::NON_FINITE_POSITION: {
                auto flat_vertex = status.element_index();
                auto keyframe = flat_vertex / vertices_per_keyframe;
                auto local_vertex = flat_vertex % vertices_per_keyframe;
                LUISA_ERROR_WITH_LOCATION(
                    "HIP motion mesh keyframe {} vertex {} has a non-finite "
                    "position component at axis {}.",
                    keyframe, local_vertex, status.component());
            }
            case HIPMotionMeshValidationError::INVALID_TRIANGLE_INDEX:
                LUISA_ERROR_WITH_LOCATION(
                    "HIP motion mesh triangle {} has an invalid local vertex "
                    "index at corner {} (each keyframe has {} vertices).",
                    status.element_index(), status.component(),
                    vertices_per_keyframe);
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "HIP motion mesh preprocessing reported unknown validation "
                    "error {} (element = {}, component = {}).",
                    luisa::to_underlying(status.error()),
                    status.element_index(), status.component());
        }
    }
};

}// namespace

HIPMesh::HIPMesh(hiprtContext ctx, const AccelOption &option) noexcept
    : _option{option}, _hiprt_ctx{ctx} {
    if (_option.motion.is_enabled()) {
        auto time_span = _option.motion.time_end - _option.motion.time_start;
        LUISA_ASSERT(std::isfinite(_option.motion.time_start) &&
                         std::isfinite(_option.motion.time_end) &&
                         _option.motion.time_start < _option.motion.time_end &&
                         std::isfinite(time_span),
                     "HIP motion mesh time range must be finite and strictly increasing "
                     "with a finite representable span (got [{}, {}]).",
                     _option.motion.time_start, _option.motion.time_end);
    }
}

void HIPMesh::_destroy_geometry_no_lock(hiprtContext ctx) noexcept {
    if (_geometry != nullptr) {
        if (_owns_direct_device_geom) {
            LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_geometry)));
        } else {
            LUISA_CHECK_HIPRT(hiprtDestroyGeometry(ctx != nullptr ? ctx : _hiprt_ctx, _geometry));
        }
        _geometry = nullptr;
        _owns_direct_device_geom = false;
    }
}

HIPMesh::~HIPMesh() noexcept {
    if (_geometry || _motion_vertex_buffer || _motion_triangle_buffer ||
        _motion_aabb_buffer || _motion_device_data) {
        // Geometry builds and traces are asynchronous, while HIPRT destruction
        // and the backing-allocation frees have no stream on which to order.
        LUISA_CHECK_HIP(hipDeviceSynchronize());
    }
    _destroy_geometry_no_lock(_hiprt_ctx);
    if (_motion_vertex_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_motion_vertex_buffer)));
    }
    if (_motion_triangle_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_motion_triangle_buffer)));
    }
    if (_motion_aabb_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_motion_aabb_buffer)));
    }
    if (_motion_device_data) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_motion_device_data)));
    }
}

void HIPMesh::build(HIPCommandEncoder &encoder, MeshBuildCommand *command) noexcept {
    const auto validated = validate_mesh_build_command(command, _option);
    std::scoped_lock lock{_mutex};

    const auto requires_build =
        _geometry == nullptr ||
        !_option.allow_update ||
        command->request() == AccelBuildRequest::FORCE_BUILD ||
        validated.new_vertex_buffer != _vertex_buffer ||
        validated.vertex_size != _vertex_buffer_size ||
        validated.vertex_stride != _vertex_stride ||
        validated.new_triangle_buffer != _triangle_buffer ||
        validated.triangle_size != _triangle_buffer_size;
    // HIPRT storage is sized by primitive count. Compacted allocations have no
    // spare capacity for another full build and must always be recreated.
    const auto old_triangle_count = _triangle_buffer_size / sizeof(Triangle);
    auto recreate_geometry =
        _geometry == nullptr || validated.triangle_count != old_triangle_count ||
        (_option.allow_compaction && requires_build);

    _vertex_buffer = validated.new_vertex_buffer;
    _vertex_buffer_size = validated.vertex_size;
    _vertex_stride = validated.vertex_stride;
    _triangle_buffer = validated.new_triangle_buffer;
    _triangle_buffer_size = validated.triangle_size;

    const auto vertex_count = validated.vertex_count;
    const auto triangle_count = validated.triangle_count;
    const auto motion_keyframe_count = validated.motion_keyframe_count;
    const auto vertices_per_keyframe = validated.vertices_per_keyframe;

    if (_option.motion.is_enabled()) {
        auto hip_stream = encoder.stream()->handle();

        // HIPRT has no native deforming-triangle primitive. Build one custom
        // AABB leaf per triangle around the union of every keyframe instead.
        // GPU preprocessing snapshots the exact positions and indices used by
        // custom intersection into mesh-owned allocations, while computing the
        // swept bounds from that same snapshot. This preserves source-resource
        // independence without a bulk readback or host synchronization.
        constexpr auto packed_position_size = sizeof(float) * 3u;
        static_assert(sizeof(AABB) == sizeof(float) * 6u);
        auto required_vertex_size = vertex_count * packed_position_size;
        if (_motion_vertex_buffer_size < required_vertex_size) {
            if (_motion_vertex_buffer) {
                LUISA_CHECK_HIP(hipFreeAsync(
                    reinterpret_cast<void *>(_motion_vertex_buffer), hip_stream));
            }
            LUISA_CHECK_HIP(hipMallocAsync(
                reinterpret_cast<void **>(&_motion_vertex_buffer),
                required_vertex_size, hip_stream));
            _motion_vertex_buffer_size = required_vertex_size;
        }
        auto required_triangle_size = triangle_count * sizeof(Triangle);
        if (_motion_triangle_buffer_size < required_triangle_size) {
            if (_motion_triangle_buffer) {
                LUISA_CHECK_HIP(hipFreeAsync(
                    reinterpret_cast<void *>(_motion_triangle_buffer), hip_stream));
            }
            LUISA_CHECK_HIP(hipMallocAsync(
                reinterpret_cast<void **>(&_motion_triangle_buffer),
                required_triangle_size, hip_stream));
            _motion_triangle_buffer_size = required_triangle_size;
        }
        auto required_aabb_size = triangle_count * sizeof(AABB);
        if (_motion_aabb_buffer_size < required_aabb_size) {
            if (_motion_aabb_buffer) {
                LUISA_CHECK_HIP(hipFreeAsync(
                    reinterpret_cast<void *>(_motion_aabb_buffer), hip_stream));
            }
            LUISA_CHECK_HIP(hipMallocAsync(
                reinterpret_cast<void **>(&_motion_aabb_buffer),
                required_aabb_size, hip_stream));
            _motion_aabb_buffer_size = required_aabb_size;
        }
        if (!_motion_device_data) {
            LUISA_CHECK_HIP(hipMallocAsync(
                reinterpret_cast<void **>(&_motion_device_data),
                sizeof(HIPMotionTriangleDeviceData), hip_stream));
        }

        hipDeviceptr_t validation_status{};
        LUISA_CHECK_HIP(hipMallocAsync(
            reinterpret_cast<void **>(&validation_status),
            sizeof(HIPMotionMeshValidationStatus), hip_stream));
        auto &preprocess = encoder.stream()->device()->motion_mesh_builtin();
        preprocess.reset_validation_status(hip_stream, validation_status);
        preprocess.pack_motion_vertices(
            hip_stream, _vertex_buffer, _motion_vertex_buffer,
            vertex_count, static_cast<uint32_t>(_vertex_stride),
            validation_status);
        preprocess.build_motion_triangle_aabbs(
            hip_stream, _triangle_buffer, _motion_vertex_buffer,
            _motion_triangle_buffer, _motion_aabb_buffer,
            static_cast<uint32_t>(triangle_count),
            static_cast<uint32_t>(vertices_per_keyframe),
            static_cast<uint32_t>(motion_keyframe_count),
            validation_status);

        uint32_t motion_flags = 0u;
        if (_option.motion.should_vanish_start) { motion_flags |= 1u << 0u; }
        if (_option.motion.should_vanish_end) { motion_flags |= 1u << 1u; }
        HIPMotionTriangleDeviceData device_data{
            .vertices = reinterpret_cast<uint64_t>(_motion_vertex_buffer),
            .triangles = reinterpret_cast<uint64_t>(_motion_triangle_buffer),
            .vertex_stride = packed_position_size,
            .vertices_per_keyframe = static_cast<uint32_t>(vertices_per_keyframe),
            .keyframe_count = static_cast<uint32_t>(motion_keyframe_count),
            .flags = motion_flags,
            .time_start = _option.motion.time_start,
            .time_end = _option.motion.time_end};

        encoder.with_upload_buffer(sizeof(device_data), [&](auto upload_buffer) noexcept {
            auto upload = upload_buffer->address();
            std::memcpy(upload, &device_data, sizeof(device_data));
            LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
                _motion_device_data, upload,
                sizeof(device_data), hip_stream));
        });

        hiprtAABBListPrimitive aabb_primitive{};
        aabb_primitive.aabbs = reinterpret_cast<hiprtDevicePtr>(
            _motion_aabb_buffer);
        aabb_primitive.aabbCount = static_cast<uint32_t>(triangle_count);
        aabb_primitive.aabbStride = sizeof(AABB);

        hiprtGeometryBuildInput build_input{};
        build_input.type = hiprtPrimitiveTypeAABBList;
        build_input.geomType = 0u;
        build_input.primitive.aabbList = aabb_primitive;

        if (requires_build) {
            auto policy = select_hiprt_geometry_build_policy(
                _hiprt_ctx, build_input, _option);
            recreate_geometry = recreate_geometry ||
                                (_geometry &&
                                 policy.options.buildFlags != _build_flags);
            if (recreate_geometry) {
                if (_geometry) {
                    // Geometry destruction has no stream parameter. Order it
                    // after all prior uses and the preprocessing above.
                    LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));
                    LUISA_CHECK_HIPRT(hiprtDestroyGeometry(
                        _hiprt_ctx, _geometry));
                    _geometry = nullptr;
                }
                LUISA_CHECK_HIPRT(hiprtCreateGeometry(
                    _hiprt_ctx, build_input, policy.options, _geometry));
                _build_flags = policy.options.buildFlags;
            }
            auto temporary_buffer =
                encoder.stream()->rt_scratch_buffer(
                    policy.temporary_buffer_size);
            LUISA_CHECK_HIPRT(hiprtBuildGeometry(
                _hiprt_ctx, hiprtBuildOperationBuild,
                build_input, policy.options, temporary_buffer,
                hip_stream, _geometry));
            if (_option.allow_compaction) {
                LUISA_CHECK_HIPRT(hiprtCompactGeometry(
                    _hiprt_ctx, hip_stream, _geometry, _geometry));
            }
        } else {
            hiprtBuildOptions build_options{};
            build_options.buildFlags = _build_flags;
            LUISA_CHECK_HIPRT(hiprtBuildGeometry(
                _hiprt_ctx, hiprtBuildOperationUpdate,
                build_input, build_options, nullptr,
                hip_stream, _geometry));
        }

        // Report deterministic device-side validation after all preceding
        // stream work has completed. Invalid inputs were sanitized by the
        // kernels, so the asynchronous HIPRT build remains memory-safe.
        auto host_status = encoder.stream()->download_pool()->allocate(
            sizeof(HIPMotionMeshValidationStatus));
        LUISA_ASSERT(host_status != nullptr,
                     "Failed to allocate HIP motion-mesh validation staging memory.");
        LUISA_CHECK_HIP(hipMemcpyDtoHAsync(
            host_status->address(), validation_status,
            sizeof(HIPMotionMeshValidationStatus), hip_stream));
        LUISA_CHECK_HIP(hipFreeAsync(
            reinterpret_cast<void *>(validation_status), hip_stream));
        encoder.add_callback(HIPMotionMeshValidationCallback::create(
            host_status, static_cast<uint32_t>(vertices_per_keyframe)));
        return;
    }

    const auto build_input = make_static_triangle_build_input(validated);
    auto hip_stream = encoder.stream()->handle();

    if (requires_build) {
        const auto profile = hip_scene_profile_enabled();
        using ProfileClock = std::chrono::steady_clock;
        auto t_start = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (profile) {
            record_mesh_profile_buckets(triangle_count);
        }
        hiprtBuildOptions build_options{};
        size_t temporary_buffer_size = 0u;
        if (triangle_count >= hip_sbvh_min_tris()) {
            auto policy = select_hiprt_geometry_build_policy(
                _hiprt_ctx, build_input, _option);
            build_options = policy.options;
            temporary_buffer_size = policy.temporary_buffer_size;
        } else {
            auto free_mem = std::numeric_limits<size_t>::max();
            build_options.buildFlags = select_static_triangle_build_flags(
                _hiprt_ctx, build_input, _option, triangle_count, false, free_mem);
            LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(
                _hiprt_ctx, build_input, build_options, temporary_buffer_size));
        }
        if (profile) {
            record_mesh_profile_builder(build_options.buildFlags, false, 1u);
        }
        auto t_after_policy = profile ? ProfileClock::now() : ProfileClock::time_point{};
        recreate_geometry = recreate_geometry ||
                            (_geometry &&
                             build_options.buildFlags != _build_flags);
        if (recreate_geometry) {
            if (profile) {
                hip_mesh_scene_profile_stats().recreate_count.fetch_add(1u, std::memory_order_relaxed);
            }
            if (_geometry) {
                LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));
                _destroy_geometry_no_lock(_hiprt_ctx);
            }
            LUISA_CHECK_HIPRT(hiprtCreateGeometry(_hiprt_ctx, build_input,
                                                  build_options, _geometry));
            _owns_direct_device_geom = false;
            _build_flags = build_options.buildFlags;
        }
        auto t_after_create = profile ? ProfileClock::now() : ProfileClock::time_point{};

        auto temp_buffer =
            encoder.stream()->rt_scratch_buffer(temporary_buffer_size);

        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationBuild,
                                             build_input, build_options,
                                             temp_buffer, hip_stream, _geometry));
        auto t_after_build = profile ? ProfileClock::now() : ProfileClock::time_point{};

        if (_option.allow_compaction) {
            if (profile) {
                hip_mesh_scene_profile_stats().compacted_count.fetch_add(1u, std::memory_order_relaxed);
            }
            LUISA_CHECK_HIPRT(hiprtCompactGeometry(
                _hiprt_ctx, hip_stream, _geometry, _geometry));
        }
        auto t_after_compact = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (profile) {
            auto &stats = hip_mesh_scene_profile_stats();
            auto delta_ns = [](auto a, auto b) noexcept {
                return static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
            };
            stats.policy_ns.fetch_add(delta_ns(t_start, t_after_policy), std::memory_order_relaxed);
            stats.create_ns.fetch_add(delta_ns(t_after_policy, t_after_create), std::memory_order_relaxed);
            stats.build_ns.fetch_add(delta_ns(t_after_create, t_after_build), std::memory_order_relaxed);
            stats.compact_ns.fetch_add(delta_ns(t_after_build, t_after_compact), std::memory_order_relaxed);
            stats.total_ns.fetch_add(delta_ns(t_start, t_after_compact), std::memory_order_relaxed);
        }
        _is_compacted = _option.allow_compaction;
    } else {
        hiprtBuildOptions build_options{};
        build_options.buildFlags = _build_flags;
        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationUpdate,
                                             build_input, build_options,
                                             nullptr, hip_stream, _geometry));
    }
    _built = true;
    _last_triangle_count = triangle_count;
}

void HIPMesh::build_batch(
    HIPCommandEncoder &encoder,
    luisa::span<MeshBuildCommand *const> commands) noexcept {
    if (commands.empty()) {
        return;
    }

    struct BatchItem {
        HIPMesh *mesh{};
        MeshBuildCommand *command{};
        ValidatedMeshBuildInput validated{};
        hiprtGeometryBuildInput build_input{};
        hiprtContext ctx{nullptr};
        hiprtBuildFlags build_flags{};
        hiprtGeometry existing_geometry{nullptr};
        hiprtBuildFlags existing_build_flags{};
        bool use_batch_builder{false};
        bool allow_compaction{false};
        bool recreate_geometry{false};
        bool destroy_existing{false};
    };

    const auto hip_stream = encoder.stream()->handle();
    const auto batch_max_prims = hip_batch_build_max_prims();
    const auto profile = hip_scene_profile_enabled();
    using ProfileClock = std::chrono::steady_clock;

    auto flush_rebuild_items = [&](std::vector<BatchItem> &items) noexcept {
        if (items.empty()) {
            return;
        }
        if (items.size() == 1u) {
            items[0].mesh->build(encoder, items[0].command);
            items.clear();
            return;
        }

        const auto t_start = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (profile) {
            hip_mesh_scene_profile_stats().batch_calls.fetch_add(1u, std::memory_order_relaxed);
        }

        size_t cached_free_memory_size = std::numeric_limits<size_t>::max();
        bool any_existing_to_destroy = false;
        for (auto &item : items) {
            const auto tri_count = item.validated.triangle_count;
            if (profile) {
                record_mesh_profile_buckets(tri_count);
            }
            item.build_input = make_static_triangle_build_input(item.validated);
            item.use_batch_builder =
                !item.mesh->_option.allow_update &&
                batch_max_prims > 0u &&
                tri_count <= batch_max_prims;
            item.build_flags = select_static_triangle_build_flags(
                item.ctx, item.build_input, item.mesh->_option,
                tri_count, item.use_batch_builder, cached_free_memory_size);
            item.allow_compaction = item.mesh->_option.allow_compaction;
            item.recreate_geometry =
                item.recreate_geometry ||
                (item.existing_geometry != nullptr &&
                 item.build_flags != item.existing_build_flags);
            item.destroy_existing =
                item.existing_geometry != nullptr && item.recreate_geometry;
            any_existing_to_destroy = any_existing_to_destroy || item.destroy_existing;
        }
        const auto t_after_policy = profile ? ProfileClock::now() : ProfileClock::time_point{};

        if (any_existing_to_destroy) {
            LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));
            for (auto &item : items) {
                if (item.destroy_existing) {
                    std::scoped_lock lock{item.mesh->_mutex};
                    item.mesh->_destroy_geometry_no_lock(item.ctx);
                    item.existing_geometry = nullptr;
                }
            }
        }

        // Partition items into homogeneous groups matching (ctx, use_batch_builder, build_flags, allow_compaction)
        // so interleaved small and large meshes coalesce into minimum kernel launches and compaction pools.
        std::vector<size_t> order(items.size());
        for (size_t i = 0u; i < items.size(); ++i) {
            order[i] = i;
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t lhs_idx, size_t rhs_idx) noexcept {
            const auto &a = items[lhs_idx];
            const auto &b = items[rhs_idx];
            if (a.ctx != b.ctx) { return a.ctx < b.ctx; }
            if (a.use_batch_builder != b.use_batch_builder) {
                return a.use_batch_builder > b.use_batch_builder;
            }
            if (a.build_flags != b.build_flags) { return a.build_flags < b.build_flags; }
            return a.allow_compaction < b.allow_compaction;
        });

        constexpr size_t max_chunk_size = 256u;
        uint64_t total_create_ns = 0u;
        uint64_t total_build_ns = 0u;
        uint64_t total_compact_ns = 0u;
        auto delta_ns = [](auto a, auto b) noexcept {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
        };

        std::vector<hiprtGeometryBuildInput> chunk_inputs;
        std::vector<hiprtGeometryBuildInput> create_inputs;
        std::vector<hiprtGeometry> create_geoms;
        std::vector<hiprtGeometry *> create_geom_ptrs;
        std::vector<hiprtGeometry> chunk_geoms;
        std::vector<hiprtGeometry> compacted_geoms;
        std::vector<hiprtGeometry *> compacted_ptrs;

        for (size_t group_begin = 0u; group_begin < order.size();) {
            const auto &key = items[order[group_begin]];
            size_t group_end = group_begin + 1u;
            while (group_end < order.size()) {
                const auto &cand = items[order[group_end]];
                if (cand.ctx != key.ctx ||
                    cand.use_batch_builder != key.use_batch_builder ||
                    cand.build_flags != key.build_flags ||
                    cand.allow_compaction != key.allow_compaction) {
                    break;
                }
                ++group_end;
            }

            for (size_t chunk_begin = group_begin; chunk_begin < group_end;
                 chunk_begin += max_chunk_size) {
                const auto chunk_end = std::min(chunk_begin + max_chunk_size, group_end);
                const auto chunk_n = static_cast<uint32_t>(chunk_end - chunk_begin);

                chunk_inputs.resize(chunk_n);
                create_inputs.clear();
                size_t max_tris = 0u;
                for (uint32_t k = 0u; k < chunk_n; ++k) {
                    const auto &item = items[order[chunk_begin + k]];
                    chunk_inputs[k] = item.build_input;
                    max_tris = std::max(max_tris, item.validated.triangle_count);
                    if (item.recreate_geometry) {
                        create_inputs.push_back(item.build_input);
                    }
                }

                hiprtBuildOptions opts{};
                opts.buildFlags = key.build_flags;
                if (key.use_batch_builder) {
                    const auto clamped = std::clamp<uint32_t>(
                        static_cast<uint32_t>(std::bit_ceil(std::max<size_t>(max_tris, 64u))),
                        64u, static_cast<uint32_t>(hiprtMaxBatchBuildMaxPrimCount));
                    opts.batchBuildMaxPrimCount = clamped;
                }

                const auto tc0 = profile ? ProfileClock::now() : ProfileClock::time_point{};
                chunk_geoms.resize(chunk_n);
                if (!create_inputs.empty()) {
                    const auto create_n = static_cast<uint32_t>(create_inputs.size());
                    create_geoms.resize(create_n);
                    create_geom_ptrs.resize(create_n);
                    for (uint32_t k = 0u; k < create_n; ++k) {
                        create_geom_ptrs[k] = &create_geoms[k];
                    }
                    LUISA_CHECK_HIPRT(hiprtCreateGeometries(
                        key.ctx, create_n, create_inputs.data(), opts, create_geom_ptrs.data()));
                    uint32_t created_idx = 0u;
                    for (uint32_t k = 0u; k < chunk_n; ++k) {
                        const auto &item = items[order[chunk_begin + k]];
                        chunk_geoms[k] = item.recreate_geometry ?
                                             create_geoms[created_idx++] :
                                             item.existing_geometry;
                    }
                } else {
                    for (uint32_t k = 0u; k < chunk_n; ++k) {
                        chunk_geoms[k] = items[order[chunk_begin + k]].existing_geometry;
                    }
                }
                const auto tc1 = profile ? ProfileClock::now() : ProfileClock::time_point{};
                if (profile) {
                    total_create_ns += delta_ns(tc0, tc1);
                    if (!create_inputs.empty()) {
                        hip_mesh_scene_profile_stats().recreate_count.fetch_add(
                            create_inputs.size(), std::memory_order_relaxed);
                    }
                }

                size_t temp_size = 0u;
                LUISA_CHECK_HIPRT(hiprtGetGeometriesBuildTemporaryBufferSize(
                    key.ctx, chunk_n, chunk_inputs.data(), opts, temp_size));
                auto temp_buffer = encoder.stream()->rt_scratch_buffer(temp_size);
                LUISA_CHECK_HIPRT(hiprtBuildGeometries(
                    key.ctx, hiprtBuildOperationBuild, chunk_n,
                    chunk_inputs.data(), opts, temp_buffer,
                    hip_stream, chunk_geoms.data()));
                const auto tb1 = profile ? ProfileClock::now() : ProfileClock::time_point{};
                if (profile) {
                    total_build_ns += delta_ns(tc1, tb1);
                    record_mesh_profile_builder(opts.buildFlags, key.use_batch_builder, chunk_n);
                }

                if (key.allow_compaction) {
                    compacted_geoms.resize(chunk_n);
                    compacted_ptrs.resize(chunk_n);
                    for (uint32_t k = 0u; k < chunk_n; ++k) {
                        compacted_ptrs[k] = &compacted_geoms[k];
                    }
                    LUISA_CHECK_HIPRT(hiprtCompactGeometries(
                        key.ctx, chunk_n, hip_stream,
                        chunk_geoms.data(), compacted_ptrs.data()));
                    chunk_geoms.swap(compacted_geoms);
                    if (profile) {
                        hip_mesh_scene_profile_stats().compacted_count.fetch_add(
                            chunk_n, std::memory_order_relaxed);
                    }
                }
                const auto tcmp1 = profile ? ProfileClock::now() : ProfileClock::time_point{};
                if (profile) {
                    total_compact_ns += delta_ns(tb1, tcmp1);
                }

                for (uint32_t k = 0u; k < chunk_n; ++k) {
                    auto &item = items[order[chunk_begin + k]];
                    auto *mesh = item.mesh;
                    std::scoped_lock lock{mesh->_mutex};
                    mesh->_geometry = chunk_geoms[k];
                    if (item.recreate_geometry) {
                        mesh->_owns_direct_device_geom = false;
                    }
                    mesh->_build_flags = opts.buildFlags;
                    mesh->_vertex_buffer = item.validated.new_vertex_buffer;
                    mesh->_vertex_buffer_size = item.validated.vertex_size;
                    mesh->_vertex_stride = item.validated.vertex_stride;
                    mesh->_triangle_buffer = item.validated.new_triangle_buffer;
                    mesh->_triangle_buffer_size = item.validated.triangle_size;
                    mesh->_is_compacted = key.allow_compaction;
                    mesh->_built = true;
                    mesh->_last_triangle_count = item.validated.triangle_count;
                }
            }
            group_begin = group_end;
        }

        if (profile) {
            const auto t_end = ProfileClock::now();
            auto &stats = hip_mesh_scene_profile_stats();
            stats.policy_ns.fetch_add(delta_ns(t_start, t_after_policy), std::memory_order_relaxed);
            stats.create_ns.fetch_add(total_create_ns, std::memory_order_relaxed);
            stats.build_ns.fetch_add(total_build_ns, std::memory_order_relaxed);
            stats.compact_ns.fetch_add(total_compact_ns, std::memory_order_relaxed);
            stats.total_ns.fetch_add(delta_ns(t_start, t_end), std::memory_order_relaxed);
        }

        items.clear();
    };

    std::vector<BatchItem> rebuild_items;
    rebuild_items.reserve(commands.size());
    std::vector<const HIPMesh *> seen_meshes;
    seen_meshes.reserve(commands.size());

    for (auto *command : commands) {
        auto *mesh = reinterpret_cast<HIPMesh *>(command->handle());
        LUISA_ASSERT(!mesh->_option.motion.is_enabled(),
                     "HIPMesh::build_batch does not accept motion meshes.");
        if (std::find(seen_meshes.begin(), seen_meshes.end(), mesh) != seen_meshes.end()) {
            flush_rebuild_items(rebuild_items);
            seen_meshes.clear();
        }

        const auto validated = validate_mesh_build_command(command, mesh->_option);
        bool requires_build = false;
        bool recreate_geometry = false;
        hiprtContext ctx{nullptr};
        hiprtGeometry existing_geom{nullptr};
        hiprtBuildFlags existing_flags{};
        {
            std::scoped_lock lock{mesh->_mutex};
            requires_build =
                mesh->_geometry == nullptr ||
                !mesh->_option.allow_update ||
                command->request() == AccelBuildRequest::FORCE_BUILD ||
                validated.new_vertex_buffer != mesh->_vertex_buffer ||
                validated.vertex_size != mesh->_vertex_buffer_size ||
                validated.vertex_stride != mesh->_vertex_stride ||
                validated.new_triangle_buffer != mesh->_triangle_buffer ||
                validated.triangle_size != mesh->_triangle_buffer_size;
            const auto old_triangle_count = mesh->_triangle_buffer_size / sizeof(Triangle);
            recreate_geometry =
                mesh->_geometry == nullptr ||
                validated.triangle_count != old_triangle_count ||
                (mesh->_option.allow_compaction && requires_build);
            ctx = mesh->_hiprt_ctx;
            existing_geom = mesh->_geometry;
            existing_flags = mesh->_build_flags;
        }

        if (!requires_build) {
            flush_rebuild_items(rebuild_items);
            seen_meshes.clear();
            mesh->build(encoder, command);
            continue;
        }

        seen_meshes.push_back(mesh);
        rebuild_items.push_back(BatchItem{
            .mesh = mesh,
            .command = command,
            .validated = validated,
            .ctx = ctx,
            .existing_geometry = existing_geom,
            .existing_build_flags = existing_flags,
            .recreate_geometry = recreate_geometry});
    }

    flush_rebuild_items(rebuild_items);
}

void HIPMesh::replace_geometry_from_archive(
    hiprtContext ctx,
    hipStream_t stream,
    hiprtGeometry new_geometry,
    uint64_t triangle_count,
    bool is_compacted,
    bool owns_direct_device_geom) noexcept {
    std::scoped_lock lock{_mutex};
    if (_geometry != nullptr) {
        if (stream != nullptr) {
            LUISA_CHECK_HIP(hipStreamSynchronize(stream));
        }
        _destroy_geometry_no_lock(ctx);
    }
    _geometry = new_geometry;
    _owns_direct_device_geom = owns_direct_device_geom && (new_geometry != nullptr);
    _build_flags = make_hiprt_build_flags(_option);
    _vertex_buffer = {};
    _vertex_buffer_size = 0u;
    _vertex_stride = 0u;
    _triangle_buffer = {};
    _triangle_buffer_size = 0u;
    _built = (new_geometry != nullptr);
    _is_compacted = is_compacted;
    _last_triangle_count = triangle_count;
}

}// namespace luisa::compute::hip
