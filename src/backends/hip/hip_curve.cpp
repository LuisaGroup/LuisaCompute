//
// HIPRT curve geometry implemented as custom AABB primitives.
//

#include <cstring>
#include <limits>

#include "hip_buffer.h"
#include "hip_check.h"
#include "hip_command_encoder.h"
#include "hip_curve.h"
#include "hip_curve_bounds.h"
#include "hip_stream.h"

namespace luisa::compute::hip {

HIPCurve::HIPCurve(hiprtContext ctx, const AccelOption &option) noexcept
    : _option{option}, _hiprt_ctx{ctx} {
    LUISA_ASSERT(!_option.motion.is_enabled(),
                 "HIP curves do not yet support motion keyframes.");
}

HIPCurve::~HIPCurve() noexcept {
    if (_geometry || _aabb_buffer || _device_data || _control_points || _segments) {
        // Geometry and all backing allocations can have been produced or used
        // asynchronously on any HIP stream. Destruction has no stream argument.
        LUISA_CHECK_HIP(hipDeviceSynchronize());
    }
    if (_geometry) { LUISA_CHECK_HIPRT(hiprtDestroyGeometry(_hiprt_ctx, _geometry)); }
    if (_aabb_buffer) { LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_aabb_buffer))); }
    if (_device_data) { LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_device_data))); }
    if (_control_points) { LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_control_points))); }
    if (_segments) { LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_segments))); }
}

void HIPCurve::build(HIPCommandEncoder &encoder, CurveBuildCommand *command) noexcept {
    auto cp_count = command->cp_count();
    auto cp_stride = command->cp_stride();
    auto seg_count = command->seg_count();
    auto basis = command->basis();
    auto cp_per_segment = segment_control_point_count(basis);
    LUISA_ASSERT(cp_per_segment != 0u, "Invalid curve basis 0x{:x}.", luisa::to_underlying(basis));
    LUISA_ASSERT(cp_stride >= sizeof(float4),
                 "Invalid control point stride {} (must be at least {}).",
                 cp_stride, sizeof(float4));
    LUISA_ASSERT(cp_count >= cp_per_segment,
                 "Curve has too few control points ({} for {}-point segments).",
                 cp_count, cp_per_segment);
    LUISA_ASSERT(seg_count > 0u, "Curve must contain at least one segment.");
    LUISA_ASSERT(cp_count <= std::numeric_limits<uint32_t>::max() &&
                     seg_count <= std::numeric_limits<uint32_t>::max() &&
                     cp_stride <= std::numeric_limits<uint32_t>::max(),
                 "HIP curves are limited to 32-bit counts and control-point strides.");

    auto cp_buffer = reinterpret_cast<const HIPBuffer *>(command->cp_buffer());
    auto seg_buffer = reinterpret_cast<const HIPBuffer *>(command->seg_buffer());
    auto cp_offset = command->cp_buffer_offset();
    auto seg_offset = command->seg_buffer_offset();
    LUISA_ASSERT(cp_offset <= cp_buffer->size_bytes() &&
                     cp_count <= (cp_buffer->size_bytes() - cp_offset) / cp_stride,
                 "Control point buffer out of range.");
    LUISA_ASSERT(seg_offset <= seg_buffer->size_bytes() &&
                     seg_count <= (seg_buffer->size_bytes() - seg_offset) / sizeof(uint32_t),
                 "Segment buffer out of range.");

    std::scoped_lock lock{_mutex};

    auto source_control_points = reinterpret_cast<hipDeviceptr_t>(
        static_cast<std::byte *>(cp_buffer->handle()) + cp_offset);
    auto source_segments = reinterpret_cast<hipDeviceptr_t>(
        static_cast<std::byte *>(seg_buffer->handle()) + seg_offset);
    auto requires_build =
        _geometry == nullptr ||
        !_option.allow_update ||
        command->request() == AccelBuildRequest::FORCE_BUILD ||
        source_control_points != _source_control_points ||
        source_segments != _source_segments ||
        cp_count != _control_point_count ||
        seg_count != _segment_count ||
        cp_stride != _control_point_stride ||
        basis != _basis;
    // A geometry allocation is sized for its primitive count. A compacted
    // allocation also has no spare capacity for a fresh full build.
    auto recreate_geometry =
        _geometry == nullptr || seg_count != _segment_count ||
        (_option.allow_compaction && requires_build);

    _source_control_points = source_control_points;
    _source_segments = source_segments;
    _control_point_count = cp_count;
    _segment_count = seg_count;
    _control_point_stride = cp_stride;
    _basis = basis;

    auto hip_stream = encoder.stream()->handle();

    // The custom-primitive BVH needs one conservative AABB per curve segment.
    // Buffer uploads preceding this build command are ordered on hip_stream;
    // synchronize once before the host-side bounds calculation so arbitrary
    // control-point strides and segment indices remain supported.
    luisa::vector<std::byte> host_cp(cp_count * cp_stride);
    luisa::vector<uint32_t> host_segments(seg_count);
    LUISA_CHECK_HIP(hipMemcpyDtoHAsync(host_cp.data(), source_control_points,
                                       host_cp.size(), hip_stream));
    LUISA_CHECK_HIP(hipMemcpyDtoHAsync(host_segments.data(), source_segments,
                                       host_segments.size() * sizeof(uint32_t), hip_stream));
    LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));

    auto load_cp = [&](size_t index) noexcept {
        float4 p{};
        std::memcpy(&p, host_cp.data() + index * cp_stride, sizeof(float4));
        return p;
    };

    luisa::vector<HIPCurveAABB> host_aabbs(seg_count);
    for (auto segment_index = 0u; segment_index < seg_count; segment_index++) {
        auto cp_begin = host_segments[segment_index];
        LUISA_ASSERT(static_cast<size_t>(cp_begin) + cp_per_segment <= cp_count,
                     "Curve segment {} starts at control point {}, but only {} control points exist.",
                     segment_index, cp_begin, cp_count);
        float4 cp[4]{};
        for (auto i = 0u; i < cp_per_segment; i++) { cp[i] = load_cp(cp_begin + i); }
        host_aabbs[segment_index] = compute_hip_curve_aabb(basis, cp);
    }

    auto required_cp_size = host_cp.size();
    auto required_segments_size = host_segments.size() * sizeof(uint32_t);
    auto required_aabb_size = host_aabbs.size() * sizeof(HIPCurveAABB);
    if (_control_points_size < required_cp_size) {
        if (_control_points) {
            LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(_control_points), hip_stream));
        }
        LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&_control_points),
                                       required_cp_size, hip_stream));
        _control_points_size = required_cp_size;
    }
    if (_segments_size < required_segments_size) {
        if (_segments) {
            LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(_segments), hip_stream));
        }
        LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&_segments),
                                       required_segments_size, hip_stream));
        _segments_size = required_segments_size;
    }
    if (_aabb_buffer_size < required_aabb_size) {
        if (_aabb_buffer) {
            LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(_aabb_buffer), hip_stream));
        }
        LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&_aabb_buffer),
                                       required_aabb_size, hip_stream));
        _aabb_buffer_size = required_aabb_size;
    }
    if (!_device_data) {
        LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&_device_data),
                                       sizeof(HIPCurveDeviceData), hip_stream));
    }
    HIPCurveDeviceData device_data{
        .control_points = reinterpret_cast<uint64_t>(_control_points),
        .segments = reinterpret_cast<uint64_t>(_segments),
        .control_point_stride = static_cast<uint32_t>(_control_point_stride),
        .basis = luisa::to_underlying(_basis),
    };

    LUISA_ASSERT(required_cp_size <= std::numeric_limits<size_t>::max() - required_segments_size &&
                     required_cp_size + required_segments_size <=
                         std::numeric_limits<size_t>::max() - required_aabb_size &&
                     required_cp_size + required_segments_size + required_aabb_size <=
                         std::numeric_limits<size_t>::max() - sizeof(device_data),
                 "HIP curve staging upload size overflow.");
    auto segments_upload_offset = required_cp_size;
    auto aabbs_upload_offset = segments_upload_offset + required_segments_size;
    auto descriptor_upload_offset = aabbs_upload_offset + required_aabb_size;
    auto upload_size = descriptor_upload_offset + sizeof(device_data);
    encoder.with_upload_buffer(upload_size, [&](auto upload_buffer) noexcept {
        auto upload = upload_buffer->address();
        std::memcpy(upload, host_cp.data(), required_cp_size);
        std::memcpy(upload + segments_upload_offset,
                    host_segments.data(), required_segments_size);
        std::memcpy(upload + aabbs_upload_offset,
                    host_aabbs.data(), required_aabb_size);
        std::memcpy(upload + descriptor_upload_offset,
                    &device_data, sizeof(device_data));
        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
            _control_points, upload, required_cp_size, hip_stream));
        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
            _segments, upload + segments_upload_offset,
            required_segments_size, hip_stream));
        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
            _aabb_buffer, upload + aabbs_upload_offset,
            required_aabb_size, hip_stream));
        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
            _device_data, upload + descriptor_upload_offset,
            sizeof(device_data), hip_stream));
    });

    hiprtAABBListPrimitive primitive{};
    primitive.aabbs = reinterpret_cast<hiprtDevicePtr>(_aabb_buffer);
    primitive.aabbCount = static_cast<uint32_t>(seg_count);
    primitive.aabbStride = sizeof(HIPCurveAABB);

    hiprtGeometryBuildInput build_input{};
    build_input.type = hiprtPrimitiveTypeAABBList;
    // Curves are the first custom function-table geometry type. Generic
    // procedural primitives retain hiprtInvalidValue and are distinguished by
    // the per-instance kind in HIPAccel::CodegenInstance.
    build_input.geomType = 0u;
    build_input.primitive.aabbList = primitive;

    hiprtBuildOptions build_options{};
    build_options.buildFlags = make_hiprt_build_flags(_option);

    if (requires_build) {
        if (recreate_geometry) {
            if (_geometry) {
                LUISA_CHECK_HIPRT(hiprtDestroyGeometry(_hiprt_ctx, _geometry));
                _geometry = nullptr;
            }
            LUISA_CHECK_HIPRT(hiprtCreateGeometry(_hiprt_ctx, build_input,
                                                  build_options, _geometry));
        }
        size_t temp_size = 0u;
        LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(
            _hiprt_ctx, build_input, build_options, temp_size));
        hipDeviceptr_t temp_buffer{};
        if (temp_size != 0u) {
            LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&temp_buffer),
                                           temp_size, hip_stream));
        }
        LUISA_CHECK_HIPRT(hiprtBuildGeometry(
            _hiprt_ctx, hiprtBuildOperationBuild, build_input, build_options,
            temp_buffer, hip_stream, _geometry));
        if (temp_buffer) {
            LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(temp_buffer), hip_stream));
        }
        if (_option.allow_compaction) {
            // HIPRT destroys the input geometry as part of compaction and permits
            // in-place replacement of the handle.
            LUISA_CHECK_HIPRT(hiprtCompactGeometry(
                _hiprt_ctx, hip_stream, _geometry, _geometry));
        }
    } else {
        LUISA_CHECK_HIPRT(hiprtBuildGeometry(
            _hiprt_ctx, hiprtBuildOperationUpdate, build_input, build_options,
            nullptr, hip_stream, _geometry));
    }
}

}// namespace luisa::compute::hip
