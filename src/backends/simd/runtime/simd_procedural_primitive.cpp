#include "simd_procedural_primitive.h"

#include <limits>

#include <luisa/core/logging.h>

#include "simd_buffer.h"

namespace luisa::compute::simd {

SIMDProceduralPrimitive::SIMDProceduralPrimitive(
    RTCDevice device, const AccelOption &option) noexcept
    : SIMDPrimitive{Kind::procedural},
      _scene{rtcNewScene(device)},
      _geometry{rtcNewGeometry(device, RTC_GEOMETRY_TYPE_USER)},
      _motion{option.motion} {
    simd_accel_set_flags(_scene, option);
    rtcSetGeometryMask(_geometry, ~0u);
    rtcSetGeometryIntersectFunction(
        _geometry, simd_procedural_intersect);
    rtcSetGeometryOccludedFunction(
        _geometry, simd_procedural_occluded);
    rtcAttachGeometry(_scene, _geometry);
    rtcReleaseGeometry(_geometry);
}

SIMDProceduralPrimitive::~SIMDProceduralPrimitive() noexcept {
    rtcReleaseScene(_scene);
}

void SIMDProceduralPrimitive::build(
    const ProceduralPrimitiveBuildCommand &command) noexcept {
    auto *buffer = reinterpret_cast<SIMDBuffer *>(command.aabb_buffer());
    LUISA_ASSERT(
        buffer != nullptr &&
            command.aabb_buffer_offset() <= buffer->size() &&
            command.aabb_buffer_size() <=
                buffer->size() - command.aabb_buffer_offset() &&
            command.aabb_buffer_size() != 0u &&
            command.aabb_buffer_size() % sizeof(AABB) == 0u,
        "Invalid SIMD procedural-primitive AABB buffer range.");
    auto count = command.aabb_buffer_size() / sizeof(AABB);
    auto primitive_count = count;
    if (_motion) {
        LUISA_ASSERT(
            count % _motion.keyframe_count == 0u,
            "SIMD motion procedural-primitive AABB count must be a "
            "multiple of the keyframe count.");
        primitive_count = count / _motion.keyframe_count;
        rtcSetGeometryTimeRange(
            _geometry, _motion.time_start, _motion.time_end);
        rtcSetGeometryTimeStepCount(
            _geometry, _motion.keyframe_count);
    } else {
        rtcSetGeometryTimeRange(_geometry, 0.0f, 1.0f);
        rtcSetGeometryTimeStepCount(_geometry, 1u);
    }
    LUISA_ASSERT(
        primitive_count <= std::numeric_limits<unsigned>::max(),
        "SIMD procedural primitive count {} exceeds Embree's limit.",
        primitive_count);
    rtcSetGeometryUserPrimitiveCount(
        _geometry, static_cast<unsigned>(primitive_count));

    struct BoundsData {
        const AABB *aabbs;
        size_t primitive_count;
    };
    auto *bytes = buffer->data() + command.aabb_buffer_offset();
    BoundsData bounds_data{
        .aabbs = reinterpret_cast<const AABB *>(bytes),
        .primitive_count = primitive_count,
    };
    rtcSetGeometryUserData(_geometry, &bounds_data);
    rtcSetGeometryBoundsFunction(
        _geometry,
        [](const RTCBoundsFunctionArguments *arguments) noexcept {
            auto *data = static_cast<const BoundsData *>(
                arguments->geometryUserPtr);
            LUISA_ASSERT(
                data != nullptr && arguments->primID < data->primitive_count,
                "Invalid SIMD procedural bounds callback invocation.");
            auto index = static_cast<size_t>(arguments->timeStep) *
                             data->primitive_count +
                         arguments->primID;
            auto &&aabb = data->aabbs[index];
            *arguments->bounds_o = RTCBounds{
                .lower_x = aabb.packed_min[0u],
                .lower_y = aabb.packed_min[1u],
                .lower_z = aabb.packed_min[2u],
                .align0 = 0.0f,
                .upper_x = aabb.packed_max[0u],
                .upper_y = aabb.packed_max[1u],
                .upper_z = aabb.packed_max[2u],
                .align1 = 0.0f,
            };
        },
        nullptr);
    rtcCommitGeometry(_geometry);
    rtcCommitScene(_scene);
    // Bounds are copied into Embree's acceleration structure during commit.
    // Never retain the stack-backed callback payload after that point.
    rtcSetGeometryBoundsFunction(_geometry, nullptr, nullptr);
    rtcSetGeometryUserData(_geometry, nullptr);
}

}// namespace luisa::compute::simd
