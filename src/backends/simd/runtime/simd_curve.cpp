#include "simd_curve.h"

#include <luisa/core/logging.h>

#include "simd_buffer.h"

namespace luisa::compute::simd {

namespace {

[[nodiscard]] RTCGeometryType embree_curve_type(
    CurveBasis basis) noexcept {
    switch (basis) {
        case CurveBasis::PIECEWISE_LINEAR:
            return RTC_GEOMETRY_TYPE_ROUND_LINEAR_CURVE;
        case CurveBasis::CUBIC_BSPLINE:
            return RTC_GEOMETRY_TYPE_ROUND_BSPLINE_CURVE;
        case CurveBasis::CATMULL_ROM:
            return RTC_GEOMETRY_TYPE_ROUND_CATMULL_ROM_CURVE;
        case CurveBasis::BEZIER:
            return RTC_GEOMETRY_TYPE_ROUND_BEZIER_CURVE;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid SIMD curve basis 0x{:x}.",
        luisa::to_underlying(basis));
}

}// namespace

SIMDCurve::SIMDCurve(
    RTCDevice device, const AccelOption &option) noexcept
    : SIMDPrimitive{Kind::curve},
      _scene{rtcNewScene(device)},
      _motion{option.motion} {
    simd_accel_set_flags(_scene, option);
}

SIMDCurve::~SIMDCurve() noexcept { rtcReleaseScene(_scene); }

void SIMDCurve::build(const CurveBuildCommand &command) noexcept {
    if (_geometry == nullptr) {
        _basis = command.basis();
        _geometry = rtcNewGeometry(
            rtcGetSceneDevice(_scene), embree_curve_type(_basis));
        rtcSetGeometryMask(_geometry, ~0u);
        rtcAttachGeometry(_scene, _geometry);
        rtcReleaseGeometry(_geometry);
    } else {
        LUISA_ASSERT(
            command.basis() == _basis,
            "A SIMD curve cannot change basis after creation.");
    }

    LUISA_ASSERT(
        command.cp_stride() >= sizeof(float4) &&
            command.cp_count() != 0u &&
            command.seg_count() != 0u,
        "Invalid SIMD curve control-point stride or element count.");
    auto *control_points =
        reinterpret_cast<SIMDBuffer *>(command.cp_buffer())->data();
    auto *segments =
        reinterpret_cast<SIMDBuffer *>(command.seg_buffer())->data();
    if (_motion) {
        LUISA_ASSERT(
            command.cp_count() % _motion.keyframe_count == 0u,
            "SIMD motion-curve control-point count must be a multiple of "
            "the keyframe count.");
        auto points_per_keyframe =
            command.cp_count() / _motion.keyframe_count;
        auto frame_stride = command.cp_stride() * points_per_keyframe;
        rtcSetGeometryTimeRange(
            _geometry, _motion.time_start, _motion.time_end);
        rtcSetGeometryTimeStepCount(
            _geometry, _motion.keyframe_count);
        for (auto frame = 0u; frame < _motion.keyframe_count; frame++) {
            rtcSetSharedGeometryBuffer(
                _geometry, RTC_BUFFER_TYPE_VERTEX, frame,
                RTC_FORMAT_FLOAT4, control_points,
                command.cp_buffer_offset() + frame * frame_stride,
                command.cp_stride(), points_per_keyframe);
        }
    } else {
        rtcSetSharedGeometryBuffer(
            _geometry, RTC_BUFFER_TYPE_VERTEX, 0u,
            RTC_FORMAT_FLOAT4, control_points,
            command.cp_buffer_offset(), command.cp_stride(),
            command.cp_count());
    }
    rtcSetSharedGeometryBuffer(
        _geometry, RTC_BUFFER_TYPE_INDEX, 0u,
        RTC_FORMAT_UINT, segments,
        command.seg_buffer_offset(), sizeof(uint32_t),
        command.seg_count());
    rtcCommitGeometry(_geometry);
    rtcCommitScene(_scene);
}

}// namespace luisa::compute::simd
