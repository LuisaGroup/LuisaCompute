#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/curve_basis.h>
#include "metal_command_encoder.h"
#include "metal_buffer.h"
#include "metal_curve.h"

namespace luisa::compute::metal {

void MetalCurve::_do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept {
    LUISA_ASSERT(_descriptor != nullptr ||
                     _compatibility_descriptor != nullptr,
                 "Curve not built.");
    resources.emplace_back(_control_point_buffer);
    resources.emplace_back(_segment_buffer);
}

MetalCurve::MetalCurve(MTL::Device *device, const AccelOption &option) noexcept
    : MetalPrimitive{device, option} {}

MetalCurve::~MetalCurve() noexcept {
    if (_descriptor) { _descriptor->release(); }
    if (_compatibility_descriptor) {
        _compatibility_descriptor->release();
    }
    if (_control_point_buffer) { _control_point_buffer->release(); }
    if (_segment_buffer) { _segment_buffer->release(); }
}

void MetalCurve::build(MetalCommandEncoder &encoder,
                       CurveBuildCommand *command) noexcept {
    std::scoped_lock lock{mutex()};

    auto cp_count = command->cp_count();
    auto seg_count = command->seg_count();
    auto keyframe_count = motion_keyframe_count();
    LUISA_ASSERT(cp_count != 0u && cp_count % keyframe_count == 0u,
                 "Invalid control point count for motion keyframes.");

    auto cp_buffer = reinterpret_cast<MetalBuffer *>(command->cp_buffer());
    auto cp_buffer_handle = cp_buffer->handle();
    auto cp_buffer_offset = command->cp_buffer_offset();
    auto cp_stride = command->cp_stride();
    LUISA_ASSERT(cp_stride >= sizeof(float4) &&
                     cp_buffer_offset + cp_count * cp_stride <=
                         cp_buffer_handle->length(),
                 "Invalid control point buffer size.");

    auto radius_buffer_handle = cp_buffer_handle;
    auto radius_buffer_offset = cp_buffer_offset + sizeof(float) * 3u;
    auto radius_stride = cp_stride;

    auto seg_buffer = reinterpret_cast<MetalBuffer *>(command->seg_buffer());
    auto seg_buffer_handle = seg_buffer->handle();
    auto seg_buffer_offset = command->seg_buffer_offset();
    LUISA_ASSERT(seg_buffer_offset + seg_count * sizeof(uint) <=
                     seg_buffer_handle->length(),
                 "Invalid segment buffer size.");

    auto [basis, end_cap] = [curve_basis = command->basis()] {
        switch (curve_basis) {
            case CurveBasis::PIECEWISE_LINEAR:
                return std::make_pair(MTL::CurveBasisLinear,
                                      MTL::CurveEndCapsSphere);
            case CurveBasis::CUBIC_BSPLINE:
                return std::make_pair(MTL::CurveBasisBSpline,
                                      MTL::CurveEndCapsNone);
            case CurveBasis::CATMULL_ROM:
                return std::make_pair(MTL::CurveBasisCatmullRom,
                                      MTL::CurveEndCapsNone);
            case CurveBasis::BEZIER:
                return std::make_pair(MTL::CurveBasisBezier,
                                      MTL::CurveEndCapsNone);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid curve basis.");
    }();
    auto cp_per_seg = segment_control_point_count(command->basis());

    auto geometry_buffers_changed =
        _control_point_buffer != cp_buffer_handle ||
        _segment_buffer != seg_buffer_handle ||
        _control_point_buffer_offset != cp_buffer_offset ||
        _control_point_count != cp_count ||
        _control_point_stride != cp_stride ||
        _segment_buffer_offset != seg_buffer_offset ||
        _segment_count != seg_count ||
        _basis != basis || _end_caps != end_cap;
    auto address_driven =
        encoder.stream()->supports_address_driven_acceleration_structures();
    auto descriptor_missing = address_driven ?
                                  _descriptor == nullptr :
                                  _compatibility_descriptor == nullptr;
    auto requires_build = handle() == nullptr ||
                          !option().allow_update ||
                          command->request() == AccelBuildRequest::FORCE_BUILD ||
                          descriptor_missing ||
                          geometry_buffers_changed;

    using GeometryDescriptor =
        MTL4::AccelerationStructureCurveGeometryDescriptor;
    using MotionGeometryDescriptor =
        MTL4::AccelerationStructureMotionCurveGeometryDescriptor;
    if (requires_build) {
        if (_control_point_buffer != cp_buffer_handle) {
            cp_buffer_handle->retain();
            if (_control_point_buffer) { _control_point_buffer->release(); }
            _control_point_buffer = cp_buffer_handle;
        }
        if (_segment_buffer != seg_buffer_handle) {
            seg_buffer_handle->retain();
            if (_segment_buffer) { _segment_buffer->release(); }
            _segment_buffer = seg_buffer_handle;
        }
        _control_point_buffer_offset = cp_buffer_offset;
        _control_point_count = cp_count;
        _control_point_stride = cp_stride;
        _segment_buffer_offset = seg_buffer_offset;
        _segment_count = seg_count;
        _basis = basis;
        _end_caps = end_cap;

        if (address_driven) {
            if (_descriptor) { _descriptor->release(); }
            _descriptor =
                MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
            _descriptor->setUsage(usage());
            _set_motion_options(_descriptor);
            if (option().motion) {
                auto geom_desc = NS::TransferPtr(
                    MotionGeometryDescriptor::alloc()->init());
                geom_desc->setControlPointStride(cp_stride);
                geom_desc->setControlPointCount(
                    cp_count / keyframe_count);
                geom_desc->setControlPointFormat(
                    MTL::AttributeFormatFloat3);
                geom_desc->setRadiusStride(radius_stride);
                geom_desc->setRadiusFormat(MTL::AttributeFormatFloat);
                geom_desc->setIndexBuffer(MTL4::BufferRange{
                    seg_buffer_handle->gpuAddress() + seg_buffer_offset,
                    seg_count * sizeof(uint)});
                geom_desc->setIndexType(MTL::IndexTypeUInt32);
                geom_desc->setSegmentCount(seg_count);
                geom_desc->setSegmentControlPointCount(cp_per_seg);
                geom_desc->setCurveType(MTL::CurveTypeRound);
                geom_desc->setCurveBasis(basis);
                geom_desc->setCurveEndCaps(end_cap);
                geom_desc->setOpaque(true);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            } else {
                auto geom_desc = NS::TransferPtr(
                    GeometryDescriptor::alloc()->init());
                geom_desc->setControlPointBuffer(MTL4::BufferRange{
                    cp_buffer_handle->gpuAddress() + cp_buffer_offset,
                    cp_count * cp_stride});
                geom_desc->setControlPointStride(cp_stride);
                geom_desc->setControlPointCount(cp_count);
                geom_desc->setControlPointFormat(
                    MTL::AttributeFormatFloat3);
                geom_desc->setRadiusBuffer(MTL4::BufferRange{
                    radius_buffer_handle->gpuAddress() +
                        radius_buffer_offset,
                    (cp_count - 1u) * radius_stride + sizeof(float)});
                geom_desc->setRadiusStride(radius_stride);
                geom_desc->setRadiusFormat(MTL::AttributeFormatFloat);
                geom_desc->setIndexBuffer(MTL4::BufferRange{
                    seg_buffer_handle->gpuAddress() + seg_buffer_offset,
                    seg_count * sizeof(uint)});
                geom_desc->setIndexType(MTL::IndexTypeUInt32);
                geom_desc->setSegmentCount(seg_count);
                geom_desc->setSegmentControlPointCount(cp_per_seg);
                geom_desc->setCurveType(MTL::CurveTypeRound);
                geom_desc->setCurveBasis(basis);
                geom_desc->setCurveEndCaps(end_cap);
                geom_desc->setOpaque(true);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            }
        } else {
            if (_compatibility_descriptor) {
                _compatibility_descriptor->release();
            }
            _compatibility_descriptor =
                MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
            _compatibility_descriptor->setUsage(usage());
            _set_motion_options(_compatibility_descriptor);
            if (option().motion) {
                auto geom_desc = NS::TransferPtr(
                    MTL::AccelerationStructureMotionCurveGeometryDescriptor::alloc()->init());
                luisa::vector<NS::Object *> cp_keyframes;
                luisa::vector<NS::Object *> radius_keyframes;
                cp_keyframes.reserve(keyframe_count);
                radius_keyframes.reserve(keyframe_count);
                auto keyframe_cp_count = cp_count / keyframe_count;
                auto pitch = cp_stride * keyframe_cp_count;
                for (auto i = 0u; i < keyframe_count; i++) {
                    auto cp_data = MTL::MotionKeyframeData::data();
                    cp_data->setBuffer(cp_buffer_handle);
                    cp_data->setOffset(cp_buffer_offset + i * pitch);
                    cp_keyframes.emplace_back(cp_data);
                    auto radius_data = MTL::MotionKeyframeData::data();
                    radius_data->setBuffer(radius_buffer_handle);
                    radius_data->setOffset(
                        radius_buffer_offset + i * pitch);
                    radius_keyframes.emplace_back(radius_data);
                }
                geom_desc->setControlPointBuffers(NS::Array::array(
                    cp_keyframes.data(), cp_keyframes.size()));
                geom_desc->setControlPointStride(cp_stride);
                geom_desc->setControlPointCount(keyframe_cp_count);
                geom_desc->setControlPointFormat(
                    MTL::AttributeFormatFloat3);
                geom_desc->setRadiusBuffers(NS::Array::array(
                    radius_keyframes.data(), radius_keyframes.size()));
                geom_desc->setRadiusStride(radius_stride);
                geom_desc->setRadiusFormat(MTL::AttributeFormatFloat);
                geom_desc->setIndexBuffer(seg_buffer_handle);
                geom_desc->setIndexBufferOffset(seg_buffer_offset);
                geom_desc->setIndexType(MTL::IndexTypeUInt32);
                geom_desc->setSegmentCount(seg_count);
                geom_desc->setSegmentControlPointCount(cp_per_seg);
                geom_desc->setCurveType(MTL::CurveTypeRound);
                geom_desc->setCurveBasis(basis);
                geom_desc->setCurveEndCaps(end_cap);
                geom_desc->setOpaque(true);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _compatibility_descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            } else {
                auto geom_desc = NS::TransferPtr(
                    MTL::AccelerationStructureCurveGeometryDescriptor::alloc()->init());
                geom_desc->setControlPointBuffer(cp_buffer_handle);
                geom_desc->setControlPointBufferOffset(cp_buffer_offset);
                geom_desc->setControlPointStride(cp_stride);
                geom_desc->setControlPointCount(cp_count);
                geom_desc->setControlPointFormat(
                    MTL::AttributeFormatFloat3);
                geom_desc->setRadiusBuffer(radius_buffer_handle);
                geom_desc->setRadiusBufferOffset(radius_buffer_offset);
                geom_desc->setRadiusStride(radius_stride);
                geom_desc->setRadiusFormat(MTL::AttributeFormatFloat);
                geom_desc->setIndexBuffer(seg_buffer_handle);
                geom_desc->setIndexBufferOffset(seg_buffer_offset);
                geom_desc->setIndexType(MTL::IndexTypeUInt32);
                geom_desc->setSegmentCount(seg_count);
                geom_desc->setSegmentControlPointCount(cp_per_seg);
                geom_desc->setCurveType(MTL::CurveTypeRound);
                geom_desc->setCurveBasis(basis);
                geom_desc->setCurveEndCaps(end_cap);
                geom_desc->setOpaque(true);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _compatibility_descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            }
        }
    }

    if (address_driven && option().motion) {
        auto geom_desc = _descriptor->geometryDescriptors()
                             ->object<MotionGeometryDescriptor>(0u);
        luisa::vector<MTL4::BufferRange> cp_ranges;
        luisa::vector<MTL4::BufferRange> radius_ranges;
        cp_ranges.reserve(keyframe_count);
        radius_ranges.reserve(keyframe_count);
        auto keyframe_cp_count = cp_count / keyframe_count;
        auto pitch = cp_stride * keyframe_cp_count;
        for (auto i = 0u; i < keyframe_count; i++) {
            auto base = cp_buffer_handle->gpuAddress() +
                        cp_buffer_offset + i * pitch;
            cp_ranges.emplace_back(base, pitch);
            radius_ranges.emplace_back(base + sizeof(float) * 3u,
                                       pitch - sizeof(float) * 3u);
        }
        auto cp_table = encoder.upload(
            cp_ranges.data(), cp_ranges.size() * sizeof(MTL4::BufferRange));
        auto radius_table = encoder.upload(
            radius_ranges.data(),
            radius_ranges.size() * sizeof(MTL4::BufferRange));
        geom_desc->setControlPointBuffers(MTL4::BufferRange{
            cp_table, cp_ranges.size() * sizeof(MTL4::BufferRange)});
        geom_desc->setRadiusBuffers(MTL4::BufferRange{
            radius_table, radius_ranges.size() * sizeof(MTL4::BufferRange)});
    }

    encoder.use_resource(cp_buffer_handle);
    encoder.use_resource(seg_buffer_handle);
    if (requires_build) {
        if (address_driven) { _do_build(encoder, _descriptor); }
        else { _do_build(encoder, _compatibility_descriptor); }
    } else {
        if (address_driven) { _do_update(encoder, _descriptor); }
        else { _do_update(encoder, _compatibility_descriptor); }
    }
}

}// namespace luisa::compute::metal
