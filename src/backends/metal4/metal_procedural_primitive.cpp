#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/aabb.h>
#include "metal_buffer.h"
#include "metal_command_encoder.h"
#include "metal_procedural_primitive.h"

namespace luisa::compute::metal {

MetalProceduralPrimitive::MetalProceduralPrimitive(
    MTL::Device *device, const AccelOption &option) noexcept
    : MetalPrimitive{device, option} {}

MetalProceduralPrimitive::~MetalProceduralPrimitive() noexcept {
    if (_descriptor) { _descriptor->release(); }
    if (_compatibility_descriptor) {
        _compatibility_descriptor->release();
    }
    if (_aabb_buffer) { _aabb_buffer->release(); }
}

void MetalProceduralPrimitive::_do_add_resources(
    luisa::vector<MTL::Resource *> &resources) const noexcept {
    LUISA_ASSERT(_descriptor != nullptr ||
                     _compatibility_descriptor != nullptr,
                 "Procedural primitive not built.");
    resources.emplace_back(_aabb_buffer);
}

void MetalProceduralPrimitive::build(
    MetalCommandEncoder &encoder,
    ProceduralPrimitiveBuildCommand *command) noexcept {
    std::scoped_lock lock{mutex()};

    auto aabb_buffer = reinterpret_cast<MetalBuffer *>(command->aabb_buffer());
    auto aabb_buffer_handle = aabb_buffer->handle();
    auto aabb_buffer_offset = command->aabb_buffer_offset();
    auto aabb_buffer_size = command->aabb_buffer_size();
    constexpr auto aabb_stride = sizeof(AABB);
    auto keyframe_count = motion_keyframe_count();
    LUISA_ASSERT(aabb_buffer_size != 0u &&
                     aabb_buffer_size % (aabb_stride * keyframe_count) == 0u,
                 "Invalid AABB buffer size.");

    auto aabb_count = aabb_buffer_size / aabb_stride;
    auto geometry_buffer_changed =
        _aabb_buffer != aabb_buffer_handle ||
        _aabb_buffer_offset != aabb_buffer_offset ||
        _aabb_count != aabb_count;
    auto address_driven =
        encoder.stream()->supports_address_driven_acceleration_structures();
    auto descriptor_missing = address_driven ?
                                  _descriptor == nullptr :
                                  _compatibility_descriptor == nullptr;
    auto requires_build = handle() == nullptr ||
                          !option().allow_update ||
                          command->request() == AccelBuildRequest::FORCE_BUILD ||
                          descriptor_missing ||
                          geometry_buffer_changed;

    using GeometryDescriptor =
        MTL4::AccelerationStructureBoundingBoxGeometryDescriptor;
    using MotionGeometryDescriptor =
        MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor;
    if (requires_build) {
        if (_aabb_buffer != aabb_buffer_handle) {
            aabb_buffer_handle->retain();
            if (_aabb_buffer) { _aabb_buffer->release(); }
            _aabb_buffer = aabb_buffer_handle;
        }
        _aabb_buffer_offset = aabb_buffer_offset;
        _aabb_count = aabb_count;

        if (address_driven) {
            if (_descriptor) { _descriptor->release(); }
            _descriptor =
                MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
            _descriptor->setUsage(usage());
            _set_motion_options(_descriptor);
            if (option().motion) {
                auto geom_desc = NS::TransferPtr(
                    MotionGeometryDescriptor::alloc()->init());
                geom_desc->setBoundingBoxCount(
                    aabb_count / keyframe_count);
                geom_desc->setBoundingBoxStride(aabb_stride);
                geom_desc->setOpaque(false);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            } else {
                auto geom_desc = NS::TransferPtr(
                    GeometryDescriptor::alloc()->init());
                geom_desc->setBoundingBoxBuffer(MTL4::BufferRange{
                    aabb_buffer_handle->gpuAddress() + aabb_buffer_offset,
                    aabb_buffer_size});
                geom_desc->setBoundingBoxCount(aabb_count);
                geom_desc->setBoundingBoxStride(aabb_stride);
                geom_desc->setOpaque(false);
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
                    MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::alloc()->init());
                luisa::vector<NS::Object *> keyframes;
                keyframes.reserve(keyframe_count);
                auto pitch = aabb_buffer_size / keyframe_count;
                for (auto i = 0u; i < keyframe_count; i++) {
                    auto data = MTL::MotionKeyframeData::data();
                    data->setBuffer(aabb_buffer_handle);
                    data->setOffset(aabb_buffer_offset + i * pitch);
                    keyframes.emplace_back(data);
                }
                geom_desc->setBoundingBoxBuffers(NS::Array::array(
                    keyframes.data(), keyframes.size()));
                geom_desc->setBoundingBoxCount(
                    aabb_count / keyframe_count);
                geom_desc->setBoundingBoxStride(aabb_stride);
                geom_desc->setOpaque(false);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _compatibility_descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            } else {
                auto geom_desc = NS::TransferPtr(
                    MTL::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()->init());
                geom_desc->setBoundingBoxBuffer(aabb_buffer_handle);
                geom_desc->setBoundingBoxBufferOffset(aabb_buffer_offset);
                geom_desc->setBoundingBoxCount(aabb_count);
                geom_desc->setBoundingBoxStride(aabb_stride);
                geom_desc->setOpaque(false);
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
        luisa::vector<MTL4::BufferRange> ranges;
        ranges.reserve(keyframe_count);
        auto pitch = aabb_buffer_size / keyframe_count;
        for (auto i = 0u; i < keyframe_count; i++) {
            ranges.emplace_back(
                aabb_buffer_handle->gpuAddress() +
                    aabb_buffer_offset + i * pitch,
                pitch);
        }
        auto table = encoder.upload(
            ranges.data(), ranges.size() * sizeof(MTL4::BufferRange));
        geom_desc->setBoundingBoxBuffers(MTL4::BufferRange{
            table, ranges.size() * sizeof(MTL4::BufferRange)});
    }

    encoder.use_resource(aabb_buffer_handle);
    if (requires_build) {
        if (address_driven) { _do_build(encoder, _descriptor); }
        else { _do_build(encoder, _compatibility_descriptor); }
    } else {
        if (address_driven) { _do_update(encoder, _descriptor); }
        else { _do_update(encoder, _compatibility_descriptor); }
    }
}

}// namespace luisa::compute::metal
