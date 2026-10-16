#include <luisa/core/logging.h>
#include "metal_command_encoder.h"
#include "metal_acceleration_structure_build.h"
#include "metal_primitive.h"

namespace luisa::compute::metal {

MetalPrimitive::MetalPrimitive(MTL::Device *device [[maybe_unused]],
                               const AccelOption &option) noexcept
    : MetalPrimitiveBase{Kind::PRIMITIVE}, _option{option} {}

MetalPrimitive::~MetalPrimitive() noexcept {
    if (_handle) { _handle->release(); }
    if (_update_buffer) { _update_buffer->release(); }
}

MTL::AccelerationStructureUsage MetalPrimitive::usage() const noexcept {
    auto u = 0u;
    switch (option().hint) {
        case AccelOption::UsageHint::FAST_TRACE:
            u |= MTL::AccelerationStructureUsageNone;
            break;
        case AccelOption::UsageHint::FAST_BUILD:
            u |= MTL::AccelerationStructureUsagePreferFastBuild;
            break;
    }
    if (option().allow_update) { u |= MTL::AccelerationStructureUsageRefit; }
    return static_cast<MTL::AccelerationStructureUsage>(u);
}

void MetalPrimitive::add_resources(luisa::vector<MTL::Resource *> &resources) noexcept {
    std::scoped_lock lock{_mutex};
    resources.emplace_back(_handle);
    _do_add_resources(resources);
}

void MetalPrimitive::_do_build(MetalCommandEncoder &encoder,
                               MTL4::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept {
    LUISA_ASSERT(descriptor != nullptr, "Invalid acceleration structure descriptor.");
    auto device = encoder.device();
    auto sizes = device->accelerationStructureSizes(descriptor);
    if (option().allow_update) {
        if (_update_buffer == nullptr ||
            _update_buffer->length() < sizes.refitScratchBufferSize) {
            if (_update_buffer != nullptr) { _update_buffer->release(); }
            if (sizes.refitScratchBufferSize != 0) {
                _update_buffer = device->newBuffer(sizes.refitScratchBufferSize,
                                                   MTL::ResourceHazardTrackingModeTracked |
                                                       MTL::ResourceStorageModePrivate);
            }
        }
    }
    if (_handle != nullptr) { _handle->release(); }
    _handle = device->newAccelerationStructure(sizes.accelerationStructureSize);
    _handle->setLabel(_name);
    auto build_buffer = device->newBuffer(sizes.buildScratchBufferSize,
                                          MTL::ResourceHazardTrackingModeTracked |
                                              MTL::ResourceStorageModePrivate);
    auto build_encoder = encoder.compute_encoder();
    _handle->retain();
    descriptor->retain();
    build_encoder->buildAccelerationStructure(
        _handle, descriptor,
        MTL4::BufferRange{build_buffer->gpuAddress(), build_buffer->length()});
    encoder.use_resource(_handle);
    encoder.use_resource(build_buffer);
    build_encoder->endEncoding();
    encoder.add_callback(FunctionCallbackContext::create([handle = _handle,
                                                          build_buffer = build_buffer,
                                                          descriptor] {
        handle->release();
        build_buffer->release();
        descriptor->release();
    }));

    auto compacted_size = uint64_t{0u};
    if (option().allow_compaction) {
        // read back the size of the compacted acceleration structure
        encoder.with_download_buffer(sizeof(uint64_t), [&](MetalStageBufferPool::Allocation *size_buffer) noexcept {
            auto size_encoder = encoder.compute_encoder();
            size_encoder->writeCompactedAccelerationStructureSize(
                _handle,
                MTL4::BufferRange{
                    size_buffer->buffer()->gpuAddress() + size_buffer->offset(),
                    sizeof(uint64_t)});
            encoder.use_resource(_handle);
            encoder.use_resource(size_buffer->buffer());
            size_encoder->endEncoding();
            encoder.add_callback(FunctionCallbackContext::create([size_buffer, &compacted_size] {
                compacted_size = *reinterpret_cast<uint64_t *>(size_buffer->data());
            }));
        });
        encoder.submit_and_wait();
        auto compacted_handle = device->newAccelerationStructure(compacted_size);
        compacted_handle->setLabel(_name);
        auto compact_encoder = encoder.compute_encoder();
        compacted_handle->retain();
        compact_encoder->copyAndCompactAccelerationStructure(_handle, compacted_handle);
        encoder.use_resource(_handle);
        encoder.use_resource(compacted_handle);
        compact_encoder->endEncoding();
        encoder.add_callback(FunctionCallbackContext::create([old_handle = _handle,
                                                              compacted_handle] {
            old_handle->release();
            compacted_handle->release();
        }));
        _handle = compacted_handle;
    }
}

void MetalPrimitive::_do_build(
    MetalCommandEncoder &encoder,
    MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept {
    build_acceleration_structure_compatibility(
        encoder, descriptor, option().allow_update,
        option().allow_compaction, _name, _handle, _update_buffer);
}

void MetalPrimitive::_do_update(MetalCommandEncoder &encoder,
                                MTL4::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept {

    LUISA_ASSERT(_handle != nullptr, "Acceleration structure not built yet.");
    LUISA_ASSERT(descriptor != nullptr, "Invalid acceleration structure descriptor.");

    auto refit_encoder = encoder.compute_encoder();
    _handle->retain();
    if (_update_buffer != nullptr) { _update_buffer->retain(); }
    descriptor->retain();
    refit_encoder->refitAccelerationStructure(
        _handle, descriptor, _handle,
        MTL4::BufferRange{
            _update_buffer == nullptr ? 0u : _update_buffer->gpuAddress(),
            _update_buffer == nullptr ? 0u : _update_buffer->length()});
    encoder.use_resource(_handle);
    encoder.use_resource(_update_buffer);
    refit_encoder->endEncoding();
    encoder.add_callback(FunctionCallbackContext::create([handle = _handle,
                                                          update_buffer = _update_buffer,
                                                          descriptor] {
        handle->release();
        descriptor->release();
        if (update_buffer != nullptr) { update_buffer->release(); }
    }));
}

void MetalPrimitive::_do_update(
    MetalCommandEncoder &encoder,
    MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept {
    refit_acceleration_structure_compatibility(
        encoder, descriptor, _handle, _update_buffer);
}

void MetalPrimitive::_set_motion_options(MTL4::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept {
    if (auto m = _option.motion) {
        descriptor->setMotionKeyframeCount(m.keyframe_count);
        descriptor->setMotionStartTime(m.time_start);
        descriptor->setMotionEndTime(m.time_end);
        if (m.should_vanish_start) {
            descriptor->setMotionStartBorderMode(MTL::MotionBorderModeVanish);
        } else {
            descriptor->setMotionStartBorderMode(MTL::MotionBorderModeClamp);
        }
        if (m.should_vanish_end) {
            descriptor->setMotionEndBorderMode(MTL::MotionBorderModeVanish);
        } else {
            descriptor->setMotionEndBorderMode(MTL::MotionBorderModeClamp);
        }
    }
}

void MetalPrimitive::_set_motion_options(
    MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept {
    if (auto m = _option.motion) {
        descriptor->setMotionKeyframeCount(m.keyframe_count);
        descriptor->setMotionStartTime(m.time_start);
        descriptor->setMotionEndTime(m.time_end);
        descriptor->setMotionStartBorderMode(
            m.should_vanish_start ? MTL::MotionBorderModeVanish :
                                    MTL::MotionBorderModeClamp);
        descriptor->setMotionEndBorderMode(
            m.should_vanish_end ? MTL::MotionBorderModeVanish :
                                  MTL::MotionBorderModeClamp);
    }
}

void MetalPrimitive::set_name(luisa::string_view name) noexcept {
    std::scoped_lock lock{_mutex};
    if (_name) {
        _name->release();
        _name = nullptr;
    }
    if (!name.empty()) {
        _name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
    }
    if (_handle) { _handle->setLabel(_name); }
}

}// namespace luisa::compute::metal
