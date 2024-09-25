#include <luisa/core/logging.h>
#include "metal_command_encoder.h"
#include "metal_primitive.h"

namespace luisa::compute::metal {

MetalPrimitive::MetalPrimitive(MTL::Device *device [[maybe_unused]],
                               const AccelOption &option) noexcept
    : _option{option} {}

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
                               MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept {
    LUISA_ASSERT(descriptor != nullptr, "Invalid acceleration structure descriptor.");
    auto device = encoder.device();
    auto sizes = device->accelerationStructureSizes(descriptor);
    if (option().allow_update) {
        if (_update_buffer == nullptr ||
            _update_buffer->length() < sizes.refitScratchBufferSize) {
            if (_update_buffer != nullptr) { _update_buffer->release(); }
            _update_buffer = device->newBuffer(sizes.refitScratchBufferSize,
                                               MTL::ResourceHazardTrackingModeTracked |
                                                   MTL::ResourceStorageModePrivate);
        }
    }
    if (_handle != nullptr) { _handle->release(); }
    _handle = device->newAccelerationStructure(sizes.accelerationStructureSize);
    _handle->setLabel(_name);
    auto build_buffer = device->newBuffer(sizes.buildScratchBufferSize,
                                          MTL::ResourceHazardTrackingModeTracked |
                                              MTL::ResourceStorageModePrivate);
    auto build_encoder = encoder.command_buffer()->accelerationStructureCommandEncoder();
    _handle->retain();
    descriptor->retain();
    build_encoder->buildAccelerationStructure(_handle, descriptor, build_buffer, 0u);
    build_encoder->endEncoding();
    encoder.add_callback(FunctionCallbackContext::create([handle = _handle,
                                                          build_buffer = build_buffer,
                                                          descriptor] {
        handle->release();
        build_buffer->release();
        descriptor->release();
    }));

    auto compacted_size = 0u;
    if (option().allow_compaction) {
        // read back the size of the compacted acceleration structure
        encoder.with_download_buffer(sizeof(uint), [&](MetalStageBufferPool::Allocation *size_buffer) noexcept {
            auto size_encoder = encoder.command_buffer()->accelerationStructureCommandEncoder();
            size_encoder->writeCompactedAccelerationStructureSize(
                _handle, size_buffer->buffer(), size_buffer->offset(), MTL::DataTypeUInt);
            size_encoder->endEncoding();
            encoder.add_callback(FunctionCallbackContext::create([size_buffer, &compacted_size] {
                compacted_size = *reinterpret_cast<uint *>(size_buffer->data());
            }));
        });
        auto submitted_command_buffer = encoder.submit({});
        submitted_command_buffer->waitUntilCompleted();
        auto compacted_handle = device->newAccelerationStructure(compacted_size);
        compacted_handle->setLabel(_name);
        auto compact_encoder = encoder.command_buffer()->accelerationStructureCommandEncoder();
        compacted_handle->retain();
        compact_encoder->copyAndCompactAccelerationStructure(_handle, compacted_handle);
        compact_encoder->endEncoding();
        encoder.add_callback(FunctionCallbackContext::create([old_handle = _handle,
                                                              compacted_handle] {
            old_handle->release();
            compacted_handle->release();
        }));
        _handle = compacted_handle;
    }
}

void MetalPrimitive::_do_update(MetalCommandEncoder &encoder,
                                MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept {

    LUISA_ASSERT(_handle != nullptr, "Acceleration structure not built yet.");
    LUISA_ASSERT(_update_buffer != nullptr, "Invalid acceleration structure update buffer.");
    LUISA_ASSERT(descriptor != nullptr, "Invalid acceleration structure descriptor.");

    auto refit_encoder = encoder.command_buffer()->accelerationStructureCommandEncoder();
    _handle->retain();
    _update_buffer->retain();
    descriptor->retain();
    refit_encoder->refitAccelerationStructure(_handle, descriptor, _handle, _update_buffer, 0u);
    refit_encoder->endEncoding();
    encoder.add_callback(FunctionCallbackContext::create([handle = _handle,
                                                          update_buffer = _update_buffer,
                                                          descriptor] {
        handle->release();
        update_buffer->release();
        descriptor->release();
    }));
}

void MetalPrimitive::_set_motion_options(MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept {
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
