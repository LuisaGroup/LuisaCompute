#include <luisa/core/logging.h>
#include "metal_device.h"
#include "metal_command_encoder.h"
#include "metal_primitive.h"
#include "metal_accel.h"

namespace luisa::compute::metal {

MetalAccel::MetalAccel(MetalDevice *device, const AccelOption &option) noexcept
    : _update{device->builtin_update_accel_instances()},
      _option{option} { _resources.reserve(reserved_primitive_count); }

MetalAccel::~MetalAccel() noexcept {
    if (_handle) { _handle->release(); }
    if (_instance_buffer) { _instance_buffer->release(); }
    if (_update_buffer) { _update_buffer->release(); }
    if (_descriptor) { _descriptor->release(); }
    if (_name) { _name->release(); }
}

void MetalAccel::build(MetalCommandEncoder &encoder, AccelBuildCommand *command) noexcept {

    std::scoped_lock lock{_mutex};

    auto device = encoder.device();
    auto instance_count = command->instance_count();
    LUISA_ASSERT(instance_count > 0u, "Empty acceleration structure is not allowed.");
    if (auto size = instance_count * sizeof(MTL::AccelerationStructureInstanceDescriptor);
        _instance_buffer == nullptr || _instance_buffer->length() < size) {
        auto old_instance_buffer = _instance_buffer;
        _instance_buffer = device->newBuffer(size, MTL::ResourceStorageModePrivate |
                                                       MTL::ResourceHazardTrackingModeTracked);
        if (old_instance_buffer) {
            auto blit_encoder = encoder.command_buffer()->blitCommandEncoder();
            blit_encoder->copyFromBuffer(old_instance_buffer, 0u,
                                         _instance_buffer, 0u,
                                         old_instance_buffer->length());
            blit_encoder->endEncoding();
            encoder.add_callback(FunctionCallbackContext::create([old_instance_buffer] {
                old_instance_buffer->release();
            }));
        }
    }
    // collect instance updates
    auto mods = command->modifications();
    auto old_instance_count = _descriptor == nullptr ? 0u : _descriptor->instanceCount();
    _primitives.resize(instance_count);
    if (auto n = static_cast<uint>(mods.size())) {
        using Mod = AccelBuildCommand::Modification;
        auto mod_buffer_size = n * sizeof(Mod);
        encoder.with_upload_buffer(mod_buffer_size, [&](MetalStageBufferPool::Allocation *mod_buffer) noexcept {
            // collect updates into the upload buffer
            auto updates = reinterpret_cast<Mod *>(mod_buffer->data());
            for (auto i = 0u; i < n; i++) {
                auto m = mods[i];
                if (m.flags & Mod::flag_primitive) {
                    _requires_rebuild = true;
                    _primitives[m.index] = reinterpret_cast<MetalPrimitive *>(m.primitive);
                }
                updates[i] = m;
            }
            // launch the update kernel
            auto compute_encoder = encoder.command_buffer()->computeCommandEncoder(MTL::DispatchTypeConcurrent);
            compute_encoder->setComputePipelineState(_update);
            compute_encoder->setBuffer(_instance_buffer, 0u, 0u);
            compute_encoder->setBuffer(mod_buffer->buffer(), mod_buffer->offset(), 1u);
            compute_encoder->setBytes(&n, sizeof(n), 2u);
            constexpr auto block_size = MetalDevice::update_accel_instances_block_size;
            auto block_count = (n + block_size - 1u) / block_size;
            compute_encoder->dispatchThreadgroups(MTL::Size{block_count, 1u, 1u},
                                                  MTL::Size{block_size, 1u, 1u});
            compute_encoder->endEncoding();
        });
    }

    // find additional handle changes due to shrinking or primitive change or rebuild
    if (_descriptor != nullptr) {
        // release old primitives due to primitive changes
        for (auto i = 0u; i < instance_count && i < old_instance_count; i++) {
            if (auto old_prim = _descriptor->instancedAccelerationStructures()
                                    ->object<MTL::AccelerationStructure>(i),
                new_prim = _primitives[i]->handle();
                old_prim != new_prim) {
                _requires_rebuild = true;
            }
        }
    }

    // find out if we need to rebuild the acceleration structure
    _requires_rebuild = _requires_rebuild /* pending rebuild */ ||
                        _descriptor == nullptr || old_instance_count != instance_count /* instance count has changed */ ||
                        _handle == nullptr /* not built before */ ||
                        !_option.allow_update /* accel cannot be refitted */ ||
                        command->request() == AccelBuildRequest::FORCE_BUILD /* rebuild is forced */;

    // prepare the descriptor
    if (_requires_rebuild) {
        if (_descriptor != nullptr) { _descriptor->release(); }
        _descriptor = MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
    }
    _descriptor->setInstanceCount(instance_count);
    _descriptor->setInstanceDescriptorBuffer(_instance_buffer);
    _descriptor->setInstanceDescriptorBufferOffset(0u);
    _descriptor->setInstanceDescriptorStride(sizeof(MTL::AccelerationStructureInstanceDescriptor));
    _descriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeDefault);
    auto usage = 0u;
    switch (_option.hint) {
        case AccelOption::UsageHint::FAST_TRACE:
            usage |= MTL::AccelerationStructureUsageNone;
            break;
        case AccelOption::UsageHint::FAST_BUILD:
            usage |= MTL::AccelerationStructureUsagePreferFastBuild;
            break;
    }
    if (_option.allow_update) { usage |= MTL::AccelerationStructureUsageRefit; }
    _descriptor->setUsage(usage);

    // update the descriptor
    if (_requires_rebuild) {
        luisa::vector<NS::Object *> objects;
        objects.reserve(instance_count);
        std::transform(_primitives.begin(), _primitives.end(), std::back_inserter(objects),
                       [](auto p) noexcept {
                           auto handle = p->handle();
#ifndef NDEBUG
                           LUISA_ASSERT(handle != nullptr, "Invalid primitive handle.");
#endif
                           return handle;
                       });
        auto instances = NS::Array::array(objects.data(), objects.size());
        _descriptor->setInstancedAccelerationStructures(instances);
    }

    if (!command->update_instance_buffer_only()) {
        if (_requires_rebuild) {
            _do_build(encoder);
        } else {
            _do_update(encoder);
        }
        _requires_rebuild = false;
    }
}

void MetalAccel::_do_update(MetalCommandEncoder &encoder) noexcept {
    LUISA_ASSERT(_handle != nullptr, "Acceleration structure is not built.");
    LUISA_ASSERT(_descriptor != nullptr, "Descriptor is not allocated.");
    LUISA_ASSERT(_instance_buffer != nullptr, "Instance buffer is not allocated.");
    LUISA_ASSERT(_update_buffer != nullptr, "Update buffer is not allocated.");
    auto command_encoder = encoder.command_buffer()->accelerationStructureCommandEncoder();
    _descriptor->retain();
    _handle->retain();
    _update_buffer->retain();
    command_encoder->refitAccelerationStructure(_handle, _descriptor, _handle, _update_buffer, 0u);
    command_encoder->endEncoding();
    encoder.add_callback(FunctionCallbackContext::create([descriptor = _descriptor,
                                                          handle = _handle,
                                                          update_buffer = _update_buffer] {
        descriptor->release();
        handle->release();
        update_buffer->release();
    }));
}

void MetalAccel::_do_build(MetalCommandEncoder &encoder) noexcept {
    LUISA_ASSERT(_descriptor != nullptr, "Descriptor is not allocated.");
    LUISA_ASSERT(_instance_buffer != nullptr, "Instance buffer is not allocated.");
    auto device = _update->device();
    auto sizes = device->accelerationStructureSizes(_descriptor);
    if (_option.allow_update) {
        if (_update_buffer == nullptr ||
            _update_buffer->length() < sizes.refitScratchBufferSize) {
            if (_update_buffer != nullptr) { _update_buffer->release(); }
            _update_buffer = device->newBuffer(sizes.refitScratchBufferSize,
                                               MTL::ResourceStorageModePrivate |
                                                   MTL::ResourceHazardTrackingModeTracked);
        }
    }
    if (_handle != nullptr) { _handle->release(); }
    _handle = device->newAccelerationStructure(sizes.accelerationStructureSize);
    _handle->setLabel(_name);
    auto build_buffer = device->newBuffer(sizes.buildScratchBufferSize,
                                          MTL::ResourceStorageModePrivate |
                                              MTL::ResourceHazardTrackingModeTracked);
    auto command_encoder = encoder.command_buffer()->accelerationStructureCommandEncoder();
    _descriptor->retain();
    _handle->retain();
    command_encoder->buildAccelerationStructure(_handle, _descriptor, build_buffer, 0u);
    command_encoder->endEncoding();
    encoder.add_callback(FunctionCallbackContext::create([descriptor = _descriptor,
                                                          handle = _handle,
                                                          build_buffer] {
        descriptor->release();
        handle->release();
        build_buffer->release();
    }));

    // update the resources used by the acceleration structure
    _resources.clear();
    for (auto prim : _primitives) { prim->add_resources(_resources); }
    std::sort(_resources.begin(), _resources.end());
    _resources.erase(std::unique(_resources.begin(), _resources.end()), _resources.cend());

    // do compaction if required
    auto compacted_size = 0u;
    if (_option.allow_compaction) {
        encoder.with_download_buffer(sizeof(uint), [&](MetalStageBufferPool::Allocation *size_buffer) noexcept {
            // read back the size of the compacted acceleration structure
            auto compaction_size_encoder = encoder.command_buffer()->accelerationStructureCommandEncoder();
            compaction_size_encoder->writeCompactedAccelerationStructureSize(
                _handle, size_buffer->buffer(), size_buffer->offset(), MTL::DataTypeUInt);
            compaction_size_encoder->endEncoding();
            encoder.add_callback(FunctionCallbackContext::create([size_buffer, &compacted_size] {
                compacted_size = *reinterpret_cast<uint *>(size_buffer->data());
            }));
        });
        auto submitted_command_buffer = encoder.submit({});
        submitted_command_buffer->waitUntilCompleted();
        // compact the acceleration structure
        auto compacted_handle = device->newAccelerationStructure(compacted_size);
        compacted_handle->setLabel(_name);
        auto compact_encoder = encoder.command_buffer()->accelerationStructureCommandEncoder();
        compacted_handle->retain();
        compact_encoder->copyAndCompactAccelerationStructure(_handle, compacted_handle);
        compact_encoder->endEncoding();
        encoder.add_callback(FunctionCallbackContext::create([old_handle = _handle, compacted_handle] {
            old_handle->release();
            compacted_handle->release();
        }));
        _handle = compacted_handle;
    }
}

void MetalAccel::set_name(luisa::string_view name) noexcept {
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

void MetalAccel::mark_resource_usages(MetalCommandEncoder &encoder,
                                      MTL::ComputeCommandEncoder *command_encoder,
                                      MTL::ResourceUsage usage) noexcept {

    std::scoped_lock lock{_mutex};

    _descriptor->retain();
    _handle->retain();
    _instance_buffer->retain();
    encoder.add_callback(FunctionCallbackContext::create([descriptor = _descriptor,
                                                          handle = _handle,
                                                          instance_buffer = _instance_buffer] {
        descriptor->release();
        handle->release();
        instance_buffer->release();
    }));
    command_encoder->useResource(_handle, MTL::ResourceUsageRead);
    command_encoder->useResource(_instance_buffer, usage);
    // FIXME: This seems unnecessary according to the Metal profiling tool, but
    //  will cause some bottom-level acceleration structures disappear in tests.
    command_encoder->useResources(_resources.data(), _resources.size(), MTL::ResourceUsageRead);
}

}// namespace luisa::compute::metal

