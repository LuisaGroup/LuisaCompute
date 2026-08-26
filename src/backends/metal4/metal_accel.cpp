#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include "metal_device.h"
#include "metal_command_encoder.h"
#include "metal_acceleration_structure_build.h"
#include "metal_primitive.h"
#include "metal_accel.h"

namespace luisa::compute::metal {

namespace {
constexpr auto metal_accel_default_instance_limit = 1u << 24u;
}// namespace

MetalAccel::MetalAccel(MetalDevice *device,
                       const AccelOption &option) noexcept
    : _update{device->builtin_update_accel_instances()},
      _option{option} {
    _resources.reserve(reserved_primitive_count);
}

MetalAccel::~MetalAccel() noexcept {
    if (_handle) { _handle->release(); }
    if (_instance_buffer) { _instance_buffer->release(); }
    if (_motion_instance_buffer) { _motion_instance_buffer->release(); }
    if (_motion_transform_buffer) { _motion_transform_buffer->release(); }
    if (_update_buffer) { _update_buffer->release(); }
    if (_descriptor) { _descriptor->release(); }
    if (_compatibility_descriptor) {
        _compatibility_descriptor->release();
    }
    if (_name) { _name->release(); }
}

void MetalAccel::build(MetalCommandEncoder &encoder,
                       AccelBuildCommand *command) noexcept {
    std::scoped_lock lock{_mutex};

    auto device = encoder.device();
    auto instance_count = command->instance_count();
    auto requires_extended_limits =
        instance_count > metal_accel_default_instance_limit;
    LUISA_ASSERT(instance_count > 0u,
                 "Empty acceleration structure is not allowed.");

    auto required_instance_buffer_size =
        static_cast<size_t>(instance_count) * sizeof(Instance);
    if (_instance_buffer == nullptr ||
        _instance_buffer->length() < required_instance_buffer_size) {
        auto old_instance_buffer = _instance_buffer;
        _instance_buffer = device->newBuffer(
            required_instance_buffer_size,
            MTL::ResourceStorageModePrivate |
                MTL::ResourceHazardTrackingModeTracked);
        LUISA_ASSERT(_instance_buffer != nullptr,
                     "Failed to allocate Metal4 acceleration-structure "
                     "instance buffer.");
        if (old_instance_buffer != nullptr) {
            auto copy_encoder = encoder.compute_encoder();
            copy_encoder->copyFromBuffer(
                old_instance_buffer, 0u, _instance_buffer, 0u,
                old_instance_buffer->length());
            encoder.use_resource(old_instance_buffer);
            encoder.use_resource(_instance_buffer);
            copy_encoder->endEncoding();
            encoder.add_callback(FunctionCallbackContext::create(
                [old_instance_buffer]() noexcept {
                    old_instance_buffer->release();
                }));
        }
    }

    auto address_driven =
        encoder.stream()->supports_address_driven_acceleration_structures();
    auto old_instance_count = address_driven ?
                                  (_descriptor == nullptr ?
                                       0u :
                                       _descriptor->instanceCount()) :
                                  (_compatibility_descriptor == nullptr ?
                                       0u :
                                       _compatibility_descriptor->instanceCount());
    _primitives.resize(instance_count);
    _instances.resize(instance_count);
    auto mods = command->modifications();
    if (auto n = static_cast<uint>(mods.size()); n != 0u) {
        using Mod = AccelBuildCommand::Modification;
        encoder.with_upload_buffer(
            static_cast<size_t>(n) * sizeof(Mod),
            [&](MetalStageBufferPool::Allocation *mod_buffer) noexcept {
                auto updates = reinterpret_cast<Mod *>(mod_buffer->data());
                for (auto i = 0u; i < n; i++) {
                    auto m = mods[i];
                    LUISA_ASSERT(m.index < instance_count,
                                 "Invalid acceleration-structure instance "
                                 "index {} for {} instances.",
                                 m.index, instance_count);
                    if (m.flags & Mod::flag_primitive) {
                        auto primitive =
                            reinterpret_cast<MetalPrimitiveBase *>(m.primitive);
                        LUISA_ASSERT(primitive != nullptr &&
                                         primitive->handle() != nullptr,
                                     "Invalid Metal primitive in instance {}.",
                                     m.index);
                        _requires_rebuild = true;
                        _primitives[m.index] = primitive;
                        auto resource_id =
                            primitive->handle()->gpuResourceID();
                        _instances[m.index].mesh_index = m.index;
                        _instances[m.index].acceleration_structure_id =
                            resource_id;
                        m.primitive =
                            luisa::bit_cast<uint64_t>(resource_id);
                    }
                    auto &instance = _instances[m.index];
                    if (m.flags & Mod::flag_visibility) {
                        instance.mask = m.vis_mask;
                    }
                    if (m.flags & Mod::flag_opaque) {
                        auto options = uint32_t{
                            MTL::AccelerationStructureInstanceOptionDisableTriangleCulling};
                        options |= static_cast<uint32_t>(
                            (m.flags & Mod::flag_opaque_on) ?
                                MTL::AccelerationStructureInstanceOptionOpaque :
                                MTL::AccelerationStructureInstanceOptionNonOpaque);
                        instance.options = static_cast<
                            MTL::AccelerationStructureInstanceOptions>(options);
                    }
                    if (m.flags & Mod::flag_transform) {
                        instance.transformation = MTL::PackedFloat4x3{
                            MTL::PackedFloat3{
                                m.affine[0u], m.affine[4u], m.affine[8u]},
                            MTL::PackedFloat3{
                                m.affine[1u], m.affine[5u], m.affine[9u]},
                            MTL::PackedFloat3{
                                m.affine[2u], m.affine[6u], m.affine[10u]},
                            MTL::PackedFloat3{
                                m.affine[3u], m.affine[7u], m.affine[11u]}};
                    }
                    if (m.flags & Mod::flag_user_id) {
                        instance.user_id = m.user_id;
                    }
                    updates[i] = m;
                }

                auto compute_encoder = encoder.compute_encoder();
                compute_encoder->setComputePipelineState(_update);
                auto table = encoder.argument_table(3u);
                table->setAddress(_instance_buffer->gpuAddress(), 0u);
                table->setAddress(
                    mod_buffer->buffer()->gpuAddress() +
                        mod_buffer->offset(),
                    1u);
                table->setAddress(encoder.upload(&n, sizeof(n)), 2u);
                compute_encoder->setArgumentTable(table);
                encoder.use_resource(_update);
                encoder.use_resource(_instance_buffer);
                encoder.use_resource(mod_buffer->buffer());
                constexpr auto block_size =
                    MetalDevice::update_accel_instances_block_size;
                auto block_count =
                    (n + block_size - 1u) / block_size;
                compute_encoder->dispatchThreadgroups(
                    MTL::Size{block_count, 1u, 1u},
                    MTL::Size{block_size, 1u, 1u});
                compute_encoder->endEncoding();
            });
    }

    _requires_rebuild =
        _requires_rebuild ||
        (address_driven ? _descriptor == nullptr :
                          _compatibility_descriptor == nullptr) ||
        old_instance_count != instance_count || _handle == nullptr ||
        _requires_extended_limits != requires_extended_limits ||
        !_option.allow_update ||
        command->request() == AccelBuildRequest::FORCE_BUILD;
    _requires_extended_limits = requires_extended_limits;

    encoder.use_resource(_instance_buffer);
    if (command->update_instance_buffer_only()) { return; }

    for (auto i = 0u; i < instance_count; i++) {
        LUISA_ASSERT(_primitives[i] != nullptr &&
                         _primitives[i]->handle() != nullptr,
                     "Metal4 acceleration-structure instance {} has no "
                     "valid primitive.",
                     i);
    }
    _prepare_motion_data(encoder);

    if (_requires_rebuild) {
        if (address_driven) {
            if (_descriptor != nullptr) { _descriptor->release(); }
            _descriptor =
                MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
        } else {
            if (_compatibility_descriptor != nullptr) {
                _compatibility_descriptor->release();
            }
            _compatibility_descriptor =
                MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
        }
    }
    auto usage = uint{0u};
    switch (_option.hint) {
        case AccelOption::UsageHint::FAST_TRACE:
            usage |= MTL::AccelerationStructureUsageNone;
            break;
        case AccelOption::UsageHint::FAST_BUILD:
            usage |= MTL::AccelerationStructureUsagePreferFastBuild;
            break;
    }
    if (_option.allow_update) {
        usage |= MTL::AccelerationStructureUsageRefit;
    }
    if (_requires_extended_limits) {
        usage |= MTL::AccelerationStructureUsageExtendedLimits;
    }
    auto descriptor_usage =
        static_cast<MTL::AccelerationStructureUsage>(usage);
    auto has_motion = _motion_mode != MotionMode::NONE;
    auto descriptor_instance_buffer =
        has_motion ? _motion_instance_buffer : _instance_buffer;
    auto descriptor_instance_buffer_size =
        has_motion ? static_cast<size_t>(instance_count) *
                         sizeof(MotionInstance) :
                     required_instance_buffer_size;
    auto descriptor_instance_stride =
        has_motion ? sizeof(MotionInstance) : sizeof(Instance);
    auto descriptor_instance_type =
        has_motion ?
            MTL::AccelerationStructureInstanceDescriptorTypeIndirectMotion :
            MTL::AccelerationStructureInstanceDescriptorTypeIndirect;
    auto motion_transform_stride =
        _motion_mode == MotionMode::MATRIX ?
            sizeof(MTL::PackedFloat4x3) :
            sizeof(MTL::ComponentTransform);
    auto motion_transform_type =
        _motion_mode == MotionMode::COMPONENT ?
            MTL::TransformTypeComponent :
            MTL::TransformTypePackedFloat4x3;
    if (address_driven) {
        _descriptor->setInstanceCount(instance_count);
        _descriptor->setInstanceDescriptorBuffer(MTL4::BufferRange{
            descriptor_instance_buffer->gpuAddress(),
            descriptor_instance_buffer_size});
        _descriptor->setInstanceDescriptorStride(descriptor_instance_stride);
        _descriptor->setInstanceDescriptorType(descriptor_instance_type);
        _descriptor->setInstanceTransformationMatrixLayout(
            MTL::MatrixLayoutColumnMajor);
        if (has_motion) {
            _descriptor->setMotionTransformBuffer(MTL4::BufferRange{
                _motion_transform_buffer->gpuAddress(),
                _motion_transform_count * motion_transform_stride});
            _descriptor->setMotionTransformCount(_motion_transform_count);
            _descriptor->setMotionTransformStride(motion_transform_stride);
            _descriptor->setMotionTransformType(motion_transform_type);
        }
        _descriptor->setUsage(descriptor_usage);
    } else {
        _compatibility_descriptor->setInstanceCount(instance_count);
        _compatibility_descriptor->setInstanceDescriptorBuffer(
            descriptor_instance_buffer);
        _compatibility_descriptor->setInstanceDescriptorBufferOffset(0u);
        _compatibility_descriptor->setInstanceDescriptorStride(
            descriptor_instance_stride);
        _compatibility_descriptor->setInstanceDescriptorType(
            descriptor_instance_type);
        _compatibility_descriptor->setInstanceTransformationMatrixLayout(
            MTL::MatrixLayoutColumnMajor);
        if (has_motion) {
            _compatibility_descriptor->setMotionTransformBuffer(
                _motion_transform_buffer);
            _compatibility_descriptor->setMotionTransformBufferOffset(0u);
            _compatibility_descriptor->setMotionTransformCount(
                _motion_transform_count);
            _compatibility_descriptor->setMotionTransformStride(
                motion_transform_stride);
            _compatibility_descriptor->setMotionTransformType(
                motion_transform_type);
        }
        _compatibility_descriptor->setUsage(descriptor_usage);
    }

    encoder.use_resource(descriptor_instance_buffer);
    encoder.use_resource(_motion_transform_buffer);
    if (address_driven) {
        if (_requires_rebuild) { _do_build(encoder); }
        else { _do_update(encoder); }
    } else {
        if (_requires_rebuild) {
            _do_build_compatibility(encoder);
        } else {
            _do_update_compatibility(encoder);
        }
    }
    _requires_rebuild = false;
}

void MetalAccel::_do_update(MetalCommandEncoder &encoder) noexcept {
    LUISA_ASSERT(_handle != nullptr,
                 "Acceleration structure is not built.");
    LUISA_ASSERT(_descriptor != nullptr,
                 "Descriptor is not allocated.");
    LUISA_ASSERT(_instance_buffer != nullptr,
                 "Instance buffer is not allocated.");

    auto command_encoder = encoder.compute_encoder();
    _descriptor->retain();
    _handle->retain();
    if (_update_buffer != nullptr) { _update_buffer->retain(); }
    command_encoder->refitAccelerationStructure(
        _handle, _descriptor, _handle,
        MTL4::BufferRange{
            _update_buffer == nullptr ? 0u : _update_buffer->gpuAddress(),
            _update_buffer == nullptr ? 0u : _update_buffer->length()});
    encoder.use_resource(_handle);
    encoder.use_resource(_instance_buffer);
    encoder.use_resource(_motion_instance_buffer);
    encoder.use_resource(_motion_transform_buffer);
    encoder.use_resource(_update_buffer);
    for (auto resource : _resources) { encoder.use_resource(resource); }
    command_encoder->endEncoding();
    encoder.add_callback(FunctionCallbackContext::create(
        [descriptor = _descriptor, handle = _handle,
         update_buffer = _update_buffer]() noexcept {
            descriptor->release();
            handle->release();
            if (update_buffer != nullptr) { update_buffer->release(); }
        }));
}

void MetalAccel::_do_build(MetalCommandEncoder &encoder) noexcept {
    LUISA_ASSERT(_descriptor != nullptr,
                 "Descriptor is not allocated.");
    LUISA_ASSERT(_instance_buffer != nullptr,
                 "Instance buffer is not allocated.");

    _resources.clear();
    for (auto primitive : _primitives) {
        primitive->add_resources(_resources);
    }
    if (_motion_instance_buffer != nullptr) {
        _resources.emplace_back(_motion_instance_buffer);
    }
    if (_motion_transform_buffer != nullptr) {
        _resources.emplace_back(_motion_transform_buffer);
    }
    luisa::sort(_resources.begin(), _resources.end());
    _resources.erase(
        std::unique(_resources.begin(), _resources.end()),
        _resources.cend());

    auto device = _update->device();
    auto sizes = device->accelerationStructureSizes(_descriptor);
    if (_option.allow_update &&
        (_update_buffer == nullptr ||
         _update_buffer->length() < sizes.refitScratchBufferSize)) {
        if (_update_buffer != nullptr) { _update_buffer->release(); }
        _update_buffer = sizes.refitScratchBufferSize == 0u ?
                             nullptr :
                             device->newBuffer(
                                 sizes.refitScratchBufferSize,
                                 MTL::ResourceStorageModePrivate |
                                     MTL::ResourceHazardTrackingModeTracked);
    }
    if (_handle != nullptr) { _handle->release(); }
    _handle =
        device->newAccelerationStructure(sizes.accelerationStructureSize);
    _handle->setLabel(_name);
    auto build_buffer = device->newBuffer(
        sizes.buildScratchBufferSize,
        MTL::ResourceStorageModePrivate |
            MTL::ResourceHazardTrackingModeTracked);
    LUISA_ASSERT(_handle != nullptr && build_buffer != nullptr,
                 "Failed to allocate Metal4 acceleration structure.");

    auto command_encoder = encoder.compute_encoder();
    _descriptor->retain();
    _handle->retain();
    command_encoder->buildAccelerationStructure(
        _handle, _descriptor,
        MTL4::BufferRange{build_buffer->gpuAddress(),
                          build_buffer->length()});
    encoder.use_resource(_handle);
    encoder.use_resource(_instance_buffer);
    encoder.use_resource(_motion_instance_buffer);
    encoder.use_resource(_motion_transform_buffer);
    encoder.use_resource(build_buffer);
    for (auto resource : _resources) { encoder.use_resource(resource); }
    command_encoder->endEncoding();
    encoder.add_callback(FunctionCallbackContext::create(
        [descriptor = _descriptor, handle = _handle,
         build_buffer]() noexcept {
            descriptor->release();
            handle->release();
            build_buffer->release();
        }));

    if (_option.allow_compaction) {
        auto compacted_size = uint64_t{0u};
        encoder.with_download_buffer(
            sizeof(compacted_size),
            [&](MetalStageBufferPool::Allocation *size_buffer) noexcept {
                auto size_encoder = encoder.compute_encoder();
                size_encoder->writeCompactedAccelerationStructureSize(
                    _handle,
                    MTL4::BufferRange{
                        size_buffer->buffer()->gpuAddress() +
                            size_buffer->offset(),
                        sizeof(compacted_size)});
                encoder.use_resource(_handle);
                encoder.use_resource(size_buffer->buffer());
                size_encoder->endEncoding();
                encoder.add_callback(FunctionCallbackContext::create(
                    [size_buffer, &compacted_size]() noexcept {
                        compacted_size =
                            *reinterpret_cast<uint64_t *>(
                                size_buffer->data());
                    }));
            });
        encoder.submit_and_wait();

        auto compacted_handle =
            device->newAccelerationStructure(compacted_size);
        LUISA_ASSERT(compacted_handle != nullptr,
                     "Failed to allocate compacted Metal4 acceleration "
                     "structure.");
        compacted_handle->setLabel(_name);
        auto compact_encoder = encoder.compute_encoder();
        compacted_handle->retain();
        compact_encoder->copyAndCompactAccelerationStructure(
            _handle, compacted_handle);
        encoder.use_resource(_handle);
        encoder.use_resource(compacted_handle);
        compact_encoder->endEncoding();
        encoder.add_callback(FunctionCallbackContext::create(
            [old_handle = _handle,
             compacted_handle]() noexcept {
                old_handle->release();
                compacted_handle->release();
            }));
        _handle = compacted_handle;
    }
}

void MetalAccel::_do_build_compatibility(
    MetalCommandEncoder &encoder) noexcept {
    LUISA_ASSERT(_compatibility_descriptor != nullptr &&
                     _instance_buffer != nullptr,
                 "Metal acceleration-structure compatibility descriptor is "
                 "not initialized.");
    _resources.clear();
    for (auto primitive : _primitives) {
        primitive->add_resources(_resources);
    }
    if (_motion_instance_buffer != nullptr) {
        _resources.emplace_back(_motion_instance_buffer);
    }
    if (_motion_transform_buffer != nullptr) {
        _resources.emplace_back(_motion_transform_buffer);
    }
    luisa::sort(_resources.begin(), _resources.end());
    _resources.erase(
        std::unique(_resources.begin(), _resources.end()),
        _resources.cend());
    build_acceleration_structure_compatibility(
        encoder, _compatibility_descriptor, _option.allow_update,
        _option.allow_compaction, _name, _handle, _update_buffer,
        {_resources.data(), _resources.size()});
}

void MetalAccel::_do_update_compatibility(
    MetalCommandEncoder &encoder) noexcept {
    LUISA_ASSERT(_compatibility_descriptor != nullptr &&
                     _handle != nullptr && _instance_buffer != nullptr,
                 "Metal acceleration-structure compatibility refit is not "
                 "initialized.");
    refit_acceleration_structure_compatibility(
        encoder, _compatibility_descriptor, _handle, _update_buffer,
        {_resources.data(), _resources.size()});
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

void MetalAccel::mark_resource_usages(
    MetalCommandEncoder &encoder) noexcept {
    std::scoped_lock lock{_mutex};
    auto descriptor = static_cast<MTL::AccelerationStructureDescriptor *>(
        _descriptor != nullptr ?
            static_cast<MTL::AccelerationStructureDescriptor *>(_descriptor) :
            _compatibility_descriptor);
    LUISA_ASSERT(descriptor != nullptr && _handle != nullptr &&
                     _instance_buffer != nullptr,
                 "Metal acceleration structure has not been built.");

    descriptor->retain();
    _handle->retain();
    _instance_buffer->retain();
    encoder.add_callback(FunctionCallbackContext::create(
        [descriptor, handle = _handle,
         instance_buffer = _instance_buffer]() noexcept {
            descriptor->release();
            handle->release();
            instance_buffer->release();
        }));
    encoder.use_resource(_handle);
    encoder.use_resource(_instance_buffer);
    encoder.use_resource(_motion_instance_buffer);
    encoder.use_resource(_motion_transform_buffer);
    for (auto resource : _resources) { encoder.use_resource(resource); }
}

}// namespace luisa::compute::metal
