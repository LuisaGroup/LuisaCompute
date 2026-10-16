#include <luisa/core/logging.h>

#include "metal_command_encoder.h"
#include "metal_acceleration_structure_build.h"

namespace luisa::compute::metal {

namespace {

[[nodiscard]] NS::SharedPtr<MTL::CommandBuffer>
new_compatibility_command_buffer(MetalCommandEncoder &encoder) noexcept {
    encoder.submit_and_wait();
    auto queue =
        encoder.stream()->acceleration_structure_compatibility_queue();
    LUISA_ASSERT(queue != nullptr,
                 "Metal acceleration-structure compatibility queue is not "
                 "available on this device.");
    auto command_buffer = queue->commandBufferWithUnretainedReferences();
    LUISA_ASSERT(command_buffer != nullptr,
                 "Failed to create Metal acceleration-structure "
                 "compatibility command buffer.");
    return NS::RetainPtr(command_buffer);
}

void wait_compatibility_command_buffer(
    const NS::SharedPtr<MTL::CommandBuffer> &command_buffer) noexcept {
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    if (auto error = command_buffer->error(); error != nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Metal acceleration-structure compatibility command buffer "
            "failed: {}.",
            error->localizedDescription()->utf8String());
    }
    LUISA_ASSERT(
        command_buffer->status() == MTL::CommandBufferStatusCompleted,
        "Metal acceleration-structure compatibility command buffer did not "
        "complete successfully (status = {}).",
        static_cast<uint32_t>(command_buffer->status()));
}

void mark_indirect_resources(
    MTL::AccelerationStructureCommandEncoder *command_encoder,
    luisa::span<MTL::Resource *const> resources) noexcept {
    if (!resources.empty()) {
        command_encoder->useResources(
            resources.data(), resources.size(),
            MTL::ResourceUsageRead);
    }
}

}// namespace

void build_acceleration_structure_compatibility(
    MetalCommandEncoder &encoder,
    MTL::AccelerationStructureDescriptor *descriptor,
    bool allow_update,
    bool allow_compaction,
    NS::String *name,
    MTL::AccelerationStructure *&handle,
    MTL::Buffer *&update_buffer,
    luisa::span<MTL::Resource *const> indirect_resources) noexcept {
    LUISA_ASSERT(descriptor != nullptr,
                 "Invalid acceleration-structure compatibility descriptor.");
    auto device = encoder.device();
    auto sizes = device->accelerationStructureSizes(descriptor);
    if (allow_update &&
        (update_buffer == nullptr ||
         update_buffer->length() < sizes.refitScratchBufferSize)) {
        if (update_buffer != nullptr) { update_buffer->release(); }
        update_buffer = sizes.refitScratchBufferSize == 0u ?
                            nullptr :
                            device->newBuffer(
                                sizes.refitScratchBufferSize,
                                MTL::ResourceHazardTrackingModeTracked |
                                    MTL::ResourceStorageModePrivate);
    }

    auto replacement =
        device->newAccelerationStructure(sizes.accelerationStructureSize);
    auto scratch = device->newBuffer(
        sizes.buildScratchBufferSize,
        MTL::ResourceHazardTrackingModeTracked |
            MTL::ResourceStorageModePrivate);
    LUISA_ASSERT(replacement != nullptr && scratch != nullptr,
                 "Failed to allocate Metal acceleration-structure "
                 "compatibility build resources.");
    replacement->setLabel(name);

    auto compacted_size_buffer = allow_compaction ?
                                     device->newBuffer(
                                         sizeof(uint64_t),
                                         MTL::ResourceStorageModeShared |
                                             MTL::ResourceHazardTrackingModeTracked) :
                                     nullptr;
    auto command_buffer = new_compatibility_command_buffer(encoder);
    auto command_encoder =
        command_buffer->accelerationStructureCommandEncoder();
    LUISA_ASSERT(command_encoder != nullptr,
                 "Failed to create Metal acceleration-structure "
                 "compatibility encoder.");
    mark_indirect_resources(command_encoder, indirect_resources);
    command_encoder->buildAccelerationStructure(
        replacement, descriptor, scratch, 0u);
    if (compacted_size_buffer != nullptr) {
        command_encoder->writeCompactedAccelerationStructureSize(
            replacement, compacted_size_buffer, 0u,
            MTL::DataTypeULong);
    }
    command_encoder->endEncoding();
    wait_compatibility_command_buffer(command_buffer);
    scratch->release();

    if (compacted_size_buffer != nullptr) {
        auto compacted_size =
            *static_cast<const uint64_t *>(
                compacted_size_buffer->contents());
        auto compacted =
            device->newAccelerationStructure(compacted_size);
        LUISA_ASSERT(compacted != nullptr,
                     "Failed to allocate compacted Metal acceleration "
                     "structure.");
        compacted->setLabel(name);
        auto compact_command_buffer =
            new_compatibility_command_buffer(encoder);
        auto compact_encoder =
            compact_command_buffer->accelerationStructureCommandEncoder();
        LUISA_ASSERT(compact_encoder != nullptr,
                     "Failed to create Metal acceleration-structure "
                     "compatibility compaction encoder.");
        mark_indirect_resources(compact_encoder, indirect_resources);
        compact_encoder->copyAndCompactAccelerationStructure(
            replacement, compacted);
        compact_encoder->endEncoding();
        wait_compatibility_command_buffer(compact_command_buffer);
        replacement->release();
        replacement = compacted;
        compacted_size_buffer->release();
    }

    if (handle != nullptr) { handle->release(); }
    handle = replacement;
}

void refit_acceleration_structure_compatibility(
    MetalCommandEncoder &encoder,
    MTL::AccelerationStructureDescriptor *descriptor,
    MTL::AccelerationStructure *handle,
    MTL::Buffer *update_buffer,
    luisa::span<MTL::Resource *const> indirect_resources) noexcept {
    LUISA_ASSERT(descriptor != nullptr && handle != nullptr,
                 "Invalid acceleration-structure compatibility refit.");
    auto command_buffer = new_compatibility_command_buffer(encoder);
    auto command_encoder =
        command_buffer->accelerationStructureCommandEncoder();
    LUISA_ASSERT(command_encoder != nullptr,
                 "Failed to create Metal acceleration-structure "
                 "compatibility refit encoder.");
    mark_indirect_resources(command_encoder, indirect_resources);
    command_encoder->refitAccelerationStructure(
        handle, descriptor, handle, update_buffer, 0u);
    command_encoder->endEncoding();
    wait_compatibility_command_buffer(command_buffer);
}

}// namespace luisa::compute::metal
