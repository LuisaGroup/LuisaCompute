#pragma once

#include <luisa/core/stl/memory.h>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalCommandEncoder;

void build_acceleration_structure_compatibility(
    MetalCommandEncoder &encoder,
    MTL::AccelerationStructureDescriptor *descriptor,
    bool allow_update,
    bool allow_compaction,
    NS::String *name,
    MTL::AccelerationStructure *&handle,
    MTL::Buffer *&update_buffer,
    luisa::span<MTL::Resource *const> indirect_resources = {}) noexcept;

void refit_acceleration_structure_compatibility(
    MetalCommandEncoder &encoder,
    MTL::AccelerationStructureDescriptor *descriptor,
    MTL::AccelerationStructure *handle,
    MTL::Buffer *update_buffer,
    luisa::span<MTL::Resource *const> indirect_resources = {}) noexcept;

}// namespace luisa::compute::metal
