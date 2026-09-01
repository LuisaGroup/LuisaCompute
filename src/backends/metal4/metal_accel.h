#pragma once

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/command.h>
#include "../common/resource_tracker.h"
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalDevice;
class MetalPrimitive;
class MetalPrimitiveBase;
class MetalCommandEncoder;

class MetalAccel {

public:
    static constexpr auto reserved_primitive_count = 1024u;

private:
    enum class MotionMode : uint8_t {
        NONE,
        MATRIX,
        COMPONENT
    };

    struct alignas(8) Instance {
        MTL::PackedFloat4x3 transformation;
        MTL::AccelerationStructureInstanceOptions options;
        uint32_t mask;
        uint32_t intersection_function_offset;
        uint32_t user_id;
        MTL::ResourceID acceleration_structure_id;
    };
    static_assert(sizeof(Instance) == 72u);
    static_assert(
        sizeof(Instance) ==
        sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));
    static_assert(offsetof(Instance, transformation) ==
                  offsetof(MTL::IndirectAccelerationStructureInstanceDescriptor,
                           transformationMatrix));
    static_assert(offsetof(Instance, options) ==
                  offsetof(MTL::IndirectAccelerationStructureInstanceDescriptor,
                           options));
    static_assert(offsetof(Instance, mask) ==
                  offsetof(MTL::IndirectAccelerationStructureInstanceDescriptor,
                           mask));
    static_assert(offsetof(Instance, intersection_function_offset) ==
                  offsetof(MTL::IndirectAccelerationStructureInstanceDescriptor,
                           intersectionFunctionTableOffset));
    static_assert(offsetof(Instance, user_id) ==
                  offsetof(MTL::IndirectAccelerationStructureInstanceDescriptor,
                           userID));
    static_assert(offsetof(Instance, acceleration_structure_id) ==
                  offsetof(MTL::IndirectAccelerationStructureInstanceDescriptor,
                           accelerationStructureID));

    using MotionInstance =
        MTL::IndirectAccelerationStructureMotionInstanceDescriptor;
    static_assert(sizeof(MotionInstance) == 48u);
    static_assert(offsetof(MotionInstance, options) == 0u);
    static_assert(offsetof(MotionInstance, accelerationStructureID) == 16u);
    static_assert(offsetof(MotionInstance, motionTransformsStartIndex) == 24u);
    static_assert(offsetof(MotionInstance, motionTransformsCount) == 28u);
    static_assert(offsetof(MotionInstance, motionStartBorderMode) == 32u);
    static_assert(offsetof(MotionInstance, motionEndTime) == 44u);
    static_assert(sizeof(MTL::PackedFloat4x3) == 48u);
    static_assert(sizeof(MTL::ComponentTransform) == 64u);

    MTL::AccelerationStructure *_handle{nullptr};
    MTL::Buffer *_instance_buffer{nullptr};
    MTL::Buffer *_motion_instance_buffer{nullptr};
    MTL::Buffer *_motion_transform_buffer{nullptr};
    MTL::Buffer *_update_buffer{nullptr};
    MTL4::InstanceAccelerationStructureDescriptor *_descriptor{nullptr};
    MTL::InstanceAccelerationStructureDescriptor *_compatibility_descriptor{nullptr};
    MTL::ComputePipelineState *_update;
    luisa::vector<Instance> _instances{};
    luisa::vector<MetalPrimitiveBase *> _primitives{};
    luisa::vector<MTL::Resource *> _resources{};
    NS::String *_name{nullptr};
    AccelOption _option;
    MotionMode _motion_mode{MotionMode::NONE};
    size_t _motion_transform_count{0u};
    bool _requires_rebuild{true};
    bool _requires_extended_limits{false};
    spin_mutex _mutex;

public:
    struct Binding {
        MTL::ResourceID handle;
        uint64_t instance_buffer;
    };

private:
    void _do_build(MetalCommandEncoder &encoder) noexcept;
    void _do_update(MetalCommandEncoder &encoder) noexcept;
    void _do_build_compatibility(MetalCommandEncoder &encoder) noexcept;
    void _do_update_compatibility(MetalCommandEncoder &encoder) noexcept;
    void _prepare_motion_data(MetalCommandEncoder &encoder) noexcept;

public:
    MetalAccel(MetalDevice *device, const AccelOption &option) noexcept;
    ~MetalAccel() noexcept;
    void build(MetalCommandEncoder &encoder, AccelBuildCommand *command) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer; }
    [[nodiscard]] auto binding() const noexcept { return Binding{_handle->gpuResourceID(), _instance_buffer->gpuAddress()}; }
    [[nodiscard]] auto pointer_to_handle() const noexcept { return const_cast<void *>(static_cast<const void *>(&_handle)); }
    [[nodiscard]] auto requires_extended_limits() const noexcept { return _requires_extended_limits; }
    void set_name(luisa::string_view name) noexcept;
    void mark_resource_usages(MetalCommandEncoder &encoder) noexcept;
};

}// namespace luisa::compute::metal
