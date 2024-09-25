#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/resource.h>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalCommandEncoder;

class MetalPrimitive {

public:
    static constexpr auto max_motion_keyframe_count = 64u;

private:
    MTL::AccelerationStructure *_handle{nullptr};
    MTL::Buffer *_update_buffer{nullptr};
    NS::String *_name{nullptr};
    AccelOption _option;
    spin_mutex _mutex;

private:
    virtual void _do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept = 0;

protected:
    void _do_build(MetalCommandEncoder &encoder, MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept;
    void _do_update(MetalCommandEncoder &encoder, MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept;
    void _set_motion_options(MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept;
    [[nodiscard]] auto &mutex() noexcept { return _mutex; }

public:
    MetalPrimitive(MTL::Device *device, const AccelOption &option) noexcept;
    virtual ~MetalPrimitive() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto option() const noexcept { return _option; }
    [[nodiscard]] MTL::AccelerationStructureUsage usage() const noexcept;
    [[nodiscard]] auto pointer_to_handle() const noexcept { return const_cast<void *>(static_cast<const void *>(&_handle)); }
    [[nodiscard]] auto motion_keyframe_count() const noexcept { return std::max<uint>(1u, _option.motion.keyframe_count); }
    void set_name(luisa::string_view name) noexcept;
    void add_resources(luisa::vector<MTL::Resource *> &resources) noexcept;
};

}// namespace luisa::compute::metal
