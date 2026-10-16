#pragma once

#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rtx/motion_transform.h>
#include "metal_primitive.h"

namespace luisa::compute::metal {

class MetalMotionInstance final : public MetalPrimitiveBase {

public:
    struct Snapshot {
        AccelMotionOption option;
        MetalPrimitive *child;
        luisa::vector<MotionInstanceTransform> keyframes;
        uint64_t build_version;
    };

private:
    AccelMotionOption _option;
    MetalPrimitive *_child{nullptr};
    luisa::vector<MotionInstanceTransform> _keyframes;
    luisa::string _name;
    uint64_t _build_version{0u};
    mutable spin_mutex _mutex;

public:
    explicit MetalMotionInstance(const AccelMotionOption &option) noexcept;
    ~MetalMotionInstance() noexcept override = default;
    void build(MotionInstanceBuildCommand *command) noexcept;
    [[nodiscard]] const AccelMotionOption &option() const noexcept {
        return _option;
    }
    [[nodiscard]] Snapshot snapshot() const noexcept;
    [[nodiscard]] MTL::AccelerationStructure *handle() const noexcept override;
    void set_name(luisa::string_view name) noexcept override;
    void add_resources(luisa::vector<MTL::Resource *> &resources) noexcept override;
};

}// namespace luisa::compute::metal
