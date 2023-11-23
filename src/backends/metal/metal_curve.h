#pragma once

#include <luisa/runtime/rhi/command.h>
#include "metal_primitive.h"

namespace luisa::compute::metal {

class MetalCurve : public MetalPrimitive {

private:
    MTL::PrimitiveAccelerationStructureDescriptor *_descriptor{nullptr};

private:
    void _do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept override;

public:
    MetalCurve(MTL::Device *device, const AccelOption &option) noexcept;
    ~MetalCurve() noexcept override;
    void build(MetalCommandEncoder &encoder, CurveBuildCommand *command) noexcept;
};

}
