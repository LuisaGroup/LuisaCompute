#pragma once

#include <luisa/runtime/rhi/command.h>
#include "metal_primitive.h"

namespace luisa::compute::metal {

class MetalCurve : public MetalPrimitive {

private:
    MTL4::PrimitiveAccelerationStructureDescriptor *_descriptor{nullptr};
    MTL::PrimitiveAccelerationStructureDescriptor *_compatibility_descriptor{nullptr};
    MTL::Buffer *_control_point_buffer{nullptr};
    MTL::Buffer *_segment_buffer{nullptr};
    size_t _control_point_buffer_offset{0u};
    size_t _control_point_count{0u};
    size_t _control_point_stride{0u};
    size_t _segment_buffer_offset{0u};
    size_t _segment_count{0u};
    MTL::CurveBasis _basis{MTL::CurveBasisLinear};
    MTL::CurveEndCaps _end_caps{MTL::CurveEndCapsNone};

private:
    void _do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept override;

public:
    MetalCurve(MTL::Device *device, const AccelOption &option) noexcept;
    ~MetalCurve() noexcept override;
    void build(MetalCommandEncoder &encoder, CurveBuildCommand *command) noexcept;
};

}
