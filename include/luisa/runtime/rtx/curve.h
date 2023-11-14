#pragma once

#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/rhi/curve_basis.h>

namespace luisa::compute {

class LC_RUNTIME_API Curve final : public Resource {

public:
    using BuildRequest = AccelBuildRequest;

private:
    CurveBasis _basis{};
    size_t _cp_count{};
    size_t _seg_count{};
    // control point buffer
    uint64_t _cp_buffer{};
    size_t _cp_buffer_offset{};
    size_t _cp_stride{};
    // radius buffer
    uint64_t _radius_buffer{};
    size_t _radius_buffer_offset{};
    size_t _radius_stride{};
    // segment buffer
    uint64_t _seg_buffer{};
    size_t _seg_buffer_offset{};

public:


};

}// namespace luisa::compute