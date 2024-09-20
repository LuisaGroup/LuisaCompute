#pragma once

#include <cuda.h>

#include <luisa/runtime/rtx/curve.h>
#include "optix_api.h"
#include "cuda_primitive.h"

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDACommandEncoder;

class CUDACurve final : public CUDAPrimitive {

private:
    optix::PrimitiveType _basis{};
    size_t _cp_count{};
    size_t _seg_count{};
    CUdeviceptr _cp_buffer{};
    size_t _cp_stride{};
    CUdeviceptr _seg_buffer{};

private:
    [[nodiscard]] optix::BuildInput _make_build_input() const noexcept override;

public:
    explicit CUDACurve(const AccelOption &option) noexcept;
    ~CUDACurve() noexcept override = default;
    void build(CUDACommandEncoder &encoder,
               CurveBuildCommand *command) noexcept;

public:
    [[nodiscard]] auto basis() const noexcept { return _basis; }
};

}// namespace luisa::compute::cuda
