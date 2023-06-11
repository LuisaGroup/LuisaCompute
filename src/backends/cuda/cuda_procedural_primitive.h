//
// Created by Mike on 3/27/2023.
//

#pragma once

#include <luisa/runtime/rtx/procedural_primitive.h>
#include <backends/cuda/optix_api.h>
#include <backends/cuda/cuda_primitive.h>

namespace luisa::compute::cuda {

class CUDACommandEncoder;

class CUDAProceduralPrimitive final : public CUDAPrimitive {

private:
    CUdeviceptr _aabb_buffer{};
    size_t _aabb_buffer_size{};

private:
    [[nodiscard]] optix::BuildInput _make_build_input() const noexcept override;

public:
    explicit CUDAProceduralPrimitive(const AccelOption &option) noexcept;
    ~CUDAProceduralPrimitive() noexcept override = default;
    void build(CUDACommandEncoder &encoder,
               ProceduralPrimitiveBuildCommand *command) noexcept;
};

}// namespace luisa::compute::cuda

