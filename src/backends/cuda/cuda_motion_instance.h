//
// Created by Mike on 2024/9/21.
//

#pragma once

#include <cuda.h>

#include <luisa/runtime/rtx/motion_instance.h>

#include "optix_api.h"
#include "cuda_primitive.h"

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDACommandEncoder;

/**
 * @brief Mesh of CUDA
 *
 */
class CUDAMotionInstance : public CUDAPrimitiveBase {

private:
    AccelMotionOption _option;
    CUdeviceptr _motion_buffer{};

public:
    CUDAMotionInstance(CUDADevice *device, const AccelMotionOption &option) noexcept;
    ~CUDAMotionInstance() noexcept override;
    void build(CUDACommandEncoder &encoder, MotionInstanceBuildCommand *command) noexcept;
};

}// namespace luisa::compute::cuda
