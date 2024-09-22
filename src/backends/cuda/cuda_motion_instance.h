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
    size_t _motion_buffer_size{};
    CUdeviceptr _motion_buffer{};
    CUDAPrimitive *_child{};

public:
    CUDAMotionInstance(CUDADevice *device, const AccelMotionOption &option) noexcept;
    ~CUDAMotionInstance() noexcept override;
    void build(CUDACommandEncoder &encoder, MotionInstanceBuildCommand *command) noexcept;
    [[nodiscard]] auto child() const noexcept { return _child; }
};

}// namespace luisa::compute::cuda
