//
// Created by Mike on 2021/12/2.
//

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_accel.h>

namespace luisa::compute::cuda {

CUDAAccel::CUDAAccel(AccelBuildHint hint) noexcept
    : _build_hint{hint} {
    LUISA_CHECK_CUDA(cuEventCreate(
        &_update_event,
        CU_EVENT_DISABLE_TIMING));
}

CUDAAccel::~CUDAAccel() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_instance_buffer));
    LUISA_CHECK_CUDA(cuMemFree(_update_buffer));
    LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer));
    LUISA_CHECK_CUDA(cuEventDestroy(_update_event));
}

}
