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

void CUDAAccel::add_instance(CUDAMesh *mesh, float4x4 transform) noexcept {
    _instance_meshes.emplace_back(mesh);
    _instance_transforms.emplace_back(transform);
}

void CUDAAccel::set_transform(size_t index, float4x4 transform) noexcept {
    _instance_transforms[index] = transform;
    _dirty_range.mark(index);
}

bool CUDAAccel::uses_buffer(CUdeviceptr handle) const noexcept {
    return std::binary_search(
        _resource_buffers.cbegin(),
        _resource_buffers.cend(),
        handle);
}

void CUDAAccel::build(CUDADevice *device, CUDAStream *stream) noexcept {
    // TODO...
    _dirty_range.clear();
}

void CUDAAccel::update(CUDADevice *device, CUDAStream *stream) noexcept {

    if (!_dirty_range.empty()) {
        auto dirty_update_buffer_size = _dirty_range.size();

        _dirty_range.clear();
    }
}

}
