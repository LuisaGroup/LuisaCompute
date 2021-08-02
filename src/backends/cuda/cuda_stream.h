//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <cuda.h>
#include <backends/cuda/cuda_ring_buffer.h>

namespace luisa::compute::cuda {

class CUDAStream {

private:
    CUstream _handle;
    CUDARingBuffer _upload_pool;

public:
    CUDAStream() noexcept;
    ~CUDAStream() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] decltype(auto) upload_pool() noexcept { return (_upload_pool); }
};

}
