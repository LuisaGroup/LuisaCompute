//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <backends/cuda/cuda_error.h>

namespace luisa::compute::cuda {

class CUDAHeap;

class CUDABuffer {

private:
    CUdeviceptr _handle;
    CUDAHeap *_heap{nullptr};

public:


};


}
