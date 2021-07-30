//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <cuda.h>
#include <backends/cuda/cuda_error.h>

namespace luisa::compute::cuda {

class CUDADevice;

class CUDAHeap {

private:
    CUDADevice *_device;
    CUmemoryPool _handle;


public:


};


}
