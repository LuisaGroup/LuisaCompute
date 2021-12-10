//
// Created by Mike on 2021/12/10.
//

#pragma once

#include <bitset>

#include <cuda.h>
#include <backends/cuda/cuda_error.h>

namespace luisa::compute::cuda {

class CUDAHeap {

public:
    struct Node {
        Node *next;
        size_t offset;
        size_t size;
    };

    static constexpr auto alignment = 256u;

private:
    CUdeviceptr _memory{0u};
    Node *_free_list{nullptr};

public:


};

}
