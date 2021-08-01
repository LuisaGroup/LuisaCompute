//
// Created by Mike on 7/30/2021.
//

#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_heap.h>

namespace luisa::compute::cuda {

CUdeviceptr CUDABuffer::handle() const noexcept {
    return _heap == nullptr ? _handle : _heap->_items[_index].buffer;
}

}// namespace luisa::compute::cuda
