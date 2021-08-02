//
// Created by Mike on 8/1/2021.
//

#include <backends/cuda/cuda_texture.h>
#include <backends/cuda/cuda_heap.h>

namespace luisa::compute::cuda {

uint64_t CUDATexture::handle() const noexcept {
    return _heap == nullptr ? _handle : _heap->_items[_index].texture;
}

}
