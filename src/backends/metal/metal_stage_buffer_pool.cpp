#include <luisa/core/pool.h>
#include "metal_stage_buffer_pool.h"

namespace luisa::compute::metal {

[[nodiscard]] static auto metal_stage_buffer_allocation_node_pool() noexcept {
    static Pool<MetalStageBufferPool::Allocation, true> pool;
    return &pool;
}

void MetalStageBufferPool::Allocation::recycle() noexcept {
    if (is_pooled()) {
        _pool->_recycle(_node);
    } else {
        _buffer->release();
    }
    metal_stage_buffer_allocation_node_pool()->destroy(this);
}

MTL::Buffer *MetalStageBufferPool::Allocation::buffer() const noexcept {
    return is_pooled() ? _pool->_buffer : _buffer;
}

size_t MetalStageBufferPool::Allocation::offset() const noexcept {
    return is_pooled() ? _node->offset() : 0u;
}

size_t MetalStageBufferPool::Allocation::size() const noexcept {
    return is_pooled() ? _node->size() : _buffer->length();
}

std::byte *MetalStageBufferPool::Allocation::data() const noexcept {
    return static_cast<std::byte *>(buffer()->contents()) + offset();
}

[[nodiscard]] inline auto metal_stage_buffer_options(bool write_combined) noexcept {
    return write_combined ?
               MTL::ResourceCPUCacheModeWriteCombined |
                   MTL::ResourceHazardTrackingModeTracked |
                   MTL::ResourceStorageModeShared :
               MTL::ResourceHazardTrackingModeTracked |
                   MTL::ResourceStorageModeShared;
}

MetalStageBufferPool::MetalStageBufferPool(MTL::Device *device,
                                           size_t size,
                                           bool write_combined) noexcept
    : _first_fit{size, 16u},
      _buffer{device->newBuffer(size, metal_stage_buffer_options(write_combined))} {}

MetalStageBufferPool::~MetalStageBufferPool() noexcept {
    _buffer->release();
}

MetalStageBufferPool::Allocation *MetalStageBufferPool::allocate(size_t size) noexcept {
    auto pool_node = [this, size] {
        std::scoped_lock lock{_mutex};
        return _first_fit.allocate(size);
    }();
    if (pool_node != nullptr) {
        return metal_stage_buffer_allocation_node_pool()->create(this, pool_node);
    }
    auto temp_buffer = _buffer->device()->newBuffer(size, _buffer->resourceOptions());
    return metal_stage_buffer_allocation_node_pool()->create(temp_buffer);
}

void MetalStageBufferPool::_recycle(FirstFit::Node *alloc) noexcept {
    std::scoped_lock lock{_mutex};
    _first_fit.free(alloc);
}

}// namespace luisa::compute::metal

