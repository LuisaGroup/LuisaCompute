#pragma once

#include <luisa/core/spin_mutex.h>
#include <luisa/core/first_fit.h>
#include "metal_api.h"
#include "metal_callback_context.h"

namespace luisa::compute::metal {

class MetalStageBufferPool {

public:
    class Allocation final : public MetalCallbackContext {

        friend class MetalStageBufferPool;

    private:
        MetalStageBufferPool *_pool;
        union {
            MTL::Buffer *_buffer;
            FirstFit::Node *_node;
        };

    public:
        Allocation(MTL::Buffer *buffer) noexcept
            : _pool{nullptr}, _buffer{buffer} {}
        Allocation(MetalStageBufferPool *pool, FirstFit::Node *node) noexcept
            : _pool{pool}, _node{node} {}
        ~Allocation() noexcept override = default;
        [[nodiscard]] auto is_pooled() const noexcept { return _pool != nullptr; }
        [[nodiscard]] MTL::Buffer *buffer() const noexcept;
        [[nodiscard]] size_t offset() const noexcept;
        [[nodiscard]] size_t size() const noexcept;
        [[nodiscard]] std::byte *data() const noexcept;
        void recycle() noexcept override;
    };

private:
    FirstFit _first_fit;
    MTL::Buffer *_buffer;
    spin_mutex _mutex;

private:
    void _recycle(FirstFit::Node *alloc) noexcept;

public:
    MetalStageBufferPool(MTL::Device *device, size_t size, bool write_combined) noexcept;
    ~MetalStageBufferPool() noexcept;
    [[nodiscard]] Allocation *allocate(size_t size) noexcept;
};

}// namespace luisa::compute::metal

