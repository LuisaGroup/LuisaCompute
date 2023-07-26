#pragma once

#include <cstddef>
#include <span>

#include <cuda.h>

#include <luisa/core/pool.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/first_fit.h>
#include "cuda_error.h"
#include "cuda_callback_context.h"

namespace luisa::compute::cuda {

// TODO: finish doc
/**
 * @brief Host buffer pool of CUDA
 * 
 */
class CUDAHostBufferPool {

public:
    static constexpr auto alignment = 16u;

public:
    class View final : public CUDACallbackContext {

    private:
        void *_handle;
        CUDAHostBufferPool *_pool{nullptr};

    public:
        explicit View(std::byte *handle) noexcept;
        View(FirstFit::Node *node, CUDAHostBufferPool *pool) noexcept;
        [[nodiscard]] auto is_pooled() const noexcept { return _pool != nullptr; }
        [[nodiscard]] auto node() const noexcept { return static_cast<FirstFit::Node *>(_handle); }
        [[nodiscard]] std::byte *address() const noexcept;
        [[nodiscard]] static View *create(std::byte *handle) noexcept;
        [[nodiscard]] static View *create(FirstFit::Node *node, CUDAHostBufferPool *pool) noexcept;
        void recycle() noexcept override;
    };

private:
    spin_mutex _mutex;
    std::byte *_memory{nullptr};
    FirstFit _first_fit;

public:
    CUDAHostBufferPool(size_t size, bool write_combined) noexcept;
    ~CUDAHostBufferPool() noexcept;
    [[nodiscard]] std::byte *memory() const noexcept { return _memory; }
    [[nodiscard]] View *allocate(size_t size, bool fallback_if_failed = true) noexcept;
    void recycle(FirstFit::Node *node) noexcept;
};

}// namespace luisa::compute::cuda

