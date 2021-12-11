//
// Created by Mike on 2021/12/10.
//

#pragma once

#include <cuda.h>

#include <core/spin_mutex.h>
#include <core/allocator.h>
#include <core/first_fit.h>
#include <core/pool.h>
#include <core/basic_types.h>
#include <backends/cuda/cuda_callback_context.h>

namespace luisa::compute::cuda {

class CUDAHeap {

public:
    struct SmallBuffer {
        CUdeviceptr address{};
        FirstFit *list{nullptr};
        FirstFit::Node *node{nullptr};
        Pool<SmallBuffer> *pool{nullptr};
    };

    class BufferFreeContext : public CUDACallbackContext {

    private:
        CUDAHeap *_heap{nullptr};
        uint64_t _buffer{};

    private:
        BufferFreeContext(CUDAHeap *heap, uint64_t buffer) noexcept
            : _heap{heap}, _buffer{buffer} {}

    public:
        [[nodiscard]] static BufferFreeContext *create(CUDAHeap *heap, uint64_t buffer) noexcept;
        void recycle() noexcept override;
    };

public:
    static constexpr auto small_buffer_alignment = 256u;
    static constexpr auto small_buffer_pool_size = 64_mb;
    static constexpr auto small_buffer_size_threshold = 4_mb;

private:
    luisa::vector<CUdeviceptr> _pool_buffers;
    luisa::vector<luisa::unique_ptr<FirstFit>> _lists;
    size_t _last_tried_list = 0u;
    luisa::vector<CUdeviceptr> _native_buffers_to_free;
    spin_mutex _mutex;

public:
    CUDAHeap() noexcept =default;
    ~CUDAHeap() noexcept;
    CUDAHeap(CUDAHeap &&) noexcept = delete;
    CUDAHeap(const CUDAHeap &) noexcept = delete;
    CUDAHeap &operator=(CUDAHeap &&) noexcept = delete;
    CUDAHeap &operator=(const CUDAHeap &) noexcept = delete;
    [[nodiscard]] uint64_t allocate(size_t size) noexcept;
    void free(uint64_t buffer) noexcept;
    [[nodiscard]] static CUdeviceptr buffer_address(uint64_t handle) noexcept;
};

}
