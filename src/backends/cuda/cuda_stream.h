//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <mutex>

#include <cuda.h>

#include <core/stl.h>
#include <core/spin_mutex.h>
#include <backends/cuda/cuda_callback_context.h>
#include <backends/cuda/cuda_host_buffer_pool.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDACallbackContext;

/**
 * @brief Stream on CUDA
 * 
 */
class CUDAStream {

public:
    static constexpr auto backed_cuda_stream_count = 1u;// TODO: enable multiple streams when IR analysis is ready
    static_assert(backed_cuda_stream_count <= 32u);     // limit of uint bits

private:
    CUDADevice *_device;
    CUDAHostBufferPool _upload_pool;
    std::mutex _mutex;
    luisa::queue<luisa::vector<CUDACallbackContext *>> _callback_lists;
    luisa::vector<CUDACallbackContext *> _current_callbacks;
    std::array<CUstream, backed_cuda_stream_count> _worker_streams{};
    std::array<CUevent, backed_cuda_stream_count> _worker_events{};
    mutable uint _used_streams{0u};
    mutable uint _round{0u};

public:
    explicit CUDAStream(CUDADevice *device) noexcept;
    ~CUDAStream() noexcept;
    /**
     * @brief Return handle of a CUDA worker stream
     *
     * @param force_first_stream enforce to get the first (main) worker stream or not
     * @return CUstream
     */
    [[nodiscard]] CUstream handle(bool force_first_stream = false) const noexcept;
    /**
     * @brief Return CUDAHostBufferPool
     * 
     * @return address of CUDAHostBufferPool
     */
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    /**
     * @brief Emplace callback context
     * 
     * @param cb callback context
     */
    void emplace_callback(CUDACallbackContext *cb) noexcept;
    /**
     * @brief Insert barrier into the main worker stream
     */
    void barrier() noexcept;
    /**
     * @brief Dispatch callbacks
     */
    void dispatch_callbacks() noexcept;
    /**
     * @brief Synchronize
     */
    void synchronize() noexcept;
    /**
     * @brief Dispatch a host function in the stream
     */
    void dispatch(luisa::move_only_function<void()> &&f) noexcept;
};

}// namespace luisa::compute::cuda
