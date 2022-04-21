//
// Created by Mike on 8/1/2021.
//

#include <mutex>

#include <core/logging.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_callback_context.h>
#include <backends/cuda/cuda_device.h>

namespace luisa::compute::cuda {

CUDAStream::CUDAStream(CUDADevice *device) noexcept
    : _device{device}, _upload_pool{64_mb, true} {
    for (auto i = 0u; i < backed_cuda_stream_count; i++) {
        LUISA_CHECK_CUDA(cuStreamCreate(&_worker_streams[i], CU_STREAM_NON_BLOCKING));
        LUISA_CHECK_CUDA(cuEventCreate(&_worker_events[i], CU_EVENT_DISABLE_TIMING));
    }
}

CUDAStream::~CUDAStream() noexcept {
    for (auto i = 0u; i < backed_cuda_stream_count; i++) {
        LUISA_CHECK_CUDA(cuStreamDestroy(_worker_streams[i]));
        LUISA_CHECK_CUDA(cuEventDestroy(_worker_events[i]));
    }
}

void CUDAStream::emplace_callback(CUDACallbackContext *cb) noexcept {
    if (cb != nullptr) { _current_callbacks.emplace_back(cb); }
}

void CUDAStream::barrier() noexcept {
    // wait for all streams to finish
    _used_streams &= ~1u;
    if (_used_streams != 0u) {
        for (auto i = 1u; i < backed_cuda_stream_count; i++) {
            if (_used_streams & (1u << i)) {
                LUISA_CHECK_CUDA(cuEventRecord(_worker_events[i], _worker_streams[i]));
                LUISA_CHECK_CUDA(cuStreamWaitEvent(_worker_streams[0], _worker_events[i], 0));
            }
        }
    }
    LUISA_CHECK_CUDA(cuEventRecord(_worker_events[0], _worker_streams[0]));
    _round = 0u;
    _used_streams = 0u;
}

void CUDAStream::synchronize() noexcept {
    _round = 0u;
    _used_streams = 0u;
    LUISA_CHECK_CUDA(cuStreamSynchronize(_worker_streams.front()));
}

void CUDAStream::dispatch(luisa::move_only_function<void()> &&f) noexcept {
    auto ptr = new_with_allocator<luisa::move_only_function<void()>>(std::move(f));
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        _worker_streams.front(), [](void *ptr) noexcept {
            auto func = static_cast<luisa::move_only_function<void()> *>(ptr);
            (*func)();
            luisa::delete_with_allocator(func);
        },
        ptr));
    _round = 0u;
    _used_streams = 0u;
}

CUstream CUDAStream::handle(bool force_first_stream) const noexcept {
    if (force_first_stream) {
        if (_round == 0u) { _round = 1u % backed_cuda_stream_count; }
        return _worker_streams.front();
    }
    auto index = _round;
    auto stream_bit = 1u << index;
    if (index != 0u && (_used_streams & stream_bit) == 0u) {
        _used_streams |= stream_bit;
        LUISA_CHECK_CUDA(cuStreamWaitEvent(_worker_streams[index], _worker_events[0], 0));
    }
    _round = (_round + 1u) % backed_cuda_stream_count;
    return _worker_streams[index];
}

void CUDAStream::dispatch_callbacks() noexcept {
    if (_current_callbacks.empty()) { return; }
    luisa::vector<CUDACallbackContext *> callbacks;
    callbacks.swap(_current_callbacks);
    std::scoped_lock lock{_mutex};
    _callback_lists.emplace(std::move(callbacks));
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        _worker_streams.front(), [](void *p) noexcept {
            constexpr auto pop = [](auto stream) -> luisa::vector<CUDACallbackContext *> {
                std::scoped_lock lock{stream->_mutex};
                if (stream->_callback_lists.empty()) [[unlikely]] {
                    LUISA_WARNING_WITH_LOCATION(
                        "Fetching stream callback from empty queue.");
                    return {};
                }
                auto callbacks = std::move(stream->_callback_lists.front());
                stream->_callback_lists.pop();
                return callbacks;
            };
            auto stream = static_cast<CUDAStream *>(p);
            auto callbacks = pop(stream);
            for (auto callback : callbacks) {
                callback->recycle();
            }
        },
        this));
}

}// namespace luisa::compute::cuda
