//
// Created by Mike on 8/1/2021.
//

#include "backends/cuda/cuda_error.h"
#include "core/logging.h"
#include <mutex>

#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_callback_context.h>

namespace luisa::compute::cuda {

CUDAStream::CUDAStream() noexcept
    : _upload_pool{64_mb, true} {
    for (auto i = 0u; i < backed_cuda_stream_count; i++) {
        LUISA_CHECK_CUDA(cuStreamCreate(
            &_worker_streams[i], CU_STREAM_NON_BLOCKING));
        LUISA_CHECK_CUDA(cuEventCreate(
            &_worker_events[i], CU_EVENT_DISABLE_TIMING));
        LUISA_INFO("Event #{}: {}", i, static_cast<void *>(_worker_events[i]));
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
    auto count = 0u;
    for (auto i = 0u; i < backed_cuda_stream_count; i++) {
        if (_used_streams.test(i)) {
            count++;
            auto event = _worker_events[i];
            constexpr auto flags = CU_EVENT_WAIT_DEFAULT;
            LUISA_CHECK_CUDA(cuEventRecord(event, _worker_streams[i]));
            LUISA_CHECK_CUDA(cuStreamWaitEvent(_worker_streams.front(), event, flags));
        }
    }
    if (count > 1u) {
        LUISA_VERBOSE_WITH_LOCATION(
            "Active concurrent CUDA streams: {}.", count);
    }
    _round = 0u;
    _used_streams.reset();
}

CUstream CUDAStream::handle(bool force_first_stream) const noexcept {
    if (force_first_stream) {
        if (_round == 0u) { _round = 1u % backed_cuda_stream_count; }
        _used_streams.set(0u);
        return _worker_streams.front();
    }
    auto index = _round;
    auto stream = _worker_streams[index];
    if (index != 0u && !_used_streams.test(index)) {
        LUISA_CHECK_CUDA(cuStreamWaitEvent(
            stream, _worker_events.front(),
            CU_EVENT_WAIT_DEFAULT));
    }
    _used_streams.set(index);
    _round = (_round + 1u) % backed_cuda_stream_count;
    return stream;
}

void CUDAStream::dispatch_callbacks() noexcept {
    if (_current_callbacks.empty()) { return; }
    auto callbacks = std::move(_current_callbacks);
    _current_callbacks = {};
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
