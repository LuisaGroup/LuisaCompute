//
// Created by Mike on 8/1/2021.
//

#include <mutex>

#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_callback_context.h>

namespace luisa::compute::cuda {

CUDAStream::CUDAStream() noexcept
    : _upload_pool{64_mb, true} {
    for (auto i = 0u; i < backed_cuda_stream_count; i++) {
        LUISA_CHECK_CUDA(cuStreamCreate(&_worker_streams[i], CU_STREAM_NON_BLOCKING));
    }
    if constexpr (backed_cuda_stream_count > 1u) {
        _worker_events.resize(backed_cuda_event_count);
        for (auto i = 0u; i < backed_cuda_event_count; i++) {
            LUISA_CHECK_CUDA(cuEventCreate(&_worker_events[i], CU_EVENT_DISABLE_TIMING));
        }
    }
}

CUDAStream::~CUDAStream() noexcept {
    for (auto s : _worker_streams) { LUISA_CHECK_CUDA(cuStreamDestroy(s)); }
    for (auto e : _worker_events) { LUISA_CHECK_CUDA(cuEventDestroy(e)); }
}

void CUDAStream::emplace_callback(CUDACallbackContext *cb) noexcept {
    if (cb != nullptr) { _current_callbacks.emplace_back(cb); }
}

void CUDAStream::barrier() noexcept {
    auto count = 0u;
    for (auto i = 1u; i < backed_cuda_stream_count; i++) {
        if (_used_streams.test(i)) {
            count++;
            auto event = _worker_events[_current_event];
            _current_event = (_current_event + 1u) % backed_cuda_event_count;
            LUISA_CHECK_CUDA(cuEventSynchronize(event));
            LUISA_CHECK_CUDA(cuEventRecord(event, _worker_streams[i]));
            LUISA_CHECK_CUDA(cuStreamWaitEvent(_worker_streams.front(), event, 0u));
        }
    }
    if (count != 0u) {
        LUISA_INFO("Active streams: {}.", count);
        LUISA_CHECK_CUDA(cuStreamSynchronize(_worker_streams.front()));
    }
    _round = 0u;
    _used_streams.reset();
}

CUstream CUDAStream::handle(bool force_first_stream) const noexcept {
    if (force_first_stream) {
        if (_round == 0u) { _round = (_round + 1u) % backed_cuda_stream_count; }
        return _worker_streams.front();
    }
    auto index = _round;
    _round = (_round + 1u) % backed_cuda_stream_count;
    _used_streams.set(index);
    return _worker_streams[index];
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
