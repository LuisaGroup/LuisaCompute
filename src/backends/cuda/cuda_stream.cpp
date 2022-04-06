//
// Created by Mike on 8/1/2021.
//

#include <mutex>

#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_callback_context.h>

namespace luisa::compute::cuda {

CUDAStream::CUDAStream() noexcept
    : _upload_pool{64_mb, true} {
    int lo, hi;
    LUISA_CHECK_CUDA(cuCtxGetStreamPriorityRange(&lo, &hi));
    LUISA_CHECK_CUDA(cuStreamCreateWithPriority(
        &_worker_streams.front(), CU_STREAM_NON_BLOCKING, hi));
    for (auto i = 1u; i < backed_cuda_stream_count; i++) {
        LUISA_CHECK_CUDA(cuStreamCreateWithPriority(
            &_worker_streams[i], CU_STREAM_NON_BLOCKING, hi));
        LUISA_CHECK_CUDA(cuEventCreate(
            &_worker_events[i], CU_EVENT_DISABLE_TIMING));
    }
}

CUDAStream::~CUDAStream() noexcept {
    LUISA_CHECK_CUDA(cuStreamDestroy(_worker_streams.front()));
    for (auto i = 1u; i < backed_cuda_stream_count; i++) {
        LUISA_CHECK_CUDA(cuStreamDestroy(_worker_streams[i]));
        LUISA_CHECK_CUDA(cuEventDestroy(_worker_events[i]));
    }
}

void CUDAStream::emplace_callback(CUDACallbackContext *cb) noexcept {
    if (cb != nullptr) { _current_callbacks.emplace_back(cb); }
}

void CUDAStream::barrier() noexcept {
    _used_streams.reset(0u);
    if (_used_streams.any()) {
        for (auto i = 1u; i < backed_cuda_stream_count; i++) {
            if (_used_streams.test(i)) {
                LUISA_CHECK_CUDA(cuEventRecord(
                    _worker_events[i], _worker_streams[i]));
                LUISA_CHECK_CUDA(cuStreamWaitEvent(
                    _worker_streams.front(), _worker_events[i],
                    CU_EVENT_WAIT_DEFAULT));
            }
        }
    }
    _round = 0u;
    _used_streams.reset();
}

CUstream CUDAStream::handle(bool force_first_stream) const noexcept {
    if (force_first_stream) { return _worker_streams.front(); }
    auto index = _round++;
    if (_round == backed_cuda_stream_count) { _round = 0u; }
    if (index != 0u) { _used_streams.set(index); }
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
