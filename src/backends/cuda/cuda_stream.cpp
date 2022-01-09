//
// Created by Mike on 8/1/2021.
//

#include <mutex>

#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_callback_context.h>

namespace luisa::compute::cuda {

CUDAStream::CUDAStream() noexcept
    : _handle{nullptr},
      _upload_pool{64_mb, true} {
    int lo, hi;
    LUISA_CHECK_CUDA(cuCtxGetStreamPriorityRange(&lo, &hi));
    LUISA_CHECK_CUDA(cuStreamCreateWithPriority(&_handle, CU_STREAM_NON_BLOCKING, hi));
}

CUDAStream::~CUDAStream() noexcept {
    LUISA_CHECK_CUDA(cuStreamDestroy(_handle));
}

void CUDAStream::emplace_callback(CUDACallbackContext *cb) noexcept {
    if (cb != nullptr) { _current_callbacks.emplace_back(cb); }
}

void CUDAStream::dispatch_callbacks() noexcept {
    if (_current_callbacks.empty()) { return; }
    auto callbacks = std::move(_current_callbacks);
    _current_callbacks = {};
    std::scoped_lock lock{_mutex};
    _callback_lists.emplace(std::move(callbacks));
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        _handle, [](void *p) noexcept {
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
