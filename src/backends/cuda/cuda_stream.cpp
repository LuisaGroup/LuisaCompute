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
    if (cb != nullptr) {
        std::scoped_lock lock{_mutex};
        _callbacks.emplace(cb);
        _any_callback = true;
    }
}

void CUDAStream::dispatch_callbacks() noexcept {
    std::scoped_lock lock{_mutex};
    if (_any_callback) {
        _any_callback = false;
        _callbacks.emplace(nullptr);
        LUISA_CHECK_CUDA(cuLaunchHostFunc(
            _handle,
            [](void *p) noexcept {
                constexpr auto pop = [](auto stream) -> CUDACallbackContext * {
                    std::scoped_lock lock{stream->_mutex};
                    if (stream->_callbacks.empty()) [[unlikely]] {
                        LUISA_WARNING_WITH_LOCATION(
                            "Fetching stream callback from empty queue.");
                        return nullptr;
                    }
                    auto callback = stream->_callbacks.front();
                    stream->_callbacks.pop();
                    return callback;
                };
                auto stream = static_cast<CUDAStream *>(p);
                while (auto callback = pop(stream)) {
                    callback->recycle();
                }
            },
            this));
    }
}

}// namespace luisa::compute::cuda
