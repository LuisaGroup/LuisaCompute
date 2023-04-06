//
// Created by Mike on 8/1/2021.
//

#include <cstdlib>
#include <mutex>

#include <nvtx3/nvToolsExtCuda.h>

#include <core/logging.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_callback_context.h>
#include <backends/cuda/cuda_device.h>

namespace luisa::compute::cuda {

[[nodiscard]] auto cuda_stream_assign_uid() noexcept {
    static std::atomic_uint64_t uid{0u};
    return uid.fetch_add(1u, std::memory_order_relaxed);
}

CUDAStream::CUDAStream(CUDADevice *device) noexcept
    : _device{device}, _upload_pool{64_M, true},
      _uid{cuda_stream_assign_uid()} {
    LUISA_CHECK_CUDA(cuStreamCreate(&_stream, CU_STREAM_NON_BLOCKING));
}

CUDAStream::~CUDAStream() noexcept {
    LUISA_CHECK_CUDA(cuStreamDestroy(_stream));
}

void CUDAStream::synchronize() noexcept {
    LUISA_CHECK_CUDA(cuStreamSynchronize(_stream));
}

void CUDAStream::callback(CUDAStream::CallbackContainer &&callbacks) noexcept {
    if (!callbacks.empty()) {
        {
            std::scoped_lock lock{_mutex};
            _callback_lists.emplace(std::move(callbacks));
        }
        LUISA_CHECK_CUDA(cuLaunchHostFunc(
            _stream, [](void *p) noexcept {
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
}

void CUDAStream::signal(CUevent event) noexcept {
    LUISA_CHECK_CUDA(cuEventRecord(event, _stream));
}

void CUDAStream::wait(CUevent event) noexcept {
    LUISA_CHECK_CUDA(cuStreamWaitEvent(_stream, event, CU_EVENT_WAIT_DEFAULT));
}

void CUDAStream::set_name(luisa::string &&name) noexcept {
    nvtxNameCuStreamA(_stream, name.c_str());
}

}// namespace luisa::compute::cuda
