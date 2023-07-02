//
// Created by Mike on 8/1/2021.
//

#include <cstdlib>
#include <mutex>

#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/logging.h>
#include "cuda_error.h"
#include "cuda_command_encoder.h"
#include "cuda_callback_context.h"
#include "cuda_device.h"
#include "cuda_stream.h"

namespace luisa::compute::cuda {

CUDAStream::CUDAStream(CUDADevice *device) noexcept
    : _device{device},
      _upload_pool{64_M, true},
      _download_pool{32_M, false} {
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
            std::scoped_lock lock{_callback_mutex};
            _callback_lists.emplace(std::move(callbacks));
        }
        LUISA_CHECK_CUDA(cuLaunchHostFunc(
            _stream, [](void *p) noexcept {
                constexpr auto pop = [](auto stream) -> luisa::vector<CUDACallbackContext *> {
                    std::scoped_lock lock{stream->_callback_mutex};
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

void CUDAStream::signal(CUdeviceptr event, uint64_t value) noexcept {
    LUISA_CHECK_CUDA(cuStreamWriteValue64(
        _stream, event, value,
        CU_STREAM_WRITE_VALUE_DEFAULT));
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        _stream, [](void *p) noexcept {
            LUISA_INFO("After signal for {}.",
                       reinterpret_cast<uint64_t>(p));
        },
        reinterpret_cast<void *>(value)));
}

void CUDAStream::wait(CUdeviceptr event, uint64_t value) noexcept {
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        _stream, [](void *p) noexcept {
            LUISA_INFO("Before wait for {}.",
                       reinterpret_cast<uint64_t>(p));
        },
        reinterpret_cast<void *>(value)));
    LUISA_CHECK_CUDA(cuStreamWaitValue64(
        _stream, event, value,
        CU_STREAM_WAIT_VALUE_GEQ));
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        _stream, [](void *p) noexcept {
            LUISA_INFO("After wait for {}.",
                       reinterpret_cast<uint64_t>(p));
        },
        reinterpret_cast<void *>(value)));
}

void CUDAStream::set_name(luisa::string &&name) noexcept {
    nvtxNameCuStreamA(_stream, name.c_str());
}

void CUDAStream::dispatch(CommandList &&command_list) noexcept {
    CUDACommandEncoder encoder{this};
    auto commands = command_list.steal_commands();
    auto callbacks = command_list.steal_callbacks();
    {
        std::scoped_lock lock{_dispatch_mutex};
        for (auto &cmd : commands) { cmd->accept(encoder); }
        encoder.commit(std::move(callbacks));
    }
}

}// namespace luisa::compute::cuda
