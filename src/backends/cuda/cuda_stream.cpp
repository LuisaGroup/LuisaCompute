//
// Created by Mike on 8/1/2021.
//

#include <cstdlib>
#include <mutex>

#include <nvtx3/nvToolsExtCuda.h>

#include <core/logging.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_command_encoder.h>
#include <backends/cuda/cuda_callback_context.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_stream.h>

namespace luisa::compute::cuda {

[[nodiscard]] auto cuda_stream_assign_uid() noexcept {
    static std::atomic_uint uid{0u};
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

void CUDAStream::signal(CUevent event) noexcept {
    LUISA_CHECK_CUDA(cuEventRecord(event, _stream));
}

void CUDAStream::wait(CUevent event) noexcept {
    LUISA_CHECK_CUDA(cuStreamWaitEvent(_stream, event, CU_EVENT_WAIT_DEFAULT));
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

class CUDAIndirectDispatch final : public CUDACallbackContext {

private:
    ShaderDispatchCommand _command;

public:
    void recycle() noexcept override {

    }
};

CUDAIndirectDispatchStream::CUDAIndirectDispatchStream(CUDAStream *parent) noexcept
    : _thread{[this] {
          while (true) {
              auto task = [this] {
                  std::unique_lock lock{_mutex};
                  _cv.wait(lock, [this] { return !_tasks.empty(); });
                  auto task = _tasks.front();
                  _tasks.pop();
                  return task;
              }();
              if (task == stop_token) { break; }
              _parent->device()->with_handle([this, task] {
                  LUISA_CHECK_CUDA(cuStreamWaitEvent(
                      _stream, _event_to_wait,
                      CU_EVENT_WAIT_DEFAULT));
                  task->recycle();
                  LUISA_CHECK_CUDA(cuEventRecord(
                      _event_to_signal, _stream));
              });
          }
      }} {}

CUDAIndirectDispatchStream::~CUDAIndirectDispatchStream() noexcept { stop(); }

void CUDAIndirectDispatchStream::enqueue(ShaderDispatchCommand *command) noexcept {

}

void CUDAIndirectDispatchStream::stop() noexcept {

}

}// namespace luisa::compute::cuda
