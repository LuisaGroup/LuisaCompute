#include <cstdlib>
#include <mutex>

#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/logging.h>

#include "cuda_error.h"
#include "cuda_callback_context.h"
#include "cuda_command_encoder.h"
#include "cuda_device.h"
#include "cuda_stream.h"
#include "cuda_event.h"

namespace luisa::compute::cuda {

CUDAStream::CUDAStream(CUDADevice *device) noexcept
    : _device{device},
      _upload_pool{64_M, true}, _download_pool{32_M, false},
      _callback_event{device->event_manager()->create()} {
    LUISA_CHECK_CUDA(cuStreamCreate(&_stream, CU_STREAM_NON_BLOCKING));
    _callback_thread = std::thread{[this] {
        for (;;) {
            auto package = [this] {
                std::unique_lock lock{_callback_mutex};
                _callback_cv.wait(lock, [this] { return !_callback_lists.empty(); });
                auto p = std::move(_callback_lists.front());
                _callback_lists.pop();
                if (p.ticket == stop_ticket && !_callback_lists.empty()) [[unlikely]] {
                    LUISA_WARNING_WITH_LOCATION(
                        "Stream callback queue is not empty "
                        "when stop ticket is received.");
                }
                return p;
            }();
            if (package.ticket == stop_ticket) { break; }
            // wait for the commands to finish
            _callback_event->synchronize(package.ticket);
            for (auto &&callback : package.callbacks) { callback->recycle(); }
            // signal the event that the callbacks have finished
            _finished_ticket.store(package.ticket, std::memory_order_release);
        }
    }};
}

CUDAStream::~CUDAStream() noexcept {
    // notify the callback thread to stop
    {
        CallbackPackage p{.ticket = stop_ticket};
        std::scoped_lock lock{_callback_mutex};
        _callback_lists.emplace(std::move(p));
    }
    _callback_cv.notify_one();
    // wait for the stream to finish
    LUISA_CHECK_CUDA(cuStreamSynchronize(_stream));
    // wait for the callback thread to stop
    _callback_thread.join();
    // destroy the events and the stream
    _device->event_manager()->destroy(_callback_event);
    LUISA_CHECK_CUDA(cuStreamDestroy(_stream));
}

void CUDAStream::synchronize() noexcept {
    auto ticket = _current_ticket.load();
    LUISA_CHECK_CUDA(cuStreamSynchronize(_stream));
    auto wait_iterations = 0u;
    constexpr auto max_wait_iterations_before_yield = 1024u;
    for (;;) {// TODO: is spinning good enough?
        if (_finished_ticket.load(std::memory_order_acquire) >= ticket) { break; }
        if (++wait_iterations >= max_wait_iterations_before_yield) {
            wait_iterations = 0u;
            std::this_thread::yield();
        }
    }
}

void CUDAStream::callback(CUDAStream::CallbackContainer &&callbacks) noexcept {
    if (!callbacks.empty()) {
        // signal that the stream has been dispatched
        auto ticket = 1u + _current_ticket.fetch_add(1u, std::memory_order_relaxed);
        _callback_event->signal(_stream, ticket);
        // enqueue callbacks
        {
            CallbackPackage package{
                .ticket = ticket,
                .callbacks = std::move(callbacks)};
            std::scoped_lock lock{_callback_mutex};
            _callback_lists.emplace(std::move(package));
        }
        // notify the callback thread
        _callback_cv.notify_one();
    }
}

void CUDAStream::signal(CUDAEvent *event, uint64_t value) noexcept {
    event->signal(_stream, value);
}

void CUDAStream::wait(CUDAEvent *event, uint64_t value) noexcept {
    event->wait(_stream, value);
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
