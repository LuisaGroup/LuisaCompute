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

class StreamCallbackSemaphoreUpdate final {

private:
    std::atomic_uint64_t *_semaphore{};
    uint64_t _value{};

private:
    [[nodiscard]] static auto _pool() noexcept {
        static Pool<StreamCallbackSemaphoreUpdate> pool;
        return &pool;
    }

public:
    [[nodiscard]] static auto create(std::atomic_uint64_t *sem, uint64_t value) noexcept {
        auto x = _pool()->create();
        x->_semaphore = sem;
        x->_value = value;
        return x;
    }
    void recycle() noexcept {
        _semaphore->store(_value, std::memory_order_release);
        _pool()->destroy(this);
    }
};

CUDAStream::CUDAStream(CUDADevice *device) noexcept
    : _device{device},
      _upload_pool{64_M, true},
      _download_pool{32_M, false} {

    // create the stream
    LUISA_CHECK_CUDA(cuStreamCreate(&_stream, CU_STREAM_DEFAULT));

    // create the callback thread
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
            [ticket = package.ticket, this] {
                static constexpr auto spins_before_yield = 1024u;
                for (;;) {
                    for (auto i = 0u; i < spins_before_yield; i++) {
                        if (_callback_semaphore.load(std::memory_order_acquire) >= ticket) { return; }
                    }
                    std::this_thread::yield();
                }
            }();
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
    _reclaim_staging(true);
    // wait for the callback thread to stop
    _callback_thread.join();
    // destroy the events before the stream they were recorded on
    for (auto event : _free_events) { LUISA_CHECK_CUDA(cuEventDestroy(event)); }
    _free_events.clear();
    // destroy the stream
    LUISA_CHECK_CUDA(cuStreamDestroy(_stream));
}

void CUDAStream::synchronize() noexcept {
    auto ticket = _current_ticket.load();
    LUISA_CHECK_CUDA(cuStreamSynchronize(_stream));
    // Only reclaim what the stream has provably passed: a concurrent dispatch() on
    // another thread may have recorded a fresh event between cuStreamSynchronize()
    // and this call, and forcing it would hand its block back to the pool while the
    // copies that use it are still queued.
    _reclaim_staging(false);
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
        // signal via cuLaunchHostFunc so the driver orders the host write after the stream's prior DMAs
        auto update = StreamCallbackSemaphoreUpdate::create(&_callback_semaphore, ticket);
        LUISA_CHECK_CUDA(cuLaunchHostFunc(
            _stream,
            [](void *data) noexcept {
                auto update = static_cast<StreamCallbackSemaphoreUpdate *>(data);
                update->recycle();
            },
            update));
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

void CUDAStream::defer_recycle(luisa::vector<CUDACallbackContext *> &&contexts) noexcept {
    if (contexts.empty()) { return; }
    std::scoped_lock lock{_staging_mutex};
    auto event = [this] {
        if (!_free_events.empty()) {
            auto e = _free_events.back();
            _free_events.pop_back();
            return e;
        }
        CUevent e{};
        LUISA_CHECK_CUDA(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING));
        return e;
    }();
    // Recorded after every command of this submission, so the block is only
    // reusable once the copies that read (upload) or wrote (download) it are
    // complete.  A stream event is a semaphore write, ~free next to the host
    // callback it replaces.
    LUISA_CHECK_CUDA(cuEventRecord(event, _stream));
    _pending_staging.emplace_back(StagingBatch{.event = event,
                                                .contexts = std::move(contexts)});
}

void CUDAStream::_reclaim_staging(bool all) noexcept {
    std::scoped_lock lock{_staging_mutex};
    if (_pending_staging.empty()) { return; }
    auto keep = size_t{0u};
    for (auto i = size_t{0u}; i < _pending_staging.size(); i++) {
        auto &batch = _pending_staging[i];
        auto done = all;
        if (!done) {
            auto result = cuEventQuery(batch.event);
            if (result == CUDA_SUCCESS) {
                done = true;
            } else if (result != CUDA_ERROR_NOT_READY) {
                LUISA_CHECK_CUDA(result);
            }
        }
        if (done) {
            for (auto &&context : batch.contexts) { context->recycle(); }
            _free_events.emplace_back(batch.event);
        } else {
            if (keep != i) { _pending_staging[keep] = std::move(batch); }
            keep++;
        }
    }
    _pending_staging.resize(keep);
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

void CUDAStream::set_log_callback(LogCallback callback) noexcept {
    if (_log_callback) {
        LUISA_WARNING_WITH_LOCATION(
            "Setting CUDAStream::log_callback more than once. "
            "Please note this is not thread-safe. You may want to "
            "synchonize the stream before setting the callback.");
    }
    _log_callback = std::move(callback);
}

void CUDAStream::dispatch(CommandList &&command_list) noexcept {
    _reclaim_staging(false);
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
