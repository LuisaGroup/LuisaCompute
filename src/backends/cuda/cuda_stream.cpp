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
    volatile uint64_t *_semaphore{};
    uint64_t _value{};

private:
    [[nodiscard]] static auto _pool() noexcept {
        static Pool<StreamCallbackSemaphoreUpdate> pool;
        return &pool;
    }

public:
    [[nodiscard]] static auto create(volatile uint64_t *sem, uint64_t value) noexcept {
        auto x = _pool()->create();
        x->_semaphore = sem;
        x->_value = value;
        return x;
    }
    void recycle() noexcept {
        *_semaphore = _value;
        _pool()->destroy(this);
    }
};

CUDAStream::CUDAStream(CUDADevice *device) noexcept
    : _device{device},
      _upload_pool{64_M, true},
      _download_pool{32_M, false} {

    // initialize the callback semaphore
    {
        // check if the device supports stream-ordered memory operations
        auto stream_mem_op_support = 0;
        LUISA_CHECK_CUDA(cuDeviceGetAttribute(&stream_mem_op_support,
                                              CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
                                              device->handle().device()));
        if (stream_mem_op_support) {
            auto callback_semaphore = static_cast<void *>(nullptr);
            LUISA_CHECK_CUDA(cuMemHostAlloc(&callback_semaphore,
                                            sizeof(uint64_t), CU_MEMHOSTALLOC_DEVICEMAP));
            _callback_semaphore = static_cast<volatile uint64_t *>(callback_semaphore);
            LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(&_callback_semaphore_device,
                                                       callback_semaphore, 0u));
        } else {
            LUISA_WARNING_WITH_LOCATION("Stream memory operation is not supported. "
                                        "LuisaCompute will use stream callbacks to "
                                        "synchronize the stream. This may cause "
                                        "performance degradation.");
            _callback_semaphore = luisa::allocate_with_allocator<uint64_t>(1u);
            _callback_semaphore_device = 0u;
        }
        *_callback_semaphore = 0u;
    }
    // create the stream
    LUISA_CHECK_CUDA(cuStreamCreate(&_stream, CU_STREAM_NON_BLOCKING));
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
                        if (*_callback_semaphore >= ticket) { return; }
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
    // wait for the callback thread to stop
    _callback_thread.join();
    // destroy the events and the stream
    auto callback_sem = const_cast<uint64_t *>(_callback_semaphore);
    if (_callback_semaphore_device != 0u) {
        LUISA_CHECK_CUDA(cuMemFreeHost(callback_sem));
    } else {
        luisa::deallocate_with_allocator(callback_sem);
    }
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
        if (_callback_semaphore_device) {
            LUISA_CHECK_CUDA(cuStreamWriteValue64(_stream, _callback_semaphore_device,
                                                  ticket, CU_STREAM_WRITE_VALUE_DEFAULT));
        } else {
            auto update = StreamCallbackSemaphoreUpdate::create(_callback_semaphore, ticket);
            LUISA_CHECK_CUDA(cuLaunchHostFunc(
                _stream,
                [](void *data) noexcept {
                    auto update = static_cast<StreamCallbackSemaphoreUpdate *>(data);
                    update->recycle();
                },
                update));
        }
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
