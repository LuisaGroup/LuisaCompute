//
// Created by mike on 1/9/26.
//

#include <cstdlib>

#include <luisa/core/stl/memory.h>
#include <luisa/core/logging.h>
#include <luisa/core/pool.h>

#include "hip_check.h"
#include "hip_device.h"
#include "hip_command_encoder.h"
#include "hip_stream.h"

namespace luisa::compute::hip {

void HIPStream::_create_callback_semaphore() noexcept {
    auto stream_mem_op_support = 0;
    LUISA_CHECK_HIP(hipDeviceGetAttribute(&stream_mem_op_support,
                                          hipDeviceAttributeCanUseStreamWaitValue,
                                          _device->device_id()));
    if (stream_mem_op_support) {
        auto callback_semaphore = static_cast<void *>(nullptr);
        LUISA_CHECK_HIP(hipHostMalloc(&callback_semaphore,
                                      sizeof(uint64_t), hipHostMallocMapped));
        _callback_semaphore = static_cast<volatile uint64_t *>(callback_semaphore);
        LUISA_CHECK_HIP(hipHostGetDevicePointer(&_callback_semaphore_device,
                                                callback_semaphore, 0u));
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "HIP device does not support stream-ordered memory operations. "
            "Stream callbacks may have higher latency.");
        _callback_semaphore = luisa::allocate_with_allocator<volatile uint64_t>(1u);
        _callback_semaphore_device = nullptr;
    }
    *_callback_semaphore = 0u;
}

void HIPStream::_destroy_callback_semaphore() noexcept {
    auto callback_sem = const_cast<uint64_t *>(_callback_semaphore);
    if (_callback_semaphore_device != nullptr) {
        LUISA_CHECK_HIP(hipHostFree(callback_sem));
    } else {
        luisa::deallocate_with_allocator(callback_sem);
    }
}

void HIPStream::_spawn_callback_thread() noexcept {
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

void HIPStream::_shutdown_callback_thread() noexcept {
    // notify the callback thread to stop
    {
        CallbackPackage p{.ticket = stop_ticket};
        std::scoped_lock lock{_callback_mutex};
        _callback_lists.emplace(std::move(p));
    }
    _callback_cv.notify_one();
    // wait for the stream to finish
    LUISA_CHECK_HIP(hipStreamSynchronize(_stream));
    // wait for the callback thread to stop
    _callback_thread.join();
}

HIPStream::HIPStream(HIPDevice *device) noexcept
    : _device{device}, _stream{},
      _upload_pool{64_M, true},
      _download_pool{64_M, false} {
    LUISA_CHECK_HIP(hipStreamCreate(&_stream));
    _create_callback_semaphore();
    _spawn_callback_thread();
    if (auto env = std::getenv("LUISA_HIP_PROFILE"); env != nullptr) {
        _profiling_enabled = (std::string_view{env} == "1" || std::string_view{env} == "true");
        if (_profiling_enabled) {
            LUISA_INFO("HIP stream profiling enabled (LUISA_HIP_PROFILE=1).");
        }
    }
}

HIPStream::~HIPStream() noexcept {
    if (_profiling_enabled && _dispatch_count > 0u) {
        LUISA_INFO("HIP stream profiling summary: {} dispatches, total GPU time = {:.3f} ms, "
                   "avg = {:.3f} ms/dispatch",
                   _dispatch_count, _total_gpu_time_ms,
                   _total_gpu_time_ms / static_cast<double>(_dispatch_count));
    }
    _destroy_callback_semaphore();
    _shutdown_callback_thread();
    LUISA_CHECK_HIP(hipStreamDestroy(_stream));
}

void HIPStream::dispatch(CommandList &&command_list) noexcept {
    HIPCommandEncoder encoder{this};
    {
        auto commands = command_list.steal_commands();
        auto callbacks = command_list.steal_callbacks();
        std::scoped_lock lock{_dispatch_mutex};

        hipEvent_t start_event{}, stop_event{};
        if (_profiling_enabled) {
            LUISA_CHECK_HIP(hipEventCreate(&start_event));
            LUISA_CHECK_HIP(hipEventCreate(&stop_event));
            LUISA_CHECK_HIP(hipEventRecord(start_event, _stream));
        }

        for (auto &cmd : commands) {
            cmd->accept(encoder);
        }

        if (_profiling_enabled) {
            LUISA_CHECK_HIP(hipEventRecord(stop_event, _stream));
            LUISA_CHECK_HIP(hipEventSynchronize(stop_event));
            float gpu_ms = 0.f;
            LUISA_CHECK_HIP(hipEventElapsedTime(&gpu_ms, start_event, stop_event));
            _total_gpu_time_ms += gpu_ms;
            _dispatch_count++;
            LUISA_INFO("HIP dispatch #{}: {} cmd(s), GPU time = {:.3f} ms (total = {:.3f} ms)",
                       _dispatch_count, commands.size(), gpu_ms, _total_gpu_time_ms);
            LUISA_CHECK_HIP(hipEventDestroy(start_event));
            LUISA_CHECK_HIP(hipEventDestroy(stop_event));
        }

        encoder.commit(std::move(callbacks));
    }
}

void HIPStream::synchronize() noexcept {
    auto ticket = _current_ticket.load();
    LUISA_CHECK_HIP(hipStreamSynchronize(_stream));
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

class HIPStreamCallbackSemaphoreUpdate {

private:
    volatile uint64_t *_semaphore{};
    uint64_t _value{};

    [[nodiscard]] static auto _pool() noexcept {
        static Pool<HIPStreamCallbackSemaphoreUpdate> pool;
        return &pool;
    }

public:
    HIPStreamCallbackSemaphoreUpdate(volatile uint64_t *semaphore, uint64_t value) noexcept
        : _semaphore{semaphore}, _value{value} {}
    static HIPStreamCallbackSemaphoreUpdate *create(volatile uint64_t *semaphore, uint64_t value) noexcept {
        return _pool()->create(semaphore, value);
    }
    void recycle() noexcept {
        *_semaphore = _value;
        _pool()->destroy(this);
    }
};

void HIPStream::callback(CallbackContainer &&callbacks) noexcept {
    if (!callbacks.empty()) {
        auto ticket = 1u + _current_ticket.fetch_add(1u, std::memory_order_relaxed);
        if (_callback_semaphore_device != nullptr) {
            LUISA_CHECK_HIP(hipStreamWriteValue64(_stream, _callback_semaphore_device, ticket, 0u));
        } else {
            auto update = HIPStreamCallbackSemaphoreUpdate::create(_callback_semaphore, ticket);
            LUISA_CHECK_HIP(hipLaunchHostFunc(
                _stream,
                [](void *data) noexcept {
                    auto update = static_cast<HIPStreamCallbackSemaphoreUpdate *>(data);
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

}// namespace luisa::compute::hip
