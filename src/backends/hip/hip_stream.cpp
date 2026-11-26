//
// Created by mike on 1/9/26.
//

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <string>

#include <luisa/core/stl/memory.h>
#include <luisa/core/logging.h>
#include <luisa/core/pool.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/core/stl/string.h>

#include "hip_check.h"
#include "hip_device.h"
#include "hip_shader.h"
#include "hip_command_encoder.h"
#include "hip_stream.h"

namespace luisa::compute::hip {

void HIPStream::_create_callback_semaphore() noexcept {
    // Always retire callbacks via hipLaunchHostFunc + host-allocated atomic
    // semaphore (matching CUDAStream) and flush the stream; WDDM/PAL does not
    // reliably flush hipStreamWriteValue64 from an un-synchronized stream.
    _callback_semaphore = luisa::allocate_with_allocator<volatile uint64_t>(1u);
    _callback_semaphore_device = nullptr;
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
                        if (std::atomic_ref<uint64_t>{*const_cast<uint64_t *>(_callback_semaphore)}
                                .load(std::memory_order_acquire) >= ticket) {
                            return;
                        }
                    }
                    std::this_thread::yield();
                }
            }();
            for (auto &&callback : package.callbacks) { callback->recycle(); }
            // signal the event that the callbacks have finished
            _finished_ticket.store(package.ticket, std::memory_order_release);
            _finished_ticket.notify_all();
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
    _notify_hiprt_build_completed_after_synchronize();
    // wait for the callback thread to stop
    _callback_thread.join();
}

void HIPStream::_notify_hiprt_build_completed_after_synchronize() noexcept {
    if (_hiprt_build_pending.exchange(false, std::memory_order_acq_rel)) {
        _device->notify_hiprt_build_completed();
    }
}

HIPStream::HIPStream(HIPDevice *device) noexcept
    : _device{device}, _stream{},
      _upload_pool{64_M, true},
      _download_pool{64_M, false} {
    LUISA_CHECK_HIP(hipStreamCreate(&_stream));
    _create_callback_semaphore();
    _spawn_callback_thread();
    if (auto env = std::getenv("LUISA_HIP_PROFILE"); env != nullptr) {
        const auto mode = luisa::string_view{env};
        if (mode == "summary" || mode == "Summary" || mode == "SUMMARY") {
            _summary_profiling_enabled = true;
            LUISA_INFO("HIP stream summary profiling enabled (LUISA_HIP_PROFILE=summary).");
        } else {
            _profiling_enabled = (mode == "1" || mode == "true");
            if (_profiling_enabled) {
                LUISA_INFO("HIP stream profiling enabled (LUISA_HIP_PROFILE=1).");
            }
        }
    }
}

void HIPStream::_drain_profile_events_after_sync() noexcept {
    if (_pending_profile_events.empty()) {
        return;
    }
    for (auto &ep : _pending_profile_events) {
        float gpu_ms = 0.f;
        LUISA_CHECK_HIP(hipEventElapsedTime(&gpu_ms, ep.start, ep.stop));
        _total_gpu_time_ms += static_cast<double>(gpu_ms);
        if (ep.tag == Command::Tag::EShaderDispatchCommand) {
            luisa::string key = ep.shader_name;
            if (key.empty()) {
                key = luisa::format("shader_0x{:x}", ep.shader_handle);
            }
            auto &agg = _shader_profile_totals[std::move(key)];
            agg.dispatches += 1u;
            agg.gpu_ms += static_cast<double>(gpu_ms);
        } else {
            _other_gpu_time_ms += static_cast<double>(gpu_ms);
        }
        LUISA_CHECK_HIP(hipEventDestroy(ep.start));
        LUISA_CHECK_HIP(hipEventDestroy(ep.stop));
    }
    _pending_profile_events.clear();
}

HIPStream::~HIPStream() noexcept {
    if (_profiling_enabled && _dispatch_count > 0u) {
        LUISA_INFO("HIP stream profiling summary: {} dispatches, total GPU time = {:.3f} ms, "
                   "avg = {:.3f} ms/dispatch",
                   _dispatch_count, _total_gpu_time_ms,
                   _total_gpu_time_ms / static_cast<double>(_dispatch_count));
    }
    _shutdown_callback_thread();
    if (_summary_profiling_enabled) {
        _drain_profile_events_after_sync();
        if (_commit_count > 0u || _sync_count > 0u) {
            struct Entry {
                luisa::string name;
                uint64_t dispatches;
                double gpu_ms;
            };
            luisa::vector<Entry> sorted;
            sorted.reserve(_shader_profile_totals.size());
            double shader_gpu_ms = 0.0;
            for (const auto &[k, v] : _shader_profile_totals) {
                shader_gpu_ms += v.gpu_ms;
                sorted.push_back(Entry{k, v.dispatches, v.gpu_ms});
            }
            std::sort(sorted.begin(), sorted.end(), [](const Entry &a, const Entry &b) {
                return a.gpu_ms > b.gpu_ms;
            });
            LUISA_INFO(
                "HIP stream summary: commits={} syncs={} host_encode_ms={:.3f} sync_wait_ms={:.3f} "
                "shader_gpu_ms={:.3f} other_gpu_ms={:.3f} total_gpu_ms={:.3f}",
                _commit_count, _sync_count, _host_encode_ms, _sync_wait_ms,
                shader_gpu_ms, _other_gpu_time_ms, _total_gpu_time_ms);
            for (const auto &e : sorted) {
                LUISA_INFO("  HIP kernel '{}': dispatches={} gpu_ms={:.3f} avg_ms={:.4f}",
                           e.name, e.dispatches, e.gpu_ms,
                           e.dispatches > 0u ? e.gpu_ms / static_cast<double>(e.dispatches) : 0.0);
            }
            std::string json;
            json.reserve(4096u);
            json += luisa::format(
                "{{\"commits\":{},\"sync_calls\":{},\"host_encode_ms\":{:.3f},\"sync_wait_ms\":{:.3f},"
                "\"shader_gpu_ms\":{:.3f},\"other_gpu_ms\":{:.3f},\"total_gpu_ms\":{:.3f},\"shaders\":[",
                _commit_count, _sync_count, _host_encode_ms, _sync_wait_ms,
                shader_gpu_ms, _other_gpu_time_ms, _total_gpu_time_ms).c_str();
            for (size_t i = 0u; i < sorted.size(); ++i) {
                if (i != 0u) { json += ','; }
                json += luisa::format(
                    "{{\"name\":\"{}\",\"dispatches\":{},\"gpu_ms\":{:.3f}}}",
                    sorted[i].name, sorted[i].dispatches, sorted[i].gpu_ms).c_str();
            }
            json += "]}";
            LUISA_INFO("LUISA_HIP_PROFILE_SUMMARY_JSON={}", json);
        }
    }
    if (_rt_scratch_buffer != nullptr) {
        LUISA_CHECK_HIP(hipFree(_rt_scratch_buffer));
    }
    if (_rt_global_stack_buffer != nullptr) {
        LUISA_CHECK_HIP(hipFree(_rt_global_stack_buffer));
    }
    _destroy_callback_semaphore();
    LUISA_CHECK_HIP(hipStreamDestroy(_stream));
}

hipDeviceptr_t HIPStream::rt_scratch_buffer(size_t required_size) noexcept {
    if (required_size == 0u) { return nullptr; }
    if (required_size > _rt_scratch_capacity) {
        constexpr auto maximum_power_of_two =
            size_t{1u}
            << (std::numeric_limits<size_t>::digits - 1u);
        LUISA_ASSERT(
            required_size <= maximum_power_of_two,
            "HIP ray-tracing scratch-buffer request {} cannot be "
            "rounded to a representable power-of-two capacity.",
            required_size);
        auto new_capacity = std::bit_ceil(required_size);
        if (_rt_scratch_buffer != nullptr) {
            // Every consumer submits all scratch uses to this stream, and
            // dispatch() serializes command encoding. Thus successive uses are
            // totally ordered and may alias. Drain the old lifetime only on
            // this rare growth path before replacing its backing allocation.
            LUISA_CHECK_HIP(hipStreamSynchronize(_stream));
            _notify_hiprt_build_completed_after_synchronize();
            LUISA_CHECK_HIP(hipFree(_rt_scratch_buffer));
        }
        LUISA_CHECK_HIP(hipMalloc(
            reinterpret_cast<void **>(&_rt_scratch_buffer),
            new_capacity));
        _rt_scratch_capacity = new_capacity;
    }
    return _rt_scratch_buffer;
}

hiprtGlobalStackBuffer
HIPStream::rt_global_stack_buffer(
    size_t required_thread_count) noexcept {
    static constexpr auto stack_size = uint32_t{64u};
    if (required_thread_count == 0u) {
        return hiprtGlobalStackBuffer{
            stack_size, 0u, nullptr};
    }
    if (required_thread_count >
        _rt_global_stack_thread_capacity) {
        constexpr auto maximum_power_of_two =
            size_t{1u}
            << (std::numeric_limits<size_t>::digits - 1u);
        LUISA_ASSERT(
            required_thread_count <= maximum_power_of_two,
            "HIPRT global-stack thread request {} cannot be rounded to a "
            "representable power-of-two capacity.",
            required_thread_count);
        const auto new_capacity =
            std::bit_ceil(required_thread_count);
        LUISA_ASSERT(
            new_capacity <=
                std::numeric_limits<uint32_t>::max(),
            "HIPRT global-stack launch requires {} physical threads; the "
            "HIPRT stack ABI supports at most {}.",
            required_thread_count,
            std::numeric_limits<uint32_t>::max());
        constexpr auto bytes_per_thread =
            size_t{stack_size} * sizeof(uint32_t);
        LUISA_ASSERT(
            new_capacity <=
                std::numeric_limits<size_t>::max() /
                    bytes_per_thread,
            "HIPRT global-stack allocation size overflow for {} threads.",
            new_capacity);
        // All users of this allocation are enqueued on this stream. Growth is
        // rare and synchronizes only the old allocation's lifetime; ordinary
        // launches remain fully asynchronous.
        if (_rt_global_stack_buffer != nullptr) {
            LUISA_CHECK_HIP(hipStreamSynchronize(_stream));
            _notify_hiprt_build_completed_after_synchronize();
            LUISA_CHECK_HIP(hipFree(
                _rt_global_stack_buffer));
        }
        LUISA_CHECK_HIP(hipMalloc(
            reinterpret_cast<void **>(
                &_rt_global_stack_buffer),
            new_capacity * bytes_per_thread));
        _rt_global_stack_thread_capacity =
            static_cast<uint32_t>(new_capacity);
        LUISA_VERBOSE(
            "Allocated per-stream HIPRT global stack for {} threads "
            "({} bytes).",
            _rt_global_stack_thread_capacity,
            new_capacity * bytes_per_thread);
    }
    return hiprtGlobalStackBuffer{
        stack_size,
        _rt_global_stack_thread_capacity,
        _rt_global_stack_buffer};
}

void HIPStream::dispatch(CommandList &&command_list) noexcept {
    HIPCommandEncoder encoder{this};
    {
        auto commands = command_list.steal_commands();
        auto callbacks = command_list.steal_callbacks();
        std::scoped_lock lock{_dispatch_mutex};

        if (_profiling_enabled) {
            struct EventPair {
                hipEvent_t start{};
                hipEvent_t stop{};
                Command::Tag tag{};
                uint64_t shader_handle{0u};
                uint3 dispatch_size{};
            };
            luisa::vector<EventPair> event_pairs;
            event_pairs.reserve(commands.size());

            for (auto &cmd : commands) {
                auto &ep = event_pairs.emplace_back();
                ep.tag = cmd->tag();
                if (ep.tag == Command::Tag::EShaderDispatchCommand) {
                    auto shader_cmd = static_cast<ShaderDispatchCommand *>(cmd.get());
                    ep.shader_handle = shader_cmd->handle();
                    if (!shader_cmd->is_indirect() && !shader_cmd->is_multiple_dispatch()) {
                        ep.dispatch_size = shader_cmd->dispatch_size();
                    }
                }
                LUISA_CHECK_HIP(hipEventCreate(&ep.start));
                LUISA_CHECK_HIP(hipEventCreate(&ep.stop));
                LUISA_CHECK_HIP(hipEventRecord(ep.start, _stream));
                cmd->accept(encoder);
                LUISA_CHECK_HIP(hipEventRecord(ep.stop, _stream));
            }

            LUISA_CHECK_HIP(hipStreamSynchronize(_stream));
            _notify_hiprt_build_completed_after_synchronize();

            _dispatch_count++;
            auto batch_total_ms = 0.0;
            for (size_t i = 0u; i < event_pairs.size(); i++) {
                auto &ep = event_pairs[i];
                float gpu_ms = 0.f;
                LUISA_CHECK_HIP(hipEventElapsedTime(&gpu_ms, ep.start, ep.stop));
                batch_total_ms += gpu_ms;
                if (ep.tag == Command::Tag::EShaderDispatchCommand) {
                    LUISA_INFO("  HIP dispatch #{} cmd[{}]: ShaderDispatch handle={:#x} "
                               "size=({},{},{}) GPU time = {:.3f} ms",
                               _dispatch_count, i, ep.shader_handle,
                               ep.dispatch_size.x, ep.dispatch_size.y, ep.dispatch_size.z,
                               gpu_ms);
                }
                LUISA_CHECK_HIP(hipEventDestroy(ep.start));
                LUISA_CHECK_HIP(hipEventDestroy(ep.stop));
            }
            _total_gpu_time_ms += batch_total_ms;
            LUISA_INFO("HIP dispatch #{}: {} cmd(s), GPU time = {:.3f} ms (total = {:.3f} ms)",
                       _dispatch_count, commands.size(), batch_total_ms, _total_gpu_time_ms);
        } else if (_summary_profiling_enabled) {
            const auto t_enc0 = std::chrono::steady_clock::now();
            _pending_profile_events.reserve(_pending_profile_events.size() + commands.size());
            for (auto &cmd : commands) {
                PendingProfileEvent ep{};
                ep.tag = cmd->tag();
                if (ep.tag == Command::Tag::EShaderDispatchCommand) {
                    auto shader_cmd = static_cast<ShaderDispatchCommand *>(cmd.get());
                    ep.shader_handle = shader_cmd->handle();
                    if (auto *shader = reinterpret_cast<const HIPShader *>(ep.shader_handle)) {
                        ep.shader_name = shader->name();
                    }
                }
                LUISA_CHECK_HIP(hipEventCreate(&ep.start));
                LUISA_CHECK_HIP(hipEventCreate(&ep.stop));
                LUISA_CHECK_HIP(hipEventRecord(ep.start, _stream));
                cmd->accept(encoder);
                LUISA_CHECK_HIP(hipEventRecord(ep.stop, _stream));
                _pending_profile_events.emplace_back(std::move(ep));
            }
            const auto t_enc1 = std::chrono::steady_clock::now();
            _host_encode_ms += std::chrono::duration<double, std::milli>(t_enc1 - t_enc0).count();
            _commit_count += 1u;
        } else {
            for (auto &cmd : commands) {
                cmd->accept(encoder);
            }
        }

        encoder.commit(std::move(callbacks));
    }
}

void HIPStream::synchronize() noexcept {
    const auto t_sync0 = _summary_profiling_enabled
                             ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
    auto ticket = _current_ticket.load();
    LUISA_CHECK_HIP(hipStreamSynchronize(_stream));
    _notify_hiprt_build_completed_after_synchronize();
    // HIP has already drained the device stream. The remaining wait is only
    // for the callback thread to recycle host-side command resources and
    // publish the completion ticket. Waiting on that atomic avoids occupying
    // a CPU core while a callback package is being retired; the callback
    // thread calls notify_all() immediately after publishing the ticket.
    for (;;) {
        auto finished = _finished_ticket.load(std::memory_order_acquire);
        if (finished >= ticket) { break; }
        _finished_ticket.wait(finished, std::memory_order_acquire);
    }
    if (_summary_profiling_enabled) {
        const auto t_sync1 = std::chrono::steady_clock::now();
        _sync_wait_ms += std::chrono::duration<double, std::milli>(t_sync1 - t_sync0).count();
        _sync_count += 1u;
        std::scoped_lock lock{_dispatch_mutex};
        _drain_profile_events_after_sync();
    }
}

HIPStream::LogCallback HIPStream::log_callback() const noexcept {
    std::scoped_lock lock{_log_callback_mutex};
    return _log_callback;
}

void HIPStream::set_log_callback(LogCallback callback) noexcept {
    std::scoped_lock lock{_log_callback_mutex};
    _log_callback = std::move(callback);
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
        std::atomic_ref<uint64_t>{*const_cast<uint64_t *>(_semaphore)}
            .store(_value, std::memory_order_release);
        _pool()->destroy(this);
    }
};

void HIPStream::callback(CallbackContainer &&callbacks) noexcept {
    if (!callbacks.empty()) {
        auto ticket = 1u + _current_ticket.fetch_add(1u, std::memory_order_relaxed);
        auto update = HIPStreamCallbackSemaphoreUpdate::create(_callback_semaphore, ticket);
        LUISA_CHECK_HIP(hipLaunchHostFunc(
            _stream,
            [](void *data) noexcept {
                auto update = static_cast<HIPStreamCallbackSemaphoreUpdate *>(data);
                update->recycle();
            },
            update));
        // Flush the WDDM/PAL command buffer immediately so callers awaiting
        // callbacks asynchronously observe completion without synchronize().
        static_cast<void>(hipStreamQuery(_stream));
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
