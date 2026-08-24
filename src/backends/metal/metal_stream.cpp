#include <luisa/core/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "metal_event.h"
#include "metal_texture.h"
#include "metal_swapchain.h"
#include "metal_command_encoder.h"
#include "metal_stream.h"

namespace luisa::compute::metal {

namespace {

struct MetalCommandBufferProfileStats {
    uint64_t count{};
    double total_ms{};
    double min_ms{};
    double max_ms{};
};

class MetalCommandBufferProfiler {

private:
    std::mutex _mutex;
    std::unordered_map<std::string, MetalCommandBufferProfileStats> _gpu_stats;
    std::unordered_map<std::string, MetalCommandBufferProfileStats> _host_stats;

private:
    static void _record(
        std::unordered_map<std::string, MetalCommandBufferProfileStats> &stats,
        const char *name, double elapsed_ms) noexcept {
        if (name == nullptr || name[0] == '\0' || elapsed_ms <= 0.0) { return; }
        auto &profile = stats[name];
        if (profile.count == 0u) {
            profile.min_ms = elapsed_ms;
            profile.max_ms = elapsed_ms;
        } else {
            profile.min_ms = std::min(profile.min_ms, elapsed_ms);
            profile.max_ms = std::max(profile.max_ms, elapsed_ms);
        }
        profile.count++;
        profile.total_ms += elapsed_ms;
    }

public:
    ~MetalCommandBufferProfiler() noexcept {
        std::vector<std::pair<std::string, MetalCommandBufferProfileStats>> gpu_stats;
        std::vector<std::pair<std::string, MetalCommandBufferProfileStats>> host_stats;
        {
            std::scoped_lock lock{_mutex};
            gpu_stats.reserve(_gpu_stats.size());
            for (auto &&item : _gpu_stats) { gpu_stats.emplace_back(item); }
            host_stats.reserve(_host_stats.size());
            for (auto &&item : _host_stats) { host_stats.emplace_back(item); }
        }
        auto by_total = [](auto &&lhs, auto &&rhs) noexcept {
            return lhs.second.total_ms > rhs.second.total_ms;
        };
        std::sort(gpu_stats.begin(), gpu_stats.end(), by_total);
        std::sort(host_stats.begin(), host_stats.end(), by_total);
        for (auto &&[stage, profile] : gpu_stats) {
            std::fprintf(
                stderr,
                "LUISA_METAL_COMMAND_BUFFER_PROFILE stage='%s' dispatches=%llu "
                "total_ms=%.6f average_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
                stage.c_str(),
                static_cast<unsigned long long>(profile.count),
                profile.total_ms,
                profile.total_ms / static_cast<double>(profile.count),
                profile.min_ms,
                profile.max_ms);
        }
        for (auto &&[operation, profile] : host_stats) {
            std::fprintf(
                stderr,
                "LUISA_METAL_STREAM_PROFILE operation='%s' calls=%llu "
                "total_ms=%.6f average_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
                operation.c_str(),
                static_cast<unsigned long long>(profile.count),
                profile.total_ms,
                profile.total_ms / static_cast<double>(profile.count),
                profile.min_ms,
                profile.max_ms);
        }
    }

    void record_gpu(const char *stage, double elapsed_ms) noexcept {
        std::scoped_lock lock{_mutex};
        _record(_gpu_stats, stage, elapsed_ms);
    }

    void record_host(const char *operation, double elapsed_ms) noexcept {
        std::scoped_lock lock{_mutex};
        _record(_host_stats, operation, elapsed_ms);
    }
};

[[nodiscard]] bool metal_command_buffer_profiling_enabled() noexcept {
    static const auto enabled =
        std::getenv("LUISA_METAL_COMMAND_BUFFER_PROFILE") != nullptr;
    return enabled;
}

[[nodiscard]] MetalCommandBufferProfiler &metal_command_buffer_profiler() noexcept {
    static MetalCommandBufferProfiler profiler;
    return profiler;
}

[[nodiscard]] double elapsed_milliseconds(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) noexcept {
    return std::chrono::duration<double, std::milli>{end - begin}.count();
}

}// namespace

MetalStream::MetalStream(MTL::Device *device,
                         size_t max_commands) noexcept
    : _queue{max_commands == 0u ?
                 device->newCommandQueue() :
                 device->newCommandQueue(max_commands)} {}

MetalStream::~MetalStream() noexcept {
    _queue->release();
}

MetalStageBufferPool *MetalStream::upload_pool() noexcept {
    {
        std::scoped_lock lock{_upload_pool_creation_mutex};
        if (_upload_pool == nullptr) {
            _upload_pool = luisa::make_unique<MetalStageBufferPool>(
                _queue->device(), 64_M, true);
        }
    }
    return _upload_pool.get();
}

MetalStageBufferPool *MetalStream::download_pool() noexcept {
    {
        std::scoped_lock lock{_download_pool_creation_mutex};
        if (_download_pool == nullptr) {
            _download_pool = luisa::make_unique<MetalStageBufferPool>(
                _queue->device(), 32_M, false);
        }
    }
    return _download_pool.get();
}

void MetalStream::signal(MetalEvent *event, uint64_t value) noexcept {
    auto command_buffer = _queue->commandBufferWithUnretainedReferences();
    event->signal(command_buffer, value);
    CallbackContainer callbacks;
    callbacks.emplace_back(event->host_signal_callback(value));
    submit(command_buffer, std::move(callbacks));
}

void MetalStream::wait(MetalEvent *event, uint64_t value) noexcept {
    auto command_buffer = _queue->commandBufferWithUnretainedReferences();
    event->wait(command_buffer, value);
    submit(command_buffer, {});
}

void MetalStream::synchronize() noexcept {
    auto profile = metal_command_buffer_profiling_enabled();
    auto synchronize_begin = profile ?
                                 std::chrono::steady_clock::now() :
                                 std::chrono::steady_clock::time_point{};
    auto command_buffer = _queue->commandBufferWithUnretainedReferences();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    auto command_buffer_done = profile ?
                                   std::chrono::steady_clock::now() :
                                   std::chrono::steady_clock::time_point{};
    if (auto error = command_buffer->error()) {
        LUISA_ERROR_WITH_LOCATION(
            "Metal synchronization command buffer failed: {}.",
            error->localizedDescription()->utf8String());
    }
    auto callback_target =
        _submitted_callback_lists.load(std::memory_order_acquire);
    while (_completed_callback_lists.load(std::memory_order_acquire) <
           callback_target) {
        std::this_thread::yield();
    }
    if (profile) {
        auto synchronize_done = std::chrono::steady_clock::now();
        auto &profiler = metal_command_buffer_profiler();
        profiler.record_host(
            "synchronize_total",
            elapsed_milliseconds(synchronize_begin, synchronize_done));
        profiler.record_host(
            "synchronize_wait_until_completed",
            elapsed_milliseconds(synchronize_begin, command_buffer_done));
        profiler.record_host(
            "synchronize_callback_drain",
            elapsed_milliseconds(command_buffer_done, synchronize_done));
    }
}

void MetalStream::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        _queue->setLabel(nullptr);
    } else {
        auto mtl_name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
        _queue->setLabel(mtl_name);
        mtl_name->release();
    }
}

void MetalStream::_encode(MetalCommandEncoder &encoder,
                          Command *command) noexcept {
    command->accept(encoder);
}

void MetalStream::_do_dispatch(MetalCommandEncoder &encoder,
                               CommandList &&list) noexcept {
    if (list.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "MetalStream::dispatch: Command list is empty.");
    } else {
        auto commands = list.steal_commands();
        auto callbacks = list.steal_callbacks();
        {
            std::scoped_lock lock{_dispatch_mutex};
            for (auto &command : commands) { _encode(encoder, command.get()); }
            encoder.submit(std::move(callbacks));
        }
    }
}

void MetalStream::dispatch(CommandList &&list) noexcept {
    MetalCommandEncoder encoder{this};
    _do_dispatch(encoder, std::move(list));
}

void MetalStream::present(MetalSwapchain *swapchain, MetalTexture *image) noexcept {
    swapchain->present(_queue, image->handle());
}

void MetalStream::submit(MTL::CommandBuffer *command_buffer,
                         MetalStream::CallbackContainer &&callbacks) noexcept {
    if (!callbacks.empty()) {
        {
            std::scoped_lock lock{_callback_mutex};
            _callback_lists.emplace(std::move(callbacks));
            _submitted_callback_lists.fetch_add(
                1u, std::memory_order_release);
        }
        command_buffer->addCompletedHandler(^(MTL::CommandBuffer *) noexcept {
            auto self = this;
            std::scoped_lock execution_lock{
                self->_callback_execution_mutex};
            auto callbacks = [self] {
                std::scoped_lock lock{self->_callback_mutex};
                if (self->_callback_lists.empty()) {
                    LUISA_WARNING_WITH_LOCATION(
                        "MetalStream::submit: Callback list is empty.");
                    return CallbackContainer{};
                }
                auto callbacks = std::move(self->_callback_lists.front());
                self->_callback_lists.pop();
                return callbacks;
            }();
            for (auto callback : callbacks) { callback->recycle(); }
            self->_completed_callback_lists.fetch_add(
                1u, std::memory_order_release);
        });
    }
    command_buffer->addCompletedHandler(^(MTL::CommandBuffer *cb) noexcept {
#ifndef NDEBUG
        if (auto logs = cb->logs()) {
            luisa_compute_metal_stream_print_function_logs(logs);
        }
#endif
        if (auto error = cb->error()) {
            LUISA_ERROR_WITH_LOCATION(
                "Metal command buffer execution failed: {}.",
                error->localizedDescription()->utf8String());
        }
    });
    if (metal_command_buffer_profiling_enabled()) {
        command_buffer->addCompletedHandler(^(MTL::CommandBuffer *cb) noexcept {
            auto begin = cb->GPUStartTime();
            auto end = cb->GPUEndTime();
            auto label = cb->label();
            if (end > begin) {
                metal_command_buffer_profiler().record_gpu(
                    label == nullptr ? "<unlabeled>" : label->utf8String(),
                    (end - begin) * 1.0e3);
            }
        });
    }
    command_buffer->commit();
}

}// namespace luisa::compute::metal
