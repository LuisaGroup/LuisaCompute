#include <luisa/core/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
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
    bool _scope_started{false};

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
    void begin_scope_once() noexcept {
        std::scoped_lock lock{_mutex};
        if (_scope_started) { return; }
        _gpu_stats.clear();
        _host_stats.clear();
        _scope_started = true;
    }

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

[[nodiscard]] const char *metal_command_buffer_profile_start_stage() noexcept {
    static const auto stage =
        std::getenv("LUISA_METAL_COMMAND_BUFFER_PROFILE_START_STAGE");
    return stage;
}

[[nodiscard]] double elapsed_milliseconds(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) noexcept {
    return std::chrono::duration<double, std::milli>{end - begin}.count();
}

inline constexpr auto shader_log_subsystem =
    std::string_view{"org.luisa.compute"};
inline constexpr auto shader_log_category =
    std::string_view{"shader"};
inline constexpr auto shader_log_bool_prefix =
    std::string_view{"__luisa_metal_bool_"};
inline constexpr auto shader_log_bool_suffix =
    std::string_view{"__"};

[[nodiscard]] luisa::string normalize_shader_log_message(
    std::string_view message) noexcept {
    luisa::string normalized;
    normalized.reserve(message.size());
    while (!message.empty()) {
        auto marker = message.find(shader_log_bool_prefix);
        if (marker == std::string_view::npos) {
            normalized.append(message);
            break;
        }
        normalized.append(message.substr(0u, marker));
        message.remove_prefix(marker + shader_log_bool_prefix.size());
        if (message.starts_with("0__")) {
            normalized.append("false");
            message.remove_prefix(1u + shader_log_bool_suffix.size());
        } else if (message.starts_with("1__")) {
            normalized.append("true");
            message.remove_prefix(1u + shader_log_bool_suffix.size());
        } else {
            normalized.append(shader_log_bool_prefix);
        }
    }
    return normalized;
}

}// namespace

MetalStream::MetalStream(MTL::Device *device,
                         size_t max_commands) noexcept
    : _max_commands{max_commands} {
    auto log_descriptor = NS::TransferPtr(
        MTL::LogStateDescriptor::alloc()->init());
    log_descriptor->setBufferSize(1_M);
    log_descriptor->setLevel(MTL::LogLevelDebug);
    NS::Error *error = nullptr;
    _log_state = device->newLogState(log_descriptor.get(), &error);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to create Metal shader log state: {}.",
            error->localizedDescription()->utf8String());
    }
    LUISA_ASSERT(_log_state != nullptr,
                 "Failed to create Metal shader log state.");
    _log_state->addLogHandler(MTL::LogHandlerFunction{
        [this](NS::String *subsystem, NS::String *category,
               MTL::LogLevel, NS::String *message) noexcept {
            _emit_shader_log(subsystem, category, message);
        }});
    auto queue_descriptor = NS::TransferPtr(
        MTL4::CommandQueueDescriptor::alloc()->init());
    NS::Error *queue_error = nullptr;
    _queue = device->newMTL4CommandQueue(
        queue_descriptor.get(), &queue_error);
    if (queue_error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to create Metal4 command queue: {}.",
            queue_error->localizedDescription()->utf8String());
    }
    LUISA_ASSERT(_queue != nullptr,
                 "Failed to create Metal4 command queue.");
    if (!device->supportsFamily(MTL::GPUFamilyApple9)) {
        _acceleration_structure_compatibility_queue =
            device->newCommandQueue();
        LUISA_ASSERT(
            _acceleration_structure_compatibility_queue != nullptr,
            "Failed to create Metal acceleration-structure compatibility "
            "queue.");
        LUISA_INFO(
            "Metal4 address-driven acceleration-structure builds require "
            "Apple9 or newer; using the synchronized Metal AS build bridge "
            "on '{}'. Shader compilation and dispatch remain Metal4 AIR.",
            device->name()->utf8String());
    }
}

MetalStream::~MetalStream() noexcept {
    synchronize();
    if (_acceleration_structure_compatibility_queue != nullptr) {
        _acceleration_structure_compatibility_queue->release();
    }
    _queue->release();
    _log_state->release();
    if (_name != nullptr) { _name->release(); }
}

void MetalStream::_emit_shader_log(
    NS::String *subsystem, NS::String *category,
    NS::String *message) const noexcept {
    if (subsystem == nullptr || category == nullptr || message == nullptr ||
        std::string_view{subsystem->utf8String()} != shader_log_subsystem ||
        std::string_view{category->utf8String()} != shader_log_category) {
        return;
    }
    auto normalized = normalize_shader_log_message(
        message->utf8String());
    LogCallback callback;
    {
        std::scoped_lock lock{_log_callback_mutex};
        callback = _log_callback;
    }
    if (callback) {
        callback(normalized);
    } else {
        LUISA_INFO("[DEVICE] {}", normalized);
    }
}

void MetalStream::set_log_callback(LogCallback callback) noexcept {
    std::scoped_lock lock{_log_callback_mutex};
    _log_callback = std::move(callback);
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
    _queue->signalEvent(event->handle(), value);
    MetalCommandEncoder encoder{this};
    encoder.compute_encoder()->endEncoding();
    encoder.add_callback(event->host_signal_callback(value));
    static_cast<void>(encoder.submit({}));
}

void MetalStream::wait(MetalEvent *event, uint64_t value) noexcept {
    if (value == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "MetalEvent::wait() is called before any signal event.");
    } else {
        _queue->wait(event->handle(), value);
    }
}

void MetalStream::synchronize() noexcept {
    auto profile = metal_command_buffer_profiling_enabled();
    auto synchronize_begin = profile ?
                                 std::chrono::steady_clock::now() :
                                 std::chrono::steady_clock::time_point{};
    MetalCommandEncoder encoder{this};
    encoder.compute_encoder()->endEncoding();
    encoder.submit_and_wait();
    auto command_buffer_done = profile ?
                                   std::chrono::steady_clock::now() :
                                   std::chrono::steady_clock::time_point{};
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
    if (_name != nullptr) {
        _name->release();
        _name = nullptr;
    }
    if (!name.empty()) {
        _name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
    }
    if (_acceleration_structure_compatibility_queue != nullptr) {
        _acceleration_structure_compatibility_queue->setLabel(_name);
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
            static_cast<void>(encoder.submit(std::move(callbacks)));
        }
    }
}

void MetalStream::dispatch(CommandList &&list) noexcept {
    MetalCommandEncoder encoder{this};
    _do_dispatch(encoder, std::move(list));
}

void MetalStream::present(MetalSwapchain *swapchain, MetalTexture *image) noexcept {
    swapchain->present(this, image->handle());
}

MetalStream::SubmissionHandle MetalStream::submit(
    MTL4::CommandBuffer *command_buffer,
    MTL4::CommandAllocator *command_allocator,
    MetalStream::CallbackContainer &&callbacks) noexcept {
    LUISA_ASSERT(command_buffer != nullptr && command_allocator != nullptr,
                 "Invalid Metal4 command-buffer submission.");
    auto submission = luisa::make_shared<Submission>();
    auto has_callbacks = !callbacks.empty();
    if (!callbacks.empty()) {
        {
            std::scoped_lock lock{_callback_mutex};
            _callback_lists.emplace(std::move(callbacks));
            _submitted_callback_lists.fetch_add(
                1u, std::memory_order_release);
        }
    }
    auto label = command_buffer->label() == nullptr ?
                     luisa::string{"<unlabeled>"} :
                     luisa::string{command_buffer->label()->utf8String()};
    if (_max_commands != 0u) {
        std::unique_lock lock{_inflight_mutex};
        _inflight_cv.wait(lock, [this]() noexcept {
            return _inflight_commands < _max_commands;
        });
        _inflight_commands++;
    }
    auto options = NS::TransferPtr(
        MTL4::CommitOptions::alloc()->init());
    options->addFeedbackHandler(MTL4::CommitFeedbackHandlerFunction{
        [this, command_buffer, command_allocator, submission,
         has_callbacks, label = std::move(label)](
            MTL4::CommitFeedback *feedback) noexcept {
            if (has_callbacks) {
                std::scoped_lock execution_lock{
                    _callback_execution_mutex};
                auto callbacks = [this] {
                    std::scoped_lock lock{_callback_mutex};
                    if (_callback_lists.empty()) {
                        LUISA_WARNING_WITH_LOCATION(
                            "MetalStream::submit: Callback list is empty.");
                        return CallbackContainer{};
                    }
                    auto callbacks = std::move(_callback_lists.front());
                    _callback_lists.pop();
                    return callbacks;
                }();
                for (auto callback : callbacks) { callback->recycle(); }
                _completed_callback_lists.fetch_add(
                    1u, std::memory_order_release);
            }
            if (metal_command_buffer_profiling_enabled()) {
                auto begin = feedback->GPUStartTime();
                auto end = feedback->GPUEndTime();
                if (end > begin) {
                    auto &profiler = metal_command_buffer_profiler();
                    auto start_stage =
                        metal_command_buffer_profile_start_stage();
                    if (start_stage != nullptr && label == start_stage) {
                        profiler.begin_scope_once();
                    }
                    profiler.record_gpu(
                        label.c_str(), (end - begin) * 1.0e3);
                }
            }
            auto error = feedback->error();
            auto error_message = error == nullptr ?
                                     luisa::string{} :
                                     luisa::string{error->localizedDescription()->utf8String()};
            command_allocator->reset();
            command_allocator->release();
            command_buffer->release();
            if (_max_commands != 0u) {
                {
                    std::scoped_lock lock{_inflight_mutex};
                    LUISA_ASSERT(_inflight_commands != 0u,
                                 "Invalid Metal4 in-flight command count.");
                    _inflight_commands--;
                }
                _inflight_cv.notify_one();
            }
            if (!error_message.empty()) {
                LUISA_ERROR_WITH_LOCATION(
                    "Metal4 command-buffer execution failed: {}.",
                    error_message);
            }
            // Publish completion last. A synchronous waiter may own the final
            // MetalStream reference and destroy the stream as soon as this
            // becomes true, so the feedback handler must not touch `this`
            // afterwards.
            submission->completed.store(true, std::memory_order_release);
        }});
    const MTL4::CommandBuffer *command_buffers[]{command_buffer};
    _queue->commit(command_buffers, 1u, options.get());
    return submission;
}

}// namespace luisa::compute::metal
