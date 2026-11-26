#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/string.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "metal_device.h"
#include "metal_event.h"
#include "metal_texture.h"
#include "metal_swapchain.h"
#include "metal_shader.h"
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
        luisa::sort(gpu_stats.begin(), gpu_stats.end(), by_total);
        luisa::sort(host_stats.begin(), host_stats.end(), by_total);
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
        std::getenv("LUISA_METAL_COMMAND_BUFFER_PROFILE") != nullptr ||
        std::getenv("LUISA_METAL4_DISPATCH_PROFILE") != nullptr;
    return enabled;
}

[[nodiscard]] bool metal4_dispatch_profiling_enabled() noexcept {
    static const auto enabled =
        std::getenv("LUISA_METAL4_DISPATCH_PROFILE") != nullptr;
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
    luisa::string_view{"org.luisa.compute"};
inline constexpr auto shader_log_category =
    luisa::string_view{"shader"};
inline constexpr auto shader_log_bool_prefix =
    luisa::string_view{"__luisa_metal_bool_"};
inline constexpr auto shader_log_bool_suffix =
    luisa::string_view{"__"};

[[nodiscard]] luisa::string normalize_shader_log_message(
    luisa::string_view message) noexcept {
    luisa::string normalized;
    normalized.reserve(message.size());
    while (!message.empty()) {
        auto marker = message.find(shader_log_bool_prefix);
        if (marker == luisa::string_view::npos) {
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
                         size_t max_commands,
                         MetalDevice *owner_device) noexcept
    : _owner_device{owner_device}, _max_commands{max_commands} {
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
    _command_buffer_options =
        MTL4::CommandBufferOptions::alloc()->init();
    LUISA_ASSERT(_command_buffer_options != nullptr,
                 "Failed to create Metal4 command-buffer options.");
    _command_buffer_options->setLogState(_log_state);
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
    _completion_event = device->newSharedEvent();
    LUISA_ASSERT(_completion_event != nullptr,
                 "Failed to create Metal4 stream completion shared event.");
    {
        auto residency_desc = NS::TransferPtr(
            MTL::ResidencySetDescriptor::alloc()->init());
        residency_desc->setInitialCapacity(4u);
        residency_desc->setLabel(MTLSTR("LuisaCompute Metal 4 Stream ResidencySet"));
        NS::Error *residency_error = nullptr;
        _stream_residency_set = device->newResidencySet(
            residency_desc.get(), &residency_error);
        if (_stream_residency_set != nullptr) {
            _queue->addResidencySet(_stream_residency_set);
        }
    }
    if (_owner_device != nullptr && _owner_device->residency_set() != nullptr) {
        _queue->addResidencySet(_owner_device->residency_set());
    }
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
    if (metal4_dispatch_profiling_enabled()) {
        MetalTimingExt timing_ext{device};
        static_cast<void>(timing_ext.begin_sample(
            reinterpret_cast<uint64_t>(this), 1u, 2048u, true));
    }
}

MetalStream::~MetalStream() noexcept {
    if (metal4_dispatch_profiling_enabled() && timing_enabled()) {
        auto sample = end_timing();
        struct KernelStat {
            uint64_t count{0};
            double total_ms{0.0};
            double max_ms{0.0};
            uint64_t total_threads{0};
            uint3 block{};
        };
        std::unordered_map<uint64_t, KernelStat> stats;
        double total_dispatch_gpu_ms = 0.0;
        for (const auto &d : sample.dispatches) {
            if (!d.valid) { continue; }
            auto ms = d.elapsed_nanoseconds * 1.0e-6;
            total_dispatch_gpu_ms += ms;
            auto &s = stats[d.shader_checksum];
            s.count++;
            s.total_ms += ms;
            s.max_ms = std::max(s.max_ms, ms);
            s.total_threads += uint64_t{d.dispatch_size.x} * d.dispatch_size.y * d.dispatch_size.z;
            s.block = d.block_size;
        }
        std::vector<std::pair<uint64_t, KernelStat>> sorted(stats.begin(), stats.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
            return a.second.total_ms > b.second.total_ms;
        });
        if (!sorted.empty()) {
            std::fprintf(stderr, "=== LUISA_METAL4_DISPATCH_PROFILE (dispatches=%zu, total_kernel_gpu_ms=%.3f, cbs=%zu, err='%s') ===\n",
                         sample.dispatches.size(), total_dispatch_gpu_ms,
                         sample.command_buffers.size(), sample.error.c_str());
            for (const auto &[cksum, s] : sorted) {
                auto name = MetalShader::lookup_debug_name(cksum);
                auto pct = total_dispatch_gpu_ms > 0.0 ? (100.0 * s.total_ms / total_dispatch_gpu_ms) : 0.0;
                std::fprintf(stderr,
                             "  KERNEL %-42s calls=%5llu gpu_ms=%9.3f (%5.1f%%) avg_us=%8.2f max_ms=%7.3f block=%ux%ux%u avg_threads=%.0f\n",
                             name.c_str(),
                             static_cast<unsigned long long>(s.count),
                             s.total_ms, pct,
                             (s.total_ms * 1000.0) / static_cast<double>(s.count),
                             s.max_ms,
                             s.block.x, s.block.y, s.block.z,
                             static_cast<double>(s.total_threads) / static_cast<double>(s.count));
            }
        }
    }
    synchronize();
    while (_active_feedback_handlers.load(std::memory_order_acquire) != 0u) {
        std::this_thread::yield();
    }
    for (auto bundle : _command_allocator_pool) {
        if (bundle.argument_table != nullptr) { bundle.argument_table->release(); }
        if (bundle.allocator != nullptr) { bundle.allocator->release(); }
    }
    if (_owner_device != nullptr && _owner_device->residency_set() != nullptr) {
        _queue->removeResidencySet(_owner_device->residency_set());
    }
    if (_stream_residency_set != nullptr) {
        _queue->removeResidencySet(_stream_residency_set);
        _stream_residency_set->endResidency();
        _stream_residency_set->release();
        _stream_residency_set = nullptr;
    }
    _command_buffer_options->release();
    if (_acceleration_structure_compatibility_queue != nullptr) {
        _acceleration_structure_compatibility_queue->release();
    }
    if (_completion_event != nullptr) { _completion_event->release(); }
    _queue->release();
    _log_state->release();
    if (_name != nullptr) { _name->release(); }
}

size_t MetalStream::_command_allocator_pool_capacity() const noexcept {
    // Ordinary streams cap in-flight submissions at four by default. Keep a
    // small bounded reserve for streams whose in-flight limit is disabled.
    return std::max(_max_commands, static_cast<size_t>(16u));
}

MetalStream::CommandAllocatorBundle MetalStream::_acquire_command_allocator() noexcept {
    {
        std::scoped_lock lock{_command_allocator_pool_mutex};
        if (!_command_allocator_pool.empty()) {
            auto bundle = _command_allocator_pool.back();
            _command_allocator_pool.pop_back();
            return bundle;
        }
    }
    auto command_allocator = device()->newCommandAllocator();
    LUISA_ASSERT(command_allocator != nullptr,
                 "Failed to create Metal4 command allocator.");
    auto descriptor = NS::TransferPtr(
        MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setInitializeBindings(true);
    descriptor->setMaxBufferBindCount(pooled_argument_table_buffer_slots);
    descriptor->setMaxTextureBindCount(pooled_argument_table_texture_slots);
    descriptor->setMaxSamplerStateBindCount(0u);
    NS::Error *error = nullptr;
    auto argument_table = device()->newArgumentTable(descriptor.get(), &error);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to create pooled Metal4 argument table: {}.",
            error->localizedDescription()->utf8String());
    }
    LUISA_ASSERT(argument_table != nullptr,
                 "Failed to create pooled Metal4 argument table.");
    return CommandAllocatorBundle{
        .allocator = command_allocator,
        .argument_table = argument_table};
}

void MetalStream::_recycle_command_allocator(
    CommandAllocatorBundle bundle) noexcept {
    LUISA_ASSERT(bundle.allocator != nullptr,
                 "Cannot recycle a null Metal4 command allocator.");
    bundle.allocator->reset();
    {
        std::scoped_lock lock{_command_allocator_pool_mutex};
        if (_command_allocator_pool.size() <
            _command_allocator_pool_capacity()) {
            _command_allocator_pool.emplace_back(bundle);
            return;
        }
    }
    if (bundle.argument_table != nullptr) { bundle.argument_table->release(); }
    bundle.allocator->release();
}

void MetalStream::_register_stream_allocation(
    const MTL::Allocation *allocation) noexcept {
    if (allocation == nullptr || _stream_residency_set == nullptr) { return; }
    std::scoped_lock lock{_stream_residency_mutex};
    _stream_residency_set->addAllocation(allocation);
    _stream_residency_set->commit();
    _stream_residency_set->requestResidency();
}

void MetalStream::_emit_shader_log(
    NS::String *subsystem, NS::String *category,
    NS::String *message) const noexcept {
    if (subsystem == nullptr || category == nullptr || message == nullptr ||
        luisa::string_view{subsystem->utf8String()} != shader_log_subsystem ||
        luisa::string_view{category->utf8String()} != shader_log_category) {
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
            _register_stream_allocation(_upload_pool->backing_buffer());
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
            _register_stream_allocation(_download_pool->backing_buffer());
        }
    }
    return _download_pool.get();
}

void MetalStream::_wait_for_completion_ticket(uint64_t target) noexcept {
    if (target == 0u ||
        _retired_ticket.load(std::memory_order_acquire) >= target ||
        _completion_event->signaledValue() >= target) {
        return;
    }
    auto spin_start = std::chrono::steady_clock::now();
    for (;;) {
        for (auto i = 0u; i < 64u; ++i) {
            if (_completion_event->signaledValue() >= target) { return; }
#if defined(__aarch64__) || defined(_M_ARM64)
            asm volatile("yield" ::: "memory");
#else
            std::this_thread::yield();
#endif
        }
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - spin_start)
                              .count();
        if (elapsed_us >= 3000) { break; }
    }
    while (_completion_event->signaledValue() < target) {
        static_cast<void>(_completion_event->waitUntilSignaledValue(target, 10u));
    }
}

void MetalStream::_retire_submissions_up_to(
    uint64_t target, bool blocking) noexcept {
    auto self_id = std::this_thread::get_id();
    if (_retiring_thread.load(std::memory_order_relaxed) == self_id) {
        return;
    }
    std::unique_lock<std::mutex> retire_lock{_retire_mutex, std::defer_lock};
    if (blocking) {
        retire_lock.lock();
    } else if (!retire_lock.try_lock()) {
        return;
    }
    _retiring_thread.store(self_id, std::memory_order_relaxed);
    struct ClearRetiringThread {
        std::atomic<std::thread::id> &slot;
        ~ClearRetiringThread() noexcept {
            slot.store(std::thread::id{}, std::memory_order_relaxed);
        }
    } clear_guard{_retiring_thread};

    if (target != 0u &&
        _retired_ticket.load(std::memory_order_acquire) >= target) {
        return;
    }
    auto signaled = _completion_event->signaledValue();
    auto limit = target == 0u ? signaled : std::min(target, signaled);
    if (_retired_ticket.load(std::memory_order_relaxed) >= limit) {
        return;
    }
    for (;;) {
        PendingSubmission item;
        {
            std::scoped_lock sub_lock{_submission_mutex};
            if (_pending_submissions.empty() ||
                _pending_submissions.front().ticket > limit) {
                break;
            }
            item = std::move(_pending_submissions.front());
            _pending_submissions.pop();
        }
        for (auto callback : item.callbacks) { callback->recycle(); }
        _recycle_command_allocator(item.allocator_bundle);
        item.command_buffer->release();
        _retired_ticket.store(item.ticket, std::memory_order_release);
        if (item.submission) {
            item.submission->completed.store(true, std::memory_order_release);
            item.submission->completed.notify_all();
        }
    }
}

void MetalStream::wait_submission(const SubmissionHandle &submission) noexcept {
    if (!submission) { return; }
    if (submission->ticket != 0u) {
        _wait_for_completion_ticket(submission->ticket);
        _retire_submissions_up_to(submission->ticket, true);
    } else {
        submission->completed.wait(false, std::memory_order_acquire);
    }
}

void MetalStream::signal(MetalEvent *event, uint64_t value) noexcept {
    {
        std::scoped_lock sub_lock{_submission_mutex};
        _has_unsubmitted_queue_ops.store(true, std::memory_order_release);
        _queue->signalEvent(event->handle(), value);
    }
    MetalCommandEncoder encoder{this};
    encoder.compute_encoder()->endEncoding();
    encoder.add_callback(event->host_signal_callback(value));
    encoder.require_async_host_completion();
    static_cast<void>(encoder.submit({}));
}

void MetalStream::wait(MetalEvent *event, uint64_t value) noexcept {
    if (value == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "MetalEvent::wait() is called before any signal event.");
    } else {
        std::scoped_lock sub_lock{_submission_mutex};
        _has_unsubmitted_queue_ops.store(true, std::memory_order_release);
        _queue->wait(event->handle(), value);
    }
}

void MetalStream::synchronize() noexcept {
    auto profile = metal_command_buffer_profiling_enabled();
    auto synchronize_begin = profile ?
                                 std::chrono::steady_clock::now() :
                                 std::chrono::steady_clock::time_point{};
    if (timing_enabled() || _has_unsubmitted_queue_ops.load(std::memory_order_acquire)) {
        MetalCommandEncoder encoder{this};
        encoder.compute_encoder()->endEncoding();
        encoder.submit_and_wait();
    } else {
        auto target = _submitted_ticket.load(std::memory_order_acquire);
        if (_retired_ticket.load(std::memory_order_acquire) < target) {
            _wait_for_completion_ticket(target);
            _retire_submissions_up_to(target, true);
        }
        std::unique_lock lock{_inflight_mutex};
        if (_inflight_commands != 0u) {
            _inflight_cv.wait(lock, [this]() noexcept {
                return _inflight_commands == 0u;
            });
        }
    }
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
    if (command->tag() != Command::Tag::EShaderDispatchCommand) { encoder.note_timing_non_dispatch_work(); }
    command->accept(encoder);
}

void MetalStream::_wait_if_overlapping_host_download(
    luisa::span<const luisa::unique_ptr<Command>> cmds) noexcept {
    if (cmds.empty()) { return; }
    luisa::fixed_vector<HostByteRange, 8u> upload_ranges;
    for (const auto &cmd : cmds) {
        if (auto range = command_host_upload_range(cmd.get()); !range.empty()) {
            upload_ranges.push_back(range);
        }
    }
    if (upload_ranges.empty()) { return; }
    luisa::fixed_vector<SubmissionHandle, 8u> wait_submissions;
    {
        auto retired = _retired_ticket.load(std::memory_order_acquire);
        std::scoped_lock lock{_host_download_mutex};
        erase_host_transfer_if(
            _pending_host_downloads,
            [retired](const PendingHostDownload &entry) noexcept {
                if (!entry.submission) { return true; }
                if (entry.submission->completed.load(std::memory_order_acquire)) { return true; }
                return entry.submission->ticket != 0u && entry.submission->ticket <= retired;
            });
        for (const auto &entry : _pending_host_downloads) {
            for (const auto &upload : upload_ranges) {
                if (entry.range.overlaps(upload)) {
                    if (std::find(wait_submissions.begin(), wait_submissions.end(), entry.submission) ==
                        wait_submissions.end()) {
                        wait_submissions.push_back(entry.submission);
                    }
                    break;
                }
            }
        }
    }
    for (const auto &submission : wait_submissions) {
        wait_submission(submission);
    }
    if (!wait_submissions.empty()) {
        auto retired = _retired_ticket.load(std::memory_order_acquire);
        std::scoped_lock lock{_host_download_mutex};
        erase_host_transfer_if(
            _pending_host_downloads,
            [retired](const PendingHostDownload &entry) noexcept {
                if (!entry.submission) { return true; }
                if (entry.submission->completed.load(std::memory_order_acquire)) { return true; }
                return entry.submission->ticket != 0u && entry.submission->ticket <= retired;
            });
    }
}

void MetalStream::_record_pending_host_downloads(
    luisa::span<const luisa::unique_ptr<Command>> cmds,
    const SubmissionHandle &submission) noexcept {
    if (cmds.empty() || !submission) { return; }
    std::scoped_lock lock{_host_download_mutex};
    for (const auto &cmd : cmds) {
        if (auto range = command_host_download_range(cmd.get()); !range.empty()) {
            _pending_host_downloads.emplace_back(
                PendingHostDownload{range, submission});
        }
    }
}

void MetalStream::_do_dispatch(MetalCommandEncoder &encoder,
                               CommandList &&list) noexcept {
    if (list.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "MetalStream::dispatch: Command list is empty.");
    } else {
        auto commands = list.steal_commands();
        auto callbacks = list.steal_callbacks();
        if (commands.empty()) {
            std::scoped_lock lock{_dispatch_mutex};
            static_cast<void>(encoder.submit(std::move(callbacks)));
            return;
        }
        size_t cursor = 0u;
        while (cursor < commands.size()) {
            auto next = find_next_host_transfer_split(
                luisa::span{commands.data(), commands.size()}, cursor);
            auto chunk = luisa::span{commands.data() + cursor, next - cursor};
            bool is_last = (next == commands.size());
            _wait_if_overlapping_host_download(chunk);
            auto first_tail_download = chunk.size();
            while (first_tail_download > 0u &&
                   chunk[first_tail_download - 1u]->tag() ==
                       Command::Tag::EBufferDownloadCommand) {
                --first_tail_download;
            }
            {
                std::scoped_lock lock{_dispatch_mutex};
                bool has_downloads = false;
                for (size_t i = 0u; i < chunk.size(); ++i) {
                    if (!command_host_download_range(chunk[i].get()).empty()) {
                        has_downloads = true;
                    }
                    encoder.set_in_tail_downloads(i >= first_tail_download);
                    _encode(encoder, chunk[i].get());
                }
                encoder.set_in_tail_downloads(false);
                auto submission = encoder.submit(
                    is_last ? std::move(callbacks) : CommandList::CallbackContainer{});
                if (has_downloads && submission) {
                    _record_pending_host_downloads(chunk, submission);
                }
            }
            cursor = next;
        }
    }
}

void MetalStream::dispatch(CommandList &&list) noexcept {
    auto profile = metal_command_buffer_profiling_enabled();
    auto begin = profile ? std::chrono::steady_clock::now() :
                           std::chrono::steady_clock::time_point{};
    MetalCommandEncoder encoder{this};
    _do_dispatch(encoder, std::move(list));
    if (profile) {
        metal_command_buffer_profiler().record_host(
            "dispatch_encode_and_submit",
            elapsed_milliseconds(begin, std::chrono::steady_clock::now()));
    }
}

void MetalStream::present(MetalSwapchain *swapchain, MetalTexture *image) noexcept {
    swapchain->present(this, image->handle());
}

MetalStream::SubmissionHandle MetalStream::submit(
    MTL4::CommandBuffer *command_buffer,
    CommandAllocatorBundle allocator_bundle,
    MetalStream::CallbackContainer &&callbacks,
    MetalTimingSubmission timing,
    bool async_host_completion) noexcept {
    LUISA_ASSERT(command_buffer != nullptr && allocator_bundle.allocator != nullptr,
                 "Invalid Metal4 command-buffer submission.");
    if (_owner_device != nullptr) {
        _owner_device->commit_residency_if_needed();
    }
    auto profile_cb = metal_command_buffer_profiling_enabled();
    if (!timing.session && !profile_cb) {
        _retire_submissions_up_to(0u, false);
        if (_max_commands != 0u) {
            for (;;) {
                uint64_t wait_ticket = 0u;
                {
                    std::scoped_lock sub_lock{_submission_mutex};
                    if (_pending_submissions.size() < _max_commands) { break; }
                    wait_ticket = _pending_submissions.front().ticket;
                }
                _wait_for_completion_ticket(wait_ticket);
                _retire_submissions_up_to(wait_ticket, true);
            }
        }
        auto submission = luisa::make_shared<Submission>();
        {
            std::scoped_lock sub_lock{_submission_mutex};
            auto ticket = _submitted_ticket.load(std::memory_order_relaxed) + 1u;
            submission->ticket = ticket;
            const MTL4::CommandBuffer *command_buffers[]{command_buffer};
            if (async_host_completion) {
                _active_feedback_handlers.fetch_add(1u, std::memory_order_release);
                auto options = NS::TransferPtr(
                    MTL4::CommitOptions::alloc()->init());
                options->addFeedbackHandler(MTL4::CommitFeedbackHandlerFunction{
                    [this, ticket](MTL4::CommitFeedback *feedback) noexcept {
                        auto error = feedback->error();
                        auto error_message =
                            error == nullptr ?
                                luisa::string{} :
                                luisa::string{error->localizedDescription()->utf8String()};
                        if (!error_message.empty()) {
                            LUISA_ERROR_WITH_LOCATION(
                                "Metal4 command-buffer execution failed: {}.",
                                error_message);
                        }
                        _wait_for_completion_ticket(ticket);
                        _retire_submissions_up_to(ticket, true);
                        _active_feedback_handlers.fetch_sub(
                            1u, std::memory_order_release);
                    }});
                _queue->commit(command_buffers, 1u, options.get());
            } else {
                _queue->commit(command_buffers, 1u);
            }
            _queue->signalEvent(_completion_event, ticket);
            _submitted_ticket.store(ticket, std::memory_order_release);
            _has_unsubmitted_queue_ops.store(false, std::memory_order_relaxed);
            _pending_submissions.emplace(PendingSubmission{
                .ticket = ticket,
                .command_buffer = command_buffer,
                .allocator_bundle = allocator_bundle,
                .callbacks = std::move(callbacks),
                .submission = submission});
        }
        return submission;
    }

    auto fast_target = _submitted_ticket.load(std::memory_order_acquire);
    if (_retired_ticket.load(std::memory_order_acquire) < fast_target) {
        _wait_for_completion_ticket(fast_target);
        _retire_submissions_up_to(fast_target, true);
    }

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
    auto label = profile_cb ?
                     (command_buffer->label() == nullptr ?
                          luisa::string{"<unlabeled>"} :
                          luisa::string{command_buffer->label()->utf8String()}) :
                     luisa::string{};
    {
        std::unique_lock lock{_inflight_mutex};
        if (_max_commands != 0u) {
            _inflight_cv.wait(lock, [this]() noexcept {
                return _inflight_commands < _max_commands;
            });
        }
        _inflight_commands++;
        _has_unsubmitted_queue_ops.store(false, std::memory_order_relaxed);
    }
    auto options = NS::TransferPtr(
        MTL4::CommitOptions::alloc()->init());
    options->addFeedbackHandler(MTL4::CommitFeedbackHandlerFunction{
        [this, command_buffer, allocator_bundle, submission,
         has_callbacks, timing, profile_cb, label = std::move(label)](
            MTL4::CommitFeedback *feedback) noexcept {
            // Record receipt before user callbacks or allocator recycling.
            if (timing.session) { timing.session->feedback(timing.ordinal, feedback); }
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
            if (timing.session) { timing.session->callbacks_finished(timing.ordinal); }
            if (profile_cb) {
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
            _recycle_command_allocator(allocator_bundle);
            command_buffer->release();
            if (!error_message.empty()) {
                LUISA_ERROR_WITH_LOCATION(
                    "Metal4 command-buffer execution failed: {}.",
                    error_message);
            }
            // Publish completion last. A synchronous waiter may own the final
            // MetalStream reference and destroy the stream as soon as this
            // becomes true, so the feedback handler must not touch `this`
            // afterwards.
            if (timing.session) { timing.session->completing(timing.ordinal); }
            submission->completed.store(true, std::memory_order_release);
            submission->completed.notify_all();
            {
                std::scoped_lock lock{_inflight_mutex};
                LUISA_ASSERT(_inflight_commands != 0u,
                             "Invalid Metal4 in-flight command count.");
                _inflight_commands--;
                _inflight_cv.notify_all();
            }
        }});
    const MTL4::CommandBuffer *command_buffers[]{command_buffer};
    if (timing.session) { timing.session->will_commit(timing.ordinal, timing.dispatch_count, timing.contains_non_dispatch_work); }
    _queue->commit(command_buffers, 1u, options.get());
    if (timing.session) { timing.session->did_commit(timing.ordinal); }
    return submission;
}

luisa::shared_ptr<MetalTimingSession> MetalStream::timing_session() noexcept {
    if (!timing_enabled()) { return {}; }
    std::scoped_lock lock{_timing_mutex};
    return _timing_session;
}

bool MetalStream::begin_timing(luisa::shared_ptr<MetalTimingSession> session) noexcept {
    // The public extension requires externally serialized sample boundaries.
    if (timing_enabled()) { return false; }
    synchronize();
    std::scoped_lock lock{_timing_mutex};
    if (_timing_session) { return false; }
    _timing_session = std::move(session);
    _timing_enabled.store(true, std::memory_order_release);
    return true;
}

Metal4TimingSample MetalStream::end_timing() noexcept {
    luisa::shared_ptr<MetalTimingSession> session;
    {
        std::scoped_lock lock{_timing_mutex};
        if (!_timing_session) { return {.error = "No Metal4 timing sample is active on this stream"}; }
        _timing_enabled.store(false, std::memory_order_release);
        session = std::move(_timing_session);
    }
    // This synchronization's own empty command buffer is deliberately not
    // attached to the disarmed sample. Earlier buffers retain their session.
    synchronize();
    return session->finish();
}

}// namespace luisa::compute::metal
