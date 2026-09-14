#include <chrono>
#include <cmath>
#include <cstring>
#include <objc/runtime.h>
#include <luisa/core/logging.h>
#include "metal_timing.h"
#include "metal_stream.h"

namespace luisa::compute::metal {
namespace {

[[nodiscard]] uint64_t host_nanoseconds() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

[[nodiscard]] bool responds(MTL::Device *device, const char *selector) noexcept {
    return class_respondsToSelector(object_getClass(reinterpret_cast<id>(device)), sel_registerName(selector));
}

[[nodiscard]] NS::SharedPtr<MTL4::CounterHeap> create_heap(MTL::Device *device, uint32_t dispatches,
                                                           luisa::string &error) noexcept {
    auto descriptor = NS::TransferPtr(MTL4::CounterHeapDescriptor::alloc()->init());
    descriptor->setType(MTL4::CounterHeapTypeTimestamp);
    descriptor->setCount(static_cast<NS::UInteger>(dispatches) * 2u);
    NS::Error *native_error = nullptr;
    auto heap = NS::TransferPtr(device->newCounterHeap(descriptor.get(), &native_error));
    if (!heap) {
        error = native_error ? native_error->localizedDescription()->utf8String() : "Metal4 timestamp heap creation failed";
    } else if (heap->count() != descriptor->count() || heap->type() != MTL4::CounterHeapTypeTimestamp) {
        error = "Metal4 timestamp heap descriptor mismatch";
        heap.reset();
    } else {
        heap->invalidateCounterRange(NS::Range::Make(0u, heap->count()));
    }
    return heap;
}

}// namespace

MetalTimingSession::MetalTimingSession(uint64_t sample_id, uint32_t max_dispatches,
                                       uint64_t frequency, NS::SharedPtr<MTL4::CounterHeap> heap) noexcept
    : _max_dispatches{max_dispatches}, _heap{std::move(heap)} {
    _sample.sample_id = sample_id;
    _sample.timestamp_frequency_hz = frequency;
    _sample.dispatch_timestamps_enabled = _heap.get() != nullptr;
    _sample.dispatches.reserve(max_dispatches);
}

uint64_t MetalTimingSession::begin_command_buffer() noexcept {
    std::scoped_lock lock{_mutex};
    auto ordinal = _sample.command_buffers.size();
    _sample.command_buffers.emplace_back(Metal4CommandBufferTiming{.ordinal = ordinal});
    return ordinal;
}

uint32_t MetalTimingSession::begin_dispatch(MTL4::ComputeCommandEncoder *encoder,
                                            uint64_t command_buffer_ordinal, uint64_t shader_checksum,
                                            uint3 dispatch_size, uint3 block_size) noexcept {
    std::scoped_lock lock{_mutex};
    if (_sample.dispatches.size() >= _max_dispatches) {
        _sample.overflow = true;
        _sample.error = "Metal4 timing sample exceeded max_dispatches; partial records must not be used as a complete sample";
        return UINT32_MAX;
    }
    auto ordinal = static_cast<uint32_t>(_sample.dispatches.size());
    _sample.dispatches.emplace_back(Metal4DispatchTiming{.ordinal = ordinal,
                                                         .command_buffer_ordinal = command_buffer_ordinal,
                                                         .shader_checksum = shader_checksum,
                                                         .dispatch_size = dispatch_size,
                                                         .block_size = block_size});
    if (_heap) { encoder->writeTimestamp(MTL4::TimestampGranularityPrecise, _heap.get(), ordinal * 2u); }
    return ordinal;
}

void MetalTimingSession::end_dispatch(MTL4::ComputeCommandEncoder *encoder, uint32_t ordinal) noexcept {
    if (_heap && ordinal != UINT32_MAX) {
        encoder->writeTimestamp(MTL4::TimestampGranularityPrecise, _heap.get(), ordinal * 2u + 1u);
    }
}

void MetalTimingSession::unsupported_indirect_dispatch() noexcept {
    std::scoped_lock lock{_mutex};
    _sample.error = "Metal4 per-dispatch timing does not support indirect command ranges";
}

void MetalTimingSession::will_commit(uint64_t ordinal, uint64_t dispatch_count, bool non_dispatch_work) noexcept {
    std::scoped_lock lock{_mutex};
    auto &record = _sample.command_buffers.at(ordinal);
    record.dispatch_count = dispatch_count;
    record.contains_non_dispatch_work = non_dispatch_work;
    record.host_commit_begin_nanoseconds = host_nanoseconds();
    _pending_feedback++;
}

void MetalTimingSession::did_commit(uint64_t ordinal) noexcept {
    auto time = host_nanoseconds();
    std::scoped_lock lock{_mutex};
    _sample.command_buffers.at(ordinal).host_commit_return_nanoseconds = time;
}

void MetalTimingSession::feedback(uint64_t ordinal, MTL4::CommitFeedback *feedback) noexcept {
    auto receipt = host_nanoseconds();
    {
        std::scoped_lock lock{_mutex};
        auto &record = _sample.command_buffers.at(ordinal);
        record.host_feedback_begin_nanoseconds = receipt;
        record.gpu_begin_seconds = feedback->GPUStartTime();
        record.gpu_end_seconds = feedback->GPUEndTime();
        record.valid = feedback->error() == nullptr && std::isfinite(record.gpu_begin_seconds) &&
                       std::isfinite(record.gpu_end_seconds) && record.gpu_begin_seconds > 0.0 &&
                       record.gpu_end_seconds > record.gpu_begin_seconds;
        if (feedback->error()) { _sample.error = feedback->error()->localizedDescription()->utf8String(); }
    }
}

void MetalTimingSession::callbacks_finished(uint64_t ordinal) noexcept {
    auto time = host_nanoseconds();
    std::scoped_lock lock{_mutex};
    _sample.command_buffers.at(ordinal).host_callbacks_end_nanoseconds = time;
}

void MetalTimingSession::completing(uint64_t ordinal) noexcept {
    auto time = host_nanoseconds();
    {
        std::scoped_lock lock{_mutex};
        _sample.command_buffers.at(ordinal).host_completion_publish_nanoseconds = time;
        _pending_feedback--;
    }
    _feedback_cv.notify_all();
}

Metal4TimingSample MetalTimingSession::finish() noexcept {
    return with_autorelease_pool([&] {
        std::unique_lock lock{_mutex};
        // GPU completion and feedback receipt are separate observations.
        // Do not assume completion-handler invocation order across commits.
        _feedback_cv.wait(lock, [&] { return _pending_feedback == 0u; });
        if (_heap && !_sample.dispatches.empty()) {
            auto count = _sample.dispatches.size() * 2u;
            auto data = _heap->resolveCounterRange(NS::Range::Make(0u, count));
            if (data == nullptr || data->length() != count * sizeof(MTL4::TimestampHeapEntry)) {
                _sample.error = "Metal4 timestamp heap resolution failed or returned an unexpected byte count";
            } else {
                auto bytes = static_cast<const std::byte *>(data->bytes());
                for (auto &record : _sample.dispatches) {
                    MTL4::TimestampHeapEntry stamps[2u]{};
                    std::memcpy(stamps, bytes + record.ordinal * sizeof(stamps), sizeof(stamps));
                    record.begin_ticks = stamps[0u].timestamp;
                    record.end_ticks = stamps[1u].timestamp;
                    record.valid = record.begin_ticks != 0u && record.end_ticks > record.begin_ticks && _sample.timestamp_frequency_hz != 0u;
                    if (record.valid) {
                        record.elapsed_nanoseconds = static_cast<double>(record.end_ticks - record.begin_ticks) * 1.0e9 /
                                                     static_cast<double>(_sample.timestamp_frequency_hz);
                    } else {
                        _sample.error = "Metal4 timestamp sample is zero, unordered, or missing its frequency";
                    }
                }
            }
        }
        for (auto &record : _sample.command_buffers) {
            // Empty synchronize submissions may have no meaningful GPU span;
            // retain that absence without invalidating dispatch evidence.
            if (!record.valid && (record.dispatch_count != 0u || record.contains_non_dispatch_work)) {
                _sample.error = "Metal4 command-buffer feedback interval is unavailable";
            }
        }
        return std::move(_sample);
    });
}

Metal4TimingCapabilities MetalTimingExt::capabilities() const noexcept {
    return with_autorelease_pool([&] {
        Metal4TimingCapabilities result;
        result.legacy_dispatch_sampling = _device->supportsCounterSampling(MTL::CounterSamplingPointAtDispatchBoundary);
        result.legacy_stage_sampling = _device->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary);
        if (!responds(_device, "newCounterHeapWithDescriptor:error:") ||
            !responds(_device, "queryTimestampFrequency") || !responds(_device, "sizeOfCounterHeapEntry:")) {
            result.error = "This Metal device does not expose MTL4 timestamp heap APIs";
            return result;
        }
        result.timestamp_frequency_hz = _device->queryTimestampFrequency();
        if (result.timestamp_frequency_hz == 0u || _device->sizeOfCounterHeapEntry(MTL4::CounterHeapTypeTimestamp) != sizeof(MTL4::TimestampHeapEntry)) {
            result.error = "Metal4 timestamp frequency or resolved entry format is unsupported";
            return result;
        }
        result.timestamp_heap = create_heap(_device, 1u, result.error).get() != nullptr;
        return result;
    });
}

bool MetalTimingExt::begin_sample(uint64_t stream_handle, uint64_t sample_id,
                                  uint32_t max_dispatches, bool capture_dispatch_timestamps) const noexcept {
    return with_autorelease_pool([&] {
        if (stream_handle == 0u || max_dispatches == 0u || max_dispatches > 1048576u) { return false; }
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        if (stream->device() != _device || stream->timing_enabled()) { return false; }
        NS::SharedPtr<MTL4::CounterHeap> heap;
        auto frequency = uint64_t{0u};
        if (capture_dispatch_timestamps) {
            auto support = capabilities();
            if (!support.timestamp_heap) {
                LUISA_WARNING("Metal4 dispatch timestamps unavailable: {}", support.error);
                return false;
            }
            frequency = support.timestamp_frequency_hz;
            luisa::string error;
            heap = create_heap(_device, max_dispatches, error);
            if (!heap) {
                LUISA_WARNING("Metal4 timing sample allocation failed: {}", error);
                return false;
            }
        }
        return stream->begin_timing(luisa::make_shared<MetalTimingSession>(sample_id, max_dispatches, frequency, std::move(heap)));
    });
}

Metal4TimingSample MetalTimingExt::end_sample(uint64_t stream_handle) const noexcept {
    if (stream_handle == 0u) { return {.error = "Invalid Metal4 timing stream"}; }
    auto stream = reinterpret_cast<MetalStream *>(stream_handle);
    if (stream->device() != _device) { return {.error = "Metal4 timing stream belongs to another device"}; }
    return stream->end_timing();
}

}// namespace luisa::compute::metal
