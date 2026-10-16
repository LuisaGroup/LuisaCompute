// Real GPU compute-pass counters for framework-neutral benchmark diagnostics.
// Interpose only the PUBLIC MTLCommandBuffer encoder factories and commit,
// on the concrete command-buffer class discovered from the selected device.
// This lets TVMx, Luisa and eager PyTorch keep their own dispatch paths without
// private framework ABIs, source replay, or changes to the pinned TVMx build.
// Sampling preserves dispatch type and existing counter attachments. It does
// not split encoders or insert barriers. Restore every IMP before returning.
#include "metal_benchmark_timing.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct Hook {
    SEL selector;
    IMP original;
};

struct Capture {
    std::mutex mutex;
    bool active{false};
    bool compute_counters{true};
    bool failed{false};
    std::string error;
    id<MTLDevice> device;
    id<MTLCounterSampleBuffer> samples;
    Class command_class{Nil};
    std::vector<Hook> hooks;
    std::vector<id<MTLCommandBuffer>> buffers;
    uint32_t passes{0u};
    MTLTimestamp cpu_start{0u};
    MTLTimestamp gpu_start{0u};
};

Capture capture;
thread_local bool inside_factory = false;

void fail(const char *message) {
    if (!capture.failed) { capture.error = message; }
    capture.failed = true;
}

IMP original(SEL selector) {
    auto found = std::find_if(capture.hooks.begin(), capture.hooks.end(),
                              [selector](auto hook) { return hook.selector == selector; });
    return found == capture.hooks.end() ? nullptr : found->original;
}

void remember(id<MTLCommandBuffer> buffer) {
    if (buffer.device.registryID != capture.device.registryID) {
        fail("timed work used a different Metal device");
    }
    if (std::find(capture.buffers.begin(), capture.buffers.end(), buffer) == capture.buffers.end()) {
        capture.buffers.emplace_back(buffer);
    }
}

MTLComputePassDescriptor *instrument(id<MTLCommandBuffer> buffer, MTLComputePassDescriptor *requested) {
    std::lock_guard lock{capture.mutex};
    if (!capture.active) { return requested; }
    remember(buffer);
    if (capture.failed) { return requested; }
    if (2u * capture.passes + 2u > capture.samples.sampleCount) {
        fail("Metal compute-pass counter capacity exceeded; reduce device repetitions");
        return requested;
    }
    // MTLComputePassDescriptor has four sample-buffer attachment slots.
    MTLComputePassDescriptor *descriptor = [requested copy];
    for (auto slot = 0u; slot < 4u; slot++) {
        auto attachment = descriptor.sampleBufferAttachments[slot];
        if (attachment.sampleBuffer == nil) {
            attachment.sampleBuffer = capture.samples;
            attachment.startOfEncoderSampleIndex = 2u * capture.passes;
            attachment.endOfEncoderSampleIndex = 2u * capture.passes + 1u;
            capture.passes++;
            return descriptor;
        }
    }
    fail("all Metal compute-pass counter attachment slots are occupied");
    return requested;
}

id<MTLComputeCommandEncoder> encoder_with_descriptor(id<MTLCommandBuffer> buffer, SEL selector, MTLComputePassDescriptor *descriptor) {
    auto invoke = reinterpret_cast<id<MTLComputeCommandEncoder> (*)(id, SEL, MTLComputePassDescriptor *)>(original(selector));
    if (inside_factory) { return invoke(buffer, selector, descriptor); }
    inside_factory = true;
    auto encoder = invoke(buffer, selector, instrument(buffer, descriptor));
    inside_factory = false;
    return encoder;
}

id<MTLComputeCommandEncoder> encoder_with_type(id<MTLCommandBuffer> buffer, SEL selector, MTLDispatchType type) {
    auto invoke = reinterpret_cast<id<MTLComputeCommandEncoder> (*)(id, SEL, MTLDispatchType)>(original(selector));
    if (inside_factory) { return invoke(buffer, selector, type); }
    auto descriptor = [MTLComputePassDescriptor computePassDescriptor];
    descriptor.dispatchType = type;
    return encoder_with_descriptor(buffer, @selector(computeCommandEncoderWithDescriptor:), descriptor);
}

id<MTLComputeCommandEncoder> encoder(id<MTLCommandBuffer> buffer, SEL selector) {
    auto invoke = reinterpret_cast<id<MTLComputeCommandEncoder> (*)(id, SEL)>(original(selector));
    if (inside_factory) { return invoke(buffer, selector); }
    auto descriptor = [MTLComputePassDescriptor computePassDescriptor];
    descriptor.dispatchType = MTLDispatchTypeSerial;
    return encoder_with_descriptor(buffer, @selector(computeCommandEncoderWithDescriptor:), descriptor);
}

void commit(id<MTLCommandBuffer> buffer, SEL selector) {
    {
        std::lock_guard lock{capture.mutex};
        if (capture.active) { remember(buffer); }
    }
    auto invoke = reinterpret_cast<void (*)(id, SEL)>(original(selector));
    invoke(buffer, selector);
}

bool hook(SEL selector, IMP replacement) {
    auto method = class_getInstanceMethod(capture.command_class, selector);
    if (method == nullptr) { return false; }
    auto previous = method_getImplementation(method);
    // Never overwrite a superclass's implementation for unrelated devices.
    class_addMethod(capture.command_class, selector, previous, method_getTypeEncoding(method));
    capture.hooks.emplace_back(Hook{selector, previous});
    class_replaceMethod(capture.command_class, selector, replacement, method_getTypeEncoding(method));
    return true;
}

void restore() {
    for (auto hook : capture.hooks) {
        auto method = class_getInstanceMethod(capture.command_class, hook.selector);
        class_replaceMethod(capture.command_class, hook.selector, hook.original, method_getTypeEncoding(method));
    }
    capture.active = false;
}

}// namespace

extern "C" int luisa_metal_timing_version() { return 2; }

extern "C" const char *luisa_metal_timing_error() { return capture.error.c_str(); }

static int begin(uint32_t max_compute_passes, bool compute_counters) {
    std::lock_guard lock{capture.mutex};
    @autoreleasepool {
        if (capture.active) {
            fail("nested Metal timing capture is not supported");
            return 0;
        }
        capture.failed = false;
        capture.error.clear();
        capture.passes = 0u;
        capture.compute_counters = compute_counters;
        capture.buffers.clear();
        capture.hooks.clear();
        if (compute_counters && (max_compute_passes == 0u || max_compute_passes > 65536u)) {
            fail("Metal timing capacity must be in [1,65536] compute passes");
            return 0;
        }
        capture.device = MTLCreateSystemDefaultDevice();
        if (capture.device == nil || (compute_counters && ![capture.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])) {
            fail("Metal compute-pass timestamp sampling is unavailable");
            return 0;
        }
        if (compute_counters) {
            id<MTLCounterSet> timestamps = nil;
            for (id<MTLCounterSet> set in capture.device.counterSets) {
                if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                    timestamps = set;
                    break;
                }
            }
            if (timestamps == nil) {
                fail("Metal timestamp counter set is unavailable");
                return 0;
            }
            auto descriptor = [MTLCounterSampleBufferDescriptor new];
            descriptor.counterSet = timestamps;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.sampleCount = 2u * max_compute_passes;
            descriptor.label = @"Luisa benchmark GPU timestamps";
            NSError *error = nil;
            capture.samples = [capture.device newCounterSampleBufferWithDescriptor:descriptor error:&error];
            if (capture.samples == nil) {
                fail(error.localizedDescription.UTF8String ?: "Metal timestamp allocation failed");
                return 0;
            }
        }
        auto queue = [capture.device newCommandQueue];
        auto probe = [queue commandBuffer];
        capture.command_class = object_getClass(probe);
        if ((compute_counters && (!hook(@selector(computeCommandEncoder), reinterpret_cast<IMP>(encoder)) ||
                                  !hook(@selector(computeCommandEncoderWithDispatchType:), reinterpret_cast<IMP>(encoder_with_type)) ||
                                  !hook(@selector(computeCommandEncoderWithDescriptor:), reinterpret_cast<IMP>(encoder_with_descriptor)))) ||
            !hook(@selector(commit), reinterpret_cast<IMP>(commit))) {
            restore();
            fail("Metal command-buffer API cannot be instrumented");
            return 0;
        }
        if (compute_counters) { [capture.device sampleTimestamps:&capture.cpu_start gpuTimestamp:&capture.gpu_start]; }
        capture.active = true;
        return 1;
    }
}

extern "C" int luisa_metal_timing_begin(uint32_t max_compute_passes) { return begin(max_compute_passes, true); }
extern "C" int luisa_metal_timing_begin_control() { return begin(0u, false); }

extern "C" int luisa_metal_timing_end(LuisaMetalTimingResult *result) {
    std::lock_guard lock{capture.mutex};
    @autoreleasepool {
        if (!capture.active) {
            fail("Metal timing end has no active capture");
            return 0;
        }
        restore();
        MTLTimestamp cpu_end = 0u, gpu_end = 0u;
        if (capture.compute_counters) { [capture.device sampleTimestamps:&cpu_end gpuTimestamp:&gpu_end]; }
        if (result == nullptr || (capture.compute_counters && capture.passes == 0u) || capture.buffers.empty()) {
            fail("Metal timing captured no required work or has no result storage");
        }
        if (capture.compute_counters && (cpu_end <= capture.cpu_start || gpu_end <= capture.gpu_start)) {
            fail("invalid Metal CPU/GPU timestamp calibration");
        }
        auto command_ns = 0.0;
        for (id<MTLCommandBuffer> buffer : capture.buffers) {
            if (buffer.status != MTLCommandBufferStatusCompleted) {
                fail("timed Metal command buffer did not complete successfully; synchronize before ending capture");
                break;
            }
            auto begin = buffer.GPUStartTime;
            auto end = buffer.GPUEndTime;
            if (!std::isfinite(begin) || !std::isfinite(end) || begin <= 0.0 || end < begin) {
                fail("invalid completed Metal command-buffer GPU timestamps");
                break;
            }
            command_ns += (end - begin) * 1e9;
        }
        if (capture.failed) {
            capture.buffers.clear();
            return 0;
        }
        if (!capture.compute_counters) {
            *result = LuisaMetalTimingResult{0.0, 0.0, command_ns, 0.0, 0.0, 0u, capture.buffers.size()};
            capture.buffers.clear();
            return 1;
        }
        auto resolved = [capture.samples resolveCounterRange:NSMakeRange(0u, 2u * capture.passes)];
        if (resolved == nil || resolved.length != 2u * capture.passes * sizeof(MTLCounterResultTimestamp)) {
            fail("Metal GPU counter resolve failed");
            capture.buffers.clear();
            return 0;
        }
        auto timestamps = static_cast<const MTLCounterResultTimestamp *>(resolved.bytes);
        auto ticks = 0.0;
        auto first = std::numeric_limits<uint64_t>::max();
        auto last = uint64_t{0u};
        for (auto i = 0u; i < capture.passes; i++) {
            auto begin = timestamps[2u * i].timestamp;
            auto end = timestamps[2u * i + 1u].timestamp;
            if (begin == MTLCounterErrorValue || end == MTLCounterErrorValue || begin == 0u || end <= begin) {
                fail("invalid Metal compute-pass GPU counter sample");
                break;
            }
            ticks += static_cast<double>(end - begin);
            first = std::min(first, begin);
            last = std::max(last, end);
        }
        // Apple documents sampleTimestamps CPU values in nanoseconds; GPU
        // counter ticks need this two-point calibration, not a guessed unit.
        auto cpu_ns = static_cast<double>(cpu_end - capture.cpu_start);
        auto gpu_ticks = static_cast<double>(gpu_end - capture.gpu_start);
        auto ns_per_tick = cpu_ns / gpu_ticks;
        *result = LuisaMetalTimingResult{
            ticks * ns_per_tick, static_cast<double>(last - first) * ns_per_tick,
            command_ns, cpu_ns, gpu_ticks, capture.passes, capture.buffers.size()};
        capture.buffers.clear();
        return capture.failed ? 0 : 1;
    }
}
