// Validate benchmark GPU timestamps, numerical transparency, all public
// encoder factories, failure paths, and restoration of instrumentation.
#include "ut/ut.hpp"
#include "metal_benchmark_timing.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <cmath>
#include <cstring>
#include <iostream>

using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        expect(device != nil) << fatal;
        NSError *error = nil;
        auto library = [device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void increment(device uint *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1u; }"
                                            options:nil
                                              error:&error];
        expect(library != nil) << fatal;
        auto function = [library newFunctionWithName:@"increment"];
        auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        expect(pipeline != nil) << fatal;
        auto queue = [device newCommandQueue];
        constexpr auto elements = 262144u;
        auto storage = [device newBufferWithLength:elements * sizeof(uint32_t) options:MTLStorageModeShared];
        std::memset(storage.contents, 0, storage.length);
        auto encode = [&](id<MTLCommandBuffer> command, uint32_t factory) {
            id<MTLComputeCommandEncoder> encoder = nil;
            if (factory == 0u) {
                encoder = [command computeCommandEncoder];
            } else if (factory == 1u) {
                encoder = [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            } else {
                encoder = [command computeCommandEncoderWithDescriptor:[MTLComputePassDescriptor computePassDescriptor]];
            }
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:storage offset:0u atIndex:0u];
            [encoder dispatchThreads:MTLSizeMake(elements, 1u, 1u) threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder endEncoding];
        };
        auto probe = [queue commandBuffer];
        auto command_class = object_getClass(probe);
        auto selector = @selector(computeCommandEncoder);
        auto original = class_getMethodImplementation(command_class, selector);

        "metal_compute_pass_timing"_test = [&] {
            expect(luisa_metal_timing_version() == 1);
            expect(luisa_metal_timing_begin(16u) == 1) << luisa_metal_timing_error() << fatal;
            auto first = [queue commandBuffer];
            encode(first, 0u);
            encode(first, 1u);
            [first commit];
            auto second = [queue commandBuffer];
            encode(second, 2u);
            [second commit];
            [first waitUntilCompleted];
            [second waitUntilCompleted];
            LuisaMetalTimingResult result{};
            expect(luisa_metal_timing_end(&result) == 1) << luisa_metal_timing_error() << fatal;
            expect(result.compute_passes == 3u);
            expect(result.command_buffers == 2u);
            expect(std::isfinite(result.compute_ns) && result.compute_ns > 0.0);
            expect(result.compute_span_ns >= result.compute_ns * .99);
            expect(result.compute_ns < result.command_buffer_ns * 1.1);
            expect(result.calibration_cpu_ns > 0.0 && result.calibration_gpu_ticks > 0.0);
            auto values = static_cast<const uint32_t *>(storage.contents);
            auto correct = true;
            for (auto i = 0u; i < elements; i++) { correct &= values[i] == 3u; }
            expect(correct);
            expect(class_getMethodImplementation(command_class, selector) == original);
            std::cout << "GPU compute " << result.compute_ns / 1000.0
                      << " us, command buffers " << result.command_buffer_ns / 1000.0 << " us\n";
        };
        "metal_timing_capacity_fails_closed"_test = [&] {
            expect(luisa_metal_timing_begin(1u) == 1) << fatal;
            auto command = [queue commandBuffer];
            encode(command, 0u);
            encode(command, 2u);
            [command commit];
            [command waitUntilCompleted];
            LuisaMetalTimingResult result{};
            expect(luisa_metal_timing_end(&result) == 0);
            expect(class_getMethodImplementation(command_class, selector) == original);
        };
        "metal_timing_empty_and_nested_fail_closed"_test = [&] {
            expect(luisa_metal_timing_begin(0u) == 0);
            expect(luisa_metal_timing_begin(1u) == 1) << fatal;
            expect(luisa_metal_timing_begin(1u) == 0);
            LuisaMetalTimingResult result{};
            expect(luisa_metal_timing_end(&result) == 0);
            expect(class_getMethodImplementation(command_class, selector) == original);
            expect(luisa_metal_timing_begin(1u) == 1) << fatal;
            expect(luisa_metal_timing_end(&result) == 0);
            expect(class_getMethodImplementation(command_class, selector) == original);
        };
        "metal_timing_requires_completed_work"_test = [&] {
            expect(luisa_metal_timing_begin(1u) == 1) << fatal;
            auto command = [queue commandBuffer];
            encode(command, 0u);
            LuisaMetalTimingResult result{};
            expect(luisa_metal_timing_end(&result) == 0);
            [command commit];
            [command waitUntilCompleted];
            expect(class_getMethodImplementation(command_class, selector) == original);
        };
    }
}
