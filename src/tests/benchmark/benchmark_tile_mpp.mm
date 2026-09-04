// Direct Metal Performance Primitives tensor-op probe. This establishes the
// target-intrinsic ceiling before the Tile lowering or planner selects it.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000

namespace {

using Clock = std::chrono::steady_clock;

struct Configuration {
    int m;
    int n;
    int k;
    int samples;
    int sample_ms;
    int warmup_ms;
    int tile_m{64};
    int tile_n{64};
    int simdgroups{4};
    bool cooperative{true};
    bool relaxed_precision{false};
    bool static_reduction{false};
    bool inline_tensors{true};
    int group_simdgroups{0};
    int cohort_rows{1};
};

struct API_AVAILABLE(macos(26.0)) Precision {
    std::string_view name;
    const char *input_msl;
    const char *accumulator_msl;
    MTLTensorDataType input_mtl;
    MTLTensorDataType output_mtl;
    size_t input_bytes;
    size_t output_bytes;
    bool bfloat_input;
};

struct Measurement {
    std::string device;
    double setup_ms{};
    double cold_ms{};
    double warmup_ms{};
    double download_ms{};
    uint64_t repetitions{};
    uint64_t thread_execution_width{};
    uint64_t static_threadgroup_bytes{};
    uint64_t max_threads_per_group{};
    std::vector<double> throughput;
    std::vector<double> latency;
    std::vector<double> gpu_throughput;
    std::vector<double> gpu_latency;
};

struct BatchTiming {
    double host_ms;
    double gpu_ms;
};

constexpr auto metal_source = R"metal(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

template<typename A, typename B, typename C>
void multiply_tile(thread A &tile_a, thread B &tile_b, thread C &tile_c) {
#if STATIC_REDUCTION
    constexpr auto descriptor = matmul2d_descriptor(TILE_M, TILE_N, REDUCTION_K,
                                                    false, false, RELAXED_PRECISION);
#else
    constexpr auto descriptor = matmul2d_descriptor(TILE_M, TILE_N, dynamic_length_v<int>,
                                                    false, false, RELAXED_PRECISION);
#endif
    matmul2d<descriptor, execution_simdgroups<SIMDGROUPS>> operation;
#if COOPERATIVE_OUTPUT
    auto result = operation.get_destination_cooperative_tensor<A, B,
                                                               ACCUMULATOR_ELEMENT>();
    operation.run(tile_a, tile_b, result);
    result.store(tile_c);
#else
    operation.run(tile_a, tile_b, tile_c);
#endif
}

kernel void mpp_gemm(uint2 group [[threadgroup_position_in_grid]],
                     uint subgroup [[simdgroup_index_in_threadgroup]],
#if INLINE_TENSORS
                     device INPUT_ELEMENT *a_data [[buffer(0)]],
                     device INPUT_ELEMENT *b_data [[buffer(1)]],
                     device ACCUMULATOR_ELEMENT *c_data [[buffer(2)]]) {
    tensor<device INPUT_ELEMENT, dextents<int, 2>, tensor_inline> a(
        a_data, dextents<int, 2>(REDUCTION_K, ROWS_M), array<int, 2>{1, REDUCTION_K});
    tensor<device INPUT_ELEMENT, dextents<int, 2>, tensor_inline> b(
        b_data, dextents<int, 2>(COLUMNS_N, REDUCTION_K), array<int, 2>{1, COLUMNS_N});
    tensor<device ACCUMULATOR_ELEMENT, dextents<int, 2>, tensor_inline> c(
        c_data, dextents<int, 2>(COLUMNS_N, ROWS_M), array<int, 2>{1, COLUMNS_N});
#else
                     tensor<device INPUT_ELEMENT, dextents<int, 2>> a,
                     tensor<device INPUT_ELEMENT, dextents<int, 2>> b,
                     tensor<device ACCUMULATOR_ELEMENT, dextents<int, 2>> c) {
#endif
    // A multi-SIMD-group operation uses the whole threadgroup. Alternatively,
    // independent single-SIMD-group operations form a spatial cohort. Memory
    // views are composed with that execution map, not the other way around.
    const int local = SIMDGROUPS == 1 ? static_cast<int>(subgroup) : 0;
    const int origin_x = (static_cast<int>(group.x) * COHORT_COLUMNS + local % COHORT_COLUMNS) * TILE_N;
    const int origin_y = (static_cast<int>(group.y) * COHORT_ROWS + local / COHORT_COLUMNS) * TILE_M;
    if (origin_x >= COLUMNS_N || origin_y >= ROWS_M) { return; }
    // A static slice promises an in-bounds tile. Only interior groups may
    // make that promise; dynamic slices retain the original tensor bounds.
    if (origin_x <= COLUMNS_N - TILE_N && origin_y <= ROWS_M - TILE_M) {
#if STATIC_REDUCTION
        auto tile_a = a.slice<REDUCTION_K, TILE_M>(0, origin_y);
        auto tile_b = b.slice<TILE_N, REDUCTION_K>(origin_x, 0);
#else
        auto tile_a = a.slice<dynamic_extent, TILE_M>(0, origin_y);
        auto tile_b = b.slice<TILE_N, dynamic_extent>(origin_x, 0);
#endif
        auto tile_c = c.slice<TILE_N, TILE_M>(origin_x, origin_y);
        multiply_tile(tile_a, tile_b, tile_c);
    } else {
        auto tile_a = a.slice(0, origin_y);
        auto tile_b = b.slice(origin_x, 0);
        auto tile_c = c.slice(origin_x, origin_y);
        multiply_tile(tile_a, tile_b, tile_c);
    }
}
)metal";

[[nodiscard]] int positive_integer(const char *text) {
    auto input = std::string_view{text};
    auto value = 0;
    auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size() || value <= 0) {
        throw std::invalid_argument{"expected a positive int32"};
    }
    return value;
}

[[nodiscard]] Precision precision(std::string_view name) API_AVAILABLE(macos(26.0)) {
    if (name == "fp32") {
        return {name, "float", "float", MTLTensorDataTypeFloat32, MTLTensorDataTypeFloat32, 4u, 4u, false};
    }
    if (name == "fp16") {
        return {name, "half", "half", MTLTensorDataTypeFloat16, MTLTensorDataTypeFloat16, 2u, 2u, false};
    }
    if (name == "fp16-fp32") {
        return {name, "half", "float", MTLTensorDataTypeFloat16, MTLTensorDataTypeFloat32, 2u, 4u, false};
    }
    if (name == "bf16") {
        return {name, "bfloat", "bfloat", MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, 2u, 2u, true};
    }
    if (name == "bf16-fp32") {
        return {name, "bfloat", "float", MTLTensorDataTypeBFloat16, MTLTensorDataTypeFloat32, 2u, 4u, true};
    }
    throw std::invalid_argument{"precision must be fp32, fp16, fp16-fp32, bf16, or bf16-fp32"};
}

[[nodiscard]] bool boolean(std::string_view text, std::string_view name) {
    if (text == "0") { return false; }
    if (text == "1") { return true; }
    throw std::invalid_argument{std::string{name} + " must be 0 or 1"};
}

[[nodiscard]] size_t elements(int rows, int columns) {
    auto count = static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns);
    if (count > std::numeric_limits<size_t>::max() / sizeof(float) ||
        count > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float)) {
        throw std::invalid_argument{"matrix size overflow"};
    }
    return static_cast<size_t>(count);
}

[[nodiscard]] double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>{Clock::now() - start}.count();
}

[[nodiscard]] std::vector<float> input_values(size_t count, uint64_t seed) {
    std::vector<float> result(count);
    for (auto i = size_t{0u}; i < count; i++) {
        result[i] = static_cast<float>(static_cast<int64_t>((i * seed + 17u) % 127u) - 63) / 64.0f;
    }
    return result;
}

[[nodiscard]] std::vector<std::byte> encode_input(const std::vector<float> &values, const Precision &mode) API_AVAILABLE(macos(26.0)) {
    std::vector<std::byte> result(values.size() * mode.input_bytes);
    if (mode.input_bytes == sizeof(float)) {
        std::memcpy(result.data(), values.data(), result.size());
    } else if (mode.bfloat_input) {
        auto output = reinterpret_cast<uint16_t *>(result.data());
        for (auto i = size_t{0u}; i < values.size(); i++) {
            auto bits = std::bit_cast<uint32_t>(values[i]);
            output[i] = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
        }
    } else {
        auto output = reinterpret_cast<_Float16 *>(result.data());
        for (auto i = size_t{0u}; i < values.size(); i++) { output[i] = static_cast<_Float16>(values[i]); }
    }
    return result;
}

void complete(id<MTLCommandBuffer> command) {
    if (command == nil) { throw std::runtime_error{"cannot create Metal command buffer"}; }
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        auto reason = command.error.localizedDescription;
        throw std::runtime_error{reason == nil ? "Metal command failed" : reason.UTF8String};
    }
}

[[nodiscard]] id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes, MTLResourceOptions options) {
    auto result = [device newBufferWithLength:bytes options:options];
    if (result == nil) { throw std::runtime_error{"Metal buffer allocation failed"}; }
    return result;
}

[[nodiscard]] MTLTensorExtents *extents(NSInteger first, NSInteger second) API_AVAILABLE(macos(26.0)) {
    NSInteger values[]{first, second};
    return [[MTLTensorExtents alloc] initWithRank:2 values:values];
}

[[nodiscard]] Measurement measure(std::string_view name, Configuration cfg, const char *path) {
    if (@available(macOS 26.0, *)) {
        if (cfg.tile_m % 8 != 0 || cfg.tile_n % 8 != 0 || (cfg.tile_m % 16 != 0 && cfg.tile_n % 16 != 0)) {
            throw std::invalid_argument{"MPP M/N tiles must be multiples of 8, with at least one a multiple of 16"};
        }
        if (cfg.static_reduction && cfg.k % 16 != 0) {
            throw std::invalid_argument{"MPP static K must be a multiple of 16; use dynamic K for a tail"};
        }
        auto group_simdgroups = cfg.group_simdgroups == 0 ? cfg.simdgroups : cfg.group_simdgroups;
        auto cohorts = cfg.simdgroups == 1 ? group_simdgroups : 1;
        if ((cfg.simdgroups != 1 && group_simdgroups != cfg.simdgroups) || cohorts % cfg.cohort_rows != 0) {
            throw std::invalid_argument{"MPP scope must be one SIMD group or the whole threadgroup; cohort rows must divide independent groups"};
        }
        auto cohort_columns = cohorts / cfg.cohort_rows;
        auto mode = precision(name);
        auto device = MTLCreateSystemDefaultDevice();
        if (device == nil || ![device supportsFamily:MTLGPUFamilyApple7]) {
            throw std::runtime_error{"MPP tensor operations require Apple GPU family 7 or newer"};
        }
        auto start = Clock::now();
        auto prefix = [NSString stringWithFormat:
                                    @"#define INPUT_ELEMENT %s\n"
                                     "#define ACCUMULATOR_ELEMENT %s\n"
                                     "#define TILE_M %d\n"
                                     "#define TILE_N %d\n"
                                     "#define SIMDGROUPS %d\n"
                                     "#define COOPERATIVE_OUTPUT %d\n"
                                     "#define RELAXED_PRECISION %d\n"
                                     "#define STATIC_REDUCTION %d\n"
                                     "#define REDUCTION_K %d\n"
                                     "#define ROWS_M %d\n"
                                     "#define COLUMNS_N %d\n"
                                     "#define INLINE_TENSORS %d\n"
                                     "#define COHORT_ROWS %d\n"
                                     "#define COHORT_COLUMNS %d\n",
                                    mode.input_msl, mode.accumulator_msl,
                                    cfg.tile_m, cfg.tile_n, cfg.simdgroups,
                                    cfg.cooperative, cfg.relaxed_precision,
                                    cfg.static_reduction, cfg.k, cfg.m, cfg.n, cfg.inline_tensors,
                                    cfg.cohort_rows, cohort_columns];
        auto source = [prefix stringByAppendingString:[NSString stringWithUTF8String:metal_source]];
        auto compile_options = [MTLCompileOptions new];
        compile_options.fastMathEnabled = cfg.relaxed_precision;
        compile_options.languageVersion = MTLLanguageVersion4_0;
        NSError *error = nil;
        auto library = [device newLibraryWithSource:source options:compile_options error:&error];
        if (library == nil) {
            throw std::runtime_error{error == nil ? "MPP shader compilation failed" : error.localizedDescription.UTF8String};
        }
        auto function = [library newFunctionWithName:@"mpp_gemm"];
        auto pipeline_descriptor = [MTLComputePipelineDescriptor new];
        pipeline_descriptor.computeFunction = function;
        pipeline_descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
        auto pipeline = [device newComputePipelineStateWithDescriptor:pipeline_descriptor
                                                              options:MTLPipelineOptionNone
                                                           reflection:nil
                                                                error:&error];
        if (pipeline == nil) {
            throw std::runtime_error{error == nil ? "MPP pipeline creation failed" : error.localizedDescription.UTF8String};
        }
        if (static_cast<NSUInteger>(group_simdgroups) * pipeline.threadExecutionWidth > pipeline.maxTotalThreadsPerThreadgroup) {
            throw std::runtime_error{"requested SIMD groups exceed the pipeline threadgroup limit"};
        }

        auto input_a = encode_input(input_values(elements(cfg.m, cfg.k), 5u), mode);
        auto input_b = encode_input(input_values(elements(cfg.k, cfg.n), 11u), mode);
        auto a_buffer = make_buffer(device, input_a.size(), MTLResourceStorageModePrivate);
        auto b_buffer = make_buffer(device, input_b.size(), MTLResourceStorageModePrivate);
        auto c_bytes = elements(cfg.m, cfg.n) * mode.output_bytes;
        auto c_buffer = make_buffer(device, c_bytes, MTLResourceStorageModePrivate);
        auto transfer_queue = [device newCommandQueue];
        if (transfer_queue == nil) { throw std::runtime_error{"Metal transfer queue unavailable"}; }
        auto upload = [&](id<MTLBuffer> destination, const std::vector<std::byte> &bytes) {
            auto staging = make_buffer(device, bytes.size(), MTLResourceStorageModeShared);
            std::memcpy(staging.contents, bytes.data(), bytes.size());
            auto command = [transfer_queue commandBuffer];
            auto encoder = [command blitCommandEncoder];
            if (encoder == nil) { throw std::runtime_error{"Metal upload encoder unavailable"}; }
            [encoder copyFromBuffer:staging sourceOffset:0 toBuffer:destination destinationOffset:0 size:bytes.size()];
            [encoder endEncoding];
            complete(command);
        };
        upload(a_buffer, input_a);
        upload(b_buffer, input_b);
        upload(c_buffer, std::vector<std::byte>(c_bytes, std::byte{0xff}));// NaNs test beta=0.

        Measurement result{};
        result.device = device.name.UTF8String;
        result.thread_execution_width = pipeline.threadExecutionWidth;
        result.static_threadgroup_bytes = pipeline.staticThreadgroupMemoryLength;
        result.max_threads_per_group = pipeline.maxTotalThreadsPerThreadgroup;
        auto group_m = static_cast<uint64_t>(cfg.tile_m) * cfg.cohort_rows;
        auto group_n = static_cast<uint64_t>(cfg.tile_n) * cohort_columns;
        auto groups = MTLSizeMake(cfg.n / group_n + (cfg.n % group_n != 0u),
                                  cfg.m / group_m + (cfg.m % group_m != 0u), 1u);
        auto threads = MTLSizeMake(static_cast<NSUInteger>(group_simdgroups) * pipeline.threadExecutionWidth, 1u, 1u);
        std::function<BatchTiming(uint64_t)> batch;
        if (cfg.inline_tensors) {
            // Inline tensors use the same buffer ABI and tracked command queue
            // as the MPS baseline. No MTLTensor resource or Metal 4 encoder is required.
            batch = [=](uint64_t repetitions) {
                @autoreleasepool {
                    auto begin = Clock::now();
                    auto command = [transfer_queue commandBuffer];
                    auto encoder = [command computeCommandEncoder];
                    if (encoder == nil) { throw std::runtime_error{"Metal compute encoder unavailable"}; }
                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:a_buffer offset:0 atIndex:0];
                    [encoder setBuffer:b_buffer offset:0 atIndex:1];
                    [encoder setBuffer:c_buffer offset:0 atIndex:2];
                    for (auto i = uint64_t{0u}; i < repetitions; i++) {
                        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
                    }
                    [encoder endEncoding];
                    complete(command);
                    return BatchTiming{milliseconds(begin), 1000.0 * (command.GPUEndTime - command.GPUStartTime)};
                }
            };
        } else {

            auto make_tensor = [&](id<MTLBuffer> storage, int height, int width, MTLTensorDataType data_type) {
                auto descriptor = [MTLTensorDescriptor new];
                descriptor.dataType = data_type;
                descriptor.dimensions = extents(width, height);
                descriptor.strides = extents(1, width);
                descriptor.usage = MTLTensorUsageCompute;
                descriptor.storageMode = storage.storageMode;
                auto tensor = [storage newTensorWithDescriptor:descriptor offset:0 error:&error];
                if (tensor == nil) {
                    throw std::runtime_error{error == nil ? "buffer-backed tensor creation failed" : error.localizedDescription.UTF8String};
                }
                return tensor;
            };
            auto tensor_a = make_tensor(a_buffer, cfg.m, cfg.k, mode.input_mtl);
            auto tensor_b = make_tensor(b_buffer, cfg.k, cfg.n, mode.input_mtl);
            auto tensor_c = make_tensor(c_buffer, cfg.m, cfg.n, mode.output_mtl);

            auto table_descriptor = [MTL4ArgumentTableDescriptor new];
            table_descriptor.maxBufferBindCount = 3;
            auto table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
            if (table == nil) {
                throw std::runtime_error{error == nil ? "Metal argument table creation failed" : error.localizedDescription.UTF8String};
            }
            [table setResource:tensor_a.gpuResourceID atBufferIndex:0];
            [table setResource:tensor_b.gpuResourceID atBufferIndex:1];
            [table setResource:tensor_c.gpuResourceID atBufferIndex:2];

            auto residency_descriptor = [MTLResidencySetDescriptor new];
            residency_descriptor.initialCapacity = 3;
            auto residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
            if (residency == nil) {
                throw std::runtime_error{error == nil ? "Metal residency set creation failed" : error.localizedDescription.UTF8String};
            }
            [residency addAllocation:tensor_a];
            [residency addAllocation:tensor_b];
            [residency addAllocation:tensor_c];
            [residency commit];

            auto queue = [device newMTL4CommandQueue];
            auto allocator = [device newCommandAllocator];
            if (queue == nil || allocator == nil) {
                throw std::runtime_error{"Metal 4 command infrastructure unavailable"};
            }
            batch = [=](uint64_t repetitions) {
                @autoreleasepool {
                    auto begin = Clock::now();
                    id<MTL4CommandBuffer> command = [device newCommandBuffer];
                    [command beginCommandBufferWithAllocator:allocator];
                    [command useResidencySet:residency];
                    auto encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:pipeline];
                    [encoder setArgumentTable:table];
                    for (auto i = uint64_t{0u}; i < repetitions; i++) {
                        if (i != 0u) {
                            // Metal 4 does not track the write-after-write hazard on C.
                            [encoder barrierAfterEncoderStages:MTLStageDispatch
                                           beforeEncoderStages:MTLStageDispatch
                                             visibilityOptions:MTL4VisibilityOptionDevice];
                        }
                        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
                    }
                    [encoder endEncoding];
                    [command endCommandBuffer];
                    auto done = dispatch_semaphore_create(0);
                    __block id<MTL4CommitFeedback> feedback = nil;
                    auto options = [MTL4CommitOptions new];
                    [options addFeedbackHandler:^(id<MTL4CommitFeedback> value) {
                        feedback = value;
                        dispatch_semaphore_signal(done);
                    }];
                    [queue commit:&command count:1 options:options];
                    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) != 0) {
                        throw std::runtime_error{"Metal 4 command timed out"};
                    }
                    if (feedback.error != nil) { throw std::runtime_error{feedback.error.localizedDescription.UTF8String}; }
                    [allocator reset];
                    return BatchTiming{milliseconds(begin), 1000.0 * (feedback.GPUEndTime - feedback.GPUStartTime)};
                }
            };
        }

        result.setup_ms = milliseconds(start);
        result.cold_ms = batch(1u).host_ms;
        start = Clock::now();
        while (milliseconds(start) < cfg.warmup_ms) { static_cast<void>(batch(8u)); }
        result.warmup_ms = milliseconds(start);
        result.repetitions = 1u;
        for (auto i = 0; i < 8; i++) {
            auto elapsed = batch(result.repetitions).host_ms;
            if (elapsed >= cfg.sample_ms * 0.8 || result.repetitions == 100000u) { break; }
            auto estimate = std::clamp(result.repetitions * static_cast<double>(cfg.sample_ms) / std::max(elapsed, 1e-6), 1.0, 100000.0);
            result.repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(estimate), result.repetitions + 1u, 100000u);
        }
        for (auto i = 0; i < cfg.samples; i++) {
            auto timing = batch(result.repetitions);
            result.throughput.push_back(1000.0 * timing.host_ms / result.repetitions);
            result.gpu_throughput.push_back(1000.0 * timing.gpu_ms / result.repetitions);
        }
        for (auto i = 0; i < cfg.samples; i++) {
            auto timing = batch(1u);
            result.latency.push_back(1000.0 * timing.host_ms);
            result.gpu_latency.push_back(1000.0 * timing.gpu_ms);
        }

        start = Clock::now();
        auto staging = make_buffer(device, c_bytes, MTLResourceStorageModeShared);
        auto command = [transfer_queue commandBuffer];
        auto encoder = [command blitCommandEncoder];
        if (encoder == nil) { throw std::runtime_error{"Metal download encoder unavailable"}; }
        [encoder copyFromBuffer:c_buffer sourceOffset:0 toBuffer:staging destinationOffset:0 size:c_bytes];
        [encoder endEncoding];
        complete(command);
        result.download_ms = milliseconds(start);
        if (std::filesystem::exists(path)) { throw std::runtime_error{"output already exists"}; }
        std::ofstream file{path, std::ios::binary};
        file.write(static_cast<const char *>(staging.contents), static_cast<std::streamsize>(c_bytes));
        if (!file) { throw std::runtime_error{"cannot write output"}; }
        return result;
    }
    throw std::runtime_error{"MPP tensor operations require macOS 26 or newer"};
}

void print_samples(std::string_view name, const std::vector<double> &samples) {
    std::cout << std::quoted(name) << ":[";
    for (auto i = size_t{0u}; i < samples.size(); i++) {
        if (i != 0u) { std::cout << ','; }
        std::cout << samples[i];
    }
    std::cout << ']';
}

}// namespace

int main(int argc, char *argv[]) {
    @autoreleasepool {
        try {
            if (argc != 9 && argc != 14 && argc != 15 && argc != 16 && argc != 18) {
                throw std::invalid_argument{
                    "Usage: benchmark_tile_mpp <fp32|fp16|fp16-fp32|bf16|bf16-fp32> M N K samples sample-ms warmup-ms output "
                    "[tile-m tile-n simdgroups cooperative-output relaxed-precision [static-reduction [inline-tensors [group-simdgroups cohort-rows]]]]"};
            }
            auto name = std::string_view{argv[1]};
            Configuration cfg{positive_integer(argv[2]), positive_integer(argv[3]), positive_integer(argv[4]),
                              positive_integer(argv[5]), positive_integer(argv[6]), positive_integer(argv[7])};
            if (argc >= 14) {
                cfg.tile_m = positive_integer(argv[9]);
                cfg.tile_n = positive_integer(argv[10]);
                cfg.simdgroups = positive_integer(argv[11]);
                cfg.cooperative = boolean(argv[12], "cooperative-output");
                cfg.relaxed_precision = boolean(argv[13], "relaxed-precision");
            }
            if (argc >= 15) { cfg.static_reduction = boolean(argv[14], "static-reduction"); }
            if (argc >= 16) { cfg.inline_tensors = boolean(argv[15], "inline-tensors"); }
            if (argc == 18) {
                cfg.group_simdgroups = positive_integer(argv[16]);
                cfg.cohort_rows = positive_integer(argv[17]);
            }
            auto result = measure(name, cfg, argv[8]);
            std::cout << std::setprecision(12)
                      << "{\"backend\":\"metal\",\"implementation\":\"mpp_tensor_ops_matmul2d\""
                      << ",\"precision\":" << std::quoted(name)
                      << ",\"m\":" << cfg.m << ",\"n\":" << cfg.n << ",\"k\":" << cfg.k
                      << ",\"device\":" << std::quoted(result.device)
                      << ",\"compiler\":" << std::quoted(__clang_version__)
                      << ",\"execution_simdgroups\":" << cfg.simdgroups
                      << ",\"group_simdgroups\":" << (cfg.group_simdgroups == 0 ? cfg.simdgroups : cfg.group_simdgroups)
                      << ",\"cohort_rows\":" << cfg.cohort_rows
                      << ",\"block\":[" << cfg.tile_m << ',' << cfg.tile_n << ']'
                      << ",\"cooperative_output\":" << (cfg.cooperative ? "true" : "false")
                      << ",\"relaxed_precision\":" << (cfg.relaxed_precision ? "true" : "false")
                      << ",\"static_reduction\":" << (cfg.static_reduction ? "true" : "false")
                      << ",\"inline_tensors\":" << (cfg.inline_tensors ? "true" : "false")
                      << ",\"command_api\":" << std::quoted(cfg.inline_tensors ? "MTLCommandQueue" : "MTL4CommandQueue")
                      << ",\"fast_math\":" << (cfg.relaxed_precision ? "true" : "false")
                      << ",\"thread_execution_width\":" << result.thread_execution_width
                      << ",\"static_threadgroup_bytes\":" << result.static_threadgroup_bytes
                      << ",\"max_threads_per_group\":" << result.max_threads_per_group
                      << ",\"setup_ms\":" << result.setup_ms << ",\"cold_call_ms\":" << result.cold_ms
                      << ",\"warmup_ms\":" << result.warmup_ms << ",\"download_ms\":" << result.download_ms
                      << ",\"repetitions\":" << result.repetitions << ',';
            print_samples("throughput_us", result.throughput);
            std::cout << ',';
            print_samples("latency_us", result.latency);
            std::cout << ',';
            print_samples("gpu_throughput_us", result.gpu_throughput);
            std::cout << ',';
            print_samples("gpu_latency_us", result.gpu_latency);
            std::cout << "}\n";
            return 0;
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}

#else

int main() {
    std::cerr << "The MPP benchmark requires a macOS 26 SDK and macOS 26 or newer at runtime.\n";
    return 1;
}

#endif
