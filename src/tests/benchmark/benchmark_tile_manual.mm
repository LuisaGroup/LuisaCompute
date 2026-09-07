// Hand-written Metal GEMM realizations used to establish an implementation
// ceiling before teaching the Tile planner about a schedule. This is not a
// system-library baseline and is intentionally restricted to exact FP32 tiles.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
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

namespace {

using Clock = std::chrono::steady_clock;

struct Configuration {
    int m;
    int n;
    int k;
    int samples;
    int sample_ms;
    int warmup_ms;
};

struct Measurement {
    std::string device;
    double setup_ms{};
    double cold_ms{};
    double warmup_ms{};
    double download_ms{};
    uint64_t repetitions{};
    uint64_t max_threadgroup_bytes{};
    uint64_t static_threadgroup_bytes{};
    std::vector<double> throughput;
    std::vector<double> latency;
};

constexpr auto metal_source = R"metal(
#include <metal_stdlib>
using namespace metal;

#if VARIANT != 2
#if VARIANT == 1
constant uint shared_slots = 2;
#else
constant uint shared_slots = 1;
#endif
#endif

inline uint compact_morton_axis(uint value) {
    value &= 0x55555555u;
    value = (value | (value >> 1)) & 0x33333333u;
    value = (value | (value >> 2)) & 0x0f0f0f0fu;
    value = (value | (value >> 4)) & 0x00ff00ffu;
    value = (value | (value >> 8)) & 0x0000ffffu;
    return value;
}

inline void copy_tile(device const float *a,
                      device const float *b,
                      threadgroup float *shared_a,
                      threadgroup float *shared_b,
                      uint block_m,
                      uint block_n,
                      uint tile,
                      uint tid) {
#pragma unroll
    for (uint chunk = 0; chunk < 2; chunk++) {
        const uint index = chunk * 1024 + tid;
        const uint row = index >> 5;
        const uint column = index & 31;
        const uint source = (block_m + row) * K + tile * 32 + column;
        float v0 = a[source];
        float v1 = a[source + 8 * K];
        float v2 = a[source + 16 * K];
        float v3 = a[source + 24 * K];
        shared_a[row * A_STRIDE + column] = v0;
        shared_a[(row + 8) * A_STRIDE + column] = v1;
        shared_a[(row + 16) * A_STRIDE + column] = v2;
        shared_a[(row + 24) * A_STRIDE + column] = v3;
    }
#pragma unroll
    for (uint chunk = 0; chunk < 2; chunk++) {
        const uint index = chunk * 1024 + tid;
        const uint row = index >> 6;
        const uint column = index & 63;
        const uint source = (tile * 32 + row) * N + block_n + column;
        float v0 = b[source];
        float v1 = b[source + 4 * N];
        float v2 = b[source + 8 * N];
        float v3 = b[source + 12 * N];
        shared_b[row * B_STRIDE + column] = v0;
        shared_b[(row + 4) * B_STRIDE + column] = v1;
        shared_b[(row + 8) * B_STRIDE + column] = v2;
        shared_b[(row + 12) * B_STRIDE + column] = v3;
    }
}

kernel void manual_gemm(device const float *a [[buffer(0)]],
                        device const float *b [[buffer(1)]],
                        device float *c [[buffer(2)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]]) {
    simdgroup_float8x8 acc[8];
#pragma unroll
    for (uint i = 0; i < 8; i++) {
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const uint subgroup = tid >> 5;
    const uint subgroup_m = subgroup >> 2;
    const uint subgroup_n = subgroup & 3;
    const uint groups_m = M / 64;
    const uint groups_n = N / 64;
#if VARIANT == 3
    const uint tile_n = compact_morton_axis(group);
    const uint tile_m = compact_morton_axis(group >> 1);
#elif VARIANT == 4
    const uint tile_m = group % groups_m;
    const uint tile_n = group / groups_m;
#else
    const uint tile_m = group / groups_n;
    const uint tile_n = group % groups_n;
#endif
    const uint block_m = tile_m * 64;
    const uint block_n = tile_n * 64;

#if VARIANT != 2
    threadgroup float shared_a[shared_slots * A_SHARED_ELEMENTS];
    threadgroup float shared_b[shared_slots * B_SHARED_ELEMENTS];
#endif

#if VARIANT == 1
    copy_tile(a, b, shared_a, shared_b, block_m, block_n, 0, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif

#pragma unroll 1
    for (uint tile = 0; tile < K / 32; tile++) {
#if VARIANT != 1 && VARIANT != 2
        copy_tile(a, b, shared_a, shared_b, block_m, block_n, tile, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
#elif VARIANT == 1
        const uint slot = tile & 1;
        const uint next = tile + 1;
        if (next < K / 32) {
            const uint next_a_base = (slot ^ 1) * A_SHARED_ELEMENTS;
            const uint next_b_base = (slot ^ 1) * B_SHARED_ELEMENTS;
            copy_tile(a, b, shared_a + next_a_base, shared_b + next_b_base,
                      block_m, block_n, next, tid);
        }
#endif

#if VARIANT == 9
        simdgroup_float8x8 af;
#else
        simdgroup_float8x8 af[4];
#endif
        simdgroup_float8x8 bf[2];
#pragma unroll
        for (uint step = 0; step < 4; step++) {
#if VARIANT == 2
            const uint contraction = tile * 32 + step * 8;
            const device float *ap = a + (block_m + subgroup_m * 32) * K + contraction;
            const device float *bp = b + contraction * N + block_n + subgroup_n * 16;
#else
#if VARIANT == 1
            const uint a_base = (tile & 1) * A_SHARED_ELEMENTS;
            const uint b_base = (tile & 1) * B_SHARED_ELEMENTS;
#else
            const uint a_base = 0;
            const uint b_base = 0;
#endif
            const threadgroup float *ap = shared_a + a_base + subgroup_m * 32 * A_STRIDE + step * 8;
            const threadgroup float *bp = shared_b + b_base + step * 8 * B_STRIDE + subgroup_n * 16;
#endif
            simdgroup_load(bf[0], bp, B_LOAD_STRIDE, 0, false);
            simdgroup_load(bf[1], bp + 8, B_LOAD_STRIDE, 0, false);
#if VARIANT == 9
            simdgroup_load(af, ap, A_LOAD_STRIDE, 0, false);
            simdgroup_multiply_accumulate(acc[0], af, bf[0], acc[0]);
            simdgroup_multiply_accumulate(acc[1], af, bf[1], acc[1]);
            simdgroup_load(af, ap + 8 * A_LOAD_STRIDE, A_LOAD_STRIDE, 0, false);
            simdgroup_multiply_accumulate(acc[2], af, bf[0], acc[2]);
            simdgroup_multiply_accumulate(acc[3], af, bf[1], acc[3]);
            simdgroup_load(af, ap + 16 * A_LOAD_STRIDE, A_LOAD_STRIDE, 0, false);
            simdgroup_multiply_accumulate(acc[4], af, bf[0], acc[4]);
            simdgroup_multiply_accumulate(acc[5], af, bf[1], acc[5]);
            simdgroup_load(af, ap + 24 * A_LOAD_STRIDE, A_LOAD_STRIDE, 0, false);
            simdgroup_multiply_accumulate(acc[6], af, bf[0], acc[6]);
            simdgroup_multiply_accumulate(acc[7], af, bf[1], acc[7]);
#else
            simdgroup_load(af[0], ap, A_LOAD_STRIDE, 0, false);
            simdgroup_load(af[1], ap + 8 * A_LOAD_STRIDE, A_LOAD_STRIDE, 0, false);
            simdgroup_load(af[2], ap + 16 * A_LOAD_STRIDE, A_LOAD_STRIDE, 0, false);
            simdgroup_load(af[3], ap + 24 * A_LOAD_STRIDE, A_LOAD_STRIDE, 0, false);
            simdgroup_multiply_accumulate(acc[0], af[0], bf[0], acc[0]);
            simdgroup_multiply_accumulate(acc[1], af[0], bf[1], acc[1]);
            simdgroup_multiply_accumulate(acc[2], af[1], bf[0], acc[2]);
            simdgroup_multiply_accumulate(acc[3], af[1], bf[1], acc[3]);
            simdgroup_multiply_accumulate(acc[4], af[2], bf[0], acc[4]);
            simdgroup_multiply_accumulate(acc[5], af[2], bf[1], acc[5]);
            simdgroup_multiply_accumulate(acc[6], af[3], bf[0], acc[6]);
            simdgroup_multiply_accumulate(acc[7], af[3], bf[1], acc[7]);
#endif
        }
#if VARIANT != 2
        threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    }

    device float *output = c + (block_m + subgroup_m * 32) * N + block_n + subgroup_n * 16;
    simdgroup_store(acc[0], output, N, 0, false);
    simdgroup_store(acc[1], output + 8, N, 0, false);
    simdgroup_store(acc[2], output + 8 * N, N, 0, false);
    simdgroup_store(acc[3], output + 8 * N + 8, N, 0, false);
    simdgroup_store(acc[4], output + 16 * N, N, 0, false);
    simdgroup_store(acc[5], output + 16 * N + 8, N, 0, false);
    simdgroup_store(acc[6], output + 24 * N, N, 0, false);
    simdgroup_store(acc[7], output + 24 * N + 8, N, 0, false);
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

[[nodiscard]] int variant(std::string_view name) {
    if (name == "shared") { return 0; }
    if (name == "double") { return 1; }
    if (name == "direct") { return 2; }
    if (name == "morton") { return 3; }
    if (name == "column") { return 4; }
    if (name == "pad-a1") { return 5; }
    if (name == "pad-b1") { return 6; }
    if (name == "pad1") { return 7; }
    if (name == "pad8") { return 8; }
    if (name == "stream-a") { return 9; }
    throw std::invalid_argument{"unknown manual GEMM variant"};
}

[[nodiscard]] size_t elements(int rows, int columns) {
    auto count = static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns);
    if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
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

[[nodiscard]] Measurement measure(std::string_view name, Configuration cfg, const char *path) {
    if (cfg.m % 64 != 0 || cfg.n % 64 != 0 || cfg.k % 32 != 0) {
        throw std::invalid_argument{"manual GEMM requires M/N multiples of 64 and K a multiple of 32"};
    }
    auto a = input_values(elements(cfg.m, cfg.k), 5u);
    auto b = input_values(elements(cfg.k, cfg.n), 11u);
    std::vector<float> c(elements(cfg.m, cfg.n), std::numeric_limits<float>::quiet_NaN());
    auto device = MTLCreateSystemDefaultDevice();
    if (device == nil) { throw std::runtime_error{"Metal device unavailable"}; }
    auto queue = [device newCommandQueue];
    if (queue == nil) { throw std::runtime_error{"Metal command queue unavailable"}; }
    auto mode = variant(name);
    auto a_stride = mode == 5 || mode == 7 ? 33 : mode == 8 ? 40 : 32;
    auto b_stride = mode == 6 || mode == 7 ? 65 : mode == 8 ? 72 : 64;
    auto prefix = [NSString stringWithFormat:@"#define M %d\n#define N %d\n#define K %d\n#define VARIANT %d\n#define A_STRIDE %d\n#define B_STRIDE %d\n#define A_SHARED_ELEMENTS %d\n#define B_SHARED_ELEMENTS %d\n#define A_LOAD_STRIDE %s\n#define B_LOAD_STRIDE %s\n",
                                               cfg.m, cfg.n, cfg.k, mode, a_stride, b_stride,
                                               64 * a_stride, 32 * b_stride,
                                               mode == 2 ? "K" : "A_STRIDE", mode == 2 ? "N" : "B_STRIDE"];
    auto source = [prefix stringByAppendingString:[NSString stringWithUTF8String:metal_source]];
    auto options = [MTLCompileOptions new];
    options.fastMathEnabled = YES;
    options.languageVersion = MTLLanguageVersion3_0;
    NSError *error = nil;
    auto start = Clock::now();
    auto library = [device newLibraryWithSource:source options:options error:&error];
    if (library == nil) {
        throw std::runtime_error{error == nil ? "Metal source compilation failed" : error.localizedDescription.UTF8String};
    }
    auto function = [library newFunctionWithName:@"manual_gemm"];
    if (function == nil) { throw std::runtime_error{"manual_gemm function unavailable"}; }
    auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw std::runtime_error{error == nil ? "Metal pipeline creation failed" : error.localizedDescription.UTF8String};
    }
    auto a_buffer = make_buffer(device, a.size() * sizeof(float), MTLResourceStorageModePrivate);
    auto b_buffer = make_buffer(device, b.size() * sizeof(float), MTLResourceStorageModePrivate);
    auto c_buffer = make_buffer(device, c.size() * sizeof(float), MTLResourceStorageModePrivate);
    auto upload = [&](id<MTLBuffer> destination, const void *data, size_t bytes) {
        auto staging = make_buffer(device, bytes, MTLResourceStorageModeShared);
        std::memcpy(staging.contents, data, bytes);
        auto command = [queue commandBuffer];
        auto blit = [command blitCommandEncoder];
        if (blit == nil) { throw std::runtime_error{"Metal upload encoder unavailable"}; }
        [blit copyFromBuffer:staging sourceOffset:0 toBuffer:destination destinationOffset:0 size:bytes];
        [blit endEncoding];
        complete(command);
    };
    upload(a_buffer, a.data(), a.size() * sizeof(float));
    upload(b_buffer, b.data(), b.size() * sizeof(float));

    Measurement result{};
    result.device = device.name.UTF8String;
    result.max_threadgroup_bytes = device.maxThreadgroupMemoryLength;
    result.static_threadgroup_bytes = pipeline.staticThreadgroupMemoryLength;
    result.setup_ms = milliseconds(start);
    auto batch = [&](uint64_t repetitions) {
        @autoreleasepool {
            auto begin = Clock::now();
            auto command = [queue commandBuffer];
            auto encoder = [command computeCommandEncoder];
            if (encoder == nil) { throw std::runtime_error{"Metal compute encoder unavailable"}; }
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:a_buffer offset:0 atIndex:0];
            [encoder setBuffer:b_buffer offset:0 atIndex:1];
            [encoder setBuffer:c_buffer offset:0 atIndex:2];
            auto groups = MTLSizeMake(static_cast<NSUInteger>(cfg.m / 64 * (cfg.n / 64)), 1u, 1u);
            auto threads = MTLSizeMake(256u, 1u, 1u);
            for (auto i = uint64_t{0u}; i < repetitions; i++) {
                [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
            }
            [encoder endEncoding];
            complete(command);
            return milliseconds(begin);
        }
    };
    result.cold_ms = batch(1u);
    start = Clock::now();
    while (milliseconds(start) < cfg.warmup_ms) { static_cast<void>(batch(8u)); }
    result.warmup_ms = milliseconds(start);
    result.repetitions = 1u;
    for (auto i = 0; i < 8; i++) {
        auto elapsed = batch(result.repetitions);
        if (elapsed >= cfg.sample_ms * 0.8 || result.repetitions == 100000u) { break; }
        auto estimate = result.repetitions * static_cast<double>(cfg.sample_ms) / std::max(elapsed, 1e-6);
        result.repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(estimate), result.repetitions + 1u, 100000u);
    }
    for (auto i = 0; i < cfg.samples; i++) {
        result.throughput.push_back(1000.0 * batch(result.repetitions) / result.repetitions);
    }
    for (auto i = 0; i < cfg.samples; i++) { result.latency.push_back(1000.0 * batch(1u)); }
    start = Clock::now();
    auto staging = make_buffer(device, c.size() * sizeof(float), MTLResourceStorageModeShared);
    auto command = [queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    if (blit == nil) { throw std::runtime_error{"Metal download encoder unavailable"}; }
    [blit copyFromBuffer:c_buffer sourceOffset:0 toBuffer:staging destinationOffset:0 size:staging.length];
    [blit endEncoding];
    complete(command);
    std::memcpy(c.data(), staging.contents, c.size() * sizeof(float));
    result.download_ms = milliseconds(start);
    if (std::filesystem::exists(path)) { throw std::runtime_error{"output already exists"}; }
    std::ofstream file{path, std::ios::binary};
    file.write(reinterpret_cast<const char *>(c.data()), static_cast<std::streamsize>(c.size() * sizeof(float)));
    if (!file) { throw std::runtime_error{"cannot write output"}; }
    return result;
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
            if (argc != 9) {
                throw std::invalid_argument{"Usage: benchmark_tile_manual VARIANT M N K samples sample-ms warmup-ms output.f32"};
            }
            auto name = std::string_view{argv[1]};
            Configuration cfg{positive_integer(argv[2]), positive_integer(argv[3]), positive_integer(argv[4]),
                              positive_integer(argv[5]), positive_integer(argv[6]), positive_integer(argv[7])};
            auto result = measure(name, cfg, argv[8]);
            std::cout << std::setprecision(12)
                      << "{\"backend\":\"metal\",\"implementation\":\"manual_simdgroup_gemm\""
                      << ",\"variant\":" << std::quoted(name) << ",\"dtype\":\"float32\""
                      << ",\"m\":" << cfg.m << ",\"n\":" << cfg.n << ",\"k\":" << cfg.k
                      << ",\"device\":" << std::quoted(result.device)
                      << ",\"threads_per_group\":256,\"block\":[64,64,32]"
                      << ",\"max_threadgroup_bytes\":" << result.max_threadgroup_bytes
                      << ",\"static_threadgroup_bytes\":" << result.static_threadgroup_bytes
                      << ",\"setup_ms\":" << result.setup_ms << ",\"cold_call_ms\":" << result.cold_ms
                      << ",\"warmup_ms\":" << result.warmup_ms << ",\"download_ms\":" << result.download_ms
                      << ",\"repetitions\":" << result.repetitions << ',';
            print_samples("throughput_us", result.throughput);
            std::cout << ',';
            print_samples("latency_us", result.latency);
            std::cout << "}\n";
            return 0;
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
