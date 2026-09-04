// Direct Apple-library baselines. Intentionally independent of TileIR/TVM.
// Compact row-major FP32 C = A * B; no transpose, packing, or reduced precision.
// Warm timing includes API/encoding/submission overhead, not setup or transfers.
#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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
    int samples{1};
    int sample_ms{1};
    int warmup_ms{1};
};

struct Measurement {
    std::string device;
    double setup_ms;
    double cold_ms;
    double warmup_ms;
    double download_ms;
    uint64_t repetitions;
    std::vector<double> throughput;
    std::vector<double> latency;
    std::vector<double> gpu_throughput;
    std::vector<double> gpu_latency;
};

[[nodiscard]] int positive_integer(const char *text) {
    auto input = std::string_view{text};
    auto value = 0;
    auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size() || value <= 0) {
        throw std::invalid_argument{"expected a positive int32"};
    }
    return value;
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

void complete(id<MTLCommandBuffer> command) {
    if (command == nil) { throw std::runtime_error{"cannot create Metal command buffer"}; }
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        auto reason = command.error.localizedDescription;
        throw std::runtime_error{reason == nil ? "Metal command failed" : reason.UTF8String};
    }
}

[[nodiscard]] id<MTLBuffer> buffer(id<MTLDevice> device, size_t bytes, MTLResourceOptions options) {
    auto result = [device newBufferWithLength:bytes options:options];
    if (result == nil) { throw std::runtime_error{"Metal buffer allocation failed"}; }
    return result;
}

void validate(const Configuration &cfg, const std::vector<float> &a,
              const std::vector<float> &b, const std::vector<float> &c) {
    for (auto i = 0; i < cfg.m; i++) {
        for (auto j = 0; j < cfg.n; j++) {
            auto expected = 0.0;
            for (auto k = 0; k < cfg.k; k++) {
                expected += static_cast<double>(a[static_cast<size_t>(i) * cfg.k + k]) * b[static_cast<size_t>(k) * cfg.n + j];
            }
            auto actual = c[static_cast<size_t>(i) * cfg.n + j];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-4 + 1e-4 * std::abs(expected)) {
                throw std::runtime_error{"full FP64-oracle comparison failed"};
            }
        }
    }
}

[[nodiscard]] Measurement measure(std::string_view backend, Configuration cfg, const char *path) {
    auto a = input_values(elements(cfg.m, cfg.k), 5u);
    auto b = input_values(elements(cfg.k, cfg.n), 11u);
    std::vector<float> c(elements(cfg.m, cfg.n), std::numeric_limits<float>::quiet_NaN());
    Measurement result{};
    double gpu_ms{};
    std::function<double(uint64_t)> batch;
    std::function<void()> download = [] {};
    auto start = Clock::now();
    if (backend == "cpu") {
        result.device = "Accelerate CPU";
        batch = [&](uint64_t repetitions) {
            auto begin = Clock::now();
            for (auto i = uint64_t{0u}; i < repetitions; i++) {
                // Deliberately use the classic LP64 API, also exposed by
                // PyTorch's Accelerate build; record this choice in JSON.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            cfg.m, cfg.n, cfg.k, 1.0f, a.data(), cfg.k,
                            b.data(), cfg.n, 0.0f, c.data(), cfg.n);
#pragma clang diagnostic pop
            }
            return milliseconds(begin);
        };
    } else if (backend == "metal") {
        auto device = MTLCreateSystemDefaultDevice();
        if (device == nil || !MPSSupportsMTLDevice(device)) { throw std::runtime_error{"MPS device unavailable; no CPU fallback"}; }
        result.device = device.name.UTF8String;
        auto queue = [device newCommandQueue];
        if (queue == nil) { throw std::runtime_error{"Metal command queue unavailable"}; }
        auto make_matrix = [&](size_t rows, size_t columns) {
            auto storage = buffer(device, rows * columns * sizeof(float), MTLResourceStorageModePrivate);
            auto descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                    columns:columns
                                                                   rowBytes:columns * sizeof(float)
                                                                   dataType:MPSDataTypeFloat32];
            auto matrix = [[MPSMatrix alloc] initWithBuffer:storage descriptor:descriptor];
            if (matrix == nil) { throw std::runtime_error{"MPS matrix creation failed"}; }
            return matrix;
        };
        auto left = make_matrix(cfg.m, cfg.k);
        auto right = make_matrix(cfg.k, cfg.n);
        auto output = make_matrix(cfg.m, cfg.n);
        auto kernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                        transposeLeft:NO
                                                       transposeRight:NO
                                                           resultRows:cfg.m
                                                        resultColumns:cfg.n
                                                      interiorColumns:cfg.k
                                                                alpha:1.0
                                                                 beta:0.0];
        if (kernel == nil) { throw std::runtime_error{"MPS GEMM creation failed"}; }
        kernel.options = MPSKernelOptionsNone;
        // Upload once. Timed calls reuse private device buffers and output.
        auto upload = [&](MPSMatrix *matrix, const std::vector<float> &values) {
            auto staging = buffer(device, values.size() * sizeof(float), MTLResourceStorageModeShared);
            std::memcpy(staging.contents, values.data(), values.size() * sizeof(float));
            auto command = [queue commandBuffer];
            auto blit = [command blitCommandEncoder];
            if (blit == nil) { throw std::runtime_error{"Metal upload encoder unavailable"}; }
            [blit copyFromBuffer:staging sourceOffset:0 toBuffer:matrix.data destinationOffset:0 size:staging.length];
            [blit endEncoding];
            complete(command);
        };
        upload(left, a);
        upload(right, b);
        upload(output, c);// NaNs ensure beta=0 does not depend on old output.
        batch = [=, &gpu_ms](uint64_t repetitions) {
            @autoreleasepool {
                auto begin = Clock::now();
                auto command = [queue commandBuffer];
                if (command == nil) { throw std::runtime_error{"Metal command buffer unavailable"}; }
                for (auto i = uint64_t{0u}; i < repetitions; i++) {
                    [kernel encodeToCommandBuffer:command leftMatrix:left rightMatrix:right resultMatrix:output];
                }
                complete(command);
                gpu_ms = 1000.0 * (command.GPUEndTime - command.GPUStartTime);
                return milliseconds(begin);
            }
        };
        download = [=, &c] {
            auto staging = buffer(device, c.size() * sizeof(float), MTLResourceStorageModeShared);
            auto command = [queue commandBuffer];
            auto blit = [command blitCommandEncoder];
            if (blit == nil) { throw std::runtime_error{"Metal download encoder unavailable"}; }
            [blit copyFromBuffer:output.data sourceOffset:0 toBuffer:staging destinationOffset:0 size:staging.length];
            [blit endEncoding];
            complete(command);
            std::memcpy(c.data(), staging.contents, c.size() * sizeof(float));
        };
    } else {
        throw std::invalid_argument{"backend must be cpu or metal"};
    }
    result.setup_ms = milliseconds(start);
    result.cold_ms = batch(1u);
    start = Clock::now();
    while (milliseconds(start) < cfg.warmup_ms) { static_cast<void>(batch(8u)); }
    result.warmup_ms = milliseconds(start);
    result.repetitions = 1u;
    for (auto i = 0; i < 8; i++) {
        auto elapsed = batch(result.repetitions);
        if (elapsed >= cfg.sample_ms * 0.8 || result.repetitions == 100000u) { break; }
        auto scaled = std::clamp(result.repetitions * cfg.sample_ms / std::max(elapsed, 1e-6), 1.0, 100000.0);
        result.repetitions = std::min(uint64_t{100000u}, std::max(result.repetitions + 1u, static_cast<uint64_t>(scaled)));
    }
    for (auto i = 0; i < cfg.samples; i++) {
        result.throughput.push_back(1000.0 * batch(result.repetitions) / result.repetitions);
        if (backend == "metal") { result.gpu_throughput.push_back(1000.0 * gpu_ms / result.repetitions); }
    }
    for (auto i = 0; i < cfg.samples; i++) {
        result.latency.push_back(1000.0 * batch(1u));
        if (backend == "metal") { result.gpu_latency.push_back(1000.0 * gpu_ms); }
    }
    start = Clock::now();
    download();
    result.download_ms = milliseconds(start);
    if (path == nullptr) {
        validate(cfg, a, b, c);
    } else {
        // The Python driver validates every element against its shared FP64
        // oracle. A successful subprocess alone is not a correctness claim.
        if (std::filesystem::exists(path)) { throw std::runtime_error{"output already exists"}; }
        std::ofstream file{path, std::ios::binary};
        file.write(reinterpret_cast<const char *>(c.data()), static_cast<std::streamsize>(c.size() * sizeof(float)));
        if (!file) { throw std::runtime_error{"cannot write output"}; }
    }
    return result;
}

void print_samples(std::string_view name, const std::vector<double> &samples) {
    std::cout << std::quoted(name) << ":[";
    auto separator = "";
    for (auto value : samples) {
        std::cout << separator << value;
        separator = ",";
    }
    std::cout << ']';
}

}// namespace

int main(int argc, char *argv[]) {
    @autoreleasepool {
        try {
            if (argc == 3 && std::string_view{argv[1]} == "--self-test") {
                for (auto cfg : {Configuration{1, 1, 1}, Configuration{7, 19, 13}, Configuration{32, 32, 32}, Configuration{17, 8, 33}}) {
                    static_cast<void>(measure(argv[2], cfg, nullptr));
                }
                std::cout << "Four shapes passed full FP64 validation, including repeated beta=0 calls.\n";
                return 0;
            }
            if (argc != 9) {
                throw std::invalid_argument{"Usage: benchmark_tile_system <cpu|metal> M N K samples sample-ms warmup-ms output.f32"};
            }
            auto backend = std::string_view{argv[1]};
            Configuration cfg{positive_integer(argv[2]), positive_integer(argv[3]), positive_integer(argv[4]),
                              positive_integer(argv[5]), positive_integer(argv[6]), positive_integer(argv[7])};
            auto result = measure(backend, cfg, argv[8]);
            std::cout << std::setprecision(12) << "{\"backend\":" << std::quoted(backend)
                      << ",\"implementation\":" << std::quoted(backend == "cpu" ? "accelerate_cblas_sgemm" : "mps_matrix_multiplication")
                      << ",\"api_variant\":" << std::quoted(backend == "cpu" ? "classic_lp64" : "MPSKernelOptionsNone")
                      << ",\"operation\":\"gemm\",\"dtype\":\"float32\",\"layout\":\"compact_row_major\""
                      << ",\"alpha\":1,\"beta\":0,\"transpose_left\":false,\"transpose_right\":false"
                      << ",\"m\":" << cfg.m << ",\"n\":" << cfg.n << ",\"k\":" << cfg.k
                      << ",\"row_bytes\":[" << static_cast<uint64_t>(cfg.k) * 4u << ',' << static_cast<uint64_t>(cfg.n) * 4u << ',' << static_cast<uint64_t>(cfg.n) * 4u << ']'
                      << ",\"device\":" << std::quoted(result.device)
                      << ",\"storage\":" << std::quoted(backend == "cpu" ? "host" : "private")
                      << ",\"batch_policy\":" << std::quoted(backend == "cpu" ? "synchronous_calls" : "one_command_buffer_per_batch")
                      << ",\"compiler\":" << std::quoted(__clang_version__)
                      << ",\"setup_ms\":" << result.setup_ms << ",\"cold_call_ms\":" << result.cold_ms
                      << ",\"warmup_ms\":" << result.warmup_ms << ",\"download_ms\":" << result.download_ms
                      << ",\"repetitions\":" << result.repetitions << ',';
            print_samples("throughput_us", result.throughput);
            std::cout << ',';
            print_samples("latency_us", result.latency);
            if (backend == "metal") {
                std::cout << ',';
                print_samples("gpu_throughput_us", result.gpu_throughput);
                std::cout << ',';
                print_samples("gpu_latency_us", result.gpu_latency);
            }
            std::cout << "}\n";
            return 0;
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
