#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <random>
#include <chrono>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>

#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include "fp8.h"

using namespace luisa;
using namespace luisa::compute;

// ==================== Test Kernel ====================
// Regular FP32 compute kernel for comparison
Kernel2D fp32_compute_kernel = [](BufferVar<uint> input_buf,
                                  BufferVar<float> out_e4m3_fp32,
                                  BufferVar<float> out_e5m2_fp32) noexcept {
    set_block_size(16, 16, 1);
    auto idx = dispatch_id().x + dispatch_id().y * dispatch_size().x;
    auto v = input_buf.read(idx).cast<float>();
    out_e4m3_fp32.write(idx, v);
    out_e5m2_fp32.write(idx, v * v);
};

Kernel2D fp8_test_kernel = [](BufferVar<uint> input_buf,
                              BufferVar<uint> out_to) noexcept {
    set_block_size(16, 16, 1);
    auto input_idx = dispatch_id().x + dispatch_id().y * dispatch_size().x;
    auto idx = input_idx * 4u;
    auto N = dispatch_size().x * dispatch_size().y * 4;

    UInt int_val = input_buf.read(input_idx);
    UInt result = 0;
    for (auto i : dynamic_range(4)) {
        auto bits = (int_val >> (i * 8)) & 255u;
        auto v = fp8e4m3_to_float()(bits);
        v *= v;
        result <<= 8u;
        result |= fp8e4m3_from_float()(v);
    }
    out_to.write(idx, result);
};

// ==================== Main ====================

int main(int argc, char **argv) {
    constexpr size_t N_SIZE = 4096;
    constexpr size_t N = N_SIZE * N_SIZE;
    constexpr int WARMUP_ITERS = 10;
    constexpr int PROFILE_ITERS = 128;

    // Generate random test values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> test_values(N);
    for (size_t i = 0; i < N; ++i) {
        test_values[i] = dist(rng);
    }

    // CPU reference results
    std::vector<uint8_t> cpu_e4m3_from(N);
    std::vector<uint8_t> cpu_e5m2_from(N);
    std::vector<float> cpu_e4m3_to(N);
    std::vector<float> cpu_e5m2_to(N);
    for (size_t i = 0; i < N; ++i) {
        cpu_e4m3_from[i] = FP8E4M3::from_float(test_values[i]);
        cpu_e5m2_from[i] = FP8E5M2::from_float(test_values[i]);
        cpu_e4m3_to[i] = FP8E4M3::to_float(cpu_e4m3_from[i]);
        cpu_e5m2_to[i] = FP8E5M2::to_float(cpu_e5m2_from[i]);
    }

    // GPU setup
    Context context{argv[0]};
    luisa::string backend_name = "dx";
    Device device = context.create_device(backend_name);
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    auto input_buf = device.create_buffer<uint>(N);
    auto out_to = device.create_buffer<uint>(2 * N);
    auto out_fp32_e4m3 = device.create_buffer<float>(N);
    auto out_fp32_e5m2 = device.create_buffer<float>(N);

    // Upload test data
    stream << input_buf.copy_from(eastl::span{test_values.data(), test_values.size()})
           << synchronize();

    auto fp8_shader = device.compile(fp8_test_kernel);
    auto fp32_shader = device.compile(fp32_compute_kernel);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        stream << fp8_shader(input_buf, out_to).dispatch(N_SIZE, N_SIZE / 4u)
               << synchronize();
        stream << fp32_shader(input_buf, out_fp32_e4m3, out_fp32_e5m2).dispatch(N_SIZE, N_SIZE)
               << synchronize();
    }

    // Profile FP8 kernel
    auto t0 = std::chrono::high_resolution_clock::now();
    CommandList cmdlist;
    for (int i = 0; i < PROFILE_ITERS; ++i) {
        cmdlist << fp8_shader(input_buf, out_to).dispatch(N_SIZE, N_SIZE / 4u);
    }
    stream << cmdlist.commit() << synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp8_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / PROFILE_ITERS;

    // Profile FP32 kernel
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < PROFILE_ITERS; ++i) {
        cmdlist << fp32_shader(input_buf, out_fp32_e4m3, out_fp32_e5m2).dispatch(N_SIZE, N_SIZE);
    }
    stream << cmdlist.commit() << synchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / PROFILE_ITERS;

    LUISA_INFO("========================================");
    LUISA_INFO("Profile Results (N={}):", N);
    LUISA_INFO("  FP8  kernel avg: {} ms", fp8_ms);
    LUISA_INFO("  FP32 kernel avg: {} ms", fp32_ms);
    if (fp8_ms > 0.0) {
        LUISA_INFO("  FP32 / FP8 ratio: {}", fp32_ms / fp8_ms);
    }
    LUISA_INFO("========================================");
}
