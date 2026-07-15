#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>

#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include "fp4.h"

using namespace luisa;
using namespace luisa::compute;

// ==================== Kernels ====================

// FP32 reference: read float, write v*v
Kernel2D fp32_compute_kernel = [](BufferVar<float> input_buf,
                                  BufferVar<float> out_buf) noexcept {
    set_block_size(16, 16, 1);
    auto idx = dispatch_id().x + dispatch_id().y * dispatch_size().x;
    auto v = input_buf.read(idx);
    out_buf.write(idx, v * v);
};

// FP4 packed kernel: each uint holds 8 nibbles
Kernel2D fp4_test_kernel = [](BufferVar<uint> input_buf,
                              BufferVar<uint> out_to) noexcept {
    set_block_size(16, 16, 1);
    auto input_idx = dispatch_id().x + dispatch_id().y * dispatch_size().x;
    UInt int_val = input_buf.read(input_idx);
    UInt result = 0;
    for (auto i : dynamic_range(8)) {
        auto bits = (int_val >> (i * 4u)) & 0x0fu;
        auto v = fp4e2m1_to_float()(bits);
        v = v * v;
        result <<= 4u;
        result |= fp4e2m1_from_float()(v);
    }
    out_to.write(input_idx, result);
};

// ==================== Main ====================

int main(int argc, char **argv) {
    constexpr size_t N_SIZE = 4096;
    constexpr size_t N = N_SIZE * N_SIZE;
    constexpr int WARMUP_ITERS = 10;
    constexpr int PROFILE_ITERS = 128;

    // Parse backend
    luisa::string backend_name = (argc > 1) ? argv[1] : "dx";

    // Generate random test values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> test_values(N);
    for (size_t i = 0; i < N; ++i) {
        test_values[i] = dist(rng);
    }

    // CPU reference: encode -> decode -> square -> re-encode
    std::vector<uint8_t> cpu_fp4_from(N);
    std::vector<float> cpu_fp4_to(N);
    std::vector<uint8_t> cpu_fp4_ref(N);
    for (size_t i = 0; i < N; ++i) {
        uint8_t nibble = FP4E2M1::from_float(test_values[i]);
        cpu_fp4_from[i] = nibble;
        float decoded = FP4E2M1::to_float(nibble);
        cpu_fp4_to[i] = decoded;
        cpu_fp4_ref[i] = FP4E2M1::from_float(decoded * decoded);
    }

    // Pack nibbles into uints (little-endian: nibble 0 in bits 0-3)
    std::vector<uint> packed_input(N / 8);
    std::vector<uint> packed_ref(N / 8);
    for (size_t j = 0; j < N / 8; ++j) {
        uint w_in = 0;
        uint w_ref = 0;
        for (int k = 0; k < 8; ++k) {
            w_in |= static_cast<uint>(cpu_fp4_from[j * 8 + k] & 0x0f) << (k * 4);
            w_ref |= static_cast<uint>(cpu_fp4_ref[j * 8 + k] & 0x0f) << (k * 4);
        }
        packed_input[j] = w_in;
        packed_ref[j] = w_ref;
    }

    // GPU setup
    Context context{argv[0]};
    Device device = context.create_device(backend_name);
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    auto input_fp4_buf = device.create_buffer<uint>(N / 8);
    auto out_fp4_buf = device.create_buffer<uint>(N / 8);
    auto input_float_buf = device.create_buffer<float>(N);
    auto out_fp32_buf = device.create_buffer<float>(N);

    // Upload
    stream << input_fp4_buf.copy_from(eastl::span{packed_input.data(), packed_input.size()})
           << input_float_buf.copy_from(eastl::span{test_values.data(), test_values.size()})
           << synchronize();

    auto fp4_shader = device.compile(fp4_test_kernel);
    auto fp32_shader = device.compile(fp32_compute_kernel);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        stream << fp4_shader(input_fp4_buf, out_fp4_buf).dispatch(N_SIZE, N_SIZE / 8u)
               << synchronize();
        stream << fp32_shader(input_float_buf, out_fp32_buf).dispatch(N_SIZE, N_SIZE)
               << synchronize();
    }

    // Profile FP4 kernel
    auto t0 = std::chrono::high_resolution_clock::now();
    CommandList cmdlist;
    for (int i = 0; i < PROFILE_ITERS; ++i) {
        cmdlist << fp4_shader(input_fp4_buf, out_fp4_buf).dispatch(N_SIZE, N_SIZE / 8u);
    }
    stream << cmdlist.commit() << synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp4_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / PROFILE_ITERS;

    // Profile FP32 kernel
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < PROFILE_ITERS; ++i) {
        cmdlist << fp32_shader(input_float_buf, out_fp32_buf).dispatch(N_SIZE, N_SIZE);
    }
    stream << cmdlist.commit() << synchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / PROFILE_ITERS;

    LUISA_INFO("========================================");
    LUISA_INFO("Profile Results (N={}):", N);
    LUISA_INFO("  FP4  kernel avg: {} ms", fp4_ms);
    LUISA_INFO("  FP32 kernel avg: {} ms", fp32_ms);
    if (fp4_ms > 0.0) {
        LUISA_INFO("  FP32 / FP4 ratio: {}", fp32_ms / fp4_ms);
    }
    LUISA_INFO("========================================");

    // Validate FP4 output against CPU reference
    std::vector<uint> gpu_packed(N / 8);
    stream << out_fp4_buf.copy_to(eastl::span{gpu_packed.data(), gpu_packed.size()})
           << synchronize();

    size_t mismatch = 0;
    float max_err = 0.0f;
    for (size_t j = 0; j < N / 8; ++j) {
        uint w = gpu_packed[j];
        for (int k = 0; k < 8; ++k) {
            uint8_t gpu_nibble = static_cast<uint8_t>((w >> (k * 4)) & 0x0f);
            uint8_t cpu_nibble = cpu_fp4_ref[j * 8 + k];
            if (gpu_nibble != cpu_nibble) {
                ++mismatch;
                float gpu_f = FP4E2M1::to_float(gpu_nibble);
                float cpu_f = FP4E2M1::to_float(cpu_nibble);
                max_err = std::max(max_err, std::abs(gpu_f - cpu_f));
            }
        }
    }

    LUISA_INFO("Validation:");
    LUISA_INFO("  Mismatches: {} / {}", mismatch, N);
    if (mismatch > 0) {
        LUISA_INFO("  Max float error: {}", max_err);
    } else {
        LUISA_INFO("  All outputs match CPU reference.");
    }
    LUISA_INFO("========================================");
}
