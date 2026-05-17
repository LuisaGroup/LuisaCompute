// Test for FP4 (E2M1) quantized 2-layer MLP.
//
// This test demonstrates:
// - CPU FP32 baseline and FP4-simulated reference
// - GPU warp-level quantized matrix multiplication with per-tensor scales
// - Dynamic activation quantization + weight-only quantization
// - ReLU activation kernel
// - Profiling with luisa::Clock
// - Packed FP4 weights (2 nibbles per byte)

#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/command_list.h>
#include <luisa/dsl/sugar.h>
#include <random>
#include <cmath>
#include <limits>
#include <vector>

#include "fp4.h"

using namespace luisa;
using namespace luisa::compute;

// FP4 E2M1 constants
constexpr float kFp4E2M1Max = 6.0f;
constexpr uint kWarpSize = 32;

// TinyMLP dimensions
constexpr uint kBatch = 4u;
constexpr uint kInDim = 128u;
constexpr uint kHiddenDim = 64u;
constexpr uint kOutDim = 10u;

// K must be a multiple of 8 so that col*(K/2) is always 4-byte aligned for uint word reads
// and each uint covers 8 packed FP4 values.
static_assert(kInDim % 8u == 0u, "kInDim must be a multiple of 8");
static_assert(kHiddenDim % 8u == 0u, "kHiddenDim must be a multiple of 8");

float compute_fp4_scale(luisa::span<const float> data) {
    float abs_max = 0.0f;
    for (auto v : data) {
        abs_max = std::max(abs_max, std::abs(v));
    }
    float scale = abs_max / kFp4E2M1Max;
    return std::max(scale, std::numeric_limits<float>::min());
}

void quantize_to_fp4_bits(luisa::span<const float> src, std::vector<uint8_t> &dst, float scale) {
    size_t packed_size = (src.size() + 1) / 2;
    dst.resize(packed_size);
    for (size_t i = 0; i < src.size(); ++i) {
        float q = std::clamp(src[i] / scale, -kFp4E2M1Max, kFp4E2M1Max);
        uint8_t nibble = FP4E2M1::from_float(q);
        size_t byte_idx = i / 2;
        if (i % 2 == 0) {
            dst[byte_idx] = nibble << 4;
        } else {
            dst[byte_idx] |= (nibble & 0x0f);
        }
    }
}

inline uint8_t get_fp4_nibble(const std::vector<uint8_t> &packed, size_t idx) {
    uint8_t byte = packed[idx / 2];
    return (idx % 2 == 0) ? unpack_fp4_upper(byte) : unpack_fp4_lower(byte);
}

void mlp_cpu_fp32(luisa::span<const float> input,
                  luisa::span<const float> w1, luisa::span<const float> b1,
                  luisa::span<const float> w2, luisa::span<const float> b2,
                  luisa::span<float> output) {
    luisa::vector<float> hidden(kBatch * kHiddenDim);
    // layer 1: x @ w1.T + b1
    for (uint m = 0; m < kBatch; ++m) {
        for (uint n = 0; n < kHiddenDim; ++n) {
            float acc = b1[n];
            for (uint k = 0; k < kInDim; ++k) {
                acc += input[m * kInDim + k] * w1[n * kInDim + k];
            }
            hidden[m * kHiddenDim + n] = std::max(acc, 0.0f);
        }
    }
    // layer 2: hidden @ w2.T + b2
    for (uint m = 0; m < kBatch; ++m) {
        for (uint n = 0; n < kOutDim; ++n) {
            float acc = b2[n];
            for (uint k = 0; k < kHiddenDim; ++k) {
                acc += hidden[m * kHiddenDim + k] * w2[n * kHiddenDim + k];
            }
            output[m * kOutDim + n] = acc;
        }
    }
}

void mlp_cpu_fp4_simulated(luisa::span<const float> input,
                           luisa::span<const float> w1, luisa::span<const float> b1,
                           luisa::span<const float> w2, luisa::span<const float> b2,
                           luisa::span<float> output) {
    // Per-tensor FP4 E2M1 weight quantization
    std::vector<uint8_t> w1_q((w1.size() + 1) / 2);
    std::vector<uint8_t> w2_q((w2.size() + 1) / 2);
    std::vector<float> w_scales = {compute_fp4_scale(w1), compute_fp4_scale(w2)};
    quantize_to_fp4_bits(w1, w1_q, w_scales[0]);
    quantize_to_fp4_bits(w2, w2_q, w_scales[1]);

    luisa::vector<float> hidden(kBatch * kHiddenDim);

    // layer 1: dynamic activation quantization
    float x_scale = compute_fp4_scale(input);
    std::vector<uint8_t> x_q((input.size() + 1) / 2);
    quantize_to_fp4_bits(input, x_q, x_scale);

    for (uint m = 0; m < kBatch; ++m) {
        for (uint n = 0; n < kHiddenDim; ++n) {
            float acc = b1[n];
            for (uint k = 0; k < kInDim; ++k) {
                float x_dq = FP4E2M1::to_float(get_fp4_nibble(x_q, m * kInDim + k)) * x_scale;
                float w_dq = FP4E2M1::to_float(get_fp4_nibble(w1_q, n * kInDim + k)) * w_scales[0];
                acc += x_dq * w_dq;
            }
            hidden[m * kHiddenDim + n] = std::max(acc, 0.0f);
        }
    }

    // layer 2: dynamic activation quantization
    float h_scale = compute_fp4_scale(hidden);
    std::vector<uint8_t> h_q((hidden.size() + 1) / 2);
    quantize_to_fp4_bits(hidden, h_q, h_scale);

    for (uint m = 0; m < kBatch; ++m) {
        for (uint n = 0; n < kOutDim; ++n) {
            float acc = b2[n];
            for (uint k = 0; k < kHiddenDim; ++k) {
                float h_dq = FP4E2M1::to_float(get_fp4_nibble(h_q, m * kHiddenDim + k)) * h_scale;
                float w_dq = FP4E2M1::to_float(get_fp4_nibble(w2_q, n * kHiddenDim + k)) * w_scales[1];
                acc += h_dq * w_dq;
            }
            output[m * kOutDim + n] = acc;
        }
    }
}

int main(int argc, char *argv[]) {
    Context ctx(argv[0]);
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    auto rand_fill = [&](luisa::vector<float> &vec) {
        for (auto &v : vec) {
            v = dist(gen);
        }
    };

    luisa::vector<float> input(kBatch * kInDim);
    luisa::vector<float> w1(kHiddenDim * kInDim);
    luisa::vector<float> b1(kHiddenDim);
    luisa::vector<float> w2(kOutDim * kHiddenDim);
    luisa::vector<float> b2(kOutDim);
    rand_fill(input);
    rand_fill(w1);
    rand_fill(b1);
    rand_fill(w2);
    rand_fill(b2);

    // CPU references
    luisa::vector<float> cpu_fp32_out(kBatch * kOutDim);
    luisa::vector<float> cpu_fp4_out(kBatch * kOutDim);

    Clock cpu_clock;
    mlp_cpu_fp32(input, w1, b1, w2, b2, cpu_fp32_out);
    auto cpu_fp32_ms = cpu_clock.toc();
    cpu_clock.tic();
    mlp_cpu_fp4_simulated(input, w1, b1, w2, b2, cpu_fp4_out);
    auto cpu_fp4_ms = cpu_clock.toc();

    // Pre-quantize weights for GPU (packed nibbles, transposed to [N, K])
    std::vector<uint8_t> w1_q_bits((w1.size() + 1) / 2);
    std::vector<uint8_t> w2_q_bits((w2.size() + 1) / 2);
    std::vector<float> w_scales = {compute_fp4_scale(w1), compute_fp4_scale(w2)};
    quantize_to_fp4_bits(w1, w1_q_bits, w_scales[0]);
    quantize_to_fp4_bits(w2, w2_q_bits, w_scales[1]);

    // w1_q_bits and w2_q_bits are already in [N, K] row-major packed layout.

    // GPU buffers
    auto input_buffer = device.create_buffer<float>(input.size());
    auto hidden_buffer = device.create_buffer<float>(kBatch * kHiddenDim);
    auto output_buffer = device.create_buffer<float>(kBatch * kOutDim);
    auto w1_buffer = device.create_byte_buffer(w1_q_bits.size());
    auto b1_buffer = device.create_buffer<float>(b1.size());
    auto w2_buffer = device.create_byte_buffer(w2_q_bits.size());
    auto b2_buffer = device.create_buffer<float>(b2.size());

    // ReLU kernel
    auto relu_kernel = [&](BufferVar<float> buffer, Var<uint> count) noexcept {
        set_block_size(128, 1, 1);
        UInt idx = dispatch_id().x;
        $if (idx < count) {
            Float val = buffer.read(idx);
            buffer.write(idx, max(val, 0.0f));
        };
    };
    auto relu_shader = device.compile<1>(std::move(relu_kernel));

    // Warp-level quantized matmul kernel with packed FP4 weights
    auto matmul_kernel = [&](BufferVar<float> a_buffer, Var<ByteBuffer> w_buffer,
                             BufferVar<float> bias_buffer, BufferVar<float> c_buffer,
                             Var<uint> M, Var<uint> N, Var<uint> K,
                             Var<float> a_scale, Var<float> w_scale) noexcept {
        set_block_size(128, 1, 1);
        set_warp_size(kWarpSize);

        auto fp4_from = fp4e2m1_from_float();
        auto fp4_to = fp4e2m1_to_float();
        auto unpack = unpack_fp4();

        UInt row = dispatch_id().y;
        UInt col = dispatch_id().x / kWarpSize;
        UInt warp_local_id = warp_lane_id();

        UInt tile_count = (K + kWarpSize - 1) / kWarpSize;
        Float acc = 0.0f;

        for (auto t : dynamic_range(tile_count)) {
            UInt k = t * kWarpSize + warp_local_id;
            Float local_v = 0.0f;
            UInt warp_word = 0;
            // Each uint covers 8 packed FP4 values = 4 bytes
            $if(warp_local_id < ((K + 7u) / 8u)) {
                UInt byte_offset = col * (K / 2u) + t * (kWarpSize / 2u) + warp_local_id * 4u;
                warp_word = w_buffer.read<uint>(byte_offset);
            };
            $if (k < K) {
                Float a_val = a_buffer.read(row * K + k);
                Half a_scaled = (a_val / a_scale).cast<half>();
                UInt a_q = fp4_from(a_scaled);
                Half a_dq_half = fp4_to(a_q);
                Float a_dq = a_dq_half.cast<float>();

                // Unpack FP4 nibble from packed byte buffer
                UInt byte_offset = col * (K / 2u) + k / 2u;
                UInt rel_byte = (k / 2u) - t * (kWarpSize / 2u);
                UInt word_lane = rel_byte / 4u;
                UInt word = warp_read_lane(warp_word, word_lane);
                UInt shift = (byte_offset % 4u) * 8u;
                UInt packed_byte = (word >> shift) & 0xffu;
                UInt w_bits = unpack(packed_byte, k & 1u);
                Half w_half = fp4_to(w_bits);
                Float w_dq = w_half.cast<float>();

                local_v = a_dq * w_dq;
            };
            acc += warp_active_sum(local_v);
        }

        $if (warp_local_id == 0) {
            $if (col < N) {
                Float result = acc * a_scale * w_scale + bias_buffer.read(col);
                c_buffer.write(row * N + col, result);
            };
        };
    };
    auto matmul_shader = device.compile<2>(std::move(matmul_kernel));

    stream << input_buffer.copy_from(luisa::span{input})
           << w1_buffer.copy_from(w1_q_bits.data())
           << b1_buffer.copy_from(luisa::span{b1})
           << w2_buffer.copy_from(w2_q_bits.data())
           << b2_buffer.copy_from(luisa::span{b2})
           << synchronize();

    // Functional run to obtain dynamic activation scale and validate correctness
    float x_scale = compute_fp4_scale(input);
    stream << matmul_shader(input_buffer, w1_buffer, b1_buffer, hidden_buffer,
                            kBatch, kHiddenDim, kInDim, x_scale, w_scales[0])
                  .dispatch(kHiddenDim * kWarpSize, kBatch)
           << relu_shader(hidden_buffer, kBatch * kHiddenDim).dispatch(kBatch * kHiddenDim)
           << synchronize();

    luisa::vector<float> hidden_host(kBatch * kHiddenDim);
    stream << hidden_buffer.copy_to(luisa::span{hidden_host})
           << synchronize();
    float h_scale = compute_fp4_scale(hidden_host);

    stream << matmul_shader(hidden_buffer, w2_buffer, b2_buffer, output_buffer,
                            kBatch, kOutDim, kHiddenDim, h_scale, w_scales[1])
                  .dispatch(kOutDim * kWarpSize, kBatch)
           << synchronize();

    luisa::vector<float> gpu_out(kBatch * kOutDim);
    stream << output_buffer.copy_to(luisa::span{gpu_out})
           << synchronize();

    // Validation: max absolute error and MSE
    float max_diff_fp4 = 0.0f;
    float mse_fp4 = 0.0f;
    float max_diff_gpu = 0.0f;
    float mse_gpu = 0.0f;
    for (uint i = 0; i < kBatch * kOutDim; ++i) {
        float diff_fp4 = cpu_fp32_out[i] - cpu_fp4_out[i];
        float diff_gpu = cpu_fp32_out[i] - gpu_out[i];
        max_diff_fp4 = std::max(max_diff_fp4, std::abs(diff_fp4));
        mse_fp4 += diff_fp4 * diff_fp4;
        max_diff_gpu = std::max(max_diff_gpu, std::abs(diff_gpu));
        mse_gpu += diff_gpu * diff_gpu;
    }
    mse_fp4 /= static_cast<float>(kBatch * kOutDim);
    mse_gpu /= static_cast<float>(kBatch * kOutDim);

    // Weight compression ratio
    size_t fp32_weight_bytes = (w1.size() + w2.size()) * sizeof(float);
    size_t fp4_weight_bytes = w1_q_bits.size() + w2_q_bits.size();
    float compression_ratio = static_cast<float>(fp32_weight_bytes) / static_cast<float>(fp4_weight_bytes);

    // GPU Warmup (10 iterations)
    Clock warmup_clock;
    auto cmdlist = CommandList::create();
    for (int iter = 0; iter < 10; ++iter) {
        cmdlist << matmul_shader(input_buffer, w1_buffer, b1_buffer, hidden_buffer,
                                kBatch, kHiddenDim, kInDim, x_scale, w_scales[0])
                          .dispatch(kHiddenDim * kWarpSize, kBatch)
                 << relu_shader(hidden_buffer, kBatch * kHiddenDim).dispatch(kBatch * kHiddenDim)
                 << matmul_shader(hidden_buffer, w2_buffer, b2_buffer, output_buffer,
                                  kBatch, kOutDim, kHiddenDim, h_scale, w_scales[1])
                          .dispatch(kOutDim * kWarpSize, kBatch);
    }
    stream << cmdlist.commit() << synchronize();
    auto warmup_ms = warmup_clock.toc();

    // GPU Timed dispatch (128 iterations, batched via CommandList)
    Clock timed_clock;
    for (int iter = 0; iter < 128; ++iter) {
        cmdlist << matmul_shader(input_buffer, w1_buffer, b1_buffer, hidden_buffer,
                                 kBatch, kHiddenDim, kInDim, x_scale, w_scales[0])
                           .dispatch(kHiddenDim * kWarpSize, kBatch)
                 << relu_shader(hidden_buffer, kBatch * kHiddenDim).dispatch(kBatch * kHiddenDim)
                 << matmul_shader(hidden_buffer, w2_buffer, b2_buffer, output_buffer,
                                  kBatch, kOutDim, kHiddenDim, h_scale, w_scales[1])
                           .dispatch(kOutDim * kWarpSize, kBatch);
    }
    stream << cmdlist.commit() << synchronize();
    auto timed_ms = timed_clock.toc();

    LUISA_INFO("===== FP4 Quantization MLP Test =====");
    LUISA_INFO("CPU FP32 time: {:.4f} ms", cpu_fp32_ms);
    LUISA_INFO("CPU FP4  time: {:.4f} ms", cpu_fp4_ms);

    LUISA_INFO("GPU warmup time (10 iters): {:.4f} ms", warmup_ms);
    LUISA_INFO("GPU timed time (128 iters): {:.4f} ms", timed_ms);
    LUISA_INFO("GPU timed per-iter: {:.4f} ms", timed_ms / 128.0);
    LUISA_INFO("CPU FP4 vs FP32: max_diff={:.6f}, mse={:.6f}", max_diff_fp4, mse_fp4);
    LUISA_INFO("GPU vs CPU FP32: max_diff={:.6f}, mse={:.6f}", max_diff_gpu, mse_gpu);
    LUISA_INFO("Weight compression ratio (FP32/FP4): {:.2f}x", compression_ratio);

    float threshold = 2.0f;
    if (max_diff_gpu > threshold) {
        LUISA_ERROR("GPU result deviation too large: {}", max_diff_gpu);
    }
}
