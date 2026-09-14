// Handwritten attention probes, NOT TileIR/XIR/SIMD compiler output.
// Both entries implement complete contiguous FP32 bottom-right causal GQA:
// Q[B,H,Q,D], K[B,KH,K,D], V[B,KH,K,DV] -> O[B,H,Q,DV]. No bias/dropout.
// Compile positive ATTN_B/H/KH/Q/K/D/DV, Q <= K and H % KH == 0. The driver
// admits finite, nonaliasing inputs and validates every output against dense
// FP64 with atol=rtol=5e-5; bitwise equivalence is NOT the numerical contract.
//
// online_neon_rows retains KV-block-16 online (max,sum,acc) softmax. Its QK
// dot products deliberately use four-lane unordered accumulation, so this is
// not a strict-MMA lowering candidate. PV vectorizes independent output axes.
// dense_accelerate instead materializes Q*K scores per head and uses classic
// LP64 FP32 BLAS. Its reduction order, possible internal FMA and memory schedule
// differ from both the online probe and the captured compiler implementation.
// QK uses BLAS alpha=1, followed by an explicit FP32 score*scale boundary.
//
// The existing ABI-0 function signature is reused, but each call is one whole
// operation, not a packet: the driver must request exactly one logical block.
// Arguments are four SIMDHostBufferView descriptors in Q,K,V,O order; the
// return-lane pointer is unused and launch metadata is never mutated. The
// caller supplies attention_probe_workspace_bytes() aligned scratch through
// launch.private_workspace. No operator copies or caller allocations occur
// here; BLAS-internal allocations, if any, remain inside the entry's timing.
// Heads and queries are serial, with no head interleaving. The online probe
// skips fully causal-masked KV blocks; dense BLAS computes the complete Q*K.
//
// Build for Apple arm64/macOS >= 15 with -ffp-contract=off and no fast math.
// Configure and query BLAS single-threading outside the timer, on the same
// calling thread. This TLS setting is not changed by either timed entry.

#include "backends/simd/llvm/llvm_schedule_codegen.h"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "The native attention probes require Apple arm64."
#endif
#if !defined(ATTN_B) || !defined(ATTN_H) || !defined(ATTN_KH) || !defined(ATTN_Q) || !defined(ATTN_K) || !defined(ATTN_D) || !defined(ATTN_DV)
#error "Define all seven attention dimensions explicitly."
#endif
#if defined(ACCELERATE_NEW_LAPACK) || defined(ACCELERATE_LAPACK_ILP64)
#error "This benchmark deliberately uses the classic LP64 CBLAS interface."
#endif
#if defined(__FAST_MATH__)
#error "The native attention probes do not admit fast-math compilation."
#endif

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

using luisa::compute::simd::SIMDHostBufferView;
using luisa::compute::simd::SIMDPacketLaunchConfig;

namespace {

static_assert(ATTN_B > 0 && ATTN_H > 0 && ATTN_KH > 0 && ATTN_Q > 0 && ATTN_K > 0 && ATTN_D > 0 && ATTN_DV > 0);
constexpr uint64_t kBatch = ATTN_B, kHeads = ATTN_H, kKeyHeads = ATTN_KH;
constexpr uint64_t kQueries = ATTN_Q, kKeys = ATTN_K, kDimension = ATTN_D, kValueDimension = ATTN_DV;
constexpr uint64_t kKeyBlock = 16u, kMaxElements = uint64_t{1u} << 26u;
static_assert(kHeads % kKeyHeads == 0u && kQueries <= kKeys);

[[nodiscard]] constexpr bool bounded_elements(std::initializer_list<uint64_t> extents) noexcept {
    auto product = uint64_t{1u};
    for (auto extent : extents) {
        if (extent == 0u || extent > kMaxElements / product) { return false; }
        product *= extent;
    }
    return true;
}

// Check before forming products, including the LP64 leading dimensions.
static_assert(bounded_elements({kBatch, kHeads, kQueries, kDimension}));
static_assert(bounded_elements({kBatch, kKeyHeads, kKeys, kDimension}));
static_assert(bounded_elements({kBatch, kKeyHeads, kKeys, kValueDimension}));
static_assert(bounded_elements({kBatch, kHeads, kQueries, kValueDimension}));
static_assert(bounded_elements({kQueries, kKeys}));
static_assert(kMaxElements <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
constexpr uint64_t kWorkspaceBytes = std::max(kQueries * kKeys, kKeyBlock) * sizeof(float);

[[nodiscard]] float dot_neon(const float *query, const float *key) noexcept {
    auto partial = vdupq_n_f32(0.0f);
    auto d = uint64_t{0u};
    for (; d + 4u <= kDimension; d += 4u) {
        auto product = vmulq_f32(vld1q_f32(query + d), vld1q_f32(key + d));
        partial = vaddq_f32(partial, product);
    }
    // This explicit four-way tree is part of the probe's relaxed QK contract.
    auto sum = vaddvq_f32(partial);
    for (; d < kDimension; d++) {
        auto product = query[d] * key[d];
        sum = sum + product;
    }
    return sum;
}

void online_row(const float *query, const float *keys, const float *values,
                float *output, float *weights, uint64_t query_index, float scale) noexcept {
    std::fill_n(output, kValueDimension, 0.0f);
    auto maximum = -std::numeric_limits<float>::infinity();
    auto sum = 0.0f;
    auto visible_keys = kKeys - kQueries + query_index + 1u;
    for (auto first = uint64_t{0u}; first < visible_keys; first += kKeyBlock) {
        auto count = std::min(kKeyBlock, visible_keys - first);
        auto next_maximum = maximum;
        for (auto j = uint64_t{0u}; j < count; j++) {
            auto score = dot_neon(query, keys + (first + j) * kDimension) * scale;
            weights[j] = score;
            next_maximum = std::max(next_maximum, score);
        }
        auto previous_scale = std::exp(maximum - next_maximum);
        auto block_sum = 0.0f;
        for (auto j = uint64_t{0u}; j < count; j++) {
            weights[j] = std::exp(weights[j] - next_maximum);
            block_sum = block_sum + weights[j];
        }
        sum = sum * previous_scale + block_sum;
        maximum = next_maximum;
        auto d = uint64_t{0u};
        for (; d + 4u <= kValueDimension; d += 4u) {
            auto accumulator = vmulq_n_f32(vld1q_f32(output + d), previous_scale);
            for (auto j = uint64_t{0u}; j < count; j++) {
                auto product = vmulq_n_f32(vld1q_f32(values + (first + j) * kValueDimension + d), weights[j]);
                accumulator = vaddq_f32(accumulator, product);
            }
            vst1q_f32(output + d, accumulator);
        }
        for (; d < kValueDimension; d++) {
            auto accumulator = output[d] * previous_scale;
            for (auto j = uint64_t{0u}; j < count; j++) {
                auto product = values[(first + j) * kValueDimension + d] * weights[j];
                accumulator = accumulator + product;
            }
            output[d] = accumulator;
        }
    }
    auto d = uint64_t{0u};
    for (; d + 4u <= kValueDimension; d += 4u) {
        vst1q_f32(output + d, vdivq_f32(vld1q_f32(output + d), vdupq_n_f32(sum)));
    }
    for (; d < kValueDimension; d++) { output[d] = output[d] / sum; }
}

void dense_softmax(float *scores, float scale) noexcept {
    for (auto q = uint64_t{0u}; q < kQueries; q++) {
        auto row = scores + q * kKeys;
        auto visible_keys = kKeys - kQueries + q + 1u;
        auto maximum = -std::numeric_limits<float>::infinity();
        for (auto k = uint64_t{0u}; k < visible_keys; k++) {
            row[k] = row[k] * scale;
            maximum = std::max(maximum, row[k]);
        }
        auto sum = 0.0f;
        for (auto k = uint64_t{0u}; k < visible_keys; k++) {
            row[k] = std::exp(row[k] - maximum);
            sum = sum + row[k];
        }
        for (auto k = uint64_t{0u}; k < visible_keys; k++) { row[k] = row[k] / sum; }
        std::fill(row + visible_keys, row + kKeys, 0.0f);
    }
}

}// namespace

extern "C" __attribute__((visibility("default"))) uint64_t attention_probe_workspace_bytes() noexcept { return kWorkspaceBytes; }
extern "C" __attribute__((visibility("default"))) uint32_t attention_probe_contract_version() noexcept { return 1u; }

extern "C" __attribute__((visibility("default"))) int32_t attention_probe_set_single_threaded() noexcept {
    if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) != 0) { return -1; }
    return BLASGetThreading() == BLAS_THREADING_SINGLE_THREADED ? 0 : -1;
}

extern "C" __attribute__((visibility("default"))) uint32_t attention_probe_threading() noexcept {
    return static_cast<uint32_t>(BLASGetThreading());
}

extern "C" __attribute__((visibility("default"))) void attention_online_neon(
    const void *arguments, void *, SIMDPacketLaunchConfig *launch, uint32_t) noexcept {
    auto views = static_cast<const SIMDHostBufferView *>(arguments);
    auto query = static_cast<const float *>(views[0].data);
    auto keys = static_cast<const float *>(views[1].data);
    auto values = static_cast<const float *>(views[2].data);
    auto output = static_cast<float *>(views[3].data);
    auto weights = static_cast<float *>(launch->private_workspace);
    auto scale = 1.0f / std::sqrt(static_cast<float>(kDimension));
    for (auto b = uint64_t{0u}; b < kBatch; b++) {
        for (auto h = uint64_t{0u}; h < kHeads; h++) {
            auto key_head = h / (kHeads / kKeyHeads);
            auto query_head = query + (b * kHeads + h) * kQueries * kDimension;
            auto key_base = keys + (b * kKeyHeads + key_head) * kKeys * kDimension;
            auto value_base = values + (b * kKeyHeads + key_head) * kKeys * kValueDimension;
            auto output_head = output + (b * kHeads + h) * kQueries * kValueDimension;
            for (auto q = uint64_t{0u}; q < kQueries; q++) {
                online_row(query_head + q * kDimension, key_base, value_base,
                           output_head + q * kValueDimension, weights, q, scale);
            }
        }
    }
}

extern "C" __attribute__((visibility("default"))) void attention_dense_accelerate(
    const void *arguments, void *, SIMDPacketLaunchConfig *launch, uint32_t) noexcept {
    auto views = static_cast<const SIMDHostBufferView *>(arguments);
    auto query = static_cast<const float *>(views[0].data);
    auto keys = static_cast<const float *>(views[1].data);
    auto values = static_cast<const float *>(views[2].data);
    auto output = static_cast<float *>(views[3].data);
    auto scores = static_cast<float *>(launch->private_workspace);
    auto scale = 1.0f / std::sqrt(static_cast<float>(kDimension));
    constexpr auto q = static_cast<int32_t>(kQueries), k = static_cast<int32_t>(kKeys);
    constexpr auto d = static_cast<int32_t>(kDimension), dv = static_cast<int32_t>(kValueDimension);
    for (auto b = uint64_t{0u}; b < kBatch; b++) {
        for (auto h = uint64_t{0u}; h < kHeads; h++) {
            auto key_head = h / (kHeads / kKeyHeads);
            auto query_head = query + (b * kHeads + h) * kQueries * kDimension;
            auto key_base = keys + (b * kKeyHeads + key_head) * kKeys * kDimension;
            auto value_base = values + (b * kKeyHeads + key_head) * kKeys * kValueDimension;
            auto output_head = output + (b * kHeads + h) * kQueries * kValueDimension;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            if constexpr (kQueries == 1u) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, k, d, 1.0f, key_base, d, query_head, 1, 0.0f, scores, 1);
            } else {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q, k, d, 1.0f, query_head, d, key_base, d, 0.0f, scores, k);
            }
            dense_softmax(scores, scale);
            if constexpr (kQueries == 1u) {
                cblas_sgemv(CblasRowMajor, CblasTrans, k, dv, 1.0f, value_base, dv, scores, 1, 0.0f, output_head, 1);
            } else {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, q, dv, k, 1.0f, scores, k, value_base, dv, 0.0f, output_head, dv);
            }
#pragma clang diagnostic pop
        }
    }
}
