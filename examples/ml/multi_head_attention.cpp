// Multi-Head Attention Example
// Implements MHA as 3 GPU kernel passes:
//   Pass 1: QK^T + Scale — each thread computes one S[i][j] = (1/√d_k) * Σ_d Q[i][d] * K[j][d]
//   Pass 2: Softmax     — each thread computes softmax for one row (max-sub, exp, normalize)
//   Pass 3: AV weighted  — each thread computes one O[i][d] = Σ_j A[i][j] * V[j][d]
//
// Results are verified against a CPU reference with tolerance 1e-4f.

#include <cmath>
#include <cstdlib>
#include <vector>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>", argv[0]);
        return 1;
    }

    LUISA_INFO("Multi-Head Attention Example");
    LUISA_INFO("Backend: {}", argv[1]);

    // ── Constants ───────────────────────────────────────────────────────
    constexpr uint batch     = 1u;
    constexpr uint num_heads = 2u;
    constexpr uint seq_len   = 8u;
    constexpr uint head_dim  = 16u;
    const float scale    = 1.0f / std::sqrt(static_cast<float>(head_dim)); // 0.25

    constexpr uint qkv_size  = batch * num_heads * seq_len * head_dim;   // 256
    constexpr uint scores_size = batch * num_heads * seq_len * seq_len;   // 128
    constexpr uint kWarpSize = 32u;                                    // NVIDIA warp size (CUDA/DX)
    constexpr uint kVecSize  = 2u;                                      // float2: 2 d-values per lane
    constexpr uint N_lanes_per_output = (head_dim + kVecSize - 1u) / kVecSize; // 8 lanes per output
    constexpr uint N_pass1   = scores_size;                               // 128 output elements
    constexpr uint N_pass2   = batch * num_heads * seq_len;               //  16 softmax rows
    constexpr uint N_pass3   = qkv_size;                                  // 256 AV output elements
    constexpr uint N_pass2_lanes = kWarpSize / 4u;                       // 4 rows per warp (8 lanes each)
    constexpr uint N_pass3_lanes = (seq_len + kVecSize - 1u) / kVecSize; // 4 lanes per AV output

    // ── Context & Device ────────────────────────────────────────────────
    Context ctx{argv[0]};
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream();

    // ── Host data initialization ────────────────────────────────────────
    luisa::vector<float> Q_host(qkv_size);
    luisa::vector<float> K_host(qkv_size);
    luisa::vector<float> V_host(qkv_size);

    // Fill with deterministic values
    for (uint i = 0u; i < qkv_size; ++i) {
        float fi = static_cast<float>(i);
        Q_host[i] = std::sin(fi * 0.13f + 0.5f) * 0.5f + 0.5f;
        K_host[i] = std::cos(fi * 0.17f + 1.0f) * 0.5f + 0.5f;
        V_host[i] = std::sin(fi * 0.11f + 2.0f) * 0.3f + 0.3f;
    }

    // ── Device buffers ──────────────────────────────────────────────────
    auto Q_buf = device.create_buffer<float>(qkv_size);
    auto K_buf = device.create_buffer<float>(qkv_size);
    auto V_buf = device.create_buffer<float>(qkv_size);
    auto S_buf = device.create_buffer<float>(scores_size);
    auto A_buf = device.create_buffer<float>(scores_size);
    auto O_buf = device.create_buffer<float>(qkv_size);

    // Upload input data
    stream << Q_buf.copy_from(luisa::span{Q_host})
           << K_buf.copy_from(luisa::span{K_host})
           << V_buf.copy_from(luisa::span{V_host})
           << synchronize();

    // ── Pass 1: QK^T + Scale (warp reduce, 4 outputs per warp) ────────
    // Each warp (32 lanes) computes 4 output elements (8 lanes each).
    // Each lane packs 2 d-values into float2 → 8×2 = 16 = head_dim.
    // warp_prefix_sum + warp_read_lane extract per-group totals.
    Kernel1D qk_dot = [&](BufferFloat Q, BufferFloat K, BufferFloat S) noexcept {
        set_block_size(256u, 1u, 1u);
        set_warp_size(kWarpSize);

        // 8 lanes per output, 4 outputs per warp
        Var idx = dispatch_id().x;
        Var lane = warp_lane_id();
        Var group_id = lane / 8u;        // 0..3: which output within this warp
        Var group_lane = lane % 8u;       // 0..7: lane within the group
        Var output_idx = (idx / kWarpSize) * 4u + group_id;

        Var head = output_idx / (seq_len * seq_len);
        Var rem  = output_idx % (seq_len * seq_len);
        Var i    = rem / seq_len;
        Var j    = rem % seq_len;

        // 2 d-values per lane, guarded by ite (no warp divergence)
        Var d0 = group_lane * kVecSize;
        Var d1 = group_lane * kVecSize + 1u;
        Var base = head * (seq_len * head_dim);
        Float2 prod = make_float2(
            ite(d0 < head_dim, Q.read(base + i * head_dim + d0) * K.read(base + j * head_dim + d0), 0.0f),
            ite(d1 < head_dim, Q.read(base + i * head_dim + d1) * K.read(base + j * head_dim + d1), 0.0f)
        );

        // warp_prefix_sum is global across all 32 lanes.
        // Each group's total = inclusive[last] - inclusive[prev_group_last]
        // (for group 0, subtract 0 since there is no previous group).
        Float2 prefix = warp_prefix_sum(prod);
        Float2 inclusive = prefix + prod;
        Var last_lane = group_id * 8u + 7u;
        Float2 incl_last = warp_read_lane(inclusive, last_lane);
        // prev_last = last_lane - 8u wraps for group 0; ite discards the read
        Float2 prev_incl = warp_read_lane(inclusive, ite(group_id == 0u, 0u, last_lane - 8u));
        Float2 group_total = incl_last - ite(group_id == 0u, make_float2(0.0f), prev_incl);

        $if (group_lane == 0) {
            S.write(output_idx, (group_total.x + group_total.y) * scale);
        };
    };

    LUISA_INFO("Compiling Pass 1 (QK^T + Scale) ...");
    auto qk_shader = device.compile(qk_dot);

    // ── Pass 2: Softmax (warp reduce, 4 rows per warp) ──────────────────
    // Each warp (32 lanes) computes 4 rows (8 lanes each = seq_len).
    // Butterfly reduction for per-group max; prefix-sum for per-group sum.
    Kernel1D softmax_kernel = [&](BufferFloat S, BufferFloat A) noexcept {
        set_block_size(256u, 1u, 1u);
        set_warp_size(kWarpSize);

        Var idx = dispatch_id().x;
        Var lane = warp_lane_id();
        Var group_id = lane / 8u;        // 0..3: which row within this warp
        Var group_lane = lane % 8u;       // 0..7 = j
        Var output_idx = (idx / kWarpSize) * 4u + group_id;

        Var head = output_idx / seq_len;
        Var i    = output_idx % seq_len;
        Var base = head * (seq_len * seq_len) + i * seq_len;
        Var j    = group_lane;           // always in [0, seq_len)

        // Phase 1: per-group max via butterfly reduction (3 steps, no branches)
        Float m = S.read(base + j);
        m = max(m, warp_read_lane(m, lane ^ 4u));
        m = max(m, warp_read_lane(m, lane ^ 2u));
        m = max(m, warp_read_lane(m, lane ^ 1u));

        // Phase 2: exp + sum — per-group sum via prefix-sum extraction
        Float e = exp(S.read(base + j) - m);
        A.write(base + j, e);
        Float prefix = warp_prefix_sum(e);
        Float inclusive = prefix + e;
        Var last_lane = group_id * 8u + 7u;
        Float incl_last = warp_read_lane(inclusive, last_lane);
        Float prev_incl = warp_read_lane(inclusive, ite(group_id == 0u, 0u, last_lane - 8u));
        Float s = incl_last - ite(group_id == 0u, 0.0f, prev_incl);

        // Phase 3: normalize in-place
        A.write(base + j, A.read(base + j) / s);
    };

    LUISA_INFO("Compiling Pass 2 (Softmax) ...");
    auto softmax_shader = device.compile(softmax_kernel);

    // ── Pass 3: AV Weighted (warp reduce, 8 outputs per warp) ──────────
    // Each warp (32 lanes) computes 8 output elements (4 lanes each).
    // Each lane packs 2 j-values into float2 → 4×2 = 8 = seq_len.
    Kernel1D av_weighted = [&](BufferFloat A, BufferFloat V, BufferFloat O) noexcept {
        set_block_size(256u, 1u, 1u);
        set_warp_size(kWarpSize);

        // 4 lanes per output, 8 outputs per warp
        Var idx = dispatch_id().x;
        Var lane = warp_lane_id();
        Var group_id = lane / 4u;        // 0..7: which output within this warp
        Var group_lane = lane % 4u;       // 0..3: lane within the group
        Var output_idx = (idx / kWarpSize) * 8u + group_id;

        Var head = output_idx / (seq_len * head_dim);
        Var rem  = output_idx % (seq_len * head_dim);
        Var i    = rem / head_dim;
        Var d    = rem % head_dim;

        // 2 j-values per lane, guarded by ite
        Var j0 = group_lane * kVecSize;
        Var j1 = group_lane * kVecSize + 1u;
        Var ai_base = head * (seq_len * seq_len) + i * seq_len;
        Var vi_head = head * (seq_len * head_dim);
        Float2 prod = make_float2(
            ite(j0 < seq_len, A.read(ai_base + j0) * V.read(vi_head + j0 * head_dim + d), 0.0f),
            ite(j1 < seq_len, A.read(ai_base + j1) * V.read(vi_head + j1 * head_dim + d), 0.0f)
        );

        // Extract per-group total via prefix sum
        Float2 prefix = warp_prefix_sum(prod);
        Float2 inclusive = prefix + prod;
        Var last_lane = group_id * 4u + 3u;
        Float2 incl_last = warp_read_lane(inclusive, last_lane);
        Float2 prev_incl = warp_read_lane(inclusive, ite(group_id == 0u, 0u, last_lane - 4u));
        Float2 group_total = incl_last - ite(group_id == 0u, make_float2(0.0f), prev_incl);

        $if (group_lane == 0) {
            O.write(output_idx, group_total.x + group_total.y);
        };
    };

    LUISA_INFO("Compiling Pass 3 (AV Weighted Sum) ...");
    auto av_shader = device.compile(av_weighted);

    // ── Dispatch all three passes ──────────────────────────────────────
    LUISA_INFO("Dispatching GPU kernels ...");
    stream << qk_shader(Q_buf, K_buf, S_buf).dispatch(N_pass1 * N_lanes_per_output)
           << softmax_shader(S_buf, A_buf).dispatch(N_pass2 * N_pass2_lanes)
           << av_shader(A_buf, V_buf, O_buf).dispatch(N_pass3 * N_pass3_lanes)
           << synchronize();

    // ── Download results ───────────────────────────────────────────────
    luisa::vector<float> O_gpu(qkv_size);
    stream << O_buf.copy_to(luisa::span{O_gpu}) << synchronize();

    // ── CPU Reference ──────────────────────────────────────────────────
    LUISA_INFO("Running CPU reference ...");

    // Allocate CPU intermediates
    std::vector<float> S_cpu(scores_size, 0.0f);
    std::vector<float> A_cpu(scores_size, 0.0f);
    std::vector<float> O_cpu(qkv_size, 0.0f);

    // Pass 1: QK^T + Scale
    for (uint h = 0u; h < num_heads; ++h) {
        for (uint i = 0u; i < seq_len; ++i) {
            for (uint j = 0u; j < seq_len; ++j) {
                float sum = 0.0f;
                for (uint d = 0u; d < head_dim; ++d) {
                    uint qi = h * seq_len * head_dim + i * head_dim + d;
                    uint ki = h * seq_len * head_dim + j * head_dim + d;
                    sum += Q_host[qi] * K_host[ki];
                }
                uint si = h * seq_len * seq_len + i * seq_len + j;
                S_cpu[si] = sum * scale;
            }
        }
    }

    // Pass 2: Softmax (row-wise)
    for (uint h = 0u; h < num_heads; ++h) {
        for (uint i = 0u; i < seq_len; ++i) {
            uint base = h * seq_len * seq_len + i * seq_len;
            // Find max
            float m = -1e30f;
            for (uint j = 0u; j < seq_len; ++j) {
                float val = S_cpu[base + j];
                if (val > m) { m = val; }
            }
            // Exp & sum
            float s = 0.0f;
            for (uint j = 0u; j < seq_len; ++j) {
                float e = std::exp(S_cpu[base + j] - m);
                A_cpu[base + j] = e;
                s += e;
            }
            // Normalize
            for (uint j = 0u; j < seq_len; ++j) {
                A_cpu[base + j] /= s;
            }
        }
    }

    // Pass 3: AV Weighted Sum
    for (uint h = 0u; h < num_heads; ++h) {
        for (uint i = 0u; i < seq_len; ++i) {
            for (uint d = 0u; d < head_dim; ++d) {
                float sum = 0.0f;
                for (uint j = 0u; j < seq_len; ++j) {
                    uint ai = h * seq_len * seq_len + i * seq_len + j;
                    uint vi = h * seq_len * head_dim + j * head_dim + d;
                    sum += A_cpu[ai] * V_host[vi];
                }
                uint oi = h * seq_len * head_dim + i * head_dim + d;
                O_cpu[oi] = sum;
            }
        }
    }

    // ── Verification ────────────────────────────────────────────────────
    float max_diff = 0.0f;
    uint  max_idx  = 0u;
    for (uint i = 0u; i < qkv_size; ++i) {
        float diff = std::abs(O_gpu[i] - O_cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx  = i;
        }
    }

    constexpr float tolerance = 1e-4f;
    bool passed = max_diff <= tolerance;

    LUISA_INFO("Verification: {}", passed ? "PASSED" : "FAILED");
    LUISA_INFO("  Max error: {} at index {} (GPU={}, CPU={})",
               max_diff, max_idx, O_gpu[max_idx], O_cpu[max_idx]);

    if (!passed) {
        // Print first few mismatches for debugging
        uint printed = 0u;
        for (uint i = 0u; i < qkv_size && printed < 5u; ++i) {
            float diff = std::abs(O_gpu[i] - O_cpu[i]);
            if (diff > tolerance) {
                LUISA_INFO("  Mismatch[{}]: GPU={}, CPU={}, diff={}", i, O_gpu[i], O_cpu[i], diff);
                ++printed;
            }
        }
    }

    return passed ? 0 : 1;
}
