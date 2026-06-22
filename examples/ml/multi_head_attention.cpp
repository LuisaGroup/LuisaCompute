// Multi-Head Latent Attention (MLA) Example
//
// This example implements two attention paths selected by `use_mla`:
//
//   use_mla = false  -- classic Multi-Head Attention (MHA) reference.
//                       Three GPU passes: QK^T + scale, softmax, AV.
//
//   use_mla = true   -- Multi-Head Latent Attention reference.
//                       Demonstrates the DeepSeek-V2/V3 techniques:
//                         * Low-rank joint compression of K/V via W_DKV.
//                         * Decoupled RoPE: content queries + positional queries,
//                           separate W_KR decoupled keys.
//                         * Matrix absorption for the content score:
//                           q^T (W_UK c_t^KV) = (W_UK^T q)^T c_t^KV,
//                           so the full per-head content key tensor is never
//                           materialised during inference.
//                         * Reduced KV cache: cache only c_t^KV (latent_dim) and
//                           k_t^R (num_heads * rope_dim) per token.
//
// Both paths are verified against a matching CPU reference with tolerance 1e-4f.

#include <cmath>
#include <cstdlib>
#include <utility>

#include <luisa/core/clock.h>
#include <luisa/core/fiber.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

// -- Constants ---------------------------------------------------------
// Kept at namespace scope so they remain constant expressions inside DSL
// lambdas (they are not captured, so they can be used as template arguments
// such as in $array<float, latent_dim>).
constexpr uint batch     = 16u;
constexpr uint num_heads = 8u;
constexpr uint seq_len   = 256u;
constexpr uint head_dim  = 64u;

// MLA-specific dimensions.
constexpr uint hidden_dim  = 512u;// input hidden size h_t
constexpr uint latent_dim  = 64u;// compressed KV dim d_c
constexpr uint rope_dim    = 16u; // decoupled RoPE dim d_h^R
constexpr uint content_dim = head_dim - rope_dim;
constexpr float rope_theta = 10000.0f;

static_assert(content_dim > 0u, "head_dim must be larger than rope_dim");
static_assert(rope_dim % 2u == 0u, "rope_dim must be even for pair-wise RoPE");

int main(int argc, char *argv[]) {

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [use_mla=1]", argv[0]);
        return 1;
    }

    LUISA_INFO("Multi-Head Attention Example");
    LUISA_INFO("Backend: {}", argv[1]);

    // -- Attention path switch ---------------------------------------------
    bool use_mla = true;
    if (argc >= 3) {
        use_mla = (std::atoi(argv[2]) != 0);
    }
    LUISA_INFO("Path: {}", use_mla ? "MLA" : "MHA");


    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    constexpr uint hidden_size  = batch * seq_len * hidden_dim;
    constexpr uint latent_size  = batch * seq_len * latent_dim;
    constexpr uint rope_size    = batch * num_heads * seq_len * rope_dim;
    constexpr uint qkv_size     = batch * num_heads * seq_len * head_dim;
    constexpr uint scores_size  = batch * num_heads * seq_len * seq_len;

    // -- Host-side index helpers (CPU reference) ---------------------------
    constexpr auto qkv_index = [](uint b, uint h, uint i, uint d) constexpr noexcept {
        return ((b * num_heads + h) * seq_len + i) * head_dim + d;
    };
    constexpr auto score_index = [](uint b, uint h, uint i, uint j) constexpr noexcept {
        return ((b * num_heads + h) * seq_len + i) * seq_len + j;
    };
    constexpr auto hidden_index = [](uint b, uint i, uint d) constexpr noexcept {
        return (b * seq_len + i) * hidden_dim + d;
    };
    constexpr auto latent_index = [](uint b, uint i, uint d) constexpr noexcept {
        return (b * seq_len + i) * latent_dim + d;
    };
    constexpr auto rope_index = [](uint b, uint h, uint i, uint d) constexpr noexcept {
        return ((b * num_heads + h) * seq_len + i) * rope_dim + d;
    };

    // -- Context & Device --------------------------------------------------
    Context ctx{argv[0]};
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream();

    // -- Host data initialization ------------------------------------------
    // MHA inputs (kept for the baseline path).
    luisa::vector<float> Q_host(qkv_size);
    luisa::vector<float> K_host(qkv_size);
    luisa::vector<float> V_host(qkv_size);

    for (uint i = 0u; i < qkv_size; ++i) {
        auto fi = static_cast<float>(i);
        Q_host[i] = std::sin(fi * 0.13f + 0.5f) * 0.5f + 0.5f;
        K_host[i] = std::cos(fi * 0.17f + 1.0f) * 0.5f + 0.5f;
        V_host[i] = std::sin(fi * 0.11f + 2.0f) * 0.3f + 0.3f;
    }

    // MLA inputs and weights.
    luisa::vector<float> H_host(hidden_size);
    luisa::vector<float> Wq_host(num_heads * head_dim * hidden_dim);
    luisa::vector<float> Wdkv_host(latent_dim * hidden_dim);
    luisa::vector<float> Wuk_host(num_heads * content_dim * latent_dim);
    luisa::vector<float> Wuv_host(num_heads * head_dim * latent_dim);
    luisa::vector<float> Wkr_host(num_heads * rope_dim * hidden_dim);

    for (uint i = 0u; i < hidden_size; ++i) {
        H_host[i] = std::sin(static_cast<float>(i) * 0.05f + 0.3f) * 0.4f + 0.4f;
    }
    for (uint i = 0u; i < Wq_host.size(); ++i) {
        Wq_host[i] = std::sin(static_cast<float>(i) * 0.07f) * 0.01f;
    }
    for (uint i = 0u; i < Wdkv_host.size(); ++i) {
        Wdkv_host[i] = std::sin(static_cast<float>(i) * 0.09f + 0.1f) * 0.02f;
    }
    for (uint i = 0u; i < Wuk_host.size(); ++i) {
        Wuk_host[i] = std::sin(static_cast<float>(i) * 0.11f + 0.2f) * 0.02f;
    }
    for (uint i = 0u; i < Wuv_host.size(); ++i) {
        Wuv_host[i] = std::sin(static_cast<float>(i) * 0.13f + 0.3f) * 0.02f;
    }
    for (uint i = 0u; i < Wkr_host.size(); ++i) {
        Wkr_host[i] = std::sin(static_cast<float>(i) * 0.15f + 0.4f) * 0.02f;
    }

    // -- Device buffers ----------------------------------------------------
    auto H_buf     = device.create_buffer<float>(hidden_size);
    auto cKV_buf   = device.create_buffer<float>(latent_size);
    auto Krope_buf = device.create_buffer<float>(rope_size);
    auto Q_buf     = device.create_buffer<float>(qkv_size);
    auto K_buf     = device.create_buffer<float>(qkv_size);
    auto V_buf     = device.create_buffer<float>(qkv_size);
    auto O_buf     = device.create_buffer<float>(qkv_size);

    auto Wq_buf   = device.create_buffer<float>(Wq_host.size());
    auto Wdkv_buf = device.create_buffer<float>(Wdkv_host.size());
    auto Wuk_buf  = device.create_buffer<float>(Wuk_host.size());
    auto Wuv_buf  = device.create_buffer<float>(Wuv_host.size());
    auto Wkr_buf  = device.create_buffer<float>(Wkr_host.size());

    // Upload everything in one batch.
    CommandList upload = CommandList::create();
    upload << Q_buf.copy_from(luisa::span{Q_host})
           << K_buf.copy_from(luisa::span{K_host})
           << V_buf.copy_from(luisa::span{V_host})
           << H_buf.copy_from(luisa::span{H_host})
           << Wq_buf.copy_from(luisa::span{Wq_host})
           << Wdkv_buf.copy_from(luisa::span{Wdkv_host})
           << Wuk_buf.copy_from(luisa::span{Wuk_host})
           << Wuv_buf.copy_from(luisa::span{Wuv_host})
           << Wkr_buf.copy_from(luisa::span{Wkr_host});
    stream << upload.commit() << synchronize();

    // -- Reusable RoPE rotation helper (DSL) -------------------------------
    // Precompute per-pair inverse frequencies (host-side constexpr) to avoid
    // pow() in every kernel call. For realistic seq_len >= 2048 this matters.
    const std::array<float, rope_dim / 2u> kInvFreqs = []() {
        std::array<float, rope_dim / 2u> f{};
        for (uint p = 0u; p < rope_dim / 2u; ++p) {
            f[p] = 1.0f / std::pow(rope_theta,
                (2.0f * static_cast<float>(p)) / static_cast<float>(rope_dim));
        }
        return f;
    }();

    Callable apply_rope_pair = [&](Float x0, Float x1, const UInt &pair, const UInt &pos) noexcept {
        $constant freqs = kInvFreqs;
        Float freq = freqs.read(pair);
        Float angle = pos.cast<float>() * freq;
        Float c = cos(angle);
        Float s = sin(angle);
        return make_float2(x0 * c - x1 * s, x0 * s + x1 * c);
    };

    ShaderOption opt{.enable_debug_info = false};
    Clock compile_clock;

    // -- Compile and dispatch the selected attention path ------------------
    // Both MLA and MHA kernels are always defined; only the selected path
    // is compiled to GPU code and dispatched.

        // -- Optimized MLA GPU kernels -----------------------------------
        // Phase 1+2: float4 vectorization + kernel fusion + online softmax.

        // Project queries: Q[b,h,i,d] = Wq[h,d,:] @ H[b,i,:] (shared-memory tiled).
        // Each block handles one (b,i) token; 32 threads cooperatively load H
        // into shared memory; the first num_heads threads compute their assigned head.
        // Grid: (batch * seq_len). Block: (32, 1).
        Kernel1D project_q_kernel = [&](BufferFloat H, BufferFloat Q, BufferFloat Wq) noexcept {
            constexpr uint kBlockSize = 32u;
            set_block_size(kBlockSize, 1u, 1u);
            set_name("mla_project_q");

            Shared<float> H_shared{hidden_dim};

            auto qkv_idx = [&](auto b_, auto h_, auto i_, auto d_) {
                return ((b_ * num_heads + h_) * seq_len + i_) * head_dim + d_;
            };

            Var idx = dispatch_id().x;
            Var b = idx / seq_len;
            Var i = idx % seq_len;
            Var tx = thread_x();
            Var hi = (b * seq_len + i) * hidden_dim;

            // -- Cooperative load of H[b,i,:] into shared memory (all 32 threads) --
            $for (e, hidden_dim / kBlockSize) {
                Var e_idx = tx * (hidden_dim / kBlockSize) + e;
                H_shared[e_idx] = H.read(hi + e_idx);
            };
            sync_block();

            // -- First num_heads threads compute Q for their assigned head --
            $if (tx < num_heads) {
                Var h = tx;
                Var h_off = h * head_dim * hidden_dim;

                $for (d, head_dim) {
                    Var acc = def(0.0f);
                    $for (e, hidden_dim / 4u) {
                        Var e4 = e * 4u;
                        Var w_base = h_off + d * hidden_dim + e4;
                        acc += Wq.read(w_base) * H_shared[e4]
                             + Wq.read(w_base + 1u) * H_shared[e4 + 1u]
                             + Wq.read(w_base + 2u) * H_shared[e4 + 2u]
                             + Wq.read(w_base + 3u) * H_shared[e4 + 3u];
                    };
                    Q.write(qkv_idx(b, h, i, d), acc);
                };

                // Apply RoPE to the positional slice.
                $for (r, rope_dim / 2u) {
                    Var d0 = content_dim + r * 2u;
                    Var d1 = d0 + 1u;
                    Float x0 = Q.read(qkv_idx(b, h, i, d0));
                    Float x1 = Q.read(qkv_idx(b, h, i, d1));
                    Float2 rot = apply_rope_pair(x0, x1, r, i);
                    Q.write(qkv_idx(b, h, i, d0), rot.x);
                    Q.write(qkv_idx(b, h, i, d1), rot.y);
                };
            };
        };

        // Fused project KV: cKV + Krope from a single H read per token.
        // Each thread handles one (b,i) token, computing cKV (latent_dim=16)
        // and Krope (num_heads×rope_dim=64) in one pass.
        Kernel1D project_kv_kernel = [&](BufferFloat H, BufferFloat cKV,
                                         BufferFloat Krope, BufferFloat Wdkv,
                                         BufferFloat Wkr) noexcept {
            set_block_size(256u, 1u, 1u);
            set_name("mla_project_kv");

            auto rope_idx = [&](auto b_, auto h_, auto i_, auto d_) {
                return ((b_ * num_heads + h_) * seq_len + i_) * rope_dim + d_;
            };

            Var idx = dispatch_id().x;

            Var b = idx / seq_len;
            Var i = idx % seq_len;
            Var hi = (b * seq_len + i) * hidden_dim;

            // --- Compress KV: cKV[b,i,d] = W_DKV[d,:] @ H[b,i,:] ---
            $for (d, latent_dim) {
                Var acc = def(0.0f);
                $for (e, hidden_dim / 4u) {
                    Var e4 = e * 4u;
                    Var w_base = d * hidden_dim + e4;
                    Var h_base = hi + e4;
                    acc += Wdkv.read(w_base) * H.read(h_base)
                         + Wdkv.read(w_base + 1u) * H.read(h_base + 1u)
                         + Wdkv.read(w_base + 2u) * H.read(h_base + 2u)
                         + Wdkv.read(w_base + 3u) * H.read(h_base + 3u);
                };
                cKV.write((b * seq_len + i) * latent_dim + d, acc);
            };

            // --- Decoupled RoPE keys: Krope[b,h,i,r] = W_KR[h,r,:] @ H[b,i,:] ---
            $for (h, num_heads) {
                Var head_off_kr = h * rope_dim * hidden_dim;
                // Compute all rope_dim values, then apply RoPE.
                $for (r, rope_dim) {
                    Var acc = def(0.0f);
                    $for (e, hidden_dim / 4u) {
                        Var e4 = e * 4u;
                        Var w_base = head_off_kr + r * hidden_dim + e4;
                        Var h_base = hi + e4;
                        acc += Wkr.read(w_base) * H.read(h_base)
                             + Wkr.read(w_base + 1u) * H.read(h_base + 1u)
                             + Wkr.read(w_base + 2u) * H.read(h_base + 2u)
                             + Wkr.read(w_base + 3u) * H.read(h_base + 3u);
                    };
                    Krope.write(rope_idx(b, h, i, r), acc);
                };
                // Apply RoPE to the just-written Krope values for this head.
                $for (r, rope_dim / 2u) {
                    Var d0 = r * 2u;
                    Var d1 = d0 + 1u;
                    Float x0 = Krope.read(rope_idx(b, h, i, d0));
                    Float x1 = Krope.read(rope_idx(b, h, i, d1));
                    Float2 rot = apply_rope_pair(x0, x1, r, i);
                    Krope.write(rope_idx(b, h, i, d0), rot.x);
                    Krope.write(rope_idx(b, h, i, d1), rot.y);
                };
            };
        };

        // Online attention: fuses mla_score + softmax + av_weighted + up_project_v.
        // Uses online softmax (numerically stable) to compute
        // O[b,h,i,:] = softmax(score[i,:]) @ V[b,h,:,:] in a single pass.
        // Eliminates S, A, and V buffers entirely — V is computed inline
        // from Wuv (up-projection weight) and cKV, avoiding VRAM for V_buf
        // and reducing kernel launch count by 1.
        Kernel1D online_attention_kernel = [&](BufferFloat Q, BufferFloat cKV,
                                               BufferFloat Wuk, BufferFloat Krope,
                                               BufferFloat Wuv, BufferFloat O) noexcept {
            set_block_size(256u, 1u, 1u);
            set_name("mla_online_attention");

            auto qkv_idx = [&](auto b_, auto h_, auto i_, auto d_) {
                return ((b_ * num_heads + h_) * seq_len + i_) * head_dim + d_;
            };
            auto latent_idx = [&](auto b_, auto j_, auto d_) {
                return (b_ * seq_len + j_) * latent_dim + d_;
            };
            auto rope_idx = [&](auto b_, auto h_, auto j_, auto r_) {
                return ((b_ * num_heads + h_) * seq_len + j_) * rope_dim + r_;
            };

            Var idx = dispatch_id().x;
            Var b = idx / (num_heads * seq_len);
            Var h = (idx / seq_len) % num_heads;
            Var i = idx % seq_len;

            Var hc_off = h * content_dim * latent_dim;

            // Online softmax state.
            Var m = def(-1.0e30f);
            Var s_norm = def(0.0f);

            // Initialize output row to zero.
            $for (d, head_dim) {
                O.write(qkv_idx(b, h, i, d), 0.0f);
            };

            // -- Hoist q_latent = Wuk[h]^T @ q_content (independent of j) --
            // Pre-compute all latent_dim q_latent values once per thread
            // using a local array to avoid seq_len× recomputation.
            $array<float, latent_dim> q_latent;
            $for (d, latent_dim) {
                Var acc = def(0.0f);
                $for (c, content_dim / 4u) {
                    Var c4 = c * 4u;
                    Var w_base = hc_off + c4 * latent_dim + d;
                    Var q_base = qkv_idx(b, h, i, c4);
                    acc += Wuk.read(w_base) * Q.read(q_base)
                         + Wuk.read(w_base + latent_dim) * Q.read(q_base + 1u)
                         + Wuk.read(w_base + 2u * latent_dim) * Q.read(q_base + 2u)
                         + Wuk.read(w_base + 3u * latent_dim) * Q.read(q_base + 3u);
                };
                q_latent[d] = acc;
            };

            $for (j, seq_len) {
                // -- Content score via matrix absorption (reusing hoisted q_latent) --
                Var content_score = def(0.0f);
                $for (d, latent_dim) {
                    content_score += q_latent[d] * cKV.read(latent_idx(b, j, d));
                };

                // -- Positional score --
                Var pos_score = def(0.0f);
                $for (r, rope_dim / 4u) {
                    Var r4 = r * 4u;
                    Var q_base = qkv_idx(b, h, i, content_dim + r4);
                    Var k_base = rope_idx(b, h, j, r4);
                    pos_score += Q.read(q_base) * Krope.read(k_base)
                               + Q.read(q_base + 1u) * Krope.read(k_base + 1u)
                               + Q.read(q_base + 2u) * Krope.read(k_base + 2u)
                               + Q.read(q_base + 3u) * Krope.read(k_base + 3u);
                };

                Var score = (content_score + pos_score) * scale;

                // -- Online softmax update --
                Var m_new = max(m, score);
                Var exp_diff = exp(m - m_new);      // exp(m - m_new) ∈ (0, 1]
                Var exp_score = exp(score - m_new); // exp(score - m_new) ∈ (0, 1]
                s_norm = s_norm * exp_diff + exp_score;

                // -- Pre-load cKV[j,:] into local array to avoid reloading it
                //     64 times (once per head_dim) inside the d-loop. --
                $array<float, latent_dim> cKV_local;
                Var ci = (b * seq_len + j) * latent_dim;
                $for (l, latent_dim / 4u) {
                    Var l4 = l * 4u;
                    Var cv_base = ci + l4;
                    cKV_local[l4]     = cKV.read(cv_base);
                    cKV_local[l4 + 1u] = cKV.read(cv_base + 1u);
                    cKV_local[l4 + 2u] = cKV.read(cv_base + 2u);
                    cKV_local[l4 + 3u] = cKV.read(cv_base + 3u);
                };

                // Update O_row: O *= exp(m - m_new), then O += V[j] * exp(score - m_new).
                // V[b,h,j,d] is computed inline from Wuv and cKV_local (fused up_project_v)
                // to eliminate the V_buf intermediate: V[b,h,j,d] = Wuv[h,d,:] @ cKV[b,j,:]
                $for (d, head_dim) {
                    Var old_o = O.read(qkv_idx(b, h, i, d));
                    // Inline up_project_v: compute V[b,h,j,d] from Wuv[h,d,:] @ cKV_local[:]
                    Var head_off_uv = h * head_dim * latent_dim;
                    Var v_val = def(0.0f);
                    $for (l, latent_dim / 4u) {
                        Var l4 = l * 4u;
                        Var w_base = head_off_uv + d * latent_dim + l4;
                        v_val += Wuv.read(w_base)     * cKV_local[l4]
                               + Wuv.read(w_base + 1u) * cKV_local[l4 + 1u]
                               + Wuv.read(w_base + 2u) * cKV_local[l4 + 2u]
                               + Wuv.read(w_base + 3u) * cKV_local[l4 + 3u];
                    };
                    O.write(qkv_idx(b, h, i, d), old_o * exp_diff + v_val * exp_score);
                };

                m = m_new;
            };

            // -- Normalize output row by softmax sum --
            $for (d, head_dim) {
                Var val = O.read(qkv_idx(b, h, i, d));
                O.write(qkv_idx(b, h, i, d), val / s_norm);
            };
        };

        // -- MHA baseline kernel ----------------------------------------
        Kernel1D mha_online_attention_kernel = [&](BufferFloat Q, BufferFloat K,
                                                    BufferFloat V, BufferFloat O) noexcept {
            set_block_size(256u, 1u, 1u);
            set_name("mha_online_attention");

            auto qkv_idx = [&](auto b_, auto h_, auto i_, auto d_) {
                return ((b_ * num_heads + h_) * seq_len + i_) * head_dim + d_;
            };

            Var idx = dispatch_id().x;
            Var b = idx / (num_heads * seq_len);
            Var h = (idx / seq_len) % num_heads;
            Var i = idx % seq_len;

            Var head_base = (b * num_heads + h) * seq_len * head_dim;
            Var qi_base = head_base + i * head_dim;

            Var m = def(-1.0e30f);
            Var s_norm = def(0.0f);

            $for (d, head_dim) {
                O.write(qkv_idx(b, h, i, d), 0.0f);
            };

            $for (j, seq_len) {
                Var score = def(0.0f);
                Var kj_base = head_base + j * head_dim;
                $for (d, head_dim / 4u) {
                    Var d4 = d * 4u;
                    Var q_off = qi_base + d4;
                    Var k_off = kj_base + d4;
                    score += Q.read(q_off) * K.read(k_off)
                           + Q.read(q_off + 1u) * K.read(k_off + 1u)
                           + Q.read(q_off + 2u) * K.read(k_off + 2u)
                           + Q.read(q_off + 3u) * K.read(k_off + 3u);
                };
                score = score * scale;

                Var m_new = max(m, score);
                Var exp_diff = exp(m - m_new);
                Var exp_score = exp(score - m_new);
                s_norm = s_norm * exp_diff + exp_score;

                Var vj_base = head_base + j * head_dim;
                $for (d, head_dim) {
                    Var old_o = O.read(qkv_idx(b, h, i, d));
                    Var v_val = V.read(vj_base + d);
                    O.write(qkv_idx(b, h, i, d), old_o * exp_diff + v_val * exp_score);
                };

                m = m_new;
            };

            $for (d, head_dim) {
                Var val = O.read(qkv_idx(b, h, i, d));
                O.write(qkv_idx(b, h, i, d), val / s_norm);
            };
        };

        // -- Compile & dispatch selected path ----------------------------
        if (use_mla) {
            LUISA_INFO("Compiling MLA kernels ...");

            opt.name = "mla_project_q";
        auto project_q_shader = device.compile<1>(project_q_kernel, opt);
        opt.name = "mla_project_kv";
        auto project_kv_shader = device.compile(project_kv_kernel, opt);
        opt.name = "mla_online_attention";
        auto online_attention_shader = device.compile(online_attention_kernel, opt);

        double compile_ms = compile_clock.toc();
        LUISA_INFO("  MLA kernels compiled in {:.2f} ms", compile_ms);

        // Warm-up dispatch (not measured) — ensures GPU clocks are at full
        // frequency and PSO creation is amortized before real measurement.
        {
            CommandList warmup = CommandList::create();
            warmup << online_attention_shader(Q_buf, cKV_buf, Wuk_buf, Krope_buf, Wuv_buf, O_buf).dispatch(batch * num_heads * seq_len);
            stream << warmup.commit() << synchronize();
        }

        LUISA_INFO("Dispatching MLA GPU kernels ...");
        Clock dispatch_clock;
        CommandList cmd_list = CommandList::create();
        cmd_list << project_q_shader(H_buf, Q_buf, Wq_buf).dispatch(batch * seq_len)
                 << project_kv_shader(H_buf, cKV_buf, Krope_buf, Wdkv_buf, Wkr_buf).dispatch(batch * seq_len)
                 << online_attention_shader(Q_buf, cKV_buf, Wuk_buf, Krope_buf, Wuv_buf, O_buf).dispatch(batch * num_heads * seq_len);
        stream << cmd_list.commit() << synchronize();
        double dispatch_ms = dispatch_clock.toc();
        LUISA_INFO("  MLA GPU dispatch + sync: {:.2f} ms", dispatch_ms);

        } else {
            LUISA_INFO("Compiling MHA kernels (baseline) ...");

            opt.name = "mha_online_attention";
            auto mha_online_shader = device.compile(mha_online_attention_kernel, opt);

            double compile_ms = compile_clock.toc();
            LUISA_INFO("  MHA kernels compiled in {:.2f} ms", compile_ms);

            LUISA_INFO("Dispatching MHA GPU kernels ...");
            Clock dispatch_clock;
            stream << mha_online_shader(Q_buf, K_buf, V_buf, O_buf)
                           .dispatch(batch * num_heads * seq_len)
                   << synchronize();
            double dispatch_ms = dispatch_clock.toc();
            LUISA_INFO("  MHA GPU dispatch + sync: {:.2f} ms", dispatch_ms);
        }

    // -- Download results --------------------------------------------------
    luisa::vector<float> O_gpu(qkv_size);
    Clock download_clock;
    stream << O_buf.copy_to(luisa::span{O_gpu}) << synchronize();
    double download_ms = download_clock.toc();
    LUISA_INFO("  Download results: {:.2f} ms", download_ms);

    // -- CPU Reference -----------------------------------------------------
    LUISA_INFO("Running CPU reference ...");
    Clock cpu_clock;

    luisa::vector<float> O_cpu(qkv_size, 0.0f);

    // Fiber scheduler for the multi-threaded CPU reference.
    luisa::fiber::scheduler cpu_scheduler;

    auto apply_rope_pair_cpu = [](float x0, float x1, uint pair, uint pos) noexcept {
        float freq = 1.0f / std::pow(rope_theta, (2.0f * static_cast<float>(pair)) / rope_dim);
        float angle = static_cast<float>(pos) * freq;
        float c = std::cos(angle);
        float s = std::sin(angle);
        return std::make_pair(x0 * c - x1 * s, x0 * s + x1 * c);
    };

    if (use_mla) {
        // CPU-side MLA reference.
        luisa::vector<float> Q_cpu(qkv_size, 0.0f);
        luisa::vector<float> Krope_cpu(rope_size, 0.0f);
        luisa::vector<float> cKV_cpu(latent_size, 0.0f);
        luisa::vector<float> V_cpu(qkv_size, 0.0f);
        luisa::vector<float> S_cpu(scores_size, 0.0f);
        luisa::vector<float> A_cpu(scores_size, 0.0f);

        // Project Q.
        luisa::fiber::parallel(batch * seq_len, [&](uint32_t idx) noexcept {
            uint b = idx / seq_len;
            uint i = idx % seq_len;
            for (uint h = 0u; h < num_heads; ++h) {
                for (uint d = 0u; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (uint e = 0u; e < hidden_dim; ++e) {
                        uint wi = h * head_dim * hidden_dim + d * hidden_dim + e;
                        acc += Wq_host[wi] * H_host[hidden_index(b, i, e)];
                    }
                    Q_cpu[qkv_index(b, h, i, d)] = acc;
                }
                // RoPE on the positional slice.
                for (uint r = 0u; r < rope_dim / 2u; ++r) {
                    uint d0 = content_dim + r * 2u;
                    uint d1 = d0 + 1u;
                    auto rot = apply_rope_pair_cpu(
                        Q_cpu[qkv_index(b, h, i, d0)],
                        Q_cpu[qkv_index(b, h, i, d1)], r, i);
                    Q_cpu[qkv_index(b, h, i, d0)] = rot.first;
                    Q_cpu[qkv_index(b, h, i, d1)] = rot.second;
                }
            }
        });

        // Compress KV.
        luisa::fiber::parallel(batch * seq_len, [&](uint32_t idx) noexcept {
            uint b = idx / seq_len;
            uint i = idx % seq_len;
            for (uint d = 0u; d < latent_dim; ++d) {
                float acc = 0.0f;
                for (uint e = 0u; e < hidden_dim; ++e) {
                    acc += Wdkv_host[d * hidden_dim + e] * H_host[hidden_index(b, i, e)];
                }
                cKV_cpu[latent_index(b, i, d)] = acc;
            }
        });

        // Up-project V.
        luisa::fiber::parallel(batch * seq_len * num_heads, [&](uint32_t idx) noexcept {
            uint b = idx / (seq_len * num_heads);
            uint h = (idx / seq_len) % num_heads;
            uint j = idx % seq_len;
            for (uint d = 0u; d < head_dim; ++d) {
                float acc = 0.0f;
                for (uint l = 0u; l < latent_dim; ++l) {
                    uint wi = h * head_dim * latent_dim + d * latent_dim + l;
                    acc += Wuv_host[wi] * cKV_cpu[latent_index(b, j, l)];
                }
                V_cpu[qkv_index(b, h, j, d)] = acc;
            }
        });

        // Decoupled RoPE keys.
        luisa::fiber::parallel(batch * seq_len * num_heads, [&](uint32_t idx) noexcept {
            uint b = idx / (seq_len * num_heads);
            uint h = (idx / seq_len) % num_heads;
            uint j = idx % seq_len;
            for (uint r = 0u; r < rope_dim; ++r) {
                float acc = 0.0f;
                for (uint e = 0u; e < hidden_dim; ++e) {
                    uint wi = h * rope_dim * hidden_dim + r * hidden_dim + e;
                    acc += Wkr_host[wi] * H_host[hidden_index(b, j, e)];
                }
                Krope_cpu[rope_index(b, h, j, r)] = acc;
            }
            for (uint r = 0u; r < rope_dim / 2u; ++r) {
                uint d0 = r * 2u;
                uint d1 = d0 + 1u;
                auto rot = apply_rope_pair_cpu(
                    Krope_cpu[rope_index(b, h, j, d0)],
                    Krope_cpu[rope_index(b, h, j, d1)], r, j);
                Krope_cpu[rope_index(b, h, j, d0)] = rot.first;
                Krope_cpu[rope_index(b, h, j, d1)] = rot.second;
            }
        });

        // MLA scores with matrix absorption.
        luisa::fiber::parallel(batch * num_heads * seq_len, [&](uint32_t idx) noexcept {
            uint b = idx / (num_heads * seq_len);
            uint h = (idx / seq_len) % num_heads;
            uint i = idx % seq_len;
            for (uint j = 0u; j < seq_len; ++j) {
                // q_latent = Wuk[h]^T @ q_content.
                float q_latent[latent_dim];
                for (uint d = 0u; d < latent_dim; ++d) {
                    float acc = 0.0f;
                    for (uint c = 0u; c < content_dim; ++c) {
                        uint wi = h * content_dim * latent_dim + c * latent_dim + d;
                        acc += Wuk_host[wi] * Q_cpu[qkv_index(b, h, i, c)];
                    }
                    q_latent[d] = acc;
                }

                float content_score = 0.0f;
                for (uint d = 0u; d < latent_dim; ++d) {
                    content_score += q_latent[d] * cKV_cpu[latent_index(b, j, d)];
                }

                float pos_score = 0.0f;
                for (uint r = 0u; r < rope_dim; ++r) {
                    pos_score += Q_cpu[qkv_index(b, h, i, content_dim + r)] *
                                 Krope_cpu[rope_index(b, h, j, r)];
                }

                S_cpu[score_index(b, h, i, j)] = (content_score + pos_score) * scale;
            }
        });

        // Softmax.
        luisa::fiber::parallel(batch * num_heads * seq_len, [&](uint32_t idx) noexcept {
            uint b = idx / (num_heads * seq_len);
            uint h = (idx / seq_len) % num_heads;
            uint i = idx % seq_len;
            uint base = score_index(b, h, i, 0);
            float m = -1e30f;
            for (uint j = 0u; j < seq_len; ++j) {
                m = std::max(m, S_cpu[base + j]);
            }
            float s = 0.0f;
            for (uint j = 0u; j < seq_len; ++j) {
                float e = std::exp(S_cpu[base + j] - m);
                A_cpu[base + j] = e;
                s += e;
            }
            for (uint j = 0u; j < seq_len; ++j) {
                A_cpu[base + j] /= s;
            }
        });

        // AV weighted sum.
        luisa::fiber::parallel(batch * num_heads * seq_len * head_dim, [&](uint32_t idx) noexcept {
            uint b = idx / (num_heads * seq_len * head_dim);
            uint h = (idx / (seq_len * head_dim)) % num_heads;
            uint i = (idx / head_dim) % seq_len;
            uint d = idx % head_dim;
            float sum = 0.0f;
            for (uint j = 0u; j < seq_len; ++j) {
                sum += A_cpu[score_index(b, h, i, j)] * V_cpu[qkv_index(b, h, j, d)];
            }
            O_cpu[qkv_index(b, h, i, d)] = sum;
        });
    } else {
        // CPU-side MHA reference (multi-threaded).
        luisa::vector<float> S_cpu(scores_size, 0.0f);
        luisa::vector<float> A_cpu(scores_size, 0.0f);

        // QK^T + scale.
        luisa::fiber::parallel(batch * num_heads * seq_len, [&](uint32_t idx) noexcept {
            uint b = idx / (num_heads * seq_len);
            uint h = (idx / seq_len) % num_heads;
            uint i = idx % seq_len;
            for (uint j = 0u; j < seq_len; ++j) {
                float sum = 0.0f;
                for (uint d = 0u; d < head_dim; ++d) {
                    uint head_off = (b * num_heads + h) * seq_len * head_dim;
                    uint qi = head_off + i * head_dim + d;
                    uint ki = head_off + j * head_dim + d;
                    sum += Q_host[qi] * K_host[ki];
                }
                S_cpu[score_index(b, h, i, j)] = sum * scale;
            }
        });

        // Softmax.
        luisa::fiber::parallel(batch * num_heads * seq_len, [&](uint32_t idx) noexcept {
            uint b = idx / (num_heads * seq_len);
            uint h = (idx / seq_len) % num_heads;
            uint i = idx % seq_len;
            uint base = score_index(b, h, i, 0);
            float m = -1e30f;
            for (uint j = 0u; j < seq_len; ++j) {
                m = std::max(m, S_cpu[base + j]);
            }
            float s = 0.0f;
            for (uint j = 0u; j < seq_len; ++j) {
                float e = std::exp(S_cpu[base + j] - m);
                A_cpu[base + j] = e;
                s += e;
            }
            for (uint j = 0u; j < seq_len; ++j) {
                A_cpu[base + j] /= s;
            }
        });

        // AV weighted sum.
        luisa::fiber::parallel(batch * num_heads * seq_len * head_dim, [&](uint32_t idx) noexcept {
            uint b = idx / (num_heads * seq_len * head_dim);
            uint h = (idx / (seq_len * head_dim)) % num_heads;
            uint i = (idx / head_dim) % seq_len;
            uint d = idx % head_dim;
            float sum = 0.0f;
            for (uint j = 0u; j < seq_len; ++j) {
                sum += A_cpu[score_index(b, h, i, j)] * V_host[qkv_index(b, h, j, d)];
            }
            O_cpu[qkv_index(b, h, i, d)] = sum;
        });
    }

    double cpu_ms = cpu_clock.toc();
    LUISA_INFO("  CPU reference completed in {:.2f} ms", cpu_ms);

    // -- Performance / memory summary --------------------------------------
    LUISA_INFO("-- Summary ----------------------------------------------");
    LUISA_INFO("  Mode: {}", use_mla ? "MLA" : "MHA");
    LUISA_INFO("  Matrix config: batch={}, heads={}, seq_len={}, head_dim={}",
               batch, num_heads, seq_len, head_dim);
    if (use_mla) {
        LUISA_INFO("  MLA config: hidden_dim={}, latent_dim={}, rope_dim={}, content_dim={}",
                   hidden_dim, latent_dim, rope_dim, content_dim);
        size_t mha_kv_bytes = (2ull * num_heads * head_dim) * sizeof(float);
        size_t mla_kv_bytes = (static_cast<size_t>(latent_dim) + num_heads * rope_dim) * sizeof(float);
        float ratio = static_cast<float>(mla_kv_bytes) / static_cast<float>(mha_kv_bytes);
        LUISA_INFO("  KV-cache per token: MHA={} B, MLA={} B, ratio={:.2f}%",
                   mha_kv_bytes, mla_kv_bytes, ratio * 100.0f);
    }

    // -- Verification ------------------------------------------------------
    float max_diff = 0.0f;
    uint max_idx = 0u;
    for (uint i = 0u; i < qkv_size; ++i) {
        float diff = std::abs(O_gpu[i] - O_cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    constexpr float tolerance = 1e-4f;
    bool passed = max_diff <= tolerance;

    LUISA_INFO("Verification: {}", passed ? "PASSED" : "FAILED");
    LUISA_INFO("  Max error: {} at index {} (GPU={}, CPU={})",
               max_diff, max_idx, O_gpu[max_idx], O_cpu[max_idx]);

    if (!passed) {
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
