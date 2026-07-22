// Multi-Head Latent Attention (MLA) Example -- DSL kernel definitions.

#include "attention_config.h"
#include "attention_kernels.h"

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/coop_vector.h>
#include <luisa/dsl/resource.h>

using namespace luisa;
using namespace luisa::compute;

namespace mla {
namespace {

// -- Reusable RoPE rotation helper (DSL) -----------------------------------
// The inverse frequencies are precomputed host-side and captured as a
// $constant, avoiding pow() in every kernel call.
using ApplyRopePairCallable = Callable<float2(float, float, uint, uint)>;

[[nodiscard]] ApplyRopePairCallable make_apply_rope_pair_callable() noexcept {
    auto inv_freqs = make_rope_inv_freqs();
    ApplyRopePairCallable apply_rope_pair = [inv_freqs](Float x0, Float x1, const UInt &pair, const UInt &pos) noexcept {
        $constant freqs = inv_freqs;
        Float freq = freqs.read(pair);
        Float angle = pos.cast<float>() * freq;
        Float c = cos(angle);
        Float s = sin(angle);
        return make_float2(x0 * c - x1 * s, x0 * s + x1 * c);
    };
    return apply_rope_pair;
}

}// namespace

// ---------------------------------------------------------------------------
// Project Q kernel (unified template)
// ---------------------------------------------------------------------------
template <bool Cooperative>
ProjectQKernel create_project_q_kernel() {
    auto apply_rope_pair = make_apply_rope_pair_callable();

    // Project queries: Q[b,h,i,d] = Wq[h,d,:] @ H[b,i,:] (shared-memory tiled).
    // Each block handles one (b,i) token; all 256 threads cooperatively load
    // H into shared memory, then each thread computes one adjacent (d, d+1)
    // output pair (512 outputs / 256 threads). RoPE is applied in registers
    // before the single write, since each thread owns a complete RoPE pair.
    // Grid: (batch * seq_len). Block: (256, 1).
    ProjectQKernel kernel = [&](BufferFloat H, BufferFloat Q,
                                 BufferFloat Wq, ByteBufferVar Wq_byte_buf) noexcept {
        constexpr uint kBlockSize = project_q_block_size;
        set_block_size(kBlockSize, 1u, 1u);
        if constexpr (Cooperative) {
            set_name("mla_project_q_coop");
        } else {
            set_name("mla_project_q");
        }

        Shared<float> H_shared{hidden_dim};

        auto qkv_idx = [&](auto b_, auto h_, auto i_, auto d_) {
            return ((b_ * num_heads + h_) * seq_len + i_) * head_dim + d_;
        };

        // One block per token; the host dispatches tokens * kBlockSize threads.
        Var idx = block_id().x;
        Var b = idx / seq_len;
        Var i = idx % seq_len;
        Var tx = thread_x();
        Var hi = (b * seq_len + i) * hidden_dim;

        // -- Cooperative load of H[b,i,:] into shared memory (all 256 threads) --
        $for (e, hidden_dim / kBlockSize) {
            Var e_idx = tx * (hidden_dim / kBlockSize) + e;
            H_shared[e_idx] = H.read(hi + e_idx);
        };
        sync_block();

        // -- Each thread computes the adjacent output pair (d, d + 1) --
        Var d = tx * 2u % head_dim;
        Var h = tx * 2u / head_dim;
        Var w_row = h * (head_dim * hidden_dim) + d * hidden_dim;

        Var acc0 = def(0.0f);
        Var acc1 = def(0.0f);

        if constexpr (Cooperative) {
            // -- Cooperative vector FMA path --
            $for (chunk, kHiddenChunks) {
                Var chunk_start = chunk * kCoopChunk;

                CoopVectorRef w0_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                CoopVectorRef w1_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                w0_ref.set_byte_offset((w_row + chunk_start) * 4u);
                w1_ref.set_byte_offset((w_row + hidden_dim + chunk_start) * 4u);
                auto w0_chunk = cooperative_vector_load<float>(Wq_byte_buf, w0_ref);
                auto w1_chunk = cooperative_vector_load<float>(Wq_byte_buf, w1_ref);

                CoopVector<float> h_chunk{kCoopChunk};
                $for (t, kCoopChunk) {
                    h_chunk[t] = H_shared[chunk_start + t];
                };

                auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                auto prod0 = cooperative_vector_fma(w0_chunk, h_chunk, zero);
                auto prod1 = cooperative_vector_fma(w1_chunk, h_chunk, zero);

                $for (t, kCoopChunk) {
                    acc0 += prod0[t];
                    acc1 += prod1[t];
                };
            };
        } else {
            // -- Scalar fallback path (4-unrolled dot product) --
            $for (e, hidden_dim / 4u) {
                Var e4 = e * 4u;
                acc0 += Wq.read(w_row + e4) * H_shared[e4]
                      + Wq.read(w_row + e4 + 1u) * H_shared[e4 + 1u]
                      + Wq.read(w_row + e4 + 2u) * H_shared[e4 + 2u]
                      + Wq.read(w_row + e4 + 3u) * H_shared[e4 + 3u];
                acc1 += Wq.read(w_row + hidden_dim + e4) * H_shared[e4]
                      + Wq.read(w_row + hidden_dim + e4 + 1u) * H_shared[e4 + 1u]
                      + Wq.read(w_row + hidden_dim + e4 + 2u) * H_shared[e4 + 2u]
                      + Wq.read(w_row + hidden_dim + e4 + 3u) * H_shared[e4 + 3u];
            };
        };

        // Fuse RoPE: rotate the pair in registers, write Q exactly once.
        $if (d >= content_dim) {
            Float2 rot = apply_rope_pair(acc0, acc1, (d - content_dim) / 2u, i);
            acc0 = rot.x;
            acc1 = rot.y;
        };
        Q.write(qkv_idx(b, h, i, d), acc0);
        Q.write(qkv_idx(b, h, i, d + 1u), acc1);
    };
    return kernel;
}

// Explicit instantiations for Project Q.
template ProjectQKernel create_project_q_kernel<true>();
template ProjectQKernel create_project_q_kernel<false>();

// ---------------------------------------------------------------------------
// Project KV kernel (unified template)
// ---------------------------------------------------------------------------
template <bool Cooperative>
ProjectKVKernel create_project_kv_kernel() {
    auto apply_rope_pair = make_apply_rope_pair_callable();

    // Fused project KV: cKV + Krope from a single H read per token.
    // Each block handles one (b,i) token: all 128 threads cooperatively stage
    // H[b,i,:] in shared memory, then threads 0..63 each compute one cKV
    // output and threads 64..127 each compute one Krope pair (RoPE applied
    // in registers before the single write).
    // Grid: (batch * seq_len). Block: (128, 1).
    //
    // The unified signature takes 7 params: H, cKV, Krope, Wdkv, Wkr,
    // Wdkv_byte_buf, Wkr_byte_buf. H_byte_buf was unused in the coop version
    // and has been dropped.
    ProjectKVKernel kernel = [&](BufferFloat H, BufferFloat cKV,
                                  BufferFloat Krope, BufferFloat Wdkv,
                                  BufferFloat Wkr,
                                  ByteBufferVar Wdkv_byte_buf,
                                  ByteBufferVar Wkr_byte_buf) noexcept {
        constexpr uint kBlockSize = project_kv_block_size;
        set_block_size(kBlockSize, 1u, 1u);
        if constexpr (Cooperative) {
            set_name("mla_project_kv_coop");
        } else {
            set_name("mla_project_kv");
        }

        Shared<float> H_shared{hidden_dim};

        auto rope_idx = [&](auto b_, auto h_, auto i_, auto d_) {
            return ((b_ * num_heads + h_) * seq_len + i_) * rope_dim + d_;
        };

        // One block per token; the host dispatches tokens * kBlockSize threads.
        Var idx = block_id().x;

        Var b = idx / seq_len;
        Var i = idx % seq_len;
        Var tx = thread_x();
        Var hi = (b * seq_len + i) * hidden_dim;

        // -- Cooperative load of H[b,i,:] into shared memory (all 128 threads) --
        $for (e, hidden_dim / kBlockSize) {
            Var e_idx = tx * (hidden_dim / kBlockSize) + e;
            H_shared[e_idx] = H.read(hi + e_idx);
        };
        sync_block();

        $if (tx < latent_dim) {
            // --- Compress KV: cKV[b,i,tx] = W_DKV[tx,:] @ H[b,i,:] ---
            Var acc = def(0.0f);
            Var w_row = tx * hidden_dim;
            if constexpr (Cooperative) {
                // -- Cooperative fma path --
                $for (chunk, kHiddenChunks) {
                    Var chunk_start = chunk * kCoopChunk;

                    CoopVectorRef wdkv_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    wdkv_ref.set_byte_offset((w_row + chunk_start) * 4u);
                    auto w_chunk = cooperative_vector_load<float>(Wdkv_byte_buf, wdkv_ref);

                    CoopVector<float> h_chunk{kCoopChunk};
                    $for (t, kCoopChunk) {
                        h_chunk[t] = H_shared[chunk_start + t];
                    };

                    auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                    auto prod = cooperative_vector_fma(w_chunk, h_chunk, zero);

                    $for (t, kCoopChunk) {
                        acc += prod[t];
                    };
                };
            } else {
                // -- Scalar fallback (4-unrolled) --
                $for (e, hidden_dim / 4u) {
                    Var e4 = e * 4u;
                    acc += Wdkv.read(w_row + e4) * H_shared[e4]
                         + Wdkv.read(w_row + e4 + 1u) * H_shared[e4 + 1u]
                         + Wdkv.read(w_row + e4 + 2u) * H_shared[e4 + 2u]
                         + Wdkv.read(w_row + e4 + 3u) * H_shared[e4 + 3u];
                };
            };
            cKV.write((b * seq_len + i) * latent_dim + tx, acc);
        } $else {
            // --- Decoupled RoPE keys: one (r, r+1) pair per thread ---
            Var p = tx - latent_dim;            // pair index, 0..63
            Var h = p / (rope_dim / 2u);        // head
            Var r = (p % (rope_dim / 2u)) * 2u; // first dim of the pair
            Var w_row = h * (rope_dim * hidden_dim) + r * hidden_dim;
            Var acc0 = def(0.0f);
            Var acc1 = def(0.0f);
            if constexpr (Cooperative) {
                // -- Cooperative fma path --
                $for (chunk, kHiddenChunks) {
                    Var chunk_start = chunk * kCoopChunk;

                    CoopVectorRef w0_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    CoopVectorRef w1_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    w0_ref.set_byte_offset((w_row + chunk_start) * 4u);
                    w1_ref.set_byte_offset((w_row + hidden_dim + chunk_start) * 4u);
                    auto w0_chunk = cooperative_vector_load<float>(Wkr_byte_buf, w0_ref);
                    auto w1_chunk = cooperative_vector_load<float>(Wkr_byte_buf, w1_ref);

                    CoopVector<float> h_chunk{kCoopChunk};
                    $for (t, kCoopChunk) {
                        h_chunk[t] = H_shared[chunk_start + t];
                    };

                    auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                    auto prod0 = cooperative_vector_fma(w0_chunk, h_chunk, zero);
                    auto prod1 = cooperative_vector_fma(w1_chunk, h_chunk, zero);

                    $for (t, kCoopChunk) {
                        acc0 += prod0[t];
                        acc1 += prod1[t];
                    };
                };
            } else {
                // -- Scalar fallback (4-unrolled) --
                $for (e, hidden_dim / 4u) {
                    Var e4 = e * 4u;
                    acc0 += Wkr.read(w_row + e4) * H_shared[e4]
                          + Wkr.read(w_row + e4 + 1u) * H_shared[e4 + 1u]
                          + Wkr.read(w_row + e4 + 2u) * H_shared[e4 + 2u]
                          + Wkr.read(w_row + e4 + 3u) * H_shared[e4 + 3u];
                    acc1 += Wkr.read(w_row + hidden_dim + e4) * H_shared[e4]
                          + Wkr.read(w_row + hidden_dim + e4 + 1u) * H_shared[e4 + 1u]
                          + Wkr.read(w_row + hidden_dim + e4 + 2u) * H_shared[e4 + 2u]
                          + Wkr.read(w_row + hidden_dim + e4 + 3u) * H_shared[e4 + 3u];
                };
            };
            // Fuse RoPE: rotate the pair in registers, write Krope exactly once.
            Float2 rot = apply_rope_pair(acc0, acc1, r / 2u, i);
            Krope.write(rope_idx(b, h, i, r), rot.x);
            Krope.write(rope_idx(b, h, i, r + 1u), rot.y);
        };
    };
    return kernel;
}

// Explicit instantiations for Project KV.
template ProjectKVKernel create_project_kv_kernel<true>();
template ProjectKVKernel create_project_kv_kernel<false>();

// ---------------------------------------------------------------------------
// Online attention kernel (unified template)
// ---------------------------------------------------------------------------
template <bool Cooperative>
OnlineAttentionKernel create_online_attention_kernel() {
    // -- Shared-memory-optimized MLA online attention kernel (fallback) --------
    // Broadcast cKV[j,:] and Krope[j,:] via shared memory to eliminate
    // 256x redundant global loads. Only latent_dim=64 threads load
    // cKV and rope_dim=16 threads load Krope; all 256 read from shared.
    //
    // -- Cooperative-vector-optimized variant -------------------------------
    // Uses cooperative_vector_fma for all dot products:
    //   - q_latent hoisting (Wuk^T @ Q_content)
    //   - content score (q_latent @ cKV)
    //   - positional score (Q_pos @ Krope)
    //   - V up-projection (Wuv @ cKV)
    OnlineAttentionKernel kernel = [&](BufferFloat Q, BufferFloat cKV,
                                        BufferFloat Wuk, BufferFloat Krope,
                                        BufferFloat Wuv, BufferFloat O,
                                        ByteBufferVar Q_byte_buf,
                                        ByteBufferVar cKV_byte_buf,
                                        ByteBufferVar Krope_byte_buf,
                                        ByteBufferVar Wuv_byte_buf) noexcept {
        set_block_size(256u, 1u, 1u);
        if constexpr (Cooperative) {
            set_name("mla_online_attention_coop");
        } else {
            set_name("mla_online_attention");
        }

        auto qkv_idx = [&](auto b_, auto h_, auto i_, auto d_) {
            return ((b_ * num_heads + h_) * seq_len + i_) * head_dim + d_;
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

        // Latent-space output accumulator kept in registers. The Wuv
        // up-projection is applied once after the loop (matrix absorption).
        $array<float, latent_dim> a_acc;

        // -- Hoist q_latent = Wuk[h]^T @ Q_content --
        $array<float, latent_dim> q_latent;
        if constexpr (Cooperative) {
            // -- Cooperative fma path (chunked over content_dim) --
            $for (d, latent_dim) {
                Var acc = def(0.0f);
                $for (chunk, kContentChunks) {
                    Var chunk_start = chunk * kCoopChunk;

                    // Load Wuk[h, chunk_start:chunk_start+16, d] into CoopVector (strided access).
                    CoopVector<float> w_chunk{kCoopChunk};
                    $for (c, kCoopChunk) {
                        Var c_idx = chunk_start + c;
                        Var w_base = hc_off + c_idx * latent_dim + d;
                        w_chunk[c] = Wuk.read(w_base);
                    };

                    // Load Q_content[chunk_start:chunk_start+16] via CoopVectorRef.
                    CoopVectorRef q_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    Var q_base = qkv_idx(b, h, i, chunk_start);
                    q_ref.set_byte_offset(q_base * 4u);
                    auto q_chunk = cooperative_vector_load<float>(Q_byte_buf, q_ref);

                    auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                    auto prod = cooperative_vector_fma(w_chunk, q_chunk, zero);

                    $for (c, kCoopChunk) {
                        acc += prod[c];
                    };
                };
                q_latent[d] = acc;
            };
        } else {
            // -- Scalar fallback (4-unrolled) --
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
        };

        // -- Hoist the loop-invariant positional query slice into registers --
        // The CoopVector load (coop path) and scalar $array (fallback path)
        // are placed inside the same `if constexpr` branch as their respective
        // j-loops so that q_pos_vec remains in scope.
        $array<float, rope_dim> q_pos;
        if constexpr (Cooperative) {
            // Cooperative vector load of the full rope_dim chunk.
            CoopVectorRef q_pos_ref{CoopRefVecType::FLOAT32, kCoopChunk};
            Var q_pos_base = qkv_idx(b, h, i, content_dim);
            q_pos_ref.set_byte_offset(q_pos_base * 4u);
            auto q_pos_vec = cooperative_vector_load<float>(Q_byte_buf, q_pos_ref);
            // Also populate the scalar array for the Wuv up-projection (shared code).
            $for (r, rope_dim) {
                q_pos[r] = q_pos_vec[r];
            };

            // -- Cooperative j-loop: load cKV/Krope directly via ByteBuf --
            $for (j, seq_len) {
                Var ci = (b * seq_len + j) * latent_dim;

                // -- Content score via matrix absorption (cooperative fma) --
                Var content_score = def(0.0f);
                $for (chunk, kLatentChunks) {
                    Var chunk_start = chunk * kCoopChunk;

                    // Load q_latent[chunk_start:chunk_start+16] into a CoopVector.
                    CoopVector<float> q_chunk{kCoopChunk};
                    $for (c, kCoopChunk) {
                        q_chunk[c] = q_latent[chunk_start + c];
                    };

                    // Load cKV[j, chunk_start:chunk_start+16] via CoopVectorRef.
                    CoopVectorRef ckv_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    ckv_ref.set_byte_offset((ci + chunk_start) * 4u);
                    auto c_chunk = cooperative_vector_load<float>(cKV_byte_buf, ckv_ref);

                    auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                    auto prod = cooperative_vector_fma(q_chunk, c_chunk, zero);

                    $for (c, kCoopChunk) {
                        content_score += prod[c];
                    };
                };

                // -- Positional score (rope_dim=16 fits exactly in one chunk) --
                Var pos_score = def(0.0f);
                {
                    CoopVectorRef k_pos_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    Var k_base = rope_idx(b, h, j, 0u);
                    k_pos_ref.set_byte_offset(k_base * 4u);
                    auto k_pos = cooperative_vector_load<float>(Krope_byte_buf, k_pos_ref);
                    auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                    auto prod = cooperative_vector_fma(q_pos_vec, k_pos, zero);
                    $for (r, rope_dim) {
                        pos_score += prod[r];
                    };
                };

                Var score = (content_score + pos_score) * attention_scale;

                // -- Online softmax update --
                Var m_new = max(m, score);
                Var exp_diff = exp(m - m_new);
                Var exp_score = exp(score - m_new);
                s_norm = s_norm * exp_diff + exp_score;

                // -- Accumulate in latent space: A *= exp_diff; A += cKV[j] * exp_score --
                // (read cKV from global memory directly — coop path doesn't use shared for this)
                $for (l, latent_dim / 4u) {
                    Var l4 = l * 4u;
                    Var cv_base = ci + l4;
                    a_acc[l4] = a_acc[l4] * exp_diff + cKV.read(cv_base) * exp_score;
                    a_acc[l4 + 1u] = a_acc[l4 + 1u] * exp_diff + cKV.read(cv_base + 1u) * exp_score;
                    a_acc[l4 + 2u] = a_acc[l4 + 2u] * exp_diff + cKV.read(cv_base + 2u) * exp_score;
                    a_acc[l4 + 3u] = a_acc[l4 + 3u] * exp_diff + cKV.read(cv_base + 3u) * exp_score;
                };

                m = m_new;
            };
        } else {
            // Scalar loads into $array.
            $for (r, rope_dim) {
                q_pos[r] = Q.read(qkv_idx(b, h, i, content_dim + r));
            };

            // -- Fallback j-loop with shared-memory broadcast of cKV/Krope tiles --
            Shared<float> cKV_shared{latent_dim};
            Shared<float> Krope_shared{rope_dim};

            $for (j, seq_len) {
                Var ci = (b * seq_len + j) * latent_dim;

                // -- Cooperative load cKV[j,:] and Krope[j,:] into shared memory --
                $if (thread_x() < latent_dim) {
                    cKV_shared[thread_x()] = cKV.read(ci + thread_x());
                };
                $if (thread_x() < rope_dim) {
                    Var k_base = rope_idx(b, h, j, 0u);
                    Krope_shared[thread_x()] = Krope.read(k_base + thread_x());
                };
                sync_block();

                // -- Content score (read cKV from shared) --
                Var content_score = def(0.0f);
                $for (d, latent_dim) {
                    content_score += q_latent[d] * cKV_shared[d];
                };

                // -- Positional score (read Krope from shared) --
                Var pos_score = def(0.0f);
                $for (r, rope_dim / 4u) {
                    Var r4 = r * 4u;
                    pos_score += q_pos[r4] * Krope_shared[r4]
                               + q_pos[r4 + 1u] * Krope_shared[r4 + 1u]
                               + q_pos[r4 + 2u] * Krope_shared[r4 + 2u]
                               + q_pos[r4 + 3u] * Krope_shared[r4 + 3u];
                };

                Var score = (content_score + pos_score) * attention_scale;

                // -- Online softmax update --
                Var m_new = max(m, score);
                Var exp_diff = exp(m - m_new);
                Var exp_score = exp(score - m_new);
                s_norm = s_norm * exp_diff + exp_score;

                // -- Accumulate in latent space: A *= exp_diff; A += cKV[j] * exp_score --
                // (cKV_shared already staged in fast shared memory)
                $for (l, latent_dim / 4u) {
                    Var l4 = l * 4u;
                    a_acc[l4] = a_acc[l4] * exp_diff + cKV_shared[l4] * exp_score;
                    a_acc[l4 + 1u] = a_acc[l4 + 1u] * exp_diff + cKV_shared[l4 + 1u] * exp_score;
                    a_acc[l4 + 2u] = a_acc[l4 + 2u] * exp_diff + cKV_shared[l4 + 2u] * exp_score;
                    a_acc[l4 + 3u] = a_acc[l4 + 3u] * exp_diff + cKV_shared[l4 + 3u] * exp_score;
                };
                // All threads must finish consuming this cKV/Krope tile
                // before the next iteration overwrites shared memory.
                sync_block();

                m = m_new;
            };
        };

        // -- Up-project from latent space once: O = Wuv[h] @ A, normalized --
        // (Wuv is read seq_len times less than a per-j up-projection)
        Var head_off_uv = h * head_dim * latent_dim;
        if constexpr (Cooperative) {
            // -- Cooperative fma path (chunked over latent_dim) --
            $for (d, head_dim) {
                Var val = def(0.0f);
                $for (chunk, kLatentChunks) {
                    Var chunk_start = chunk * kCoopChunk;

                    // Load Wuv[d, chunk_start:chunk_start+16] via CoopVectorRef.
                    CoopVectorRef wuv_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    Var w_base = head_off_uv + d * latent_dim + chunk_start;
                    wuv_ref.set_byte_offset(w_base * 4u);
                    auto w_chunk = cooperative_vector_load<float>(Wuv_byte_buf, wuv_ref);

                    // Load a_acc[chunk_start:chunk_start+16] into CoopVector.
                    CoopVector<float> a_chunk{kCoopChunk};
                    $for (c, kCoopChunk) {
                        a_chunk[c] = a_acc[chunk_start + c];
                    };

                    auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                    auto prod = cooperative_vector_fma(w_chunk, a_chunk, zero);

                    $for (c, kCoopChunk) {
                        val += prod[c];
                    };
                };
                O.write(qkv_idx(b, h, i, d), val / s_norm);
            };
        } else {
            // -- Scalar fallback (4-unrolled) --
            $for (d, head_dim) {
                Var val = def(0.0f);
                $for (l, latent_dim / 4u) {
                    Var l4 = l * 4u;
                    Var w_base = head_off_uv + d * latent_dim + l4;
                    val += Wuv.read(w_base) * a_acc[l4]
                         + Wuv.read(w_base + 1u) * a_acc[l4 + 1u]
                         + Wuv.read(w_base + 2u) * a_acc[l4 + 2u]
                         + Wuv.read(w_base + 3u) * a_acc[l4 + 3u];
                };
                O.write(qkv_idx(b, h, i, d), val / s_norm);
            };
        };
    };
    return kernel;
}

// Explicit instantiations for Online Attention.
template OnlineAttentionKernel create_online_attention_kernel<true>();
template OnlineAttentionKernel create_online_attention_kernel<false>();

// ---------------------------------------------------------------------------
// MHA kernel (unchanged)
// ---------------------------------------------------------------------------
MhaOnlineAttentionKernel create_mha_online_attention_kernel() {
    // -- Shared-memory-optimized MHA kernel (broadcast K/V via shared memory) --
    // A single 32-thread wave consumes each shared K/V tile. Keeping the
    // consumer cohort within one wave avoids cross-wave progress skew while
    // still amortizing each global K/V load over 32 query rows.
    // Keys/values are consumed kTile tokens per iteration: all 32 threads
    // cooperatively fill the tile, halving the barrier count per token.
    MhaOnlineAttentionKernel kernel = [&](BufferFloat Q, BufferFloat K,
                                          BufferFloat V, BufferFloat O) noexcept {
        constexpr uint kBlockSize = 32u;
        constexpr uint kTile = 8u;
        static_assert(seq_len % kTile == 0u, "seq_len must be a multiple of the j tile");
        static_assert((kTile * head_dim) % kBlockSize == 0u);
        set_block_size(kBlockSize, 1u, 1u);
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

        // Output accumulator kept in registers; written to global memory once.
        $array<float, head_dim> o_acc;

        // Hoist the loop-invariant query row into registers.
        $array<float, head_dim> q_local;
        $for (d, head_dim) {
            q_local[d] = Q.read(qi_base + d);
        };

        Shared<float> K_shared{kTile * head_dim};
        Shared<float> V_shared{kTile * head_dim};

        $for (jt, seq_len / kTile) {
            // All 32 threads cooperatively load the kTile-token K/V tile
            // (2 * kTile * head_dim floats, 32 per thread).
            Var kt_base = head_base + jt * (kTile * head_dim);
            $for (e, kTile * head_dim / kBlockSize) {
                Var e_idx = thread_x() + e * kBlockSize;
                K_shared[e_idx] = K.read(kt_base + e_idx);
                V_shared[e_idx] = V.read(kt_base + e_idx);
            };
            sync_block();

            $for (jj, kTile) {
                Var tile_base = jj * head_dim;

                // Score = Q[i] · K[j] (read K from shared)
                Var score = def(0.0f);
                $for (d, head_dim / 4u) {
                    Var d4 = d * 4u;
                    score += q_local[d4] * K_shared[tile_base + d4]
                           + q_local[d4 + 1u] * K_shared[tile_base + d4 + 1u]
                           + q_local[d4 + 2u] * K_shared[tile_base + d4 + 2u]
                           + q_local[d4 + 3u] * K_shared[tile_base + d4 + 3u];
                };
                score = score * attention_scale;

                Var m_new = max(m, score);
                Var exp_diff = exp(m - m_new);
                Var exp_score = exp(score - m_new);
                s_norm = s_norm * exp_diff + exp_score;

                // Update O with V from shared (register accumulator)
                $for (d, head_dim) {
                    o_acc[d] = o_acc[d] * exp_diff + V_shared[tile_base + d] * exp_score;
                };

                m = m_new;
            };
            // All threads must finish consuming this K/V tile before
            // the next tile overwrites shared memory.
            sync_block();
        };

        // Normalize output row by softmax sum (single global write)
        $for (d, head_dim) {
            O.write(qkv_idx(b, h, i, d), o_acc[d] / s_norm);
        };
    };
    return kernel;
}

}// namespace mla
