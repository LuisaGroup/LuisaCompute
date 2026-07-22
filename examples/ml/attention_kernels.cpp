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

ProjectQKernel create_project_q_kernel() {
    auto apply_rope_pair = make_apply_rope_pair_callable();

    // Project queries: Q[b,h,i,d] = Wq[h,d,:] @ H[b,i,:] (shared-memory tiled).
    // Each block handles one (b,i) token; 32 threads cooperatively load H
    // into shared memory; the first num_heads threads compute their assigned head.
    // Grid: (batch * seq_len). Block: (32, 1).
    ProjectQKernel kernel = [&](BufferFloat H, BufferFloat Q, BufferFloat Wq) noexcept {
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
    return kernel;
}

ProjectQCoopKernel create_project_q_coop_kernel() {
    auto apply_rope_pair = make_apply_rope_pair_callable();

    // -- Cooperative-vector-optimized project Q kernel -----------------
    // Uses cooperative_vector_fma for the Wq @ H dot product.
    ProjectQCoopKernel kernel = [&](BufferFloat H, BufferFloat Q, BufferFloat Wq, ByteBufferVar Wq_byte_buf) noexcept {
        constexpr uint kBlockSize = 32u;
        set_block_size(kBlockSize, 1u, 1u);
        set_name("mla_project_q_coop");

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
                // Chunked dot product over hidden_dim=512 (kHiddenChunks=32 chunks of 16).
                $for (chunk, kHiddenChunks) {
                    Var chunk_start = chunk * kCoopChunk;

                    // Load Wq[d, chunk_start:chunk_start+16] via CoopVectorRef (direct buffer link).
                    CoopVectorRef w_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    Var w_base = h_off + d * hidden_dim + chunk_start;
                    w_ref.set_byte_offset(w_base * 4u);
                    auto w_chunk = cooperative_vector_load<float>(Wq_byte_buf, w_ref);

                    // Load H_shared[chunk_start:chunk_start+16] into CoopVector.
                    CoopVector<float> h_chunk{kCoopChunk};
                    $for (i, kCoopChunk) {
                        h_chunk[i] = H_shared[chunk_start + i];
                    };

                    auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                    auto prod = cooperative_vector_fma(w_chunk, h_chunk, zero);

                    $for (i, kCoopChunk) {
                        acc += prod[i];
                    };
                };
                Q.write(qkv_idx(b, h, i, d), acc);
            };

            // Apply RoPE to the positional slice (same as regular kernel).
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
    return kernel;
}

ProjectKVKernel create_project_kv_kernel() {
    auto apply_rope_pair = make_apply_rope_pair_callable();

    // Fused project KV: cKV + Krope from a single H read per token.
    // Each thread handles one (b,i) token, computing cKV (latent_dim=16)
    // and Krope (num_heads×rope_dim=64) in one pass.
    ProjectKVKernel kernel = [&](BufferFloat H, BufferFloat cKV,
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
    return kernel;
}

ProjectKVCoopKernel create_project_kv_coop_kernel() {
    auto apply_rope_pair = make_apply_rope_pair_callable();

    // -- Cooperative-vector-optimized project KV kernel ----------------
    // Uses cooperative_vector_fma for both cKV and Krope dot products.
    // The Krope computation benefits from rope_dim=16 fitting exactly
    // in a single cooperative vector chunk.
    ProjectKVCoopKernel kernel = [&](BufferFloat H, BufferFloat cKV,
                                     BufferFloat Krope, BufferFloat Wdkv,
                                     BufferFloat Wkr,
                                     ByteBufferVar H_byte_buf,
                                     ByteBufferVar Wdkv_byte_buf,
                                     ByteBufferVar Wkr_byte_buf) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("mla_project_kv_coop");

        auto rope_idx = [&](auto b_, auto h_, auto i_, auto d_) {
            return ((b_ * num_heads + h_) * seq_len + i_) * rope_dim + d_;
        };

        Var idx = dispatch_id().x;

        Var b = idx / seq_len;
        Var i = idx % seq_len;
        Var hi = (b * seq_len + i) * hidden_dim;

        // --- cKV[b,i,d] = W_DKV[d,:] @ H[b,i,:] (cooperative fma) ---
        // latent_dim=64 values, each dot product over hidden_dim=512 (32 chunks).
        $for (d, latent_dim) {
            Var acc = def(0.0f);
            $for (chunk, kHiddenChunks) {
                Var chunk_start = chunk * kCoopChunk;

                CoopVectorRef wdkv_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                Var w_base = d * hidden_dim + chunk_start;
                wdkv_ref.set_byte_offset(w_base * 4u);
                auto w_chunk = cooperative_vector_load<float>(Wdkv_byte_buf, wdkv_ref);

                CoopVectorRef h_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                Var h_base = hi + chunk_start;
                h_ref.set_byte_offset(h_base * 4u);
                auto h_chunk = cooperative_vector_load<float>(H_byte_buf, h_ref);

                auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                auto prod = cooperative_vector_fma(w_chunk, h_chunk, zero);

                $for (j, kCoopChunk) {
                    acc += prod[j];
                };
            };
            cKV.write((b * seq_len + i) * latent_dim + d, acc);
        };

        // --- Krope[b,h,i,r] = W_KR[h,r,:] @ H[b,i,:] (cooperative fma) ---
        // rope_dim=16 fits exactly in one chunk per output element.
        $for (h, num_heads) {
            Var head_off_kr = h * rope_dim * hidden_dim;
            $for (r, rope_dim) {
                Var acc = def(0.0f);
                $for (chunk, kHiddenChunks) {
                    Var chunk_start = chunk * kCoopChunk;

                    CoopVectorRef wkr_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    Var w_base = head_off_kr + r * hidden_dim + chunk_start;
                    wkr_ref.set_byte_offset(w_base * 4u);
                    auto w_chunk = cooperative_vector_load<float>(Wkr_byte_buf, wkr_ref);

                    CoopVectorRef h_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    Var h_base = hi + chunk_start;
                    h_ref.set_byte_offset(h_base * 4u);
                    auto h_chunk = cooperative_vector_load<float>(H_byte_buf, h_ref);

                    auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                    auto prod = cooperative_vector_fma(w_chunk, h_chunk, zero);

                    $for (j, kCoopChunk) {
                        acc += prod[j];
                    };
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
    return kernel;
}

OnlineAttentionCoopKernel create_online_attention_coop_kernel() {
    // -- Cooperative-vector-optimized online attention kernel ----------
    // Uses cooperative_vector_fma for all dot products:
    //   - q_latent hoisting (Wuk^T @ Q_content)
    //   - content score (q_latent @ cKV)
    //   - positional score (Q_pos @ Krope)
    //   - V up-projection (Wuv @ cKV)
    OnlineAttentionCoopKernel kernel = [&](BufferFloat Q, BufferFloat cKV,
                                           BufferFloat Wuk, BufferFloat Krope,
                                           BufferFloat Wuv, BufferFloat O,
                                           ByteBufferVar Q_byte_buf,
                                           ByteBufferVar cKV_byte_buf,
                                           ByteBufferVar Krope_byte_buf,
                                           ByteBufferVar Wuv_byte_buf) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("mla_online_attention_coop");

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

        // -- Hoist q_latent = Wuk[h]^T @ q_content (cooperative fma, chunked) --
        // content_dim=48 in kContentChunks=3 chunks of kCoopChunk=16.
        $array<float, latent_dim> q_latent;
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

        $for (j, seq_len) {
            // -- Content score via matrix absorption (cooperative fma) --
            // Process latent_dim=64 in kLatentChunks=4 chunks of size kCoopChunk=16.
            // Each chunk uses cooperative_vector_fma for element-wise multiply.
            Var content_score = def(0.0f);
            Var ci = (b * seq_len + j) * latent_dim;
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

                // Element-wise fma: q_chunk * c_chunk + 0.
                // cooperative_vector_splat creates a CoopVector with all equal elements.
                auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                auto prod = cooperative_vector_fma(q_chunk, c_chunk, zero);

                // Manual reduction: sum the 16 element-wise products.
                $for (c, kCoopChunk) {
                    content_score += prod[c];
                };
            };

            // -- Positional score (rope_dim=16 fits exactly in one chunk) --
            Var pos_score = def(0.0f);
            {
                CoopVectorRef q_pos_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                CoopVectorRef k_pos_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                Var q_base = qkv_idx(b, h, i, content_dim);
                Var k_base = rope_idx(b, h, j, 0u);
                q_pos_ref.set_byte_offset(q_base * 4u);
                k_pos_ref.set_byte_offset(k_base * 4u);
                auto q_pos = cooperative_vector_load<float>(Q_byte_buf, q_pos_ref);
                auto k_pos = cooperative_vector_load<float>(Krope_byte_buf, k_pos_ref);
                auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                auto prod = cooperative_vector_fma(q_pos, k_pos, zero);
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

            // -- Pre-load cKV[j,:] into local array (same as regular kernel) --
            $array<float, latent_dim> cKV_local;
            $for (l, latent_dim / 4u) {
                Var l4 = l * 4u;
                Var cv_base = ci + l4;
                cKV_local[l4] = cKV.read(cv_base);
                cKV_local[l4 + 1u] = cKV.read(cv_base + 1u);
                cKV_local[l4 + 2u] = cKV.read(cv_base + 2u);
                cKV_local[l4 + 3u] = cKV.read(cv_base + 3u);
            };

            // Update O_row: O *= exp_diff, then O += V[j] * exp_score.
            // V[b,h,j,d] = Wuv[h,d,:] @ cKV[b,j,:] (cooperative fma, chunked).
            $for (d, head_dim) {
                Var old_o = O.read(qkv_idx(b, h, i, d));
                Var head_off_uv = h * head_dim * latent_dim;
                Var v_val = def(0.0f);

                // Chunked dot product using cooperative_vector_fma.
                $for (chunk, kLatentChunks) {
                    Var chunk_start = chunk * kCoopChunk;

                    // Load Wuv[d, chunk_start:chunk_start+16] via CoopVectorRef.
                    CoopVectorRef wuv_ref{CoopRefVecType::FLOAT32, kCoopChunk};
                    Var w_base = head_off_uv + d * latent_dim + chunk_start;
                    wuv_ref.set_byte_offset(w_base * 4u);
                    auto w_chunk = cooperative_vector_load<float>(Wuv_byte_buf, wuv_ref);

                    // Load cKV_local[chunk_start:chunk_start+16] into CoopVector.
                    CoopVector<float> c_chunk{kCoopChunk};
                    $for (c, kCoopChunk) {
                        c_chunk[c] = cKV_local[chunk_start + c];
                    };

                    auto zero = cooperative_vector_splat<float>(0.0f, kCoopChunk);
                    auto prod = cooperative_vector_fma(w_chunk, c_chunk, zero);

                    $for (c, kCoopChunk) {
                        v_val += prod[c];
                    };
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
    return kernel;
}

OnlineAttentionKernel create_online_attention_kernel() {
    // -- Shared-memory-optimized MLA online attention kernel --------
    // Broadcast cKV[j,:] and Krope[j,:] via shared memory to eliminate
    // 256x redundant global loads. Only latent_dim=64 threads load
    // cKV and rope_dim=16 threads load Krope; all 256 read from shared.
    OnlineAttentionKernel kernel = [&](BufferFloat Q, BufferFloat cKV,
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

        // -- Hoist q_latent = Wuk[h]^T @ q_content (same as regular kernel) --
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

        // Shared memory for this j iteration
        Shared<float> cKV_shared{latent_dim};
        Shared<float> Krope_shared{rope_dim};

        $for (j, seq_len) {
            Var ci = (b * seq_len + j) * latent_dim;

            // -- Cooperative load cKV[j,:] and Krope[j,:] into shared memory --
            // latent_dim=64 threads load cKV (1 float each),
            // rope_dim=16 threads load Krope (overlaps with cKV load threads).
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
                Var q_base = qkv_idx(b, h, i, content_dim + r4);
                pos_score += Q.read(q_base) * Krope_shared[r4]
                           + Q.read(q_base + 1u) * Krope_shared[r4 + 1u]
                           + Q.read(q_base + 2u) * Krope_shared[r4 + 2u]
                           + Q.read(q_base + 3u) * Krope_shared[r4 + 3u];
            };

            Var score = (content_score + pos_score) * attention_scale;

            // -- Online softmax update --
            Var m_new = max(m, score);
            Var exp_diff = exp(m - m_new);
            Var exp_score = exp(score - m_new);
            s_norm = s_norm * exp_diff + exp_score;

            // -- Pre-load cKV[j,:] from shared memory into local array --
            // (No redundant global loads: cKV_shared already in fast shared memory)
            $array<float, latent_dim> cKV_local;
            $for (l, latent_dim / 4u) {
                Var l4 = l * 4u;
                cKV_local[l4] = cKV_shared[l4];
                cKV_local[l4 + 1u] = cKV_shared[l4 + 1u];
                cKV_local[l4 + 2u] = cKV_shared[l4 + 2u];
                cKV_local[l4 + 3u] = cKV_shared[l4 + 3u];
            };

            // -- Update O_row: O *= exp_diff, then O += V[j] * exp_score --
            $for (d, head_dim) {
                Var old_o = O.read(qkv_idx(b, h, i, d));
                Var head_off_uv = h * head_dim * latent_dim;
                Var v_val = def(0.0f);
                $for (l, latent_dim / 4u) {
                    Var l4 = l * 4u;
                    Var w_base = head_off_uv + d * latent_dim + l4;
                    v_val += Wuv.read(w_base) * cKV_local[l4]
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
    return kernel;
}

MhaOnlineAttentionKernel create_mha_online_attention_kernel() {
    // -- Shared-memory-optimized MHA kernel (broadcast K/V via shared memory) --
    // A single 32-thread wave consumes each shared K/V tile. Keeping the
    // consumer cohort within one wave avoids cross-wave progress skew while
    // still amortizing each global K/V load over 32 query rows.
    MhaOnlineAttentionKernel kernel = [&](BufferFloat Q, BufferFloat K,
                                          BufferFloat V, BufferFloat O) noexcept {
        set_block_size(32u, 1u, 1u);
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

        Shared<float> K_shared{head_dim};
        Shared<float> V_shared{head_dim};

        $for (d, head_dim) {
            O.write(qkv_idx(b, h, i, d), 0.0f);
        };

        $for (j, seq_len) {
            // Thread 0 loads K[j,:] and V[j,:] into shared memory, then all
            // 32 query threads consume the tile.
            Var kj_base = head_base + j * head_dim;
            $if (thread_x() == 0u) {
                $for (d, head_dim) {
                    K_shared[d] = K.read(kj_base + d);
                    V_shared[d] = V.read(kj_base + d);
                };
            };
            sync_block();

            // Score = Q[i] · K[j] (read K from shared)
            Var score = def(0.0f);
            $for (d, head_dim / 4u) {
                Var d4 = d * 4u;
                Var q_off = qi_base + d4;
                score += Q.read(q_off) * K_shared[d4]
                       + Q.read(q_off + 1u) * K_shared[d4 + 1u]
                       + Q.read(q_off + 2u) * K_shared[d4 + 2u]
                       + Q.read(q_off + 3u) * K_shared[d4 + 3u];
            };
            score = score * attention_scale;

            Var m_new = max(m, score);
            Var exp_diff = exp(m - m_new);
            Var exp_score = exp(score - m_new);
            s_norm = s_norm * exp_diff + exp_score;

            // Update O with V from shared
            $for (d, head_dim) {
                Var old_o = O.read(qkv_idx(b, h, i, d));
                O.write(qkv_idx(b, h, i, d), old_o * exp_diff + V_shared[d] * exp_score);
            };

            m = m_new;
            // All threads must finish consuming this K/V tile before
            // thread 0 is allowed to overwrite shared memory for j + 1.
            sync_block();
        };

        $for (d, head_dim) {
            Var val = O.read(qkv_idx(b, h, i, d));
            O.write(qkv_idx(b, h, i, d), val / s_norm);
        };
    };
    return kernel;
}

}// namespace mla
