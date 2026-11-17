// Multi-Head Latent Attention (MLA) Example -- CPU reference implementation.

#include "attention_config.h"
#include "attention_cpu_reference.h"

#include <algorithm>
#include <cmath>

#include <luisa/core/fiber.h>

namespace mla {
namespace {

void run_mla_cpu_reference(const AttentionHostData &data, luisa::vector<float> &O_cpu) {
    // CPU-side MLA reference.
    luisa::vector<float> Q_cpu(qkv_size, 0.0f);
    luisa::vector<float> Krope_cpu(rope_size, 0.0f);
    luisa::vector<float> cKV_cpu(latent_size, 0.0f);
    luisa::vector<float> V_cpu(qkv_size, 0.0f);
    luisa::vector<float> S_cpu(scores_size, 0.0f);
    luisa::vector<float> A_cpu(scores_size, 0.0f);

    auto &&H_host = data.h;
    auto &&Wq_host = data.wq;
    auto &&Wdkv_host = data.wdkv;
    auto &&Wuk_host = data.wuk;
    auto &&Wuv_host = data.wuv;
    auto &&Wkr_host = data.wkr;

    // Project Q.
    luisa::fiber::parallel(batch * seq_len, [&](uint32_t idx) noexcept {
        uint32_t b = idx / seq_len;
        uint32_t i = idx % seq_len;
        for (uint32_t h = 0u; h < num_heads; ++h) {
            for (uint32_t d = 0u; d < head_dim; ++d) {
                float acc = 0.0f;
                for (uint32_t e = 0u; e < hidden_dim; ++e) {
                    uint32_t wi = h * head_dim * hidden_dim + d * hidden_dim + e;
                    acc += Wq_host[wi] * H_host[hidden_index(b, i, e)];
                }
                Q_cpu[qkv_index(b, h, i, d)] = acc;
            }
            // RoPE on the positional slice.
            for (uint32_t r = 0u; r < rope_dim / 2u; ++r) {
                uint32_t d0 = content_dim + r * 2u;
                uint32_t d1 = d0 + 1u;
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
        uint32_t b = idx / seq_len;
        uint32_t i = idx % seq_len;
        for (uint32_t d = 0u; d < latent_dim; ++d) {
            float acc = 0.0f;
            for (uint32_t e = 0u; e < hidden_dim; ++e) {
                acc += Wdkv_host[d * hidden_dim + e] * H_host[hidden_index(b, i, e)];
            }
            cKV_cpu[latent_index(b, i, d)] = acc;
        }
    });

    // Up-project V.
    luisa::fiber::parallel(batch * seq_len * num_heads, [&](uint32_t idx) noexcept {
        uint32_t b = idx / (seq_len * num_heads);
        uint32_t h = (idx / seq_len) % num_heads;
        uint32_t j = idx % seq_len;
        for (uint32_t d = 0u; d < head_dim; ++d) {
            float acc = 0.0f;
            for (uint32_t l = 0u; l < latent_dim; ++l) {
                uint32_t wi = h * head_dim * latent_dim + d * latent_dim + l;
                acc += Wuv_host[wi] * cKV_cpu[latent_index(b, j, l)];
            }
            V_cpu[qkv_index(b, h, j, d)] = acc;
        }
    });

    // Decoupled RoPE keys.
    luisa::fiber::parallel(batch * seq_len * num_heads, [&](uint32_t idx) noexcept {
        uint32_t b = idx / (seq_len * num_heads);
        uint32_t h = (idx / seq_len) % num_heads;
        uint32_t j = idx % seq_len;
        for (uint32_t r = 0u; r < rope_dim; ++r) {
            float acc = 0.0f;
            for (uint32_t e = 0u; e < hidden_dim; ++e) {
                uint32_t wi = h * rope_dim * hidden_dim + r * hidden_dim + e;
                acc += Wkr_host[wi] * H_host[hidden_index(b, j, e)];
            }
            Krope_cpu[rope_index(b, h, j, r)] = acc;
        }
        for (uint32_t r = 0u; r < rope_dim / 2u; ++r) {
            uint32_t d0 = r * 2u;
            uint32_t d1 = d0 + 1u;
            auto rot = apply_rope_pair_cpu(
                Krope_cpu[rope_index(b, h, j, d0)],
                Krope_cpu[rope_index(b, h, j, d1)], r, j);
            Krope_cpu[rope_index(b, h, j, d0)] = rot.first;
            Krope_cpu[rope_index(b, h, j, d1)] = rot.second;
        }
    });

    // MLA scores with matrix absorption.
    luisa::fiber::parallel(batch * num_heads * seq_len, [&](uint32_t idx) noexcept {
        uint32_t b = idx / (num_heads * seq_len);
        uint32_t h = (idx / seq_len) % num_heads;
        uint32_t i = idx % seq_len;
        for (uint32_t j = 0u; j < seq_len; ++j) {
            // q_latent = Wuk[h]^T @ q_content.
            float q_latent[latent_dim];
            for (uint32_t d = 0u; d < latent_dim; ++d) {
                float acc = 0.0f;
                for (uint32_t c = 0u; c < content_dim; ++c) {
                    uint32_t wi = h * content_dim * latent_dim + c * latent_dim + d;
                    acc += Wuk_host[wi] * Q_cpu[qkv_index(b, h, i, c)];
                }
                q_latent[d] = acc;
            }

            float content_score = 0.0f;
            for (uint32_t d = 0u; d < latent_dim; ++d) {
                content_score += q_latent[d] * cKV_cpu[latent_index(b, j, d)];
            }

            float pos_score = 0.0f;
            for (uint32_t r = 0u; r < rope_dim; ++r) {
                pos_score += Q_cpu[qkv_index(b, h, i, content_dim + r)] *
                             Krope_cpu[rope_index(b, h, j, r)];
            }

            S_cpu[score_index(b, h, i, j)] = (content_score + pos_score) * attention_scale;
        }
    });

    // Softmax.
    luisa::fiber::parallel(batch * num_heads * seq_len, [&](uint32_t idx) noexcept {
        uint32_t b = idx / (num_heads * seq_len);
        uint32_t h = (idx / seq_len) % num_heads;
        uint32_t i = idx % seq_len;
        uint32_t base = score_index(b, h, i, 0);
        float m = -1e30f;
        for (uint32_t j = 0u; j < seq_len; ++j) {
            m = std::max(m, S_cpu[base + j]);
        }
        float s = 0.0f;
        for (uint32_t j = 0u; j < seq_len; ++j) {
            float e = std::exp(S_cpu[base + j] - m);
            A_cpu[base + j] = e;
            s += e;
        }
        for (uint32_t j = 0u; j < seq_len; ++j) {
            A_cpu[base + j] /= s;
        }
    });

    // AV weighted sum.
    luisa::fiber::parallel(batch * num_heads * seq_len * head_dim, [&](uint32_t idx) noexcept {
        uint32_t b = idx / (num_heads * seq_len * head_dim);
        uint32_t h = (idx / (seq_len * head_dim)) % num_heads;
        uint32_t i = (idx / head_dim) % seq_len;
        uint32_t d = idx % head_dim;
        float sum = 0.0f;
        for (uint32_t j = 0u; j < seq_len; ++j) {
            sum += A_cpu[score_index(b, h, i, j)] * V_cpu[qkv_index(b, h, j, d)];
        }
        O_cpu[qkv_index(b, h, i, d)] = sum;
    });
}

void run_mha_cpu_reference(const AttentionHostData &data, luisa::vector<float> &O_cpu) {
    // CPU-side MHA reference (multi-threaded).
    luisa::vector<float> S_cpu(scores_size, 0.0f);
    luisa::vector<float> A_cpu(scores_size, 0.0f);

    auto &&Q_host = data.q;
    auto &&K_host = data.k;
    auto &&V_host = data.v;

    // QK^T + scale.
    luisa::fiber::parallel(batch * num_heads * seq_len, [&](uint32_t idx) noexcept {
        uint32_t b = idx / (num_heads * seq_len);
        uint32_t h = (idx / seq_len) % num_heads;
        uint32_t i = idx % seq_len;
        for (uint32_t j = 0u; j < seq_len; ++j) {
            float sum = 0.0f;
            for (uint32_t d = 0u; d < head_dim; ++d) {
                uint32_t head_off = (b * num_heads + h) * seq_len * head_dim;
                uint32_t qi = head_off + i * head_dim + d;
                uint32_t ki = head_off + j * head_dim + d;
                sum += Q_host[qi] * K_host[ki];
            }
            S_cpu[score_index(b, h, i, j)] = sum * attention_scale;
        }
    });

    // Softmax.
    luisa::fiber::parallel(batch * num_heads * seq_len, [&](uint32_t idx) noexcept {
        uint32_t b = idx / (num_heads * seq_len);
        uint32_t h = (idx / seq_len) % num_heads;
        uint32_t i = idx % seq_len;
        uint32_t base = score_index(b, h, i, 0);
        float m = -1e30f;
        for (uint32_t j = 0u; j < seq_len; ++j) {
            m = std::max(m, S_cpu[base + j]);
        }
        float s = 0.0f;
        for (uint32_t j = 0u; j < seq_len; ++j) {
            float e = std::exp(S_cpu[base + j] - m);
            A_cpu[base + j] = e;
            s += e;
        }
        for (uint32_t j = 0u; j < seq_len; ++j) {
            A_cpu[base + j] /= s;
        }
    });

    // AV weighted sum.
    luisa::fiber::parallel(batch * num_heads * seq_len * head_dim, [&](uint32_t idx) noexcept {
        uint32_t b = idx / (num_heads * seq_len * head_dim);
        uint32_t h = (idx / (seq_len * head_dim)) % num_heads;
        uint32_t i = (idx / head_dim) % seq_len;
        uint32_t d = idx % head_dim;
        float sum = 0.0f;
        for (uint32_t j = 0u; j < seq_len; ++j) {
            sum += A_cpu[score_index(b, h, i, j)] * V_host[qkv_index(b, h, j, d)];
        }
        O_cpu[qkv_index(b, h, i, d)] = sum;
    });
}

}// namespace

luisa::vector<float> run_cpu_reference(const AttentionHostData &data, bool use_mla) {
    luisa::vector<float> O_cpu(qkv_size, 0.0f);

    // Fiber scheduler for the multi-threaded CPU reference.
    luisa::fiber::scheduler cpu_scheduler;

    if (use_mla) {
        run_mla_cpu_reference(data, O_cpu);
    } else {
        run_mha_cpu_reference(data, O_cpu);
    }
    return O_cpu;
}

}// namespace mla
